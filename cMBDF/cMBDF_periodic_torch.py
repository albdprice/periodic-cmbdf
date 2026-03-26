"""
Vectorized periodic cMBDF in PyTorch.

Eliminates Python-level loops in the 3-body computation by:
1. Precomputing all valid triplets as index tensors from neighbor lists
2. Batch computing all angles/distances/cutoffs as tensor operations
3. Batch interpolating convolution grids
4. Using scatter_add for per-atom accumulation

Supports CPU and GPU. Differentiable via autograd.

Usage:
    from cMBDF_periodic_torch import generate_mbdf_periodic
    reps = generate_mbdf_periodic(charges_list, coords_list, cells_list,
                                   pbc=(True,True,True), rcut=6.0, device='cuda')
"""

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list


def _hermite_np(x, degree, a=1):
    """Hermite polynomials for convolution precomputation (NumPy)."""
    if degree == 1:
        return -2 * a * x
    elif degree == 2:
        return 4 * (a * x) ** 2 - 2 * a
    elif degree == 3:
        return -8 * (a * x) ** 3 + 12 * a * x
    elif degree == 4:
        return 16 * (a * x) ** 4 - 48 * (a * x) ** 2 + 12 * a ** 2
    elif degree == 5:
        return -32 * (a * x) ** 5 + 160 * (a * x) ** 3 - 120 * (a * x)


def get_convolutions(rstep=0.0008, rcut=6.0, alpha_list=[1.5, 5.0],
                     n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                     astep=0.0002, nAs=4, device='cpu'):
    """Precompute convolution grids as torch tensors."""
    step_r = rcut / next_fast_len(int(rcut / rstep))
    astep_actual = np.pi / next_fast_len(int(np.pi / astep))
    rgrid = np.arange(0.0, rcut, step_r)
    rgrid2 = np.arange(-rcut, rcut, step_r)
    agrid = np.arange(0.0, np.pi, astep_actual)
    agrid2 = np.arange(-np.pi, np.pi, astep_actual)

    size_r = len(rgrid)
    gaussian_r = np.exp(-a1 * rgrid2**2) * np.sqrt(a1 / np.pi)
    m = order + 1

    fms = [gaussian_r] + [
        gaussian_r * _hermite_np(rgrid2, i, np.sqrt(a1)) * ((-np.sqrt(a1))**m)
        for i in range(1, m + 1)]

    temp1, temp2 = [], []
    for i in range(m):
        fm = fms[i]
        ta, tn = [], []
        for alpha in alpha_list:
            arr = fftconvolve(np.exp(-alpha * rgrid), fm, mode='same') * step_r
            ta.append(arr / np.max(np.abs(arr)))
        temp1.append(np.array(ta))
        for n in n_list:
            arr = fftconvolve(1 / (2.2508 * (rgrid + 1)**n), fm, mode='same')[:size_r] * step_r
            tn.append(arr / np.max(np.abs(arr)))
        temp2.append(np.array(tn))

    rconvs = torch.tensor(
        np.concatenate((np.asarray(temp1), np.asarray(temp2)), axis=1),
        dtype=torch.float64, device=device)

    gaussian_a = np.exp(-a2 * agrid2**2) * np.sqrt(a2 / np.pi)
    fms_a = [gaussian_a] + [
        gaussian_a * _hermite_np(agrid2, i, np.sqrt(a2)) * ((-np.sqrt(a2))**m)
        for i in range(1, m + 1)]

    temp_ang = []
    for i in range(m):
        fm = fms_a[i]
        temp = []
        for n in range(1, nAs + 1):
            arr = fftconvolve(np.cos(n * agrid), fm, mode='same') * astep_actual
            temp.append(arr / np.max(np.abs(arr)))
        temp_ang.append(np.array(temp))

    aconvs = torch.tensor(np.asarray(temp_ang), dtype=torch.float64, device=device)

    meta = {'rstep': step_r, 'astep': astep_actual, 'rcut': rcut}
    return rconvs, aconvs, meta


def _interp(grid, idx):
    """Differentiable linear interpolation into 1D grid."""
    max_idx = grid.shape[-1] - 1
    fl = idx.long().clamp(0, max_idx)
    ce = (fl + 1).clamp(0, max_idx)
    f = (idx - fl.to(idx.dtype)).clamp(0.0, 1.0)
    return grid[fl] + f * (grid[ce] - grid[fl])


def build_neighbor_data_torch(charges, coords, cell, pbc, cutoff, device='cpu'):
    """
    Build neighbor list via ASE and return torch tensors.

    Returns:
        idx_i: (n_pairs,) int tensor — central atom index
        idx_j: (n_pairs,) int tensor — neighbor atom index
        rij_vec: (n_pairs, 3) displacement vectors (i -> j with image)
        rij_dist: (n_pairs,) distances
        pair_offsets: (n_atoms+1,) start/end of each atom's neighbors
        nbr_pos: (n_pairs, 3) actual neighbor positions (for R_jk computation)
    """
    atoms = Atoms(numbers=charges.astype(int), positions=coords,
                  cell=cell, pbc=pbc)
    ii, jj, DD, dd = ase_neighbor_list('ijDd', atoms, cutoff)

    order = np.argsort(ii, kind='stable')
    ii, jj, DD, dd = ii[order], jj[order], DD[order], dd[order]

    n_atoms = len(charges)
    offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    for i in range(n_atoms):
        offsets[i + 1] = offsets[i] + np.sum(ii == i)

    nbr_pos = coords[ii] + DD  # image-shifted neighbor positions

    return (torch.tensor(ii, dtype=torch.long, device=device),
            torch.tensor(jj, dtype=torch.long, device=device),
            torch.tensor(DD, dtype=torch.float64, device=device),
            torch.tensor(dd, dtype=torch.float64, device=device),
            torch.tensor(offsets, dtype=torch.long, device=device),
            torch.tensor(nbr_pos, dtype=torch.float64, device=device))


def compute_rep_periodic(charges_np, coords_np, cell_np, pbc,
                         rconvs, aconvs, meta, rcut=6.0, n_atm=2.0):
    """
    Compute periodic cMBDF for a single structure — fully vectorized.

    Args:
        charges_np, coords_np, cell_np: numpy arrays
        pbc: tuple of bools
        rconvs: (m1, n1, grid_r) torch tensor
        aconvs: (m2, n2, grid_a) torch tensor
        meta: dict with rstep, astep
        rcut, n_atm: representation parameters

    Returns:
        (N, nrs + nAs) torch tensor
    """
    device = rconvs.device
    dtype = rconvs.dtype
    N = len(charges_np)

    m1, n1_w, grid_r = rconvs.shape
    m2, n2_w, grid_a = aconvs.shape
    nrs = m1 * n1_w
    nAs = m2 * n2_w
    rstep = meta['rstep']
    astep = meta['astep']

    charges = torch.tensor(charges_np, dtype=dtype, device=device)

    # --- Build neighbor list ---
    idx_i, idx_j, rij_vec, rij_dist, offsets, nbr_pos = \
        build_neighbor_data_torch(charges_np, coords_np, cell_np, pbc, rcut, device)

    n_pairs = idx_i.shape[0]
    if n_pairs == 0:
        return torch.zeros(N, nrs + nAs, dtype=dtype, device=device)

    # --- 2-body (vectorized over all pairs) ---
    fcut_ij = 0.5 * (torch.cos(torch.pi * rij_dist / rcut) + 1.0)
    pref_2b = torch.sqrt(charges[idx_i] * charges[idx_j]) * fcut_ij

    r_idx = rij_dist / rstep
    rconvs_flat = rconvs.reshape(nrs, grid_r)  # (nrs, grid_r)

    # Interpolate all pairs for all channels at once
    # r_idx: (n_pairs,) -> lookup for each channel
    rep_2b = torch.zeros(N, nrs, dtype=dtype, device=device)
    for c in range(nrs):
        vals = _interp(rconvs_flat[c], r_idx) * pref_2b  # (n_pairs,)
        rep_2b.scatter_add_(0, idx_i.unsqueeze(1).expand(-1, 1),
                            vals.unsqueeze(1))[:, 0]
        # Actually scatter_add needs matching dims, let me do it properly:
    # Redo 2-body with proper scatter_add
    rep_2b = torch.zeros(N, nrs, dtype=dtype, device=device)
    for c in range(nrs):
        vals = _interp(rconvs_flat[c], r_idx) * pref_2b  # (n_pairs,)
        rep_2b[:, c].scatter_add_(0, idx_i, vals)

    # --- 3-body: enumerate all triplets from neighbor pairs ---
    # For each atom i, triplets are all pairs (p, q) where p < q
    # in i's neighbor list. We build flat triplet index arrays.

    triplet_i_list = []
    triplet_p_list = []  # pair index for j neighbor
    triplet_q_list = []  # pair index for k neighbor

    offsets_np = offsets.cpu().numpy()
    for i in range(N):
        start = offsets_np[i]
        end = offsets_np[i + 1]
        n_neigh = end - start
        if n_neigh < 2:
            continue
        # All pairs (p, q) with p < q within i's neighbor range
        local_p, local_q = torch.triu_indices(n_neigh, n_neigh, offset=1, device=device)
        triplet_i_list.append(torch.full((local_p.shape[0],), i, dtype=torch.long, device=device))
        triplet_p_list.append(local_p + start)
        triplet_q_list.append(local_q + start)

    if len(triplet_i_list) == 0:
        return torch.cat([rep_2b, torch.zeros(N, nAs, dtype=dtype, device=device)], dim=1)

    tri_i = torch.cat(triplet_i_list)    # (n_triplets,) central atom
    tri_p = torch.cat(triplet_p_list)    # (n_triplets,) pair index for j
    tri_q = torch.cat(triplet_q_list)    # (n_triplets,) pair index for k
    n_triplets = tri_i.shape[0]

    # Gather vectors and distances for j and k neighbors
    rij_v = rij_vec[tri_p]               # (n_triplets, 3)
    rik_v = rij_vec[tri_q]               # (n_triplets, 3)
    rij_n = rij_dist[tri_p]              # (n_triplets,)
    rik_n = rij_dist[tri_q]              # (n_triplets,)

    # R_jk: image j position to image k position
    rkj_v = nbr_pos[tri_q] - nbr_pos[tri_p]  # (n_triplets, 3)
    rkj_n = torch.norm(rkj_v, dim=1)         # (n_triplets,)

    # Filter degenerate triplets
    valid = rkj_n > 1e-10
    if not valid.all():
        tri_i = tri_i[valid]
        rij_v = rij_v[valid]
        rik_v = rik_v[valid]
        rij_n = rij_n[valid]
        rik_n = rik_n[valid]
        rkj_v = rkj_v[valid]
        rkj_n = rkj_n[valid]
        tri_p = tri_p[valid]
        tri_q = tri_q[valid]
        n_triplets = tri_i.shape[0]

    # Charges
    j_idx = idx_j[tri_p]
    k_idx = idx_j[tri_q]
    z_i = charges[tri_i]
    z_j = charges[j_idx]
    z_k = charges[k_idx]

    # Cutoff values
    fc_ij = 0.5 * (torch.cos(torch.pi * rij_n / rcut) + 1.0)
    fc_ik = 0.5 * (torch.cos(torch.pi * rik_n / rcut) + 1.0)
    fc_jk = torch.where(rkj_n < rcut,
                         0.5 * (torch.cos(torch.pi * rkj_n / rcut) + 1.0),
                         torch.zeros_like(rkj_n))

    # Three angles of each triangle
    cos1 = ((rij_v * rik_v).sum(dim=1) / (rij_n * rik_n)).clamp(-1.0, 1.0)
    cos2 = ((rij_v * rkj_v).sum(dim=1) / (rij_n * rkj_n)).clamp(-1.0, 1.0)
    cos3 = ((-rkj_v * rik_v).sum(dim=1) / (rkj_n * rik_n)).clamp(-1.0, 1.0)

    ang1 = torch.acos(cos1)
    ang2 = torch.acos(cos2)
    ang3 = torch.acos(cos3)

    # ATM damping and charge prefactor
    atm = (rij_n * rik_n * rkj_n) ** n_atm
    charge_3b = torch.pow(z_i * z_j * z_k, 1.0 / 3.0)
    pref_3b = charge_3b * fc_ij * fc_ik * fc_jk

    # Angular grid indices (continuous, for interpolation)
    a1_idx = ang1 / astep
    a2_idx = ang2 / astep
    a3_idx = ang3 / astep

    # Evaluate all angular convolutions and accumulate
    aconvs_flat = aconvs.reshape(nAs, grid_a)  # (nAs, grid_a)
    rep_3b = torch.zeros(N, nAs, dtype=dtype, device=device)

    for c in range(nAs):
        i2 = c % n2_w  # weighting function index

        v1 = _interp(aconvs_flat[c], a1_idx)  # (n_triplets,)
        v2 = _interp(aconvs_flat[c], a2_idx)
        v3 = _interp(aconvs_flat[c], a3_idx)

        if i2 == 0:
            contrib = (pref_3b * v1 * cos2 * cos3) / atm
        else:
            contrib = (pref_3b * v1) / atm

        # Factor 2 for (j,k)/(k,j) symmetry
        rep_3b[:, c].scatter_add_(0, tri_i, 2.0 * contrib)

    return torch.cat([rep_2b, rep_3b], dim=1)


def generate_mbdf_periodic_batched(nuclear_charges, coords, cells,
                                    pbc=(True, True, True), rcut=6.0, n_atm=2.0,
                                    device='cpu', rstep=0.0008, alpha_list=[1.5, 5.0],
                                    n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                                    astep=0.0002, nAs=4):
    """
    Batched GPU processing: concatenate all structures into one super-structure,
    build a combined neighbor list with no cross-structure pairs, and run the
    vectorized computation once for all atoms.

    Returns list of (n_atoms_i, n_features) tensors (one per structure).
    """
    rconvs, aconvs, meta = get_convolutions(
        rstep=rstep, rcut=rcut, alpha_list=alpha_list, n_list=n_list,
        order=order, a1=a1, a2=a2, astep=astep, nAs=nAs, device=device)

    m1, n1_w, grid_r = rconvs.shape
    m2, n2_w, grid_a = aconvs.shape
    nrs = m1 * n1_w
    nAs_total = m2 * n2_w
    rstep_val = meta['rstep']
    astep_val = meta['astep']
    dtype = torch.float64

    if not isinstance(pbc[0], (list, tuple)):
        pbc_list = [pbc] * len(nuclear_charges)
    else:
        pbc_list = pbc

    # === Step 1: Build all neighbor lists and concatenate ===
    all_idx_i = []
    all_idx_j = []
    all_rij_vec = []
    all_rij_dist = []
    all_nbr_pos = []
    all_charges_cat = []
    all_offsets = []  # per-structure pair range offsets
    atom_offset = 0
    pair_offset = 0
    struct_atom_ranges = []  # (start, end) atom indices per structure

    for q, r, c, p in zip(nuclear_charges, coords, cells, pbc_list):
        q = np.asarray(q, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        n_atoms = len(q)

        ii, jj, DD, dd, offsets, nbr_pos_s = build_neighbor_data_torch(
            q, r, c, p, rcut, 'cpu')

        # Shift indices by atom_offset
        all_idx_i.append(ii + atom_offset)
        all_idx_j.append(jj + atom_offset)
        all_rij_vec.append(DD)
        all_rij_dist.append(dd)
        all_nbr_pos.append(nbr_pos_s)
        all_charges_cat.append(torch.tensor(q, dtype=dtype))

        # Track per-atom offsets for triplet enumeration
        shifted_offsets = offsets.clone()
        shifted_offsets += pair_offset
        all_offsets.append((shifted_offsets, atom_offset, n_atoms))

        struct_atom_ranges.append((atom_offset, atom_offset + n_atoms))
        atom_offset += n_atoms
        pair_offset += len(ii)

    # Concatenate everything
    cat_idx_i = torch.cat(all_idx_i).to(device)
    cat_idx_j = torch.cat(all_idx_j).to(device)
    cat_rij_vec = torch.cat(all_rij_vec).to(device)
    cat_rij_dist = torch.cat(all_rij_dist).to(device)
    cat_nbr_pos = torch.cat(all_nbr_pos).to(device)
    cat_charges = torch.cat(all_charges_cat).to(device)
    N_total = atom_offset
    n_pairs_total = len(cat_idx_i)

    # === Step 2: 2-body (vectorized over ALL pairs from ALL structures) ===
    fcut_ij = 0.5 * (torch.cos(torch.pi * cat_rij_dist / rcut) + 1.0)
    pref_2b = torch.sqrt(cat_charges[cat_idx_i] * cat_charges[cat_idx_j]) * fcut_ij

    r_idx = cat_rij_dist / rstep_val
    rconvs_flat = rconvs.reshape(nrs, grid_r)

    rep_2b = torch.zeros(N_total, nrs, dtype=dtype, device=device)
    for ch in range(nrs):
        vals = _interp(rconvs_flat[ch], r_idx) * pref_2b
        rep_2b[:, ch].scatter_add_(0, cat_idx_i, vals)

    # === Step 3: 3-body (enumerate triplets per structure, then concatenate) ===
    triplet_i_list = []
    triplet_p_list = []
    triplet_q_list = []

    for shifted_offsets, a_off, n_at in all_offsets:
        offsets_np = shifted_offsets.numpy()
        for i_local in range(n_at):
            i_global = i_local + a_off
            start = offsets_np[i_local]
            end = offsets_np[i_local + 1]
            n_neigh = end - start
            if n_neigh < 2:
                continue
            lp, lq = torch.triu_indices(n_neigh, n_neigh, offset=1)
            triplet_i_list.append(torch.full((lp.shape[0],), i_global, dtype=torch.long))
            triplet_p_list.append(lp + start)
            triplet_q_list.append(lq + start)

    if len(triplet_i_list) > 0:
        tri_i = torch.cat(triplet_i_list).to(device)
        tri_p = torch.cat(triplet_p_list).to(device)
        tri_q = torch.cat(triplet_q_list).to(device)

        rij_v = cat_rij_vec[tri_p]
        rik_v = cat_rij_vec[tri_q]
        rij_n = cat_rij_dist[tri_p]
        rik_n = cat_rij_dist[tri_q]

        rkj_v = cat_nbr_pos[tri_q] - cat_nbr_pos[tri_p]
        rkj_n = torch.norm(rkj_v, dim=1)

        valid = rkj_n > 1e-10
        if not valid.all():
            tri_i, tri_p, tri_q = tri_i[valid], tri_p[valid], tri_q[valid]
            rij_v, rik_v = rij_v[valid], rik_v[valid]
            rij_n, rik_n, rkj_v, rkj_n = rij_n[valid], rik_n[valid], rkj_v[valid], rkj_n[valid]

        j_idx = cat_idx_j[tri_p]
        k_idx = cat_idx_j[tri_q]

        fc_ij = 0.5 * (torch.cos(torch.pi * rij_n / rcut) + 1.0)
        fc_ik = 0.5 * (torch.cos(torch.pi * rik_n / rcut) + 1.0)
        fc_jk = torch.where(rkj_n < rcut,
                             0.5 * (torch.cos(torch.pi * rkj_n / rcut) + 1.0),
                             torch.zeros_like(rkj_n))

        cos1 = ((rij_v * rik_v).sum(1) / (rij_n * rik_n)).clamp(-1, 1)
        cos2 = ((rij_v * rkj_v).sum(1) / (rij_n * rkj_n)).clamp(-1, 1)
        cos3 = ((-rkj_v * rik_v).sum(1) / (rkj_n * rik_n)).clamp(-1, 1)

        ang1 = torch.acos(cos1)
        atm = (rij_n * rik_n * rkj_n) ** n_atm
        charge_3b = torch.pow(cat_charges[tri_i] * cat_charges[j_idx] * cat_charges[k_idx], 1.0/3.0)
        pref_3b = charge_3b * fc_ij * fc_ik * fc_jk

        a1_idx = ang1 / astep_val
        aconvs_flat = aconvs.reshape(nAs_total, grid_a)

        rep_3b = torch.zeros(N_total, nAs_total, dtype=dtype, device=device)
        for ch in range(nAs_total):
            i2 = ch % n2_w
            v1 = _interp(aconvs_flat[ch], a1_idx)
            if i2 == 0:
                contrib = (pref_3b * v1 * cos2 * cos3) / atm
            else:
                contrib = (pref_3b * v1) / atm
            rep_3b[:, ch].scatter_add_(0, tri_i, 2.0 * contrib)
    else:
        rep_3b = torch.zeros(N_total, nAs_total, dtype=dtype, device=device)

    rep_full = torch.cat([rep_2b, rep_3b], dim=1)

    # === Step 4: De-batch into per-structure results ===
    results = []
    for start, end in struct_atom_ranges:
        results.append(rep_full[start:end])

    return results


def generate_mbdf_periodic(nuclear_charges, coords, cells,
                           pbc=(True, True, True), rcut=6.0, n_atm=2.0,
                           pad=None, device='cpu',
                           rstep=0.0008, alpha_list=[1.5, 5.0],
                           n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                           astep=0.0002, nAs=4):
    """
    Generate periodic cMBDF representations for a batch of structures.

    Args:
        nuclear_charges: list of charge arrays
        coords: list of coordinate arrays
        cells: list of (3,3) lattice vector arrays
        pbc: tuple or list of tuples
        rcut: cutoff radius (default 6.0 Å)
        n_atm: ATM damping exponent
        pad: zero-pad to this many atoms
        device: 'cpu' or 'cuda'

    Returns:
        (n_structures, pad, n_features) torch tensor
    """
    rconvs, aconvs, meta = get_convolutions(
        rstep=rstep, rcut=rcut, alpha_list=alpha_list, n_list=n_list,
        order=order, a1=a1, a2=a2, astep=astep, nAs=nAs, device=device)

    lengths = [len(q) for q in nuclear_charges]
    if pad is None:
        pad = max(lengths)

    if not isinstance(pbc[0], (list, tuple)):
        pbc_list = [pbc] * len(nuclear_charges)
    else:
        pbc_list = pbc

    reps = []
    for q, r, c, p in zip(nuclear_charges, coords, cells, pbc_list):
        q = np.asarray(q, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)

        rep = compute_rep_periodic(q, r, c, p, rconvs, aconvs, meta, rcut, n_atm)

        N = len(q)
        if pad > N:
            padding = torch.zeros(pad - N, rep.shape[1], dtype=rep.dtype, device=rep.device)
            rep = torch.cat([rep, padding], dim=0)

        reps.append(rep)

    return torch.stack(reps, dim=0)
