"""
cMBDF representation in PyTorch — differentiable implementation.

Provides the same representation as cMBDF.py but using differentiable
grid interpolation, enabling:
- Automatic gradient computation (forces, end-to-end learning)
- GPU acceleration
- Integration with neural network architectures

Usage:
    from cMBDF_torch import generate_mbdf
    reps = generate_mbdf(nuclear_charges, coords, rcut=10.0)
"""

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len


def fcut(rij, rcut):
    """Smooth cosine cutoff function."""
    return 0.5 * (torch.cos(torch.pi * rij / rcut) + 1.0)


def _hermite_np(x, degree, a=1):
    """Hermite polynomials (NumPy, for convolution precomputation only)."""
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


def get_convolutions(rstep=0.0008, rcut=10.0, alpha_list=[1.5, 5.0],
                     n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                     astep=0.0002, nAs=4, device='cpu'):
    """
    Precompute convolution grids as torch tensors.

    Returns (rconvs, aconvs, meta) where:
        rconvs: (m, n_weight, grid_size) radial convolutions
        aconvs: (m, nAs, grid_size) angular convolutions
        meta: dict with grid spacing info
    """
    step_r = rcut / next_fast_len(int(rcut / rstep))
    astep_actual = np.pi / next_fast_len(int(np.pi / astep))
    rgrid = np.arange(0.0, rcut, step_r)
    rgrid2 = np.arange(-rcut, rcut, step_r)
    agrid = np.arange(0.0, np.pi, astep_actual)
    agrid2 = np.arange(-np.pi, np.pi, astep_actual)

    size_r = len(rgrid)
    gaussian_r = np.exp(-a1 * (rgrid2 ** 2)) * np.sqrt(a1 / np.pi)
    m = order + 1

    fms = [gaussian_r] + [
        gaussian_r * _hermite_np(rgrid2, i, np.sqrt(a1)) * ((-np.sqrt(a1)) ** m)
        for i in range(1, m + 1)
    ]

    temp1, temp2 = [], []
    for i in range(m):
        fm = fms[i]
        temp_alpha, temp_n = [], []
        for alpha in alpha_list:
            gn = np.exp(-alpha * rgrid)
            arr = fftconvolve(gn, fm, mode='same') * step_r
            arr = arr / np.max(np.abs(arr))
            temp_alpha.append(arr)
        temp1.append(np.array(temp_alpha))

        for n in n_list:
            gn = 2.2508 * ((rgrid + 1) ** n)
            arr = fftconvolve(1 / gn, fm, mode='same')[:size_r] * step_r
            arr = arr / np.max(np.abs(arr))
            temp_n.append(arr)
        temp2.append(np.array(temp_n))

    rconvs = np.concatenate((np.asarray(temp1), np.asarray(temp2)), axis=1)

    # Angular convolutions
    gaussian_a = np.exp(-a2 * (agrid2 ** 2)) * np.sqrt(a2 / np.pi)
    fms_a = [gaussian_a] + [
        gaussian_a * _hermite_np(agrid2, i, np.sqrt(a2)) * ((-np.sqrt(a2)) ** m)
        for i in range(1, m + 1)
    ]

    temp_ang = []
    for i in range(m):
        fm = fms_a[i]
        temp = []
        for n in range(1, nAs + 1):
            gn = np.cos(n * agrid)
            arr = fftconvolve(gn, fm, mode='same') * astep_actual
            arr = arr / np.max(np.abs(arr))
            temp.append(arr)
        temp_ang.append(np.array(temp))

    aconvs = np.asarray(temp_ang)

    rconvs_t = torch.tensor(rconvs, dtype=torch.float64, device=device)
    aconvs_t = torch.tensor(aconvs, dtype=torch.float64, device=device)

    meta = {
        'rstep': step_r,
        'astep': astep_actual,
        'rcut': rcut,
        'rgrid_size': size_r,
        'agrid_size': len(agrid),
    }

    return rconvs_t, aconvs_t, meta


def _interp_lookup(grid, idx):
    """
    Differentiable linear interpolation into a 1D grid.

    Args:
        grid: (grid_size,) tensor
        idx: (...,) continuous index tensor

    Returns:
        (...,) interpolated values with gradients flowing through idx
    """
    max_idx = grid.shape[-1] - 1
    idx_floor = idx.long().clamp(0, max_idx)
    idx_ceil = (idx_floor + 1).clamp(0, max_idx)
    frac = (idx - idx_floor.to(idx.dtype)).clamp(0.0, 1.0)
    return grid[idx_floor] + frac * (grid[idx_ceil] - grid[idx_floor])


def _compute_rep(charges, coods, rconvs, aconvs, meta,
                 cutoff_r=10.0, n_atm=2.0, smooth_cutoff=True):
    """
    Core cMBDF computation: 2-body + 3-body features.

    Iterates over unique triplets (i < j < k) matching the original Numba
    implementation. The 2-body part is fully vectorized.

    Args:
        charges: (N,) float64 tensor
        coods: (N, 3) float64 tensor (can require grad)
        rconvs: (m1, n1, grid_r) tensor
        aconvs: (m2, n2, grid_a) tensor
        meta: dict with rstep, astep
        cutoff_r, n_atm, smooth_cutoff: representation parameters

    Returns:
        (N, nrs + nAs) feature tensor
    """
    N = charges.shape[0]
    m1, n1_w, grid_r = rconvs.shape
    m2, n2_w, grid_a = aconvs.shape
    nrs = m1 * n1_w
    nAs = m2 * n2_w
    rstep = meta['rstep']
    astep = meta['astep']
    dtype = coods.dtype
    device = coods.device

    # --- Pairwise geometry ---
    diff = coods.unsqueeze(1) - coods.unsqueeze(0)  # (N, N, 3)
    dmat = torch.norm(diff, dim=-1)                  # (N, N)
    mask = (dmat > 0) & (dmat < cutoff_r)            # (N, N)

    # --- 2-body (vectorized) ---
    charge_2b = torch.sqrt(charges.unsqueeze(1) * charges.unsqueeze(0))
    pref_2b = charge_2b.clone()
    if smooth_cutoff:
        fc = torch.where(mask, fcut(dmat, cutoff_r), torch.zeros_like(dmat))
        pref_2b = pref_2b * fc
    pref_2b = torch.where(mask, pref_2b, torch.zeros_like(pref_2b))

    r_idx = dmat / rstep
    rconvs_flat = rconvs.reshape(nrs, grid_r)

    rep_2b = torch.zeros(N, nrs, dtype=dtype, device=device)
    for c in range(nrs):
        vals = _interp_lookup(rconvs_flat[c], r_idx) * pref_2b
        rep_2b[:, c] = vals.sum(dim=1)

    # --- 3-body (loop over unique triplets) ---
    rep_3b = torch.zeros(N, nAs, dtype=dtype, device=device)
    aconvs_flat = aconvs.reshape(nAs, grid_a)

    # Precompute cutoff values
    if smooth_cutoff:
        fcut_mat = torch.where(mask, fcut(dmat, cutoff_r), torch.zeros_like(dmat))

    for i in range(N):
        for j in range(i + 1, N):
            if not mask[i, j]:
                continue

            rij_vec = diff[i, j]
            rij_n = dmat[i, j]

            for k in range(j + 1, N):
                if not mask[i, k]:
                    continue

                rik_vec = diff[i, k]
                rik_n = dmat[i, k]

                rkj_vec = coods[k] - coods[j]
                rkj_n = torch.norm(rkj_vec)

                # Three angles
                cos1 = ((rij_vec * rik_vec).sum() / (rij_n * rik_n)).clamp(-1.0, 1.0)
                cos2 = ((rij_vec * rkj_vec).sum() / (rij_n * rkj_n)).clamp(-1.0, 1.0)
                cos3 = ((-rkj_vec * rik_vec).sum() / (rkj_n * rik_n)).clamp(-1.0, 1.0)

                ang1 = torch.acos(cos1)
                ang2 = torch.acos(cos2)
                ang3 = torch.acos(cos3)

                atm = (rij_n * rik_n * rkj_n) ** n_atm

                pref = torch.pow(charges[i] * charges[j] * charges[k], 1.0 / 3.0)
                if smooth_cutoff:
                    pref = pref * fcut_mat[i, j] * fcut_mat[i, k] * fcut(rkj_n, cutoff_r)

                a1 = ang1 / astep
                a2_idx = ang2 / astep
                a3 = ang3 / astep

                for c in range(nAs):
                    i2 = c % n2_w

                    v1 = _interp_lookup(aconvs_flat[c], a1.unsqueeze(0)).squeeze(0)
                    v2 = _interp_lookup(aconvs_flat[c], a2_idx.unsqueeze(0)).squeeze(0)
                    v3 = _interp_lookup(aconvs_flat[c], a3.unsqueeze(0)).squeeze(0)

                    if i2 == 0:
                        c1 = (pref * v1 * cos2 * cos3) / atm
                        c2 = (pref * v2 * cos1 * cos3) / atm
                        c3 = (pref * v3 * cos2 * cos1) / atm
                    else:
                        c1 = (pref * v1) / atm
                        c2 = (pref * v2) / atm
                        c3 = (pref * v3) / atm

                    # Original code sets 2 symmetric slots per atom per triplet,
                    # then einsum sums over all slots → factor of 2 per atom
                    rep_3b[i, c] = rep_3b[i, c] + 2.0 * c1
                    rep_3b[j, c] = rep_3b[j, c] + 2.0 * c2
                    rep_3b[k, c] = rep_3b[k, c] + 2.0 * c3

    return torch.cat([rep_2b, rep_3b], dim=-1)


def get_cmbdf(charges, coods, rconvs, aconvs, meta, pad=None,
              rcut=10.0, n_atm=2.0, smooth_cutoff=True):
    """
    Returns the local cMBDF representation for a single molecule.

    Args:
        charges: (N,) nuclear charges (array or tensor)
        coods: (N, 3) coordinates (array or tensor)
        rconvs, aconvs: precomputed convolution grids
        meta: grid metadata
        pad: zero-pad to this many atoms
        rcut: cutoff radius
        n_atm: ATM damping exponent
        smooth_cutoff: apply cosine cutoff

    Returns:
        (pad, n_features) tensor
    """
    device = rconvs.device
    dtype = rconvs.dtype

    if not isinstance(charges, torch.Tensor):
        charges = torch.tensor(np.asarray(charges, dtype=np.float64),
                               dtype=dtype, device=device)
    if not isinstance(coods, torch.Tensor):
        coods = torch.tensor(np.asarray(coods, dtype=np.float64),
                             dtype=dtype, device=device)

    N = charges.shape[0]
    if pad is None:
        pad = N

    rep = _compute_rep(charges, coods, rconvs, aconvs, meta,
                       rcut, n_atm, smooth_cutoff)

    if pad > N:
        padding = torch.zeros(pad - N, rep.shape[1], dtype=dtype, device=device)
        rep = torch.cat([rep, padding], dim=0)

    return rep


def generate_mbdf(nuclear_charges, coords, rcut=10.0, n_atm=2.0,
                  smooth_cutoff=True, pad=None, device='cpu',
                  alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                  order=4, a1=2.0, a2=2.0, rstep=0.0008,
                  astep=0.0002, nAs=4):
    """
    Generate cMBDF representations for a batch of molecules.

    Args:
        nuclear_charges: list of arrays of nuclear charges
        coords: list of arrays of coordinates
        rcut: cutoff radius in Angstrom
        n_atm: ATM damping exponent
        smooth_cutoff: apply cosine cutoff (default True)
        pad: max atoms for zero-padding (auto if None)
        device: 'cpu' or 'cuda'

    Returns:
        (n_molecules, pad, n_features) torch tensor
    """
    rconvs, aconvs, meta = get_convolutions(
        rstep=rstep, rcut=rcut, alpha_list=alpha_list, n_list=n_list,
        order=order, a1=a1, a2=a2, astep=astep, nAs=nAs, device=device
    )

    lengths = [len(q) for q in nuclear_charges]
    if pad is None:
        pad = max(lengths)

    reps = []
    for q, r in zip(nuclear_charges, coords):
        rep = get_cmbdf(q, r, rconvs, aconvs, meta, pad, rcut, n_atm, smooth_cutoff)
        reps.append(rep)

    return torch.stack(reps, dim=0)
