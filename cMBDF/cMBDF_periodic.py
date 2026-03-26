"""
Periodic cMBDF: Extension to periodic / solid-state systems.

Extends cMBDF to handle periodic boundary conditions via neighbor lists.
Uses ASE for neighbor list construction (prototyping; Numba replacement later).
The representation generation is Numba-JIT compiled for speed.

Usage:
    from cMBDF_periodic import generate_mbdf_periodic

    # Bulk silicon (diamond cubic, 2 atoms)
    charges = np.array([14.0, 14.0])
    coords = np.array([[0, 0, 0], [1.3575, 1.3575, 1.3575]])
    cell = np.array([[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]])

    reps = generate_mbdf_periodic([charges], [coords], [cell],
                                   pbc=(True, True, True), rcut=6.0)
"""

import numpy as np
import numba as nb
from numpy import einsum
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len
from ase import Atoms
from ase.neighborlist import neighbor_list


# Import convolution precomputation from cMBDF (identical for periodic)
from cMBDF import get_convolutions, fcut, hermite_polynomial


# van der Waals radii (Angstrom) indexed by atomic number, from cMBDF_4body.py
rvdw = np.array((1.0,  # 0 Ghost
    1.20, 1.40, 1.82, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,  # 1-10
    2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31,  # 11-20
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.63, 1.40, 1.39,        # 21-30
    1.87, 2.11, 1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 1.0, 1.0,    # 31-40
    1.0, 1.0, 1.0, 1.0, 1.0, 1.63, 1.72, 1.58, 1.93, 2.17,       # 41-50
    2.06, 2.06, 1.98, 2.16, 3.43, 2.49,                            # 51-56
    *([1.0]*15),  # 57-71 (lanthanides)
    *([1.0]*6),   # 72-77
    1.75, 1.66, 1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.20,        # 78-86
    3.48, 2.83, 1.0, 1.0, 1.0, 1.86,                              # 87-92
    *([1.0]*8),   # 93-100
))


def get_convolutions_element_specific(elems, rstep=0.0008, rcut=10.0,
                                       alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                                       order=4, a2=2.0, astep=0.0002, nAs=4):
    """
    Compute element-specific radial convolutions + universal angular convolutions.

    Each unique element gets its own radial convolution set using vdW radius as
    the Gaussian width. Angular convolutions are shared across elements.

    Args:
        elems: array of unique atomic numbers present in the dataset

    Returns:
        (rconvs_dict, aconvs, elem_to_idx) where:
            rconvs_dict: dict mapping Z -> (m, n_weight, grid_size) radial convolutions
            aconvs: (2, m, nAs, grid_size) angular convolutions (with gradients)
            elem_to_idx: dict mapping Z -> index in rconvs array
    """
    step_r = rcut / next_fast_len(int(rcut / rstep))
    astep_actual = np.pi / next_fast_len(int(np.pi / astep))
    rgrid = np.arange(0.0, rcut, step_r)
    rgrid2 = np.arange(-rcut, rcut, step_r)
    agrid = np.arange(0.0, np.pi, astep_actual)
    agrid2 = np.arange(-np.pi, np.pi, astep_actual)

    size_r = len(rgrid)
    m = order + 1

    # Element-specific radial convolutions
    rconvs_list = []
    elem_to_idx = {}

    for idx, elem in enumerate(elems):
        elem_to_idx[int(elem)] = idx
        a1 = rvdw[int(elem)] ** 2
        norm = np.sqrt((2 * np.pi) * a1)
        gaussian = np.exp(-a1 * (rgrid2 ** 2)) / norm

        fms = [gaussian] + [gaussian * hermite_polynomial(rgrid2, i, a1)
                            for i in range(1, m + 1)]

        temp1, temp2 = [], []
        for i in range(m):
            fm = fms[i]
            temp_alpha = []
            for alpha in alpha_list:
                gn = np.exp(-alpha * rgrid)
                arr = fftconvolve(gn, fm, mode='same') * step_r
                arr = arr / np.max(np.abs(arr))
                temp_alpha.append(arr)
            temp1.append(np.array(temp_alpha))

            temp_n = []
            for n in n_list:
                gn = 2.2508 * ((rgrid + 1) ** n)
                arr = fftconvolve(1 / gn, fm, mode='same')[:size_r] * step_r
                arr = arr / np.max(np.abs(arr))
                temp_n.append(arr)
            temp2.append(np.array(temp_n))

        rconv = np.concatenate((np.asarray(temp1), np.asarray(temp2)), axis=1)
        rconvs_list.append(rconv)

    # Stack into (n_elems, m, n_weight, grid_size) array
    rconvs_arr = np.asarray(rconvs_list)

    # Universal angular convolutions (same as standard cMBDF)
    gaussian_a = np.exp(-a2 * (agrid2 ** 2)) * np.sqrt(a2 / np.pi)
    fms_a = [gaussian_a] + [gaussian_a * hermite_polynomial(agrid2, i, np.sqrt(a2))
                            * ((-np.sqrt(a2)) ** m) for i in range(1, m + 1)]

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

    return rconvs_arr, aconvs, elem_to_idx


def build_neighbor_data(charges, coords, cell, pbc, cutoff_r, use_ase=False):
    """
    Build neighbor list for a periodic structure.

    Uses Numba-compiled neighbor list by default. Set use_ase=True
    for the ASE fallback (useful for debugging).

    Returns arrays suitable for Numba-compiled generate_data_periodic:
        neighbor_j:     (n_pairs,) int array — index of neighbor atom j
        neighbor_pos:   (n_pairs, 3) float array — position of j (with image shift)
        neighbor_rij:   (n_pairs, 3) float array — vector from i to j
        neighbor_dist:  (n_pairs,) float array — |r_ij|
        pair_offsets:   (n_atoms+1,) int array — neighbor_j[pair_offsets[i]:pair_offsets[i+1]]
                        gives the neighbors of atom i
    """
    n_atoms = len(charges)

    if not use_ase:
        # Use Numba-compiled neighbor list (faster, no ASE dependency)
        from neighbor_list_numba import build_neighbor_data_numba
        return build_neighbor_data_numba(charges, coords, cell, pbc, cutoff_r)

    # ASE fallback
    atoms = Atoms(
        numbers=charges.astype(int),
        positions=coords,
        cell=cell,
        pbc=pbc
    )

    idx_i, idx_j, dist_vec, dist = neighbor_list('ijDd', atoms, cutoff_r)

    sort_order = np.argsort(idx_i, kind='stable')
    idx_i = idx_i[sort_order]
    idx_j = idx_j[sort_order]
    dist_vec = dist_vec[sort_order]
    dist = dist[sort_order]

    neighbor_pos = coords[idx_i] + dist_vec

    pair_offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    for i in range(n_atoms):
        pair_offsets[i + 1] = pair_offsets[i] + np.sum(idx_i == i)

    return (idx_j.astype(np.int64),
            neighbor_pos.astype(np.float64),
            dist_vec.astype(np.float64),
            dist.astype(np.float64),
            pair_offsets)


@nb.jit(nopython=True)
def generate_data_periodic(size, charges, coods,
                           nbr_j, nbr_pos, nbr_rij, nbr_dist, pair_offsets,
                           rconvs, aconvs, cutoff_r=6.0, n_atm=2.0):
    """
    Compute cMBDF features for a periodic structure using neighbor lists.

    Smooth cutoff is always applied (mandatory for periodic systems).

    Args:
        size: number of atoms
        charges: (size,) nuclear charges
        coods: (size, 3) coordinates
        nbr_j: (n_pairs,) neighbor indices
        nbr_pos: (n_pairs, 3) neighbor positions (with image shifts)
        nbr_rij: (n_pairs, 3) displacement vectors i->j
        nbr_dist: (n_pairs,) distances
        pair_offsets: (size+1,) start/end indices per atom
        rconvs, aconvs: convolution grids (first element of tuple from get_convolutions)
        cutoff_r: radial cutoff
        n_atm: ATM damping exponent

    Returns:
        rep: (size, nrs + nAs) feature matrix
    """
    rconvs = rconvs[0]
    aconvs = aconvs[0]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1 * n1
    nAs = m2 * n2
    rstep = cutoff_r / rconvs.shape[-1]
    astep = np.pi / aconvs.shape[-1]

    rep_2b = np.zeros((size, nrs))
    rep_3b = np.zeros((size, nAs))

    for i in range(size):
        z_i = charges[i]
        start_i = pair_offsets[i]
        end_i = pair_offsets[i + 1]
        n_neigh_i = end_i - start_i

        # --- 2-body: sum over neighbors of i ---
        for p in range(start_i, end_i):
            j = nbr_j[p]
            rij_norm = nbr_dist[p]

            if rij_norm > 0 and rij_norm < cutoff_r:
                z_j = charges[j]
                fcutij = 0.5 * (np.cos(np.pi * rij_norm / cutoff_r) + 1.0)
                pref = np.sqrt(z_i * z_j) * fcutij
                ind = int(rij_norm / rstep)
                if ind >= rconvs.shape[-1]:
                    ind = rconvs.shape[-1] - 1

                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        rep_2b[i, id2] += pref * rconvs[i1][i2][ind]
                        id2 += 1

        # --- 3-body: loop over pairs of neighbors (j, k) of i ---
        for p in range(start_i, end_i):
            rij_norm = nbr_dist[p]
            if rij_norm <= 0 or rij_norm >= cutoff_r:
                continue

            j = nbr_j[p]
            rij_vec = nbr_rij[p]
            z_j = charges[j]
            fcutij = 0.5 * (np.cos(np.pi * rij_norm / cutoff_r) + 1.0)

            for q in range(p + 1, end_i):
                rik_norm = nbr_dist[q]
                if rik_norm <= 0 or rik_norm >= cutoff_r:
                    continue

                k = nbr_j[q]
                rik_vec = nbr_rij[q]
                z_k = charges[k]
                fcutik = 0.5 * (np.cos(np.pi * rik_norm / cutoff_r) + 1.0)

                # R_jk: vector from neighbor j image to neighbor k image
                rkj_vec = nbr_pos[q] - nbr_pos[p]
                rkj_norm = np.sqrt(rkj_vec[0]**2 + rkj_vec[1]**2 + rkj_vec[2]**2)

                if rkj_norm < 1e-10:
                    continue

                fcutjk = 0.5 * (np.cos(np.pi * min(rkj_norm, cutoff_r) / cutoff_r) + 1.0)
                if rkj_norm >= cutoff_r:
                    fcutjk = 0.0

                # Three angles of the triangle
                dot_ij_ik = rij_vec[0]*rik_vec[0] + rij_vec[1]*rik_vec[1] + rij_vec[2]*rik_vec[2]
                cos1 = min(1.0, max(dot_ij_ik / (rij_norm * rik_norm), -1.0))

                dot_ij_kj = rij_vec[0]*rkj_vec[0] + rij_vec[1]*rkj_vec[1] + rij_vec[2]*rkj_vec[2]
                cos2 = min(1.0, max(dot_ij_kj / (rij_norm * rkj_norm), -1.0))

                dot_mk_ik = -rkj_vec[0]*rik_vec[0] - rkj_vec[1]*rik_vec[1] - rkj_vec[2]*rik_vec[2]
                cos3 = min(1.0, max(dot_mk_ik / (rkj_norm * rik_norm), -1.0))

                ang1 = np.arccos(cos1)
                ang2 = np.arccos(cos2)
                ang3 = np.arccos(cos3)

                ind1 = int(ang1 / astep)
                ind2 = int(ang2 / astep)
                ind3 = int(ang3 / astep)

                # Clamp to grid bounds
                max_ang_idx = aconvs.shape[-1] - 1
                ind1 = min(ind1, max_ang_idx)
                ind2 = min(ind2, max_ang_idx)
                ind3 = min(ind3, max_ang_idx)

                atm = (rij_norm * rik_norm * rkj_norm) ** n_atm

                charge_3b = (z_i * z_j * z_k) ** (1.0 / 3.0)
                pref = charge_3b * fcutij * fcutik * fcutjk

                id2 = 0
                for i1 in range(m2):
                    for i2 in range(n2):
                        v1 = aconvs[i1][i2][ind1]
                        v2 = aconvs[i1][i2][ind2]
                        v3 = aconvs[i1][i2][ind3]

                        if i2 == 0:
                            c1 = (pref * v1 * cos2 * cos3) / atm
                            c2 = (pref * v2 * cos1 * cos3) / atm
                            c3 = (pref * v3 * cos2 * cos1) / atm
                        else:
                            c1 = (pref * v1) / atm
                            c2 = (pref * v2) / atm
                            c3 = (pref * v3) / atm

                        # Atom i gets all three angle contributions
                        # (both symmetric slots, same as molecular version)
                        rep_3b[i, id2] += 2.0 * c1

                        id2 += 1

                # Note on 3-body accumulation:
                # In the molecular version, each unique triplet (i<j<k) distributes
                # contributions to atoms i, j, and k. Here, we loop over neighbors
                # of atom i, so atom i is always the central atom. Each atom
                # accumulates its OWN 3-body features from its own neighbor pairs.
                # The factor 2.0 accounts for the two symmetric slots (j,k) and (k,j)
                # in the original threeb[i][j][k] and threeb[i][k][j] arrays.

    return np.hstack((rep_2b, rep_3b))


@nb.jit(nopython=True)
def generate_data_periodic_elemspec(size, charges, coods,
                                     nbr_j, nbr_pos, nbr_rij, nbr_dist, pair_offsets,
                                     rconvs_arr, aconvs, elem_indices,
                                     cutoff_r=6.0, n_atm=2.0):
    """
    Element-specific variant: uses per-element radial convolution grids.

    rconvs_arr: (n_elems, m1, n1, grid_r) — one set per element
    elem_indices: (size,) int array — maps each atom to its element index in rconvs_arr
    """
    m1, n1 = rconvs_arr.shape[1], rconvs_arr.shape[2]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1 * n1
    nAs = m2 * n2
    rstep = cutoff_r / rconvs_arr.shape[-1]
    astep = np.pi / aconvs.shape[-1]

    rep_2b = np.zeros((size, nrs))
    rep_3b = np.zeros((size, nAs))

    for i in range(size):
        z_i = charges[i]
        start_i = pair_offsets[i]
        end_i = pair_offsets[i + 1]

        # --- 2-body ---
        for p in range(start_i, end_i):
            j = nbr_j[p]
            rij_norm = nbr_dist[p]

            if rij_norm > 0 and rij_norm < cutoff_r:
                z_j = charges[j]
                fcutij = 0.5 * (np.cos(np.pi * rij_norm / cutoff_r) + 1.0)
                pref = np.sqrt(z_i * z_j) * fcutij
                ind = int(rij_norm / rstep)
                if ind >= rconvs_arr.shape[-1]:
                    ind = rconvs_arr.shape[-1] - 1

                # Use neighbor j's element-specific convolution
                j_elem_idx = elem_indices[j]

                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        rep_2b[i, id2] += pref * rconvs_arr[j_elem_idx][i1][i2][ind]
                        id2 += 1

        # --- 3-body (identical to universal version) ---
        for p in range(start_i, end_i):
            rij_norm = nbr_dist[p]
            if rij_norm <= 0 or rij_norm >= cutoff_r:
                continue

            j = nbr_j[p]
            rij_vec = nbr_rij[p]
            z_j = charges[j]
            fcutij = 0.5 * (np.cos(np.pi * rij_norm / cutoff_r) + 1.0)

            for q in range(p + 1, end_i):
                rik_norm = nbr_dist[q]
                if rik_norm <= 0 or rik_norm >= cutoff_r:
                    continue

                k = nbr_j[q]
                rik_vec = nbr_rij[q]
                z_k = charges[k]
                fcutik = 0.5 * (np.cos(np.pi * rik_norm / cutoff_r) + 1.0)

                rkj_vec = nbr_pos[q] - nbr_pos[p]
                rkj_norm = np.sqrt(rkj_vec[0]**2 + rkj_vec[1]**2 + rkj_vec[2]**2)

                if rkj_norm < 1e-10:
                    continue

                fcutjk = 0.5 * (np.cos(np.pi * min(rkj_norm, cutoff_r) / cutoff_r) + 1.0)
                if rkj_norm >= cutoff_r:
                    fcutjk = 0.0

                dot_ij_ik = rij_vec[0]*rik_vec[0] + rij_vec[1]*rik_vec[1] + rij_vec[2]*rik_vec[2]
                cos1 = min(1.0, max(dot_ij_ik / (rij_norm * rik_norm), -1.0))

                dot_ij_kj = rij_vec[0]*rkj_vec[0] + rij_vec[1]*rkj_vec[1] + rij_vec[2]*rkj_vec[2]
                cos2 = min(1.0, max(dot_ij_kj / (rij_norm * rkj_norm), -1.0))

                dot_mk_ik = -rkj_vec[0]*rik_vec[0] - rkj_vec[1]*rik_vec[1] - rkj_vec[2]*rik_vec[2]
                cos3 = min(1.0, max(dot_mk_ik / (rkj_norm * rik_norm), -1.0))

                ang1 = np.arccos(cos1)
                ang2 = np.arccos(cos2)
                ang3 = np.arccos(cos3)

                ind1 = min(int(ang1 / astep), aconvs.shape[-1] - 1)
                ind2 = min(int(ang2 / astep), aconvs.shape[-1] - 1)
                ind3 = min(int(ang3 / astep), aconvs.shape[-1] - 1)

                atm = (rij_norm * rik_norm * rkj_norm) ** n_atm
                charge_3b = (z_i * z_j * z_k) ** (1.0 / 3.0)
                pref = charge_3b * fcutij * fcutik * fcutjk

                id2 = 0
                for i1 in range(m2):
                    for i2 in range(n2):
                        v1 = aconvs[i1][i2][ind1]
                        v2 = aconvs[i1][i2][ind2]
                        v3 = aconvs[i1][i2][ind3]

                        if i2 == 0:
                            c1 = (pref * v1 * cos2 * cos3) / atm
                        else:
                            c1 = (pref * v1) / atm

                        rep_3b[i, id2] += 2.0 * c1
                        id2 += 1

    return np.hstack((rep_2b, rep_3b))


def get_cmbdf_periodic(charges, coords, cell, pbc, convs,
                       pad=None, rcut=6.0, n_atm=2.0,
                       elem_specific=False, elem_convs=None, elem_to_idx=None):
    """
    Returns the local cMBDF representation for a periodic structure.

    Args:
        charges: (N,) nuclear charges
        coords: (N, 3) Cartesian coordinates
        cell: (3, 3) lattice vectors
        pbc: (3,) or bool — periodic boundary conditions
        convs: (rconvs, aconvs) from get_convolutions (universal mode)
        pad: zero-pad to this many atoms
        rcut: radial cutoff (default 6.0 for solids)
        n_atm: ATM damping exponent
        elem_specific: if True, use element-specific radial convolutions
        elem_convs: (rconvs_arr, aconvs) from get_convolutions_element_specific
        elem_to_idx: dict mapping Z -> index in rconvs_arr

    Returns:
        (pad, n_features) array
    """
    size = len(charges)
    if pad is None:
        pad = size

    # Build neighbor list
    nbr_j, nbr_pos, nbr_rij, nbr_dist, pair_offsets = build_neighbor_data(
        charges, coords, cell, pbc, rcut)

    if elem_specific and elem_convs is not None:
        rconvs_arr, aconvs = elem_convs
        # Build elem_indices array for Numba
        elem_indices = np.array([elem_to_idx[int(z)] for z in charges], dtype=np.int64)
        m1, n1 = rconvs_arr.shape[1], rconvs_arr.shape[2]
        m2, n2 = aconvs.shape[0], aconvs.shape[1]
        desc_size = m1 * n1 + m2 * n2

        rep = generate_data_periodic_elemspec(
            size, charges, coords, nbr_j, nbr_pos, nbr_rij, nbr_dist,
            pair_offsets, rconvs_arr, aconvs, elem_indices, rcut, n_atm)
    else:
        rconvs, aconvs = convs
        m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
        m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]
        desc_size = m1 * n1 + m2 * n2

        rep = generate_data_periodic(
            size, charges, coords, nbr_j, nbr_pos, nbr_rij, nbr_dist,
            pair_offsets, rconvs, aconvs, rcut, n_atm)

    # Zero-pad
    if pad > size:
        rep = np.vstack([rep, np.zeros((pad - size, desc_size))])

    return rep


def normalize_per_element(reps, nuclear_charges, mode='mean'):
    """
    Normalize cMBDF features per element across a dataset.

    For each element Z and each feature f, divides by the mean (or max)
    of that feature across all atoms of element Z in the training set.

    Args:
        reps: (n_structures, max_atoms, n_features) array
        nuclear_charges: list of arrays of nuclear charges
        mode: 'mean' or 'max'

    Returns:
        reps_norm: normalized array (same shape)
        norm_factors: dict mapping Z -> (n_features,) array of normalization factors
                      (save this for applying to test data)
    """
    n_feat = reps.shape[-1]

    # Collect per-element feature values
    elem_features = {}
    for i, charges in enumerate(nuclear_charges):
        n_atoms = len(charges)
        for a in range(n_atoms):
            z = int(charges[a])
            if z not in elem_features:
                elem_features[z] = []
            elem_features[z].append(reps[i, a, :])

    # Compute normalization factors
    norm_factors = {}
    for z, feats in elem_features.items():
        feats_arr = np.array(feats)
        if mode == 'mean':
            factors = np.mean(np.abs(feats_arr), axis=0)
        elif mode == 'max':
            factors = np.max(np.abs(feats_arr), axis=0)
        else:
            raise ValueError("mode must be 'mean' or 'max'")
        # Avoid division by zero
        factors[factors == 0] = 1.0
        norm_factors[z] = factors

    # Apply normalization
    reps_norm = reps.copy()
    for i, charges in enumerate(nuclear_charges):
        n_atoms = len(charges)
        for a in range(n_atoms):
            z = int(charges[a])
            reps_norm[i, a, :] = reps[i, a, :] / norm_factors[z]

    return reps_norm, norm_factors


def apply_normalization(reps, nuclear_charges, norm_factors):
    """
    Apply pre-computed normalization factors to new data.

    Args:
        reps: (n_structures, max_atoms, n_features) array
        nuclear_charges: list of arrays of nuclear charges
        norm_factors: dict from normalize_per_element

    Returns:
        reps_norm: normalized array
    """
    reps_norm = reps.copy()
    for i, charges in enumerate(nuclear_charges):
        n_atoms = len(charges)
        for a in range(n_atoms):
            z = int(charges[a])
            if z in norm_factors:
                reps_norm[i, a, :] = reps[i, a, :] / norm_factors[z]
    return reps_norm


from joblib import Parallel, delayed

def generate_mbdf_periodic(nuclear_charges, coords, cells, pbc=(True, True, True),
                           convs='None', alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                           n_jobs=-1, a1=2.0, pad=None, rstep=0.0008,
                           rcut=6.0, astep=0.0002, nAs=4, order=4,
                           progress_bar=False, a2=2.0, n_atm=2.0,
                           elem_specific=False):
    """
    Generate periodic cMBDF representations for a batch of structures.

    Args:
        nuclear_charges: list of arrays of nuclear charges
        coords: list of arrays of Cartesian coordinates
        cells: list of (3,3) arrays of lattice vectors
        pbc: tuple of 3 bools, or list of tuples (per structure)
        convs: precomputed convolutions (or 'None' to compute)
        rcut: radial cutoff (default 6.0 Å for solids)
        n_atm: ATM damping exponent
        elem_specific: use element-specific radial basis (vdW radii)
        (remaining: convolution hyperparameters)

    Returns:
        (n_structures, pad, n_features) array
    """
    lengths, charges = [], []

    for i in range(len(nuclear_charges)):
        q, r = nuclear_charges[i], coords[i]
        assert q.shape[0] == r.shape[0], \
            "charges and coordinates length mismatch at index %d" % i
        lengths.append(len(q))
        charges.append(q.astype(np.float64))

    if pad is None:
        pad = max(lengths)

    # Compute convolutions
    elem_convs = None
    elem_to_idx = None

    if elem_specific:
        all_elems = np.unique(np.concatenate(charges))
        rconvs_arr, aconvs_es, elem_to_idx = get_convolutions_element_specific(
            all_elems, rstep, rcut, alpha_list, n_list, order, a2, astep, nAs)
        elem_convs = (rconvs_arr, aconvs_es)

    if type(convs) == str and not elem_specific:
        convs = get_convolutions(rstep, rcut, alpha_list, n_list, order,
                                 a1, a2, astep, nAs, gradients=False)

    # Handle pbc: single tuple for all, or per-structure
    if isinstance(pbc, (list, np.ndarray)) and len(pbc) == len(nuclear_charges):
        pbc_list = pbc
    else:
        pbc_list = [pbc] * len(nuclear_charges)

    coords_list = [np.asarray(r, dtype=np.float64) for r in coords]
    cells_list = [np.asarray(c, dtype=np.float64) for c in cells]

    if progress_bar:
        from tqdm import tqdm
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf_periodic)(
                charge, cood, cell, pbc_i, convs, pad, rcut, n_atm,
                elem_specific, elem_convs, elem_to_idx)
            for charge, cood, cell, pbc_i in tqdm(
                list(zip(charges, coords_list, cells_list, pbc_list))))
    else:
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf_periodic)(
                charge, cood, cell, pbc_i, convs, pad, rcut, n_atm,
                elem_specific, elem_convs, elem_to_idx)
            for charge, cood, cell, pbc_i in zip(
                charges, coords_list, cells_list, pbc_list))

    return np.asarray(reps)
