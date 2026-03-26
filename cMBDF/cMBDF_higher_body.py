"""
True 4-body (dihedral) and 5-body (improper torsion) extensions for periodic cMBDF.

Extends cMBDF_periodic.py with:
- True 4-body: dihedral angles from quadruplets (i,j,k,l)
- True 5-body: out-of-plane angles from quintuplets (toggleable)

Both use precomputed FFT convolution kernels on the appropriate angular domains.
Designed for GPU readiness (vectorizable with scatter_add).

Usage:
    from cMBDF_higher_body import generate_mbdf_periodic_higher

    reps = generate_mbdf_periodic_higher(
        charges, coords, cells, pbc=(True,True,True),
        rcut=6.0, max_body=4,  # or max_body=5
    )
"""

import numpy as np
import numba as nb
from scipy.signal import fftconvolve
from scipy.fft import next_fast_len

from cMBDF_periodic import (
    build_neighbor_data, generate_data_periodic,
    get_convolutions, normalize_per_element
)
from cMBDF import hermite_polynomial


def get_dihedral_convolutions(rstep=0.001, order=4, a_dih=1.0, n_dih=4, device='cpu'):
    """
    Compute convolution kernels for the dihedral angle domain [-π, π].

    The dihedral angle φ ∈ [-π, π] is convolved with:
    - Hermite-Gaussian basis functions (derivative orders 0..order)
    - Fourier weighting functions cos(nφ) for n=1..n_dih

    Returns:
        dconvs: (order+1, n_dih, grid_size) array of convolution values
        meta_dih: dict with step size and grid info
    """
    # Solid angle domain [0, 2π]
    dstep = 2 * np.pi / next_fast_len(int(2 * np.pi / rstep))
    dgrid = np.arange(0.0, 2 * np.pi, dstep)
    dgrid2 = np.arange(-2 * np.pi, 2 * np.pi, dstep)

    size = len(dgrid)
    gaussian = np.exp(-a_dih * (dgrid2 ** 2)) * np.sqrt(a_dih / np.pi)

    m = order + 1
    fms = [gaussian] + [
        gaussian * hermite_polynomial(dgrid2, i, np.sqrt(a_dih)) * ((-np.sqrt(a_dih)) ** m)
        for i in range(1, m + 1)
    ]

    temp = []
    for i in range(m):
        fm = fms[i]
        t = []
        for n in range(1, n_dih + 1):
            gn = np.cos(n * dgrid)
            arr = fftconvolve(gn, fm, mode='same') * dstep
            arr = arr / np.max(np.abs(arr))
            t.append(arr)
        temp.append(np.array(t))

    dconvs = np.asarray(temp)
    meta_dih = {'dstep': dstep, 'grid_size': size, 'offset': np.pi}  # φ + π maps [-π,π] -> [0, 2π]

    return dconvs, meta_dih


def get_fivebody_convolutions(rstep=0.001, order=3, a_5b=1.0, n_5b=3):
    """
    Compute convolution kernels for 5-body out-of-plane angle domain [0, π].

    The out-of-plane angle ψ ∈ [0, π] is the angle between the vector r_im
    and the plane formed by (r_ij, r_ik, r_il).

    Returns:
        fconvs: (order+1, n_5b, grid_size) array
        meta_5b: dict with step info
    """
    fstep = np.pi / next_fast_len(int(np.pi / rstep))
    fgrid = np.arange(0.0, np.pi, fstep)
    fgrid2 = np.arange(-np.pi, np.pi, fstep)

    size = len(fgrid)
    gaussian = np.exp(-a_5b * (fgrid2 ** 2)) * np.sqrt(a_5b / np.pi)

    m = order + 1
    fms = [gaussian] + [
        gaussian * hermite_polynomial(fgrid2, i, np.sqrt(a_5b)) * ((-np.sqrt(a_5b)) ** m)
        for i in range(1, m + 1)
    ]

    temp = []
    for i in range(m):
        fm = fms[i]
        t = []
        for n in range(1, n_5b + 1):
            gn = np.cos(n * fgrid)
            arr = fftconvolve(gn, fm, mode='same') * fstep
            arr = arr / np.max(np.abs(arr))
            t.append(arr)
        temp.append(np.array(t))

    fconvs = np.asarray(temp)
    meta_5b = {'fstep': fstep, 'grid_size': size}

    return fconvs, meta_5b


@nb.jit(nopython=True)
def compute_dihedral(r_ij, r_ik, r_jk, r_jl):
    """
    Compute dihedral angle between planes (i,j,k) and (j,k,l).

    The dihedral is the angle between:
    n1 = r_ij × r_ik  (normal to plane through i,j,k)
    n2 = r_jk × r_jl  (normal to plane through j,k,l)

    Returns angle in [-π, π].
    """
    # Normal to plane ijk: n1 = r_ji × r_jk = -r_ij × (r_ik - r_ij)
    # Simpler: use the standard definition via sequential bonds
    # For atoms i-j-k-l, dihedral is around the j-k bond
    # b1 = j-i, b2 = k-j, b3 = l-k
    b1 = -r_ij  # j -> i
    b2 = r_jk   # j -> k
    b3 = r_jl - r_jk  # k -> l

    # Normal vectors
    n1 = np.array([
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0]
    ])
    n2 = np.array([
        b2[1] * b3[2] - b2[2] * b3[1],
        b2[2] * b3[0] - b2[0] * b3[2],
        b2[0] * b3[1] - b2[1] * b3[0]
    ])

    n1_norm = np.sqrt(n1[0]**2 + n1[1]**2 + n1[2]**2)
    n2_norm = np.sqrt(n2[0]**2 + n2[1]**2 + n2[2]**2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
    cos_phi = min(1.0, max(-1.0, cos_phi))

    # Sign from cross product
    cross = np.array([
        n1[1]*n2[2] - n1[2]*n2[1],
        n1[2]*n2[0] - n1[0]*n2[2],
        n1[0]*n2[1] - n1[1]*n2[0]
    ])
    b2_norm = np.sqrt(b2[0]**2 + b2[1]**2 + b2[2]**2)
    if b2_norm > 0:
        sign = cross[0]*b2[0] + cross[1]*b2[1] + cross[2]*b2[2]
    else:
        sign = 0.0

    phi = np.arccos(cos_phi)
    if sign < 0:
        phi = -phi

    return phi


@nb.jit(nopython=True)
def compute_oop_angle(r_ij, r_ik, r_il, r_im):
    """
    Compute out-of-plane angle for 5-body term.

    The angle ψ between vector r_im and the plane defined by (r_ij, r_ik, r_il).
    ψ = π/2 - arcsin(|r_im · n| / |r_im|) where n is the plane normal.

    Returns angle in [0, π].
    """
    # Normal to plane (ij, ik, il)
    v1 = r_ik - r_ij
    v2 = r_il - r_ij
    n = np.array([
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ])
    n_norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    rm_norm = np.sqrt(r_im[0]**2 + r_im[1]**2 + r_im[2]**2)

    if n_norm < 1e-10 or rm_norm < 1e-10:
        return 0.0

    # Angle between r_im and the plane = arccos(|r_im · n| / (|r_im| |n|))
    dot = abs(r_im[0]*n[0] + r_im[1]*n[1] + r_im[2]*n[2])
    cos_psi = min(1.0, dot / (rm_norm * n_norm))

    return np.arccos(cos_psi)


@nb.jit(nopython=True)
def generate_4body_periodic(size, charges, nbr_j, nbr_pos, nbr_rij, nbr_dist,
                             pair_offsets, dconvs, cutoff_r=6.0, n_atm=1.0):
    """
    Compute true 4-body features using rotationally invariant tetrahedron volume.

    For each unique triplet of neighbors (j,k,l) of central atom i, the
    4-body invariant is the solid angle Ω subtended by the triangle (j,k,l)
    as seen from i:
        Ω = 2·arctan(|r_ij · (r_ik × r_il)| / D)
    where D = r_ij·r_ik·r_il + (r_ij·r_ik)·r_il + (r_ij·r_il)·r_ik + (r_ik·r_il)·r_ij

    This is a proper 4-body rotational invariant that doesn't depend on
    neighbor ordering. Ω ∈ [0, 2π].
    """
    m_d, n_d = dconvs.shape[0], dconvs.shape[1]
    n_4b = m_d * n_d
    grid_d = dconvs.shape[2]
    # Convolutions defined on [0, 2π] for solid angle
    omega_step = 2.0 * np.pi / grid_d

    rep_4b = np.zeros((size, n_4b))

    for i in range(size):
        z_i = charges[i]
        start_i = pair_offsets[i]
        end_i = pair_offsets[i + 1]

        if end_i - start_i < 3:
            continue

        for p in range(start_i, end_i):
            if nbr_dist[p] <= 0 or nbr_dist[p] >= cutoff_r:
                continue
            rij = nbr_rij[p]
            rij_n = nbr_dist[p]
            z_j = charges[nbr_j[p]]
            fcij = 0.5 * (np.cos(np.pi * rij_n / cutoff_r) + 1.0)

            # Unit vectors
            uij = rij / rij_n

            for q in range(p + 1, end_i):
                if nbr_dist[q] <= 0 or nbr_dist[q] >= cutoff_r:
                    continue
                rik = nbr_rij[q]
                rik_n = nbr_dist[q]
                z_k = charges[nbr_j[q]]
                fcik = 0.5 * (np.cos(np.pi * rik_n / cutoff_r) + 1.0)

                uik = rik / rik_n

                for s in range(q + 1, end_i):
                    if nbr_dist[s] <= 0 or nbr_dist[s] >= cutoff_r:
                        continue
                    ril = nbr_rij[s]
                    ril_n = nbr_dist[s]
                    z_l = charges[nbr_j[s]]
                    fcil = 0.5 * (np.cos(np.pi * ril_n / cutoff_r) + 1.0)

                    uil = ril / ril_n

                    # Solid angle via Van Oosterom-Strackee formula
                    # Triple product: uij · (uik × uil)
                    cross_kl = np.array([
                        uik[1]*uil[2] - uik[2]*uil[1],
                        uik[2]*uil[0] - uik[0]*uil[2],
                        uik[0]*uil[1] - uik[1]*uil[0]
                    ])
                    numerator = abs(uij[0]*cross_kl[0] + uij[1]*cross_kl[1] + uij[2]*cross_kl[2])

                    # Dot products
                    dij_ik = uij[0]*uik[0] + uij[1]*uik[1] + uij[2]*uik[2]
                    dij_il = uij[0]*uil[0] + uij[1]*uil[1] + uij[2]*uil[2]
                    dik_il = uik[0]*uil[0] + uik[1]*uil[1] + uik[2]*uil[2]

                    denominator = 1.0 + dij_ik + dij_il + dik_il
                    if denominator < 1e-10:
                        denominator = 1e-10

                    omega = 2.0 * np.arctan2(numerator, denominator)
                    # omega ∈ [0, 2π]

                    idx = int(omega / omega_step)
                    if idx >= grid_d:
                        idx = grid_d - 1
                    if idx < 0:
                        idx = 0

                    # Charge and damping
                    charge = (z_i * z_j * z_k * z_l) ** 0.25
                    atm = (rij_n * rik_n * ril_n) ** n_atm
                    pref = charge * fcij * fcik * fcil / atm

                    id2 = 0
                    for i1 in range(m_d):
                        for i2 in range(n_d):
                            rep_4b[i, id2] += pref * dconvs[i1][i2][idx]
                            id2 += 1

    return rep_4b


@nb.jit(nopython=True)
def _accumulate_5b(rep_5b, i, pref, psi, fconvs, m_f, n_f, grid_f, fstep):
    """Accumulate one OOP angle contribution."""
    idx = int(psi / fstep)
    if idx >= grid_f:
        idx = grid_f - 1
    if idx < 0:
        idx = 0
    id2 = 0
    for i1 in range(m_f):
        for i2 in range(n_f):
            rep_5b[i, id2] += pref * fconvs[i1][i2][idx]
            id2 += 1


@nb.jit(nopython=True)
def generate_5body_periodic(size, charges, nbr_j, nbr_pos, nbr_rij, nbr_dist,
                             pair_offsets, fconvs, cutoff_r=6.0, n_atm=1.0):
    """
    Compute true 5-body (out-of-plane) features for a periodic structure.

    For each central atom i, enumerate all ordered quadruplets (j,k,l,m) from
    i's neighbor list. Compute the out-of-plane angle of m relative to the
    plane (i,j,k,l). Accumulate features.
    """
    m_f, n_f = fconvs.shape[0], fconvs.shape[1]
    n_5b = m_f * n_f
    grid_f = fconvs.shape[2]
    fstep = np.pi / grid_f

    rep_5b = np.zeros((size, n_5b))

    for i in range(size):
        z_i = charges[i]
        start_i = pair_offsets[i]
        end_i = pair_offsets[i + 1]
        n_neigh = end_i - start_i

        if n_neigh < 4:
            continue

        for p in range(start_i, end_i):
            if nbr_dist[p] <= 0 or nbr_dist[p] >= cutoff_r:
                continue
            rij = nbr_rij[p]
            rij_n = nbr_dist[p]
            z_j = charges[nbr_j[p]]
            fcij = 0.5 * (np.cos(np.pi * rij_n / cutoff_r) + 1.0)

            for q in range(p + 1, end_i):
                if nbr_dist[q] <= 0 or nbr_dist[q] >= cutoff_r:
                    continue
                rik = nbr_rij[q]
                rik_n = nbr_dist[q]
                z_k = charges[nbr_j[q]]
                fcik = 0.5 * (np.cos(np.pi * rik_n / cutoff_r) + 1.0)

                for s in range(q + 1, end_i):
                    if nbr_dist[s] <= 0 or nbr_dist[s] >= cutoff_r:
                        continue
                    ril = nbr_rij[s]
                    ril_n = nbr_dist[s]
                    z_l = charges[nbr_j[s]]
                    fcil = 0.5 * (np.cos(np.pi * ril_n / cutoff_r) + 1.0)

                    for t in range(s + 1, end_i):
                        if nbr_dist[t] <= 0 or nbr_dist[t] >= cutoff_r:
                            continue
                        rim = nbr_rij[t]
                        rim_n = nbr_dist[t]
                        z_m = charges[nbr_j[t]]
                        fcim = 0.5 * (np.cos(np.pi * rim_n / cutoff_r) + 1.0)

                        charge = (z_i * z_j * z_k * z_l * z_m) ** 0.2
                        atm = (rij_n * rik_n * ril_n * rim_n) ** n_atm
                        pref = charge * fcij * fcik * fcil * fcim / atm

                        # 5-body invariant: use the angle between r_im
                        # and the plane (r_ij, r_ik, r_il) — this is
                        # symmetric in j,k,l but distinguishes m.
                        # Accumulate all 4 choices of the out-of-plane atom.
                        _accumulate_5b(rep_5b, i, pref,
                                       compute_oop_angle(rij, rik, ril, rim),
                                       fconvs, m_f, n_f, grid_f, fstep)
                        _accumulate_5b(rep_5b, i, pref,
                                       compute_oop_angle(rij, rik, rim, ril),
                                       fconvs, m_f, n_f, grid_f, fstep)
                        _accumulate_5b(rep_5b, i, pref,
                                       compute_oop_angle(rij, ril, rim, rik),
                                       fconvs, m_f, n_f, grid_f, fstep)
                        _accumulate_5b(rep_5b, i, pref,
                                       compute_oop_angle(rik, ril, rim, rij),
                                       fconvs, m_f, n_f, grid_f, fstep)

    return rep_5b


def get_cmbdf_higher_body(charges, coords, cell, pbc, convs,
                           dconvs=None, fconvs=None, meta_dih=None, meta_5b=None,
                           pad=None, rcut=6.0, n_atm=2.0, n_atm_4b=1.0,
                           max_body=4):
    """
    Compute p-cMBDF with optional 4-body and 5-body terms.

    Args:
        max_body: 3 (standard), 4 (+ dihedral), or 5 (+ out-of-plane)
    """
    size = len(charges)
    if pad is None:
        pad = size

    # Build neighbor list
    nbr_j, nbr_pos, nbr_rij, nbr_dist, pair_offsets = build_neighbor_data(
        charges, coords, cell, pbc, rcut)

    # 2+3 body (standard periodic cMBDF)
    rconvs, aconvs = convs
    rep_23 = generate_data_periodic(
        size, charges, coords, nbr_j, nbr_pos, nbr_rij, nbr_dist,
        pair_offsets, rconvs, aconvs, rcut, n_atm)

    parts = [rep_23]

    # 4-body (dihedral)
    if max_body >= 4 and dconvs is not None:
        rep_4b = generate_4body_periodic(
            size, charges, nbr_j, nbr_pos, nbr_rij, nbr_dist,
            pair_offsets, dconvs, rcut, n_atm_4b)
        parts.append(rep_4b)

    # 5-body (out-of-plane)
    if max_body >= 5 and fconvs is not None:
        rep_5b = generate_5body_periodic(
            size, charges, nbr_j, nbr_pos, nbr_rij, nbr_dist,
            pair_offsets, fconvs, rcut, n_atm_4b)
        parts.append(rep_5b)

    rep = np.hstack(parts)

    if pad > size:
        rep = np.vstack([rep, np.zeros((pad - size, rep.shape[1]))])

    return rep


from joblib import Parallel, delayed

def generate_mbdf_periodic_higher(nuclear_charges, coords, cells,
                                    pbc=(True, True, True),
                                    rcut=6.0, n_atm=2.0, n_atm_4b=1.0,
                                    max_body=4, order_4b=4, n_dih=4,
                                    order_5b=3, n_5b=3,
                                    n_jobs=-1, pad=None,
                                    progress_bar=False,
                                    **conv_kwargs):
    """
    Generate periodic cMBDF with higher body-order terms.

    Args:
        max_body: 3 (40 dim), 4 (40 + 4b_dim), or 5 (40 + 4b_dim + 5b_dim)
        order_4b: derivative order for 4-body convolutions (default 4 -> 5 orders)
        n_dih: number of dihedral weighting functions (default 4)
        order_5b: derivative order for 5-body (default 3 -> 4 orders)
        n_5b: number of 5-body weighting functions (default 3)

    Feature dimensions:
        3-body: 40 (5 orders × 4 radial + 5 orders × 4 angular)
        4-body: (order_4b+1) × n_dih (default: 5×4 = 20)
        5-body: (order_5b+1) × n_5b (default: 4×3 = 12)
        Total with max_body=4: 60
        Total with max_body=5: 72
    """
    lengths, charges_list = [], []
    for i in range(len(nuclear_charges)):
        q = nuclear_charges[i]
        lengths.append(len(q))
        charges_list.append(q.astype(np.float64))

    if pad is None:
        pad = max(lengths)

    # Standard 2+3 body convolutions
    rstep = conv_kwargs.get('rstep', 0.0008)
    alpha_list = conv_kwargs.get('alpha_list', [1.5, 5.0])
    n_list = conv_kwargs.get('n_list', [3.0, 5.0])
    order = conv_kwargs.get('order', 4)
    a1 = conv_kwargs.get('a1', 2.0)
    a2 = conv_kwargs.get('a2', 2.0)
    astep = conv_kwargs.get('astep', 0.0002)
    nAs = conv_kwargs.get('nAs', 4)

    convs = get_convolutions(rstep, rcut, alpha_list, n_list, order,
                             a1, a2, astep, nAs, gradients=False)

    # 4-body convolutions
    dconvs, meta_dih = None, None
    if max_body >= 4:
        dconvs, meta_dih = get_dihedral_convolutions(
            rstep=0.001, order=order_4b, a_dih=1.0, n_dih=n_dih)

    # 5-body convolutions
    fconvs, meta_5b = None, None
    if max_body >= 5:
        fconvs, meta_5b = get_fivebody_convolutions(
            rstep=0.001, order=order_5b, a_5b=1.0, n_5b=n_5b)

    # PBC handling
    if isinstance(pbc, (list, np.ndarray)) and len(pbc) == len(nuclear_charges):
        pbc_list = pbc
    else:
        pbc_list = [pbc] * len(nuclear_charges)

    coords_list = [np.asarray(r, dtype=np.float64) for r in coords]
    cells_list = [np.asarray(c, dtype=np.float64) for c in cells]

    if progress_bar:
        from tqdm import tqdm
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf_higher_body)(
                charge, cood, cell, pbc_i, convs, dconvs, fconvs,
                meta_dih, meta_5b, pad, rcut, n_atm, n_atm_4b, max_body)
            for charge, cood, cell, pbc_i in tqdm(
                list(zip(charges_list, coords_list, cells_list, pbc_list))))
    else:
        reps = Parallel(n_jobs=n_jobs)(
            delayed(get_cmbdf_higher_body)(
                charge, cood, cell, pbc_i, convs, dconvs, fconvs,
                meta_dih, meta_5b, pad, rcut, n_atm, n_atm_4b, max_body)
            for charge, cood, cell, pbc_i in zip(
                charges_list, coords_list, cells_list, pbc_list))

    return np.asarray(reps)
