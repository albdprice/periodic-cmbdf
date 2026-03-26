"""
Numba-compiled periodic neighbor list.

Replaces ASE NeighborList with a pure Numba implementation for speed.
Uses fractional coordinates and image cell enumeration.

Drop-in replacement for build_neighbor_data in cMBDF_periodic.py.
"""
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def _build_neighbor_list(positions, cell, inv_cell, pbc, cutoff, n_atoms):
    """
    Core neighbor list construction in Numba.

    For each atom i, finds all atoms j (including periodic images) within cutoff.

    Args:
        positions: (n_atoms, 3) Cartesian coordinates
        cell: (3, 3) lattice vectors (rows)
        inv_cell: (3, 3) inverse of cell matrix
        pbc: (3,) bool array — periodic in each direction
        cutoff: radial cutoff
        n_atoms: number of atoms

    Returns:
        idx_i, idx_j: (n_pairs,) int arrays of atom indices
        disp_vec: (n_pairs, 3) displacement vectors (i -> j with image)
        dist: (n_pairs,) distances
    """
    # Determine number of image cells needed per direction
    # n_images[d] = ceil(cutoff / perpendicular height along d)
    # Perpendicular height = volume / (area of opposite face)
    # For orthorhombic: just cell diagonal lengths
    # For general: use the reciprocal lattice vector lengths

    # Compute reciprocal lengths (how many images needed)
    rec = np.zeros(3)
    for d in range(3):
        # Cross product of the other two cell vectors
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        cross = np.array([
            cell[d1, 1] * cell[d2, 2] - cell[d1, 2] * cell[d2, 1],
            cell[d1, 2] * cell[d2, 0] - cell[d1, 0] * cell[d2, 2],
            cell[d1, 0] * cell[d2, 1] - cell[d1, 1] * cell[d2, 0]
        ])
        # Volume component
        vol_comp = abs(cell[d, 0] * cross[0] + cell[d, 1] * cross[1] + cell[d, 2] * cross[2])
        cross_norm = np.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)
        if cross_norm > 0:
            height = vol_comp / cross_norm
        else:
            height = cutoff + 1  # degenerate cell, skip this direction
        rec[d] = height

    n_images = np.zeros(3, dtype=np.int64)
    for d in range(3):
        if pbc[d]:
            n_images[d] = int(np.ceil(cutoff / rec[d]))
        else:
            n_images[d] = 0

    # Pre-allocate output arrays (generous initial size, will trim)
    max_pairs = n_atoms * n_atoms * (2 * n_images[0] + 1) * (2 * n_images[1] + 1) * (2 * n_images[2] + 1)
    # Cap at reasonable size to avoid memory issues
    max_pairs = min(max_pairs, n_atoms * 500)

    out_i = np.zeros(max_pairs, dtype=np.int64)
    out_j = np.zeros(max_pairs, dtype=np.int64)
    out_disp = np.zeros((max_pairs, 3), dtype=np.float64)
    out_dist = np.zeros(max_pairs, dtype=np.float64)
    count = 0

    for i in range(n_atoms):
        pos_i = positions[i]

        for j in range(n_atoms):
            pos_j = positions[j]
            base_disp = pos_j - pos_i

            for nx in range(-n_images[0], n_images[0] + 1):
                for ny in range(-n_images[1], n_images[1] + 1):
                    for nz in range(-n_images[2], n_images[2] + 1):
                        # Skip self-interaction in the home cell
                        if i == j and nx == 0 and ny == 0 and nz == 0:
                            continue

                        # Image shift in Cartesian
                        shift = np.float64(nx) * cell[0] + np.float64(ny) * cell[1] + np.float64(nz) * cell[2]
                        disp = base_disp + shift

                        d = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)

                        if d < cutoff and d > 1e-10:
                            if count >= max_pairs:
                                # Need to grow arrays — rare, but handle it
                                new_max = max_pairs * 2
                                new_i = np.zeros(new_max, dtype=np.int64)
                                new_j = np.zeros(new_max, dtype=np.int64)
                                new_disp = np.zeros((new_max, 3), dtype=np.float64)
                                new_dist = np.zeros(new_max, dtype=np.float64)
                                new_i[:max_pairs] = out_i
                                new_j[:max_pairs] = out_j
                                new_disp[:max_pairs] = out_disp
                                new_dist[:max_pairs] = out_dist
                                out_i = new_i
                                out_j = new_j
                                out_disp = new_disp
                                out_dist = new_dist
                                max_pairs = new_max

                            out_i[count] = i
                            out_j[count] = j
                            out_disp[count] = disp
                            out_dist[count] = d
                            count += 1

    return out_i[:count], out_j[:count], out_disp[:count], out_dist[:count]


def build_neighbor_data_numba(charges, coords, cell, pbc, cutoff_r):
    """
    Build neighbor list for a periodic structure using Numba.

    Drop-in replacement for build_neighbor_data (ASE version) in cMBDF_periodic.py.

    Returns:
        neighbor_j:     (n_pairs,) int array
        neighbor_pos:   (n_pairs, 3) float array — image-shifted positions
        neighbor_rij:   (n_pairs, 3) float array — displacement vectors
        neighbor_dist:  (n_pairs,) float array — distances
        pair_offsets:   (n_atoms+1,) int array — per-atom pair ranges
    """
    n_atoms = len(charges)
    cell = np.asarray(cell, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)

    # Handle pbc as bool array
    if isinstance(pbc, bool):
        pbc_arr = np.array([pbc, pbc, pbc])
    else:
        pbc_arr = np.array(pbc, dtype=np.bool_)

    inv_cell = np.linalg.inv(cell) if np.linalg.det(cell) != 0 else np.eye(3)

    # Run Numba kernel
    idx_i, idx_j, disp_vec, dist = _build_neighbor_list(
        coords, cell, inv_cell, pbc_arr, cutoff_r, n_atoms)

    # Sort by i for efficient per-atom access (stable sort preserves j order)
    order = np.argsort(idx_i, kind='stable')
    idx_i = idx_i[order]
    idx_j = idx_j[order]
    disp_vec = disp_vec[order]
    dist = dist[order]

    # Image-shifted neighbor positions
    nbr_pos = coords[idx_i] + disp_vec

    # Build pair_offsets
    pair_offsets = np.zeros(n_atoms + 1, dtype=np.int64)
    for i in range(n_atoms):
        pair_offsets[i + 1] = pair_offsets[i] + np.sum(idx_i == i)

    return (idx_j.astype(np.int64),
            nbr_pos.astype(np.float64),
            disp_vec.astype(np.float64),
            dist.astype(np.float64),
            pair_offsets)
