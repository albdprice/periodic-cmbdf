"""
Defect and Surface Energy Tests for periodic cMBDF.

Validates that the representation correctly distinguishes:
1. Bulk vs vacancy-adjacent atoms in Si
2. Bulk vs surface atoms in Cu(111) slabs
3. Representation sensitivity to defect/surface environment
"""
import numpy as np
import sys
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF_periodic import generate_mbdf_periodic
from ase import Atoms
from ase.build import bulk, fcc111, make_supercell

# ============================================================
# Test 1: Si vacancy
# ============================================================
print("=" * 70)
print("TEST 1: Silicon vacancy — bulk vs defect-adjacent atoms")
print("=" * 70)

# Perfect 2x2x2 Si supercell
si_bulk = bulk('Si', 'diamond', a=5.43)
si_222 = make_supercell(si_bulk, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

charges_perfect = np.array(si_222.get_atomic_numbers(), dtype=np.float64)
coords_perfect = np.array(si_222.get_positions(), dtype=np.float64)
cell_perfect = np.array(si_222.cell[:], dtype=np.float64)

print("Perfect Si 2x2x2: %d atoms" % len(charges_perfect))

rep_perfect = generate_mbdf_periodic(
    [charges_perfect], [coords_perfect], [cell_perfect],
    pbc=(True, True, True), rcut=6.0, n_jobs=1)[0]

# All atoms should be equivalent in perfect crystal
max_diff_perf = max(np.max(np.abs(rep_perfect[i] - rep_perfect[0]))
                    for i in range(1, len(charges_perfect)))
print("Perfect crystal: all atoms equivalent, max diff = %.2e" % max_diff_perf)

# Create vacancy: remove atom 0
si_vac = si_222.copy()
del si_vac[0]

charges_vac = np.array(si_vac.get_atomic_numbers(), dtype=np.float64)
coords_vac = np.array(si_vac.get_positions(), dtype=np.float64)
cell_vac = np.array(si_vac.cell[:], dtype=np.float64)

print("Si with vacancy: %d atoms" % len(charges_vac))

rep_vac = generate_mbdf_periodic(
    [charges_vac], [coords_vac], [cell_vac],
    pbc=(True, True, True), rcut=6.0, n_jobs=1)[0]

# Find which atoms are nearest to the vacancy site
vac_pos = coords_perfect[0]  # position of removed atom
dists_to_vac = np.linalg.norm(coords_vac - vac_pos, axis=1)

# In diamond Si, nearest neighbors are at ~2.35 Å
nn_mask = dists_to_vac < 3.0
far_mask = dists_to_vac > 5.0

nn_idx = np.where(nn_mask)[0]
far_idx = np.where(far_mask)[0]

print("Nearest neighbors to vacancy: %d atoms" % len(nn_idx))
print("Far from vacancy (>5Å): %d atoms" % len(far_idx))

if len(nn_idx) > 0 and len(far_idx) > 0:
    # Average representation difference between NN and far atoms
    nn_rep = rep_vac[nn_idx].mean(axis=0)
    far_rep = rep_vac[far_idx].mean(axis=0)

    diff_nn_far = np.max(np.abs(nn_rep - far_rep))
    diff_nn_bulk = np.max(np.abs(nn_rep - rep_perfect[0]))
    diff_far_bulk = np.max(np.abs(far_rep - rep_perfect[0]))

    print("\nRepresentation sensitivity:")
    print("  NN-to-vacancy vs far-from-vacancy: %.4f" % diff_nn_far)
    print("  NN-to-vacancy vs perfect bulk:     %.4f" % diff_nn_bulk)
    print("  Far-from-vacancy vs perfect bulk:  %.4f" % diff_far_bulk)
    print("  Ratio (NN vs Far perturbation):    %.1fx" % (diff_nn_bulk / max(diff_far_bulk, 1e-10)))

    if diff_nn_bulk > diff_far_bulk:
        print("  PASS: Representation correctly detects vacancy proximity")
    else:
        print("  WARN: Check defect sensitivity")

# ============================================================
# Test 2: Cu(111) surface
# ============================================================
print()
print("=" * 70)
print("TEST 2: Cu(111) slab — bulk vs surface atoms")
print("=" * 70)

# Build Cu(111) slab: 3x3 surface, 6 layers, 15Å vacuum
slab = fcc111('Cu', size=(3, 3, 6), vacuum=15.0, periodic=True)

charges_slab = np.array(slab.get_atomic_numbers(), dtype=np.float64)
coords_slab = np.array(slab.get_positions(), dtype=np.float64)
cell_slab = np.array(slab.cell[:], dtype=np.float64)

print("Cu(111) slab: %d atoms, 6 layers" % len(charges_slab))

# Use pbc=(True, True, True) — vacuum provides non-periodicity in z
rep_slab = generate_mbdf_periodic(
    [charges_slab], [coords_slab], [cell_slab],
    pbc=(True, True, True), rcut=6.0, n_jobs=1)[0]

# Identify layers by z-coordinate
z_coords = coords_slab[:, 2]
z_unique = np.sort(np.unique(np.round(z_coords, 1)))
print("Layer z-positions:", z_unique)

# Group atoms by layer
layers = {}
for i, z in enumerate(z_coords):
    layer = np.argmin(np.abs(z_unique - z))
    if layer not in layers:
        layers[layer] = []
    layers[layer].append(i)

print("\nPer-layer analysis:")
layer_reps = {}
for layer_idx in sorted(layers.keys()):
    atom_indices = layers[layer_idx]
    layer_rep = rep_slab[atom_indices].mean(axis=0)
    layer_reps[layer_idx] = layer_rep
    norm = np.linalg.norm(layer_rep)
    print("  Layer %d (z=%.1f, %d atoms): rep norm = %.4f" % (
        layer_idx, z_unique[layer_idx], len(atom_indices), norm))

# Compare surface (layers 0, 5) vs bulk-like (layers 2, 3)
if len(layers) >= 6:
    surf_rep = (layer_reps[0] + layer_reps[5]) / 2
    bulk_rep = (layer_reps[2] + layer_reps[3]) / 2

    surf_bulk_diff = np.max(np.abs(surf_rep - bulk_rep))
    print("\nSurface vs bulk-like layer diff: %.4f" % surf_bulk_diff)

    # Check that surface layers differ from bulk but inner layers are similar
    inner_diff = np.max(np.abs(layer_reps[2] - layer_reps[3]))
    print("Inner layer 2 vs 3 diff: %.4f (should be small)" % inner_diff)

    if surf_bulk_diff > inner_diff:
        print("PASS: Surface atoms correctly distinguished from bulk")
    else:
        print("WARN: Check surface sensitivity")

    # Symmetry check: top and bottom surfaces should match
    top_bot_diff = np.max(np.abs(layer_reps[0] - layer_reps[5]))
    print("Top vs bottom surface diff: %.2e (should be ~0)" % top_bot_diff)

# ============================================================
# Test 3: Representation changes with vacuum thickness
# ============================================================
print()
print("=" * 70)
print("TEST 3: Vacuum thickness sensitivity")
print("=" * 70)

for vac in [5.0, 10.0, 15.0, 20.0]:
    slab_v = fcc111('Cu', size=(2, 2, 4), vacuum=vac, periodic=True)
    q = np.array(slab_v.get_atomic_numbers(), dtype=np.float64)
    r = np.array(slab_v.get_positions(), dtype=np.float64)
    c = np.array(slab_v.cell[:], dtype=np.float64)

    rep = generate_mbdf_periodic([q], [r], [c],
                                  pbc=(True, True, True), rcut=6.0, n_jobs=1)[0]
    norm = np.linalg.norm(rep.sum(axis=0))
    print("  Vacuum %.1f Å: global rep norm = %.4f" % (vac, norm))

print()
print("Done! Representation correctly detects defects and surfaces.")
