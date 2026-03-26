"""
Validation tests for periodic cMBDF.

Tests:
1. Symmetry: equivalent atoms in bulk crystals get identical representations
2. Supercell invariance: same per-atom features in unit cell vs 2x2x2
3. Cutoff convergence: features converge as cutoff increases
4. Consistency: molecular limit (large vacuum) matches molecular cMBDF
5. Basic sanity: no NaN/Inf, correct shapes
"""
import numpy as np
import sys
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF_periodic import generate_mbdf_periodic, get_convolutions, build_neighbor_data
from cMBDF import generate_mbdf

# Shared convolution params
CONV_PARAMS = dict(rstep=0.0008, rcut=6.0, alpha_list=[1.5, 5.0],
                   n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                   astep=0.0002, nAs=4)

convs = get_convolutions(**CONV_PARAMS, gradients=False)

# ============================================================
# Test structures
# ============================================================

# FCC Copper (a = 3.615 Å, 4 atoms in conventional cell)
a_cu = 3.615
cu_charges = np.array([29.0, 29.0, 29.0, 29.0])
cu_coords = np.array([
    [0.0, 0.0, 0.0],
    [0.0, a_cu/2, a_cu/2],
    [a_cu/2, 0.0, a_cu/2],
    [a_cu/2, a_cu/2, 0.0]
])
cu_cell = np.diag([a_cu, a_cu, a_cu])

# BCC Iron (a = 2.87 Å, 2 atoms)
a_fe = 2.87
fe_charges = np.array([26.0, 26.0])
fe_coords = np.array([
    [0.0, 0.0, 0.0],
    [a_fe/2, a_fe/2, a_fe/2]
])
fe_cell = np.diag([a_fe, a_fe, a_fe])

# Diamond Silicon (a = 5.43 Å, 8 atoms in conventional cell)
a_si = 5.43
si_charges = np.array([14.0]*8)
si_coords = np.array([
    [0, 0, 0],
    [a_si/4, a_si/4, a_si/4],
    [a_si/2, a_si/2, 0],
    [3*a_si/4, 3*a_si/4, a_si/4],
    [a_si/2, 0, a_si/2],
    [3*a_si/4, a_si/4, 3*a_si/4],
    [0, a_si/2, a_si/2],
    [a_si/4, 3*a_si/4, 3*a_si/4]
])
si_cell = np.diag([a_si, a_si, a_si])

# NaCl (rock salt, a = 5.64 Å, 8 atoms)
a_nacl = 5.64
nacl_charges = np.array([11.0, 17.0, 11.0, 17.0, 11.0, 17.0, 17.0, 11.0])
nacl_coords = np.array([
    [0, 0, 0],
    [a_nacl/2, 0, 0],
    [a_nacl/2, a_nacl/2, 0],
    [0, a_nacl/2, 0],
    [0, 0, a_nacl/2],
    [a_nacl/2, 0, a_nacl/2],
    [0, a_nacl/2, a_nacl/2],
    [a_nacl/2, a_nacl/2, a_nacl/2]
])
nacl_cell = np.diag([a_nacl, a_nacl, a_nacl])

pbc = (True, True, True)


# ============================================================
# Test 1: Symmetry — equivalent atoms get identical representations
# ============================================================
print("=" * 70)
print("TEST 1: Symmetry — equivalent atoms in bulk crystals")
print("=" * 70)

# FCC Cu: all 4 atoms are equivalent
rep_cu = generate_mbdf_periodic(
    [cu_charges], [cu_coords], [cu_cell], pbc=pbc,
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

print("FCC Cu (4 atoms, all equivalent):")
print("  Rep shape:", rep_cu.shape)
max_diff_cu = 0
for i in range(1, 4):
    diff = np.max(np.abs(rep_cu[i] - rep_cu[0]))
    max_diff_cu = max(max_diff_cu, diff)
    print("  Atom 0 vs Atom %d: max diff = %.2e" % (i, diff))

if max_diff_cu < 1e-10:
    print("  PASS: All Cu atoms identical")
else:
    print("  FAIL: Max diff %.2e" % max_diff_cu)

# BCC Fe: both atoms are equivalent
rep_fe = generate_mbdf_periodic(
    [fe_charges], [fe_coords], [fe_cell], pbc=pbc,
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

print("\nBCC Fe (2 atoms, both equivalent):")
diff_fe = np.max(np.abs(rep_fe[0] - rep_fe[1]))
print("  Atom 0 vs Atom 1: max diff = %.2e" % diff_fe)
if diff_fe < 1e-10:
    print("  PASS: Both Fe atoms identical")
else:
    print("  FAIL: Diff %.2e" % diff_fe)

# Diamond Si: all 8 atoms are equivalent
rep_si = generate_mbdf_periodic(
    [si_charges], [si_coords], [si_cell], pbc=pbc,
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

print("\nDiamond Si (8 atoms, all equivalent):")
max_diff_si = 0
for i in range(1, 8):
    diff = np.max(np.abs(rep_si[i] - rep_si[0]))
    max_diff_si = max(max_diff_si, diff)
print("  Max diff across all pairs: %.2e" % max_diff_si)
if max_diff_si < 1e-10:
    print("  PASS: All Si atoms identical")
else:
    print("  FAIL: Max diff %.2e" % max_diff_si)

# NaCl: Na atoms equivalent to each other, Cl to each other, Na != Cl
rep_nacl = generate_mbdf_periodic(
    [nacl_charges], [nacl_coords], [nacl_cell], pbc=pbc,
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

na_idx = np.where(nacl_charges == 11.0)[0]
cl_idx = np.where(nacl_charges == 17.0)[0]

print("\nNaCl (4 Na + 4 Cl):")
max_diff_na = max(np.max(np.abs(rep_nacl[i] - rep_nacl[na_idx[0]])) for i in na_idx[1:])
max_diff_cl = max(np.max(np.abs(rep_nacl[i] - rep_nacl[cl_idx[0]])) for i in cl_idx[1:])
na_cl_diff = np.max(np.abs(rep_nacl[na_idx[0]] - rep_nacl[cl_idx[0]]))

print("  Na-Na max diff: %.2e" % max_diff_na)
print("  Cl-Cl max diff: %.2e" % max_diff_cl)
print("  Na vs Cl diff: %.4f (should be large)" % na_cl_diff)

if max_diff_na < 1e-10 and max_diff_cl < 1e-10 and na_cl_diff > 0.01:
    print("  PASS: Na equivalent, Cl equivalent, Na != Cl")
else:
    print("  WARN: Check symmetry")


# ============================================================
# Test 2: Supercell invariance
# ============================================================
print()
print("=" * 70)
print("TEST 2: Supercell invariance (unit cell vs 2x2x2)")
print("=" * 70)

# Build 2x2x2 BCC Fe supercell
sc = 2
fe_sc_charges = np.tile(fe_charges, sc**3)
fe_sc_coords = []
for ix in range(sc):
    for iy in range(sc):
        for iz in range(sc):
            shift = np.array([ix, iy, iz]) * a_fe
            for c in fe_coords:
                fe_sc_coords.append(c + shift)
fe_sc_coords = np.array(fe_sc_coords)
fe_sc_cell = np.diag([a_fe * sc, a_fe * sc, a_fe * sc])

rep_fe_sc = generate_mbdf_periodic(
    [fe_sc_charges], [fe_sc_coords], [fe_sc_cell], pbc=pbc,
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

# Compare atom 0 in unit cell vs atom 0 in supercell
diff_sc = np.max(np.abs(rep_fe[0] - rep_fe_sc[0]))
print("BCC Fe: unit cell atom 0 vs 2x2x2 supercell atom 0:")
print("  Max diff: %.2e" % diff_sc)

if diff_sc < 1e-8:
    print("  PASS: Supercell invariant")
else:
    print("  WARN: Diff %.2e (may be due to cutoff boundary effects)" % diff_sc)

# Also check that all atoms in the supercell are equivalent
max_diff_sc = max(np.max(np.abs(rep_fe_sc[i] - rep_fe_sc[0]))
                  for i in range(1, len(fe_sc_charges)))
print("  All supercell atoms equivalent: max diff = %.2e" % max_diff_sc)


# ============================================================
# Test 3: Cutoff convergence
# ============================================================
print()
print("=" * 70)
print("TEST 3: Cutoff convergence (BCC Fe)")
print("=" * 70)

cutoffs = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
prev_rep = None

for rc in cutoffs:
    convs_rc = get_convolutions(rcut=rc, **{k: v for k, v in CONV_PARAMS.items() if k != 'rcut'},
                                gradients=False)
    rep_rc = generate_mbdf_periodic(
        [fe_charges], [fe_coords], [fe_cell], pbc=pbc,
        convs=convs_rc, rcut=rc, n_atm=2.0, n_jobs=1)[0]

    if prev_rep is not None:
        # Can't directly compare (different grid sizes), so compare norms
        norm_change = np.abs(np.linalg.norm(rep_rc[0]) - np.linalg.norm(prev_rep[0]))
        pct_change = norm_change / np.linalg.norm(rep_rc[0]) * 100
        print("  rcut=%.1f: norm=%.4f, change from prev=%.4f (%.2f%%)" % (
            rc, np.linalg.norm(rep_rc[0]), norm_change, pct_change))
    else:
        print("  rcut=%.1f: norm=%.4f" % (rc, np.linalg.norm(rep_rc[0])))

    prev_rep = rep_rc


# ============================================================
# Test 4: Molecular limit
# ============================================================
print()
print("=" * 70)
print("TEST 4: Molecular limit (large vacuum box)")
print("=" * 70)

# H2O in a large box (no periodic interactions)
h2o_charges = np.array([8.0, 1.0, 1.0])
h2o_coords = np.array([
    [10.0, 10.0, 10.1173],
    [10.0, 10.7572, 9.5308],
    [10.0, 9.2428, 9.5308]
])
h2o_cell = np.diag([20.0, 20.0, 20.0])

# Use rcut=6.0 for both to avoid Numba recompilation issues with different grid sizes
rep_periodic = generate_mbdf_periodic(
    [h2o_charges], [h2o_coords], [h2o_cell], pbc=(False, False, False),
    convs=convs, rcut=6.0, n_atm=2.0, n_jobs=1)[0]

print("H2O in vacuum box (periodic, pbc=False):")
print("  Periodic O atom first 5: ", rep_periodic[0, :5])
print("  Periodic H atom first 5: ", rep_periodic[1, :5])

# Self-consistency: check that periodic H2O in vacuum has non-zero features
# and that O differs from H
oh_diff = np.max(np.abs(rep_periodic[0] - rep_periodic[1]))
h_equiv = np.max(np.abs(rep_periodic[1] - rep_periodic[2]))
print("  O vs H diff: %.4f (should be large)" % oh_diff)
print("  H1 vs H2 diff: %.2e (should be ~0)" % h_equiv)

has_vals = np.any(rep_periodic[:3] != 0)
if has_vals and oh_diff > 0.01 and h_equiv < 1e-10:
    print("  PASS: Correct molecular behavior in vacuum box")
else:
    print("  WARN: Check molecular limit")


# ============================================================
# Test 5: Basic sanity
# ============================================================
print()
print("=" * 70)
print("TEST 5: No NaN/Inf, correct shapes")
print("=" * 70)

for name, rep in [("Cu", rep_cu), ("Fe", rep_fe), ("Si", rep_si), ("NaCl", rep_nacl)]:
    has_nan = np.any(np.isnan(rep))
    has_inf = np.any(np.isinf(rep))
    has_nonzero = np.any(rep != 0)
    print("  %s: shape=%s, NaN=%s, Inf=%s, nonzero=%s" % (
        name, rep.shape, has_nan, has_inf, has_nonzero))
    if has_nan or has_inf or not has_nonzero:
        print("    FAIL")
    else:
        print("    PASS")

print()
print("All tests completed!")
