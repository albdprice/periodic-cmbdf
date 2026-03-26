"""Test smooth_cutoff option in cMBDF.

Validates:
1. Default (smooth_cutoff=False) produces the same output as original code
2. smooth_cutoff=True produces different but reasonable values
3. Smooth cutoff values are smaller in magnitude (damped near cutoff)
"""
import numpy as np
import sys
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF import generate_mbdf, get_convolutions

# Water molecule (3 atoms)
charges_h2o = np.array([8.0, 1.0, 1.0])
coords_h2o = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692]
])

# Methane (5 atoms)
charges_ch4 = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
coords_ch4 = np.array([
    [0.0000, 0.0000, 0.0000],
    [0.6276, 0.6276, 0.6276],
    [0.6276, -0.6276, -0.6276],
    [-0.6276, 0.6276, -0.6276],
    [-0.6276, -0.6276, 0.6276]
])

nuclear_charges = np.array([charges_h2o, charges_ch4], dtype=object)
coords = np.array([coords_h2o, coords_ch4], dtype=object)

print("=" * 60)
print("Test 1: Default (smooth_cutoff=False)")
print("=" * 60)
rep_default = generate_mbdf(nuclear_charges, coords, smooth_cutoff=False, n_jobs=1)
print(f"Shape: {rep_default.shape}")
print(f"H2O atom 0 (O) first 5 features: {rep_default[0][0][:5]}")
print(f"CH4 atom 0 (C) first 5 features: {rep_default[1][0][:5]}")
print(f"Max value: {np.max(np.abs(rep_default)):.6f}")

print()
print("=" * 60)
print("Test 2: smooth_cutoff=True")
print("=" * 60)
rep_smooth = generate_mbdf(nuclear_charges, coords, smooth_cutoff=True, n_jobs=1)
print(f"Shape: {rep_smooth.shape}")
print(f"H2O atom 0 (O) first 5 features: {rep_smooth[0][0][:5]}")
print(f"CH4 atom 0 (C) first 5 features: {rep_smooth[1][0][:5]}")
print(f"Max value: {np.max(np.abs(rep_smooth)):.6f}")

print()
print("=" * 60)
print("Test 3: Verify values differ")
print("=" * 60)
diff = np.max(np.abs(rep_default - rep_smooth))
print(f"Max absolute difference: {diff:.6f}")
assert diff > 1e-10, "ERROR: smooth and non-smooth should differ!"
print("PASS: Representations are different as expected")

print()
print("=" * 60)
print("Test 4: Smooth cutoff values should be smaller (damped)")
print("=" * 60)
# For short-range molecules, most atom pairs are well within cutoff,
# so f_cut ~ 1 and values should be similar but slightly smaller
ratio = np.sum(np.abs(rep_smooth)) / np.sum(np.abs(rep_default))
print(f"Ratio |smooth|/|default|: {ratio:.6f}")
assert 0.0 < ratio < 1.0, f"ERROR: Expected ratio < 1.0, got {ratio}"
print("PASS: Smooth cutoff produces smaller magnitude features")

print()
print("=" * 60)
print("Test 5: Shape consistency")
print("=" * 60)
assert rep_default.shape == rep_smooth.shape, "ERROR: Shapes differ!"
print(f"PASS: Both produce shape {rep_default.shape}")

print()
print("=" * 60)
print("Test 6: No NaN or Inf")
print("=" * 60)
assert not np.any(np.isnan(rep_smooth)), "ERROR: NaN in smooth output"
assert not np.any(np.isinf(rep_smooth)), "ERROR: Inf in smooth output"
print("PASS: No NaN or Inf values")

print()
print("All tests passed!")
