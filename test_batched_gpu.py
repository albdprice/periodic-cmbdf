"""
Test and benchmark batched molecular GPU processing.

Compares:
1. Numba CPU (per-molecule, sequential)
2. Torch GPU per-structure (current approach)
3. Torch GPU batched (all structures in one kernel call)
"""
import numpy as np
import torch
import time
import sys
import gc
import os

sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
if device == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))

# Load QM9 data for molecules
qm9 = np.load('/home/albd/projects/cmbdf/data/qm9_parsed.npz', allow_pickle=True)
qm9_charges = qm9['charges']
qm9_coords = qm9['coords']
qm9_natoms = np.array([len(q) for q in qm9_charges])

# Also load MP for solids
mp = np.load('/home/albd/projects/cmbdf/data/mp_eform_parsed.npz', allow_pickle=True)
mp_charges = mp['charges']
mp_coords = mp['coords']
mp_cells = mp['cells']
mp_natoms = mp['n_atoms']

import cMBDF
from cMBDF_periodic import generate_mbdf_periodic as numba_periodic
import cMBDF_periodic_torch as torch_periodic

np_convs = cMBDF.get_convolutions(rcut=10.0, gradients=False)

# ============================================================
# Test: Batched correctness
# ============================================================
print("\n" + "=" * 70)
print("TEST: Batched vs per-structure correctness (solids)")
print("=" * 70)

np.random.seed(42)
mask = mp_natoms <= 20
idx = np.random.choice(np.where(mask)[0], 10, replace=False)
q_list = list(mp_charges[idx])
r_list = list(mp_coords[idx])
c_list = list(mp_cells[idx])

# Per-structure
reps_per = torch_periodic.generate_mbdf_periodic(
    q_list, r_list, c_list, pbc=(True,True,True), rcut=6.0, device=device)

# Batched
reps_batched = torch_periodic.generate_mbdf_periodic_batched(
    q_list, r_list, c_list, pbc=(True,True,True), rcut=6.0, device=device)

# Compare — batched returns list of variable-size tensors
print("Per-structure shape:", reps_per.shape)
print("Batched: %d structures" % len(reps_batched))

max_diff = 0
for i in range(10):
    n_at = len(q_list[i])
    per = reps_per[i, :n_at].cpu().numpy()
    bat = reps_batched[i].cpu().numpy()
    diff = np.max(np.abs(per - bat))
    max_diff = max(max_diff, diff)

print("Max diff between per-struct and batched: %.2e" % max_diff)
print("PASS" if max_diff < 1e-10 else "FAIL")

# ============================================================
# Benchmark: Batched vs per-structure on solids
# ============================================================
print("\n" + "=" * 70)
print("BENCHMARK: Batched GPU vs per-structure GPU vs Numba (solids)")
print("=" * 70)

for max_atoms, n_struct in [(15, 500), (25, 300), (40, 200)]:
    mask = mp_natoms <= max_atoms
    valid = np.where(mask)[0]
    np.random.seed(42)
    idx = np.random.choice(valid, n_struct, replace=False)
    q_list = list(mp_charges[idx])
    r_list = list(mp_coords[idx])
    c_list = list(mp_cells[idx])
    avg_at = np.mean([len(q) for q in q_list])

    # Numba
    gc.collect()
    t0 = time.perf_counter()
    _ = numba_periodic(q_list, r_list, c_list, pbc=(True,True,True),
                       rcut=6.0, n_atm=2.0, n_jobs=1)
    t_numba = time.perf_counter() - t0

    # Torch GPU per-structure
    gc.collect()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch_periodic.generate_mbdf_periodic(
            q_list, r_list, c_list, pbc=(True,True,True), rcut=6.0, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_per = time.perf_counter() - t0

    # Torch GPU batched
    gc.collect()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch_periodic.generate_mbdf_periodic_batched(
            q_list, r_list, c_list, pbc=(True,True,True), rcut=6.0, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0

    print("≤%d atoms (avg %.0f), N=%d:" % (max_atoms, avg_at, n_struct))
    print("  Numba:         %.2fs (%4.0f struct/s)" % (t_numba, n_struct / t_numba))
    print("  Torch per-str: %.2fs (%4.0f struct/s)" % (t_per, n_struct / t_per))
    print("  Torch batched: %.2fs (%4.0f struct/s)" % (t_batched, n_struct / t_batched))
    print("  Batched speedup vs Numba: %.2fx" % (t_numba / t_batched))
    print("  Batched speedup vs per-struct: %.2fx" % (t_per / t_batched))

# ============================================================
# Benchmark: Molecules (non-periodic, batched)
# ============================================================
print("\n" + "=" * 70)
print("BENCHMARK: Molecules (non-periodic, QM9)")
print("=" * 70)

for n_mol in [200, 500, 1000]:
    np.random.seed(42)
    idx = np.random.choice(len(qm9_charges), n_mol, replace=False)
    q_list = list(qm9_charges[idx])
    r_list = [r + 25.0 for r in qm9_coords[idx]]  # center in box
    c_list = [np.diag([50.0, 50.0, 50.0])] * n_mol

    # Numba molecular
    gc.collect()
    q_arr = np.array(list(qm9_charges[idx]), dtype=object)
    r_arr = np.array(list(qm9_coords[idx]), dtype=object)
    t0 = time.perf_counter()
    _ = cMBDF.generate_mbdf(q_arr, r_arr, convs=np_convs, rcut=10.0,
                            n_atm=2.0, n_jobs=1, smooth_cutoff=True)
    t_numba = time.perf_counter() - t0

    # Torch GPU batched
    gc.collect()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch_periodic.generate_mbdf_periodic_batched(
            q_list, r_list, c_list, pbc=(False,False,False), rcut=10.0, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_batched = time.perf_counter() - t0

    print("N=%d molecules:" % n_mol)
    print("  Numba:         %.2fs (%4.0f mol/s)" % (t_numba, n_mol / t_numba))
    print("  Torch batched: %.2fs (%4.0f mol/s)" % (t_batched, n_mol / t_batched))
    print("  Speedup: %.2fx" % (t_numba / t_batched))

print("\nDone!")
