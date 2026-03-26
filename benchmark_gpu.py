"""
GPU Benchmark: cMBDF representation generation.

Compares:
  1. Numba CPU (original cMBDF.py)
  2. PyTorch CPU (cMBDF_torch.py)
  3. PyTorch GPU (cMBDF_torch.py on CUDA)

Across varying dataset sizes and molecule sizes.
"""
import numpy as np
import torch
import time
import sys
import gc
import os

sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')

# ============================================================
# Setup
# ============================================================
print("=" * 70)
print("cMBDF GPU Benchmark")
print("=" * 70)

device_name = "CPU only"
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("GPU: %s (%.1f GB)" % (device_name, torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print("WARNING: No CUDA GPU available, skipping GPU benchmarks")
print()

# Load QM9 data
data = np.load('/home/albd/projects/cmbdf/data/qm9_parsed.npz', allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']

# Precompute convolutions once (shared across all runs)
import cMBDF
import cMBDF_torch

CONV_PARAMS = dict(rstep=0.0008, rcut=10.0, alpha_list=[1.5, 5.0],
                   n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                   astep=0.0002, nAs=4)
# Full params for generate_mbdf (includes rcut and n_atm)
GEN_PARAMS = dict(rcut=10.0, n_atm=2.0, rstep=0.0008,
                  alpha_list=[1.5, 5.0], n_list=[3.0, 5.0],
                  order=4, a1=2.0, a2=2.0, astep=0.0002, nAs=4)

print("Precomputing convolutions...")
np_convs = cMBDF.get_convolutions(**CONV_PARAMS, gradients=False)
t_rconvs_cpu, t_aconvs_cpu, meta = cMBDF_torch.get_convolutions(**CONV_PARAMS, device='cpu')
if torch.cuda.is_available():
    t_rconvs_gpu, t_aconvs_gpu, meta_gpu = cMBDF_torch.get_convolutions(**CONV_PARAMS, device='cuda')
print("Done.\n")


def bench_numba_cpu(charges, coords, n_runs=3):
    """Benchmark original Numba CPU implementation."""
    # Warmup (JIT compilation)
    _ = cMBDF.generate_mbdf(charges[:2], coords[:2], convs=np_convs,
                            rcut=10.0, n_atm=2.0, n_jobs=1)

    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        _ = cMBDF.generate_mbdf(charges, coords, convs=np_convs,
                                rcut=10.0, n_atm=2.0, n_jobs=1)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def bench_numba_cpu_parallel(charges, coords, n_runs=3):
    """Benchmark Numba CPU with joblib parallelism."""
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        _ = cMBDF.generate_mbdf(charges, coords, convs=np_convs,
                                rcut=10.0, n_atm=2.0, n_jobs=-1)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def bench_torch_cpu(charges, coords, n_runs=3):
    """Benchmark PyTorch CPU implementation."""
    # Warmup
    _ = cMBDF_torch.generate_mbdf(charges[:2], coords[:2],
                                   device='cpu', **GEN_PARAMS)

    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = cMBDF_torch.generate_mbdf(charges, coords,
                                           device='cpu', **GEN_PARAMS)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def bench_torch_gpu(charges, coords, n_runs=3):
    """Benchmark PyTorch GPU implementation."""
    if not torch.cuda.is_available():
        return float('nan')

    # Warmup
    _ = cMBDF_torch.generate_mbdf(charges[:2], coords[:2],
                                   device='cuda', **GEN_PARAMS)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        gc.collect()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = cMBDF_torch.generate_mbdf(charges, coords,
                                           device='cuda', **GEN_PARAMS)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


# ============================================================
# Benchmark 1: Varying dataset size (fixed molecule composition)
# ============================================================
print("=" * 70)
print("BENCHMARK 1: Varying dataset size")
print("=" * 70)

# Sort molecules by size for consistent subsets
sizes = np.array([len(q) for q in all_charges])
# Use molecules with 5-15 heavy atoms (typical QM9 range)
mask = (sizes >= 5) & (sizes <= 20)
valid_idx = np.where(mask)[0]
np.random.seed(42)
np.random.shuffle(valid_idx)

dataset_sizes = [50, 100, 250, 500, 1000]
results_ds = []

print("\n%-8s | %-12s | %-12s | %-12s | %-12s | %-10s" % (
    "N_mol", "Numba 1-core", "Numba all", "Torch CPU", "Torch GPU", "GPU speedup"))
print("-" * 80)

for n in dataset_sizes:
    idx = valid_idx[:n]
    charges = all_charges[idx]
    coords = all_coords[idx]

    t_numba1 = bench_numba_cpu(charges, coords, n_runs=2)
    t_numba_p = bench_numba_cpu_parallel(charges, coords, n_runs=2)
    t_torch_cpu = bench_torch_cpu(charges, coords, n_runs=2)
    t_torch_gpu = bench_torch_gpu(charges, coords, n_runs=2)

    speedup_vs_numba1 = t_numba1 / t_torch_gpu if not np.isnan(t_torch_gpu) else float('nan')
    speedup_vs_parallel = t_numba_p / t_torch_gpu if not np.isnan(t_torch_gpu) else float('nan')

    results_ds.append({
        'n': n, 'numba1': t_numba1, 'numba_p': t_numba_p,
        'torch_cpu': t_torch_cpu, 'torch_gpu': t_torch_gpu,
        'speedup_1core': speedup_vs_numba1,
        'speedup_parallel': speedup_vs_parallel
    })

    print("%-8d | %10.3fs  | %10.3fs  | %10.3fs  | %10.3fs  | %.1fx/%.1fx" % (
        n, t_numba1, t_numba_p, t_torch_cpu, t_torch_gpu,
        speedup_vs_numba1, speedup_vs_parallel))

# ============================================================
# Benchmark 2: Varying molecule size (fixed dataset size = 100)
# ============================================================
print()
print("=" * 70)
print("BENCHMARK 2: Varying molecule size (N=100 molecules each)")
print("=" * 70)

size_bins = [(3, 6), (7, 10), (11, 15), (16, 20), (21, 27)]
results_ms = []

print("\n%-12s | %-12s | %-12s | %-12s | %-12s | %-10s" % (
    "Atoms/mol", "Numba 1-core", "Numba all", "Torch CPU", "Torch GPU", "GPU speedup"))
print("-" * 85)

for lo, hi in size_bins:
    mask_bin = (sizes >= lo) & (sizes <= hi)
    bin_idx = np.where(mask_bin)[0]
    if len(bin_idx) < 100:
        print("%-12s | skipped (only %d molecules)" % ("%d-%d" % (lo, hi), len(bin_idx)))
        continue

    np.random.seed(42)
    idx = np.random.choice(bin_idx, 100, replace=False)
    charges = all_charges[idx]
    coords = all_coords[idx]
    avg_size = np.mean([len(q) for q in charges])

    t_numba1 = bench_numba_cpu(charges, coords, n_runs=2)
    t_numba_p = bench_numba_cpu_parallel(charges, coords, n_runs=2)
    t_torch_cpu = bench_torch_cpu(charges, coords, n_runs=2)
    t_torch_gpu = bench_torch_gpu(charges, coords, n_runs=2)

    speedup = t_numba1 / t_torch_gpu if not np.isnan(t_torch_gpu) else float('nan')

    results_ms.append({
        'bin': "%d-%d" % (lo, hi), 'avg_atoms': avg_size,
        'numba1': t_numba1, 'numba_p': t_numba_p,
        'torch_cpu': t_torch_cpu, 'torch_gpu': t_torch_gpu,
        'speedup': speedup
    })

    print("%-12s | %10.3fs  | %10.3fs  | %10.3fs  | %10.3fs  | %.1fx" % (
        "%d-%d (avg %.0f)" % (lo, hi, avg_size),
        t_numba1, t_numba_p, t_torch_cpu, t_torch_gpu, speedup))

# ============================================================
# Benchmark 3: Gradient computation (PyTorch only)
# ============================================================
print()
print("=" * 70)
print("BENCHMARK 3: Gradient computation (PyTorch, N=50 molecules)")
print("=" * 70)

idx = valid_idx[:50]
charges = all_charges[idx]
coords = all_coords[idx]

# Forward only (no grad)
gc.collect()
t0 = time.perf_counter()
with torch.no_grad():
    reps = cMBDF_torch.generate_mbdf(charges, coords,
                                      device='cpu', **GEN_PARAMS)
t_fwd = time.perf_counter() - t0

# Forward + backward (with grad) — need per-molecule since coords have different shapes
gc.collect()
rconvs_cpu, aconvs_cpu, meta_cpu = cMBDF_torch.get_convolutions(**CONV_PARAMS, device='cpu')
t0 = time.perf_counter()
total_loss = torch.tensor(0.0, dtype=torch.float64)
for q, r in zip(charges, coords):
    coords_t = torch.tensor(np.asarray(r, dtype=np.float64), requires_grad=True)
    charges_t = torch.tensor(np.asarray(q, dtype=np.float64))
    rep = cMBDF_torch._compute_rep(charges_t, coords_t, rconvs_cpu, aconvs_cpu,
                                    meta_cpu, 10.0, 2.0, True)
    loss = rep.sum()
    loss.backward()
t_fwd_bwd_cpu = time.perf_counter() - t0

print("Forward only (CPU, 50 mol):       %.3fs" % t_fwd)
print("Forward + backward (CPU, 50 mol): %.3fs" % t_fwd_bwd_cpu)
print("Gradient overhead:                %.1fx" % (t_fwd_bwd_cpu / t_fwd))

if torch.cuda.is_available():
    gc.collect()
    rconvs_g, aconvs_g, meta_g = cMBDF_torch.get_convolutions(**CONV_PARAMS, device='cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for q, r in zip(charges, coords):
        coords_t = torch.tensor(np.asarray(r, dtype=np.float64), device='cuda', requires_grad=True)
        charges_t = torch.tensor(np.asarray(q, dtype=np.float64), device='cuda')
        rep = cMBDF_torch._compute_rep(charges_t, coords_t, rconvs_g, aconvs_g,
                                        meta_g, 10.0, 2.0, True)
        loss = rep.sum()
        loss.backward()
    torch.cuda.synchronize()
    t_fwd_bwd_gpu = time.perf_counter() - t0
    print("Forward + backward (GPU, 50 mol): %.3fs" % t_fwd_bwd_gpu)
    print("GPU speedup for gradients:        %.1fx" % (t_fwd_bwd_cpu / t_fwd_bwd_gpu))

# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nThroughput (molecules/second):")
print("%-20s | %-15s | %-15s" % ("Method", "Small mol (5-10)", "Large mol (16-20)"))
print("-" * 55)

for r in results_ms:
    if r['bin'] in ['7-10', '16-20']:
        numba_rate = 100 / r['numba1']
        gpu_rate = 100 / r['torch_gpu'] if not np.isnan(r['torch_gpu']) else 0
        print("%-20s | %-15s | %-15s" % (
            "Numba (1-core)",
            "%.0f mol/s" % (100 / r['numba1']),
            "%.0f mol/s" % (100 / r['numba1'])))

for r in results_ms:
    if r['bin'] == '7-10':
        small_gpu = 100 / r['torch_gpu'] if not np.isnan(r['torch_gpu']) else 0
    if r['bin'] == '16-20':
        large_gpu = 100 / r['torch_gpu'] if not np.isnan(r['torch_gpu']) else 0

if torch.cuda.is_available():
    for r in results_ms:
        if r['bin'] in ['7-10']:
            print("%-20s | %-15s" % ("Torch GPU", "%.0f mol/s" % (100 / r['torch_gpu'])), end="")
        if r['bin'] in ['16-20']:
            print(" | %-15s" % ("%.0f mol/s" % (100 / r['torch_gpu'])))

print("\nBenchmark complete!")
print("Device: %s" % device_name)
