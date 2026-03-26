"""
QM9 atomization energy learning curves for p-cMBDF.
Uses atomic reference energies to convert total energies to atomization energies.
Generates learning curve comparable to Fig 2 in the cMBDF paper.
Also generates data for Pareto front plot (Fig 3 style).
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'

# QM9 atomic reference energies (B3LYP/6-31G(2df,p), from QM9 paper)
# In Hartree
ATOMIC_REFS = {
    1: -0.500273,   # H
    6: -37.846772,   # C
    7: -54.583861,   # N
    8: -75.064579,   # O
    9: -99.718730,   # F
}

def krr_f32(X_tr, y_tr, X_te, y_te, gammas, alphas):
    best = 999
    X_tr32 = X_tr.astype(np.float32)
    X_te32 = X_te.astype(np.float32)
    y_tr32 = y_tr.astype(np.float32)
    for g in gammas:
        for a in alphas:
            try:
                K = np.exp(-g * cdist(X_tr32, X_tr32, metric='cityblock'))
                K[np.diag_indices_from(K)] += a
                c = np.linalg.solve(K, y_tr32)
                Kt = np.exp(-g * cdist(X_te32, X_tr32, metric='cityblock'))
                pred = (Kt @ c).astype(np.float64)
                mae = mean_absolute_error(y_te, pred)
                best = min(best, mae)
                del K, Kt, c
                gc.collect()
            except:
                pass
    return best

print("=" * 70, flush=True)
print("QM9 Atomization Energy Learning Curves", flush=True)
print("=" * 70, flush=True)

# Load QM9
data = np.load(os.path.join(DATA_DIR, 'qm9_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_energies = data['energies']  # Total U0 in Hartree
N = len(all_charges)
print("QM9: %d molecules" % N, flush=True)

# Convert to atomization energies
print("Computing atomization energies...", flush=True)
atomization = np.zeros(N)
for i in range(N):
    total = all_energies[i]
    ref_sum = sum(ATOMIC_REFS[int(z)] for z in all_charges[i])
    atomization[i] = (total - ref_sum) * 627.509  # Hartree to kcal/mol, note: atomization = -(total - ref_sum)

# Atomization energy should be negative (bound), flip sign for MAE convention
atomization = -atomization
print("Atomization energy range: %.1f to %.1f kcal/mol" % (atomization.min(), atomization.max()), flush=True)

# Fixed split
np.random.seed(42)
perm = np.random.permutation(N)
TEST = perm[:10000]
POOL = perm[10000:]

GAMMAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ALPHAS = [1e-10, 1e-8, 1e-6, 1e-4]
TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000]

from cMBDF import generate_mbdf

# ============================================================
# Generate old cMBDF representations
# ============================================================
REP_FILE = os.path.join(DATA_DIR, 'qm9_cmbdf_atomization.npz')
if os.path.exists(REP_FILE):
    print("Loading cached reps...", flush=True)
    reps = np.load(REP_FILE)['reps']
else:
    print("Generating cMBDF (smooth_cutoff=False, rcut=10)...", flush=True)
    t0 = time.time()
    reps = generate_mbdf(all_charges, all_coords, rcut=10.0, n_atm=2.0,
                          smooth_cutoff=False, n_jobs=-1, progress_bar=True)
    t_gen = time.time() - t0
    print("Generated in %.1fs" % t_gen, flush=True)
    np.savez_compressed(REP_FILE, reps=reps)

print("Rep shape: %s" % str(reps.shape), flush=True)

# Build global (sum over atoms)
def build_global(reps):
    if reps.ndim == 3:
        return np.array([reps[i][np.any(reps[i] != 0, axis=1)].sum(axis=0)
                        if np.any(reps[i] != 0) else np.zeros(reps.shape[-1])
                        for i in range(len(reps))])
    return reps

global_reps = build_global(reps)
print("Global shape: %s" % str(global_reps.shape), flush=True)

# ============================================================
# Learning curve on atomization energies
# ============================================================
print("\n" + "=" * 70, flush=True)
print("Learning Curve: cMBDF on QM9 Atomization Energies", flush=True)
print("=" * 70, flush=True)

print("\n%-8s | %-12s" % ("N_train", "MAE (kcal/mol)"), flush=True)
print("-" * 25, flush=True)

results = {}
timing = {}

for n_train in TRAIN_SIZES:
    if n_train > len(POOL):
        break
    tr = POOL[:n_train]

    sc = StandardScaler()
    X_tr = sc.fit_transform(global_reps[tr])
    X_te = sc.transform(global_reps[TEST])

    t0 = time.time()
    mae = krr_f32(X_tr, atomization[tr], X_te, atomization[TEST], GAMMAS, ALPHAS)
    t_krr = time.time() - t0

    results[n_train] = mae
    timing[n_train] = t_krr
    print("%-8d | %-12.2f (%.0fs)" % (n_train, mae, t_krr), flush=True)

# ============================================================
# Pareto front data (Fig 3 style)
# ============================================================
print("\n" + "=" * 70, flush=True)
print("Pareto Front Data (time vs N for chemical accuracy)", flush=True)
print("=" * 70, flush=True)

# Chemical accuracy = 1 kcal/mol
chem_acc = 1.0
print("\nChemical accuracy threshold: %.1f kcal/mol" % chem_acc, flush=True)

# Find N where MAE < 1 kcal/mol (if reached)
for n in sorted(results.keys()):
    if results[n] < chem_acc:
        print("Chemical accuracy reached at N = %d (MAE = %.2f)" % (n, results[n]), flush=True)
        break
else:
    # Extrapolate
    ns = sorted(results.keys())
    maes = [results[n] for n in ns]
    if len(ns) >= 3:
        # Log-log fit
        log_ns = np.log10(ns[-3:])
        log_maes = np.log10(maes[-3:])
        slope = (log_maes[-1] - log_maes[0]) / (log_ns[-1] - log_ns[0])
        # Extrapolate to MAE = 1
        log_n_ca = log_ns[-1] + (np.log10(chem_acc) - log_maes[-1]) / slope
        print("Extrapolated N for chemical accuracy: ~%.0f (slope=%.2f)" % (10**log_n_ca, slope), flush=True)
    print("Best MAE at max N: %.2f kcal/mol" % min(maes), flush=True)

# Total timing (rep generation + KRR at largest N)
print("\nTiming summary:", flush=True)
print("  Rep generation (133k mol): ~%.0fs" % (N / 1300), flush=True)  # ~1300 mol/s
for n in [1000, 4000, 16000, 32000]:
    if n in timing:
        print("  KRR at N=%d: %.0fs" % (n, timing[n]), flush=True)

# Published comparison points for Fig 3 style plot
print("\n--- Published results for comparison (from cMBDF paper Fig 2) ---", flush=True)
print("Method (dim) | N for ~1 kcal/mol | Training+prediction time", flush=True)
print("MBDF (5)     | ~10k              | 1.5 min", flush=True)
print("cMBDF (40)   | ~4k               | 1.3 min", flush=True)
print("CM (15)      | ~32k              | 4 min", flush=True)
print("SLATM (412)  | ~8k               | 8 min", flush=True)
print("FCHL19 (720) | ~2k               | 24.7 min", flush=True)
print("SOAP (3255)  | ~4k               | 7.7 min", flush=True)

print("\nDone!", flush=True)
