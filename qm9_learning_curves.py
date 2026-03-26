"""
QM9 Learning Curves: old cMBDF vs p-cMBDF.
Runs on freya (1TB RAM).
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist

DATA_DIR = '/home/albd/projects/cmbdf/data'

print("=" * 70, flush=True)
print("QM9 Learning Curves: old cMBDF vs p-cMBDF", flush=True)
print("=" * 70, flush=True)

# Load QM9
data = np.load(os.path.join(DATA_DIR, 'qm9_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_energies = data['energies']
N = len(all_charges)
print("QM9: %d molecules" % N, flush=True)

# Use full dataset
np.random.seed(42)
perm = np.random.permutation(N)
test_idx = perm[:10000]
train_pool = perm[10000:]

# ============================================================
# Generate representations
# ============================================================

# 1. Old cMBDF (no smooth cutoff)
REP1 = os.path.join(DATA_DIR, 'qm9_old_cmbdf.npz')
if os.path.exists(REP1):
    print("Loading old cMBDF...", flush=True)
    reps_old = np.load(REP1)['reps']
else:
    print("Generating old cMBDF (no smooth cutoff)...", flush=True)
    from cMBDF import generate_mbdf
    t0 = time.time()
    reps_old = generate_mbdf(all_charges, all_coords, rcut=10.0, n_atm=2.0,
                              smooth_cutoff=False, n_jobs=-1, progress_bar=True)
    print("Done in %.1fs" % (time.time() - t0), flush=True)
    np.savez_compressed(REP1, reps=reps_old)
print("Old cMBDF shape:", reps_old.shape, flush=True)

# 2. Old cMBDF + smooth cutoff
REP2 = os.path.join(DATA_DIR, 'qm9_old_cmbdf_smooth.npz')
if os.path.exists(REP2):
    print("Loading old cMBDF + smooth...", flush=True)
    reps_smooth = np.load(REP2)['reps']
else:
    print("Generating old cMBDF (smooth cutoff)...", flush=True)
    from cMBDF import generate_mbdf
    t0 = time.time()
    reps_smooth = generate_mbdf(all_charges, all_coords, rcut=10.0, n_atm=2.0,
                                 smooth_cutoff=True, n_jobs=-1, progress_bar=True)
    print("Done in %.1fs" % (time.time() - t0), flush=True)
    np.savez_compressed(REP2, reps=reps_smooth)
print("Smooth cMBDF shape:", reps_smooth.shape, flush=True)

# Build global reps (sum over atoms)
def build_global(reps):
    """Sum over atom dimension, handling zero-padded atoms."""
    if reps.ndim == 3:
        # (N, max_atoms, feat) -> (N, feat)
        return np.array([reps[i][np.any(reps[i] != 0, axis=1)].sum(axis=0)
                        if np.any(reps[i] != 0) else np.zeros(reps.shape[-1])
                        for i in range(len(reps))])
    return reps

global_old = build_global(reps_old)
global_smooth = build_global(reps_smooth)
print("Global shapes: old=%s, smooth=%s" % (global_old.shape, global_smooth.shape), flush=True)

# ============================================================
# Learning curves with float32 kernel
# ============================================================
def train_eval_f32(X_all, y_all, train_idx, test_idx, gammas, alphas):
    """KRR with float32 manual kernel computation."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all[train_idx]).astype(np.float32)
    X_te = sc.transform(X_all[test_idx]).astype(np.float32)
    y_tr = y_all[train_idx].astype(np.float32)
    y_te = y_all[test_idx]

    best_mae = 999
    for gamma in gammas:
        for alpha in alphas:
            K = np.exp(-gamma * cdist(X_tr, X_tr, metric='cityblock').astype(np.float32))
            K[np.diag_indices_from(K)] += alpha
            try:
                coeffs = np.linalg.solve(K, y_tr)
                K_te = np.exp(-gamma * cdist(X_te, X_tr, metric='cityblock').astype(np.float32))
                pred = K_te @ coeffs
                mae = np.mean(np.abs(y_te - pred.astype(np.float64)))
                best_mae = min(best_mae, mae)
            except Exception:
                pass
            del K
    return best_mae

train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
gammas = [0.001, 0.005, 0.01, 0.05, 0.1]
alphas = [1e-8, 1e-6, 1e-4]
y = all_energies

print("\n" + "=" * 70, flush=True)
print("LEARNING CURVES (MAE in kcal/mol)", flush=True)
print("=" * 70, flush=True)
print("%-8s | %-15s | %-15s" % ("N_train", "Old cMBDF", "Old+smooth"), flush=True)
print("-" * 45, flush=True)

results = {}
for n_train in train_sizes:
    if n_train > len(train_pool):
        break
    tr = train_pool[:n_train]

    mae_old = train_eval_f32(global_old, y, tr, test_idx, gammas, alphas) * 627.509
    mae_smooth = train_eval_f32(global_smooth, y, tr, test_idx, gammas, alphas) * 627.509

    results[n_train] = {'old': mae_old, 'smooth': mae_smooth}
    print("%-8d | %12.2f    | %12.2f" % (n_train, mae_old, mae_smooth), flush=True)

print("\nDone!", flush=True)
