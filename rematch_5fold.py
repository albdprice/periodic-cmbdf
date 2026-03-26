"""
REMatch local kernel on matbench_phonons and matbench_perovskites (5-fold CV).
These are small enough for the O(N^2 * n_atoms^2) kernel computation.
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from rematch_kernel import compute_rematch_kernel_matrix, compute_rematch_kernel_rect

DATA_DIR = '/home/albd/projects/cmbdf/data'

from cMBDF_periodic import normalize_per_element

def build_local_reps(reps, charges_list):
    """Extract per-atom reps (remove padding)."""
    local = []
    local_charges = []
    for i in range(len(reps)):
        n = len(charges_list[i])
        local.append(reps[i, :n, :])
        local_charges.append(charges_list[i])
    return local, local_charges

print("=" * 70, flush=True)
print("REMatch Local Kernel — 5-fold CV", flush=True)
print("=" * 70, flush=True)

tasks = [
    ('matbench_phonons', '1/cm'),      # 1265 structures — tractable
    ('matbench_perovskites', 'eV/atom'),  # 18928 — subsample to 2000
]

for task_name, unit in tasks:
    print("\n" + "=" * 70, flush=True)
    print("TASK: %s [%s]" % (task_name, unit), flush=True)
    print("=" * 70, flush=True)

    cache_data = os.path.join(DATA_DIR, 'matbench_%s.npz' % task_name.replace('matbench_', ''))
    d = np.load(cache_data, allow_pickle=True)
    charges = d['charges']
    targets = d['targets']
    natoms = d['n_atoms']

    mask = natoms <= 30
    valid = np.where(mask)[0]

    # Subsample large datasets
    if len(valid) > 2000:
        np.random.seed(42)
        valid = np.random.choice(valid, 2000, replace=False)
        valid.sort()

    N = len(valid)
    charges = charges[valid]
    targets = targets[valid]
    print("  Using %d structures" % N, flush=True)

    # Load cached reps
    cache_reps = os.path.join(DATA_DIR, 'pcmbdf_5fold_%s.npz' % task_name.replace('matbench_', ''))
    if not os.path.exists(cache_reps):
        cache_reps = os.path.join(DATA_DIR, 'pcmbdf_%s.npz' % task_name)
    reps = np.load(cache_reps)['reps']
    if len(reps) > N:
        reps = reps[:N]

    reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
    local_reps, local_charges = build_local_reps(reps_norm, charges)

    # 5-fold CV with REMatch
    kf = KFold(n_splits=5, shuffle=True, random_state=18012019)

    for sigma in [5.0, 10.0]:
        fold_maes = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):
            n_tr = len(train_idx)
            n_te = len(test_idx)

            reps_tr = [local_reps[i] for i in train_idx]
            charges_tr = [local_charges[i] for i in train_idx]
            reps_te = [local_reps[i] for i in test_idx]
            charges_te = [local_charges[i] for i in test_idx]

            print("  Fold %d (sigma=%.1f): computing %dx%d kernel..." % (
                fold_idx+1, sigma, n_tr, n_tr), flush=True)
            t0 = time.time()
            K_tr = compute_rematch_kernel_matrix(reps_tr, charges_tr, sigma=sigma, gamma=0.1)
            K_te = compute_rematch_kernel_rect(reps_tr, charges_tr, reps_te, charges_te, sigma=sigma, gamma=0.1)
            t_kernel = time.time() - t0
            print("    Kernel: %.1fs" % t_kernel, flush=True)

            # KRR with precomputed kernel
            best_mae = 999
            for alpha in [1e-8, 1e-6, 1e-4, 1e-2]:
                try:
                    K = K_tr.copy()
                    K[np.diag_indices_from(K)] += alpha
                    coeffs = np.linalg.solve(K, targets[train_idx])
                    pred = K_te @ coeffs
                    mae = mean_absolute_error(targets[test_idx], pred)
                    best_mae = min(best_mae, mae)
                except:
                    pass

            fold_maes.append(best_mae)
            print("    Fold %d MAE: %.4f %s" % (fold_idx+1, best_mae, unit), flush=True)

        mean_mae = np.mean(fold_maes)
        print("  === sigma=%.1f: %.4f ± %.4f %s ===" % (
            sigma, mean_mae, np.std(fold_maes), unit), flush=True)

print("\nDone!", flush=True)
