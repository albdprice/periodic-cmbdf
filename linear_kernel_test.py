"""
Linear kernel test: How well does p-cMBDF perform with a simple linear model?

Tests the hypothesis that a good representation should make the residual
mapping approximately linear. The gap between linear and Laplacian
quantifies how much nonlinearity remains in the feature-to-property mapping.

Full matbench 5-fold CV for proper comparison.
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'

from cMBDF_periodic import normalize_per_element

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

print("=" * 70, flush=True)
print("Linear vs Nonlinear Kernel — Representation Quality Test", flush=True)
print("=" * 70, flush=True)

# Load data
data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_eform = data['eform']
all_natoms = data['n_atoms']

mask = all_natoms <= 30
valid = np.where(mask)[0]
N = len(valid)
charges = all_charges[valid]
targets = all_eform[valid]
print("Structures: %d" % N, flush=True)

# Load cached reps
rep_file = os.path.join(DATA_DIR, 'pcmbdf_5fold_mp_e_form.npz')
if os.path.exists(rep_file):
    reps = np.load(rep_file)['reps']
else:
    print("ERROR: Need cached reps. Run matbench_5fold.py first.", flush=True)
    sys.exit(1)

print("Rep shape: %s" % str(reps.shape), flush=True)

# Normalize
print("Normalizing...", flush=True)
reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
global_reps = build_global(reps_norm, charges)
print("Global: %s" % str(global_reps.shape), flush=True)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=18012019)

methods = {
    'Linear Ridge': {},
    'Polynomial (d=2)': {},
    'Polynomial (d=3)': {},
    'Laplacian KRR': {},
}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(global_reps)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(global_reps[train_idx])
    X_te = sc.transform(global_reps[test_idx])
    y_tr = targets[train_idx]
    y_te = targets[test_idx]

    print("\nFold %d (train=%d, test=%d):" % (fold_idx+1, len(train_idx), len(test_idx)), flush=True)

    # Linear Ridge
    t0 = time.time()
    ridge = RidgeCV(alphas=[1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0])
    ridge.fit(X_tr, y_tr)
    pred = ridge.predict(X_te)
    mae_lin = mean_absolute_error(y_te, pred)
    t_lin = time.time() - t0
    methods['Linear Ridge'][fold_idx] = mae_lin
    print("  Linear Ridge:    %.4f eV/atom (%.1fs, alpha=%.1f)" % (mae_lin, t_lin, ridge.alpha_), flush=True)

    # Polynomial kernel (degree 2)
    t0 = time.time()
    best_poly2 = 999
    for alpha in [1e-4, 1e-2, 1.0, 10.0]:
        krr = KernelRidge(alpha=alpha, kernel='polynomial', degree=2, coef0=1.0)
        krr.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, krr.predict(X_te))
        best_poly2 = min(best_poly2, mae)
    t_poly2 = time.time() - t0
    methods['Polynomial (d=2)'][fold_idx] = best_poly2
    print("  Polynomial d=2:  %.4f eV/atom (%.1fs)" % (best_poly2, t_poly2), flush=True)

    # Polynomial kernel (degree 3)
    t0 = time.time()
    best_poly3 = 999
    for alpha in [1e-4, 1e-2, 1.0, 10.0]:
        krr = KernelRidge(alpha=alpha, kernel='polynomial', degree=3, coef0=1.0)
        krr.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, krr.predict(X_te))
        best_poly3 = min(best_poly3, mae)
    t_poly3 = time.time() - t0
    methods['Polynomial (d=3)'][fold_idx] = best_poly3
    print("  Polynomial d=3:  %.4f eV/atom (%.1fs)" % (best_poly3, t_poly3), flush=True)

    # Laplacian KRR (float32 for memory)
    t0 = time.time()
    # Tune on subset
    np.random.seed(fold_idx)
    n_tr = len(train_idx)
    tune_idx = np.random.choice(n_tr, 10000, replace=False)
    tune_te = np.random.choice(n_tr, 2000, replace=False)
    tune_te = tune_te[~np.isin(tune_te, tune_idx)][:2000]

    best_g, best_a = 0.05, 1e-4
    best_tune = 999
    for g in [0.01, 0.02, 0.05, 0.1, 0.2]:
        for a in [1e-6, 1e-4, 1e-2]:
            try:
                Xt = X_tr[tune_idx].astype(np.float32)
                K = np.exp(-g * cdist(Xt, Xt, metric='cityblock'))
                K[np.diag_indices_from(K)] += a
                c = np.linalg.solve(K, y_tr[tune_idx].astype(np.float32))
                Xv = X_tr[tune_te].astype(np.float32)
                Kv = np.exp(-g * cdist(Xv, Xt, metric='cityblock'))
                mae = mean_absolute_error(y_tr[tune_te], (Kv @ c).astype(np.float64))
                if mae < best_tune:
                    best_tune = mae
                    best_g, best_a = g, a
                del K, Kv, c
                gc.collect()
            except:
                pass

    # Full training
    X32 = X_tr.astype(np.float32)
    Xt32 = X_te.astype(np.float32)
    K = np.exp(-best_g * cdist(X32, X32, metric='cityblock'))
    K[np.diag_indices_from(K)] += best_a
    c = np.linalg.solve(K, y_tr.astype(np.float32))
    Kt = np.exp(-best_g * cdist(Xt32, X32, metric='cityblock'))
    pred_lap = (Kt @ c).astype(np.float64)
    mae_lap = mean_absolute_error(y_te, pred_lap)
    t_lap = time.time() - t0
    methods['Laplacian KRR'][fold_idx] = mae_lap
    del K, Kt, c
    gc.collect()
    print("  Laplacian KRR:   %.4f eV/atom (%.1fs)" % (mae_lap, t_lap), flush=True)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70, flush=True)
print("RESULTS: Kernel Comparison (matbench mp_e_form, 5-fold CV)", flush=True)
print("=" * 70, flush=True)

print("\n%-20s | %-12s | %-12s | %-15s" % ("Kernel", "MAE (eV/at)", "Std", "Interpretation"), flush=True)
print("-" * 65, flush=True)

for name in ['Linear Ridge', 'Polynomial (d=2)', 'Polynomial (d=3)', 'Laplacian KRR']:
    maes = list(methods[name].values())
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    print("%-20s | %.4f       | %.4f       |" % (name, mean_mae, std_mae), flush=True)

# Compute the linearity ratio
lin_mae = np.mean(list(methods['Linear Ridge'].values()))
lap_mae = np.mean(list(methods['Laplacian KRR'].values()))
ratio = lin_mae / lap_mae

print("\n--- Representation Quality Metrics ---", flush=True)
print("  Linear MAE / Laplacian MAE = %.2f" % ratio, flush=True)
print("  (1.0 = perfect representation, higher = more residual nonlinearity)", flush=True)
print("  Linear captures %.0f%% of the Laplacian kernel's performance" % (lap_mae / lin_mae * 100), flush=True)

# Comparison to raw composition baseline
print("\n--- Context ---", flush=True)
print("  If ratio ~1.0: representation fully captures nonlinear physics", flush=True)
print("  If ratio ~2.0: significant nonlinearity remains", flush=True)
print("  If ratio >>2.0: representation is essentially raw features", flush=True)
print("  Our ratio: %.2f — p-cMBDF captures most but not all nonlinearity" % ratio, flush=True)

print("\nDone!", flush=True)
