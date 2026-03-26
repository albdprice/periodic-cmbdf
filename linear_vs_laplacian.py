"""
Linear vs Laplacian kernel — quick comparison on odin while
freya runs the full polynomial test.
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
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
print("Linear vs Laplacian Kernel (matbench 5-fold CV)", flush=True)
print("=" * 70, flush=True)

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_eform = data['eform']
all_natoms = data['n_atoms']

mask = all_natoms <= 30
valid = np.where(mask)[0]
charges = all_charges[valid]
targets = all_eform[valid]
N = len(valid)
print("Structures: %d" % N, flush=True)

reps = np.load(os.path.join(DATA_DIR, 'pcmbdf_5fold_mp_e_form.npz'))['reps']
reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
global_reps = build_global(reps_norm, charges)

kf = KFold(n_splits=5, shuffle=True, random_state=18012019)

linear_maes = []
laplacian_maes = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(global_reps)):
    sc = StandardScaler()
    X_tr = sc.fit_transform(global_reps[train_idx])
    X_te = sc.transform(global_reps[test_idx])
    y_tr = targets[train_idx]
    y_te = targets[test_idx]
    n_tr = len(train_idx)

    print("\nFold %d (train=%d, test=%d):" % (fold_idx+1, n_tr, len(test_idx)), flush=True)

    # Linear Ridge
    t0 = time.time()
    ridge = RidgeCV(alphas=[1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ridge.fit(X_tr, y_tr)
    mae_lin = mean_absolute_error(y_te, ridge.predict(X_te))
    linear_maes.append(mae_lin)
    print("  Linear Ridge:  %.4f eV/atom (%.1fs)" % (mae_lin, time.time()-t0), flush=True)

    # Laplacian KRR (float32)
    t0 = time.time()
    np.random.seed(fold_idx)
    tune_idx = np.random.choice(n_tr, 10000, replace=False)
    tune_te = np.setdiff1d(np.arange(n_tr), tune_idx)[:2000]

    best_g, best_a, best_tune = 0.05, 1e-4, 999
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
                    best_tune, best_g, best_a = mae, g, a
                del K, Kv, c; gc.collect()
            except: pass

    X32 = X_tr.astype(np.float32)
    Xt32 = X_te.astype(np.float32)
    K = np.exp(-best_g * cdist(X32, X32, metric='cityblock'))
    K[np.diag_indices_from(K)] += best_a
    c = np.linalg.solve(K, y_tr.astype(np.float32))
    Kt = np.exp(-best_g * cdist(Xt32, X32, metric='cityblock'))
    mae_lap = mean_absolute_error(y_te, (Kt @ c).astype(np.float64))
    laplacian_maes.append(mae_lap)
    del K, Kt, c; gc.collect()
    print("  Laplacian KRR: %.4f eV/atom (%.1fs)" % (mae_lap, time.time()-t0), flush=True)

lin_mean = np.mean(linear_maes)
lap_mean = np.mean(laplacian_maes)
print("\n" + "=" * 70, flush=True)
print("Linear Ridge:  %.4f +/- %.4f" % (lin_mean, np.std(linear_maes)), flush=True)
print("Laplacian KRR: %.4f +/- %.4f" % (lap_mean, np.std(laplacian_maes)), flush=True)
print("Ratio: %.2f" % (lin_mean / lap_mean), flush=True)
print("Done!", flush=True)
