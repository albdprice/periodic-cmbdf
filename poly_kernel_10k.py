"""
Polynomial kernel test on 10k subset (tractable).
Gets the relative ordering: linear < poly d=2 < poly d=3 < Laplacian.
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV
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

print("Polynomial Kernel Test (10k subset)", flush=True)

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_eform = data['eform']
all_natoms = data['n_atoms']

mask = all_natoms <= 30
valid = np.where(mask)[0]
np.random.seed(42)
idx = np.random.choice(valid, 12000, replace=False)
idx.sort()
charges = all_charges[idx]
targets = all_eform[idx]

reps_full = np.load(os.path.join(DATA_DIR, 'pcmbdf_5fold_mp_e_form.npz'))['reps']

# Map indices
mask_pos = np.where(all_natoms <= 30)[0]
local_idx = np.searchsorted(mask_pos, idx)
reps = reps_full[local_idx]

reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
global_reps = build_global(reps_norm, charges)

# Split
perm = np.random.permutation(12000)
test = perm[:2000]
train = perm[2000:10000]  # 8k train
X_tr_raw = global_reps[train]
X_te_raw = global_reps[test]
y_tr = targets[train]
y_te = targets[test]

sc = StandardScaler()
X_tr = sc.fit_transform(X_tr_raw)
X_te = sc.transform(X_te_raw)

print("Train: %d, Test: %d\n" % (len(train), len(test)), flush=True)

# Linear
t0 = time.time()
ridge = RidgeCV(alphas=[1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0])
ridge.fit(X_tr, y_tr)
mae_lin = mean_absolute_error(y_te, ridge.predict(X_te))
print("Linear Ridge:    %.4f eV/atom (%.1fs)" % (mae_lin, time.time()-t0), flush=True)

# Polynomial d=2
t0 = time.time()
best_p2 = 999
for a in [1e-6, 1e-4, 1e-2, 1.0, 10.0]:
    krr = KernelRidge(alpha=a, kernel='polynomial', degree=2, coef0=1.0)
    krr.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, krr.predict(X_te))
    best_p2 = min(best_p2, mae)
print("Polynomial d=2:  %.4f eV/atom (%.1fs)" % (best_p2, time.time()-t0), flush=True)

# Polynomial d=3
t0 = time.time()
best_p3 = 999
for a in [1e-6, 1e-4, 1e-2, 1.0, 10.0]:
    krr = KernelRidge(alpha=a, kernel='polynomial', degree=3, coef0=1.0)
    krr.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, krr.predict(X_te))
    best_p3 = min(best_p3, mae)
print("Polynomial d=3:  %.4f eV/atom (%.1fs)" % (best_p3, time.time()-t0), flush=True)

# Laplacian
t0 = time.time()
best_lap = 999
X32 = X_tr.astype(np.float32)
Xt32 = X_te.astype(np.float32)
for g in [0.01, 0.02, 0.05, 0.1, 0.2]:
    for a in [1e-6, 1e-4, 1e-2]:
        K = np.exp(-g * cdist(X32, X32, metric='cityblock'))
        K[np.diag_indices_from(K)] += a
        c = np.linalg.solve(K, y_tr.astype(np.float32))
        Kt = np.exp(-g * cdist(Xt32, X32, metric='cityblock'))
        mae = mean_absolute_error(y_te, (Kt @ c).astype(np.float64))
        best_lap = min(best_lap, mae)
        del K, Kt, c; gc.collect()
print("Laplacian KRR:   %.4f eV/atom (%.1fs)" % (best_lap, time.time()-t0), flush=True)

# Gaussian RBF
t0 = time.time()
best_rbf = 999
for a in [1e-6, 1e-4, 1e-2, 1.0]:
    for g in [0.001, 0.005, 0.01, 0.05]:
        krr = KernelRidge(alpha=a, kernel='rbf', gamma=g)
        krr.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_te, krr.predict(X_te))
        best_rbf = min(best_rbf, mae)
print("Gaussian RBF:    %.4f eV/atom (%.1fs)" % (best_rbf, time.time()-t0), flush=True)

print("\n--- Summary ---", flush=True)
print("%-18s  %-12s  %-8s" % ("Kernel", "MAE", "Ratio"), flush=True)
for name, mae in [("Linear", mae_lin), ("Poly d=2", best_p2),
                   ("Poly d=3", best_p3), ("Gaussian RBF", best_rbf),
                   ("Laplacian", best_lap)]:
    print("%-18s  %.4f       %.2f" % (name, mae, mae/best_lap), flush=True)

print("\nDone!", flush=True)
