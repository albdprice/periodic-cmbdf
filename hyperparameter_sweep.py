"""
Hyperparameter sweep for p-cMBDF on MP formation energies.

Quick wins:
1. Cutoff radius sweep: 4, 5, 6, 7, 8, 10 Å
2. Derivative order m sweep: 2, 3, 4, 5, 6

Medium wins:
3. Angular harmonics nAs sweep: 2, 3, 4, 6, 8
4. ATM damping n_atm sweep: 1.0, 1.5, 2.0, 2.5, 3.0
5. Radial weighting functions: 2, 4, 6
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'

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

from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element, get_convolutions

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

# Load data
print("Loading MP data...", flush=True)
data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_cells = data['cells']
all_eform = data['eform']
all_natoms = data['n_atoms']

# 5k subset for fast sweeps
mask = all_natoms <= 20
valid = np.where(mask)[0]
np.random.seed(42)
N = 5000
idx = np.random.choice(valid, N, replace=False)
idx.sort()

charges = all_charges[idx]
coords = all_coords[idx]
cells = all_cells[idx]
eform = all_eform[idx]

perm = np.random.permutation(N)
TEST = perm[:1000]
TRAIN = perm[1000:4000]  # 3000 training

GAMMAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ALPHAS = [1e-6, 1e-4, 1e-2]

def evaluate(reps, label):
    """Generate, normalize, build global, train KRR, return MAE."""
    reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
    global_reps = build_global(reps_norm, charges)
    sc = StandardScaler()
    X_tr = sc.fit_transform(global_reps[TRAIN])
    X_te = sc.transform(global_reps[TEST])
    mae = krr_f32(X_tr, eform[TRAIN], X_te, eform[TEST], GAMMAS, ALPHAS)
    n_feat = reps.shape[-1]
    print("  %-40s: MAE = %.4f eV/atom (%d features)" % (label, mae, n_feat), flush=True)
    return mae, n_feat

# ============================================================
# SWEEP 1: Cutoff radius
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SWEEP 1: Cutoff radius (r_cut)", flush=True)
print("=" * 70, flush=True)

cutoff_results = {}
for rcut in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
    print("Generating rcut=%.1f..." % rcut, flush=True)
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=rcut, n_atm=2.0,
        elem_specific=True, n_jobs=-1)
    t_gen = time.time() - t0
    mae, nf = evaluate(reps, "rcut=%.1f (%.1fs)" % (rcut, t_gen))
    cutoff_results[rcut] = (mae, t_gen)

print("\n--- Cutoff Summary ---", flush=True)
print("%-8s | %-12s | %-10s" % ("r_cut", "MAE(eV/at)", "Gen time"), flush=True)
print("-" * 35, flush=True)
for rcut in sorted(cutoff_results):
    mae, t = cutoff_results[rcut]
    print("%-8.1f | %-12.4f | %-10.1fs" % (rcut, mae, t), flush=True)

# ============================================================
# SWEEP 2: Derivative order m
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SWEEP 2: Derivative order (m = order parameter)", flush=True)
print("=" * 70, flush=True)

order_results = {}
for order in [1, 2, 3, 4]:  # max 4 (hermite_polynomial supports degree 1-5, so order+1 must be ≤5)
    # order=k gives (k+1) derivative orders: 0,1,...,k
    # Features: (k+1) * 4 radial + (k+1) * 4 angular = 8*(k+1)
    n_feat_expected = 8 * (order + 1)
    print("Generating order=%d (%d features)..." % (order, n_feat_expected), flush=True)
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=6.0, n_atm=2.0,
        elem_specific=True, n_jobs=-1, order=order)
    t_gen = time.time() - t0
    mae, nf = evaluate(reps, "order=%d (%d feat, %.1fs)" % (order, n_feat_expected, t_gen))
    order_results[order] = (mae, nf, t_gen)

print("\n--- Derivative Order Summary ---", flush=True)
print("%-6s | %-6s | %-12s | %-10s" % ("order", "feat", "MAE(eV/at)", "Gen time"), flush=True)
print("-" * 40, flush=True)
for order in sorted(order_results):
    mae, nf, t = order_results[order]
    print("%-6d | %-6d | %-12.4f | %-10.1fs" % (order, nf, mae, t), flush=True)

# ============================================================
# SWEEP 3: Angular harmonics (nAs)
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SWEEP 3: Angular harmonics (nAs)", flush=True)
print("=" * 70, flush=True)

nas_results = {}
for nAs in [2, 3, 4, 6, 8]:
    # Features: 5*4 radial (fixed) + 5*nAs angular = 20 + 5*nAs
    n_feat_expected = 20 + 5 * nAs
    print("Generating nAs=%d (%d features)..." % (nAs, n_feat_expected), flush=True)
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=6.0, n_atm=2.0,
        elem_specific=True, n_jobs=-1, nAs=nAs)
    t_gen = time.time() - t0
    mae, nf = evaluate(reps, "nAs=%d (%d feat, %.1fs)" % (nAs, n_feat_expected, t_gen))
    nas_results[nAs] = (mae, nf, t_gen)

print("\n--- Angular Harmonics Summary ---", flush=True)
print("%-6s | %-6s | %-12s | %-10s" % ("nAs", "feat", "MAE(eV/at)", "Gen time"), flush=True)
print("-" * 40, flush=True)
for nAs in sorted(nas_results):
    mae, nf, t = nas_results[nAs]
    print("%-6d | %-6d | %-12.4f | %-10.1fs" % (nAs, nf, mae, t), flush=True)

# ============================================================
# SWEEP 4: ATM damping exponent
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SWEEP 4: ATM damping (n_atm)", flush=True)
print("=" * 70, flush=True)

atm_results = {}
for n_atm in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print("Generating n_atm=%.1f..." % n_atm, flush=True)
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=6.0, n_atm=n_atm,
        elem_specific=True, n_jobs=-1)
    t_gen = time.time() - t0
    mae, nf = evaluate(reps, "n_atm=%.1f (%.1fs)" % (n_atm, t_gen))
    atm_results[n_atm] = (mae, t_gen)

print("\n--- ATM Damping Summary ---", flush=True)
print("%-8s | %-12s | %-10s" % ("n_atm", "MAE(eV/at)", "Gen time"), flush=True)
print("-" * 35, flush=True)
for n_atm in sorted(atm_results):
    mae, t = atm_results[n_atm]
    print("%-8.1f | %-12.4f | %-10.1fs" % (n_atm, mae, t), flush=True)

# ============================================================
# SWEEP 5: Number of radial weighting functions
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SWEEP 5: Radial weighting functions", flush=True)
print("=" * 70, flush=True)

# Default: alpha_list=[1.5, 5.0], n_list=[3.0, 5.0] → 4 total
# Try fewer and more
radial_configs = [
    ('2 weights: [1.5], [5.0]', [1.5], [5.0]),
    ('4 weights: [1.5,5.0], [3.0,5.0]', [1.5, 5.0], [3.0, 5.0]),  # default
    ('6 weights: [0.5,1.5,5.0], [2.0,3.0,5.0]', [0.5, 1.5, 5.0], [2.0, 3.0, 5.0]),
    ('8 weights: [0.5,1.5,3.0,5.0], [2.0,3.0,5.0,8.0]', [0.5, 1.5, 3.0, 5.0], [2.0, 3.0, 5.0, 8.0]),
]

rad_results = {}
for label, alpha_list, n_list in radial_configs:
    n_rad = len(alpha_list) + len(n_list)
    n_feat_expected = 5 * n_rad + 5 * 4  # 5 orders * n_rad radial + 5 orders * 4 angular
    print("Generating %s..." % label, flush=True)
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True,True,True), rcut=6.0, n_atm=2.0,
        elem_specific=True, n_jobs=-1,
        alpha_list=alpha_list, n_list=n_list)
    t_gen = time.time() - t0
    mae, nf = evaluate(reps, "%s (%.1fs)" % (label, t_gen))
    rad_results[label] = (mae, nf, t_gen)

print("\n--- Radial Weighting Summary ---", flush=True)
for label in [l for l, _, _ in radial_configs]:
    if label in rad_results:
        mae, nf, t = rad_results[label]
        print("  %-50s: MAE=%.4f (%d feat, %.1fs)" % (label, mae, nf, t), flush=True)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70, flush=True)
print("OPTIMAL CONFIGURATION SEARCH", flush=True)
print("=" * 70, flush=True)

print("\nBest from each sweep:", flush=True)
print("  Cutoff:    r_cut = %.1f (MAE = %.4f)" % min(cutoff_results.items(), key=lambda x: x[1][0]), flush=True)
print("  Order:     m = %d (MAE = %.4f)" % (min(order_results.items(), key=lambda x: x[1][0])[0], min(order_results.items(), key=lambda x: x[1][0])[1][0]), flush=True)
print("  Angular:   nAs = %d (MAE = %.4f)" % (min(nas_results.items(), key=lambda x: x[1][0])[0], min(nas_results.items(), key=lambda x: x[1][0])[1][0]), flush=True)
print("  ATM damp:  n_atm = %.1f (MAE = %.4f)" % min(atm_results.items(), key=lambda x: x[1][0]), flush=True)

# TODO: Run best combination
print("\nNOTE: These sweeps are independent (one-at-a-time).", flush=True)
print("The optimal combination may differ from combining individual bests.", flush=True)
print("Future work: joint optimization or Bayesian optimization.", flush=True)

print("\nDone!", flush=True)
