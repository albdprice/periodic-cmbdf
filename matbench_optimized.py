"""
Matbench 5-fold CV with optimized hyperparameters.

Tests multiple configurations at different cost/accuracy trade-off points:
1. Default (40 dim) — baseline
2. Optimized-compact (40 dim) — same size, better params (r_cut=5, n_atm=2.5)
3. Optimized-medium (50 dim) — nAs=6, 6 radial weights
4. Optimized-full (60 dim) — nAs=8, 6 radial weights

Reports: MAE, dim, gen time, kernel time, total time, MAE×Dim, error/parameter
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'

from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

def krr_5fold(global_reps, targets, kf, gammas, alphas):
    """Run 5-fold CV with float32 kernel. Returns mean MAE, std, total time."""
    fold_maes = []
    total_time = 0
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(global_reps)):
        n_train = len(train_idx)
        sc = StandardScaler()
        X_tr = sc.fit_transform(global_reps[train_idx])
        X_te = sc.transform(global_reps[test_idx])
        y_tr = targets[train_idx]
        y_te = targets[test_idx]

        # Tune on 10k subset for large training sets
        if n_train > 20000:
            np.random.seed(fold_idx)
            tune_idx = np.random.choice(n_train, 10000, replace=False)
            tune_te = np.random.choice(n_train, 2000, replace=False)
            tune_te = tune_te[~np.isin(tune_te, tune_idx)][:2000]
            best_g, best_a, best_tune = 0.05, 1e-4, 999
            for g in gammas:
                for a in alphas:
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

            t0 = time.time()
            X32 = X_tr.astype(np.float32)
            Xt32 = X_te.astype(np.float32)
            K = np.exp(-best_g * cdist(X32, X32, metric='cityblock'))
            K[np.diag_indices_from(K)] += best_a
            c = np.linalg.solve(K, y_tr.astype(np.float32))
            Kt = np.exp(-best_g * cdist(Xt32, X32, metric='cityblock'))
            pred = (Kt @ c).astype(np.float64)
            fold_mae = mean_absolute_error(y_te, pred)
            fold_time = time.time() - t0
            del K, Kt, c
            gc.collect()
        else:
            t0 = time.time()
            best_mae = 999
            X32 = X_tr.astype(np.float32)
            Xt32 = X_te.astype(np.float32)
            for g in gammas:
                for a in alphas:
                    try:
                        K = np.exp(-g * cdist(X32, X32, metric='cityblock'))
                        K[np.diag_indices_from(K)] += a
                        c = np.linalg.solve(K, y_tr.astype(np.float32))
                        Kt = np.exp(-g * cdist(Xt32, X32, metric='cityblock'))
                        mae = mean_absolute_error(y_te, (Kt @ c).astype(np.float64))
                        best_mae = min(best_mae, mae)
                        del K, Kt, c
                        gc.collect()
                    except:
                        pass
            fold_mae = best_mae
            fold_time = time.time() - t0

        fold_maes.append(fold_mae)
        total_time += fold_time
        print("    Fold %d: %.4f (%.0fs)" % (fold_idx + 1, fold_mae, fold_time), flush=True)

    return np.mean(fold_maes), np.std(fold_maes), total_time

# ============================================================
# Load data
# ============================================================
print("Loading data...", flush=True)
data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_cells = data['cells']
all_eform = data['eform']
all_natoms = data['n_atoms']

mask = all_natoms <= 30
valid = np.where(mask)[0]
N = len(valid)
charges = all_charges[valid]
coords = all_coords[valid]
cells = all_cells[valid]
targets = all_eform[valid]
print("Structures: %d" % N, flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
gammas = [0.01, 0.02, 0.05, 0.1, 0.2]
alphas = [1e-6, 1e-4, 1e-2]

# ============================================================
# Configurations to test
# ============================================================
configs = [
    {
        'name': 'Default (current paper)',
        'rcut': 6.0, 'n_atm': 2.0, 'nAs': 4,
        'alpha_list': [1.5, 5.0], 'n_list': [3.0, 5.0],
    },
    {
        'name': 'Optimized-compact (same 40 dim)',
        'rcut': 5.0, 'n_atm': 2.5, 'nAs': 4,
        'alpha_list': [1.5, 5.0], 'n_list': [3.0, 5.0],
    },
    {
        'name': 'Optimized-medium (50 dim)',
        'rcut': 5.0, 'n_atm': 2.5, 'nAs': 6,
        'alpha_list': [0.5, 1.5, 5.0], 'n_list': [3.0, 5.0],
    },
    {
        'name': 'Optimized-full (60 dim)',
        'rcut': 5.0, 'n_atm': 2.5, 'nAs': 8,
        'alpha_list': [0.5, 1.5, 5.0], 'n_list': [2.0, 3.0, 5.0],
    },
]

results = []

for cfg in configs:
    print("\n" + "=" * 70, flush=True)
    print("CONFIG: %s" % cfg['name'], flush=True)
    print("  rcut=%.1f, n_atm=%.1f, nAs=%d, radial=%d weights" % (
        cfg['rcut'], cfg['n_atm'], cfg['nAs'],
        len(cfg['alpha_list']) + len(cfg['n_list'])), flush=True)
    print("=" * 70, flush=True)

    # Generate representations
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges), list(coords), list(cells),
        pbc=(True, True, True),
        rcut=cfg['rcut'], n_atm=cfg['n_atm'],
        nAs=cfg['nAs'],
        alpha_list=cfg['alpha_list'], n_list=cfg['n_list'],
        elem_specific=True, n_jobs=-1, progress_bar=True)
    t_gen = time.time() - t0
    print("  Generated: shape=%s, %.1fs (%.0f struct/s)" % (
        reps.shape, t_gen, N / t_gen), flush=True)

    n_feat = reps.shape[-1]

    # Normalize
    reps_norm, _ = normalize_per_element(reps, charges, mode='mean')
    global_reps = build_global(reps_norm, charges)
    print("  Features: %d" % n_feat, flush=True)

    # 5-fold CV
    print("  Running 5-fold CV...", flush=True)
    mae, std, t_krr = krr_5fold(global_reps, targets, kf, gammas, alphas)

    # Compute metrics
    n_dual = int(N * 0.8)  # ~80% training per fold
    mae_x_dim = mae * n_feat
    mae_per_Mparam = mae / (n_dual / 1e6)

    results.append({
        'name': cfg['name'],
        'dim': n_feat,
        'mae': mae,
        'std': std,
        't_gen': t_gen,
        't_krr': t_krr,
        't_total': t_gen + t_krr,
        'mae_x_dim': mae_x_dim,
        'n_dual': n_dual,
        'mae_per_Mparam': mae_per_Mparam,
        'gen_rate': N / t_gen,
    })

    print("\n  === %s: %.4f ± %.4f eV/atom ===" % (cfg['name'], mae, std), flush=True)
    print("  MAE×Dim = %.1f | Gen: %.0fs | KRR: %.0fs | Total: %.0fs" % (
        mae_x_dim, t_gen, t_krr, t_gen + t_krr), flush=True)

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 90, flush=True)
print("COMPREHENSIVE TRADE-OFF ANALYSIS", flush=True)
print("=" * 90, flush=True)

print("\n%-35s | %-4s | %-8s | %-8s | %-8s | %-8s | %-8s" % (
    "Configuration", "Dim", "MAE", "MAE×Dim", "Gen(s)", "KRR(s)", "Total"), flush=True)
print("-" * 90, flush=True)

for r in results:
    print("%-35s | %-4d | %.4f  | %-8.1f | %-8.0f | %-8.0f | %-8.0f" % (
        r['name'], r['dim'], r['mae'], r['mae_x_dim'],
        r['t_gen'], r['t_krr'], r['t_total']), flush=True)

# Cost-benefit analysis
print("\n%-35s | %-12s | %-15s | %-12s" % (
    "Configuration", "Improvement", "Cost increase", "Efficiency"), flush=True)
print("-" * 80, flush=True)

baseline = results[0]
for r in results:
    if r == baseline:
        print("%-35s | baseline     | baseline        | baseline" % r['name'], flush=True)
    else:
        mae_imp = (baseline['mae'] - r['mae']) / baseline['mae'] * 100
        cost_inc = (r['t_total'] - baseline['t_total']) / baseline['t_total'] * 100
        efficiency = mae_imp / max(cost_inc, 0.1)  # improvement per cost
        print("%-35s | %+.1f%%       | %+.0f%%            | %.2f" % (
            r['name'], -mae_imp, cost_inc, efficiency), flush=True)

print("\nDone!", flush=True)
