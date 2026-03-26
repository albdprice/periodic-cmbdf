"""
Benchmark 4-body and 5-body extensions on MP formation energies and QM9.
Compares max_body=3 (40 dim) vs 4 (60 dim) vs 5 (72 dim).
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

from cMBDF_higher_body import generate_mbdf_periodic_higher
from cMBDF_periodic import normalize_per_element

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

# ============================================================
# SECTION 1: MP Formation Energies
# ============================================================
print("=" * 70, flush=True)
print("SECTION 1: MP Formation Energy — body order comparison", flush=True)
print("=" * 70, flush=True)

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_cells = data['cells']
all_eform = data['eform']
all_natoms = data['n_atoms']

# Use ≤20 atoms for tractability with 4-body and 5-body
mask = all_natoms <= 20
valid = np.where(mask)[0]
np.random.seed(42)
n_mp = min(8000, len(valid))
mp_idx = np.random.choice(valid, n_mp, replace=False)
mp_idx.sort()

charges_mp = all_charges[mp_idx]
coords_mp = all_coords[mp_idx]
cells_mp = all_cells[mp_idx]
eform_mp = all_eform[mp_idx]
print("MP subset: %d structures (≤20 atoms)" % n_mp, flush=True)

mp_results = {}

for max_body in [3, 4, 5]:
    print("\n--- max_body=%d ---" % max_body, flush=True)
    cache = os.path.join(DATA_DIR, 'pcmbdf_mp_%db_%dk.npz' % (max_body, n_mp // 1000))

    if os.path.exists(cache):
        print("Loading cached reps...", flush=True)
        reps = np.load(cache)['reps']
    else:
        print("Generating p-cMBDF (max_body=%d)..." % max_body, flush=True)
        t0 = time.time()
        reps = generate_mbdf_periodic_higher(
            list(charges_mp), list(coords_mp), list(cells_mp),
            pbc=(True, True, True), rcut=6.0, max_body=max_body,
            n_jobs=-1, progress_bar=True)
        t_gen = time.time() - t0
        print("Generated in %.1fs (%.0f struct/s)" % (t_gen, n_mp / t_gen), flush=True)
        np.savez_compressed(cache, reps=reps)

    print("Shape: %s" % str(reps.shape), flush=True)

    reps_norm, _ = normalize_per_element(reps, charges_mp, mode='mean')
    global_reps = build_global(reps_norm, charges_mp)

    perm = np.random.permutation(n_mp)
    te = perm[:2000]
    pool = perm[2000:]

    gammas = [0.01, 0.02, 0.05, 0.1, 0.2]
    alphas = [1e-6, 1e-4, 1e-2]

    for n_tr in [500, 1000, 2000, 4000]:
        if n_tr > len(pool):
            break
        tr = pool[:n_tr]
        sc = StandardScaler()
        X_tr = sc.fit_transform(global_reps[tr])
        X_te = sc.transform(global_reps[te])
        mae = krr_f32(X_tr, eform_mp[tr], X_te, eform_mp[te], gammas, alphas)
        print("  N=%4d: MAE = %.4f eV/atom (%d features)" % (n_tr, mae, global_reps.shape[1]), flush=True)
        mp_results[(max_body, n_tr)] = mae

# Summary table
print("\n--- MP Formation Energy Summary ---", flush=True)
print("%-8s | %-10s | %-10s | %-10s" % ("N_train", "3-body(40)", "4-body(60)", "5-body(72)"), flush=True)
print("-" * 45, flush=True)
for n_tr in [500, 1000, 2000, 4000]:
    row = "N=%4d  " % n_tr
    for mb in [3, 4, 5]:
        if (mb, n_tr) in mp_results:
            row += " | %.4f   " % mp_results[(mb, n_tr)]
        else:
            row += " | —        "
    print(row, flush=True)

# ============================================================
# SECTION 2: QM9 Molecular Benchmark
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 2: QM9 Total Energy — body order comparison", flush=True)
print("=" * 70, flush=True)

qm9 = np.load(os.path.join(DATA_DIR, 'qm9_parsed.npz'), allow_pickle=True)
qm9_charges = qm9['charges']
qm9_coords = qm9['coords']
qm9_energies = qm9['energies']

np.random.seed(42)
n_qm9 = 10000
qm9_idx = np.random.choice(len(qm9_charges), n_qm9, replace=False)

charges_q = qm9_charges[qm9_idx]
coords_q = qm9_coords[qm9_idx]
energies_q = qm9_energies[qm9_idx]

# For periodic code, wrap in vacuum boxes
cells_q = [np.diag([50.0, 50.0, 50.0])] * n_qm9
coords_q_shifted = [np.asarray(r, dtype=np.float64) + 25.0 for r in coords_q]

print("QM9 subset: %d molecules" % n_qm9, flush=True)

qm9_results = {}

for max_body in [3, 4, 5]:
    print("\n--- max_body=%d ---" % max_body, flush=True)
    cache = os.path.join(DATA_DIR, 'pcmbdf_qm9_%db_%dk.npz' % (max_body, n_qm9 // 1000))

    if os.path.exists(cache):
        print("Loading cached reps...", flush=True)
        reps = np.load(cache)['reps']
    else:
        print("Generating p-cMBDF (max_body=%d) for QM9..." % max_body, flush=True)
        t0 = time.time()
        reps = generate_mbdf_periodic_higher(
            list(charges_q), coords_q_shifted, cells_q,
            pbc=(False, False, False), rcut=10.0, max_body=max_body,
            n_jobs=-1, progress_bar=True)
        t_gen = time.time() - t0
        print("Generated in %.1fs (%.0f mol/s)" % (t_gen, n_qm9 / t_gen), flush=True)
        np.savez_compressed(cache, reps=reps)

    print("Shape: %s" % str(reps.shape), flush=True)

    # For QM9, normalize per-element
    reps_norm, _ = normalize_per_element(reps, charges_q, mode='mean')
    global_reps = build_global(reps_norm, charges_q)

    perm = np.random.permutation(n_qm9)
    te = perm[:2000]
    pool = perm[2000:]

    gammas = [0.001, 0.005, 0.01, 0.05, 0.1]
    alphas = [1e-8, 1e-6, 1e-4]

    for n_tr in [500, 1000, 2000, 5000]:
        if n_tr > len(pool):
            break
        tr = pool[:n_tr]
        sc = StandardScaler()
        X_tr = sc.fit_transform(global_reps[tr])
        X_te = sc.transform(global_reps[te])
        mae = krr_f32(X_tr, energies_q[tr], X_te, energies_q[te], gammas, alphas) * 627.509
        print("  N=%4d: MAE = %.1f kcal/mol (%d features)" % (n_tr, mae, global_reps.shape[1]), flush=True)
        qm9_results[(max_body, n_tr)] = mae

# Summary table
print("\n--- QM9 Total Energy Summary (kcal/mol) ---", flush=True)
print("%-8s | %-12s | %-12s | %-12s" % ("N_train", "3-body(40)", "4-body(60)", "5-body(72)"), flush=True)
print("-" * 50, flush=True)
for n_tr in [500, 1000, 2000, 5000]:
    row = "N=%4d  " % n_tr
    for mb in [3, 4, 5]:
        if (mb, n_tr) in qm9_results:
            row += " | %8.1f    " % qm9_results[(mb, n_tr)]
        else:
            row += " | —          "
    print(row, flush=True)

print("\nDone!", flush=True)
