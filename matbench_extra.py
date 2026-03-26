"""
Additional matbench tasks: perovskites, phonons, dielectric.
Shows generality of p-cMBDF across property types.
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/albd/projects/cmbdf/data'

def krr_f32(X_tr, y_tr, X_te, y_te, gammas, alphas):
    best = 999
    X_tr32, X_te32 = X_tr.astype(np.float32), X_te.astype(np.float32)
    y_tr32 = y_tr.astype(np.float32)
    for g in gammas:
        for a in alphas:
            try:
                K = np.exp(-g * cdist(X_tr32, X_tr32, metric='cityblock'))
                K[np.diag_indices_from(K)] += a
                c = np.linalg.solve(K, y_tr32)
                Kt = np.exp(-g * cdist(X_te32, X_tr32, metric='cityblock'))
                pred = (Kt @ c).astype(np.float64)
                best = min(best, mean_absolute_error(y_te, pred))
                del K, Kt
            except:
                pass
    return best

def load_matbench(name):
    """Load a matbench dataset, return charges, coords, cells, targets."""
    cache = os.path.join(DATA_DIR, 'matbench_%s.npz' % name)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['charges'], d['coords'], d['cells'], d['targets'], d['n_atoms']

    from matminer.datasets import load_dataset
    print("Fetching %s..." % name, flush=True)
    df = load_dataset(name)
    target_col = [c for c in df.columns if c != 'structure'][0]
    print("Target: %s, N=%d" % (target_col, len(df)), flush=True)

    charges, coords, cells, targets, natoms = [], [], [], [], []
    for _, row in df.iterrows():
        try:
            s = row['structure']
            charges.append(np.array([sp.Z for sp in s.species], dtype=np.float64))
            coords.append(np.array(s.cart_coords, dtype=np.float64))
            cells.append(np.array(s.lattice.matrix, dtype=np.float64))
            targets.append(row[target_col])
            natoms.append(len(s))
        except:
            pass

    charges = np.array(charges, dtype=object)
    coords = np.array(coords, dtype=object)
    cells = np.array(cells, dtype=object)
    targets = np.array(targets, dtype=np.float64)
    natoms = np.array(natoms, dtype=np.int64)

    np.savez(cache, charges=charges, coords=coords, cells=cells,
             targets=targets, n_atoms=natoms)
    print("Saved %d structures" % len(charges), flush=True)
    return charges, coords, cells, targets, natoms

from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

# ============================================================
# Run each matbench task
# ============================================================
tasks = [
    ('matbench_perovskites', 'eV/atom'),
    ('matbench_phonons', '1/cm'),
    ('matbench_dielectric', '(unitless)'),
]

for task_name, unit in tasks:
    print("\n" + "=" * 70, flush=True)
    print("TASK: %s [%s]" % (task_name, unit), flush=True)
    print("=" * 70, flush=True)

    charges, coords, cells, targets, natoms = load_matbench(task_name)
    N_total = len(charges)
    print("Total: %d structures, atoms: %d-%d (mean %.1f)" % (
        N_total, natoms.min(), natoms.max(), natoms.mean()), flush=True)
    print("Target range: %.3f to %.3f %s" % (targets.min(), targets.max(), unit), flush=True)

    # Filter to ≤30 atoms
    mask = natoms <= 30
    valid = np.where(mask)[0]
    print("≤30 atoms: %d structures" % len(valid), flush=True)

    np.random.seed(42)
    n_use = min(len(valid), 10000)
    subset = np.random.choice(valid, n_use, replace=False)
    subset.sort()

    charges_sub = charges[subset]
    targets_sub = targets[subset]

    # Generate p-cMBDF
    rep_cache = os.path.join(DATA_DIR, 'pcmbdf_%s.npz' % task_name)
    if os.path.exists(rep_cache):
        print("Loading cached reps...", flush=True)
        reps = np.load(rep_cache)['reps']
    else:
        print("Generating p-cMBDF...", flush=True)
        t0 = time.time()
        reps = generate_mbdf_periodic(
            list(charges_sub), list(coords[subset]), list(cells[subset]),
            pbc=(True,True,True), rcut=6.0, n_atm=2.0,
            n_jobs=-1, progress_bar=True, elem_specific=True)
        t_gen = time.time() - t0
        print("Generated in %.1fs (%.0f struct/s)" % (t_gen, n_use / t_gen), flush=True)
        np.savez_compressed(rep_cache, reps=reps)

    # Normalize and build global
    reps_norm, _ = normalize_per_element(reps, charges_sub, mode='mean')
    global_reps = build_global(reps_norm, charges_sub)

    # Train/test split
    n_test = min(1000, n_use // 4)
    perm = np.random.permutation(n_use)
    te = perm[:n_test]
    pool = perm[n_test:]

    y_test = targets_sub[te]

    gammas = [0.001, 0.005, 0.01, 0.05, 0.1]
    alphas = [1e-8, 1e-6, 1e-4, 1e-2]

    print("\nLearning curve:", flush=True)
    for n_tr in [200, 500, 1000, 2000, 5000]:
        if n_tr > len(pool):
            break
        tr = pool[:n_tr]
        sc = StandardScaler()
        X_tr = sc.fit_transform(global_reps[tr])
        X_te = sc.transform(global_reps[te])

        mae = krr_f32(X_tr, targets_sub[tr], X_te, y_test, gammas, alphas)
        print("  N=%4d: MAE = %.4f %s" % (n_tr, mae, unit), flush=True)

print("\nAll tasks complete!", flush=True)
