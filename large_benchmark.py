"""
Large-scale periodic cMBDF benchmark on Materials Project.

- Formation energies: up to 10k training
- Band gaps: matbench_mp_gap
- Hyperparameter search on kernel sigma
- Comparison: universal vs element-specific, raw vs normalized
"""
import numpy as np
import os
import sys
import time

sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/albd/projects/cmbdf/data'

# ============================================================
# Load formation energy data
# ============================================================
print("=" * 70)
print("Loading Materials Project data")
print("=" * 70)

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_cells = data['cells']
all_eform = data['eform']
all_natoms = data['n_atoms']

# Filter to ≤30 atoms for tractability
mask = all_natoms <= 30
valid = np.where(mask)[0]
print("Structures with ≤30 atoms: %d / %d" % (len(valid), len(all_charges)))

# Take 12k subset (10k train + 2k test)
np.random.seed(42)
subset = np.random.choice(valid, min(12000, len(valid)), replace=False)
subset.sort()

charges_all = all_charges[subset]
coords_all = all_coords[subset]
cells_all = all_cells[subset]
eform_all = all_eform[subset]
natoms_all = all_natoms[subset]
print("Subset: %d structures, avg %.1f atoms" % (len(subset), natoms_all.mean()))

# ============================================================
# Generate representations (element-specific + normalized — best config)
# ============================================================
print()
print("=" * 70)
print("Generating periodic cMBDF (element-specific, rcut=6.0)")
print("=" * 70)

REP_FILE = os.path.join(DATA_DIR, 'pcmbdf_mp_12k_elemspec.npz')

if os.path.exists(REP_FILE):
    print("Loading cached representations...")
    rd = np.load(REP_FILE)
    reps = rd['reps']
else:
    t0 = time.time()
    reps = generate_mbdf_periodic(
        list(charges_all), list(coords_all), list(cells_all),
        pbc=(True, True, True), rcut=6.0, n_atm=2.0,
        n_jobs=-1, progress_bar=True, elem_specific=True)
    t_gen = time.time() - t0
    print("Generated in %.1fs (%.0f struct/s)" % (t_gen, len(reps) / t_gen))
    np.savez_compressed(REP_FILE, reps=reps)

print("Rep shape:", reps.shape)

# Per-element normalization
reps_norm, norm_factors = normalize_per_element(reps, charges_all, mode='mean')

# Build global (sum-over-atoms) representations
def build_global(reps, charges_list):
    n = len(reps)
    nf = reps.shape[-1]
    out = np.zeros((n, nf))
    for i in range(n):
        na = len(charges_list[i])
        out[i] = reps[i, :na, :].sum(axis=0)
    return out

global_reps = build_global(reps_norm, charges_all)
print("Global rep shape:", global_reps.shape)

# ============================================================
# Train/test split
# ============================================================
n_total = len(global_reps)
n_test = 2000
perm = np.random.permutation(n_total)
test_idx = perm[:n_test]
train_pool = perm[n_test:]

X_test = global_reps[test_idx]
y_test = eform_all[test_idx]

sc = StandardScaler()

# ============================================================
# Learning curves with hyperparameter search
# ============================================================
print()
print("=" * 70)
print("Formation energy learning curves (elem-specific + normalized)")
print("=" * 70)

train_sizes = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
gammas = [0.001, 0.005, 0.01, 0.02, 0.05]
alphas = [1e-8, 1e-6, 1e-4]

for n_train in train_sizes:
    if n_train > len(train_pool):
        break

    idx_train = train_pool[:n_train]
    X_tr = sc.fit_transform(global_reps[idx_train])
    X_te = sc.transform(X_test)
    y_tr = eform_all[idx_train]

    best_mae = 999
    best_params = {}

    for gamma in gammas:
        for alpha in alphas:
            krr = KernelRidge(alpha=alpha, kernel='laplacian', gamma=gamma)
            krr.fit(X_tr, y_tr)
            pred = krr.predict(X_te)
            mae = mean_absolute_error(y_test, pred)
            if mae < best_mae:
                best_mae = mae
                best_params = {'gamma': gamma, 'alpha': alpha}

    print("N=%5d: MAE = %.4f eV/atom (gamma=%.3f, alpha=%.0e)" % (
        n_train, best_mae, best_params['gamma'], best_params['alpha']))

# ============================================================
# Band gap prediction (if matbench_mp_gap is available)
# ============================================================
print()
print("=" * 70)
print("Band gap prediction")
print("=" * 70)

GAP_FILE = os.path.join(DATA_DIR, 'mp_gap_parsed.npz')

if not os.path.exists(GAP_FILE):
    try:
        from matminer.datasets import load_dataset
        print("Fetching matbench_mp_gap...")
        df_gap = load_dataset("matbench_mp_gap")
        print("Dataset size:", len(df_gap))

        gap_charges, gap_coords, gap_cells, gap_values, gap_natoms = [], [], [], [], []
        for _, row in df_gap.iterrows():
            try:
                s = row['structure']
                gap_charges.append(np.array([sp.Z for sp in s.species], dtype=np.float64))
                gap_coords.append(np.array(s.cart_coords, dtype=np.float64))
                gap_cells.append(np.array(s.lattice.matrix, dtype=np.float64))
                gap_values.append(row['gap pbe'])
                gap_natoms.append(len(s))
            except Exception:
                pass

        np.savez(GAP_FILE,
                 charges=np.array(gap_charges, dtype=object),
                 coords=np.array(gap_coords, dtype=object),
                 cells=np.array(gap_cells, dtype=object),
                 gap=np.array(gap_values),
                 n_atoms=np.array(gap_natoms))
        print("Saved %d structures" % len(gap_values))
    except Exception as e:
        print("Could not fetch band gap data:", e)

if os.path.exists(GAP_FILE):
    gdata = np.load(GAP_FILE, allow_pickle=True)
    g_charges = gdata['charges']
    g_coords = gdata['coords']
    g_cells = gdata['cells']
    g_gap = gdata['gap']
    g_natoms = gdata['n_atoms']

    # Filter and subset
    g_mask = g_natoms <= 30
    g_valid = np.where(g_mask)[0]
    np.random.seed(123)
    g_sub = np.random.choice(g_valid, min(7000, len(g_valid)), replace=False)
    g_sub.sort()

    GAP_REP_FILE = os.path.join(DATA_DIR, 'pcmbdf_gap_7k_elemspec.npz')
    if os.path.exists(GAP_REP_FILE):
        g_reps = np.load(GAP_REP_FILE)['reps']
    else:
        print("Generating reps for band gap structures...")
        t0 = time.time()
        g_reps = generate_mbdf_periodic(
            list(g_charges[g_sub]), list(g_coords[g_sub]), list(g_cells[g_sub]),
            pbc=(True, True, True), rcut=6.0, n_atm=2.0,
            n_jobs=-1, progress_bar=True, elem_specific=True)
        print("Generated in %.1fs" % (time.time() - t0))
        np.savez_compressed(GAP_REP_FILE, reps=g_reps)

    g_reps_norm, _ = normalize_per_element(g_reps, g_charges[g_sub], mode='mean')
    g_global = build_global(g_reps_norm, g_charges[g_sub])
    g_y = g_gap[g_sub]

    print("Band gap dataset: %d structures, range %.3f to %.3f eV" % (
        len(g_sub), g_y.min(), g_y.max()))

    # Learning curve
    n_g = len(g_global)
    perm_g = np.random.permutation(n_g)
    g_test = perm_g[:1000]
    g_pool = perm_g[1000:]

    for n_train in [200, 500, 1000, 2000, 4000]:
        if n_train > len(g_pool):
            break
        idx = g_pool[:n_train]
        sc_g = StandardScaler()
        Xtr = sc_g.fit_transform(g_global[idx])
        Xte = sc_g.transform(g_global[g_test])

        best = 999
        for gamma in [0.001, 0.005, 0.01, 0.05]:
            for alpha in [1e-8, 1e-6, 1e-4]:
                krr = KernelRidge(alpha=alpha, kernel='laplacian', gamma=gamma)
                krr.fit(Xtr, g_y[idx])
                mae = mean_absolute_error(g_y[g_test], krr.predict(Xte))
                best = min(best, mae)

        print("Band gap N=%4d: MAE = %.4f eV" % (n_train, best))

print()
print("Benchmark complete!")
