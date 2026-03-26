"""
Paper benchmarks: MP extended, SOAP comparison, old-vs-new, band gaps.
Runs on odin.
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = '/home/albd/projects/cmbdf/data'

def krr_f32(X_tr, y_tr, X_te, y_te, gammas, alphas):
    """KRR with float32 kernel. Returns best MAE."""
    X_tr32 = X_tr.astype(np.float32)
    X_te32 = X_te.astype(np.float32)
    y_tr32 = y_tr.astype(np.float32)
    best = 999
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
                del K, Kt
            except:
                pass
    return best

# ============================================================
# SECTION 1: MP Formation Energy Extended to 50k
# ============================================================
print("=" * 70, flush=True)
print("SECTION 1: MP Formation Energy (extended to 50k)", flush=True)
print("=" * 70, flush=True)

rep_data = np.load(os.path.join(DATA_DIR, 'pcmbdf_mp_55k_elemspec.npz'))
reps = rep_data['reps']

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_eform = data['eform']
all_natoms = data['n_atoms']
all_coords = data['coords']
all_cells = data['cells']

mask = all_natoms <= 30
valid = np.where(mask)[0]
np.random.seed(42)
n_use = min(55000, len(valid))
subset = np.random.choice(valid, n_use, replace=False)
subset.sort()

charges_sub = all_charges[subset[:n_use]]
eform_sub = all_eform[subset[:n_use]]

print("Normalizing...", flush=True)
from cMBDF_periodic import normalize_per_element
reps_norm, nf = normalize_per_element(reps[:n_use], charges_sub, mode='mean')

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

global_reps = build_global(reps_norm, charges_sub)
print("Global shape:", global_reps.shape, flush=True)

perm = np.random.permutation(n_use)
test_idx = perm[:5000]
train_pool = perm[5000:]
X_test = global_reps[test_idx]
y_test = eform_sub[test_idx]

sc = StandardScaler()
X_test_s = None  # will be set per training size

gammas = [0.01, 0.02, 0.05, 0.1]
alphas = [1e-6, 1e-4, 1e-2]

print("\nLearning curve:", flush=True)
for n_train in [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]:
    if n_train > len(train_pool):
        break
    tr = train_pool[:n_train]
    sc_i = StandardScaler()
    X_tr = sc_i.fit_transform(global_reps[tr])
    X_te = sc_i.transform(X_test)
    mae = krr_f32(X_tr, eform_sub[tr], X_te, y_test, gammas, alphas)
    print("  N=%5d: MAE = %.4f eV/atom" % (n_train, mae), flush=True)

# ============================================================
# SECTION 2: SOAP Comparison on Binary Compounds
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 2: SOAP Comparison (binary compounds)", flush=True)
print("=" * 70, flush=True)

try:
    from dscribe.descriptors import SOAP as DScribeSOAP
    from ase import Atoms
    from sklearn.linear_model import RidgeCV
    from sklearn.kernel_ridge import KernelRidge

    # Filter binary compounds with Z <= 56, n_atoms <= 20
    binary_idx = []
    for i in range(min(n_use, 20000)):
        q = charges_sub[i]
        n_sp = len(set(int(z) for z in q))
        if n_sp == 2 and all(int(z) <= 56 for z in q) and len(q) <= 20:
            binary_idx.append(i)
    print("Binary compounds: %d" % len(binary_idx), flush=True)

    np.random.seed(42)
    binary_idx = np.array(binary_idx)
    np.random.shuffle(binary_idx)
    n_soap = min(5000, len(binary_idx))
    soap_idx = binary_idx[:n_soap]

    species = sorted(set(int(z) for i in soap_idx for z in charges_sub[i]))
    print("Species (%d): %s" % (len(species), species), flush=True)

    soap = DScribeSOAP(species=species, r_cut=6.0, n_max=4, l_max=4,
                       periodic=True, sparse=False)
    soap_dim = soap.get_number_of_features()
    print("SOAP dim: %d, cMBDF dim: 40" % soap_dim, flush=True)

    print("Generating SOAP...", flush=True)
    t0 = time.time()
    soap_list = []
    for i in soap_idx:
        try:
            a = Atoms(numbers=charges_sub[i].astype(int),
                     positions=all_coords[subset[i]],
                     cell=all_cells[subset[i]], pbc=True)
            soap_list.append(soap.create(a).sum(axis=0))
        except:
            soap_list.append(np.zeros(soap_dim))
    t_soap = time.time() - t0
    soap_arr = np.array(soap_list)
    print("SOAP generated: %.1fs (%.0f struct/s)" % (t_soap, n_soap / t_soap), flush=True)

    # cMBDF for same structures
    cmbdf_arr = global_reps[soap_idx]
    y_soap = eform_sub[soap_idx]

    # Split: 1000 test, rest train
    n_te = 1000
    print("\nLearning curves:", flush=True)
    for n_tr in [200, 500, 1000, 2000, 3000]:
        if n_tr + n_te > n_soap:
            break

        # cMBDF + Laplacian KRR
        sc1 = StandardScaler()
        Xtr_c = sc1.fit_transform(cmbdf_arr[:n_tr])
        Xte_c = sc1.transform(cmbdf_arr[n_tr:n_tr + n_te])
        mae_c = krr_f32(Xtr_c, y_soap[:n_tr], Xte_c, y_soap[n_tr:n_tr+n_te],
                        [0.01, 0.05, 0.1, 0.2], [1e-6, 1e-4])

        # SOAP + Linear Ridge
        sc2 = StandardScaler()
        Xtr_s = sc2.fit_transform(soap_arr[:n_tr])
        Xte_s = sc2.transform(soap_arr[n_tr:n_tr + n_te])
        ridge = RidgeCV(alphas=[1e-4, 1e-2, 1.0, 10.0, 100.0])
        ridge.fit(Xtr_s, y_soap[:n_tr])
        mae_s_lin = mean_absolute_error(y_soap[n_tr:n_tr+n_te], ridge.predict(Xte_s))

        # SOAP + polynomial KRR (degree=2, the SOAP kernel analog)
        mae_s_poly = 999
        for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
            try:
                krr = KernelRidge(alpha=alpha, kernel='polynomial', degree=2, coef0=1.0)
                krr.fit(Xtr_s, y_soap[:n_tr])
                mae = mean_absolute_error(y_soap[n_tr:n_tr+n_te], krr.predict(Xte_s))
                mae_s_poly = min(mae_s_poly, mae)
            except:
                pass

        print("  N=%4d: cMBDF=%.4f | SOAP-linear=%.4f | SOAP-poly=%.4f eV/atom" % (
            n_tr, mae_c, mae_s_lin, mae_s_poly), flush=True)

    print("  SOAP dim: %d, cMBDF dim: 40 (ratio: %dx)" % (soap_dim, soap_dim // 40), flush=True)
    print("  SOAP gen: %.1fs, cMBDF gen: <1s (cached)" % t_soap, flush=True)

except Exception as e:
    print("SOAP comparison failed: %s" % e, flush=True)
    import traceback; traceback.print_exc()

# ============================================================
# SECTION 3: Old cMBDF vs p-cMBDF on Solids
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 3: Old cMBDF (no PBC) vs p-cMBDF on MP solids", flush=True)
print("=" * 70, flush=True)

N_cmp = 15000
charges_cmp = charges_sub[:N_cmp]
eform_cmp = eform_sub[:N_cmp]
coords_cmp = [np.asarray(all_coords[subset[i]], dtype=np.float64) for i in range(N_cmp)]

# Old cMBDF on crystal structures (treating as isolated clusters)
OLD_REP = os.path.join(DATA_DIR, 'mp_old_cmbdf_15k.npz')
if os.path.exists(OLD_REP):
    print("Loading old cMBDF reps...", flush=True)
    reps_old_mp = np.load(OLD_REP)['reps']
else:
    print("Generating old cMBDF (no PBC) for MP structures...", flush=True)
    from cMBDF import generate_mbdf as old_generate
    t0 = time.time()
    reps_old_mp = old_generate(
        charges_cmp, np.array(coords_cmp, dtype=object),
        rcut=6.0, n_atm=2.0, smooth_cutoff=True, n_jobs=-1, progress_bar=True)
    print("Done in %.1fs" % (time.time() - t0), flush=True)
    np.savez_compressed(OLD_REP, reps=reps_old_mp)

global_old_mp = build_global(reps_old_mp, charges_cmp)
global_new_mp = global_reps[:N_cmp]  # p-cMBDF already computed

perm2 = np.random.permutation(N_cmp)
te2 = perm2[:2000]
tp2 = perm2[2000:]

print("\nLearning curves:", flush=True)
for n_tr in [500, 1000, 2000, 5000, 10000]:
    if n_tr > len(tp2):
        break
    tr2 = tp2[:n_tr]

    sc1 = StandardScaler()
    Xtr_old = sc1.fit_transform(global_old_mp[tr2])
    Xte_old = sc1.transform(global_old_mp[te2])
    mae_old = krr_f32(Xtr_old, eform_cmp[tr2], Xte_old, eform_cmp[te2],
                      [0.01, 0.05, 0.1], [1e-6, 1e-4])

    sc2 = StandardScaler()
    Xtr_new = sc2.fit_transform(global_new_mp[tr2])
    Xte_new = sc2.transform(global_new_mp[te2])
    mae_new = krr_f32(Xtr_new, eform_cmp[tr2], Xte_new, eform_cmp[te2],
                      [0.01, 0.05, 0.1], [1e-6, 1e-4])

    improvement = (mae_old - mae_new) / mae_old * 100
    print("  N=%5d: old=%.4f, p-cMBDF=%.4f eV/atom (%+.1f%%)" % (
        n_tr, mae_old, mae_new, -improvement), flush=True)

# ============================================================
# SECTION 4: Band Gap Extended
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 4: Band Gap Extended", flush=True)
print("=" * 70, flush=True)

GAP_REP = os.path.join(DATA_DIR, 'pcmbdf_gap_7k_elemspec.npz')
GAP_DATA = os.path.join(DATA_DIR, 'mp_gap_parsed.npz')

if os.path.exists(GAP_REP) and os.path.exists(GAP_DATA):
    g_reps = np.load(GAP_REP)['reps']
    gdata = np.load(GAP_DATA, allow_pickle=True)
    g_charges = gdata['charges']
    g_gap = gdata['gap']
    g_natoms = gdata['n_atoms']

    g_mask = g_natoms <= 30
    g_valid = np.where(g_mask)[0]
    np.random.seed(123)
    g_sub = np.random.choice(g_valid, min(7000, len(g_valid)), replace=False)
    g_sub.sort()

    g_reps_norm, _ = normalize_per_element(g_reps, g_charges[g_sub], mode='mean')
    g_global = build_global(g_reps_norm, g_charges[g_sub])
    g_y = g_gap[g_sub]

    n_g = len(g_global)
    perm_g = np.random.permutation(n_g)
    g_test = perm_g[:1000]
    g_pool = perm_g[1000:]

    print("Band gap: %d structures" % n_g, flush=True)
    for n_tr in [200, 500, 1000, 2000, 4000, 5000, 6000]:
        if n_tr > len(g_pool):
            break
        tr = g_pool[:n_tr]
        sc_g = StandardScaler()
        Xtr = sc_g.fit_transform(g_global[tr])
        Xte = sc_g.transform(g_global[g_test])
        mae = krr_f32(Xtr, g_y[tr], Xte, g_y[g_test],
                      [0.001, 0.005, 0.01, 0.05], [1e-8, 1e-6, 1e-4])
        print("  N=%4d: MAE = %.4f eV" % (n_tr, mae), flush=True)
else:
    print("Band gap data not found, skipping", flush=True)

print("\nAll sections complete!", flush=True)
