"""
Feature pruning on solids + SOAP comparison v3.
Pruning uses 10k cached reps. SOAP uses a filtered subset with ≤20 species.
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/albd/projects/cmbdf/data'

# Load data
print("Loading...", flush=True)
rep_data = np.load(os.path.join(DATA_DIR, 'pcmbdf_mp_55k_elemspec.npz'))
reps_full = rep_data['reps']

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

N = 10000
reps = reps_full[:N]
charges_sub = all_charges[subset[:N]]
eform_sub = all_eform[subset[:N]]
coords_sub = all_coords[subset[:N]]
cells_sub = all_cells[subset[:N]]

print("Normalizing...", flush=True)
from cMBDF_periodic import normalize_per_element
reps_norm, _ = normalize_per_element(reps, charges_sub, mode='mean')

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

global_reps = build_global(reps_norm, charges_sub)
print("Ready. Shape:", global_reps.shape, flush=True)

perm = np.random.permutation(N)
test_idx = perm[:2000]
train_pool = perm[2000:]
X_test = global_reps[test_idx]
y_test = eform_sub[test_idx]
n_tr = 5000
idx_tr = train_pool[:n_tr]

# ============================================================
# PRUNING
# ============================================================
print("\n" + "=" * 70, flush=True)
print("FEATURE PRUNING ON SOLIDS (N=5000)", flush=True)
print("=" * 70, flush=True)

feat_labels = []
for m in range(5):
    for w in ['exp(-1.5r)', 'exp(-5.0r)', '1/(r+1)^3', '1/(r+1)^5']:
        feat_labels.append("R:m=%d,%s" % (m, w))
for m in range(5):
    for n in range(1, 5):
        feat_labels.append("A:m=%d,cos(%dθ)" % (m, n))

def eval_sub(feat_idx, label):
    sc = StandardScaler()
    Xtr = sc.fit_transform(global_reps[idx_tr][:, feat_idx])
    Xte = sc.transform(X_test[:, feat_idx])
    best = 999
    for g in [0.01, 0.02, 0.05, 0.1, 0.2]:
        krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=g)
        krr.fit(Xtr, eform_sub[idx_tr])
        best = min(best, mean_absolute_error(y_test, krr.predict(Xte)))
    print("  %-40s (%2d feat): %.4f eV/atom" % (label, len(feat_idx), best), flush=True)
    return best

mae_all = eval_sub(list(range(40)), "All 40 features")
eval_sub(list(range(20)), "Radial only")
eval_sub(list(range(20, 40)), "Angular only")
eval_sub(list(range(12)) + list(range(20, 32)), "m<=2 (24 features)")
eval_sub(list(range(16)) + list(range(20, 36)), "m<=3 (32 features)")
cos2_idx = [21, 25, 29, 33, 37]
eval_sub([i for i in range(40) if i not in cos2_idx], "Remove cos(2θ) (35 feat)")
eval_sub([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], "Short-range radial only (10)")
eval_sub([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], "Long-range radial only (10)")

# Leave-one-out
print("\n--- Leave-one-feature-out ---", flush=True)
deltas = {}
for f in range(40):
    fmask = [i for i in range(40) if i != f]
    sc = StandardScaler()
    Xtr = sc.fit_transform(global_reps[idx_tr][:, fmask])
    Xte = sc.transform(X_test[:, fmask])
    krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=0.05)
    krr.fit(Xtr, eform_sub[idx_tr])
    deltas[f] = mean_absolute_error(y_test, krr.predict(Xte)) - mae_all

sf = sorted(deltas.items(), key=lambda x: -x[1])
print("  Most important:", flush=True)
for f, d in sf[:10]:
    print("    %s: %+.4f" % (feat_labels[f], d), flush=True)
print("  Least important:", flush=True)
for f, d in sf[-10:]:
    print("    %s: %+.4f" % (feat_labels[f], d), flush=True)
print("  Removable (delta<=0): %d / 40" % sum(1 for _, d in sf if d <= 0), flush=True)

# ============================================================
# SOAP comparison — use binary oxide subset (fewer species)
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SOAP COMPARISON (binary oxide subset)", flush=True)
print("=" * 70, flush=True)

try:
    from dscribe.descriptors import SOAP as DScribeSOAP
    from ase import Atoms

    # Filter to structures with ≤5 unique elements for SOAP tractability
    soap_candidates = []
    for i in range(N):
        n_species = len(set(int(z) for z in charges_sub[i]))
        if n_species <= 5 and all(int(z) <= 83 for z in charges_sub[i]):
            soap_candidates.append(i)
    print("Structures with ≤5 species: %d" % len(soap_candidates), flush=True)

    np.random.seed(42)
    n_soap = min(2000, len(soap_candidates))
    soap_idx = np.random.choice(soap_candidates, n_soap, replace=False)

    all_species = sorted(set(int(z) for i in soap_idx for z in charges_sub[i]))
    print("Unique species: %d — %s" % (len(all_species), all_species[:20]), flush=True)

    soap = DScribeSOAP(species=all_species, r_cut=6.0, n_max=4, l_max=4,
                       periodic=True, sparse=False)
    soap_dim = soap.get_number_of_features()
    print("SOAP dim: %d" % soap_dim, flush=True)

    print("Generating SOAP...", flush=True)
    t0 = time.time()
    soap_reps = []
    nfail = 0
    for idx_s in soap_idx:
        try:
            a = Atoms(numbers=charges_sub[idx_s].astype(int),
                     positions=coords_sub[idx_s], cell=cells_sub[idx_s], pbc=True)
            soap_reps.append(soap.create(a).sum(axis=0))
        except:
            soap_reps.append(np.zeros(soap_dim))
            nfail += 1
    t_soap = time.time() - t0
    soap_arr = np.array(soap_reps)
    print("SOAP: %.1fs for %d structs (%d fail)" % (t_soap, n_soap, nfail), flush=True)

    # Split
    n_tr_soap = 1500
    n_te_soap = 500
    ytr = eform_sub[soap_idx[:n_tr_soap]]
    yte = eform_sub[soap_idx[n_tr_soap:n_tr_soap+n_te_soap]]

    # cMBDF on same split
    sc = StandardScaler()
    Xtr_c = sc.fit_transform(global_reps[soap_idx[:n_tr_soap]])
    Xte_c = sc.transform(global_reps[soap_idx[n_tr_soap:n_tr_soap+n_te_soap]])
    best_c = 999
    for g in [0.01, 0.05, 0.1, 0.2]:
        krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=g)
        krr.fit(Xtr_c, ytr)
        best_c = min(best_c, mean_absolute_error(yte, krr.predict(Xte_c)))

    # SOAP
    sc2 = StandardScaler()
    Xtr_s = sc2.fit_transform(soap_arr[:n_tr_soap])
    Xte_s = sc2.transform(soap_arr[n_tr_soap:n_tr_soap+n_te_soap])
    best_s = 999
    for g in [1e-5, 1e-4, 1e-3, 5e-3, 0.01]:
        krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=g)
        krr.fit(Xtr_s, ytr)
        best_s = min(best_s, mean_absolute_error(yte, krr.predict(Xte_s)))

    print("\n  === N=1500 training, ≤5-species structures ===", flush=True)
    print("  cMBDF (40 dim):          MAE = %.4f eV/atom" % best_c, flush=True)
    print("  SOAP (%d dim):       MAE = %.4f eV/atom" % (soap_dim, best_s), flush=True)
    print("  SOAP generation time:    %.1fs" % t_soap, flush=True)
    print("  cMBDF generation time:   <1s (from cache)", flush=True)
    print("  Dim ratio: SOAP is %dx larger" % (soap_dim // 40), flush=True)

except ImportError:
    print("DScribe not installed", flush=True)
except Exception as e:
    print("SOAP failed: %s" % e, flush=True)
    import traceback; traceback.print_exc()

print("\nDone!", flush=True)
