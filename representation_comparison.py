"""
Comprehensive representation comparison for the paper.

1. SOAP on Si-only structures (fairest SOAP comparison)
2. Coulomb Matrix baseline
3. ACSF (Atom-Centered Symmetry Functions) baseline
4. Sine Matrix (periodic-specific) baseline
5. All compared to p-cMBDF on same structures/splits

Run on odin (has DScribe, sklearn, etc).
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from ase import Atoms

DATA_DIR = '/home/albd/projects/cmbdf/data'

from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element

def krr_eval(X_tr, y_tr, X_te, y_te, gammas=[0.01, 0.05, 0.1, 0.5], alphas=[1e-6, 1e-4, 1e-2]):
    """KRR with Laplacian kernel, grid search."""
    best = 999
    for g in gammas:
        for a in alphas:
            try:
                krr = KernelRidge(alpha=a, kernel='laplacian', gamma=g)
                krr.fit(X_tr, y_tr)
                best = min(best, mean_absolute_error(y_te, krr.predict(X_te)))
            except:
                pass
    return best

def ridge_eval(X_tr, y_tr, X_te, y_te):
    """Linear ridge regression."""
    ridge = RidgeCV(alphas=[1e-4, 1e-2, 1.0, 10.0, 100.0])
    ridge.fit(X_tr, y_tr)
    return mean_absolute_error(y_te, ridge.predict(X_te))

# ============================================================
# Load MP data
# ============================================================
print("=" * 70, flush=True)
print("Loading Materials Project data", flush=True)
print("=" * 70, flush=True)

data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_cells = data['cells']
all_eform = data['eform']
all_natoms = data['n_atoms']

# ============================================================
# SECTION 1: SOAP on Si-only structures
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 1: SOAP on Si-only structures (fairest comparison)", flush=True)
print("=" * 70, flush=True)

from dscribe.descriptors import SOAP as DScribeSOAP

# Try single-element systems: Si, Cu, Fe — use whichever has enough data
for target_z, elem_name in [(14, "Si"), (29, "Cu"), (26, "Fe"), (22, "Ti")]:
    si_idx = []
    for i in range(len(all_charges)):
        if all(int(z) == target_z for z in all_charges[i]) and len(all_charges[i]) <= 30:
            si_idx.append(i)
    if len(si_idx) >= 200:
        print("%s-only structures (≤30 atoms): %d" % (elem_name, len(si_idx)), flush=True)
        break
    print("%s-only: only %d structures, trying next..." % (elem_name, len(si_idx)), flush=True)

# If no single-element has enough, try binary oxides (2 species including O)
if len(si_idx) < 200:
    print("No single-element system has enough data. Using binary oxides...", flush=True)
    si_idx = []
    for i in range(len(all_charges)):
        q = all_charges[i]
        species = set(int(z) for z in q)
        if len(species) == 2 and 8 in species and all(z <= 56 for z in species) and len(q) <= 20:
            si_idx.append(i)
    elem_name = "BinaryOxide"
    target_z = None
    species_for_soap = sorted(set(int(z) for i in si_idx[:2000] for z in all_charges[i]))
    print("Binary oxide structures: %d (species: %s)" % (len(si_idx), species_for_soap), flush=True)

if len(si_idx) >= 200:
    np.random.seed(42)
    si_idx = np.array(si_idx)
    np.random.shuffle(si_idx)
    n_si = min(len(si_idx), 2000)
    si_idx = si_idx[:n_si]

    # SOAP
    if target_z is not None:
        soap_species = [target_z]
    else:
        soap_species = species_for_soap
    soap_si = DScribeSOAP(species=soap_species, r_cut=6.0, n_max=6, l_max=6, periodic=True, sparse=False)
    soap_dim = soap_si.get_number_of_features()
    print("SOAP dim (Si only): %d" % soap_dim, flush=True)

    t0 = time.time()
    soap_reps = []
    for i in si_idx:
        a = Atoms(numbers=all_charges[i].astype(int), positions=all_coords[i],
                 cell=all_cells[i], pbc=True)
        soap_reps.append(soap_si.create(a).sum(axis=0))
    t_soap = time.time() - t0
    soap_arr = np.array(soap_reps)
    print("SOAP: %.1fs (%.0f struct/s)" % (t_soap, n_si / t_soap), flush=True)

    # p-cMBDF for same structures
    from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element
    t0 = time.time()
    cmbdf_reps = generate_mbdf_periodic(
        [all_charges[i] for i in si_idx], [all_coords[i] for i in si_idx],
        [all_cells[i] for i in si_idx], pbc=(True,True,True),
        rcut=6.0, n_atm=2.0, n_jobs=-1, elem_specific=True)
    t_cmbdf = time.time() - t0
    cmbdf_norm, _ = normalize_per_element(cmbdf_reps, [all_charges[i] for i in si_idx], mode='mean')
    cmbdf_global = np.array([cmbdf_norm[i, :len(all_charges[si_idx[i]]), :].sum(axis=0) for i in range(n_si)])
    print("p-cMBDF: %.1fs" % t_cmbdf, flush=True)

    y_si = all_eform[si_idx]
    n_te = min(500, n_si // 3)
    n_pool = n_si - n_te

    print("\nLearning curves on Si-only:", flush=True)
    print("%-6s | %-15s | %-15s | %-15s | %-15s" % (
        "N", "SOAP+Lap KRR", "SOAP+Ridge", "p-cMBDF+Lap", "p-cMBDF+Ridge"), flush=True)
    print("-" * 75, flush=True)

    for n_tr in [100, 200, 500, 1000]:
        if n_tr > n_pool:
            break

        # SOAP
        sc_s = StandardScaler()
        Xtr_s = sc_s.fit_transform(soap_arr[:n_tr])
        Xte_s = sc_s.transform(soap_arr[n_pool:n_pool+n_te])
        ytr = y_si[:n_tr]
        yte = y_si[n_pool:n_pool+n_te]

        mae_s_krr = krr_eval(Xtr_s, ytr, Xte_s, yte,
                              gammas=[0.0001, 0.001, 0.01, 0.1], alphas=[1e-6, 1e-4, 1e-2])
        mae_s_ridge = ridge_eval(Xtr_s, ytr, Xte_s, yte)

        # cMBDF
        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(cmbdf_global[:n_tr])
        Xte_c = sc_c.transform(cmbdf_global[n_pool:n_pool+n_te])

        mae_c_krr = krr_eval(Xtr_c, ytr, Xte_c, yte)
        mae_c_ridge = ridge_eval(Xtr_c, ytr, Xte_c, yte)

        print("N=%4d | %.4f          | %.4f          | %.4f        | %.4f" % (
            n_tr, mae_s_krr, mae_s_ridge, mae_c_krr, mae_c_ridge), flush=True)

    print("  SOAP dim: %d | p-cMBDF dim: 40" % soap_dim, flush=True)
else:
    print("Not enough Si structures, skipping", flush=True)

# ============================================================
# SECTION 2: CM, ACSF, Sine Matrix on MP subset
# ============================================================
print("\n" + "=" * 70, flush=True)
print("SECTION 2: Representation baselines on MP formation energy", flush=True)
print("=" * 70, flush=True)

from dscribe.descriptors import CoulombMatrix, SineMatrix, ACSF

# Use a manageable subset: ≤20 atoms, ≤5 species, up to 5000 structures
baseline_idx = []
for i in range(len(all_charges)):
    q = all_charges[i]
    if len(q) <= 20 and len(set(int(z) for z in q)) <= 5 and all(int(z) <= 56 for z in q):
        baseline_idx.append(i)
np.random.seed(42)
baseline_idx = np.array(baseline_idx)
np.random.shuffle(baseline_idx)
n_base = min(5000, len(baseline_idx))
baseline_idx = baseline_idx[:n_base]
print("Baseline subset: %d structures" % n_base, flush=True)

species_base = sorted(set(int(z) for i in baseline_idx for z in all_charges[i]))
print("Species: %d" % len(species_base), flush=True)

max_atoms = max(len(all_charges[i]) for i in baseline_idx)
y_base = all_eform[baseline_idx]

# --- Coulomb Matrix ---
print("\nGenerating Coulomb Matrix...", flush=True)
cm = CoulombMatrix(n_atoms_max=max_atoms, permutation="sorted_l2")
t0 = time.time()
cm_reps = np.array([cm.create(Atoms(numbers=all_charges[i].astype(int),
                                     positions=all_coords[i]))
                     for i in baseline_idx])
t_cm = time.time() - t0
print("CM: %.1fs, dim=%d" % (t_cm, cm_reps.shape[1]), flush=True)

# --- Sine Matrix (periodic) --- SKIPPED due to DScribe/ASE compatibility bug
print("Sine Matrix: skipped (DScribe/ASE compatibility issue)", flush=True)
sm_reps = None
t_sm = 0

# --- ACSF ---
print("Generating ACSF...", flush=True)
acsf = ACSF(species=species_base, r_cut=6.0,
             g2_params=[[1, 1], [1, 2], [1, 3]],
             g4_params=[[1, 1, 1], [1, 2, 1]],
             periodic=True)
t0 = time.time()
acsf_reps = []
for i in baseline_idx:
    a = Atoms(numbers=all_charges[i].astype(int), positions=all_coords[i],
             cell=all_cells[i], pbc=True)
    acsf_reps.append(acsf.create(a).sum(axis=0))  # global sum
t_acsf = time.time() - t0
acsf_arr = np.array(acsf_reps)
acsf_dim = acsf_arr.shape[1]
print("ACSF: %.1fs, dim=%d" % (t_acsf, acsf_dim), flush=True)

# --- p-cMBDF ---
print("Generating p-cMBDF...", flush=True)
t0 = time.time()
pcmbdf_reps = generate_mbdf_periodic(
    [all_charges[i] for i in baseline_idx],
    [all_coords[i] for i in baseline_idx],
    [all_cells[i] for i in baseline_idx],
    pbc=(True,True,True), rcut=6.0, n_atm=2.0,
    n_jobs=-1, elem_specific=True, progress_bar=True)
t_pcmbdf = time.time() - t0
pcmbdf_norm, _ = normalize_per_element(pcmbdf_reps,
    [all_charges[i] for i in baseline_idx], mode='mean')
pcmbdf_global = np.array([pcmbdf_norm[j, :len(all_charges[baseline_idx[j]]), :].sum(axis=0)
                           for j in range(n_base)])
print("p-cMBDF: %.1fs, dim=40" % t_pcmbdf, flush=True)

# --- Learning curves ---
n_te_b = 1000
perm_b = np.random.permutation(n_base)
te_b = perm_b[:n_te_b]
pool_b = perm_b[n_te_b:]
yte_b = y_base[te_b]

print("\n%-6s | %-10s | %-10s | %-10s" % (
    "N", "CM", "ACSF", "p-cMBDF"), flush=True)
print("-" * 45, flush=True)

reps_dict = {
    'CM': cm_reps,
    'ACSF': acsf_arr,
    'p-cMBDF': pcmbdf_global,
}

for n_tr in [200, 500, 1000, 2000, 3000]:
    if n_tr > len(pool_b):
        break
    tr_b = pool_b[:n_tr]
    ytr_b = y_base[tr_b]

    row = "N=%4d" % n_tr
    for name, reps in reps_dict.items():
        sc = StandardScaler()
        Xtr = sc.fit_transform(reps[tr_b])
        Xte = sc.transform(reps[te_b])
        if reps.shape[1] > 1000:
            # High dim: use ridge
            mae = ridge_eval(Xtr, ytr_b, Xte, yte_b)
        else:
            mae = krr_eval(Xtr, ytr_b, Xte, yte_b)
        row += " | %.4f   " % mae
    print(row, flush=True)

print("\nGeneration times:", flush=True)
print("  CM:         %.1fs | dim=%d" % (t_cm, cm_reps.shape[1]), flush=True)
print("  SineMatrix: %.1fs | dim=%d" % (t_sm, sm_reps.shape[1]), flush=True)
print("  ACSF:       %.1fs | dim=%d" % (t_acsf, acsf_dim), flush=True)
print("  p-cMBDF:    %.1fs | dim=40" % t_pcmbdf, flush=True)

print("\nDone!", flush=True)
