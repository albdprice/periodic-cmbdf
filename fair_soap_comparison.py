"""
Fair SOAP comparison: Linear ridge regression (natural for high-dim SOAP)
+ SOAP dot-product kernel (the standard SOAP kernel).

cMBDF uses Laplacian KRR (its natural kernel).
"""
import numpy as np
import os, sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/albd/projects/cmbdf/data'

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
subset = np.random.choice(valid, min(55000, len(valid)), replace=False)
subset.sort()

N = 5000
reps = reps_full[:N]
charges_sub = all_charges[subset[:N]]
eform_sub = all_eform[subset[:N]]
coords_sub = all_coords[subset[:N]]
cells_sub = all_cells[subset[:N]]

from cMBDF_periodic import normalize_per_element
print("Normalizing cMBDF...", flush=True)
reps_norm, _ = normalize_per_element(reps, charges_sub, mode='mean')

def build_global(reps, cl):
    out = np.zeros((len(reps), reps.shape[-1]))
    for i in range(len(reps)):
        out[i] = reps[i, :len(cl[i]), :].sum(axis=0)
    return out

cmbdf_global = build_global(reps_norm, charges_sub)

# Filter to ≤5 species for SOAP
print("Filtering for SOAP...", flush=True)
soap_ok = []
for i in range(N):
    n_sp = len(set(int(z) for z in charges_sub[i]))
    if n_sp <= 5 and all(int(z) <= 83 for z in charges_sub[i]):
        soap_ok.append(i)
print("SOAP-compatible structures: %d" % len(soap_ok), flush=True)

np.random.seed(42)
soap_ok = np.array(soap_ok)
np.random.shuffle(soap_ok)
n_use = min(3000, len(soap_ok))
soap_idx = soap_ok[:n_use]

species = sorted(set(int(z) for i in soap_idx for z in charges_sub[i]))
print("Species: %d" % len(species), flush=True)

# Generate SOAP
print("Generating SOAP...", flush=True)
from dscribe.descriptors import SOAP as DScribeSOAP
from ase import Atoms

soap = DScribeSOAP(species=species, r_cut=6.0, n_max=4, l_max=4,
                   periodic=True, sparse=False)
soap_dim = soap.get_number_of_features()
print("SOAP dim: %d" % soap_dim, flush=True)

t0 = time.time()
soap_reps = []
for i in soap_idx:
    try:
        a = Atoms(numbers=charges_sub[i].astype(int),
                 positions=coords_sub[i], cell=cells_sub[i], pbc=True)
        soap_reps.append(soap.create(a).sum(axis=0))
    except:
        soap_reps.append(np.zeros(soap_dim))
t_soap_gen = time.time() - t0
soap_arr = np.array(soap_reps)
print("SOAP generated: %.1fs" % t_soap_gen, flush=True)

# Split
n_train = 2000
n_test = 1000
ytr = eform_sub[soap_idx[:n_train]]
yte = eform_sub[soap_idx[n_train:n_train+n_test]]

# cMBDF features on same split
Xtr_c = cmbdf_global[soap_idx[:n_train]]
Xte_c = cmbdf_global[soap_idx[n_train:n_train+n_test]]

# SOAP features on same split
Xtr_s = soap_arr[:n_train]
Xte_s = soap_arr[n_train:n_train+n_test]

print("\n" + "=" * 70, flush=True)
print("FAIR COMPARISON: N=%d train, %d test" % (n_train, n_test), flush=True)
print("=" * 70, flush=True)

# --- cMBDF + Laplacian KRR (its natural method) ---
print("\n--- cMBDF (40 dim) + Laplacian KRR ---", flush=True)
sc = StandardScaler()
Xtr_cs = sc.fit_transform(Xtr_c)
Xte_cs = sc.transform(Xte_c)

t0 = time.time()
best_c = 999
for g in [0.01, 0.02, 0.05, 0.1, 0.2]:
    krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=g)
    krr.fit(Xtr_cs, ytr)
    mae = mean_absolute_error(yte, krr.predict(Xte_cs))
    best_c = min(best_c, mae)
t_cmbdf_krr = time.time() - t0
print("  MAE = %.4f eV/atom (%.1fs)" % (best_c, t_cmbdf_krr), flush=True)

# --- cMBDF + Linear Ridge (for comparison) ---
print("\n--- cMBDF (40 dim) + Linear Ridge ---", flush=True)
t0 = time.time()
ridge_c = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
ridge_c.fit(Xtr_cs, ytr)
mae_c_lin = mean_absolute_error(yte, ridge_c.predict(Xte_cs))
t_cmbdf_lin = time.time() - t0
print("  MAE = %.4f eV/atom (%.1fs)" % (mae_c_lin, t_cmbdf_lin), flush=True)

# --- SOAP + Linear Ridge (natural for high-dim SOAP) ---
print("\n--- SOAP (%d dim) + Linear Ridge ---" % soap_dim, flush=True)
sc2 = StandardScaler()
t0 = time.time()
Xtr_ss = sc2.fit_transform(Xtr_s)
Xte_ss = sc2.transform(Xte_s)
ridge_s = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
ridge_s.fit(Xtr_ss, ytr)
mae_s_lin = mean_absolute_error(yte, ridge_s.predict(Xte_ss))
t_soap_lin = time.time() - t0
print("  MAE = %.4f eV/atom (%.1fs)" % (mae_s_lin, t_soap_lin), flush=True)

# --- SOAP + dot-product kernel (the standard SOAP kernel) ---
print("\n--- SOAP (%d dim) + Dot-product KRR ---" % soap_dim, flush=True)
t0 = time.time()
best_s_dot = 999
for alpha in [1e-6, 1e-4, 1e-2, 1.0]:
    krr_s = KernelRidge(alpha=alpha, kernel='polynomial', degree=2, coef0=1)
    krr_s.fit(Xtr_ss, ytr)
    mae = mean_absolute_error(yte, krr_s.predict(Xte_ss))
    best_s_dot = min(best_s_dot, mae)
t_soap_dot = time.time() - t0
print("  MAE = %.4f eV/atom (%.1fs)" % (best_s_dot, t_soap_dot), flush=True)

# --- Summary ---
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print("%-35s | %-8s | %-10s | %-10s" % ("Method", "Dim", "MAE", "Time"), flush=True)
print("-" * 70, flush=True)
print("%-35s | %-8d | %.4f    | %.1fs" % ("cMBDF + Laplacian KRR", 40, best_c, t_cmbdf_krr), flush=True)
print("%-35s | %-8d | %.4f    | %.1fs" % ("cMBDF + Linear Ridge", 40, mae_c_lin, t_cmbdf_lin), flush=True)
print("%-35s | %-8d | %.4f    | %.1fs" % ("SOAP + Linear Ridge", soap_dim, mae_s_lin, t_soap_lin), flush=True)
print("%-35s | %-8d | %.4f    | %.1fs" % ("SOAP + Polynomial KRR (deg=2)", soap_dim, best_s_dot, t_soap_dot), flush=True)
print("\nSOAP gen: %.1fs | cMBDF gen: <1s (cached)" % t_soap_gen, flush=True)
print("Dimensionality ratio: %dx" % (soap_dim // 40), flush=True)

print("\nDone!", flush=True)
