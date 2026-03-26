"""
cMBDF Feature Analysis on QM9

Performs:
1. Download/parse QM9 dataset
2. Generate cMBDF representations
3. Feature correlation analysis
4. Feature importance via KRR leave-one-feature-out
5. Ablation study: radial vs angular, feature subsets
6. LASSO feature selection
"""
import numpy as np
import os
import sys
import time
import urllib.request
import tarfile

sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF import generate_mbdf, get_convolutions

DATA_DIR = '/home/albd/projects/cmbdf/data'
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# STEP 1: Download and parse QM9
# ============================================================
print("=" * 70)
print("STEP 1: Loading QM9 dataset")
print("=" * 70)

QM9_NPZ = os.path.join(DATA_DIR, 'qm9_parsed.npz')

if os.path.exists(QM9_NPZ):
    print("Loading cached QM9 data...")
    data = np.load(QM9_NPZ, allow_pickle=True)
    all_charges = data['charges']
    all_coords = data['coords']
    all_energies = data['energies']
    print("Loaded %d molecules" % len(all_charges))
else:
    QM9_TAR = os.path.join(DATA_DIR, 'dsgdb9nsd.xyz.tar.bz2')
    QM9_DIR = os.path.join(DATA_DIR, 'qm9_xyz')

    if not os.path.exists(QM9_TAR):
        url = 'https://ndownloader.figshare.com/files/3195389'
        print("Downloading QM9 from figshare...")
        os.system('curl -L -o "%s" "%s"' % (QM9_TAR, url))
        print("Downloaded.")

    if not os.path.isdir(QM9_DIR) or len(os.listdir(QM9_DIR)) == 0:
        print("Extracting...")
        os.makedirs(QM9_DIR, exist_ok=True)
        with tarfile.open(QM9_TAR, 'r:bz2') as tar:
            tar.extractall(QM9_DIR, filter='data')
        print("Extracted %d files." % len(os.listdir(QM9_DIR)))

    # Element symbol to atomic number
    elem2z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    xyz_files = sorted([f for f in os.listdir(QM9_DIR) if f.endswith('.xyz')])
    print("Found %d xyz files" % len(xyz_files))

    all_charges = []
    all_coords = []
    all_energies = []
    n_skipped = 0

    for fname in xyz_files:
        path = os.path.join(QM9_DIR, fname)
        with open(path, 'r') as f:
            lines = f.readlines()

        try:
            n_atoms = int(lines[0].strip())
            # Line 2: tab-separated properties
            # gdb N, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv
            props = lines[1].strip().split('\t')
            energy_U0 = float(props[12])  # Internal energy at 0K in Hartree

            charges = []
            coords = []
            for i in range(2, 2 + n_atoms):
                parts = lines[i].strip().replace('*^', 'e').split('\t')
                elem = parts[0].strip()
                x = float(parts[1].strip())
                y = float(parts[2].strip())
                z = float(parts[3].strip())
                charges.append(elem2z[elem])
                coords.append([x, y, z])

            all_charges.append(np.array(charges, dtype=np.float64))
            all_coords.append(np.array(coords, dtype=np.float64))
            all_energies.append(energy_U0)
        except Exception:
            n_skipped += 1
            continue

    if n_skipped > 0:
        print("Skipped %d problematic files" % n_skipped)

    all_charges = np.array(all_charges, dtype=object)
    all_coords = np.array(all_coords, dtype=object)
    all_energies = np.array(all_energies, dtype=np.float64)

    np.savez(QM9_NPZ, charges=all_charges, coords=all_coords, energies=all_energies)
    print("Parsed and saved %d molecules" % len(all_charges))

N_total = len(all_charges)
print("Total molecules: %d" % N_total)
print("Energy range: %.2f to %.2f Ha" % (all_energies.min(), all_energies.max()))

# Use a 5000-molecule subset for analysis (fast enough for feature analysis)
np.random.seed(42)
N_subset = 5000
idx = np.random.choice(N_total, N_subset, replace=False)
idx.sort()

charges_sub = all_charges[idx]
coords_sub = all_coords[idx]
energies_sub = all_energies[idx]

print("Using subset of %d molecules" % N_subset)

# ============================================================
# STEP 2: Generate cMBDF representations
# ============================================================
print()
print("=" * 70)
print("STEP 2: Generating cMBDF representations")
print("=" * 70)

REP_NPZ = os.path.join(DATA_DIR, 'cmbdf_qm9_5k.npz')

if os.path.exists(REP_NPZ):
    print("Loading cached representations...")
    rep_data = np.load(REP_NPZ)
    reps = rep_data['reps']
    energies_sub = rep_data['energies']
    charges_flat = rep_data['charges_flat']
else:
    t0 = time.time()
    reps = generate_mbdf(charges_sub, coords_sub, rcut=10.0, n_atm=2.0,
                         smooth_cutoff=False, n_jobs=-1, progress_bar=True)
    t1 = time.time()
    print("Generated in %.1f seconds" % (t1 - t0))
    print("Shape: %s" % str(reps.shape))

    # Flatten charges for element-resolved analysis
    charges_flat = np.concatenate(charges_sub)

    np.savez(REP_NPZ, reps=reps, energies=energies_sub, charges_flat=charges_flat)

print("Representation shape:", reps.shape)
n_mol, max_atoms, n_feat = reps.shape
nrs = 20  # radial features
nAs = 20  # angular features

# Feature labels
feat_labels = []
alpha_list = [1.5, 5.0]
n_list = [3.0, 5.0]
for m in range(5):  # derivative order 0-4
    for j, wt in enumerate(['exp(-1.5r)', 'exp(-5.0r)', '1/(r+1)^3', '1/(r+1)^5']):
        feat_labels.append("R:m=%d,%s" % (m, wt))
for m in range(5):
    for n in range(1, 5):
        feat_labels.append("A:m=%d,cos(%dθ)" % (m, n))

print("Feature labels (first 5):", feat_labels[:5])
print("Feature labels (20-24):", feat_labels[20:25])

# ============================================================
# STEP 3: Feature correlation analysis
# ============================================================
print()
print("=" * 70)
print("STEP 3: Feature correlation analysis")
print("=" * 70)

# Collect all non-zero atom representations
all_atom_reps = []
for i in range(n_mol):
    n_atoms_i = int(np.sum(np.any(reps[i] != 0, axis=1)))
    if n_atoms_i > 0:
        all_atom_reps.append(reps[i, :n_atoms_i, :])

all_atom_reps = np.vstack(all_atom_reps)
print("Total atom representations: %d" % len(all_atom_reps))

# Pearson correlation matrix
corr_matrix = np.corrcoef(all_atom_reps.T)  # (40, 40)

print("\nFeature-feature correlation matrix (40x40):")
print("Strongest positive correlations (off-diagonal):")
upper_tri = np.triu_indices(n_feat, k=1)
corr_vals = corr_matrix[upper_tri]
sorted_idx = np.argsort(-np.abs(corr_vals))

for rank in range(10):
    idx_in_tri = sorted_idx[rank]
    i_feat = upper_tri[0][idx_in_tri]
    j_feat = upper_tri[1][idx_in_tri]
    r = corr_vals[idx_in_tri]
    print("  %s <-> %s: r = %.4f" % (feat_labels[i_feat], feat_labels[j_feat], r))

# Count highly correlated pairs
n_high = np.sum(np.abs(corr_vals) > 0.95)
n_moderate = np.sum(np.abs(corr_vals) > 0.80)
print("\nPairs with |r| > 0.95: %d" % n_high)
print("Pairs with |r| > 0.80: %d" % n_moderate)

# Feature variance
feat_var = np.var(all_atom_reps, axis=0)
print("\nFeature variance (sorted ascending):")
var_order = np.argsort(feat_var)
for i in range(5):
    fi = var_order[i]
    print("  %s: var = %.6f" % (feat_labels[fi], feat_var[fi]))
print("  ...")
for i in range(-3, 0):
    fi = var_order[i]
    print("  %s: var = %.4f" % (feat_labels[fi], feat_var[fi]))

# Near-constant features
near_const = np.sum(feat_var < 1e-6)
print("\nNear-constant features (var < 1e-6): %d" % near_const)

# Save correlation matrix
np.save(os.path.join(DATA_DIR, 'corr_matrix.npy'), corr_matrix)

# ============================================================
# STEP 4: Feature importance via KRR
# ============================================================
print()
print("=" * 70)
print("STEP 4: Feature importance via KRR leave-one-feature-out")
print("=" * 70)

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Build global representation: sum over atoms (size-extensive)
global_reps = np.array([reps[i].sum(axis=0) for i in range(n_mol)])
print("Global rep shape:", global_reps.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    global_reps, energies_sub, test_size=0.2, random_state=42)

print("Train: %d, Test: %d" % (len(X_train), len(X_test)))

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Baseline: all features
print("\nTraining KRR with all 40 features...")
krr = KernelRidge(alpha=1e-8, kernel='laplacian', gamma=0.01)
krr.fit(X_train_s, y_train)
y_pred = krr.predict(X_test_s)
mae_baseline = mean_absolute_error(y_test, y_pred) * 627.509  # Ha to kcal/mol
print("Baseline MAE (all 40 features): %.2f kcal/mol" % mae_baseline)

# Leave-one-feature-out
print("\nLeave-one-feature-out analysis:")
feature_importance = {}

for f in range(n_feat):
    feat_mask = np.ones(n_feat, dtype=bool)
    feat_mask[f] = False

    X_tr = X_train_s[:, feat_mask]
    X_te = X_test_s[:, feat_mask]

    krr_f = KernelRidge(alpha=1e-8, kernel='laplacian', gamma=0.01)
    krr_f.fit(X_tr, y_train)
    y_pred_f = krr_f.predict(X_te)
    mae_f = mean_absolute_error(y_test, y_pred_f) * 627.509
    delta = mae_f - mae_baseline
    feature_importance[f] = delta

# Sort by importance (largest MAE increase = most important)
sorted_feats = sorted(feature_importance.items(), key=lambda x: -x[1])

print("\nMost important features (removing causes largest MAE increase):")
for rank, (f, delta) in enumerate(sorted_feats[:10]):
    print("  %2d. %s: +%.3f kcal/mol" % (rank + 1, feat_labels[f], delta))

print("\nLeast important features (removing barely changes MAE):")
for f, delta in sorted_feats[-10:]:
    print("  %s: %+.3f kcal/mol" % (feat_labels[f], delta))

# ============================================================
# STEP 5: LASSO feature selection
# ============================================================
print()
print("=" * 70)
print("STEP 5: LASSO feature selection")
print("=" * 70)

from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train_s, y_train)
y_pred_lasso = lasso.predict(X_test_s)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso) * 627.509
print("LASSO MAE: %.2f kcal/mol" % mae_lasso)
print("LASSO alpha: %.6f" % lasso.alpha_)

coefs = np.abs(lasso.coef_)
lasso_order = np.argsort(-coefs)
print("\nLASSO feature ranking (by |coefficient|):")
for rank in range(n_feat):
    f = lasso_order[rank]
    print("  %2d. %s: |coef| = %.6f%s" % (
        rank + 1, feat_labels[f], coefs[f],
        " (ZERO)" if coefs[f] == 0 else ""))

n_nonzero = np.sum(coefs > 0)
n_zero = np.sum(coefs == 0)
print("\nNon-zero LASSO coefficients: %d / %d" % (n_nonzero, n_feat))
print("Zero (eliminated) features: %d" % n_zero)

# ============================================================
# STEP 6: Ablation study
# ============================================================
print()
print("=" * 70)
print("STEP 6: Ablation study — feature subsets")
print("=" * 70)

def train_eval_krr(X_tr, X_te, y_tr, y_te, label):
    """Train KRR and return MAE in kcal/mol."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    krr = KernelRidge(alpha=1e-8, kernel='laplacian', gamma=0.01)
    krr.fit(Xtr, y_tr)
    pred = krr.predict(Xte)
    mae = mean_absolute_error(y_te, pred) * 627.509
    return mae

print("\n--- Body-order ablation ---")

# Radial only (features 0-19)
mae_rad = train_eval_krr(X_train[:, :nrs], X_test[:, :nrs], y_train, y_test, "radial")
print("Radial only (20 feat):  %.2f kcal/mol" % mae_rad)

# Angular only (features 20-39)
mae_ang = train_eval_krr(X_train[:, nrs:], X_test[:, nrs:], y_train, y_test, "angular")
print("Angular only (20 feat): %.2f kcal/mol" % mae_ang)

# All
print("All features (40 feat): %.2f kcal/mol" % mae_baseline)

print("\n--- Derivative order ablation (radial) ---")
for m in range(5):
    # Each derivative order m has 4 weighting functions
    feat_idx = list(range(m * 4, (m + 1) * 4))
    mae_m = train_eval_krr(X_train[:, feat_idx], X_test[:, feat_idx],
                            y_train, y_test, "radial m=%d" % m)
    print("Radial m=%d (4 feat): %.2f kcal/mol" % (m, mae_m))

print("\n--- Derivative order ablation (angular) ---")
for m in range(5):
    feat_idx = list(range(nrs + m * 4, nrs + (m + 1) * 4))
    mae_m = train_eval_krr(X_train[:, feat_idx], X_test[:, feat_idx],
                            y_train, y_test, "angular m=%d" % m)
    print("Angular m=%d (4 feat): %.2f kcal/mol" % (m, mae_m))

print("\n--- Weighting function ablation (radial) ---")
wt_names = ['exp(-1.5r)', 'exp(-5.0r)', '1/(r+1)^3', '1/(r+1)^5']
for w in range(4):
    # Feature w, w+4, w+8, w+12, w+16 (one per derivative order)
    feat_idx = [w + m * 4 for m in range(5)]
    mae_w = train_eval_krr(X_train[:, feat_idx], X_test[:, feat_idx],
                            y_train, y_test, wt_names[w])
    print("Radial %s (5 feat): %.2f kcal/mol" % (wt_names[w], mae_w))

print("\n--- Weighting function ablation (angular) ---")
for w in range(4):
    feat_idx = [nrs + w + m * 4 for m in range(5)]
    mae_w = train_eval_krr(X_train[:, feat_idx], X_test[:, feat_idx],
                            y_train, y_test, "cos(%dθ)" % (w + 1))
    print("Angular cos(%dθ) (5 feat): %.2f kcal/mol" % (w + 1, mae_w))

print("\n--- Cumulative derivative order (adding higher orders) ---")
for max_m in range(5):
    feat_idx = list(range((max_m + 1) * 4))  # radial 0..max_m
    feat_idx += list(range(nrs, nrs + (max_m + 1) * 4))  # angular 0..max_m
    n_f = len(feat_idx)
    mae_cum = train_eval_krr(X_train[:, feat_idx], X_test[:, feat_idx],
                              y_train, y_test, "m<=<%d" % max_m)
    print("m <= %d (%d feat): %.2f kcal/mol" % (max_m, n_f, mae_cum))

print("\n--- Minimal feature sets ---")
# Top N features by importance
for n_top in [5, 10, 15, 20, 25, 30, 35]:
    top_feats = [f for f, _ in sorted_feats[:n_top]]
    mae_top = train_eval_krr(X_train[:, top_feats], X_test[:, top_feats],
                              y_train, y_test, "top %d" % n_top)
    pct = (1 - mae_top / mae_baseline) * 100 if mae_top > mae_baseline else 100
    print("Top %2d features: %.2f kcal/mol (%.1f%% of baseline accuracy)" % (
        n_top, mae_top, 100 - abs(mae_top - mae_baseline) / mae_baseline * 100))

# ============================================================
# STEP 7: Summary
# ============================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

# Group importance by body order
rad_importance = np.array([feature_importance[f] for f in range(nrs)])
ang_importance = np.array([feature_importance[f] for f in range(nrs, n_feat)])

print("\nAverage importance by body order:")
print("  2-body (radial):  mean delta = %.4f kcal/mol" % np.mean(rad_importance))
print("  3-body (angular): mean delta = %.4f kcal/mol" % np.mean(ang_importance))

# Group by derivative order
print("\nAverage importance by derivative order m:")
for m in range(5):
    rad_m = np.mean([feature_importance[m * 4 + w] for w in range(4)])
    ang_m = np.mean([feature_importance[nrs + m * 4 + w] for w in range(4)])
    print("  m=%d: radial=%.4f, angular=%.4f" % (m, rad_m, ang_m))

# Pruning recommendations
print("\nPRUNING RECOMMENDATIONS:")
removable = [f for f, delta in sorted_feats if delta < 0.05]
print("Features removable with <0.05 kcal/mol impact: %d / %d" % (len(removable), n_feat))
for f in removable:
    print("  - %s (delta=%.4f)" % (feat_labels[f], feature_importance[f]))

print("\nDone!")
