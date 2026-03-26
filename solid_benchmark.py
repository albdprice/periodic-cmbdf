"""
Solid-state ML benchmark: periodic cMBDF on formation energies.

Uses matbench_mp_e_form dataset (Materials Project formation energies).
Compares universal vs element-specific cMBDF with KRR.
"""
import numpy as np
import os
import sys
import time

sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')

DATA_DIR = '/home/albd/projects/cmbdf/data'
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Step 1: Load dataset
# ============================================================
print("=" * 70)
print("STEP 1: Loading Materials Project formation energy data")
print("=" * 70)

MP_NPZ = os.path.join(DATA_DIR, 'mp_eform_parsed.npz')

if os.path.exists(MP_NPZ):
    print("Loading cached data...")
    data = np.load(MP_NPZ, allow_pickle=True)
    all_charges = data['charges']
    all_coords = data['coords']
    all_cells = data['cells']
    all_eform = data['eform']
    all_n_atoms = data['n_atoms']
    print("Loaded %d structures" % len(all_charges))
else:
    print("Fetching matbench_mp_e_form via matminer...")
    from matminer.datasets import load_dataset
    df = load_dataset("matbench_mp_e_form")
    print("Dataset size: %d" % len(df))

    all_charges = []
    all_coords = []
    all_cells = []
    all_eform = []
    all_n_atoms = []
    n_skipped = 0

    for idx, row in df.iterrows():
        struct = row['structure']
        eform = row['e_form']

        try:
            charges = np.array([s.Z for s in struct.species], dtype=np.float64)
            coords = np.array(struct.cart_coords, dtype=np.float64)
            cell = np.array(struct.lattice.matrix, dtype=np.float64)

            all_charges.append(charges)
            all_coords.append(coords)
            all_cells.append(cell)
            all_eform.append(eform)
            all_n_atoms.append(len(charges))
        except Exception:
            n_skipped += 1

    all_charges = np.array(all_charges, dtype=object)
    all_coords = np.array(all_coords, dtype=object)
    all_cells = np.array(all_cells, dtype=object)
    all_eform = np.array(all_eform, dtype=np.float64)
    all_n_atoms = np.array(all_n_atoms, dtype=np.int64)

    np.savez(MP_NPZ, charges=all_charges, coords=all_coords,
             cells=all_cells, eform=all_eform, n_atoms=all_n_atoms)
    print("Parsed %d structures (skipped %d)" % (len(all_charges), n_skipped))

N = len(all_charges)
print("Total: %d structures" % N)
print("Formation energy range: %.3f to %.3f eV/atom" % (all_eform.min(), all_eform.max()))
print("Atoms per structure: %d to %d (mean %.1f)" % (
    all_n_atoms.min(), all_n_atoms.max(), all_n_atoms.mean()))

# Filter to manageable sizes for this benchmark
size_mask = all_n_atoms <= 40  # keep structures with <= 40 atoms
print("Structures with <= 40 atoms: %d" % np.sum(size_mask))

# Use a 2000-structure subset for speed
np.random.seed(42)
valid_idx = np.where(size_mask)[0]
subset_idx = np.random.choice(valid_idx, min(2000, len(valid_idx)), replace=False)
subset_idx.sort()

charges_sub = all_charges[subset_idx]
coords_sub = all_coords[subset_idx]
cells_sub = all_cells[subset_idx]
eform_sub = all_eform[subset_idx]
n_atoms_sub = all_n_atoms[subset_idx]

print("Using subset: %d structures" % len(subset_idx))
print("Unique elements: %s" % sorted(set(int(z) for q in charges_sub for z in q)))

# ============================================================
# Step 2: Generate cMBDF representations
# ============================================================
print()
print("=" * 70)
print("STEP 2: Generating periodic cMBDF representations")
print("=" * 70)

from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element

# Universal basis
print("\n--- Universal basis ---")
t0 = time.time()
reps_univ = generate_mbdf_periodic(
    list(charges_sub), list(coords_sub), list(cells_sub),
    pbc=(True, True, True), rcut=6.0, n_atm=2.0,
    n_jobs=-1, progress_bar=True, elem_specific=False)
t_univ = time.time() - t0
print("Shape: %s, Time: %.1fs" % (str(reps_univ.shape), t_univ))

# Element-specific basis
print("\n--- Element-specific basis ---")
t0 = time.time()
reps_elem = generate_mbdf_periodic(
    list(charges_sub), list(coords_sub), list(cells_sub),
    pbc=(True, True, True), rcut=6.0, n_atm=2.0,
    n_jobs=-1, progress_bar=True, elem_specific=True)
t_elem = time.time() - t0
print("Shape: %s, Time: %.1fs" % (str(reps_elem.shape), t_elem))

# ============================================================
# Step 3: Build global representations and train KRR
# ============================================================
print()
print("=" * 70)
print("STEP 3: KRR learning curves on formation energies")
print("=" * 70)

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def build_global_rep(reps, charges_list):
    """Sum over atoms → global (per-structure) representation."""
    n_struct = len(reps)
    n_feat = reps.shape[-1]
    global_rep = np.zeros((n_struct, n_feat))
    for i in range(n_struct):
        n_at = len(charges_list[i])
        global_rep[i] = reps[i, :n_at, :].sum(axis=0)
    return global_rep

def run_learning_curve(X, y, train_sizes, label, n_trials=3):
    """Train KRR at various training sizes and return MAEs."""
    results = {}
    n_total = len(X)

    for n_train in train_sizes:
        if n_train >= n_total - 200:
            continue

        maes = []
        for trial in range(n_trials):
            np.random.seed(trial)
            perm = np.random.permutation(n_total)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:n_train + 200]

            sc = StandardScaler()
            X_tr = sc.fit_transform(X[train_idx])
            X_te = sc.transform(X[test_idx])

            krr = KernelRidge(alpha=1e-6, kernel='laplacian', gamma=0.01)
            krr.fit(X_tr, y[train_idx])
            pred = krr.predict(X_te)
            mae = mean_absolute_error(y[test_idx], pred)
            maes.append(mae)

        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        results[n_train] = (mean_mae, std_mae)
        print("  %s N=%4d: MAE = %.4f +/- %.4f eV/atom" % (label, n_train, mean_mae, std_mae))

    return results

# Build global reps
global_univ = build_global_rep(reps_univ, charges_sub)
global_elem = build_global_rep(reps_elem, charges_sub)

# Per-atom normalization
reps_univ_norm, norm_factors_u = normalize_per_element(reps_univ, charges_sub, mode='mean')
reps_elem_norm, norm_factors_e = normalize_per_element(reps_elem, charges_sub, mode='mean')

global_univ_norm = build_global_rep(reps_univ_norm, charges_sub)
global_elem_norm = build_global_rep(reps_elem_norm, charges_sub)

train_sizes = [100, 200, 400, 800, 1200, 1600]

print("\n--- Universal basis (raw) ---")
res_u = run_learning_curve(global_univ, eform_sub, train_sizes, "Univ-raw")

print("\n--- Universal basis (per-element normalized) ---")
res_un = run_learning_curve(global_univ_norm, eform_sub, train_sizes, "Univ-norm")

print("\n--- Element-specific basis (raw) ---")
res_e = run_learning_curve(global_elem, eform_sub, train_sizes, "Elem-raw")

print("\n--- Element-specific basis (per-element normalized) ---")
res_en = run_learning_curve(global_elem_norm, eform_sub, train_sizes, "Elem-norm")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print("SUMMARY — Formation energy prediction (eV/atom)")
print("=" * 70)

print("\n%-20s" % "N_train", end="")
for n in sorted(res_u.keys()):
    print(" | N=%-5d" % n, end="")
print()
print("-" * (22 + 10 * len(res_u)))

for label, res in [("Universal", res_u), ("Univ+norm", res_un),
                    ("Elem-specific", res_e), ("Elem+norm", res_en)]:
    print("%-20s" % label, end="")
    for n in sorted(res.keys()):
        print(" | %.4f" % res[n][0], end="")
    print()

print("\nBenchmark complete!")
print("Representation generation: universal %.1fs, elem-specific %.1fs" % (t_univ, t_elem))
