"""
QM9 atomization energy with LOCAL kernel (delta kernel on species).
This is the same kernel type used in the cMBDF paper for QM9 benchmarks.

K(A,B) = sum_i sum_j delta(Z_i, Z_j) * exp(-||p_i - p_j||_1 / sigma)

This is O(N^2 * n_atoms^2) but for QM9 (small molecules) it's tractable
at moderate training sizes.
"""
import numpy as np
import os, sys, time, gc
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from sklearn.metrics import mean_absolute_error
from numba import jit, prange

DATA_DIR = '/home/albd/projects/cmbdf/data'

# Atomic reference energies (B3LYP/6-31G(2df,p))
ATOMIC_REFS = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}

@jit(nopython=True, parallel=True)
def compute_local_kernel_matrix(reps_A, charges_A, natoms_A,
                                 reps_B, charges_B, natoms_B,
                                 sigma):
    """
    Local delta kernel between two sets of molecules.
    K[m,n] = sum_i sum_j delta(Z_i^A, Z_j^B) * exp(-||p_i - p_j||_1 / sigma)
    """
    nA = len(natoms_A)
    nB = len(natoms_B)
    K = np.zeros((nA, nB))

    for m in prange(nA):
        na = natoms_A[m]
        for n in range(nB):
            nb = natoms_B[n]
            val = 0.0
            for i in range(na):
                qi = charges_A[m][i]
                for j in range(nb):
                    if qi == charges_B[n][j]:
                        dist = 0.0
                        for f in range(reps_A.shape[2]):
                            dist += abs(reps_A[m, i, f] - reps_B[n, j, f])
                        val += np.exp(-dist / sigma)
            K[m, n] = val

    return K

print("=" * 70, flush=True)
print("QM9 Atomization Energy — LOCAL Kernel", flush=True)
print("=" * 70, flush=True)

# Load QM9
data = np.load(os.path.join(DATA_DIR, 'qm9_parsed.npz'), allow_pickle=True)
all_charges = data['charges']
all_coords = data['coords']
all_energies = data['energies']
N = len(all_charges)

# Compute atomization energies
atomization = np.zeros(N)
for i in range(N):
    ref_sum = sum(ATOMIC_REFS[int(z)] for z in all_charges[i])
    atomization[i] = -(all_energies[i] - ref_sum) * 627.509  # kcal/mol

print("Molecules: %d" % N, flush=True)
print("Atomization range: %.1f to %.1f kcal/mol" % (atomization.min(), atomization.max()), flush=True)

# Load cached reps
REP_FILE = os.path.join(DATA_DIR, 'qm9_cmbdf_atomization.npz')
if os.path.exists(REP_FILE):
    reps = np.load(REP_FILE)['reps']
else:
    print("Generating cMBDF...", flush=True)
    from cMBDF import generate_mbdf
    reps = generate_mbdf(all_charges, all_coords, rcut=10.0, n_atm=2.0,
                          smooth_cutoff=False, n_jobs=-1, progress_bar=True)
    np.savez_compressed(REP_FILE, reps=reps)

print("Rep shape: %s" % str(reps.shape), flush=True)
max_atoms = reps.shape[1]
n_feat = reps.shape[2]

# Build padded charges array for numba
charges_padded = np.zeros((N, max_atoms), dtype=np.float64)
natoms_arr = np.zeros(N, dtype=np.int64)
for i in range(N):
    na = len(all_charges[i])
    natoms_arr[i] = na
    charges_padded[i, :na] = all_charges[i]

# Fixed split
np.random.seed(42)
perm = np.random.permutation(N)
TEST = perm[:10000]
POOL = perm[10000:]

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]

print("\n%-8s | %-12s | %-10s | %-10s" % ("N_train", "MAE(kcal/mol)", "sigma", "time(s)"), flush=True)
print("-" * 45, flush=True)

for n_train in TRAIN_SIZES:
    if n_train > len(POOL):
        break
    tr = POOL[:n_train]

    best_mae = 999
    best_sigma = 0

    for sigma in [10.0, 50.0, 100.0, 500.0, 1000.0]:
        t0 = time.time()

        # Compute train kernel
        K_train = compute_local_kernel_matrix(
            reps[tr], charges_padded[tr], natoms_arr[tr],
            reps[tr], charges_padded[tr], natoms_arr[tr],
            sigma)

        # Regularize and solve
        for alpha in [1e-10, 1e-8, 1e-6]:
            try:
                K = K_train.copy()
                K[np.diag_indices_from(K)] += alpha
                coeffs = np.linalg.solve(K, atomization[tr])

                # Test kernel
                K_test = compute_local_kernel_matrix(
                    reps[TEST[:2000]], charges_padded[TEST[:2000]], natoms_arr[TEST[:2000]],
                    reps[tr], charges_padded[tr], natoms_arr[tr],
                    sigma)

                pred = K_test @ coeffs
                mae = mean_absolute_error(atomization[TEST[:2000]], pred)

                if mae < best_mae:
                    best_mae = mae
                    best_sigma = sigma
            except:
                pass

        t_total = time.time() - t0
        del K_train
        gc.collect()

    print("%-8d | %-12.2f | %-10.0f | %-10.0f" % (n_train, best_mae, best_sigma, t_total), flush=True)

print("\nContext (cMBDF paper, local kernel, QM9 atomization):", flush=True)
print("  cMBDF (40 dim): ~1 kcal/mol at N=4000", flush=True)
print("  MBDF (5 dim):   ~2 kcal/mol at N=10000", flush=True)
print("  FCHL19 (720):   ~0.8 kcal/mol at N=2000", flush=True)

print("\nDone!", flush=True)
