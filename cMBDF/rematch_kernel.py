"""
REMatch (Regularized Entropy Match) kernel for local atomic representations.

Instead of summing atomic features into a global vector and using a standard
kernel, REMatch compares structures by finding the optimal soft assignment
between their atoms, respecting element types.

This is the kernel used in SOAP-based ML (De et al., PCCP 2016) and should
significantly improve over the global sum kernel for periodic cMBDF.

K_RE(A, B) = Tr(P^T * C) where:
  C[i,j] = k(a_i, b_j) is the local kernel between atoms
  P is the optimal transport matrix (regularized via entropy)
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def local_kernel_matrix(reps_a, charges_a, reps_b, charges_b, sigma, metric='laplacian'):
    """
    Compute local kernel between atoms of two structures.

    Only compares atoms of the same element (delta kernel on species).

    Args:
        reps_a: (n_a, d) atomic features for structure A
        charges_a: (n_a,) charges
        reps_b: (n_b, d) atomic features for structure B
        charges_b: (n_b,) charges
        sigma: kernel width

    Returns:
        C: (n_a, n_b) kernel matrix
    """
    n_a, n_b = len(reps_a), len(reps_b)
    C = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            if charges_a[i] == charges_b[j]:
                if metric == 'laplacian':
                    dist = np.sum(np.abs(reps_a[i] - reps_b[j]))
                else:
                    dist = np.sqrt(np.sum((reps_a[i] - reps_b[j])**2))
                C[i, j] = np.exp(-dist / sigma)

    return C


def rematch_kernel(C, gamma=0.01, max_iter=100, tol=1e-6):
    """
    Compute the REMatch kernel value from a local kernel matrix.

    Uses the Sinkhorn algorithm to find the entropy-regularized optimal
    transport between two sets of atoms.

    Args:
        C: (n_a, n_b) local kernel matrix
        gamma: regularization (higher = more entropic, closer to average)
        max_iter: max Sinkhorn iterations
        tol: convergence tolerance

    Returns:
        scalar kernel value K_RE
    """
    n_a, n_b = C.shape
    if n_a == 0 or n_b == 0:
        return 0.0

    # Cost matrix (negative log of kernel)
    # Add small epsilon to avoid log(0)
    K = C + 1e-16

    # Sinkhorn algorithm
    u = np.ones(n_a) / n_a
    v = np.ones(n_b) / n_b

    K_reg = K ** (1.0 / gamma)

    for _ in range(max_iter):
        u_new = 1.0 / (n_a * K_reg @ v)
        v_new = 1.0 / (n_b * K_reg.T @ u_new)

        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            break
        u = u_new
        v = v_new

    # Transport plan
    P = np.diag(u) @ K_reg @ np.diag(v)

    # REMatch kernel value
    return np.sum(P * C)


def compute_rematch_kernel_matrix(reps_list, charges_list, sigma=1.0,
                                    gamma=0.01, metric='laplacian'):
    """
    Compute the full REMatch kernel matrix for a set of structures.

    Args:
        reps_list: list of (n_atoms_i, d) arrays — per-atom features
        charges_list: list of (n_atoms_i,) arrays — nuclear charges
        sigma: local kernel width
        gamma: REMatch regularization

    Returns:
        K: (N, N) kernel matrix
    """
    N = len(reps_list)
    K = np.zeros((N, N))

    for i in range(N):
        K[i, i] = rematch_kernel(
            local_kernel_matrix(reps_list[i], charges_list[i],
                                reps_list[i], charges_list[i], sigma, metric),
            gamma)
        for j in range(i + 1, N):
            C = local_kernel_matrix(reps_list[i], charges_list[i],
                                    reps_list[j], charges_list[j], sigma, metric)
            K[i, j] = rematch_kernel(C, gamma)
            K[j, i] = K[i, j]

    return K


def compute_rematch_kernel_rect(reps_train, charges_train,
                                 reps_test, charges_test,
                                 sigma=1.0, gamma=0.01, metric='laplacian'):
    """
    Compute rectangular REMatch kernel matrix (test vs train).
    """
    n_train = len(reps_train)
    n_test = len(reps_test)
    K = np.zeros((n_test, n_train))

    for i in range(n_test):
        for j in range(n_train):
            C = local_kernel_matrix(reps_test[i], charges_test[i],
                                    reps_train[j], charges_train[j], sigma, metric)
            K[i, j] = rematch_kernel(C, gamma)

    return K


def run_rematch_benchmark(reps, charges_list, eform, n_train=1000, n_test=500,
                           sigma=1.0, gamma=0.01, alpha=1e-6):
    """
    Train KRR with REMatch kernel and return MAE.
    """
    np.random.seed(42)
    perm = np.random.permutation(len(reps))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]

    # Extract per-atom reps (remove zero-padding)
    def get_local(idx):
        local_reps = []
        local_charges = []
        for i in idx:
            n_at = len(charges_list[i])
            local_reps.append(reps[i, :n_at, :])
            local_charges.append(charges_list[i])
        return local_reps, local_charges

    reps_tr, q_tr = get_local(train_idx)
    reps_te, q_te = get_local(test_idx)

    print("  Computing train kernel (%d x %d)..." % (n_train, n_train))
    t0 = time.time()
    K_train = compute_rematch_kernel_matrix(reps_tr, q_tr, sigma, gamma)
    t_ktrain = time.time() - t0
    print("  Train kernel: %.1fs" % t_ktrain)

    print("  Computing test kernel (%d x %d)..." % (n_test, n_train))
    t0 = time.time()
    K_test = compute_rematch_kernel_rect(reps_tr, q_tr, reps_te, q_te, sigma, gamma)
    t_ktest = time.time() - t0
    print("  Test kernel: %.1fs" % t_ktest)

    # KRR with precomputed kernel
    K_train += alpha * np.eye(n_train)
    coeffs = np.linalg.solve(K_train, eform[train_idx])
    pred = K_test @ coeffs

    mae = mean_absolute_error(eform[test_idx], pred)
    return mae


if __name__ == '__main__':
    import os
    import time
    import sys
    sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')

    DATA_DIR = '/home/albd/projects/cmbdf/data'

    print("=" * 70)
    print("REMatch Local Kernel Benchmark")
    print("=" * 70)

    # Load reps
    rep_data = np.load(os.path.join(DATA_DIR, 'pcmbdf_mp_55k_elemspec.npz'))
    reps = rep_data['reps'][:5000]

    data = np.load(os.path.join(DATA_DIR, 'mp_eform_parsed.npz'), allow_pickle=True)
    all_charges = data['charges']
    all_eform = data['eform']
    all_natoms = data['n_atoms']

    mask = all_natoms <= 30
    valid = np.where(mask)[0]
    np.random.seed(42)
    subset = np.random.choice(valid, min(55000, len(valid)), replace=False)
    subset.sort()

    charges_sub = all_charges[subset[:5000]]
    eform_sub = all_eform[subset[:5000]]

    # Normalize
    from cMBDF_periodic import normalize_per_element
    reps_norm, _ = normalize_per_element(reps, charges_sub, mode='mean')

    # REMatch is expensive — start with small N
    # Compare: global kernel vs REMatch at same training size
    print("\n--- Global sum kernel baseline ---")
    from sklearn.preprocessing import StandardScaler

    def build_global(reps, charges_list):
        n = len(reps)
        out = np.zeros((n, reps.shape[-1]))
        for i in range(n):
            out[i] = reps[i, :len(charges_list[i]), :].sum(axis=0)
        return out

    global_reps = build_global(reps_norm, charges_sub)

    for n_tr in [500, 1000]:
        np.random.seed(42)
        perm = np.random.permutation(5000)
        tr, te = perm[:n_tr], perm[n_tr:n_tr+500]
        sc = StandardScaler()
        Xtr = sc.fit_transform(global_reps[tr])
        Xte = sc.transform(global_reps[te])
        best = 999
        for g in [0.01, 0.05, 0.1]:
            krr = KernelRidge(alpha=1e-4, kernel='laplacian', gamma=g)
            krr.fit(Xtr, eform_sub[tr])
            best = min(best, mean_absolute_error(eform_sub[te], krr.predict(Xte)))
        print("  Global kernel N=%d: MAE = %.4f eV/atom" % (n_tr, best))

    print("\n--- REMatch local kernel ---")
    for n_tr in [200, 500]:
        for sigma in [1.0, 5.0, 10.0]:
            mae = run_rematch_benchmark(reps_norm, charges_sub, eform_sub,
                                        n_train=n_tr, n_test=500,
                                        sigma=sigma, gamma=0.1, alpha=1e-6)
            print("  REMatch N=%d, sigma=%.1f: MAE = %.4f eV/atom" % (n_tr, sigma, mae))

    print("\nDone!")
