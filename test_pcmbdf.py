"""
Test unified p-cMBDF API: Numba vs Torch backend, all body orders.
"""
import numpy as np
import sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from pcmbdf import generate_pcmbdf

# Test structure
a_cu = 3.615
cu_q = np.array([29.0]*4)
cu_r = np.array([[0,0,0],[0,a_cu/2,a_cu/2],[a_cu/2,0,a_cu/2],[a_cu/2,a_cu/2,0]])
cu_c = np.diag([a_cu]*3)

a_nacl = 5.64
nacl_q = np.array([11.0, 17.0, 11.0, 17.0, 11.0, 17.0, 17.0, 11.0])
nacl_r = np.array([[0,0,0],[a_nacl/2,0,0],[a_nacl/2,a_nacl/2,0],[0,a_nacl/2,0],
                   [0,0,a_nacl/2],[a_nacl/2,0,a_nacl/2],[0,a_nacl/2,a_nacl/2],[a_nacl/2,a_nacl/2,a_nacl/2]])
nacl_c = np.diag([a_nacl]*3)

pbc = (True, True, True)

print("=" * 60, flush=True)
print("Unified p-cMBDF API Test", flush=True)
print("=" * 60, flush=True)

for max_body in [3, 4, 5]:
    print("\n--- max_body=%d ---" % max_body, flush=True)

    # Numba backend
    t0 = time.time()
    reps_numba, _ = generate_pcmbdf(
        [cu_q], [cu_r], [cu_c], max_body=max_body, backend='numba',
        normalize=False, n_jobs=1)
    t_numba = time.time() - t0
    print("Numba: shape=%s, time=%.2fs" % (str(reps_numba.shape), t_numba), flush=True)

    # Symmetry
    max_diff = max(np.max(np.abs(reps_numba[0, i] - reps_numba[0, 0])) for i in range(1, 4))
    print("  Cu symmetry: %.2e %s" % (max_diff, "PASS" if max_diff < 1e-6 else "FAIL"), flush=True)

    # Torch backend
    t0 = time.time()
    reps_torch, _ = generate_pcmbdf(
        [cu_q], [cu_r], [cu_c], max_body=max_body, backend='torch',
        normalize=False, device='cpu')
    t_torch = time.time() - t0
    print("Torch: shape=%s, time=%.2fs" % (str(reps_torch.shape), t_torch), flush=True)

    # Symmetry
    max_diff_t = max(np.max(np.abs(reps_torch[0, i] - reps_torch[0, 0])) for i in range(1, 4))
    print("  Cu symmetry: %.2e %s" % (max_diff_t, "PASS" if max_diff_t < 1e-6 else "FAIL"), flush=True)

    # Agreement between backends (within interpolation tolerance)
    agreement = np.max(np.abs(reps_numba - reps_torch))
    n_feat = reps_numba.shape[-1]
    print("  Numba vs Torch max diff: %.2e (n_feat=%d)" % (agreement, n_feat), flush=True)

    # NaCl test
    reps_nacl, _ = generate_pcmbdf(
        [nacl_q], [nacl_r], [nacl_c], max_body=max_body, backend='numba',
        normalize=False, n_jobs=1)
    na_idx = np.where(nacl_q == 11.0)[0]
    cl_idx = np.where(nacl_q == 17.0)[0]
    na_diff = max(np.max(np.abs(reps_nacl[0, i] - reps_nacl[0, na_idx[0]])) for i in na_idx[1:])
    na_cl = np.max(np.abs(reps_nacl[0, na_idx[0]] - reps_nacl[0, cl_idx[0]]))
    print("  NaCl Na-Na: %.2e, Na-Cl: %.4f %s" % (
        na_diff, na_cl, "PASS" if na_diff < 1e-6 and na_cl > 0.01 else "FAIL"), flush=True)

# Test with normalization
print("\n--- Normalization test ---", flush=True)
reps_norm, nf = generate_pcmbdf(
    [cu_q, nacl_q], [cu_r, nacl_r], [cu_c, nacl_c],
    max_body=4, backend='numba', normalize=True, n_jobs=1)
print("Normalized: shape=%s, norm_factors keys: %s" % (
    str(reps_norm.shape), sorted(nf.keys())), flush=True)
print("PASS" if nf is not None and len(nf) > 0 else "FAIL", flush=True)

print("\nAll tests done!", flush=True)
