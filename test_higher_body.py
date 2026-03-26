"""
Test 4-body and 5-body extensions.
Validate: correct shapes, symmetry, non-zero values, no NaN/Inf.
"""
import numpy as np
import sys, time
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
from cMBDF_higher_body import generate_mbdf_periodic_higher

# Test structures
a_cu = 3.615
cu_q = np.array([29.0]*4)
cu_r = np.array([[0,0,0],[0,a_cu/2,a_cu/2],[a_cu/2,0,a_cu/2],[a_cu/2,a_cu/2,0]])
cu_c = np.diag([a_cu]*3)

a_si = 5.43
si_q = np.array([14.0]*8)
si_r = np.array([[0,0,0],[a_si/4,a_si/4,a_si/4],
                 [a_si/2,a_si/2,0],[3*a_si/4,3*a_si/4,a_si/4],
                 [a_si/2,0,a_si/2],[3*a_si/4,a_si/4,3*a_si/4],
                 [0,a_si/2,a_si/2],[a_si/4,3*a_si/4,3*a_si/4]])
si_c = np.diag([a_si]*3)

pbc = (True, True, True)

for max_body in [3, 4, 5]:
    print("=" * 60, flush=True)
    print("max_body = %d" % max_body, flush=True)
    print("=" * 60, flush=True)

    # FCC Cu
    t0 = time.time()
    rep_cu = generate_mbdf_periodic_higher(
        [cu_q], [cu_r], [cu_c], pbc=pbc, rcut=6.0, max_body=max_body, n_jobs=1)[0]
    t = time.time() - t0
    print("FCC Cu: shape=%s, time=%.2fs" % (str(rep_cu.shape), t), flush=True)
    print("  NaN: %s, Inf: %s, nonzero: %s" % (
        np.any(np.isnan(rep_cu)), np.any(np.isinf(rep_cu)), np.any(rep_cu != 0)), flush=True)

    # Symmetry: all Cu atoms should be equivalent
    max_diff = max(np.max(np.abs(rep_cu[i] - rep_cu[0])) for i in range(1, 4))
    print("  Symmetry: max diff = %.2e %s" % (max_diff, "PASS" if max_diff < 1e-8 else "FAIL"), flush=True)

    # Diamond Si
    t0 = time.time()
    rep_si = generate_mbdf_periodic_higher(
        [si_q], [si_r], [si_c], pbc=pbc, rcut=6.0, max_body=max_body, n_jobs=1)[0]
    t = time.time() - t0
    print("Diamond Si: shape=%s, time=%.2fs" % (str(rep_si.shape), t), flush=True)

    max_diff_si = max(np.max(np.abs(rep_si[i] - rep_si[0])) for i in range(1, 8))
    print("  Symmetry: max diff = %.2e %s" % (max_diff_si, "PASS" if max_diff_si < 1e-8 else "FAIL"), flush=True)

    # Feature breakdown
    if max_body == 3:
        print("  Features: 2+3 body = %d" % rep_cu.shape[1], flush=True)
    elif max_body == 4:
        print("  Features: 2+3 body = 40, 4-body = %d, total = %d" % (
            rep_cu.shape[1] - 40, rep_cu.shape[1]), flush=True)
    elif max_body == 5:
        n_4b = 20  # 5 orders * 4 weights
        n_5b = rep_cu.shape[1] - 40 - n_4b
        print("  Features: 2+3 body = 40, 4-body = %d, 5-body = %d, total = %d" % (
            n_4b, n_5b, rep_cu.shape[1]), flush=True)

    # Check that higher body features are non-zero
    if max_body >= 4:
        fourb = rep_cu[:, 40:]
        print("  4+5 body features: min=%.4f, max=%.4f, mean=%.4f" % (
            fourb.min(), fourb.max(), np.mean(np.abs(fourb))), flush=True)

print("\nDone!", flush=True)
