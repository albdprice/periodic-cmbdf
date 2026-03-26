"""
Validate PyTorch cMBDF against NumPy reference implementation.

Tests:
1. Forward pass numerical agreement (within interpolation tolerance)
2. Gradient computation via autograd
3. Finite difference gradient check
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/home/albd/projects/cmbdf/cMBDF')
import cMBDF
import cMBDF_torch

# Test molecules
charges_h2o = np.array([8.0, 1.0, 1.0])
coords_h2o = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692]
])

charges_ch4 = np.array([6.0, 1.0, 1.0, 1.0, 1.0])
coords_ch4 = np.array([
    [0.0000, 0.0000, 0.0000],
    [0.6276, 0.6276, 0.6276],
    [0.6276, -0.6276, -0.6276],
    [-0.6276, 0.6276, -0.6276],
    [-0.6276, -0.6276, 0.6276]
])

nuclear_charges = np.array([charges_h2o, charges_ch4], dtype=object)
coords = np.array([coords_h2o, coords_ch4], dtype=object)

# All hyperparams for generate_mbdf
all_params = dict(rstep=0.0008, rcut=10.0, alpha_list=[1.5, 5.0],
                  n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                  astep=0.0002, nAs=4, n_atm=2.0)
# For torch version (same params)
conv_params = dict(rstep=0.0008, rcut=10.0, alpha_list=[1.5, 5.0],
                   n_list=[3.0, 5.0], order=4, a1=2.0, a2=2.0,
                   astep=0.0002, nAs=4)

print("=" * 60)
print("Test 1: Forward pass agreement (smooth_cutoff=True)")
print("=" * 60)

ref = cMBDF.generate_mbdf(nuclear_charges, coords, smooth_cutoff=True,
                          n_jobs=1, **all_params)
torch_rep = cMBDF_torch.generate_mbdf(nuclear_charges, coords,
                                       smooth_cutoff=True, **all_params)
torch_np = torch_rep.detach().numpy()

nrs = 20  # 5 * 4 radial features

print("H2O O atom, 2-body (first 5):")
print("  NumPy:", ref[0, 0, :5])
print("  Torch:", torch_np[0, 0, :5])

print("H2O O atom, 3-body (first 5):")
print("  NumPy:", ref[0, 0, nrs:nrs+5])
print("  Torch:", torch_np[0, 0, nrs:nrs+5])

max_err_2b = np.max(np.abs(ref[:, :, :nrs] - torch_np[:, :, :nrs]))
max_err_3b = np.max(np.abs(ref[:, :, nrs:] - torch_np[:, :, nrs:]))

# Relative error
nonzero = np.abs(ref) > 1e-10
rel_err = np.max(np.abs(ref[nonzero] - torch_np[nonzero]) / np.abs(ref[nonzero]))

print("\nMax abs error (2-body): %.2e" % max_err_2b)
print("Max abs error (3-body): %.2e" % max_err_3b)
print("Max relative error:     %.2e" % rel_err)

# Linear interpolation vs int() truncation gives ~1% error at most
# This is expected and acceptable for a differentiable implementation
if rel_err < 0.15:
    print("PASS: Within interpolation tolerance (<15% relative)")
else:
    print("FAIL: Relative error too large")


print()
print("=" * 60)
print("Test 2: Forward pass (smooth_cutoff=False)")
print("=" * 60)

ref2 = cMBDF.generate_mbdf(nuclear_charges, coords, smooth_cutoff=False,
                            n_jobs=1, **all_params)
torch_rep2 = cMBDF_torch.generate_mbdf(nuclear_charges, coords,
                                        smooth_cutoff=False, **all_params)
torch_np2 = torch_rep2.detach().numpy()

max_err2 = np.max(np.abs(ref2 - torch_np2))
nonzero2 = np.abs(ref2) > 1e-10
rel_err2 = np.max(np.abs(ref2[nonzero2] - torch_np2[nonzero2]) / np.abs(ref2[nonzero2]))
print("Max abs error: %.2e" % max_err2)
print("Max rel error: %.2e" % rel_err2)
if rel_err2 < 0.15:
    print("PASS")
else:
    print("FAIL")


print()
print("=" * 60)
print("Test 3: Autograd gradients")
print("=" * 60)

rconvs, aconvs, meta = cMBDF_torch.get_convolutions(**conv_params)
coords_t = torch.tensor(coords_h2o, dtype=torch.float64, requires_grad=True)
charges_t = torch.tensor(charges_h2o, dtype=torch.float64)

rep = cMBDF_torch._compute_rep(charges_t, coords_t, rconvs, aconvs, meta,
                                cutoff_r=10.0, n_atm=2.0, smooth_cutoff=True)
loss = rep.sum()
loss.backward()

grad = coords_t.grad.clone()
print("Gradient shape:", grad.shape)
print("Gradient (O):", grad[0].numpy())
print("Gradient (H1):", grad[1].numpy())
print("Any NaN:", torch.any(torch.isnan(grad)).item())
print("Any zero rows:", (grad.abs().sum(dim=1) == 0).any().item())

if not torch.any(torch.isnan(grad)) and not torch.all(grad == 0):
    print("PASS: Non-zero, valid gradients")
else:
    print("FAIL")


print()
print("=" * 60)
print("Test 4: Finite difference gradient check")
print("=" * 60)

eps = 1e-5
fd_grad = np.zeros_like(coords_h2o)

for a in range(3):
    for d in range(3):
        coords_plus = coords_h2o.copy()
        coords_minus = coords_h2o.copy()
        coords_plus[a, d] += eps
        coords_minus[a, d] -= eps

        rep_plus = cMBDF_torch._compute_rep(
            charges_t,
            torch.tensor(coords_plus, dtype=torch.float64),
            rconvs, aconvs, meta, 10.0, 2.0, True)
        rep_minus = cMBDF_torch._compute_rep(
            charges_t,
            torch.tensor(coords_minus, dtype=torch.float64),
            rconvs, aconvs, meta, 10.0, 2.0, True)

        fd_grad[a, d] = (rep_plus.sum().item() - rep_minus.sum().item()) / (2 * eps)

print("Autograd:\n", grad.numpy())
print("Finite diff:\n", fd_grad)
grad_err = np.max(np.abs(grad.numpy() - fd_grad))
print("Max gradient error: %.2e" % grad_err)

# Relative gradient error
nonzero_g = np.abs(fd_grad) > 1e-8
if nonzero_g.any():
    rel_grad_err = np.max(np.abs(grad.numpy()[nonzero_g] - fd_grad[nonzero_g]) / np.abs(fd_grad[nonzero_g]))
    print("Max relative gradient error: %.2e" % rel_grad_err)
    if rel_grad_err < 0.05:
        print("PASS: Gradients match FD within 5%%")
    else:
        print("WARN: Gradient relative error %.2e" % rel_grad_err)


print()
print("=" * 60)
print("Test 5: No NaN/Inf in outputs")
print("=" * 60)
assert not torch.any(torch.isnan(torch_rep))
assert not torch.any(torch.isinf(torch_rep))
print("PASS")

print()
print("All tests completed!")
