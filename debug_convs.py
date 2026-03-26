import numpy as np
import sys
sys.path.insert(0, "/home/albd/projects/cmbdf/cMBDF")
import cMBDF
import cMBDF_torch

params_conv = dict(rstep=0.0008, rcut=10.0, alpha_list=[1.5,5.0],
                   n_list=[3.0,5.0], order=4, a1=2.0, a2=2.0,
                   astep=0.0002, nAs=4)

# NumPy convolutions
np_convs = cMBDF.get_convolutions(**params_conv, gradients=False)
np_rconvs = np_convs[0][0]  # shape (m, n, grid)
np_aconvs = np_convs[1][0]

# Torch convolutions
t_rconvs, t_aconvs, meta = cMBDF_torch.get_convolutions(**params_conv)

print("NumPy rconvs shape:", np_rconvs.shape)
print("Torch rconvs shape:", t_rconvs.shape)
print("NumPy aconvs shape:", np_aconvs.shape)
print("Torch aconvs shape:", t_aconvs.shape)

diff_r = np.max(np.abs(np_rconvs - t_rconvs.numpy()))
diff_a = np.max(np.abs(np_aconvs - t_aconvs.numpy()))
print("Max rconvs diff:", diff_r)
print("Max aconvs diff:", diff_a)

# Check specific grid points
idx = 1196
print("np rconv[0,0,%d]: %.10f" % (idx, np_rconvs[0, 0, idx]))
print("torch rconv[0,0,%d]: %.10f" % (idx, t_rconvs[0, 0, idx].item()))

# Check if grid sizes match
print("np rgrid size:", np_rconvs.shape[-1])
print("torch rgrid size:", t_rconvs.shape[-1])
print("np agrid size:", np_aconvs.shape[-1])
print("torch agrid size:", t_aconvs.shape[-1])
