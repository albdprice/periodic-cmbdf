"""
p-cMBDF: Unified periodic convolutional Many-Body Distribution Functionals.

Unified API for generating periodic cMBDF representations with:
- Configurable body order: max_body = 3, 4, or 5
- Configurable backend: 'numba' (CPU, production) or 'torch' (GPU-accelerated)
- Element-specific radial basis option
- Per-element normalization

Usage:
    from pcmbdf import generate_pcmbdf

    # 3-body, Numba CPU (fastest for small structures)
    reps = generate_pcmbdf(charges, coords, cells, max_body=3, backend='numba')

    # 4-body, GPU (best for large structures or batches)
    reps = generate_pcmbdf(charges, coords, cells, max_body=4, backend='torch')

    # 5-body, toggleable
    reps = generate_pcmbdf(charges, coords, cells, max_body=5, backend='numba')

Feature dimensions:
    max_body=3: 40 features (20 radial + 20 angular)
    max_body=4: 60 features (40 + 20 solid angle)
    max_body=5: 72 features (40 + 20 solid angle + 12 OOP angle)
"""

import numpy as np


def generate_pcmbdf(nuclear_charges, coords, cells,
                     pbc=(True, True, True),
                     max_body=3, backend='numba',
                     rcut=6.0, n_atm=2.0, n_atm_4b=1.0,
                     elem_specific=True,
                     normalize=True, norm_mode='mean',
                     pad=None, n_jobs=-1, progress_bar=False,
                     device='cuda',
                     # Convolution hyperparameters
                     order=4, order_4b=4, n_dih=4,
                     order_5b=3, n_5b=3,
                     **conv_kwargs):
    """
    Generate p-cMBDF representations.

    Args:
        nuclear_charges: list of arrays of nuclear charges
        coords: list of arrays of Cartesian coordinates
        cells: list of (3,3) lattice vectors
        pbc: periodic boundary conditions
        max_body: 3 (40 dim), 4 (60 dim), or 5 (72 dim)
        backend: 'numba' (CPU) or 'torch' (GPU if available)
        rcut: radial cutoff in Angstrom
        n_atm: ATM damping for 2+3 body
        n_atm_4b: ATM damping for 4+5 body
        elem_specific: use element-specific radial basis
        normalize: apply per-element normalization
        norm_mode: 'mean' or 'max'
        pad: zero-pad to this many atoms
        n_jobs: parallelism for Numba backend
        progress_bar: show tqdm progress
        device: 'cpu' or 'cuda' for torch backend

    Returns:
        reps: (n_structures, pad, n_features) array or tensor
        norm_factors: dict (if normalize=True) or None
    """
    if backend == 'numba':
        return _generate_numba(
            nuclear_charges, coords, cells, pbc, max_body,
            rcut, n_atm, n_atm_4b, elem_specific, normalize, norm_mode,
            pad, n_jobs, progress_bar,
            order, order_4b, n_dih, order_5b, n_5b, **conv_kwargs)
    elif backend == 'torch':
        return _generate_torch(
            nuclear_charges, coords, cells, pbc, max_body,
            rcut, n_atm, n_atm_4b, elem_specific, normalize, norm_mode,
            pad, device, progress_bar,
            order, order_4b, n_dih, order_5b, n_5b, **conv_kwargs)
    else:
        raise ValueError("backend must be 'numba' or 'torch'")


def _generate_numba(nuclear_charges, coords, cells, pbc, max_body,
                     rcut, n_atm, n_atm_4b, elem_specific, normalize, norm_mode,
                     pad, n_jobs, progress_bar,
                     order, order_4b, n_dih, order_5b, n_5b, **conv_kwargs):
    """Numba CPU backend."""
    if max_body == 3:
        from cMBDF_periodic import generate_mbdf_periodic, normalize_per_element
        reps = generate_mbdf_periodic(
            nuclear_charges, coords, cells, pbc=pbc,
            rcut=rcut, n_atm=n_atm, elem_specific=elem_specific,
            n_jobs=n_jobs, pad=pad, progress_bar=progress_bar,
            order=order, **conv_kwargs)
    else:
        from cMBDF_higher_body import generate_mbdf_periodic_higher
        reps = generate_mbdf_periodic_higher(
            nuclear_charges, coords, cells, pbc=pbc,
            rcut=rcut, n_atm=n_atm, n_atm_4b=n_atm_4b,
            max_body=max_body,
            order_4b=order_4b, n_dih=n_dih,
            order_5b=order_5b, n_5b=n_5b,
            n_jobs=n_jobs, pad=pad, progress_bar=progress_bar,
            order=order, **conv_kwargs)

    norm_factors = None
    if normalize:
        from cMBDF_periodic import normalize_per_element
        reps, norm_factors = normalize_per_element(reps, nuclear_charges, mode=norm_mode)

    return reps, norm_factors


def _generate_torch(nuclear_charges, coords, cells, pbc, max_body,
                     rcut, n_atm, n_atm_4b, elem_specific, normalize, norm_mode,
                     pad, device, progress_bar,
                     order, order_4b, n_dih, order_5b, n_5b, **conv_kwargs):
    """PyTorch GPU backend with vectorized higher body orders."""
    import torch
    from cMBDF_periodic_torch import (
        get_convolutions, build_neighbor_data_torch, _interp
    )

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", flush=True)
        device = 'cpu'

    # Precompute convolutions
    rconvs, aconvs, meta = get_convolutions(
        rcut=rcut, order=order, device=device,
        **{k: v for k, v in conv_kwargs.items()
           if k in ['rstep', 'alpha_list', 'n_list', 'a1', 'a2', 'astep', 'nAs']})

    m1, n1_w, grid_r = rconvs.shape
    m2, n2_w, grid_a = aconvs.shape
    nrs = m1 * n1_w
    nAs = m2 * n2_w
    dtype = torch.float64

    # 4-body convolutions
    dconvs_t = None
    n_4b = 0
    if max_body >= 4:
        from cMBDF_higher_body import get_dihedral_convolutions
        dconvs_np, meta_dih = get_dihedral_convolutions(
            order=order_4b, n_dih=n_dih)
        dconvs_t = torch.tensor(dconvs_np, dtype=dtype, device=device)
        n_4b = dconvs_t.shape[0] * dconvs_t.shape[1]

    # 5-body convolutions
    fconvs_t = None
    n_5b_feat = 0
    if max_body >= 5:
        from cMBDF_higher_body import get_fivebody_convolutions
        fconvs_np, meta_5b = get_fivebody_convolutions(
            order=order_5b, n_5b=n_5b)
        fconvs_t = torch.tensor(fconvs_np, dtype=dtype, device=device)
        n_5b_feat = fconvs_t.shape[0] * fconvs_t.shape[1]

    n_feat_total = nrs + nAs + n_4b + n_5b_feat

    if not isinstance(pbc[0], (list, tuple)):
        pbc_list = [pbc] * len(nuclear_charges)
    else:
        pbc_list = pbc

    lengths = [len(q) for q in nuclear_charges]
    if pad is None:
        pad_val = max(lengths)
    else:
        pad_val = pad

    all_reps = []

    for struct_idx, (q, r, c, p) in enumerate(zip(nuclear_charges, coords, cells, pbc_list)):
        q_np = np.asarray(q, dtype=np.float64)
        r_np = np.asarray(r, dtype=np.float64)
        c_np = np.asarray(c, dtype=np.float64)
        N = len(q_np)

        # Build neighbor list
        idx_i, idx_j, rij_vec, rij_dist, offsets, nbr_pos = \
            build_neighbor_data_torch(q_np, r_np, c_np, p, rcut, device)

        charges = torch.tensor(q_np, dtype=dtype, device=device)
        n_pairs = idx_i.shape[0]

        # ---- 2-body (vectorized) ----
        if n_pairs > 0:
            fcut_ij = 0.5 * (torch.cos(torch.pi * rij_dist / rcut) + 1.0)
            pref_2b = torch.sqrt(charges[idx_i] * charges[idx_j]) * fcut_ij
            r_idx = rij_dist / meta['rstep']
            rconvs_flat = rconvs.reshape(nrs, grid_r)
            rep_2b = torch.zeros(N, nrs, dtype=dtype, device=device)
            for ch in range(nrs):
                vals = _interp(rconvs_flat[ch], r_idx) * pref_2b
                rep_2b[:, ch].scatter_add_(0, idx_i, vals)
        else:
            rep_2b = torch.zeros(N, nrs, dtype=dtype, device=device)

        # ---- 3-body (vectorized triplets) ----
        rep_3b = torch.zeros(N, nAs, dtype=dtype, device=device)
        offsets_np = offsets.cpu().numpy()

        tri_i_list, tri_p_list, tri_q_list = [], [], []
        for i in range(N):
            start, end = offsets_np[i], offsets_np[i+1]
            nn = end - start
            if nn < 2:
                continue
            lp, lq = torch.triu_indices(nn, nn, offset=1, device=device)
            tri_i_list.append(torch.full((lp.shape[0],), i, dtype=torch.long, device=device))
            tri_p_list.append(lp + start)
            tri_q_list.append(lq + start)

        if tri_i_list:
            tri_i = torch.cat(tri_i_list)
            tri_p = torch.cat(tri_p_list)
            tri_q = torch.cat(tri_q_list)

            rij_v = rij_vec[tri_p]
            rik_v = rij_vec[tri_q]
            rij_n = rij_dist[tri_p]
            rik_n = rij_dist[tri_q]
            rkj_v = nbr_pos[tri_q] - nbr_pos[tri_p]
            rkj_n = torch.norm(rkj_v, dim=1)

            valid = rkj_n > 1e-10
            if not valid.all():
                tri_i, tri_p, tri_q = tri_i[valid], tri_p[valid], tri_q[valid]
                rij_v, rik_v = rij_v[valid], rik_v[valid]
                rij_n, rik_n = rij_n[valid], rik_n[valid]
                rkj_v, rkj_n = rkj_v[valid], rkj_n[valid]

            fc_ij = 0.5 * (torch.cos(torch.pi * rij_n / rcut) + 1.0)
            fc_ik = 0.5 * (torch.cos(torch.pi * rik_n / rcut) + 1.0)
            fc_jk = torch.where(rkj_n < rcut,
                                 0.5 * (torch.cos(torch.pi * rkj_n / rcut) + 1.0),
                                 torch.zeros_like(rkj_n))

            j_idx = idx_j[tri_p]
            k_idx = idx_j[tri_q]
            cos1 = ((rij_v * rik_v).sum(1) / (rij_n * rik_n)).clamp(-1, 1)
            cos2 = ((rij_v * rkj_v).sum(1) / (rij_n * rkj_n)).clamp(-1, 1)
            cos3 = ((-rkj_v * rik_v).sum(1) / (rkj_n * rik_n)).clamp(-1, 1)

            ang1 = torch.acos(cos1)
            atm_3b = (rij_n * rik_n * rkj_n) ** n_atm
            charge_3b = torch.pow(charges[tri_i] * charges[j_idx] * charges[k_idx], 1.0/3.0)
            pref_3b = charge_3b * fc_ij * fc_ik * fc_jk

            aconvs_flat = aconvs.reshape(nAs, grid_a)
            a1_idx = ang1 / meta['astep']

            for ch in range(nAs):
                i2 = ch % n2_w
                v1 = _interp(aconvs_flat[ch], a1_idx)
                if i2 == 0:
                    contrib = (pref_3b * v1 * cos2 * cos3) / atm_3b
                else:
                    contrib = (pref_3b * v1) / atm_3b
                rep_3b[:, ch].scatter_add_(0, tri_i, 2.0 * contrib)

        parts = [rep_2b, rep_3b]

        # ---- 4-body (vectorized quadruplets — solid angle) ----
        if max_body >= 4 and dconvs_t is not None and tri_i_list:
            rep_4b = torch.zeros(N, n_4b, dtype=dtype, device=device)
            dconvs_flat = dconvs_t.reshape(n_4b, dconvs_t.shape[-1])
            grid_4b = dconvs_t.shape[-1]
            omega_step = 2.0 * np.pi / grid_4b

            # Enumerate quadruplets: for each atom i, all (p,q,s) with p<q<s
            quad_i_list, quad_p_list, quad_q_list, quad_s_list = [], [], [], []
            for i in range(N):
                start, end = offsets_np[i], offsets_np[i+1]
                nn = end - start
                if nn < 3:
                    continue
                # All combinations of 3 from nn neighbors
                for a in range(nn):
                    for b in range(a+1, nn):
                        for cc in range(b+1, nn):
                            quad_i_list.append(i)
                            quad_p_list.append(start + a)
                            quad_q_list.append(start + b)
                            quad_s_list.append(start + cc)

            if quad_i_list:
                qi = torch.tensor(quad_i_list, dtype=torch.long, device=device)
                qp = torch.tensor(quad_p_list, dtype=torch.long, device=device)
                qq = torch.tensor(quad_q_list, dtype=torch.long, device=device)
                qs = torch.tensor(quad_s_list, dtype=torch.long, device=device)

                # Unit vectors from central atom to each neighbor
                uij = rij_vec[qp] / rij_dist[qp].unsqueeze(1)
                uik = rij_vec[qq] / rij_dist[qq].unsqueeze(1)
                uil = rij_vec[qs] / rij_dist[qs].unsqueeze(1)

                # Solid angle via Van Oosterom-Strackee
                cross_kl = torch.cross(uik, uil, dim=1)
                numerator = torch.abs((uij * cross_kl).sum(dim=1))

                dij_ik = (uij * uik).sum(dim=1)
                dij_il = (uij * uil).sum(dim=1)
                dik_il = (uik * uil).sum(dim=1)
                denominator = (1.0 + dij_ik + dij_il + dik_il).clamp(min=1e-10)

                omega = 2.0 * torch.atan2(numerator, denominator)

                # Grid index
                omega_idx = omega / omega_step

                # Charge and damping
                rij_n_q = rij_dist[qp]
                rik_n_q = rij_dist[qq]
                ril_n_q = rij_dist[qs]

                fc_j = 0.5 * (torch.cos(torch.pi * rij_n_q / rcut) + 1.0)
                fc_k = 0.5 * (torch.cos(torch.pi * rik_n_q / rcut) + 1.0)
                fc_l = 0.5 * (torch.cos(torch.pi * ril_n_q / rcut) + 1.0)

                z_j = charges[idx_j[qp]]
                z_k = charges[idx_j[qq]]
                z_l = charges[idx_j[qs]]

                charge_4b = torch.pow(charges[qi] * z_j * z_k * z_l, 0.25)
                atm_4b = (rij_n_q * rik_n_q * ril_n_q) ** n_atm_4b
                pref_4b = charge_4b * fc_j * fc_k * fc_l / atm_4b

                for ch in range(n_4b):
                    vals = _interp(dconvs_flat[ch], omega_idx) * pref_4b
                    rep_4b[:, ch].scatter_add_(0, qi, vals)

            parts.append(rep_4b)

        # ---- 5-body (vectorized quintuplets — OOP angle) ----
        if max_body >= 5 and fconvs_t is not None and tri_i_list:
            rep_5b = torch.zeros(N, n_5b_feat, dtype=dtype, device=device)
            fconvs_flat = fconvs_t.reshape(n_5b_feat, fconvs_t.shape[-1])
            grid_5b = fconvs_t.shape[-1]
            fstep = np.pi / grid_5b

            # Enumerate quintuplets: for each atom i, all (p,q,s,t) with p<q<s<t
            quint_i_list, quint_p_list, quint_q_list, quint_s_list, quint_t_list = [], [], [], [], []
            for i in range(N):
                start, end = offsets_np[i], offsets_np[i+1]
                nn = end - start
                if nn < 4:
                    continue
                for a in range(nn):
                    for b in range(a+1, nn):
                        for cc in range(b+1, nn):
                            for dd in range(cc+1, nn):
                                quint_i_list.append(i)
                                quint_p_list.append(start + a)
                                quint_q_list.append(start + b)
                                quint_s_list.append(start + cc)
                                quint_t_list.append(start + dd)

            if quint_i_list:
                q5i = torch.tensor(quint_i_list, dtype=torch.long, device=device)
                q5p = torch.tensor(quint_p_list, dtype=torch.long, device=device)
                q5q = torch.tensor(quint_q_list, dtype=torch.long, device=device)
                q5s = torch.tensor(quint_s_list, dtype=torch.long, device=device)
                q5t = torch.tensor(quint_t_list, dtype=torch.long, device=device)

                rij_5 = rij_vec[q5p]
                rik_5 = rij_vec[q5q]
                ril_5 = rij_vec[q5s]
                rim_5 = rij_vec[q5t]

                # Compute OOP angle: angle between rim and plane(rij, rik, ril)
                # Symmetrize: accumulate all 4 choices of OOP atom
                def _oop_batch(r1, r2, r3, r_oop):
                    v1 = r2 - r1
                    v2 = r3 - r1
                    n = torch.cross(v1, v2, dim=1)
                    n_norm = torch.norm(n, dim=1, keepdim=True).clamp(min=1e-10)
                    r_norm = torch.norm(r_oop, dim=1, keepdim=True).clamp(min=1e-10)
                    cos_psi = torch.abs((r_oop * n).sum(dim=1)) / (r_norm.squeeze() * n_norm.squeeze())
                    return torch.acos(cos_psi.clamp(0, 1))

                rij_n5 = rij_dist[q5p]
                rik_n5 = rij_dist[q5q]
                ril_n5 = rij_dist[q5s]
                rim_n5 = rij_dist[q5t]

                fc5_j = 0.5 * (torch.cos(torch.pi * rij_n5 / rcut) + 1.0)
                fc5_k = 0.5 * (torch.cos(torch.pi * rik_n5 / rcut) + 1.0)
                fc5_l = 0.5 * (torch.cos(torch.pi * ril_n5 / rcut) + 1.0)
                fc5_m = 0.5 * (torch.cos(torch.pi * rim_n5 / rcut) + 1.0)

                z5_j = charges[idx_j[q5p]]
                z5_k = charges[idx_j[q5q]]
                z5_l = charges[idx_j[q5s]]
                z5_m = charges[idx_j[q5t]]

                charge_5b = torch.pow(charges[q5i] * z5_j * z5_k * z5_l * z5_m, 0.2)
                atm_5b = (rij_n5 * rik_n5 * ril_n5 * rim_n5) ** n_atm_4b
                pref_5b = charge_5b * fc5_j * fc5_k * fc5_l * fc5_m / atm_5b

                # All 4 OOP angles
                psi1 = _oop_batch(rij_5, rik_5, ril_5, rim_5)
                psi2 = _oop_batch(rij_5, rik_5, rim_5, ril_5)
                psi3 = _oop_batch(rij_5, ril_5, rim_5, rik_5)
                psi4 = _oop_batch(rik_5, ril_5, rim_5, rij_5)

                for psi in [psi1, psi2, psi3, psi4]:
                    psi_idx = psi / fstep
                    for ch in range(n_5b_feat):
                        vals = _interp(fconvs_flat[ch], psi_idx) * pref_5b
                        rep_5b[:, ch].scatter_add_(0, q5i, vals)

            parts.append(rep_5b)

        # Concatenate all body orders
        rep = torch.cat(parts, dim=1)

        # Pad
        if pad_val > N:
            padding = torch.zeros(pad_val - N, rep.shape[1], dtype=dtype, device=device)
            rep = torch.cat([rep, padding], dim=0)

        all_reps.append(rep.cpu().numpy())

        if progress_bar and (struct_idx + 1) % 100 == 0:
            print("  %d/%d" % (struct_idx + 1, len(nuclear_charges)), flush=True)

    reps = np.array(all_reps)

    norm_factors = None
    if normalize:
        from cMBDF_periodic import normalize_per_element
        reps, norm_factors = normalize_per_element(reps, nuclear_charges, mode=norm_mode)

    return reps, norm_factors
