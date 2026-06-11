import numpy as np

from tb_spinor import *

import numpy as np

####################################################
# wigner transform reduction: center-relative
# then diagonalize at fixed (k_super, r0, k_p)
####################################################

def reshape_h_full_to_2d(h_full, n_spin=2):
    """
    Convert tb._gen_ham output to a plain 2D matrix.

    Supported shapes:
      - (n, n)
      - (n_orb, n_spin, n_orb, n_spin)  -> reshape to (n_orb*n_spin, n_orb*n_spin)
    """
    h_full = np.asarray(h_full)
    if h_full.ndim == 2:
        return h_full.astype(np.complex128, copy=False)

    if h_full.ndim == 4:
        if h_full.shape[1] != n_spin or h_full.shape[3] != n_spin:
            raise ValueError(f"unexpected spin axes in h_full with shape {h_full.shape}")
        # h_full is (orb_i, s_i, orb_j, s_j); flatten (orb_i,s_i) and (orb_j,s_j)
        return h_full.reshape(h_full.shape[0] * n_spin, h_full.shape[2] * n_spin).astype(
            np.complex128, copy=False
        )

    raise ValueError(f"unsupported h_full.ndim={h_full.ndim}, shape={h_full.shape}")


def build_wigner_cell_block_idx(base_norb, sc_vectors):
    """
    Build mapping from internal supercell cell r to the reduced subspace indices
    inside the flattened 2D spinful Hamiltonian.

    Ordering assumption used throughout:
      spinful flat index = 2 * orb + s
      spinless orb index = cell_idx * base_norb + wf_idx
    Therefore:
      flat_index = 2 * (cell_idx*base_norb + wf_idx) + s
    """
    sc_vectors = np.asarray(sc_vectors, dtype=np.int64)
    if sc_vectors.ndim == 1:
        sc_vectors = sc_vectors.reshape(1, -1)

    ncell, dim_r = sc_vectors.shape
    red_dim = 2 * base_norb  # fixed center r0: base_norb orbitals * 2 spins

    cell_block_idx = np.zeros((ncell, red_dim), dtype=np.int64)
    for r in range(ncell):
        for wf in range(base_norb):
            orb = r * base_norb + wf  # spinless orb index in supercell ordering
            cell_block_idx[r, 2 * wf + 0] = 2 * orb + 0
            cell_block_idx[r, 2 * wf + 1] = 2 * orb + 1

    return cell_block_idx, sc_vectors


def _wigner_reduced_ham_from_full(
    h_full_2d,
    k_p,
    *,
    cell_block_idx,
    cell_vectors,
    target_r0_cell_idx=0,
    phase_2pi=True,
    wigner_norm_mode="pairs",
):
    """
    Build reduced wigner hamiltonian at fixed center r0 and fixed k_p.

    Discrete center condition:
        r0 = (r + r') / 2
    implemented as:
        cell_vectors[r] + cell_vectors[r'] == 2 * cell_vectors[target_r0_cell_idx]
    """
    h_full_2d = np.asarray(h_full_2d)
    cell_vectors = np.asarray(cell_vectors, dtype=np.int64)
    cell_block_idx = np.asarray(cell_block_idx, dtype=np.int64)

    ncell, red_dim = cell_block_idx.shape
    dim_r = cell_vectors.shape[1]

    k_p = np.asarray(k_p, dtype=np.float64).reshape(-1)
    if k_p.shape != (dim_r,):
        raise ValueError(f"k_p must have shape ({dim_r},), got {k_p.shape}")

    r0_vec = cell_vectors[int(target_r0_cell_idx)]
    center2 = 2 * r0_vec

    pairs = []
    for r in range(ncell):
        for rp in range(ncell):
            if np.all(cell_vectors[r] + cell_vectors[rp] == center2):
                delta = cell_vectors[rp] - cell_vectors[r]
                pairs.append((r, rp, delta))

    if len(pairs) == 0:
        raise ValueError(
            "no (r,r') pairs satisfy the requested r0 center; "
            "check target_r0_cell_idx and sc_vectors bookkeeping."
        )

    if wigner_norm_mode == "pairs":
        n_delta = len(pairs)
    elif wigner_norm_mode == "ncell":
        n_delta = ncell
    else:
        raise ValueError("wigner_norm_mode must be 'pairs' or 'ncell'")

    norm = np.sqrt(float(n_delta))
    h_red = np.zeros((red_dim, red_dim), dtype=np.complex128)

    for r, rp, delta in pairs:
        arg = float(np.dot(k_p, delta))
        if phase_2pi:
            phase = np.exp(-2j * np.pi * arg)
        else:
            phase = np.exp(-1j * arg)

        idx_r = cell_block_idx[r]    # indices into h_full_2d
        idx_rp = cell_block_idx[rp]  # indices into h_full_2d
        h_red += phase * h_full_2d[np.ix_(idx_r, idx_rp)]

    h_red /= norm

    # enforce hermiticity numerically
    h_red = 0.5 * (h_red + h_red.conjugate().T)
    return h_red


def compute_wigner_bands_for_fixed_k_and_r0(
    tb_up,
    tb_dn,
    k_super,
    k_p_list,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
    target_r0_cell_idx=0,
    eig_vectors=False,
    phase_2pi=True,
    wigner_norm_mode="pairs",
):
    """
    Assemble rotated Hamiltonian H_full(k_super) using tb_rot._gen_ham,
    reshape to 2D, perform center-relative (wigner) reduction at fixed r0,
    then diagonalize for each k_p in k_p_list.

    Returns:
      evals: (red_dim, nk_p)
      evecs: (red_dim, nk_p, red_dim) if eig_vectors=True
      meta : dict
    """
    # 1) build primitive spinful model (your existing pipeline)
    tb_spin_prim = build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=fermi_level
    )

    # 2) build rotated model (primitive or magnetic supercell)
    if sc_red_lat is not None:
        tb_rot = apply_local_spin_rotation_from_cellwf_samples(
            tb_spin_prim, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
        )
        _, sc_vectors = tb_spin_prim.make_supercell(
            sc_red_lat, return_sc_vectors=True, to_home=to_home
        )
    else:
        tb_rot = apply_local_spin_rotation_from_cellwf_samples(
            tb_spin_prim, u_samples=u_samples, sc_red_lat=None, to_home=to_home
        )
        sc_vectors = np.zeros((1, tb_rot._dim_r), dtype=np.int64)

    base_norb = int(tb_spin_prim._norb)  # spinless orbitals per internal cell
    ncell = int(np.asarray(sc_vectors).shape[0])

    # 3) build wigner reduction mapping
    cell_block_idx, cell_vectors = build_wigner_cell_block_idx(base_norb, sc_vectors)
    red_dim = 2 * base_norb
    meta = {
        "base_norb": base_norb,
        "ncell": ncell,
        "dim_r": int(cell_vectors.shape[1]),
        "cell_vectors": np.asarray(cell_vectors, dtype=np.int64),
        "target_r0_cell_idx": int(target_r0_cell_idx),
    }

    ncell_check = int(cell_vectors.shape[0])
    if not (0 <= int(target_r0_cell_idx) < ncell_check):
        raise ValueError(f"target_r0_cell_idx must be in [0, {ncell_check-1}]")

    # 4) normalize k_p_list shape
    k_p_list = np.asarray(k_p_list, dtype=np.float64)
    dim_r = int(cell_vectors.shape[1])

    if dim_r == 1:
        if k_p_list.ndim == 1:
            k_p_list = k_p_list.reshape(-1, 1)
        elif not (k_p_list.ndim == 2 and k_p_list.shape[1] == 1):
            raise ValueError("for dim_r==1, k_p_list must be shape (nk,) or (nk,1)")
    else:
        if k_p_list.ndim != 2 or k_p_list.shape[1] != dim_r:
            raise ValueError(f"k_p_list must have shape (nk, {dim_r}), got {k_p_list.shape}")

    nk_p = int(k_p_list.shape[0])

    # 5) normalize k_super shape
    k_super = np.asarray(k_super, dtype=np.float64)
    if tb_rot._dim_k == 1:
        if k_super.ndim == 0:
            k_super = np.array([float(k_super)], dtype=np.float64)
        elif not (k_super.ndim == 1 and k_super.shape == (1,)):
            raise ValueError("k_super must be scalar or shape (1,) when dim_k==1")
    else:
        if k_super.ndim != 1 or k_super.shape[0] != tb_rot._dim_k:
            raise ValueError(f"k_super must have shape ({tb_rot._dim_k},) for dim_k>1")

    # 6) build h_full and reshape to 2D
    h_full_raw = tb_rot._gen_ham(k_super)
    h_full_2d = reshape_h_full_to_2d(h_full_raw, n_spin=2)

    expected_norb_sc_spinless = base_norb * ncell
    expected_dim2d = 2 * expected_norb_sc_spinless  # spinful
    if h_full_2d.shape[0] != expected_dim2d or h_full_2d.shape[1] != expected_dim2d:
        raise ValueError(
            "h_full_2d size does not match the wigner bookkeeping assumptions.\n"
            f"  expected_dim2d={expected_dim2d}\n"
            f"  h_full_2d.shape={h_full_2d.shape}\n"
            "  Check that orbitals in tb_rot._gen_ham are ordered as "
            "orb = cell_idx*base_norb + wf_idx (spinful flatten 2*orb+s)."
        )

    # 7) diagonalize reduced Hamiltonian for each k_p
    evals = np.zeros((red_dim, nk_p), dtype=np.float64)
    evecs = np.zeros((red_dim, nk_p, red_dim), dtype=np.complex128) if eig_vectors else None

    for ip in range(nk_p):
        k_p = k_p_list[ip]
        h_red = _wigner_reduced_ham_from_full(
            h_full_2d,
            k_p,
            cell_block_idx=cell_block_idx,
            cell_vectors=cell_vectors,
            target_r0_cell_idx=int(target_r0_cell_idx),
            phase_2pi=phase_2pi,
            wigner_norm_mode=wigner_norm_mode,
        )
        w, v = np.linalg.eigh(h_red)
        evals[:, ip] = w.real
        if eig_vectors:
            evecs[:, ip, :] = v

    if eig_vectors:
        return evals, evecs, meta
    return evals, meta
