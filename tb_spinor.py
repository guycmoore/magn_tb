import numpy as np
import copy
from pythtb import tb_model
from numba import njit, prange
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


def _as_unitary(u, tol=1e-8):
    u = np.asarray(u, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError(f"u must have shape (2,2), got {u.shape}")
    err = np.max(np.abs(u.conj().T @ u - np.eye(2, dtype=complex)))
    if err > tol:
        raise ValueError(f"u not unitary: max ||u†u - I|| = {err}")
    return u


def build_spinful_from_collinear_intersection(tb_up, tb_dn, fermi_level=0.0):
    """
    Builds nspin=2 spinor model with diagonal spin blocks using the INTERSECTION
    of hopping keys (i,j,R) present in both tb_up and tb_dn:
      onsite: [[e_up, 0], [0, e_dn]]
      hop:    [[t_up, 0], [0, t_dn]]
    """
    if tb_up._nspin != 1 or tb_dn._nspin != 1:
        raise ValueError("expected both inputs to have nspin=1 (spinless).")

    for attr in ["_dim_k", "_dim_r", "_norb", "_per"]:
        if getattr(tb_up, attr) != getattr(tb_dn, attr):
            raise ValueError(f"models not compatible: tb_up.{attr} != tb_dn.{attr}")

    tol = 1e-2
    if np.max(np.abs(tb_up._lat - tb_dn._lat)) > tol:
        raise ValueError("lattice vectors differ between tb models.")
    if np.max(np.abs(tb_up._orb - tb_dn._orb)) > tol:
        raise ValueError("orbital positions differ between tb models.")

    dim_k = tb_up._dim_k
    dim_r = tb_up._dim_r
    lat = tb_up.get_lat()
    orb = tb_up.get_orb()
    per = copy.deepcopy(tb_up._per)
    norb = tb_up.get_num_orbitals()

    tb_spin = tb_model(dim_k, dim_r, lat=lat, orb=orb, per=per, nspin=2)

    # onsite
    onsite = []
    for i in range(norb):
        e_up = tb_up._site_energies[i] - fermi_level
        e_dn = tb_dn._site_energies[i] - fermi_level
        onsite.append(
            np.array([[complex(e_up), 0.0],
                      [0.0, complex(e_dn)]], dtype=complex)
        )
    tb_spin.set_onsite(onsite)

    # hopping maps
    def hop_key(h, tb):
        i = int(h[1])
        j = int(h[2])
        if tb._dim_k == 0:
            return (i, j)
        r = np.array(h[3], dtype=int)
        return (i, j, tuple(r.tolist()))

    up_map = {hop_key(h, tb_up): complex(h[0]) for h in tb_up._hoppings}
    dn_map = {hop_key(h, tb_dn): complex(h[0]) for h in tb_dn._hoppings}

    common_keys = set(up_map.keys()) & set(dn_map.keys())
    if len(common_keys) == 0:
        raise ValueError("No common hopping keys found between tb_up and tb_dn.")

    for key in common_keys:
        if dim_k == 0:
            i, j = key
            amp_spin = np.array([[up_map[key], 0.0],
                                  [0.0, dn_map[key]]], dtype=complex)
            tb_spin.set_hop(amp_spin, i, j, mode="set")
        else:
            i, j, r_tup = key
            r = np.array(r_tup, dtype=int)
            amp_spin = np.array([[up_map[key], 0.0],
                                  [0.0, dn_map[key]]], dtype=complex)
            tb_spin.set_hop(amp_spin, i, j, ind_R=r, mode="set")

    return tb_spin


def apply_local_spin_rotation_from_cellwf_samples(
    tb_spinful,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    su2_tol=1e-8,
):
    """
    Assigns u_i per orbital i using:
      u_samples[(cell_r_int_tuple, wf_index)] -> 2x2 unitary spinor matrix
    Rotates onsite and hopping blocks:
      eps' = u eps u†
      t'   = u_i t u_j†
    """
    if tb_spinful._nspin != 2:
        raise ValueError("tb_spinful must have nspin=2.")

    base_norb = tb_spinful._norb
    dim_r = tb_spinful._dim_r
    dim_k = tb_spinful._dim_k

    if sc_red_lat is not None:
        sc_tb, sc_vectors = tb_spinful.make_supercell(
            sc_red_lat, return_sc_vectors=True, to_home=to_home
        )
    else:
        sc_tb = tb_spinful
        sc_vectors = [np.zeros(dim_r, dtype=int)]

    num_sc = len(sc_vectors)
    if sc_tb._norb != base_norb * num_sc:
        raise ValueError("unexpected supercell orbital count; check sc_red_lat.")

    u_list = [None] * sc_tb._norb
    for sc_i, cell_r in enumerate(sc_vectors):
        cell_key = tuple(int(x) for x in np.asarray(cell_r, dtype=int))
        for wf_i in range(base_norb):
            orb_i = sc_i * base_norb + wf_i
            key = (cell_key, int(wf_i))
            if key not in u_samples:
                raise KeyError(
                    f"missing u_samples for key={key}. "
                    f"you need u for every (cell, wf_index) in the supercell."
                )
            u_list[orb_i] = _as_unitary(u_samples[key], tol=su2_tol)

    lat = sc_tb.get_lat()
    orb = sc_tb.get_orb()
    per = copy.deepcopy(sc_tb._per)

    tb_rot = tb_model(dim_k, dim_r, lat=lat, orb=orb, per=per, nspin=2)

    # onsite rotation: eps' = u eps u†
    onsite_rot = []
    for i in range(sc_tb._norb):
        eps = np.array(sc_tb._site_energies[i], dtype=complex)
        ui = u_list[i]
        onsite_rot.append(ui @ eps @ ui.conj().T)
    tb_rot.set_onsite(onsite_rot)

    # hopping rotation: t' = u_i t u_j†
    for h in sc_tb._hoppings:
        amp = np.array(h[0], dtype=complex)
        i = int(h[1])
        j = int(h[2])
        ui = u_list[i]
        uj = u_list[j]
        amp_p = ui @ amp @ uj.conj().T

        if dim_k == 0:
            tb_rot.set_hop(amp_p, i, j, mode="set")
        else:
            ind_r = np.array(h[3], dtype=int)
            tb_rot.set_hop(amp_p, i, j, ind_R=ind_r, mode="set")

    return tb_rot


# ---------------------------------------------------------------------------
# 1. Unoptimized Reference Serial Solver
# ---------------------------------------------------------------------------

def compute_rotated_bands_serial(
    tb_up,
    tb_dn,
    k_vec,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
    eig_vectors=True
):
    """
    Clean, unoptimized serial reference solver.
    Evaluates and diagonalizes H(k) one k-point at a time sequentially.
    """
    tb_spin = build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=fermi_level
    )
    tb_rot = apply_local_spin_rotation_from_cellwf_samples(
        tb_spin, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
    )

    k_arr = np.asarray(k_vec, dtype=float)
    nk = k_arr.shape[0]

    evals_list = []
    evecs_list = []

    for ik in range(nk):
        kpt = k_arr[ik]
        h = np.asarray(tb_rot._gen_ham(kpt), dtype=np.complex128)
        if h.ndim == 4:
            h = h.reshape(h.shape[0] * h.shape[1], h.shape[2] * h.shape[3])

        if eig_vectors:
            w, v = np.linalg.eigh(h)
            evals_list.append(w.real)
            evecs_list.append(v)
        else:
            w = np.linalg.eigvalsh(h)
            evals_list.append(w.real)

    evals = np.stack(evals_list, axis=1)  # (nband, nk)
    if eig_vectors:
        evecs = np.stack(evecs_list, axis=1)  # (nband, nk, nband)
        return evals, evecs
    return evals


# ---------------------------------------------------------------------------
# 2. Optimized Multi-Process + Numba Solver
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=False)
def _phases_from_k_and_r(k_arr, r_arr):
    nk = k_arr.shape[0]
    nh = r_arr.shape[0]
    out = np.empty((nk, nh), dtype=np.complex128)
    twopi = 2.0 * np.pi
    for ik in prange(nk):
        for ih in range(nh):
            x = 0.0
            for d in range(k_arr.shape[1]):
                x += k_arr[ik, d] * r_arr[ih, d]
            ang = twopi * x
            out[ik, ih] = np.cos(ang) + 1j * np.sin(ang)
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _assemble_h_all_k_numba(phases, onsite_flat, hop_i, hop_j, hop_amp_flat, nband):
    nk = phases.shape[0]
    nh = phases.shape[1]
    h_all = np.zeros((nk, nband, nband), dtype=np.complex128)

    for ik in prange(nk):
        for a in range(nband):
            for b in range(nband):
                h_all[ik, a, b] = onsite_flat[a, b]

        for ih in range(nh):
            p = phases[ik, ih]
            pc = np.conjugate(p)
            i0 = hop_i[ih]
            j0 = hop_j[ih]

            for s1 in range(2):
                for s2 in range(2):
                    a = 2 * i0 + s1
                    b = 2 * j0 + s2
                    t = hop_amp_flat[ih, a, b]
                    h_all[ik, a, b] += p * t

                    # Correct Hermitian conjugation: (T_ij)^\dagger = (T_ij)^T*
                    a_conj = 2 * j0 + s1
                    b_conj = 2 * i0 + s2
                    t_conj = np.conjugate(hop_amp_flat[ih, 2 * i0 + s2, 2 * j0 + s1])
                    h_all[ik, a_conj, b_conj] += pc * t_conj

    return h_all


def _prepare_tb_arrays_for_numba(tb_rot):
    norb = tb_rot._norb
    nband = 2 * norb
    dim_k = tb_rot._dim_k

    onsite_flat = np.zeros((nband, nband), dtype=np.complex128)
    for i in range(norb):
        eps = np.asarray(tb_rot._site_energies[i], dtype=np.complex128)
        onsite_flat[2*i:2*i+2, 2*i:2*i+2] = eps

    per = np.array(tb_rot._per, dtype=np.int64) if dim_k > 0 else np.zeros((0,), dtype=np.int64)

    nhop = len(tb_rot._hoppings)
    hop_i = np.zeros(nhop, dtype=np.int64)
    hop_j = np.zeros(nhop, dtype=np.int64)
    hop_amp_flat = np.zeros((nhop, nband, nband), dtype=np.complex128)
    r_arr = np.zeros((nhop, dim_k), dtype=np.float64)

    for ih, h in enumerate(tb_rot._hoppings):
        amp2 = np.asarray(h[0], dtype=np.complex128)
        i = int(h[1])
        j = int(h[2])

        hop_i[ih] = i
        hop_j[ih] = j
        hop_amp_flat[ih, 2*i:2*i+2, 2*j:2*j+2] = amp2

        if dim_k > 0:
            rv_full = -tb_rot._orb[i, :] + tb_rot._orb[j, :] + np.asarray(h[3], dtype=np.float64)
            rv_k = rv_full[per] if len(per) > 0 else np.zeros((dim_k,), dtype=np.float64)
            r_arr[ih, :] = rv_k

    return onsite_flat, hop_i, hop_j, hop_amp_flat, r_arr, nband


def _solve_single_h(h, eig_vectors=True):
    if eig_vectors:
        w, v = np.linalg.eigh(h)
        return w.real, v
    else:
        w = np.linalg.eigvalsh(h)
        return w.real


def compute_rotated_bands_parallel(
    tb_up,
    tb_dn,
    k_vec,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
    eig_vectors=True,
    n_jobs=-1
):
    """
    Optimized solver combining Numba fast matrix assembly with multi-process
    parallel diagonalization across k-points using joblib.
    """
    tb_spin = build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=fermi_level
    )
    tb_rot = apply_local_spin_rotation_from_cellwf_samples(
        tb_spin, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
    )

    k_arr = np.asarray(k_vec, dtype=float)
    if k_arr.ndim == 1:
        k_arr = k_arr.reshape(-1, 1)

    onsite_flat, hop_i, hop_j, hop_amp_flat, r_arr, nband = _prepare_tb_arrays_for_numba(tb_rot)
    phases = _phases_from_k_and_r(k_arr, r_arr)
    h_all = _assemble_h_all_k_numba(phases, onsite_flat, hop_i, hop_j, hop_amp_flat, nband)

    nk = h_all.shape[0]

    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_solve_single_h)(h_all[ik], eig_vectors=eig_vectors)
            for ik in range(nk)
        )

    if eig_vectors:
        evals = np.stack([r[0] for r in results], axis=1)  # (nband, nk)
        evecs = np.stack([r[1] for r in results], axis=1)  # (nband, nk, nband)
        return evals, evecs

    evals = np.stack(results, axis=1)  # (nband, nk)
    return evals


# Backwards compatibility alias
compute_rotated_bands_from_cellwf_samples_alt = compute_rotated_bands_parallel


# ---------------------------------------------------------------------------
# 3. Density of States (DOS) & Projected DOS (PDOS) Helper
# ---------------------------------------------------------------------------

def compute_total_and_projected_dos(
    evals,
    evecs,
    kpoints,
    projector=None,
    energy_grid=None,
    e_min=None,
    e_max=None,
    n_energy=2000,
    sigma=0.02,
    k_weights=None,
    return_components=False,
):
    """
    Compute total DOS and projected DOS (PDOS) from band eigenvalues and eigenvectors.
    """
    evals = np.asarray(evals, dtype=np.float64)
    kp = np.asarray(kpoints)

    if evals.ndim != 2:
        raise ValueError("evals must be 2D: (n_band,n_k) or (n_k,n_band).")

    nk_from_k = kp.shape[0] if kp.ndim >= 1 else None
    if nk_from_k is None:
        raise ValueError("could not infer n_k from kpoints.")

    if evals.shape[1] == nk_from_k:
        e_bn = evals
    elif evals.shape[0] == nk_from_k:
        e_bn = evals.T
    else:
        raise ValueError(f"evals shape {evals.shape} incompatible with n_k={nk_from_k}.")

    n_band, n_k = e_bn.shape

    v = np.asarray(evecs)
    if v.ndim != 3:
        raise ValueError("evecs must be 3D.")

    if v.shape[0] == n_band and v.shape[1] == n_k:
        v_bkn = v
    elif v.shape[0] == n_k and v.shape[2] == n_band:
        v_bkn = np.transpose(v, (2, 0, 1))
    else:
        raise ValueError(f"unsupported evecs shape {v.shape}.")

    n_basis = v_bkn.shape[2]

    if k_weights is None:
        wk = np.full(n_k, 1.0 / n_k, dtype=np.float64)
    else:
        wk = np.asarray(k_weights, dtype=np.float64).reshape(-1)
        if wk.shape[0] != n_k:
            raise ValueError("k_weights must have length n_k.")
        s = wk.sum()
        if s <= 0:
            raise ValueError("k_weights sum must be positive.")
        wk = wk / s

    if energy_grid is not None:
        energy = np.asarray(energy_grid, dtype=np.float64).reshape(-1)
        if energy.size < 2:
            raise ValueError("energy_grid must have at least 2 points.")
    else:
        emin_data = float(np.min(e_bn))
        emax_data = float(np.max(e_bn))
        if e_min is None:
            e_min = emin_data - 5.0 * sigma
        if e_max is None:
            e_max = emax_data + 5.0 * sigma
        energy = np.linspace(float(e_min), float(e_max), int(n_energy), dtype=np.float64)

    proj_mode = "identity"
    proj_indices = None
    proj_mask = None
    proj_matrix = None

    if projector is None:
        proj_mode = "identity"

    elif isinstance(projector, dict):
        if "matrix" in projector and projector["matrix"] is not None:
            proj_mode = "matrix"
            proj_matrix = np.asarray(projector["matrix"], dtype=np.complex128)
        elif "mask" in projector and projector["mask"] is not None:
            proj_mode = "mask"
            proj_mask = np.asarray(projector["mask"], dtype=bool).reshape(-1)
        elif "indices" in projector and projector["indices"] is not None:
            proj_mode = "indices"
            proj_indices = np.asarray(projector["indices"], dtype=np.int64).reshape(-1)
        else:
            proj_mode = "identity"

    else:
        arr = np.asarray(projector)
        if arr.ndim == 2:
            proj_mode = "matrix"
            proj_matrix = np.asarray(arr, dtype=np.complex128)
        elif arr.ndim == 1 and arr.dtype == bool:
            proj_mode = "mask"
            proj_mask = arr.astype(bool)
        elif arr.ndim == 1:
            proj_mode = "indices"
            proj_indices = arr.astype(np.int64)
        else:
            raise ValueError("projector format not recognized.")

    if proj_mode == "matrix":
        if proj_matrix.shape != (n_basis, n_basis):
            raise ValueError(f"projector matrix must be shape ({n_basis},{n_basis}).")
    elif proj_mode == "mask":
        if proj_mask.shape[0] != n_basis:
            raise ValueError(f"projector mask must have length {n_basis}.")
        proj_indices = np.where(proj_mask)[0]
    elif proj_mode == "indices":
        if proj_indices.size == 0:
            proj_indices = np.array([], dtype=np.int64)
        if np.any(proj_indices < 0) or np.any(proj_indices >= n_basis):
            raise ValueError(f"projector indices must be in [0,{n_basis-1}].")

    weights = np.zeros((n_band, n_k), dtype=np.float64)

    if proj_mode == "identity":
        weights[:, :] = 1.0

    elif proj_mode in ("mask", "indices"):
        if proj_indices.size == 0:
            weights[:, :] = 0.0
        else:
            sub = v_bkn[:, :, proj_indices]
            weights = np.sum(np.abs(sub) ** 2, axis=2).real

    elif proj_mode == "matrix":
        tmp = np.einsum("bkc,ca->bka", np.conjugate(v_bkn), proj_matrix, optimize=True)
        weights = np.einsum("bka,bka->bk", tmp, v_bkn, optimize=True).real

    dos = np.zeros_like(energy, dtype=np.float64)
    pdos = np.zeros_like(energy, dtype=np.float64)

    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    for ik in range(n_k):
        wk_i = wk[ik]
        e_k = e_bn[:, ik]
        w_k = weights[:, ik]
        for ib in range(n_band):
            de = (energy - e_k[ib]) / sigma
            g = norm * np.exp(-0.5 * de * de)
            dos += wk_i * g
            pdos += wk_i * w_k[ib] * g

    out = {"energy": energy, "dos": dos, "pdos": pdos}
    if return_components:
        out["weights"] = weights
    return out