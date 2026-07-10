import numpy as np
from numba import njit, prange
import copy
from pythtb import tb_model

from numba import njit, prange
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False

def _as_unitary(u, tol=1e-8):
    u = np.asarray(u, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError(f"u must have shape (2,2), got {u.shape}")
    err = np.max(np.abs(u.conj().T @ u - np.eye(2, dtype=complex)))
    if err > tol:
        raise ValueError(f"u not unitary: max ||u†u - I|| = {err}")
    return u

# FIXME: Low tolerance by Wannier centers.
def _check_compatible_spinless_models(tb_a, tb_b, tol=1e-2):
    if tb_a._nspin != 1 or tb_b._nspin != 1:
        raise ValueError("expected both inputs to have nspin=1 (spinless).")

    for attr in ["_dim_k", "_dim_r", "_norb", "_per"]:
        if getattr(tb_a, attr) != getattr(tb_b, attr):
            raise ValueError(f"models not compatible: tb_a.{attr} != tb_b.{attr}")

    if np.max(np.abs(tb_a._lat - tb_b._lat)) > tol:
        raise ValueError("lattice vectors differ between tb models.")
    if np.max(np.abs(tb_a._orb - tb_b._orb)) > tol:
        raise ValueError("orbital positions differ between tb models.")

    # hopping connectivity check by keys (i,j,r)
    def hop_key(h, tb):
        i = int(h[1])
        j = int(h[2])
        if tb._dim_k == 0:
            return (i, j)
        r = np.array(h[3], dtype=int)
        return (i, j, tuple(r.tolist()))

    keys_a = [hop_key(h, tb_a) for h in tb_a._hoppings]
    keys_b = [hop_key(h, tb_b) for h in tb_b._hoppings]
    if sorted(keys_a) != sorted(keys_b):
        raise ValueError("hopping connectivity differs between tb_up and tb_dn.")


def build_spinful_from_collinear(tb_up, tb_dn, fermi_level=0.0):
    """
    builds nspin=2 spinor model with diagonal spin blocks:
      onsite: [[e_up,0],[0,e_dn]]
      hop:    [[t_up,0],[0,t_dn]]
    """
    _check_compatible_spinless_models(tb_up, tb_dn)

    dim_k = tb_up._dim_k
    dim_r = tb_up._dim_r
    lat = tb_up.get_lat()
    orb = tb_up.get_orb()
    per = copy.deepcopy(tb_up._per)
    norb = tb_up.get_num_orbitals()

    tb_spin = tb_model(dim_k, dim_r, lat=lat, orb=orb, per=per, nspin=2)

    onsite = []
    for i in range(norb):
        e_up = tb_up._site_energies[i] - fermi_level
        e_dn = tb_dn._site_energies[i] - fermi_level
        onsite.append(np.array([[complex(e_up), 0.0],
                                 [0.0, complex(e_dn)]], dtype=complex))
    tb_spin.set_onsite(onsite)

    # map tb_dn hoppings by key
    def hop_key(h):
        i = int(h[1])
        j = int(h[2])
        if dim_k == 0:
            return (i, j)
        r = np.array(h[3], dtype=int)
        return (i, j, tuple(r.tolist()))

    dn_map = {hop_key(h): h[0] for h in tb_dn._hoppings}

    for h in tb_up._hoppings:
        key = hop_key(h)
        amp_up = complex(h[0])
        amp_dn = complex(dn_map[key])

        amp_spin = np.array([[amp_up, 0.0],
                              [0.0, amp_dn]], dtype=complex)

        i = int(h[1])
        j = int(h[2])
        if dim_k == 0:
            tb_spin.set_hop(amp_spin, i, j, mode="set")
        else:
            ind_r = np.array(h[3], dtype=int)
            tb_spin.set_hop(amp_spin, i, j, ind_R=ind_r, mode="set")

    return tb_spin

def build_spinful_from_collinear_intersection(tb_up, tb_dn, fermi_level=0.0):
    """
    builds nspin=2 spinor model with diagonal spin blocks:
      onsite: [[e_up,0],[0,e_dn]]
      hop:    [[t_up,0],[0,t_dn]]

    Uses ONLY the INTERSECTION of hopping keys (i,j,R) present in both tb_up and tb_dn.
    Hoppings not present in both are excluded.
    """
    # --- geometry compatibility (but DO NOT check hopping connectivity equality) ---
    if tb_up._nspin != 1 or tb_dn._nspin != 1:
        raise ValueError("expected both inputs to have nspin=1 (spinless).")

    for attr in ["_dim_k", "_dim_r", "_norb", "_per"]:
        if getattr(tb_up, attr) != getattr(tb_dn, attr):
            raise ValueError(f"models not compatible: tb_a.{attr} != tb_b.{attr}")

    # lattice vectors and orbital positions must match for this approach
    tol = 1e-2 # FIXME
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

    # --- onsite ---
    onsite = []
    for i in range(norb):
        e_up = tb_up._site_energies[i] - fermi_level
        e_dn = tb_dn._site_energies[i] - fermi_level
        onsite.append(
            np.array([[complex(e_up), 0.0],
                      [0.0, complex(e_dn)]], dtype=complex)
        )
    tb_spin.set_onsite(onsite)

    # --- hopping maps ---
    def hop_key(h, tb):
        i = int(h[1])
        j = int(h[2])
        if tb._dim_k == 0:
            return (i, j)
        r = np.array(h[3], dtype=int)
        return (i, j, tuple(r.tolist()))

    up_map = {}
    for h in tb_up._hoppings:
        k = hop_key(h, tb_up)
        up_map[k] = complex(h[0])

    dn_map = {}
    for h in tb_dn._hoppings:
        k = hop_key(h, tb_dn)
        dn_map[k] = complex(h[0])

    # --- intersection keys only ---
    common_keys = set(up_map.keys()) & set(dn_map.keys())
    if len(common_keys) == 0:
        raise ValueError("No common hopping keys found between tb_up and tb_dn.")

    # --- set hoppings for common keys only ---
    for key in common_keys:
        if dim_k == 0:
            i, j = key
            amp_up = up_map[key]
            amp_dn = dn_map[key]
            amp_spin = np.array([[amp_up, 0.0],
                                  [0.0, amp_dn]], dtype=complex)
            tb_spin.set_hop(amp_spin, i, j, mode="set")
        else:
            i, j, r_tup = key
            r = np.array(r_tup, dtype=int)
            amp_up = up_map[key]
            amp_dn = dn_map[key]
            amp_spin = np.array([[amp_up, 0.0],
                                  [0.0, amp_dn]], dtype=complex)
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
    assigns u_i per orbital i using:
      u_samples[(cell_r_int_tuple, wf_index)]  -> 2x2 unitary spinor matrix

    rotation convention (your choice: u rotates the magnetization field):
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

    # build u_list for each orbital in sc_tb
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

    # build rotated model with identical geometry
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
        amp = np.array(h[0], dtype=complex)  # 2x2
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


def compute_rotated_bands_from_cellwf_samples(
    tb_up,
    tb_dn,
    k_vec,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
):
    """
    end-to-end:
      tb_up/tb_dn (nspin=1) -> spinful tb_spinful (nspin=2)
      apply u_samples rotation (optionally on a supercell)
      diagonalize on k_vec
    """
    # tb_spin = build_spinful_from_collinear(tb_up, tb_dn, fermi_level=fermi_level)
    tb_spin = build_spinful_from_collinear_intersection(tb_up, tb_dn, fermi_level=fermi_level)
    
    tb_rot = apply_local_spin_rotation_from_cellwf_samples(
        tb_spin, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
    )
    e = tb_rot.solve_all(k_vec)  # (n_bands, n_k)
    return e


def compute_rotated_bands_from_cellwf_samples_alt(
    tb_up,
    tb_dn,
    k_vec,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
    eig_vectors=False,
    eig_backend="parallel",          # "parallel" or "stacked"
    n_jobs=-1,
    limit_blas_threads=True,
    blas_threads=1,
):
    """
    Alternative to compute_rotated_bands_from_cellwf_samples with optional eigenvectors.

    returns
    -------
    evals : ndarray, shape (n_bands, n_k)
    evecs : ndarray, shape (n_bands, n_k, n_bands), optional
        returned only if eig_vectors=True
    """
    tb_spin = build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=fermi_level
    )

    tb_rot = apply_local_spin_rotation_from_cellwf_samples(
        tb_spin, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
    )

    k_arr = np.asarray(k_vec, dtype=float)
    if tb_rot._dim_k == 1:
        if k_arr.ndim == 1:
            k_arr = k_arr.reshape(-1, 1)
        elif not (k_arr.ndim == 2 and k_arr.shape[1] == 1):
            raise ValueError("for dim_k=1, k_vec must be shape (nk,) or (nk,1)")
    else:
        if k_arr.ndim != 2 or k_arr.shape[1] != tb_rot._dim_k:
            raise ValueError(f"k_vec must have shape (nk, {tb_rot._dim_k})")

    nk = k_arr.shape[0]

    ctx = threadpool_limits(limits=blas_threads) if limit_blas_threads else _nullcontext()
    with ctx:
        if eig_backend == "parallel":
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_solve_one_k)(tb_rot, k_arr[ik], eig_vectors=eig_vectors)
                for ik in range(nk)
            )
        elif eig_backend == "stacked":
            results = [_solve_one_k(tb_rot, k_arr[ik], eig_vectors=eig_vectors) for ik in range(nk)]
        else:
            raise ValueError("eig_backend must be 'parallel' or 'stacked'.")

    if eig_vectors:
        evals = np.stack([r[0] for r in results], axis=1)  # (nb, nk)
        evecs = np.stack([r[1] for r in results], axis=1)  # (nb, nk, nb)
        return evals, evecs

    evals = np.stack(results, axis=1)  # (nb, nk)
    return evals

def _solve_one_k(tb_model_spinful, kpt, eig_vectors=False):
    kpt = np.asarray(kpt, dtype=float)
    h = np.asarray(tb_model_spinful._gen_ham(kpt), dtype=np.complex128)

    # pythtb nspin=2 may return (norb,2,norb,2); flatten to (2*norb,2*norb)
    if h.ndim == 4:
        h = h.reshape(h.shape[0] * h.shape[1], h.shape[2] * h.shape[3])

    if eig_vectors:
        w, v = np.linalg.eigh(h)
        return w.real, v
    else:
        w = np.linalg.eigvalsh(h)
        return w.real
    
##

# ---------------- numba kernels ----------------

@njit(parallel=True, cache=True, fastmath=False)
def _phases_from_k_and_r(k_arr, r_arr):
    """
    phase[ik, ih] = exp(2π i k·r)
    k_arr: (nk, dim_k)
    r_arr: (nhop, dim_k)  integer lattice vectors for periodic directions only
    """
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
def _assemble_h_all_k_numba(
    phases,                # (nk, nhop)
    onsite_flat,           # (nband, nband)
    hop_i, hop_j,          # (nhop,)
    hop_amp_flat,          # (nhop, nband, nband)
    nband,
):
    """
    Build H(k) for all k:
      H(k) = onsite + sum_h [ phase_h * T_h(i,j) + conj(phase_h) * T_h(i,j)† ]
    where T_h contributes to block (i,j) only.
    """
    nk = phases.shape[0]
    nh = phases.shape[1]
    h_all = np.zeros((nk, nband, nband), dtype=np.complex128)

    for ik in prange(nk):
        # onsite
        for a in range(nband):
            for b in range(nband):
                h_all[ik, a, b] = onsite_flat[a, b]

        # hoppings (+ h.c.)
        for ih in range(nh):
            p = phases[ik, ih]
            pc = np.conjugate(p)
            i0 = hop_i[ih]
            j0 = hop_j[ih]

            # add p * T to block (i0,j0)
            for s1 in range(2):
                for s2 in range(2):
                    a = 2 * i0 + s1
                    b = 2 * j0 + s2
                    t = hop_amp_flat[ih, a, b]
                    h_all[ik, a, b] += p * t
                    # add h.c. to (j0,i0)
                    h_all[ik, b, a] += pc * np.conjugate(t)

    return h_all


# ---------------- prep helpers ----------------

def _prepare_tb_arrays_for_numba(tb_rot):
    """
    Convert pythtb model internals into compact arrays for numba assembly.
    Assumes nspin=2.
    """
    if tb_rot._nspin != 2:
        raise ValueError("tb_rot must have nspin=2")

    norb = tb_rot._norb
    nband = 2 * norb
    dim_k = tb_rot._dim_k

    # onsite
    onsite_flat = np.zeros((nband, nband), dtype=np.complex128)
    for i in range(norb):
        eps = np.asarray(tb_rot._site_energies[i], dtype=np.complex128)
        # eps is 2x2
        onsite_flat[2*i:2*i+2, 2*i:2*i+2] = eps

    # periodic direction selection for k·R
    per = np.array(tb_rot._per, dtype=np.int64) if dim_k > 0 else np.zeros((0,), dtype=np.int64)

    # hoppings
    nhop = len(tb_rot._hoppings)
    hop_i = np.zeros(nhop, dtype=np.int64)
    hop_j = np.zeros(nhop, dtype=np.int64)
    hop_amp_flat = np.zeros((nhop, nband, nband), dtype=np.complex128)
    r_arr = np.zeros((nhop, dim_k), dtype=np.float64)

    for ih, h in enumerate(tb_rot._hoppings):
        amp2 = np.asarray(h[0], dtype=np.complex128)   # 2x2
        i = int(h[1])
        j = int(h[2])

        hop_i[ih] = i
        hop_j[ih] = j

        # place only the (i,j) 2x2 block in full flattened matrix
        hop_amp_flat[ih, 2*i:2*i+2, 2*j:2*j+2] = amp2

        if dim_k > 0:
            rv_full = np.asarray(h[3], dtype=np.float64)  # length dim_r
            rv_k = rv_full[per] if len(per) > 0 else np.zeros((dim_k,), dtype=np.float64)
            r_arr[ih, :] = rv_k

    return onsite_flat, hop_i, hop_j, hop_amp_flat, r_arr, nband


# ---------------- main optimized function ----------------

def compute_rotated_bands_from_cellwf_samples_numba(
    tb_up,
    tb_dn,
    k_vec,
    u_samples,
    sc_red_lat=None,
    to_home=True,
    fermi_level=0.0,
    eig_vectors=False,
    numba_threads=None,          # None -> keep current numba thread setting
    limit_blas_threads=True,
    blas_threads=1,
):
    """
    Numba-optimized alternative:
      1) build rotated tb model
      2) numba-parallel assemble H(k) for all k
      3) diagonalize each H(k)

    returns
    -------
    evals : (n_bands, nk)
    evecs : (n_bands, nk, n_bands)  [if eig_vectors=True]
    """
    from numba import set_num_threads

    # build rotated model (same physics path as your original)
    tb_spin = build_spinful_from_collinear_intersection(
        tb_up, tb_dn, fermi_level=fermi_level
    )
    tb_rot = apply_local_spin_rotation_from_cellwf_samples(
        tb_spin, u_samples=u_samples, sc_red_lat=sc_red_lat, to_home=to_home
    )

    dim_k = tb_rot._dim_k
    k_arr = np.asarray(k_vec, dtype=np.float64)
    if dim_k == 1:
        if k_arr.ndim == 1:
            k_arr = k_arr.reshape(-1, 1)
        elif not (k_arr.ndim == 2 and k_arr.shape[1] == 1):
            raise ValueError("for dim_k=1, k_vec must be shape (nk,) or (nk,1)")
    else:
        if k_arr.ndim != 2 or k_arr.shape[1] != dim_k:
            raise ValueError(f"k_vec must have shape (nk, {dim_k})")

    if numba_threads is not None:
        set_num_threads(int(numba_threads))

    onsite_flat, hop_i, hop_j, hop_amp_flat, r_arr, nband = _prepare_tb_arrays_for_numba(tb_rot)

    # assemble all H(k) with numba parallelization over k
    if dim_k > 0 and r_arr.shape[0] > 0:
        phases = _phases_from_k_and_r(k_arr, r_arr)
    else:
        # dim_k==0 case or no hoppings: phase = 1
        phases = np.ones((k_arr.shape[0], r_arr.shape[0]), dtype=np.complex128)

    h_all = _assemble_h_all_k_numba(
        phases,
        onsite_flat,
        hop_i, hop_j,
        hop_amp_flat,
        nband,
    )

    nk = h_all.shape[0]
    evals = np.zeros((nband, nk), dtype=np.float64)
    evecs = np.zeros((nband, nk, nband), dtype=np.complex128) if eig_vectors else None

    # control BLAS threads during eigh loop
    ctx = threadpool_limits(limits=blas_threads) if limit_blas_threads else _nullcontext()
    with ctx:
        for ik in range(nk):
            w, v = np.linalg.eigh(h_all[ik])
            evals[:, ik] = w.real
            if eig_vectors:
                evecs[:, ik, :] = v

    if eig_vectors:
        return evals, evecs
    return evals

## 

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
    Compute total DOS and projected DOS (PDOS) from eigensystem data using Gaussian broadening.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (n_band, n_k) or (n_k, n_band).
    evecs : ndarray
        Eigenvectors in the same basis as your TB Hamiltonian.
        Supported shapes:
          - (n_band, n_k, n_basis)
          - (n_k, n_basis, n_band)
        Assumes columns are eigenvectors if matrix-like per k.
    kpoints : ndarray
        k-point list, shape (n_k,) or (n_k, dim_k). Used for nk consistency only.
    projector : None, array-like, or dict
        General projector definition (kept in this main function by design):
          - None:
              PDOS == total DOS (identity projector).
          - 1D list/array of basis indices:
              project onto those basis states.
          - bool mask, shape (n_basis,):
              True entries are selected basis states.
          - 2D matrix P, shape (n_basis, n_basis):
              Hermitian projector/weight operator used as <psi|P|psi>.
          - dict:
              {
                "indices": [...],      # optional
                "mask": bool_array,    # optional
                "matrix": P            # optional
              }
              Priority: matrix > mask > indices.
    energy_grid : ndarray or None
        If provided, use this grid directly.
    e_min, e_max : float or None
        Energy window (used if energy_grid is None). If None, inferred from eval range ± 5*sigma.
    n_energy : int
        Number of energy points if energy_grid is None.
    sigma : float
        Gaussian broadening width (same energy unit as evals).
    k_weights : ndarray or None
        Optional k-point weights, shape (n_k,). If None, uniform weights.
        Will be normalized to sum to 1.
    return_components : bool
        If True, also return per-(band,k) projection weights.

    Returns
    -------
    out : dict
        {
          "energy":   (nE,),
          "dos":      (nE,),
          "pdos":     (nE,),
          "weights":  (n_band, n_k)   # only if return_components=True
        }

    Notes
    -----
    - DOS normalization here is such that integral over E gives number of bands (for uniform normalized k weights).
    - PDOS integral gives average projected weight summed over bands.
    """
    # ----------------------------
    # 0) shape normalization
    # ----------------------------
    evals = np.asarray(evals, dtype=np.float64)
    kp = np.asarray(kpoints)

    if evals.ndim != 2:
        raise ValueError("evals must be 2D: (n_band,n_k) or (n_k,n_band).")

    # infer nk from kpoints
    nk_from_k = kp.shape[0] if kp.ndim >= 1 else None
    if nk_from_k is None:
        raise ValueError("could not infer n_k from kpoints.")

    # make evals -> (n_band, n_k)
    if evals.shape[1] == nk_from_k:
        e_bn = evals
    elif evals.shape[0] == nk_from_k:
        e_bn = evals.T
    else:
        raise ValueError(f"evals shape {evals.shape} incompatible with n_k={nk_from_k} from kpoints.")

    n_band, n_k = e_bn.shape

    # evecs normalization to (n_band, n_k, n_basis)
    v = np.asarray(evecs)
    if v.ndim != 3:
        raise ValueError("evecs must be 3D.")

    # case A: (n_band, n_k, n_basis)
    if v.shape[0] == n_band and v.shape[1] == n_k:
        v_bkn = v
    # case B: (n_k, n_basis, n_band) -> transpose to (n_band, n_k, n_basis)
    elif v.shape[0] == n_k and v.shape[2] == n_band:
        v_bkn = np.transpose(v, (2, 0, 1))
    else:
        raise ValueError(
            f"unsupported evecs shape {v.shape}. "
            f"Expected (n_band,n_k,n_basis) or (n_k,n_basis,n_band)."
        )

    n_basis = v_bkn.shape[2]

    # ----------------------------
    # 1) k weights
    # ----------------------------
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

    # ----------------------------
    # 2) build energy grid
    # ----------------------------
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

    # ----------------------------
    # 3) projector handling (in main function, as requested)
    # ----------------------------
    # default identity projector -> PDOS == DOS
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

    # ----------------------------
    # 4) per-state projection weights w_{b,k}
    # ----------------------------
    weights = np.zeros((n_band, n_k), dtype=np.float64)

    if proj_mode == "identity":
        weights[:, :] = 1.0

    elif proj_mode in ("mask", "indices"):
        # w = sum_{a in subset} |c_a|^2
        if proj_indices.size == 0:
            weights[:, :] = 0.0
        else:
            sub = v_bkn[:, :, proj_indices]               # (b,k,nsel)
            weights = np.sum(np.abs(sub) ** 2, axis=2).real

    elif proj_mode == "matrix":
        # w = <psi|P|psi> for each (b,k)
        # vectorized over (b,k):
        # tmp_{b,k,a} = sum_c v*_{b,k,c} P_{c,a}
        tmp = np.einsum("bkc,ca->bka", np.conjugate(v_bkn), proj_matrix, optimize=True)
        weights = np.einsum("bka,bka->bk", tmp, v_bkn, optimize=True).real

    # ----------------------------
    # 5) Gaussian DOS accumulation
    # ----------------------------
    dos = np.zeros_like(energy, dtype=np.float64)
    pdos = np.zeros_like(energy, dtype=np.float64)

    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    # loop over k then bands for locality
    for ik in range(n_k):
        wk_i = wk[ik]
        e_k = e_bn[:, ik]             # (n_band,)
        w_k = weights[:, ik]          # (n_band,)
        # accumulate each band Gaussian on grid
        for ib in range(n_band):
            de = (energy - e_k[ib]) / sigma
            g = norm * np.exp(-0.5 * de * de)
            dos += wk_i * g
            pdos += wk_i * w_k[ib] * g

    out = {"energy": energy, "dos": dos, "pdos": pdos}
    if return_components:
        out["weights"] = weights
    return out