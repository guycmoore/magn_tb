import numpy as np
from numba import njit, prange
import copy
from pythtb import tb_model


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
