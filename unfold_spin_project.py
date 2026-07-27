import numpy as np

# ---------------------------
# Pauli matrices
# ---------------------------
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _build_sc_translation_list(sc_red_lat):
    """
    For diagonal integer supercell matrix diag(N1,N2,N3), return internal translations n:
      n = (n1,n2,n3), ni=0..Ni-1
    in the same order as pythtb.make_supercell for simple diagonal case.
    """
    S = np.asarray(sc_red_lat, dtype=int)
    if S.shape != (3, 3):
        raise ValueError("sc_red_lat must be 3x3")
    if not np.all(S == np.diag(np.diag(S))):
        raise NotImplementedError("Only diagonal supercells supported in this helper.")

    N1, N2, N3 = np.diag(S)
    out = []
    for n1 in range(N1):
        for n2 in range(N2):
            for n3 in range(N3):
                out.append(np.array([n1, n2, n3], dtype=int))
    return np.array(out, dtype=int)  # (Nsc,3)


def build_reference_spinful(tb_up, tb_dn, fermi_level=0.0):
    """
    Same convention as your code: spinful collinear model with diagonal spin blocks.
    """
    from tb_spinor import build_spinful_from_collinear_intersection
    return build_spinful_from_collinear_intersection(tb_up, tb_dn, fermi_level=fermi_level)


def eigensystem_at_k(tb_model_spinful, kpt):
    """
    Return eigenvalues and eigenvectors of full spinful Hamiltonian at kpt.
    evec columns are eigenvectors.
    """
    h = np.asarray(tb_model_spinful._gen_ham(np.asarray(kpt, dtype=float)), dtype=np.complex128)
    if h.ndim == 4:
        h = h.reshape(h.shape[0] * h.shape[1], h.shape[2] * h.shape[3])
    w, v = np.linalg.eigh(h)
    return w.real, v  # v: (nbasis, nbasis), columns


def projector_onto_ref_state_in_sc(ref_vec, k_p, k_s, r_int_list, base_norb):
    """
    Embed primitive-cell ref eigenvector into supercell basis at fixed unfolding sector
    via Bloch phase exp(i2pi (k_p-k_s)·R_int).

    ref_vec: (2*base_norb,)
    returns embedded vec phi in supercell basis (2*base_norb*Nsc,)
    """
    nq = len(r_int_list)
    nb_prim = 2 * base_norb
    nb_sc = nb_prim * nq
    phi = np.zeros(nb_sc, dtype=complex)

    dq = np.asarray(k_p) - np.asarray(k_s)  # should match n/N
    for iq, R in enumerate(r_int_list):
        phase = np.exp(2j * np.pi * np.dot(dq, R))
        i0 = iq * nb_prim
        phi[i0:i0 + nb_prim] = phase * ref_vec

    # normalize
    phi /= np.sqrt(nq)
    return phi


def spin_expectation_from_coeffs(coeff):
    """
    coeff: (...,2) spinor coefficients
    """
    c_up = coeff[..., 0]
    c_dn = coeff[..., 1]
    sx = 2.0 * np.real(np.conjugate(c_up) * c_dn)
    sy = 2.0 * np.imag(np.conjugate(c_dn) * c_up)
    sz = np.abs(c_up) ** 2 - np.abs(c_dn) ** 2
    return sx, sy, sz


def project_sc_bands_on_reference(
    tb_up,
    tb_dn,
    E_sc,
    V_sc,            # shape expected (nband, nk, nbasis) OR (nk, nbasis, nband)
    k_s_list,
    sc_red_lat,
    fermi_level=0.0,
    nb_ref_keep=None
):
    """
    Main routine:
      - For each k_s and each unfolding sector n -> k_p
      - Diagonalize reference model at k_p
      - Project each SC eigenstate onto each reference eigenstate
      - Return spectral weights and spin polarization of projected reference character

    Returns dict with:
      weights[J,ik,nq,iref]
      spin_xyz[J,ik,nq,iref,3]
      k_p[ik,nq,3]
      E_ref[ik,nq,iref]
    """
    tb_ref = build_reference_spinful(tb_up, tb_dn, fermi_level=fermi_level)

    k_s_arr = np.asarray(k_s_list, dtype=float)
    nk = k_s_arr.shape[0]

    # normalize V_sc to (nband, nk, nbasis) with columns convention
    V = np.asarray(V_sc)
    if V.ndim != 3:
        raise ValueError("V_sc must be 3D")
    if V.shape[1] == nk:
        # assume (nband,nk,nbasis): each [J,ik,:] is eigenvector components
        V_bkn = V
    elif V.shape[0] == nk:
        # maybe (nk,nbasis,nband): convert
        V_bkn = np.transpose(V, (2, 0, 1))
    else:
        raise ValueError("Unrecognized V_sc shape")

    nband, _, nbasis = V_bkn.shape
    base_norb = tb_up._norb

    r_int = _build_sc_translation_list(sc_red_lat)
    nq = len(r_int)

    if nb_ref_keep is None:
        nb_ref_keep = 2 * base_norb

    weights = np.zeros((nband, nk, nq, nb_ref_keep), dtype=float)
    spin_xyz = np.zeros((nband, nk, nq, nb_ref_keep, 3), dtype=float)
    E_ref = np.zeros((nk, nq, nb_ref_keep), dtype=float)
    k_p_all = np.zeros((nk, nq, 3), dtype=float)

    # unfolding vectors q = n/N
    S = np.asarray(sc_red_lat, dtype=int)
    N1, N2, N3 = np.diag(S)
    q_list = np.array([[n1 / N1, n2 / N2, n3 / N3] for n1, n2, n3 in r_int], dtype=float)

    for ik in range(nk):
        k_s = k_s_arr[ik]
        psi_sc = V_bkn[:, ik, :]  # (J,nbasis)

        for iq in range(nq):
            q = q_list[iq]
            k_p = k_s + q
            k_p_all[ik, iq] = k_p

            e_ref, v_ref = eigensystem_at_k(tb_ref, k_p)  # v_ref columns
            E_ref[ik, iq, :] = e_ref[:nb_ref_keep]

            for ir in range(nb_ref_keep):
                ref_vec = v_ref[:, ir]  # primitive basis
                phi = projector_onto_ref_state_in_sc(ref_vec, k_p, k_s, r_int, base_norb)  # embedded

                # overlaps for all SC bands J
                ov = psi_sc @ np.conjugate(phi)  # (J,)
                w = np.abs(ov) ** 2
                weights[:, ik, iq, ir] = w.real

                # spin of projected component: use overlap amplitude * ref spin expectation
                # (simple and robust approximation)
                ref_spin = np.zeros(3, dtype=float)
                ref_orb_spin = ref_vec.reshape(base_norb, 2)
                sx = np.sum(np.conjugate(ref_orb_spin) @ SIGMA_X * ref_orb_spin).real
                sy = np.sum(np.conjugate(ref_orb_spin) @ SIGMA_Y * ref_orb_spin).real
                sz = np.sum(np.conjugate(ref_orb_spin) @ SIGMA_Z * ref_orb_spin).real
                ref_spin[:] = [sx, sy, sz]

                spin_xyz[:, ik, iq, ir, :] = w[:, None] * ref_spin[None, :]

    return {
        "weights": weights,
        "spin_xyz": spin_xyz,
        "k_p": k_p_all,
        "E_ref": E_ref,
    }


def rgb_from_spin(spin_vec, weight, eps=1e-12):
    """
    Map spin vector (sx,sy,sz) to RGB by |component| mixing:
      R~|sx|, G~|sy|, B~|sz|, scaled by weight.
    """
    s = np.asarray(spin_vec, dtype=float)
    mag = np.linalg.norm(s, axis=-1, keepdims=True)
    unit = s / np.maximum(mag, eps)
    rgb = np.abs(unit)
    return np.clip(weight[..., None] * rgb, 0.0, 1.0)

