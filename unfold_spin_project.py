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


def projector_onto_ref_state_in_sc(ref_vec, k_p, k_s, r_int_list, base_norb, u_cell_list=None):
    """
    Embed primitive-cell ref eigenvector into supercell basis at fixed unfolding sector
    via Bloch phase exp(i2pi (k_p-k_s)·R_int) and optional local spin rotation u_i per cell i.

    ref_vec: (2*base_norb,) primitive reference eigenvector
    u_cell_list: list of 2x2 complex matrices, length nq = len(r_int_list) (one per cell i in supercell)
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

        if u_cell_list is not None and u_cell_list[iq] is not None:
            ui = u_cell_list[iq]
            # apply ui to each 2-component orbital spinor in ref_vec
            ref_vec_loc = (ref_vec.reshape(base_norb, 2) @ ui.T).ravel()
            phi[i0:i0 + nb_prim] = phase * ref_vec_loc
        else:
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
    u_samples=None,  # dict mapping (cell_key, wf_idx) -> 2x2 unitary matrix (or (cell_key,) -> 2x2)
    fermi_level=0.0,
    nb_ref_keep=None
):
    """
    Main routine for local-frame spin-resolved band unfolding:
      - For each k_s and each unfolding sector n -> k_p
      - Diagonalize reference collinear model at k_p
      - Embed reference state into supercell with local cell spin frame u_i
      - Project supercell eigenstate onto local-frame reference state
      - Compute local adiabatic weights and lab/local frame spin expectations

    Returns dict with:
      weights[J,ik,nq,iref]       : local adiabatic projection weight
      spin_lab[J,ik,nq,iref,3]    : lab-frame 3D spin expectation
      spin_local[J,ik,nq,iref,3]  : local-frame 3D spin expectation
      k_p[ik,nq,3]                : primitive momenta
      E_ref[ik,nq,iref]           : primitive reference energies
    """
    tb_ref = build_reference_spinful(tb_up, tb_dn, fermi_level=fermi_level)

    k_s_arr = np.asarray(k_s_list, dtype=float)
    nk = k_s_arr.shape[0]

    # normalize V_sc to (nband, nk, nbasis) with columns convention
    V = np.asarray(V_sc)
    if V.ndim != 3:
        raise ValueError("V_sc must be 3D")
    
    if V.shape[0] == nk:
        # (nk, nbasis, nband) -> (nband, nk, nbasis)
        V_bkn = np.transpose(V, (2, 0, 1))
    elif V.shape[1] == nk:
        # Check if shape is (nband, nk, nbasis) vs (nbasis, nk, nband)
        # compute_rotated_bands_from_cellwf_samples_alt returns (nbasis, nk, nband)
        # where V_sc[:, ik, j] is the j-th eigenvector.
        # So transpose (2, 1, 0) gives (nband, nk, nbasis)
        V_bkn = np.transpose(V, (2, 1, 0))
    else:
        raise ValueError("Unrecognized V_sc shape")

    nband, _, nbasis = V_bkn.shape
    base_norb = tb_up._norb

    r_int = _build_sc_translation_list(sc_red_lat)
    nq = len(r_int)

    # extract u_cell_list if u_samples provided
    u_cell_list = None
    if u_samples is not None:
        u_cell_list = []
        for iq, R in enumerate(r_int):
            cell_key = tuple(int(x) for x in R)
            # check for cell_key in u_samples
            if (cell_key, 0) in u_samples:
                u_cell_list.append(u_samples[(cell_key, 0)])
            elif cell_key in u_samples:
                u_cell_list.append(u_samples[cell_key])
            else:
                u_cell_list.append(None)

    if nb_ref_keep is None:
        nb_ref_keep = 2 * base_norb

    weights = np.zeros((nband, nk, nq, nb_ref_keep), dtype=float)
    spin_lab = np.zeros((nband, nk, nq, nb_ref_keep, 3), dtype=float)
    spin_local = np.zeros((nband, nk, nq, nb_ref_keep, 3), dtype=float)
    E_ref = np.zeros((nk, nq, nb_ref_keep), dtype=float)
    k_p_all = np.zeros((nk, nq, 3), dtype=float)

    # unfolding vectors q = n/N
    S = np.asarray(sc_red_lat, dtype=int)
    N1, N2, N3 = np.diag(S)
    q_list = np.array([[n1 / N1, n2 / N2, n3 / N3] for n1, n2, n3 in r_int], dtype=float)
    r_int_arr = np.array(r_int, dtype=float)

    for ik in range(nk):
        k_s = k_s_arr[ik]
        psi_sc = V_bkn[:, ik, :]  # (nband, nbasis)
        psi_sc_3d = psi_sc.reshape(nband, nq, base_norb, 2)

        # precompute local-frame supercell wavefunctions if u_cell_list provided
        psi_loc_3d = np.zeros_like(psi_sc_3d)
        for i_cell in range(nq):
            if u_cell_list is not None and u_cell_list[i_cell] is not None:
                ui = u_cell_list[i_cell]
                psi_loc_3d[:, i_cell, :, :] = psi_sc_3d[:, i_cell, :, :] @ ui.conj()
            else:
                psi_loc_3d[:, i_cell, :, :] = psi_sc_3d[:, i_cell, :, :]

        for iq in range(nq):
            q = q_list[iq]
            k_p = k_s + q
            k_p_all[ik, iq] = k_p

            e_ref, v_ref = eigensystem_at_k(tb_ref, k_p)  # v_ref columns
            E_ref[ik, iq, :] = e_ref[:nb_ref_keep]

            # Fourier transform supercell wavefunctions into primitive sector q
            phase = np.exp(-2j * np.pi * (q @ r_int_arr.T)) / np.sqrt(nq) # (nq,)
            c_q_lab = np.einsum('i, Jimb -> Jmb', phase, psi_sc_3d)       # (J, base_norb, 2)
            c_q_loc = np.einsum('i, Jimb -> Jmb', phase, psi_loc_3d)      # (J, base_norb, 2)

            sx_lab = np.sum(2.0 * np.real(np.conj(c_q_lab[:, :, 0]) * c_q_lab[:, :, 1]), axis=1)
            sy_lab = np.sum(2.0 * np.imag(np.conj(c_q_lab[:, :, 1]) * c_q_lab[:, :, 0]), axis=1)
            sz_lab = np.sum(np.abs(c_q_lab[:, :, 0])**2 - np.abs(c_q_lab[:, :, 1])**2, axis=1)

            sx_loc = np.sum(2.0 * np.real(np.conj(c_q_loc[:, :, 0]) * c_q_loc[:, :, 1]), axis=1)
            sy_loc = np.sum(2.0 * np.imag(np.conj(c_q_loc[:, :, 1]) * c_q_loc[:, :, 0]), axis=1)
            sz_loc = np.sum(np.abs(c_q_loc[:, :, 0])**2 - np.abs(c_q_loc[:, :, 1])**2, axis=1)

            s_lab_vec = np.stack([sx_lab, sy_lab, sz_lab], axis=-1)   # (J, 3)
            s_loc_vec = np.stack([sx_loc, sy_loc, sz_loc], axis=-1)   # (J, 3)

            for ir in range(nb_ref_keep):
                ref_vec = v_ref[:, ir]  # primitive basis
                phi = projector_onto_ref_state_in_sc(ref_vec, k_p, k_s, r_int, base_norb, u_cell_list=u_cell_list)

                # overlaps for all SC bands J
                ov = psi_sc @ np.conjugate(phi)  # (J,)
                w = np.abs(ov) ** 2
                weights[:, ik, iq, ir] = w.real

                spin_lab[:, ik, iq, ir, :] = s_lab_vec
                spin_local[:, ik, iq, ir, :] = s_loc_vec

    return {
        "weights": weights,
        "spin_lab": spin_lab,
        "spin_local": spin_local,
        "spin_xyz": spin_lab,  # backwards compatibility alias
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


