import numpy as np


# -----------------------------
# Pauli matrices (spin-1/2)
# -----------------------------
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _su2_from_ref_to_n_single(sref, n, eps=1e-12):
    """
    SU(2) spinor rotation u such that:
        u sigma_z u^dagger = (n·sigma)  if sref == +z-hat
    More generally:
        if sref is the reference unit vector, u aligns sref -> n in SO(3).

    We build it from axis-angle with:
        axis = sref x n
        cos(theta) = sref · n
        u = cos(theta/2) I - i sin(theta/2) (axis_hat · sigma)

    Returns:
      u: (2,2) unitary matrix
    """
    sref = np.asarray(sref, dtype=float)
    n = np.asarray(n, dtype=float)

    sref = sref / np.linalg.norm(sref)
    n = n / np.linalg.norm(n)

    c = float(np.dot(sref, n))          # cos(theta)
    a = np.cross(sref, n)
    a_norm = float(np.linalg.norm(a))   # sin(theta)

    # parallel: theta ~ 0
    if a_norm < eps and c > 0:
        return np.eye(2, dtype=complex)

    # antiparallel: theta ~ pi; axis ambiguous -> pick a deterministic perpendicular axis
    if a_norm < eps and c < 0:
        # choose a trial axis not parallel to sref
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(trial, sref)) > 1.0 - eps:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)

        axis = np.cross(sref, trial)
        axis = axis / np.linalg.norm(axis)
        theta = np.pi
    else:
        axis = a / a_norm
        theta = np.arctan2(a_norm, c)

    sigma = axis[0] * _SIGMA_X + axis[1] * _SIGMA_Y + axis[2] * _SIGMA_Z
    u = np.cos(theta / 2.0) * np.eye(2, dtype=complex) - 1.0j * np.sin(theta / 2.0) * sigma
    return u


def su2_from_ref_to_n(sref, n, eps=1e-12):
    """
    Vectorized wrapper.

    Args:
      sref: (3,) reference unit vector (e.g. [0,0,1])
      n: (..., 3) local unit vectors
    Returns:
      u: (..., 2, 2) SU(2) matrices
    """
    sref = np.asarray(sref, dtype=float)
    n = np.asarray(n, dtype=float)

    if n.shape[-1] != 3:
        raise ValueError(f"n must have last dim 3, got {n.shape}")

    out_shape = n.shape[:-1]
    u = np.zeros(out_shape + (2, 2), dtype=complex)

    it = np.nditer(np.empty(out_shape), flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        u[idx] = _su2_from_ref_to_n_single(sref, n[idx], eps=eps)
        it.iternext()

    return u


def su2_fix_sign_continuity(u_samples, order=None):
    """
    Fix the +/- sign gauge ambiguity for SU(2) rotations by enforcing continuity
    (choose sign so neighboring matrices have positive overlap).

    Args:
      u_samples: (N,2,2)
      order: optional array of indices specifying traversal order (default 0..N-1)

    Returns:
      u_fixed: (N,2,2)
    """
    u_samples = np.asarray(u_samples, dtype=complex)
    if u_samples.ndim != 3 or u_samples.shape[1:] != (2, 2):
        raise ValueError(f"u_samples must have shape (N,2,2), got {u_samples.shape}")

    npts = u_samples.shape[0]
    if order is None:
        order = np.arange(npts, dtype=int)
    else:
        order = np.asarray(order, dtype=int)

    u_fixed = u_samples.copy()

    for t in range(1, len(order)):
        i_prev = order[t - 1]
        i_cur = order[t]

        # overlap heuristic: if Re(tr(u_prev† u_cur)) < 0 flip sign
        overlap = np.real(np.trace(u_fixed[i_prev].conj().T @ u_fixed[i_cur]))
        if overlap < 0.0:
            u_fixed[i_cur] *= -1.0

    return u_fixed


def check_alignment(u_samples, n_samples):
    """
    Sanity check:
      For each point, compute u sigma_z u† and compare with (n·sigma)
    Assumes the construction uses sref = +z-hat. If you use a different sref,
    you’d need a different check.

    Args:
      u_samples: (N,2,2)
      n_samples: (N,3)
    """
    u_samples = np.asarray(u_samples, dtype=complex)
    n_samples = np.asarray(n_samples, dtype=float)

    if u_samples.shape[0] != n_samples.shape[0]:
        raise ValueError("u_samples and n_samples must match in number of points")

    sigma_z = _SIGMA_Z
    sigma_x = _SIGMA_X
    sigma_y = _SIGMA_Y

    udag = np.conjugate(np.transpose(u_samples, (0, 2, 1)))
    rot = u_samples @ sigma_z @ udag  # (N,2,2)

    # Extract components via traces: n_a = (1/2) Tr( (n·sigma) sigma_a )
    # since Tr(sigma_a sigma_b)=2 delta_ab
    nx = 0.5 * np.real(np.trace(rot @ sigma_x, axis1=1, axis2=2))
    ny = 0.5 * np.real(np.trace(rot @ sigma_y, axis1=1, axis2=2))
    nz = 0.5 * np.real(np.trace(rot @ sigma_z, axis1=1, axis2=2))

    n_check = np.stack([nx, ny, nz], axis=-1)
    return np.max(np.linalg.norm(n_check - n_samples, axis=-1))


def u_samples_to_dict(u_samples, cell_r_int, wf_index=0):
    """
    Convert (N,2,2) u_samples to a dict keyed by:
      (cell_r_int_tuple, wf_index)

    Args:
      u_samples: (N,2,2)
      cell_r_int: (N,dim_r) integers (supercell translation in reduced coords)
      wf_index: int or (N,) array

    Returns:
      dict mapping (tuple(cell), int(wf)) -> (2,2) complex
    """
    u_samples = np.asarray(u_samples, dtype=complex)
    cell_r_int = np.asarray(cell_r_int, dtype=int)

    if u_samples.ndim != 3 or u_samples.shape[1:] != (2, 2):
        raise ValueError(f"u_samples must be (N,2,2), got {u_samples.shape}")
    if cell_r_int.ndim != 2 or cell_r_int.shape[0] != u_samples.shape[0]:
        raise ValueError("cell_r_int must be (N,dim_r) with matching N")

    npts = u_samples.shape[0]

    if np.isscalar(wf_index):
        wf_index_arr = int(wf_index) * np.ones(npts, dtype=int)
    else:
        wf_index_arr = np.asarray(wf_index, dtype=int)
        if wf_index_arr.shape[0] != npts:
            raise ValueError("wf_index array length must match N")

    out = {}
    for i in range(npts):
        cell_tup = tuple(cell_r_int[i].tolist())
        out[(cell_tup, int(wf_index_arr[i]))] = u_samples[i]
    return out


def magnetization_texture_skyrmion_00(x, y, z=0.0, r0=1.0):
    """
    Analytic unit magnetization texture matching your notebook formula:

      mx = 2 x / (x^2 + y^2 + r0^2)
      my = 2 y / (x^2 + y^2 + r0^2)
      mz = (x^2 + y^2 - r0^2) / (x^2 + y^2 + r0^2)

    Returns:
      n: unit vector, shape (..., 3)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x * x + y * y
    denom = r2 + float(r0) ** 2

    mx = 2.0 * x / denom
    my = 2.0 * y / denom
    mz = (r2 - float(r0) ** 2) / denom

    n = np.stack([mx, my, mz], axis=-1)
    # numerical normalization safeguard
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def magnetization_collinear(x, y, z=0.0, r0=1.0, vec=(1.0, 0.0, 0.0)):
    """
    Uniform collinear magnetization along a user-supplied 3D unit vector `vec`.
    Default is +x ([1.0, 0.0, 0.0]).
    """
    x_arr = np.asarray(x, dtype=float)
    v = np.asarray(vec, dtype=float)
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm

    mx = np.full_like(x_arr, v[0])
    my = np.full_like(x_arr, v[1])
    mz = np.full_like(x_arr, v[2])
    return np.stack([mx, my, mz], axis=-1)


