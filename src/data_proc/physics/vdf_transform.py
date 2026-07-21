import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DEFAULT_HERMITE_ORDER = 22


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")

    return vector / norm


def build_rotation_matrix(b_field, bulk_velocity):
    """
    Build the (B, v_perp, B x v_perp) rotation matrix.

    Ports the frame construction from ``HERMITE::getRotatedVDF`` in
    https://github.com/ivanzait/vlasiator/blob/dev_rotation/hermite/vdf_tools.cpp,
    with the electric field replaced by the bulk flow velocity. Unlike a
    literal port, ``bulk_velocity`` is Gram-Schmidt orthogonalized against
    ``b_field`` before use: the C++ code builds its rotation matrix as the
    raw rows ``(B_hat, E_hat, B_hat x E_hat)`` and uses the transpose as the
    inverse, which is only exact when ``E_hat`` is perpendicular to
    ``B_hat`` (approximately true for E via the frozen-in condition, away
    from reconnection sites). The bulk velocity is generally not
    perpendicular to B -- there is often strong field-aligned flow,
    especially in the exhaust regions this project's classifier targets --
    so orthogonalizing keeps the matrix a true orthonormal rotation (its
    transpose is always exactly its inverse).

    Returns a 3x3 orthonormal matrix with rows (b_hat, v_perp_hat, b_hat x v_perp_hat).
    """

    b_hat = unit_vector(b_field)
    velocity = np.asarray(bulk_velocity, dtype=float)

    v_perp = velocity - np.dot(velocity, b_hat) * b_hat
    v_perp_norm = np.linalg.norm(v_perp)
    if v_perp_norm == 0:
        raise ValueError(
            "Bulk velocity is parallel to B; cannot build a perpendicular axis"
        )
    v_perp_hat = v_perp / v_perp_norm

    exb_hat = np.cross(b_hat, v_perp_hat)

    return np.stack([b_hat, v_perp_hat, exb_hat], axis=0)


def compute_local_lmn_basis(dBx_dx, dBx_dz, dBy_dx, dBy_dz, dBz_dx, dBz_dz, j_vector):
    """
    LMN boundary-normal basis at one point via Minimum Gradient Analysis
    (MGA) / Minimum Directional Derivative Analysis (MDD) applied to the
    magnetic field Jacobian G = grad(B) (Alho et al. 2024, Ann. Geophys.
    42, 145, https://doi.org/10.5194/angeo-42-145-2024).

    Moved here from physics.current_layer, which used it to expand a
    peak-|J| core selection along the local boundary normal -- that margin
    step is no longer part of current-layer cell selection (see
    physics.current_layer's module docstring), so this function currently
    has no caller. Kept as a future option for an (L, M, N)-aligned VDF
    coordinate transform, alongside the (B, v_perp, B x v_perp) frame
    build_rotation_matrix above builds -- an alternative frame for cases
    where the boundary-normal direction is more physically meaningful than
    the bulk-velocity-derived one.

    Assumes a y-invariant (2D xz-plane) domain (no d/dy data), so G's
    middle column is exactly zero -- a consequence of that grid, not an
    approximation of the method; a 3D caller would need to pass the real
    dBx_dy/dBy_dy/dBz_dy Jacobian entries instead of hardcoding zero.

    L: eigenvector of the largest eigenvalue of G^T G (MGA) -- the
    field-aligned/outflow direction.
    N: eigenvector of the largest eigenvalue of G G^T (MDD), orthogonalized
    against L -- the boundary-normal direction. Sign fixed so
    (N x L) . J > 0 (right-handed), per the paper's convention.
    M = N x L.

    Returns (l_hat, m_hat, n_hat), each a unit 3-vector, or None if the
    Jacobian carries no usable gradient information at this point.
    """

    G = np.array(
        [
            [dBx_dx, 0.0, dBx_dz],
            [dBy_dx, 0.0, dBy_dz],
            [dBz_dx, 0.0, dBz_dz],
        ],
        dtype=float,
    )

    if not np.any(G):
        return None

    _mga_eigvals, mga_eigvecs = np.linalg.eigh(G.T @ G)
    l_hat = mga_eigvecs[:, -1]

    _mdd_eigvals, mdd_eigvecs = np.linalg.eigh(G @ G.T)
    n_raw = mdd_eigvecs[:, -1]

    n_hat = n_raw - np.dot(n_raw, l_hat) * l_hat
    n_norm = np.linalg.norm(n_hat)
    if n_norm == 0:
        return None
    n_hat = n_hat / n_norm

    j_vector = np.asarray(j_vector, dtype=float)
    if np.dot(np.cross(n_hat, l_hat), j_vector) < 0:
        n_hat = -n_hat

    m_hat = np.cross(n_hat, l_hat)

    return l_hat, m_hat, n_hat


def rotate_bounds(v_limits, rotation_matrix):
    """Axis-aligned bounding box, in the rotated frame, of a rotated velocity-space cuboid."""

    v_limits = np.asarray(v_limits, dtype=float)
    lo, hi = v_limits[:3], v_limits[3:]

    corners = np.array(
        [
            [
                hi[0] if corner_index & 1 else lo[0],
                hi[1] if corner_index & 2 else lo[1],
                hi[2] if corner_index & 4 else lo[2],
            ]
            for corner_index in range(8)
        ]
    )
    rotated_corners = corners @ rotation_matrix.T

    return np.concatenate([rotated_corners.min(axis=0), rotated_corners.max(axis=0)])


def compute_new_shape(v_limits, shape, new_v_limits):
    """Rotated-frame grid shape that preserves the original frame's cell spacing."""

    v_limits = np.asarray(v_limits, dtype=float)
    shape = np.asarray(shape, dtype=float)
    spacing = (v_limits[3:] - v_limits[:3]) / (shape - 1)

    new_v_limits = np.asarray(new_v_limits, dtype=float)
    new_extent = new_v_limits[3:] - new_v_limits[:3]
    new_shape = np.floor(new_extent / spacing).astype(int) + 1

    return tuple(int(value) for value in new_shape)


def get_rotated_vdf(vdf, shape, v_limits, b_field, bulk_velocity, new_v_limits=None, new_shape=None):
    """
    Rotate a dense VDF into a (B, v_perp, B x v_perp) aligned frame.

    Builds a regular grid covering the rotated bounding box at
    approximately the original velocity-cell spacing, then fills it by
    trilinearly interpolating the original VDF at each new grid point's
    coordinates in the original frame (``old = R.T @ new``, exact because
    the rotation matrix is orthonormal).

    Uses ``scipy.interpolate.RegularGridInterpolator`` rather than a
    hand-rolled trilinear routine. Validated against a literal Python port
    of the C++ trilinear interpolation: results match to floating-point
    precision everywhere except a thin one-cell-wide shell just outside the
    low edge of each axis, where the C++ code's ``static_cast<int>`` (which
    truncates toward zero instead of flooring) misclassifies slightly
    out-of-domain points as in-bounds. This implementation correctly treats
    those points as outside the domain (filled with zero).

    new_v_limits, new_shape : optional
        By default (both ``None``) the rotated grid's bounding box/shape is
        computed automatically (``rotate_bounds``/``compute_new_shape``),
        and both vary per call depending on ``b_field``'s orientation --
        fine for a single VDF (e.g. before ``vdf_to_hermite_spectra``, whose
        output size depends only on the Hermite order, not this shape). Pass
        both explicitly to force a fixed output grid instead (e.g. the
        original ``v_limits``/``shape``) when rotating many VDFs that must
        end up the same shape as each other, such as flattening into
        fixed-length PCA feature rows (see ``ml_models.features``).

    Returns (rotated_vdf, new_shape, new_v_limits, rotation_matrix).
    """

    shape = tuple(int(value) for value in shape)
    v_limits = np.asarray(v_limits, dtype=float)

    rotation_matrix = build_rotation_matrix(b_field, bulk_velocity)

    if new_v_limits is None:
        new_v_limits = rotate_bounds(v_limits, rotation_matrix)
    else:
        new_v_limits = np.asarray(new_v_limits, dtype=float)
    if new_shape is None:
        new_shape = compute_new_shape(v_limits, shape, new_v_limits)
    else:
        new_shape = tuple(int(value) for value in new_shape)

    axes_new = [
        np.linspace(new_v_limits[axis], new_v_limits[axis + 3], new_shape[axis])
        for axis in range(3)
    ]
    grid_new = np.stack(np.meshgrid(*axes_new, indexing="ij"), axis=-1).reshape(-1, 3)

    # Points on the new (rotated) grid, expressed in the original frame.
    # rotation_matrix maps old -> new (new = R @ old); since R is
    # orthonormal, R.T is its exact inverse, so old = R.T @ new. In
    # row-vector form (as used here), that is old_row = new_row @ R.
    grid_old = grid_new @ rotation_matrix

    axes_old = [
        np.linspace(v_limits[axis], v_limits[axis + 3], shape[axis])
        for axis in range(3)
    ]
    interpolator = RegularGridInterpolator(
        axes_old,
        vdf,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    rotated_vdf = interpolator(grid_old).reshape(new_shape).astype(vdf.dtype, copy=False)

    return (
        rotated_vdf,
        new_shape,
        tuple(float(value) for value in new_v_limits),
        rotation_matrix,
    )


def hermite_polynomials(v_axis, order):
    """Gaussian-weighted physicists' Hermite polynomials, orders 0..order-1, evaluated at v_axis. Shape (order, len(v_axis))."""

    v_axis = np.asarray(v_axis)
    hp = np.zeros((order, len(v_axis)))
    hp[0, :] = np.exp(-0.5 * v_axis**2)
    if order > 1:
        hp[1, :] = 2 * v_axis * np.exp(-0.5 * v_axis**2)
    for n in range(2, order):
        hp[n, :] = 2 * v_axis * hp[n - 1, :] - 2 * (n - 1) * hp[n - 2, :]
    return hp


def normalized_hermite_basis(shape, v_limits, order, vth, u, axis):
    """Normalized Hermite basis along one velocity axis, shape (order, shape[axis])."""

    v_axis = np.linspace(v_limits[axis], v_limits[axis + 3], shape[axis])
    v_axis = (v_axis - u[axis]) / vth
    hermite_vals = hermite_polynomials(v_axis, order)
    for n in range(order):
        norm_const = math.sqrt((2**n) * math.factorial(n) * math.sqrt(math.pi) * vth)
        hermite_vals[n, :] /= norm_const
    return hermite_vals


def compute_drift_velocity(vdf, shape, v_limits):
    """Bulk drift velocity [ux, uy, uz] of a dense 3D VDF."""

    vx = np.linspace(v_limits[0], v_limits[3], shape[0])
    vy = np.linspace(v_limits[1], v_limits[4], shape[1])
    vz = np.linspace(v_limits[2], v_limits[5], shape[2])

    dv1 = (v_limits[3] - v_limits[0]) / shape[0]
    dv2 = (v_limits[4] - v_limits[1]) / shape[1]
    dv3 = (v_limits[5] - v_limits[2]) / shape[2]
    cell_volume = dv1 * dv2 * dv3

    n = float(vdf.sum()) * cell_volume
    if n == 0:
        return np.zeros(3)

    u = np.array(
        [
            np.sum(vdf * vx[:, None, None]),
            np.sum(vdf * vy[None, :, None]),
            np.sum(vdf * vz[None, None, :]),
        ]
    ) * cell_volume

    return u / n


def compute_thermal_velocity(vdf, shape, v_limits, u):
    """Isotropic thermal velocity of a dense 3D VDF around drift velocity u, or 0.0 if the VDF has zero density."""

    vx = np.linspace(v_limits[0], v_limits[3], shape[0])
    vy = np.linspace(v_limits[1], v_limits[4], shape[1])
    vz = np.linspace(v_limits[2], v_limits[5], shape[2])

    dv = (v_limits[3] - v_limits[0]) / shape[0]
    dv3 = dv**3

    n = float(vdf.sum()) * dv3
    if n == 0:
        return 0.0

    Pxx = np.sum(vdf * (vx[:, None, None] - u[0]) ** 2) * dv3
    Pyy = np.sum(vdf * (vy[None, :, None] - u[1]) ** 2) * dv3
    Pzz = np.sum(vdf * (vz[None, None, :] - u[2]) ** 2) * dv3

    return math.sqrt((Pxx + Pyy + Pzz) / (3 * n))


def compute_density(vdf, shape, v_limits):
    """Number density n [m^-3] of a dense 3D VDF (zeroth moment)."""

    dv = (v_limits[3] - v_limits[0]) / shape[0]

    return float(vdf.sum()) * dv**3


def compute_thermal_velocity_components(vdf, shape, v_limits, u):
    """
    Per-axis thermal velocity [vthx, vthy, vthz] (m/s) of a dense 3D VDF
    around drift velocity u -- the anisotropic counterpart to
    compute_thermal_velocity, which combines Pxx/Pyy/Pzz into one isotropic
    scalar; this keeps them separate (e.g. vthx/vthy/vthz in a
    (B, v_perp, B x v_perp)-rotated frame are then thermal speeds parallel
    to B and along each perpendicular direction -- a real, physically
    meaningful anisotropy, especially for current sheets/shocked plasma).

    Returns zeros if the VDF has zero density.
    """

    vx = np.linspace(v_limits[0], v_limits[3], shape[0])
    vy = np.linspace(v_limits[1], v_limits[4], shape[1])
    vz = np.linspace(v_limits[2], v_limits[5], shape[2])

    dv = (v_limits[3] - v_limits[0]) / shape[0]
    dv3 = dv**3

    n = float(vdf.sum()) * dv3
    if n == 0:
        return np.zeros(3)

    Pxx = np.sum(vdf * (vx[:, None, None] - u[0]) ** 2) * dv3
    Pyy = np.sum(vdf * (vy[None, :, None] - u[1]) ** 2) * dv3
    Pzz = np.sum(vdf * (vz[None, None, :] - u[2]) ** 2) * dv3

    return np.array([math.sqrt(Pxx / n), math.sqrt(Pyy / n), math.sqrt(Pzz / n)])


def compute_hermite_spectra(vdf, shape, v_limits, order, vth, u):
    """    Project a dense 3D VDF onto a separable Hermite basis.
    Vectorized, mathematically equivalent to summing
    ``vdf[i, j, k] * Hx[nx, i] * Hy[ny, j] * Hz[nz, k] * dv**3`` over the
    velocity grid for every Hermite mode ``(nx, ny, nz)``.
    """

    dv = (v_limits[3] - v_limits[0]) / shape[0]

    hermite_x = normalized_hermite_basis(shape, v_limits, order, vth, u, 0)
    hermite_y = normalized_hermite_basis(shape, v_limits, order, vth, u, 1)
    hermite_z = normalized_hermite_basis(shape, v_limits, order, vth, u, 2)

    spectra = np.einsum(
        "ijk,ni,mj,lk->nml",
        vdf,
        hermite_x,
        hermite_y,
        hermite_z,
        optimize=True,
    )

    return spectra * dv**3


def reconstruct_from_hermite_spectra(spectra, shape, v_limits, vth, u):
    """
    Inverse of compute_hermite_spectra: evaluate the truncated Hermite
    series at every grid point, given its spectra and the same vth/u that
    built it (the basis functions' width/center -- must match what
    compute_hermite_spectra used, or the reconstruction is meaningless).

    Since normalized_hermite_basis is L2-orthonormal (integral H_n * H_m dv
    = delta_nm), this is compute_hermite_spectra's exact mathematical
    inverse (up to the truncation at `order` terms) -- same basis, same
    separable structure, just summing spectra * basis instead of
    integrating field * basis. No dv**3 factor here: that was specific to
    approximating a continuous inner product (an integral) with a discrete
    Riemann sum over the grid; this step evaluates a already-computed
    series, not another projection.

    A meaningful verification tool: reconstruct a VDF from its own
    (possibly log-space, see vdf_to_hermite_spectra_log) spectra and
    compare against the original -- how well the truncated series
    resembles the input tells you directly whether `order` is high enough
    to actually capture the VDF's shape, rather than trusting the
    spectra's numbers on faith.

    Returns the reconstructed field, shape `shape` (same grid the spectra
    were computed on).
    """

    order = spectra.shape[0]
    hermite_x = normalized_hermite_basis(shape, v_limits, order, vth, u, 0)
    hermite_y = normalized_hermite_basis(shape, v_limits, order, vth, u, 1)
    hermite_z = normalized_hermite_basis(shape, v_limits, order, vth, u, 2)

    return np.einsum(
        "nml,ni,mj,lk->ijk",
        spectra,
        hermite_x,
        hermite_y,
        hermite_z,
        optimize=True,
    )


def vdf_to_hermite_spectra(vdf, shape, v_limits, order=DEFAULT_HERMITE_ORDER):
    """Convert a dense 3D VDF into its Hermite spectra, shape (order, order, order)."""

    u = compute_drift_velocity(vdf, shape, v_limits)
    vth = compute_thermal_velocity(vdf, shape, v_limits, u)

    if vth == 0:
        return np.zeros((order, order, order))

    return compute_hermite_spectra(vdf, shape, v_limits, order, vth, u)


def vdf_to_hermite_spectra_log(vdf, shape, v_limits, sparsity_threshold, order=DEFAULT_HERMITE_ORDER):
    """
    Convert a dense 3D VDF into the Hermite spectra of its log10, rather
    than its raw linear value.

    u (drift velocity) and vth (thermal speed) -- which set the basis
    functions' center/width -- are still computed from the raw, linear-scale
    vdf (compute_drift_velocity/compute_thermal_velocity): those are
    genuine physical moments (real bulk flow, real thermal spread), not
    something a log-density-weighted average would mean anything for. Only
    the basis PROJECTION itself uses log10(vdf) -- mirroring
    ml_models.features' existing raw-pixel representation, which
    log-scales for the same reason: a VDF's peak (near u) is orders of
    magnitude above its wings, so a projection against the raw linear
    values is dominated by the peak, and non-Maxwellian structure that
    shows up in the wings/tails (beams, heating, a reconnection exhaust)
    is comparatively invisible. Working in log space instead keeps the
    wings' contribution to the spectra comparable in scale to the core's.

    sparsity_threshold : float
        The VDF's own sparsity floor (vdf_tools.get_vdf_plot_threshold's
        "MinValue" -- below this, the simulation's sparse storage already
        treats a cell as noise and stores it as exactly 0, so log10 of it
        is undefined). The projected quantity is log10(vdf / sparsity_threshold)
        -- i.e. log10(vdf) with log10(sparsity_threshold) subtracted back
        out (the "patch") -- rather than plain log10(vdf) floored at
        log10(sparsity_threshold). That keeps below-threshold cells at
        exactly 0, preserving the raw VDF's compact support (its velocity-
        space "footprint" is a small fraction of the full grid -- see
        vdf_tools.VdfExtractor): a raw grid point with vdf == 0 contributes
        nothing to compute_hermite_spectra's projection integral, however
        wide the Hermite basis functions are, because 0 times anything is
        0. Substituting a literal log10(sparsity_threshold) FLOOR value
        there instead (a large, non-zero negative constant, not 0) does NOT
        preserve that: that constant background then gets integrated
        against the basis functions over the ENTIRE grid -- typically far
        larger than the real VDF's actual footprint (99%+ empty on this
        fixture) -- dominating every coefficient with a background term
        that depends only on each sample's own u/vth (which set the basis
        functions' shape), not on the VDF's actual physical shape. Caught
        during development: an earlier version used the floored-not-offset
        form and every sample's spectra came out order 1e9-1e10 in
        magnitude, swamping the real signal (see TESTING.md).

    Returns Hermite spectra, shape (order, order, order), or all-zero if
    the VDF has zero density (thermal velocity undefined -- happens before
    the log step, so sparsity_threshold plays no role in that case).
    """

    vdf = np.asarray(vdf, dtype=float)
    u = compute_drift_velocity(vdf, shape, v_limits)
    vth = compute_thermal_velocity(vdf, shape, v_limits, u)

    if vth == 0:
        return np.zeros((order, order, order))

    log_floor = math.log10(float(sparsity_threshold))
    positive_mask = vdf > 0
    log_vdf = np.zeros_like(vdf)
    log_vdf[positive_mask] = np.log10(vdf[positive_mask]) - log_floor

    return compute_hermite_spectra(log_vdf, shape, v_limits, order, vth, u)
