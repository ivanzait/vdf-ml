import numpy as np
from scipy.interpolate import RegularGridInterpolator


def unit_vector(vector):
    """
    Normalize a 3-vector to unit length.

    Parameters
    ----------
    vector : array-like of float
        Input vector.

    Returns
    -------
    numpy.ndarray
        Unit vector.
    """

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

    Parameters
    ----------
    b_field : array-like of float
        Magnetic field vector ``[Bx, By, Bz]``.
    bulk_velocity : array-like of float
        Bulk flow velocity vector ``[Vx, Vy, Vz]``.

    Returns
    -------
    numpy.ndarray
        ``3x3`` orthonormal rotation matrix with rows
        ``(b_hat, v_perp_hat, b_hat x v_perp_hat)``.
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


def rotate_bounds(v_limits, rotation_matrix):
    """
    Find the axis-aligned bounding box of a rotated velocity-space cuboid.

    Parameters
    ----------
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    rotation_matrix : numpy.ndarray
        ``3x3`` rotation matrix mapping the original frame to the rotated
        frame (``new = R @ old``).

    Returns
    -------
    numpy.ndarray
        Rotated-frame extent
        ``[nx_min, ny_min, nz_min, nx_max, ny_max, nz_max]``.
    """

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
    """
    Compute a rotated-frame grid shape that preserves the original spacing.

    Parameters
    ----------
    v_limits : array-like of float
        Original velocity mesh extent.
    shape : tuple of int
        Original VDF grid shape ``(nx, ny, nz)``.
    new_v_limits : array-like of float
        Rotated-frame velocity mesh extent from ``rotate_bounds``.

    Returns
    -------
    tuple of int
        Rotated-frame grid shape.
    """

    v_limits = np.asarray(v_limits, dtype=float)
    shape = np.asarray(shape, dtype=float)
    spacing = (v_limits[3:] - v_limits[:3]) / (shape - 1)

    new_v_limits = np.asarray(new_v_limits, dtype=float)
    new_extent = new_v_limits[3:] - new_v_limits[:3]
    new_shape = np.floor(new_extent / spacing).astype(int) + 1

    return tuple(int(value) for value in new_shape)


def get_rotated_vdf(vdf, shape, v_limits, b_field, bulk_velocity):
    """
    Rotate a dense VDF into a (B, v_perp, B x v_perp) aligned frame.

    Builds a regular grid covering the rotated bounding box at
    approximately the original velocity-cell spacing, then fills it by
    trilinearly interpolating the original VDF at each new grid point's
    coordinates in the original frame (``old = R.T @ new``, exact because
    the rotation matrix is orthonormal).

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with axis order ``[vx, vy, vz]`` and shape
        ``shape``.
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    b_field : array-like of float
        Magnetic field vector ``[Bx, By, Bz]``.
    bulk_velocity : array-like of float
        Bulk flow velocity vector ``[Vx, Vy, Vz]``.

    Returns
    -------
    rotated_vdf : numpy.ndarray
        Dense VDF array in the rotated frame.
    new_shape : tuple of int
        Rotated-frame grid shape.
    new_v_limits : tuple of float
        Rotated-frame velocity mesh extent.
    rotation_matrix : numpy.ndarray
        The ``3x3`` rotation matrix used, with rows
        ``(b_hat, v_perp_hat, b_hat x v_perp_hat)``.

    Notes
    -----
    Interpolation uses ``scipy.interpolate.RegularGridInterpolator`` rather
    than a hand-rolled trilinear routine. Validated against a literal Python
    port of the C++ trilinear interpolation: results match to floating-point
    precision everywhere except a thin one-cell-wide shell just outside the
    low edge of each axis, where the C++ code's ``static_cast<int>`` (which
    truncates toward zero instead of flooring) misclassifies slightly
    out-of-domain points as in-bounds. This implementation correctly treats
    those points as outside the domain (filled with zero).
    """

    shape = tuple(int(value) for value in shape)
    v_limits = np.asarray(v_limits, dtype=float)

    rotation_matrix = build_rotation_matrix(b_field, bulk_velocity)

    new_v_limits = rotate_bounds(v_limits, rotation_matrix)
    new_shape = compute_new_shape(v_limits, shape, new_v_limits)

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
