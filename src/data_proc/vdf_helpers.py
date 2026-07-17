import math

import numpy as np
import analysator as pt
from scipy.interpolate import RegularGridInterpolator

R_EARTH = 6.371e6
DEFAULT_HERMITE_ORDER = 22


def coord_re_to_m(coord_re):
    return np.array(coord_re, dtype=float) * R_EARTH


def create_coordinate_name(coord_re):
    return (
        f"x{coord_re[0]:g}_y{coord_re[1]:g}_z{coord_re[2]:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def iter_enabled_regions_re(points_config, names_key, default_region_name="tail"):
    """Yield (region_name, region_re) for each region configured under ``names_key``."""

    points_config = points_config or {}
    regions_re = points_config.get("regions_re")

    if regions_re is None:
        region_re = points_config.get("region_re")
        if region_re is not None:
            yield default_region_name, region_re
        return

    region_names = points_config.get(names_key)
    if region_names is None:
        region_names = list(regions_re)
    elif isinstance(region_names, str):
        region_names = [region_names]

    for region_name in region_names:
        if region_name not in regions_re:
            raise ValueError(
                f"points.{names_key} contains unknown region "
                f"{region_name!r}"
            )

        yield str(region_name), regions_re[region_name]


def find_matching_region_name_re(coord_re, points_config, names_key):
    """Return the name of the first configured region containing ``coord_re``, or None."""

    points_config = points_config or {}
    if (
            points_config.get("regions_re") is None
            and points_config.get("region_re") is None
    ):
        return "all"

    for region_name, region_re in iter_enabled_regions_re(
            points_config=points_config,
            names_key=names_key,
    ):
        if is_coord_in_region_re(coord_re, region_re):
            return region_name

    return None


def is_coord_in_region_re(coord_re, region_re):
    coord_re = np.asarray(coord_re, dtype=float)

    for axis_index, axis_name in enumerate(("x", "y", "z")):
        lower_re, upper_re = get_region_axis_bounds_re(region_re, axis_name)

        if lower_re is not None and coord_re[axis_index] < lower_re:
            return False

        if upper_re is not None and coord_re[axis_index] > upper_re:
            return False

    return True


def create_region_mask_re(coords_re, region_re):
    """Boolean mask selecting rows of ``coords_re`` (shape ``(n, 3)``) inside ``region_re``."""

    coords_re = np.asarray(coords_re, dtype=float)
    selected = np.ones(coords_re.shape[0], dtype=bool)

    for axis_index, axis_name in enumerate(("x", "y", "z")):
        lower_re, upper_re = get_region_axis_bounds_re(region_re, axis_name)

        if lower_re is not None:
            selected &= coords_re[:, axis_index] >= lower_re

        if upper_re is not None:
            selected &= coords_re[:, axis_index] <= upper_re

    return selected


def get_region_axis_bounds_re(region_re, axis_name):
    """Return (lower, upper) bounds in RE for one axis, from *_between/*_abs_max/*_min/*_max keys."""

    between = region_re.get(f"{axis_name}_between")
    if between is not None:
        lower_re, upper_re = between
        return min(lower_re, upper_re), max(lower_re, upper_re)

    abs_max_re = region_re.get(f"{axis_name}_abs_max")
    if abs_max_re is not None:
        abs_max_re = float(abs_max_re)
        return -abs_max_re, abs_max_re

    lower_re = region_re.get(f"{axis_name}_min")
    upper_re = region_re.get(f"{axis_name}_max")
    if lower_re is not None:
        lower_re = float(lower_re)
    if upper_re is not None:
        upper_re = float(upper_re)

    if lower_re is not None and upper_re is not None:
        return min(lower_re, upper_re), max(lower_re, upper_re)

    return lower_re, upper_re


def get_cellid_with_vdf(reader, coord_re, pop="avgs"):
    coord_m = coord_re_to_m(coord_re)
    cid = reader.get_cellid_with_vdf(coord_m, pop=pop)

    return int(cid)


def cell_has_vdf(reader, cid, pop="avgs"):
    try:
        velocity_cells = reader.read_velocity_cells(int(cid), pop)
    except Exception:
        return False

    return len(velocity_cells) > 0


def get_vdf_cellid_set(reader, pop="avgs"):
    try:
        cellids = reader.read(
            mesh="SpatialGrid",
            tag="CELLSWITHBLOCKS",
            name=pop,
        )
    except Exception:
        cellids = reader.read(
            mesh="SpatialGrid",
            tag="CELLSWITHBLOCKS",
        )

    return {int(cid) for cid in np.atleast_1d(cellids)}


def get_vdf_cells_with_coords_re(reader, pop="avgs"):
    """Return (cellids, coords_re) for all spatial cells that contain a VDF."""

    cellids = np.asarray(sorted(get_vdf_cellid_set(reader, pop=pop)), dtype=int)
    if len(cellids) == 0:
        return cellids, np.empty((0, 3), dtype=float)

    try:
        coords = reader.get_cell_coordinates(cellids)
    except Exception:
        coords = [reader.get_cell_coordinates(int(cid)) for cid in cellids]

    coords_re = np.asarray(coords, dtype=float) / R_EARTH

    return cellids, coords_re


def get_nearest_vdf_cellid(coord_re, vdf_cellids, vdf_coords_re):
    if len(vdf_cellids) == 0:
        raise ValueError("No velocity distributions found")

    coord_re = np.asarray(coord_re, dtype=float)
    distances_squared = np.sum((vdf_coords_re - coord_re) ** 2, axis=1)
    nearest_index = int(np.argmin(distances_squared))

    return int(vdf_cellids[nearest_index])


def get_spatial_index_range(reader, axis_name, axis_index, min_value, max_value):
    """Convert coordinate bounds in meters to clipped fsgrid index bounds for one axis."""

    mesh_size = np.asarray(reader.get_spatial_mesh_size(), dtype=int)
    n_cells = int(mesh_size[axis_index])
    axis_min = float(reader.read_parameter(f"{axis_name}min"))
    axis_max = float(reader.read_parameter(f"{axis_name}max"))
    cell_size = (axis_max - axis_min) / n_cells

    lower_index = int(np.floor((min_value - axis_min) / cell_size))
    upper_index = int(np.floor((max_value - axis_min) / cell_size))

    lower_index = max(0, min(n_cells - 1, lower_index))
    upper_index = max(0, min(n_cells - 1, upper_index))

    return min(lower_index, upper_index), max(lower_index, upper_index)


def get_vdf_cellids_in_box(
    reader,
    coord_re,
    box_config,
    pop="avgs",
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """Return {box position name: cell ID} for VDF cells inside a box around ``coord_re``."""

    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(
            reader=reader,
            pop=pop,
        )

    if len(vdf_cellids) == 0:
        return {}

    center_re = np.asarray(coord_re, dtype=float)
    half_widths_re = np.array(
        [
            float(box_config["x_half_width_re"]),
            float(box_config["y_half_width_re"]),
            float(box_config["z_half_width_re"]),
        ],
        dtype=float,
    )

    lower_bounds = center_re - half_widths_re
    upper_bounds = center_re + half_widths_re
    active_axes = half_widths_re > 0

    if np.any(active_axes):
        selected = np.all(
            (vdf_coords_re[:, active_axes] >= lower_bounds[active_axes])
            & (vdf_coords_re[:, active_axes] <= upper_bounds[active_axes]),
            axis=1,
        )
    else:
        selected = np.ones(len(vdf_cellids), dtype=bool)

    selected_cellids = vdf_cellids[selected]
    if cell_has_vdf_func is not None:
        selected_cellids = np.asarray(
            [cid for cid in selected_cellids if cell_has_vdf_func(int(cid))],
            dtype=int,
        )

    return {
        f"box_{index:04d}": int(cid)
        for index, cid in enumerate(selected_cellids)
    }


def get_b_field(reader, cid):
    return np.asarray(reader.read_variable("B", int(cid)), dtype=float)


def get_bulk_velocity(reader, cid):
    return np.asarray(reader.read_variable("V", int(cid)), dtype=float)


def get_velocity_cell_size_from_extent(extent, vdf_shape, axis="vy"):
    extent = np.asarray(extent, dtype=float)
    axis_map = {"vx": 0, "vy": 1, "vz": 2}
    axis_index = axis_map[axis]

    vmin = extent[axis_index]
    vmax = extent[axis_index + 3]

    return float((vmax - vmin) / vdf_shape[axis_index])


def get_vdf_plot_parameters(reader, cid, vdf_shape, pop="avgs"):
    """Return (extent, dv, threshold) for a VDF sample, from an open reader."""

    extent, dv = get_vdf_plot_axes_parameters(
        reader=reader,
        vdf_shape=vdf_shape,
        pop=pop,
    )
    threshold = get_vdf_plot_threshold(
        reader=reader,
        cid=cid,
    )

    return extent, dv, threshold


def get_vdf_plot_axes_parameters(reader, vdf_shape, pop="avgs"):
    extent = np.asarray(
        reader.get_velocity_mesh_extent(pop=pop),
        dtype=float
    )

    dv = get_velocity_cell_size_from_extent(
        extent=extent,
        vdf_shape=vdf_shape,
    )

    return extent, dv


def get_vdf_plot_threshold(reader, cid):
    return float(reader.read_variable("MinValue", int(cid)))


def get_vdf_plot_parameters_from_file(file_location, cid, vdf_shape, pop="avgs"):
    """Return (extent, dv, threshold) for a VDF sample, opening the VLSV file directly."""

    reader = pt.vlsvfile.VlsvReader(str(file_location))

    return get_vdf_plot_parameters(
        reader=reader,
        cid=cid,
        vdf_shape=vdf_shape,
        pop=pop,
    )


def create_xz_slice(vdf):
    """Middle xz slice of a 3D VDF array."""

    mid_y = vdf.shape[1] // 2
    return vdf[:, mid_y, :]


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


def get_rotated_vdf(vdf, shape, v_limits, b_field, bulk_velocity):
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

    Returns (rotated_vdf, new_shape, new_v_limits, rotation_matrix).
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


def compute_hermite_spectra(vdf, shape, v_limits, order, vth, u):
    """
    Project a dense 3D VDF onto a separable Hermite basis.

    Vectorized, mathematically equivalent to summing
    ``vdf[i, j, k] * Hx[nx, i] * Hy[ny, j] * Hz[nz, k] * dv**3`` over the
    velocity grid for every Hermite mode ``(nx, ny, nz)``.

    Returns spectra with shape (order, order, order).
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


def vdf_to_hermite_spectra(vdf, shape, v_limits, order=DEFAULT_HERMITE_ORDER):
    """Convert a dense 3D VDF into its Hermite spectra, shape (order, order, order)."""

    u = compute_drift_velocity(vdf, shape, v_limits)
    vth = compute_thermal_velocity(vdf, shape, v_limits, u)

    if vth == 0:
        return np.zeros((order, order, order))

    return compute_hermite_spectra(vdf, shape, v_limits, order, vth, u)
