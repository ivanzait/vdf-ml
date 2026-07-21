"""VLSV/VDF technical tools: coordinate/region helpers, cellid lookups, VDF-plot-parameter readers, and dense VDF extraction."""

import numpy as np
import analysator as pt

R_EARTH = 6.371e6


# =====================================================================
# Masking -- coordinate/region matching, box membership
# =====================================================================

def coord_re_to_m(coord_re):
    """Convert a coordinate in Earth radii to meters."""

    return np.array(coord_re, dtype=float) * R_EARTH


def create_coordinate_name(coord_re):
    """Filesystem-safe name for a coordinate, e.g. x-12_y0_z0p5 -> xm12_y0_z0p5."""

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
    """Whether one coordinate falls inside a region_re box."""

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


# =====================================================================
# Primitives -- cellid/coordinate/field lookups against an open reader
# =====================================================================

def get_cellid_with_vdf(reader, coord_re, pop="avgs"):
    """Spatial cell ID with a VDF nearest to a coordinate in Earth radii."""

    coord_m = coord_re_to_m(coord_re)
    cid = reader.get_cellid_with_vdf(coord_m, pop=pop)

    return int(cid)


def get_vdf_cellid_set(reader, pop="avgs"):
    """Set of every spatial cell ID that carries a VDF for a population."""

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
    """Nearest VDF-carrying cell ID to a coordinate, from an already-loaded cellid/coords pair."""

    if len(vdf_cellids) == 0:
        raise ValueError("No velocity distributions found")

    coord_re = np.asarray(coord_re, dtype=float)
    distances_squared = np.sum((vdf_coords_re - coord_re) ** 2, axis=1)
    nearest_index = int(np.argmin(distances_squared))

    return int(vdf_cellids[nearest_index])


def get_b_field(reader, cid):
    """Magnetic field vector B at one cell."""

    return np.asarray(reader.read_variable("B", int(cid)), dtype=float)


def get_bulk_velocity(reader, cid):
    """Bulk flow velocity vector V at one cell."""

    return np.asarray(reader.read_variable("V", int(cid)), dtype=float)


# =====================================================================
# VDF plot parameters -- velocity-mesh extent/spacing/threshold, xz slicing
# =====================================================================

def get_velocity_cell_size_from_extent(extent, vdf_shape, axis="vy"):
    """Velocity cell size dv along one axis, from the mesh extent and dense VDF shape."""

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
    """Return (extent, dv): the velocity mesh extent and cell size for a population."""

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
    """Sparsity threshold (MinValue) below which a cell's VDF values are noise."""

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
    """Middle xz slice of a 3D VDF array, shape (vx, vy, vz) -> (vx, vz)."""

    mid_y = vdf.shape[1] // 2
    return vdf[:, mid_y, :]


# =====================================================================
# Extractor -- dense VDF extraction from sparse VLSV velocity-space data
# =====================================================================

class VdfExtractor:
    """Extract dense VDF arrays from an open reader, caching the sorted velocity-space ordering once per population."""

    def __init__(self, reader, pop="avgs", dtype=np.float32):
        self.reader = reader
        self.pop = pop
        self.dtype = dtype

        size = reader.get_velocity_mesh_size(pop)
        self.vdf_shape = (
            4 * int(size[0]),
            4 * int(size[1]),
            4 * int(size[2]),
        )
        self.n_velocity_cells = int(np.prod(self.vdf_shape))
        self.sorted_velocity_indices = create_sorted_velocity_indices(
            reader=reader,
            n_velocity_cells=self.n_velocity_cells,
            pop=pop,
        )

    def extract(self, cid, box=-1):
        """Extract one dense 3D VDF (axis order [vx, vy, vz]), optionally cropped to +/-box around its peak."""

        if int(cid) <= 0:
            raise ValueError("cid must be positive")

        velocity_cells = self.reader.read_velocity_cells(int(cid), self.pop)
        dist = np.zeros(self.n_velocity_cells, dtype=self.dtype)
        dist[list(velocity_cells.keys())] = list(velocity_cells.values())

        vdf = dist[self.sorted_velocity_indices].reshape(self.vdf_shape)
        max_indices = np.unravel_index(np.argmax(vdf), vdf.shape)
        box = int(box)

        if box > 0:
            i, j, k = max_indices
            data = vdf[
                (i - box):(i + box),
                (j - box):(j + box),
                (k - box):(k + box),
            ]
        else:
            data = vdf

        data = np.swapaxes(data, 2, 0)

        return np.asarray(data, dtype=self.dtype)


def create_sorted_velocity_indices(reader, n_velocity_cells, pop="avgs"):
    """Indices that sort a population's velocity cells into vx/vy/vz order, used to densify the VLSV velocity block layout."""

    vids = np.arange(int(n_velocity_cells))
    velocity_coords = reader.get_velocity_cell_coordinates(vids, pop)
    sorted_indices = vids

    for axis_index in range(3):
        axis_order = np.argsort(
            velocity_coords[:, axis_index],
            kind="stable",
        )
        velocity_coords = velocity_coords[axis_order]
        sorted_indices = sorted_indices[axis_order]

    return sorted_indices


def extract_vdf(file_location, cid, box=-1, pop="avgs"):
    """Extract one dense 3D VDF directly from a file path (opens its own reader); see VdfExtractor to reuse a reader across many cells."""

    reader = pt.vlsvfile.VlsvReader(file_location)
    extractor = VdfExtractor(reader=reader, pop=pop)

    return extractor.extract(cid=cid, box=box)
