"""VDF-cell selection around a detected X/O point ("which cells belong to this point")."""

import numpy as np
from matplotlib.path import Path as MplPath

from src.data_proc.vdf_tools import R_EARTH, get_b_field

SELECTION_METHOD = "union_physical_priority"


def get_point_selection_config(config, point_kind):
    """Return the ``x_selection``/``o_selection`` config block for one point kind."""

    points_config = (config or {}).get("points", config or {})

    if point_kind == "x":
        return points_config.get("x_selection", {})

    if point_kind == "o":
        return points_config.get("o_selection", {})

    raise ValueError(f"Unknown point kind: {point_kind}")


def get_manual_config_re(config, point_kind):
    """Return manual point-box half-widths in Earth radii: x/y/z_half_width_re."""

    selection_config = get_point_selection_config(
        config=config,
        point_kind=point_kind,
    )
    box_config = selection_config.get("manual_re", {})
    required_keys = ("x_half_width_re", "z_half_width_re")
    missing_keys = [
        key
        for key in required_keys
        if key not in box_config
    ]

    if missing_keys:
        raise ValueError(
            f"points.{point_kind}_selection.manual_re is missing "
            f"required keys: {missing_keys}"
        )

    return {
        "x_half_width_re": float(box_config["x_half_width_re"]),
        "y_half_width_re": float(
            box_config.get(
                "y_half_width_re",
                selection_config.get("y_half_width_re", 0.0),
            )
        ),
        "z_half_width_re": float(box_config["z_half_width_re"]),
    }


def compute_b_perp_di_box_geometry(reader, point_record, half_width_di_normal, outflow_aspect_ratio):
    """
    In-plane box geometry for ``get_vdf_cellids_in_b_perp_di_box``, factored
    out so plotting code can draw exactly the box that was selected instead
    of re-deriving it.

    Returns ``(b_hat_inplane, perp_hat_inplane, half_width_normal_m,
    half_width_outflow_m)``, or ``None`` if the local B field has no
    in-plane (xz) component.
    """

    b_field = get_b_field(reader=reader, cid=point_record["cellid"])
    b_inplane = np.array([b_field[0], b_field[2]], dtype=float)
    b_inplane_norm = np.linalg.norm(b_inplane)
    if b_inplane_norm == 0:
        return None

    b_hat_inplane = b_inplane / b_inplane_norm
    perp_hat_inplane = np.array([-b_hat_inplane[1], b_hat_inplane[0]])

    half_width_normal_m = float(half_width_di_normal) * float(point_record["di_m"])
    half_width_outflow_m = half_width_normal_m * float(outflow_aspect_ratio)

    return b_hat_inplane, perp_hat_inplane, half_width_normal_m, half_width_outflow_m


def get_vdf_cellids_in_b_perp_di_box(reader, config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells in a rectangular box around an X point, in-plane axes
    aligned perpendicular/parallel to the local B field (read at the X
    point's own cell). Out-of-plane (y) is left open, same as
    y_half_width_re=0 does for the Hessian-box selector.

    Proxy for the ion diffusion region: short side perpendicular to B (~normal
    to the current layer near the X point), half-width half_width_di_normal *
    d_i; long side along B (~outflow direction along the current sheet),
    outflow_aspect_ratio times wider (aspect ratio 1:10 by default, matching
    the IDR's normal-vs-outflow extent).

    Both in-plane axes are bounded deliberately: leaving either open turns
    the box into an infinite strip, which (with several X-point candidates
    spread across a snapshot) sweeps in cells tens of Re away.

    Config keys (under x_selection): half_width_di_normal (half-width in
    d_i along the perpendicular-to-B/normal direction) and
    outflow_aspect_ratio (multiplier for the along-B/outflow half-width,
    default 10.0).
    """

    if len(vdf_cellids) == 0 or point_record.get("di_m") is None:
        return {}

    x_selection = get_point_selection_config(config=config, point_kind="x")
    half_width_di_normal = x_selection.get("half_width_di_normal")
    if half_width_di_normal is None:
        return {}
    outflow_aspect_ratio = x_selection.get("outflow_aspect_ratio", 10.0)

    geometry = compute_b_perp_di_box_geometry(
        reader=reader,
        point_record=point_record,
        half_width_di_normal=half_width_di_normal,
        outflow_aspect_ratio=outflow_aspect_ratio,
    )
    if geometry is None:
        return {}
    b_hat_inplane, perp_hat_inplane, half_width_normal_m, half_width_outflow_m = geometry

    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_xz_m = (vdf_coords_re - center_re)[:, [0, 2]] * R_EARTH
    perp_projection_m = offsets_xz_m @ perp_hat_inplane
    along_projection_m = offsets_xz_m @ b_hat_inplane

    selected = (
        (np.abs(perp_projection_m) <= half_width_normal_m)
        & (np.abs(along_projection_m) <= half_width_outflow_m)
    )

    return {
        f"x_bperp_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_vdf_cellids_in_flux_contour(config, point_record, vdf_cellids, vdf_coords_re):
    """Select VDF cells inside an O-point's closed flux contour."""

    if len(vdf_cellids) == 0:
        return {}

    contour_vertices_re = point_record.get("contour_vertices_re")
    if contour_vertices_re is None:
        return {}

    o_selection = get_point_selection_config(
        config=config,
        point_kind="o",
    )
    y_half_width_re = float(o_selection.get("y_half_width_re", 0.0))
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re

    contour_path = MplPath(np.asarray(contour_vertices_re, dtype=float))
    selected = (
        contour_path.contains_points(vdf_coords_re[:, [0, 2]])
        & (
            (y_half_width_re <= 0)
            | (np.abs(offsets_re[:, 1]) <= y_half_width_re)
        )
    )

    return {
        f"o_island_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_vdf_cellids_in_gyroradius_circle(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells within radius_gyroradii * rho_i of an O point, in-plane
    (xz) Euclidean distance. Out-of-plane (y) is left open, same as
    y_half_width_re=0 does elsewhere.

    Config keys (under o_selection): radius_gyroradii (multiplier on the
    thermal ion gyroradius rho_i_re computed by add_thermal_gyroradius,
    default 1.0).
    """

    if len(vdf_cellids) == 0 or point_record.get("rho_i_re") is None:
        return {}

    o_selection = get_point_selection_config(
        config=config,
        point_kind="o",
    )
    radius_gyroradii = o_selection.get("radius_gyroradii", 1.0)
    radius_re = float(radius_gyroradii) * float(point_record["rho_i_re"])
    y_half_width_re = float(o_selection.get("y_half_width_re", 0.0))

    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re
    in_plane_distance_re = np.linalg.norm(offsets_re[:, [0, 2]], axis=1)

    selected = (
        (in_plane_distance_re <= radius_re)
        & (
            (y_half_width_re <= 0)
            | (np.abs(offsets_re[:, 1]) <= y_half_width_re)
        )
    )

    return {
        f"o_gyro_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_o_point_cellids_by_method(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells for one O point using o_selection.selection_method:
    "gyroradius" (default) -> get_vdf_cellids_in_gyroradius_circle, round area
    sized by the local thermal ion gyroradius; "flux_contour" ->
    get_vdf_cellids_in_flux_contour, the O point's closed flux island.
    """

    o_selection = get_point_selection_config(config=config, point_kind="o")
    selection_method = o_selection.get("selection_method", "gyroradius")

    if selection_method == "gyroradius":
        return get_vdf_cellids_in_gyroradius_circle(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    if selection_method == "flux_contour":
        return get_vdf_cellids_in_flux_contour(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    raise ValueError(f"Unknown o_selection.selection_method: {selection_method}")


def create_point_sample_metadata(config, point_record):
    """Metadata fields describing a point-selected sample: source point, selection box, hessian/flux info."""

    point_kind = point_record["point_kind"]
    coord_re = point_record["coord_re"]
    box_config = get_manual_config_re(
        config=config,
        point_kind=point_kind,
    )
    metadata = {
        "point_kind": point_kind,
        "selection_method": SELECTION_METHOD,
        "source_point_x_re": float(coord_re[0]),
        "source_point_y_re": float(coord_re[1]),
        "source_point_z_re": float(coord_re[2]),
        "source_point_flux": float(point_record["flux"]),
        "selection_box_x_half_width_re": box_config["x_half_width_re"],
        "selection_box_y_half_width_re": box_config["y_half_width_re"],
        "selection_box_z_half_width_re": box_config["z_half_width_re"],
    }
    if "region_name" in point_record:
        metadata["region_name"] = point_record["region_name"]

    if point_kind == "x":
        eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
        metadata.update(
            {
                "rho": optional_float(point_record.get("rho")),
                "di_re": optional_float(point_record.get("di_re")),
                "hessian_e0_x": float(eigvecs[0, 0]),
                "hessian_e0_z": float(eigvecs[1, 0]),
                "hessian_e1_x": float(eigvecs[0, 1]),
                "hessian_e1_z": float(eigvecs[1, 1]),
            }
        )

    if point_kind == "o":
        metadata.update(
            {
                "boundary_flux": optional_float(point_record.get("boundary_flux")),
                "search_flux": optional_float(point_record.get("search_flux")),
                "core_fraction": optional_float(point_record.get("core_fraction")),
                "temperature_k": optional_float(point_record.get("temperature_k")),
                "rho_i_re": optional_float(point_record.get("rho_i_re")),
            }
        )

    return metadata


def optional_float(value):
    """``float(value)``, or ``nan`` when ``value`` is missing."""

    if value is None:
        return float("nan")

    return float(value)
