import analysator as pt
import numpy as np
from matplotlib.path import Path as MplPath

from src.data_proc.config import create_timestep_path
from src.data_proc.point_topology import find_point_records
from src.data_proc.vdf_helpers import R_EARTH, get_cellid_with_vdf

SELECTION_METHOD = "union_physical_priority"


def iter_labeled_coords(config):
    """Yield (class_name, label, coord_re) for each static coordinate under config.class_coords_re."""

    labels = config["labels"]
    class_coords = config.get("class_coords_re", {})

    for class_name, coords in class_coords.items():
        label = labels[class_name]

        for coord_re in coords:
            yield class_name, int(label), coord_re


def create_point_label_data(config, timestep, reader=None):
    """
    Detect X/O points for one timestep and attach class_name/label to each.

    Raw detected records (before removing X/O points that share a VDF cell)
    are kept alongside the filtered, labeled ones so downstream code can use
    every detected point for nearest-point distance/vector metadata.

    Returns a dict: point_labeled_coords, rejected_cellids, raw_x_point_records,
    raw_o_point_records.
    """

    points_config = config.get("points")
    labels = config["labels"]

    x_class_name = points_config["x_class_name"]
    o_class_name = points_config["o_class_name"]

    flux_file_location = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )

    if reader is None:
        bulk_file_location = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        reader = pt.vlsvfile.VlsvReader(str(bulk_file_location))

    raw_x_point_records, raw_o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )

    x_point_records, o_point_records, rejected_cellids = remove_shared_cellid_points(
        reader=reader,
        x_point_records=raw_x_point_records,
        o_point_records=raw_o_point_records,
    )

    point_labeled_coords = []

    for point_record in x_point_records:
        labeled_record = dict(point_record)
        labeled_record["class_name"] = x_class_name
        labeled_record["label"] = int(labels[x_class_name])
        point_labeled_coords.append(labeled_record)

    for point_record in o_point_records:
        labeled_record = dict(point_record)
        labeled_record["class_name"] = o_class_name
        labeled_record["label"] = int(labels[o_class_name])
        point_labeled_coords.append(labeled_record)

    return {
        "point_labeled_coords": point_labeled_coords,
        "rejected_cellids": rejected_cellids,
        "raw_x_point_records": raw_x_point_records,
        "raw_o_point_records": raw_o_point_records,
    }


def remove_shared_cellid_points(reader, x_point_records, o_point_records):
    """Drop X and O records that resolve to the same VDF cell (kept in neither). Returns (x_records, o_records, shared_cellids)."""

    x_points_by_cellid = group_point_records_by_cellid(reader, x_point_records)
    o_points_by_cellid = group_point_records_by_cellid(reader, o_point_records)

    shared_cellids = set(x_points_by_cellid) & set(o_points_by_cellid)

    filtered_x_point_records = [
        point_records[0]
        for cellid, point_records in x_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    filtered_o_point_records = [
        point_records[0]
        for cellid, point_records in o_points_by_cellid.items()
        if cellid not in shared_cellids
    ]

    return filtered_x_point_records, filtered_o_point_records, shared_cellids


def group_point_records_by_cellid(reader, point_records):
    """Group point records by the VDF cell ID their coord_re resolves to."""

    point_records_by_cellid = {}

    for point_record in point_records:
        cellid = get_cellid_with_vdf(reader, point_record["coord_re"])
        point_records_by_cellid.setdefault(cellid, []).append(point_record)

    return point_records_by_cellid


def is_point_record(labeled_coord):
    """Whether ``labeled_coord`` is a detected X/O point record (vs. a static coordinate tuple)."""
    return isinstance(labeled_coord, dict) and labeled_coord.get("is_point_record")


def unpack_labeled_coord(labeled_coord):
    """Return ``(class_name, label, coord_re)`` from a static tuple or detected point record."""

    if is_point_record(labeled_coord):
        return (
            labeled_coord["class_name"],
            int(labeled_coord["label"]),
            labeled_coord["coord_re"],
        )

    class_name, label, coord_re = labeled_coord
    return class_name, int(label), coord_re


def get_point_selection_result(config, point_record, vdf_cellids, vdf_coords_re):
    """
    Select VDF cells for a detected X/O point.

    Combines two selections and keeps the union: the physical detector (a
    Hessian-aligned box for X points, a closed flux contour for O points) and
    a fixed manual box around the point. Cells found by both keep the
    physical-method metadata; cells found by only one method are tagged
    accordingly (see ``create_combined_selection_metadata``).

    Returns a dict with ``cellids_by_position``, ``rejected_cellids``, and
    ``metadata_by_position``.
    """

    physical_cellids_by_position = get_physical_point_cellids_by_position(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )
    manual_cellids_by_position = get_vdf_cellids_in_manual(
        config=config,
        point_record=point_record,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
    )

    physical_cellids = set(invert_cellids_by_position(physical_cellids_by_position))
    manual_cellids = set(invert_cellids_by_position(manual_cellids_by_position))

    cellids_by_position = {}
    metadata_by_position = {}
    for position, cid in physical_cellids_by_position.items():
        cid = int(cid)
        manual_selected = cid in manual_cellids
        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="both" if manual_selected else "physical_only",
            physical_selected=True,
            manual_selected=manual_selected,
            plot_selection_method="physical",
        )

    for position, cid in manual_cellids_by_position.items():
        cid = int(cid)
        if cid in physical_cellids:
            continue

        cellids_by_position[position] = cid
        metadata_by_position[position] = create_combined_selection_metadata(
            selection_agreement="manual_only",
            physical_selected=False,
            manual_selected=True,
            plot_selection_method="manual",
        )

    return create_point_selection_result(
        cellids_by_position=cellids_by_position,
        metadata_by_position=metadata_by_position,
    )


def get_physical_point_cellids_by_position(
    config,
    point_record,
    vdf_cellids,
    vdf_coords_re,
):
    """Physical-detector selection for one X/O point: Hessian box for X, flux contour for O."""

    point_kind = point_record["point_kind"]

    if point_kind == "x":
        return get_vdf_cellids_in_hessian_di_box(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    if point_kind == "o":
        return get_vdf_cellids_in_flux_contour(
            config=config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )

    raise ValueError(f"Unknown point kind: {point_kind}")


def create_point_selection_result(
    cellids_by_position,
    rejected_cellids=None,
    metadata_by_position=None,
):
    """Normalize a point-selection result into cellids_by_position/rejected_cellids/metadata_by_position."""

    return {
        "cellids_by_position": {
            position: int(cid)
            for position, cid in cellids_by_position.items()
        },
        "rejected_cellids": {
            int(cid)
            for cid in (rejected_cellids or set())
        },
        "metadata_by_position": metadata_by_position or {},
    }


def invert_cellids_by_position(cellids_by_position):
    """Return {VDF cell ID: first selection position found} from a position-to-cellid mapping."""

    positions_by_cellid = {}
    for position, cid in cellids_by_position.items():
        positions_by_cellid.setdefault(int(cid), position)

    return positions_by_cellid


def create_combined_selection_metadata(
    selection_agreement,
    physical_selected,
    manual_selected,
    plot_selection_method,
):
    """Metadata describing whether a cell was found by the physical detector, the manual box, or both."""

    return {
        "selection_agreement": selection_agreement,
        "physical_selected": bool(physical_selected),
        "manual_selected": bool(manual_selected),
        "plot_selection_method": plot_selection_method,
    }


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


def get_vdf_cellids_in_manual(config, point_record, vdf_cellids, vdf_coords_re):
    """Select VDF cells inside a fixed axis-aligned box around a point."""

    if len(vdf_cellids) == 0:
        return {}

    point_kind = point_record["point_kind"]
    box_config = get_manual_config_re(
        config=config,
        point_kind=point_kind,
    )
    half_widths_re = np.asarray(
        [
            box_config["x_half_width_re"],
            box_config["y_half_width_re"],
            box_config["z_half_width_re"],
        ],
        dtype=float,
    )
    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re

    active_axes = half_widths_re > 0
    if np.any(active_axes):
        selected = np.all(
            np.abs(offsets_re[:, active_axes]) <= half_widths_re[active_axes],
            axis=1,
        )
    else:
        selected = np.ones(len(vdf_cellids), dtype=bool)

    return {
        f"{point_kind}_box_{index:04d}": int(cid)
        for index, cid in enumerate(vdf_cellids[selected])
    }


def get_vdf_cellids_in_hessian_di_box(config, point_record, vdf_cellids, vdf_coords_re):
    """Select VDF cells in a Hessian-eigenvector-aligned X-point box, scaled by local d_i."""

    if len(vdf_cellids) == 0 or point_record.get("di_m") is None:
        return {}

    x_selection = get_point_selection_config(
        config=config,
        point_kind="x",
    )
    half_width_di = x_selection.get("half_width_di", {})
    half_widths_m = np.asarray(
        [
            float(half_width_di["eigenvector_0"]),
            float(half_width_di["eigenvector_1"]),
        ],
        dtype=float,
    ) * float(point_record["di_m"])
    y_half_width_re = float(x_selection.get("y_half_width_re", 0.0))

    center_re = np.asarray(point_record["coord_re"], dtype=float)
    offsets_re = vdf_coords_re - center_re
    offsets_xz_m = offsets_re[:, [0, 2]] * R_EARTH
    eigvecs = np.asarray(point_record["eigvecs"], dtype=float)
    projections_m = offsets_xz_m @ eigvecs

    selected = (
        (np.abs(projections_m[:, 0]) <= half_widths_m[0])
        & (np.abs(projections_m[:, 1]) <= half_widths_m[1])
        & (
            (y_half_width_re <= 0)
            | (np.abs(offsets_re[:, 1]) <= y_half_width_re)
        )
    )

    return {
        f"x_di_{index:04d}": int(cid)
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
            }
        )

    return metadata


def optional_float(value):
    """``float(value)``, or ``nan`` when ``value`` is missing."""

    if value is None:
        return float("nan")

    return float(value)
