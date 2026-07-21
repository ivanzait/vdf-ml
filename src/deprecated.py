"""
DEPRECATED -- old student-authored data-extraction/labeling pipeline.

Everything in this file implements the OLD label scheme
(lobe/exhaust/o_point/x_point/dayside, driven by static class_coords_re plus
manual/Hessian point selection) used by ``scripts/data_proc/create_dataset.py``
and its supporting scripts. It has been superseded by the physics-driven
labeling pipeline in ``src/data_proc/labeling/`` (X/O points from the
topology detector, Shue-model magnetosphere regions), used by
``scripts/data_proc/extract_data.py``.

Nothing here is called by any current tool. It is kept, consolidated into
this single file, only so the still-functional old pipeline
(create_dataset.py / backfill_dataset_metadata.py / plot_dataset_labels.py /
plot_dataset_random_class_samples.py / plot_dataset_all.py) keeps working
unchanged. Do not build new functionality on top of this file -- extend
src/data_proc/labeling/ instead.
"""

import csv
import mmap
import os
import stat
import tempfile
import time
from pathlib import Path

import analysator as pt
import matplotlib
import numpy as np
from joblib import Parallel, delayed
from matplotlib.patches import Polygon, Rectangle

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.data_proc.batches import iter_index_batches
from src.data_proc.config import create_path, create_timestep_path, load_config
from src.data_proc.dataset_io import save_metadata
from src.data_proc.labeling.point_labels import (
    create_point_sample_metadata,
    get_manual_config_re,
    get_point_selection_config,
    get_vdf_cellids_in_flux_contour,
)
from src.data_proc.physics.point_topology import find_point_records
from src.data_proc.plot_tools import (
    draw_o_point_search_areas,
    get_regions_re_boxre,
    plot_prepared_vdf_xz_slice_on_axis,
    plot_vdf_xz_slice,
    plot_vdf_xz_slice_from_physical_xz,
    prepare_vdf_xz_plot,
)
from src.data_proc.vdf_tools import (
    R_EARTH,
    VdfExtractor,
    create_region_mask_re,
    get_b_field,
    get_bulk_velocity,
    get_cellid_with_vdf,
    get_nearest_vdf_cellid,
    get_vdf_cells_with_coords_re,
    get_vdf_plot_axes_parameters,
    get_vdf_plot_threshold,
    iter_enabled_regions_re,
)
from src.data_proc.physics.vdf_transform import (
    DEFAULT_HERMITE_ORDER,
    get_rotated_vdf,
    vdf_to_hermite_spectra,
)


# =====================================================================
# From dataset_metadata.py -- X/O-point distance/vector metadata columns
# for the old label scheme.
# =====================================================================

POINT_REFERENCE_METADATA_COLUMNS = (
    "distance_to_x_point_re",
    "distance_to_o_point_re",
    "vdf_to_x_point_dx_re",
    "vdf_to_x_point_dy_re",
    "vdf_to_x_point_dz_re",
    "vdf_to_o_point_dx_re",
    "vdf_to_o_point_dy_re",
    "vdf_to_o_point_dz_re",
)

OMITTED_DATASET_METADATA_COLUMNS = (
    "simulation_time",
    "neighbor_position",
    "region_name",
    "selection_box_y_half_width_re",
    "rho",
    "core_fraction",
    "selection_agreement",
    "x_point_available",
    "o_point_available",
    # Write-only: computed but never read back by any downstream code.
    # Kept here (rather than only skipped at write time) so backfill also
    # strips them from datasets created before this was true. Note:
    # source_point_y_re is NOT included here even though it's write-only --
    # unlike these, it's still needed internally (add_point_reference_metadata
    # reads it back off sample_spec after this filter already ran once), so
    # it's excluded later instead, in create_sample_metadata_row.
    "vdf_x_re",
    "vdf_y_re",
    "vdf_z_re",
)


def create_point_reference_metadata_arrays(
    vdf_coords_re,
    x_point_coords_re=None,
    o_point_coords_re=None,
    point_kinds=None,
    source_point_coords_re=None,
):
    """
    Compute, for every VDF sample, distance/vector to a reference X point and a reference O point.

    Without ``point_kinds``/``source_point_coords_re``, every sample uses its
    closest detected X point and closest detected O point. When per-sample
    provenance is given, X-selected samples use their own source X point and
    O-selected samples use their own source O point instead (the opposite
    point kind still uses the closest detected point). Vectors point from the
    VDF center to the reference point, in Earth radii.

    Returns a dict of 8 arrays (see ``POINT_REFERENCE_METADATA_COLUMNS``),
    NaN where a sample has no reference point of that kind.
    """

    vdf_coords_re = np.asarray(vdf_coords_re, dtype=float)
    if vdf_coords_re.ndim != 2 or vdf_coords_re.shape[1:] != (3,):
        raise ValueError("vdf_coords_re must have shape (n_samples, 3)")
    if not np.all(np.isfinite(vdf_coords_re)):
        raise ValueError("vdf_coords_re must contain finite coordinates")

    x_point_coords_re = _normalize_point_coords_re(
        x_point_coords_re,
        argument_name="x_point_coords_re",
    )
    o_point_coords_re = _normalize_point_coords_re(
        o_point_coords_re,
        argument_name="o_point_coords_re",
    )

    normalized_point_kinds = None
    normalized_source_coords_re = None
    if point_kinds is not None:
        point_kinds = np.asarray(point_kinds, dtype=object)
        if point_kinds.shape != (len(vdf_coords_re),):
            raise ValueError("point_kinds must have shape (n_samples,)")
        normalized_point_kinds = np.asarray(
            [
                "" if value is None else str(value).strip().lower()
                for value in point_kinds
            ],
            dtype=object,
        )
        invalid_point_kinds = ~np.isin(
            normalized_point_kinds,
            ("", "x", "o"),
        )
        if np.any(invalid_point_kinds):
            invalid_values = sorted(set(normalized_point_kinds[invalid_point_kinds]))
            raise ValueError(f"Unknown point kinds: {invalid_values}")

        if source_point_coords_re is None:
            if np.any(normalized_point_kinds != ""):
                raise ValueError(
                    "source_point_coords_re is required for X/O samples"
                )
        else:
            normalized_source_coords_re = np.asarray(
                source_point_coords_re,
                dtype=float,
            )
            if normalized_source_coords_re.shape != vdf_coords_re.shape:
                raise ValueError(
                    "source_point_coords_re must have shape (n_samples, 3)"
                )
            point_mask = normalized_point_kinds != ""
            if not np.all(np.isfinite(normalized_source_coords_re[point_mask])):
                raise ValueError(
                    "Source coordinates must be finite for X/O samples"
                )
    elif source_point_coords_re is not None:
        raise ValueError(
            "point_kinds is required when source_point_coords_re is provided"
        )

    distances_by_kind = {}
    vectors_by_kind = {}
    for point_kind, point_coords_re in (
        ("x", x_point_coords_re),
        ("o", o_point_coords_re),
    ):
        distances, vectors = _find_nearest_point_geometry(
            vdf_coords_re=vdf_coords_re,
            point_coords_re=point_coords_re,
        )

        if normalized_point_kinds is not None:
            source_mask = normalized_point_kinds == point_kind
            if np.any(source_mask):
                source_vectors = (
                    normalized_source_coords_re[source_mask]
                    - vdf_coords_re[source_mask]
                )
                vectors[source_mask] = source_vectors
                distances[source_mask] = np.linalg.norm(
                    source_vectors,
                    axis=1,
                )

        distances_by_kind[point_kind] = distances
        vectors_by_kind[point_kind] = vectors

    return {
        "distance_to_x_point_re": distances_by_kind["x"],
        "distance_to_o_point_re": distances_by_kind["o"],
        "vdf_to_x_point_dx_re": vectors_by_kind["x"][:, 0],
        "vdf_to_x_point_dy_re": vectors_by_kind["x"][:, 1],
        "vdf_to_x_point_dz_re": vectors_by_kind["x"][:, 2],
        "vdf_to_o_point_dx_re": vectors_by_kind["o"][:, 0],
        "vdf_to_o_point_dy_re": vectors_by_kind["o"][:, 1],
        "vdf_to_o_point_dz_re": vectors_by_kind["o"][:, 2],
    }


def _normalize_point_coords_re(point_coords_re, argument_name):
    """Coerce point coordinates to a finite (n_points, 3) array; None/empty becomes (0, 3)."""

    if point_coords_re is None:
        return np.empty((0, 3), dtype=float)

    point_coords_re = np.asarray(point_coords_re, dtype=float)
    if point_coords_re.size == 0:
        return np.empty((0, 3), dtype=float)
    if point_coords_re.shape == (3,):
        point_coords_re = point_coords_re.reshape(1, 3)
    if point_coords_re.ndim != 2 or point_coords_re.shape[1:] != (3,):
        raise ValueError(f"{argument_name} must have shape (n_points, 3)")
    if not np.all(np.isfinite(point_coords_re)):
        raise ValueError(f"{argument_name} must contain finite coordinates")

    return point_coords_re


def _find_nearest_point_geometry(vdf_coords_re, point_coords_re):
    """For each VDF sample, the distance and vector to its closest point_coords_re entry. NaN if there are none."""

    n_samples = len(vdf_coords_re)
    best_distance_squared = np.full(n_samples, np.inf, dtype=float)
    best_vectors = np.full((n_samples, 3), np.nan, dtype=float)

    for point_coord_re in point_coords_re:
        vectors = point_coord_re - vdf_coords_re
        distance_squared = np.einsum("ij,ij->i", vectors, vectors)
        closer = distance_squared < best_distance_squared
        best_distance_squared[closer] = distance_squared[closer]
        best_vectors[closer] = vectors[closer]

    distances = np.full(n_samples, np.nan, dtype=float)
    has_reference = np.isfinite(best_distance_squared)
    distances[has_reference] = np.sqrt(best_distance_squared[has_reference])
    return distances, best_vectors


def create_vdf_spatial_metadata(
    vdf_coord_re,
    point_kind=None,
    source_point_coord_re=None,
):
    """Distance/vector fields from a VDF center to point_kind's source point (NaN-filled if point_kind is None)."""

    vdf_coord_re = np.asarray(vdf_coord_re, dtype=float)
    if vdf_coord_re.shape != (3,) or not np.all(np.isfinite(vdf_coord_re)):
        raise ValueError("vdf_coord_re must contain three finite coordinates")

    metadata = {
        column: float("nan")
        for column in POINT_REFERENCE_METADATA_COLUMNS
    }

    point_kind = "" if point_kind is None else str(point_kind).strip().lower()
    if point_kind not in {"", "x", "o"}:
        raise ValueError(f"Unknown point_kind: {point_kind!r}")
    if not point_kind:
        return metadata
    if source_point_coord_re is None:
        raise ValueError(
            f"Source coordinates are required for {point_kind}-points"
        )

    source_point_coord_re = np.asarray(source_point_coord_re, dtype=float)
    if (
        source_point_coord_re.shape != (3,)
        or not np.all(np.isfinite(source_point_coord_re))
    ):
        raise ValueError(
            "source_point_coord_re must contain three finite coordinates"
        )
    vector_re = source_point_coord_re - vdf_coord_re
    metadata[f"distance_to_{point_kind}_point_re"] = float(
        np.linalg.norm(vector_re)
    )
    for component_name, component in zip(("dx", "dy", "dz"), vector_re):
        metadata[
            f"vdf_to_{point_kind}_point_{component_name}_re"
        ] = float(component)
    return metadata


def backfill_dataset_spatial_metadata(dataset_dir, config, n_jobs=4):
    """
    Recompute X/O-point distance/vector metadata for an existing dataset, in place.

    Reruns topology detection once per timestep from the configured bulk and
    flux files (all raw detected points are used for closest-point lookups,
    even ones that produced no saved sample) and rewrites
    ``POINT_REFERENCE_METADATA_COLUMNS``. Existing X/O rows keep their own
    source point for the same-kind reference. Also migrates the legacy
    ``reconnection`` class name to ``x_point`` and drops
    ``OMITTED_DATASET_METADATA_COLUMNS``. Every source row is validated
    against ``config`` before anything is recomputed, and the CSV is replaced
    atomically (write to a temp file, read it back to confirm it round-trips,
    then rename over the original) so a failure never leaves a half-written
    ``metadata.csv``.

    Use this instead of recreating a dataset from VLSV files when only the
    point-topology settings changed -- it skips the expensive VDF extraction.
    """

    metadata_path = Path(dataset_dir) / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")

    with open(metadata_path, newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset metadata has no header: {metadata_path}")
        original_fieldnames = list(reader.fieldnames)
        if len(original_fieldnames) != len(set(original_fieldnames)):
            raise ValueError(
                f"Dataset metadata has duplicate columns: {metadata_path}"
            )
        fieldnames = [
            column
            for column in original_fieldnames
            if column not in OMITTED_DATASET_METADATA_COLUMNS
        ]
        rows = list(reader)
    for row in rows:
        for column in OMITTED_DATASET_METADATA_COLUMNS:
            row.pop(column, None)

    if not isinstance(config, dict):
        raise TypeError("config must be a dataset creation config dictionary")
    required_config_keys = {
        "file_template_bulk",
        "file_template_flux",
        "points",
    }
    missing_config_keys = required_config_keys - set(config)
    if missing_config_keys:
        raise ValueError(
            f"Dataset config is missing keys: {sorted(missing_config_keys)}"
        )
    if not isinstance(config["points"], dict):
        raise TypeError("config.points must be a dictionary")

    required_columns = {"timestep", "cid", "class_name", "file_location"}
    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"Dataset metadata is missing columns: {sorted(missing_columns)}"
        )

    source_columns = (
        "source_point_x_re",
        "source_point_y_re",
        "source_point_z_re",
    )
    present_source_columns = set(source_columns) & set(fieldnames)
    if present_source_columns and present_source_columns != set(source_columns):
        raise ValueError(
            "Point source metadata must contain all of "
            "source_point_x_re, source_point_y_re, and source_point_z_re"
        )
    if "point_kind" not in fieldnames and present_source_columns:
        raise ValueError("Point metadata requires a point_kind column")
    has_source_columns = set(source_columns).issubset(fieldnames)
    row_keys = []
    row_indices_by_timestep = {}
    cellids_by_timestep = {}
    for row_number, row in enumerate(rows, start=2):
        class_name = str(row.get("class_name", "")).strip()
        if class_name == "reconnection":
            row["class_name"] = "x_point"
            class_name = "x_point"

        file_location = str(row["file_location"]).strip()
        if file_location == "":
            raise ValueError(f"Metadata row {row_number} has no file_location")
        try:
            timestep = int(row["timestep"])
            cid = int(row["cid"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metadata row {row_number} has invalid timestep or cid"
            ) from error

        configured_bulk_path = create_timestep_path(
            path_template=config["file_template_bulk"],
            timestep=timestep,
        )
        if Path(file_location) != Path(configured_bulk_path):
            raise ValueError(
                f"Metadata row {row_number} bulk file does not match the "
                f"configured template: {file_location!r} != "
                f"{str(configured_bulk_path)!r}"
            )

        point_kind = str(row.get("point_kind", "")).strip().lower()
        if point_kind not in {"", "x", "o"}:
            raise ValueError(
                f"Metadata row {row_number} has invalid point_kind "
                f"{point_kind!r}"
            )
        expected_point_kind = {
            "x_point": "x",
            "o_point": "o",
        }.get(class_name)
        if expected_point_kind is not None and point_kind != expected_point_kind:
            raise ValueError(
                f"Metadata row {row_number} class {class_name!r} requires "
                f"point_kind={expected_point_kind!r} and source coordinates"
            )
        if point_kind and expected_point_kind is None:
            raise ValueError(
                f"Metadata row {row_number} has point_kind={point_kind!r} "
                f"but class {class_name!r} is not an X/O-point class"
            )
        if point_kind and not has_source_columns:
            raise ValueError(
                "X/O-point rows require source_point_x_re, "
                "source_point_y_re, and source_point_z_re"
            )
        source_point_coord_re = None
        if point_kind:
            try:
                source_point_coord_re = tuple(
                    float(row[column]) for column in source_columns
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Metadata row {row_number} has invalid source coordinates"
                ) from error
            if not np.all(np.isfinite(source_point_coord_re)):
                raise ValueError(
                    f"Metadata row {row_number} has invalid source coordinates"
                )

        row_keys.append((timestep, cid, point_kind, source_point_coord_re))
        row_index = len(row_keys) - 1
        row_indices_by_timestep.setdefault(timestep, []).append(row_index)
        cellids_by_timestep.setdefault(timestep, set()).add(cid)

    missing_files = []
    for timestep in sorted(cellids_by_timestep):
        for path_template in (
            config["file_template_bulk"],
            config["file_template_flux"],
        ):
            file_path = create_timestep_path(
                path_template=path_template,
                timestep=timestep,
            )
            if not file_path.is_file():
                missing_files.append(str(file_path))
    if missing_files:
        raise FileNotFoundError(
            f"Configured bulk/flux files do not exist: {missing_files[:5]}"
        )

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")

    print(
        f"Recomputing topology for {len(cellids_by_timestep)} timesteps "
        f"with n_jobs={n_jobs}"
    )
    timestep_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_read_timestep_spatial_data)(
            timestep=timestep,
            cellids=cellids,
            config=config,
        )
        for timestep, cellids in cellids_by_timestep.items()
    )

    for (
        timestep,
        cellids,
        coords_re,
        x_point_coords_re,
        o_point_coords_re,
    ) in timestep_results:
        coord_by_cellid = {
            int(cid): coord_re
            for cid, coord_re in zip(cellids, coords_re)
        }
        row_indices = row_indices_by_timestep[timestep]
        vdf_coords_re = np.asarray(
            [coord_by_cellid[row_keys[index][1]] for index in row_indices],
            dtype=float,
        )
        point_kinds = np.asarray(
            [row_keys[index][2] for index in row_indices],
            dtype=object,
        )
        source_point_coords_re = np.full(vdf_coords_re.shape, np.nan, dtype=float)
        for local_index, row_index in enumerate(row_indices):
            source_point_coord_re = row_keys[row_index][3]
            if source_point_coord_re is not None:
                source_point_coords_re[local_index] = source_point_coord_re

        reference_metadata = create_point_reference_metadata_arrays(
            vdf_coords_re=vdf_coords_re,
            x_point_coords_re=x_point_coords_re,
            o_point_coords_re=o_point_coords_re,
            point_kinds=point_kinds,
            source_point_coords_re=source_point_coords_re,
        )

        for local_index, row_index in enumerate(row_indices):
            row = rows[row_index]
            for column, values in reference_metadata.items():
                row[column] = _format_metadata_value(values[local_index])

    output_fieldnames = fieldnames + [
        column
        for column in POINT_REFERENCE_METADATA_COLUMNS
        if column not in fieldnames
    ]

    original_mode = stat.S_IMODE(metadata_path.stat().st_mode)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(
                temp_file,
                fieldnames=output_fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        with open(temp_path, newline="") as temp_file:
            reader = csv.DictReader(temp_file)
            written_fieldnames = list(reader.fieldnames or [])
            written_rows = list(reader)
        if written_fieldnames != output_fieldnames or written_rows != rows:
            raise RuntimeError("Temporary metadata validation failed")

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, metadata_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    return metadata_path


def _read_timestep_spatial_data(timestep, cellids, config):
    """Rerun topology detection and read VDF-cell coordinates for one timestep. Returns (timestep, cellids, coords_re, x_point_coords_re, o_point_coords_re)."""

    bulk_file_path = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    flux_file_path = create_timestep_path(
        path_template=config["file_template_flux"],
        timestep=timestep,
    )
    cellids = np.asarray(sorted(cellids), dtype=int)
    reader = pt.vlsvfile.VlsvReader(str(bulk_file_path))
    try:
        coords = np.asarray(reader.get_cell_coordinates(cellids), dtype=float)
        if coords.ndim == 1 and len(cellids) == 1:
            coords = coords.reshape(1, 3)
        if coords.shape != (len(cellids), 3):
            raise ValueError("Vector coordinate lookup returned invalid shape")
    except Exception:
        coords = np.asarray(
            [reader.get_cell_coordinates(int(cid)) for cid in cellids],
            dtype=float,
        )

    coords_re = coords / R_EARTH
    if coords_re.shape != (len(cellids), 3) or not np.all(
        np.isfinite(coords_re)
    ):
        raise ValueError(f"Invalid cell coordinates read from {bulk_file_path}")

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_path,
        points_config=config["points"],
    )

    return (
        int(timestep),
        cellids,
        coords_re,
        [point_record["coord_re"] for point_record in x_point_records],
        [point_record["coord_re"] for point_record in o_point_records],
    )


def _format_metadata_value(value):
    """CSV representation of one derived metric: repr(float), or '' for NaN."""

    value = float(value)
    return "" if np.isnan(value) else repr(value)


# =====================================================================
# From point_selection.py -- old-scheme point selection (manual/Hessian
# boxes, union of physical+manual selection with agreement metadata).
# =====================================================================

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


# =====================================================================
# From dataset_sampling.py -- old-scheme sample-spec planning/extraction.
# =====================================================================

def create_timestep_sample_specs_for_timestep(config, timestep):
    """Plan sample specs for one timestep: open a reader, find labeled/point coords, build specs. Returns (timestep, sample_specs)."""

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep,
    )
    planning_start = time.perf_counter()

    reader_start = time.perf_counter()
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    reader_elapsed = time.perf_counter() - reader_start

    vdf_cells_start = time.perf_counter()
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
    vdf_cells_elapsed = time.perf_counter() - vdf_cells_start

    point_start = time.perf_counter()
    labeled_coords = list(iter_labeled_coords(config))
    point_label_data = create_point_label_data(
        config=config,
        timestep=timestep,
        reader=reader,
    )
    labeled_coords.extend(point_label_data["point_labeled_coords"])
    point_elapsed = time.perf_counter() - point_start

    specs_start = time.perf_counter()
    sample_specs = create_timestep_sample_specs(
        config=config,
        timestep=timestep,
        labeled_coords=labeled_coords,
        rejected_cellids=point_label_data["rejected_cellids"],
        reader=reader,
        vdf_cellids=vdf_cellids,
        vdf_coords_re=vdf_coords_re,
        raw_x_point_records=point_label_data["raw_x_point_records"],
        raw_o_point_records=point_label_data["raw_o_point_records"],
    )
    hermite_config = config.get("hermite", {})
    hermite_enabled = bool(hermite_config.get("enabled", False))
    hermite_order = int(hermite_config.get("order", DEFAULT_HERMITE_ORDER))
    hermite_rotate = bool(hermite_config.get("rotate", False))
    for sample_spec in sample_specs:
        sample_spec["hermite_enabled"] = hermite_enabled
        sample_spec["hermite_order"] = hermite_order
        sample_spec["hermite_rotate"] = hermite_rotate

    specs_elapsed = time.perf_counter() - specs_start
    planning_elapsed = time.perf_counter() - planning_start

    print(
        f"Timestep {int(timestep)} planning: "
        f"reader={reader_elapsed:.2f}s, "
        f"vdf_cells={vdf_cells_elapsed:.2f}s, "
        f"points={point_elapsed:.2f}s, "
        f"samples={specs_elapsed:.2f}s, "
        f"total={planning_elapsed:.2f}s, "
        f"n={len(sample_specs)}"
    )

    return int(timestep), sample_specs


def create_timestep_sample_specs(
    config,
    timestep,
    labeled_coords,
    rejected_cellids=None,
    reader=None,
    vdf_cellids=None,
    vdf_coords_re=None,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """
    Build VDF sample specs for one timestep.

    ``labeled_coords`` mixes static (class_name, label, coord_re) tuples with
    detected X/O point records; each point record is expanded into one or
    more VDF cells via ``get_point_selection_result`` (physical + manual
    union). Background classes then fill unclaimed cells in the configured
    sampling regions. Cell IDs claimed by more than one class are dropped
    from all of them (see ``find_conflicting_cellids``).

    ``raw_x_point_records``/``raw_o_point_records`` are the full topology
    detection output (including points that produced no VDF sample) and are
    used only for nearest-point distance/vector metadata
    (``add_point_reference_metadata``); if omitted, this falls back to just
    the point records that made it into ``labeled_coords``.

    Returns a list of sample spec dicts.
    """

    file_location = create_timestep_path(
        path_template=config["file_template_bulk"],
        timestep=timestep
    )

    if reader is None:
        reader = pt.vlsvfile.VlsvReader(str(file_location))

    labeled_coords = list(labeled_coords)
    if raw_x_point_records is None:
        raw_x_point_records = [
            labeled_coord
            for labeled_coord in labeled_coords
            if is_point_record(labeled_coord)
            and str(labeled_coord.get("point_kind", "")).lower() == "x"
        ]
    else:
        raw_x_point_records = list(raw_x_point_records)
    if raw_o_point_records is None:
        raw_o_point_records = [
            labeled_coord
            for labeled_coord in labeled_coords
            if is_point_record(labeled_coord)
            and str(labeled_coord.get("point_kind", "")).lower() == "o"
        ]
    else:
        raw_o_point_records = list(raw_o_point_records)
    rejected_cellids = {int(cid) for cid in (rejected_cellids or set())}
    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
    vdf_coord_by_cellid = {
        int(cid): tuple(float(value) for value in coord_re)
        for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
    }

    sample_specs = []
    seen_class_cellids = set()

    for labeled_coord in labeled_coords:
        class_name, label, coord_re = unpack_labeled_coord(labeled_coord)
        sample_metadata = {}
        metadata_by_position = {}

        if is_point_record(labeled_coord):
            selection_result = get_point_selection_result(
                config=config,
                point_record=labeled_coord,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            cellids_by_position = selection_result["cellids_by_position"]
            rejected_cellids.update(selection_result["rejected_cellids"])
            metadata_by_position = selection_result["metadata_by_position"]
            sample_metadata = create_point_sample_metadata(
                config=config,
                point_record=labeled_coord,
            )
        else:
            cid = get_nearest_vdf_cellid(
                coord_re=coord_re,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
            cellids_by_position = {"closest": cid}

        for neighbor_position, cid in cellids_by_position.items():
            class_cellid = (class_name, int(cid))

            if class_cellid in seen_class_cellids:
                continue

            seen_class_cellids.add(class_cellid)
            sample_spec = {
                "file_location": file_location,
                "cid": int(cid),
                "label": int(label),
                "class_name": class_name,
                "coord_re": coord_re,
                "vdf_coord_re": vdf_coord_by_cellid[int(cid)],
                "timestep": int(timestep),
            }
            sample_spec.update(
                {
                    key: value
                    for key, value in sample_metadata.items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_spec.update(
                {
                    key: value
                    for key, value in metadata_by_position.get(
                        neighbor_position,
                        {},
                    ).items()
                    if key not in OMITTED_DATASET_METADATA_COLUMNS
                }
            )
            sample_specs.append(sample_spec)

    conflicting_cellids = find_conflicting_cellids(sample_specs)
    rejected_cellids.update(conflicting_cellids)
    sample_specs = remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=rejected_cellids,
    )

    sample_specs.extend(
        create_background_region_sample_specs(
            config=config,
            reader=reader,
            file_location=file_location,
            existing_sample_specs=sample_specs,
            timestep=timestep,
            rejected_cellids=rejected_cellids,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
            vdf_coord_by_cellid=vdf_coord_by_cellid,
        )
    )

    sample_specs = remove_conflicting_cellids(
        sample_specs=sample_specs,
        conflicting_cellids=find_conflicting_cellids(sample_specs),
    )
    add_point_reference_metadata(
        sample_specs=sample_specs,
        raw_x_point_records=raw_x_point_records,
        raw_o_point_records=raw_o_point_records,
    )
    return sample_specs


def add_point_reference_metadata(
    sample_specs,
    raw_x_point_records=None,
    raw_o_point_records=None,
):
    """
    Add source-or-nearest X/O distance/vector fields to sample specs, in place.

    X-selected samples measure from their own source X point; O-selected
    samples from their own source O point. Every other sample measures from
    the closest raw detected point of each kind. See
    ``create_point_reference_metadata_arrays`` for the actual computation.
    """

    if not sample_specs:
        return sample_specs

    raw_x_point_records = list(raw_x_point_records or [])
    raw_o_point_records = list(raw_o_point_records or [])
    vdf_coords_re = np.asarray(
        [sample_spec["vdf_coord_re"] for sample_spec in sample_specs],
        dtype=float,
    )
    point_kinds = np.empty(len(sample_specs), dtype=object)
    source_point_coords_re = np.full(vdf_coords_re.shape, np.nan, dtype=float)

    for index, sample_spec in enumerate(sample_specs):
        point_kind = sample_spec.get("point_kind")
        point_kind = "" if point_kind is None else str(point_kind).strip().lower()
        point_kinds[index] = point_kind
        if point_kind in {"x", "o"}:
            source_point_coords_re[index] = (
                sample_spec["source_point_x_re"],
                sample_spec["source_point_y_re"],
                sample_spec["source_point_z_re"],
            )

    reference_metadata = create_point_reference_metadata_arrays(
        vdf_coords_re=vdf_coords_re,
        x_point_coords_re=[
            point_record["coord_re"]
            for point_record in raw_x_point_records
        ],
        o_point_coords_re=[
            point_record["coord_re"]
            for point_record in raw_o_point_records
        ],
        point_kinds=point_kinds,
        source_point_coords_re=source_point_coords_re,
    )

    for column, values in reference_metadata.items():
        for sample_spec, value in zip(sample_specs, values):
            sample_spec[column] = float(value)

    return sample_specs


def create_background_region_sample_specs(
    config,
    reader,
    file_location,
    existing_sample_specs,
    timestep,
    rejected_cellids=None,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
    vdf_coord_by_cellid=None,
):
    """Fill background-class samples from configured spatial regions, excluding cells already claimed by point/static classes."""

    labels = config["labels"]
    excluded_cellids = {
        int(sample_spec["cid"])
        for sample_spec in existing_sample_specs
    }
    excluded_cellids.update(int(cid) for cid in (rejected_cellids or set()))
    seen_background_class_cellids = set()
    sample_specs = []
    if vdf_coord_by_cellid is None:
        if vdf_cellids is not None and vdf_coords_re is not None:
            vdf_coord_by_cellid = {
                int(cid): tuple(float(value) for value in coord_re)
                for cid, coord_re in zip(vdf_cellids, vdf_coords_re)
            }
        else:
            vdf_coord_by_cellid = {}

    points_config = config.get("points", {})
    for region_name, region_re in iter_enabled_regions_re(
            points_config=points_config,
            names_key="background_region_names",
    ):
        class_name = get_region_background_class_name(
            points_config=points_config,
            region_name=region_name,
            region_re=region_re,
        )
        if class_name not in labels:
            raise ValueError(
                f"Region {region_name!r} uses background class "
                f"{class_name!r}, but labels has no such class"
            )
        label = int(labels[class_name])

        for cid in get_vdf_cellids_in_region_re(
            reader=reader,
            region_re=region_re,
            cell_has_vdf_func=cell_has_vdf_func,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        ):
            cid = int(cid)
            background_class_cellid = (class_name, cid)
            if (
                    cid in excluded_cellids
                    or background_class_cellid in seen_background_class_cellids
            ):
                continue

            seen_background_class_cellids.add(background_class_cellid)
            coord_re = vdf_coord_by_cellid.get(cid)
            if coord_re is None:
                coord_re = (
                    np.asarray(reader.get_cell_coordinates(cid), dtype=float)
                    / R_EARTH
                )
            sample_specs.append(
                {
                    "file_location": file_location,
                    "cid": cid,
                    "label": label,
                    "class_name": class_name,
                    "coord_re": coord_re,
                    "vdf_coord_re": coord_re,
                    "timestep": int(timestep),
                }
            )

    return sample_specs


def get_region_background_class_name(points_config, region_name, region_re):
    """Background class name for a region: region_re.background_class_name, else points.background_class_names[name], else 'other' for tail, else the region name."""

    class_name = region_re.get("background_class_name")
    if class_name is not None:
        return str(class_name)

    class_names = (points_config or {}).get("background_class_names", {})
    if region_name in class_names:
        return str(class_names[region_name])

    if region_name == "tail":
        return str((points_config or {}).get("background_class_name", "other"))

    return str(region_name)


def get_vdf_cellids_in_region_re(
    reader,
    region_re,
    cell_has_vdf_func=None,
    vdf_cellids=None,
    vdf_coords_re=None,
):
    """VDF cell IDs inside one configured spatial region."""

    if vdf_cellids is None or vdf_coords_re is None:
        vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)

    if len(vdf_cellids) == 0:
        return []

    selected = create_region_mask_re(vdf_coords_re, region_re)

    selected_cellids = vdf_cellids[selected]
    if cell_has_vdf_func is not None:
        selected_cellids = [
            int(cid)
            for cid in selected_cellids
            if cell_has_vdf_func(int(cid))
        ]

    return [int(cid) for cid in selected_cellids]


def find_conflicting_cellids(sample_specs):
    """Cell IDs assigned to more than one class name."""

    class_names_by_cellid = {}

    for sample_spec in sample_specs:
        cid = int(sample_spec["cid"])
        class_name = sample_spec["class_name"]
        class_names_by_cellid.setdefault(cid, set()).add(class_name)

    return {
        cid
        for cid, class_names in class_names_by_cellid.items()
        if len(class_names) > 1
    }


def remove_conflicting_cellids(sample_specs, conflicting_cellids):
    """Drop every sample spec whose cell ID is in conflicting_cellids."""

    if not conflicting_cellids:
        return sample_specs

    conflicting_cellids = {int(cid) for cid in conflicting_cellids}

    return [
        sample_spec
        for sample_spec in sample_specs
        if int(sample_spec["cid"]) not in conflicting_cellids
    ]


def iter_timestep_sample_specs(sample_specs):
    """Extract each planned sample's VDF (raw or Hermite spectra) and metadata. Yields {vdf, label, metadata}."""

    if not sample_specs:
        return

    file_location = sample_specs[0]["file_location"]
    timestep = int(sample_specs[0].get("timestep", 0))
    extraction_start = time.perf_counter()

    print_memory_usage(f"timestep {timestep} before reader")
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    extractor = VdfExtractor(reader=reader)
    hermite_enabled = bool(sample_specs[0].get("hermite_enabled", False))
    v_limits = None
    hermite_order = DEFAULT_HERMITE_ORDER
    hermite_rotate = False
    if hermite_enabled:
        v_limits, _dv = get_vdf_plot_axes_parameters(
            reader=reader,
            vdf_shape=extractor.vdf_shape,
        )
        hermite_order = int(
            sample_specs[0].get("hermite_order", DEFAULT_HERMITE_ORDER)
        )
        hermite_rotate = bool(sample_specs[0].get("hermite_rotate", False))
    print_memory_usage(f"timestep {timestep} after reader")

    print(f"Timestep {timestep}: extracting {len(sample_specs)} samples")

    for sample_spec in sample_specs:
        coord_re = sample_spec["coord_re"]
        cid = int(sample_spec["cid"])

        vdf = extractor.extract(cid=cid)
        if hermite_enabled:
            vdf_for_hermite = vdf
            shape_for_hermite = extractor.vdf_shape
            v_limits_for_hermite = v_limits
            if hermite_rotate:
                b_field = get_b_field(reader=reader, cid=cid)
                bulk_velocity = get_bulk_velocity(reader=reader, cid=cid)
                (
                    vdf_for_hermite,
                    shape_for_hermite,
                    v_limits_for_hermite,
                    _rotation_matrix,
                ) = get_rotated_vdf(
                    vdf=vdf,
                    shape=extractor.vdf_shape,
                    v_limits=v_limits,
                    b_field=b_field,
                    bulk_velocity=bulk_velocity,
                )
            sample_array = vdf_to_hermite_spectra(
                vdf=vdf_for_hermite,
                shape=shape_for_hermite,
                v_limits=v_limits_for_hermite,
                order=hermite_order,
            ).astype(np.float32, copy=False)
        else:
            sample_array = vdf.astype(np.float32, copy=False)

        yield {
            "vdf": sample_array,
            "label": sample_spec["label"],
            "metadata": create_sample_metadata_row(
                sample_spec=sample_spec,
                cid=cid,
                coord_re=coord_re,
                file_location=file_location,
            ),
        }

    extraction_elapsed = time.perf_counter() - extraction_start
    print_memory_usage(f"timestep {timestep} extraction complete")
    print(f"Timestep {timestep} extraction: {extraction_elapsed:.2f} s")


def create_sample_metadata_row(sample_spec, cid, coord_re, file_location):
    """Build one metadata.csv row: identity fields + X/O-point distance/vector fields."""

    metadata_row = {
        "timestep": int(sample_spec["timestep"]),
        "cid": int(cid),
        "label": sample_spec["label"],
        "class_name": sample_spec["class_name"],
        "x_re": float(coord_re[0]),
        "y_re": float(coord_re[1]),
        "z_re": float(coord_re[2]),
        "file_location": str(file_location),
    }
    point_kind = sample_spec.get("point_kind")
    if point_kind is not None:
        point_kind = str(point_kind).strip().lower()
    source_point_coord_re = None
    if point_kind in {"x", "o"}:
        source_point_coord_re = (
            sample_spec["source_point_x_re"],
            sample_spec["source_point_y_re"],
            sample_spec["source_point_z_re"],
        )
    if all(
        column in sample_spec
        for column in POINT_REFERENCE_METADATA_COLUMNS
    ):
        spatial_metadata = {
            column: sample_spec[column]
            for column in POINT_REFERENCE_METADATA_COLUMNS
        }
    else:
        spatial_metadata = create_vdf_spatial_metadata(
            vdf_coord_re=sample_spec["vdf_coord_re"],
            point_kind=point_kind,
            source_point_coord_re=source_point_coord_re,
        )
    internal_keys = {
        "file_location",
        "cid",
        "label",
        "class_name",
        "coord_re",
        "vdf_coord_re",
        "timestep",
        "hermite_order",
        "hermite_enabled",
        "hermite_rotate",
        # source_point_y_re is write-only (unlike source_point_x_re/z_re,
        # which draw_manual_point_search_boxes below reads back): needed
        # internally above and by add_point_reference_metadata, but
        # excluded from metadata.csv here.
        "source_point_y_re",
        *OMITTED_DATASET_METADATA_COLUMNS,
        *POINT_REFERENCE_METADATA_COLUMNS,
    }

    for key, value in sample_spec.items():
        if key not in internal_keys:
            metadata_row[key] = value

    metadata_row.update(spatial_metadata)

    return metadata_row


def write_timestep_samples(X, y, metadata, timestep_samples, sample_index):
    """Write extracted samples into the output X/y arrays and append their metadata rows. Returns the next sample_index."""

    for sample in timestep_samples:
        X[sample_index] = sample["vdf"]
        y[sample_index] = sample["label"]

        metadata_row = {"sample_index": sample_index}
        metadata_row.update(sample["metadata"])
        metadata.append(metadata_row)

        sample_index += 1

    return sample_index


def iter_chunks(items, chunk_size):
    """Yield consecutive slices of items with at most chunk_size entries."""

    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def get_point_class_names(config):
    """Configured X/O point class names, from points.x_class_name/o_class_name."""

    points_config = config.get("points", {})
    class_names = {
        points_config.get("x_class_name"),
        points_config.get("o_class_name"),
    }

    return {class_name for class_name in class_names if class_name is not None}


def print_memory_usage(label):
    """Print current process resident memory usage, tagged with label."""

    print(f"Memory {label}: {get_memory_usage_mb():.1f} MB")


def get_memory_usage_mb():
    """Current process resident memory usage in MB."""

    import resource
    import sys

    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    value_kb = float(line.split()[1])
                    return value_kb / 1024.0
        return 0.0
    except FileNotFoundError:
        # /proc is Linux-only; fall back to getrusage for other platforms
        # (e.g. macOS, where ru_maxrss is reported in bytes, not KB).
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return max_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else max_rss / 1024.0


# =====================================================================
# From dataset_extraction.py -- old-scheme memmap-batched extraction
# (serial and parallel-by-timestep-chunk paths).
# =====================================================================

def release_memmap_pages(array):

    mmap_object = getattr(array, "_mmap", None)
    madvise = getattr(mmap_object, "madvise", None)
    dontneed = getattr(mmap, "MADV_DONTNEED", None)

    if madvise is None or dontneed is None:
        return False

    try:
        madvise(dontneed)
    except (OSError, ValueError):
        return False

    return True

def flush_and_release_memmaps(*arrays):

    for array in arrays:
        array.flush()

    for array in arrays:
        release_memmap_pages(array)

def get_worker_count(n_jobs, config_name):

    if n_jobs == 0:
        raise ValueError(f"{config_name} must be non-zero")

    if n_jobs < 0:
        return os.cpu_count() or 1

    return max(1, n_jobs)

def plan_dataset_sample_specs(config, timesteps, planning_n_jobs):

    print_memory_usage("before planning")
    planning_start = time.perf_counter()

    if planning_n_jobs == 1:
        sample_spec_results = [
            create_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        ]
    else:
        sample_spec_results = Parallel(n_jobs=planning_n_jobs)(
            delayed(create_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )

    return dict(sample_spec_results), time.perf_counter() - planning_start

def find_first_nonempty_timestep(sample_counts_by_timestep, timesteps):

    for timestep_index, timestep in enumerate(timesteps):
        if sample_counts_by_timestep[int(timestep)] > 0:
            return timestep_index, int(timestep)

    raise ValueError("No samples were found for the requested timesteps")

def extract_first_sample_from_specs(sample_specs):

    print_memory_usage("before first extraction")
    sample_iter = iter_timestep_sample_specs(sample_specs)

    try:
        first_sample = next(sample_iter)
    except StopIteration as error:
        raise ValueError("No samples were found for the first timestep") from error

    print_memory_usage("after first extraction")

    return first_sample, sample_iter

def write_remaining_timesteps(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):

    if extraction_n_jobs == 1:
        return write_timesteps_serial(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=X,
            y=y,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
        )

    return write_timesteps_parallel(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )

def write_timesteps_serial(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
):

    for timestep in timesteps:
        sample_specs = sample_specs_by_timestep.pop(int(timestep))
        sample_index = write_timestep_samples(
            X=X,
            y=y,
            metadata=metadata,
            timestep_samples=iter_timestep_sample_specs(sample_specs),
            sample_index=sample_index,
        )
        flush_and_release_memmaps(X, y)
        print_memory_usage(f"after timestep {int(timestep)} memmap release")

    return sample_index

def write_timesteps_parallel(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):

    with tempfile.TemporaryDirectory(
        prefix="extraction_",
        dir=Path(X.filename).parent,
    ) as temp_dir:
        temp_dir = Path(temp_dir)

        for timestep_chunk in iter_chunks(timesteps, extraction_worker_count):
            chunk_start = int(timestep_chunk[0])
            chunk_end = int(timestep_chunk[-1])
            print_memory_usage(
                f"before extraction chunk {chunk_start}-{chunk_end}"
            )
            chunk_specs_by_timestep = {
                int(timestep): sample_specs_by_timestep.pop(int(timestep))
                for timestep in timestep_chunk
            }
            chunk_results = extract_timestep_chunk_parallel(
                sample_specs_by_timestep=chunk_specs_by_timestep,
                timestep_chunk=timestep_chunk,
                extraction_n_jobs=extraction_n_jobs,
                temp_dir=temp_dir,
                sample_shape=X.shape[1:],
                dtype=X.dtype,
            )
            print_memory_usage(
                f"after extraction chunk {chunk_start}-{chunk_end}"
            )

            for extracted_timestep in chunk_results:
                sample_index = write_extracted_timestep(
                    X=X,
                    y=y,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    sample_index=sample_index,
                )

            print_memory_usage(f"after writing chunk {chunk_start}-{chunk_end}")
            flush_and_release_memmaps(X, y)
            print_memory_usage(
                f"after memmap release chunk {chunk_start}-{chunk_end}"
            )

    return sample_index

def extract_timestep_samples_to_temp(sample_specs, temp_dir, sample_shape, dtype):

    if not sample_specs:
        return {
            "n_samples": 0,
            "metadata": [],
            "X_path": None,
            "y_path": None,
        }

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir)
    X_path = temp_dir / f"timestep_{timestep}_X.npy"
    y_path = temp_dir / f"timestep_{timestep}_y.npy"

    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *sample_shape),
    )
    y_temp = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(len(sample_specs),),
    )

    metadata = []
    sample_index = write_timestep_samples(
        X=X_temp,
        y=y_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(sample_specs),
        sample_index=0,
    )

    if sample_index != len(sample_specs):
        raise RuntimeError(
            f"Expected to extract {len(sample_specs)} samples for timestep "
            f"{timestep}, extracted {sample_index}"
        )

    flush_and_release_memmaps(X_temp, y_temp)

    return {
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
        "y_path": str(y_path),
    }

def write_extracted_timestep(X, y, metadata, extracted_timestep, sample_index):

    n_samples = int(extracted_timestep["n_samples"])
    if n_samples == 0:
        return sample_index

    timestep_metadata = extracted_timestep["metadata"]
    if len(timestep_metadata) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} metadata rows, got {len(timestep_metadata)}"
        )

    X_temp = np.load(extracted_timestep["X_path"], mmap_mode="r")
    y_temp = np.load(extracted_timestep["y_path"], mmap_mode="r")
    write_end = sample_index + n_samples

    X[sample_index:write_end] = X_temp[:n_samples]
    y[sample_index:write_end] = y_temp[:n_samples]

    for metadata_row in timestep_metadata:
        output_row = dict(metadata_row)
        output_row["sample_index"] = sample_index + int(
            metadata_row["sample_index"]
        )
        metadata.append(output_row)

    release_memmap_pages(X_temp)
    release_memmap_pages(y_temp)

    return write_end

def extract_timestep_chunk_parallel(
    sample_specs_by_timestep,
    timestep_chunk,
    extraction_n_jobs,
    temp_dir,
    sample_shape,
    dtype,
):

    try:
        parallel = Parallel(n_jobs=extraction_n_jobs, return_as="generator")
    except TypeError:
        parallel = Parallel(n_jobs=extraction_n_jobs)

    return parallel(
        delayed(extract_timestep_samples_to_temp)(
            sample_specs=sample_specs_by_timestep[int(timestep)],
            temp_dir=temp_dir,
            sample_shape=sample_shape,
            dtype=dtype,
        )
        for timestep in timestep_chunk
    )


# =====================================================================
# From dataset_creation.py -- old-scheme dataset creation orchestration
# (create_dataset() + its memmap helper; load_dataset/save_metadata/
# print_vdf_statistics are generic and live on in src/data_proc/dataset_io.py
# instead; print_dataset_info was removed with inspect_dataset.py).
# =====================================================================

def create_dataset(
    config,
    start_timestep,
    n_timesteps,
    dataset_kind,
):
    """
    Create and save a labeled VDF dataset.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    start_timestep : int
        First timestep to include.
    n_timesteps : int
        Number of consecutive timesteps to process.
    dataset_kind : {"train", "test"}
        Output dataset split name.
    """

    total_start = time.perf_counter()
    timesteps = list(range(start_timestep, start_timestep + n_timesteps))

    output_dirs = config["output_dirs"]
    output_dir = output_dirs[dataset_kind]

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset kind: {dataset_kind}")
    print(f"Output directory: {outdir}")

    creation_config = config.get("creation", {})
    default_n_jobs = int(creation_config.get("n_jobs", 1))
    planning_n_jobs = int(
        creation_config.get("planning_n_jobs", default_n_jobs)
    )
    extraction_n_jobs = int(
        creation_config.get("extraction_n_jobs", 1)
    )

    planning_worker_count = get_worker_count(
        planning_n_jobs,
        "creation.planning_n_jobs",
    )
    extraction_worker_count = get_worker_count(
        extraction_n_jobs,
        "creation.extraction_n_jobs",
    )

    print(
        f"Planning jobs: {planning_n_jobs} "
        f"({planning_worker_count} workers)"
    )
    print(
        f"Extraction jobs: {extraction_n_jobs} "
        f"({extraction_worker_count} workers)"
    )

    sample_specs_by_timestep, planning_elapsed = plan_dataset_sample_specs(
        config=config,
        timesteps=timesteps,
        planning_n_jobs=planning_n_jobs,
    )
    sample_counts_by_timestep = {
        int(timestep): len(sample_specs)
        for timestep, sample_specs in sample_specs_by_timestep.items()
    }
    n_samples = sum(sample_counts_by_timestep.values())

    print(f"Samples: {n_samples}")
    print(f"Timing planning: {planning_elapsed:.2f} s")
    print_memory_usage("after planning")

    if n_samples == 0:
        raise ValueError("No samples were found for the requested timesteps")

    extraction_start = time.perf_counter()
    metadata = []
    sample_index = 0

    first_timestep_index, first_timestep = find_first_nonempty_timestep(
        sample_counts_by_timestep=sample_counts_by_timestep,
        timesteps=timesteps,
    )
    first_sample_specs = sample_specs_by_timestep[first_timestep]
    first_sample, first_sample_iter = extract_first_sample_from_specs(
        first_sample_specs
    )

    X, y = create_memmap_dataset(
        outdir=outdir,
        n_samples=n_samples,
        sample_shape=first_sample["vdf"].shape,
        dtype=np.float32,
    )
    print_memory_usage("after memmap creation")

    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=[first_sample],
        sample_index=sample_index,
    )
    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=first_sample_iter,
        sample_index=sample_index,
    )
    sample_specs_by_timestep.pop(first_timestep, None)
    flush_and_release_memmaps(X, y)
    print_memory_usage("after first timestep memmap release")

    remaining_timesteps = timesteps[first_timestep_index + 1:]
    sample_index = write_remaining_timesteps(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=remaining_timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )

    if sample_index != n_samples:
        raise RuntimeError(
            f"Expected to write {n_samples} samples, wrote {sample_index}"
        )

    extraction_elapsed = time.perf_counter() - extraction_start
    save_start = time.perf_counter()

    print_memory_usage("before flush")
    flush_and_release_memmaps(X, y)
    print_memory_usage("after flush and memmap release")

    save_metadata(
        outdir=outdir,
        metadata=metadata,
    )

    save_elapsed = time.perf_counter() - save_start
    total_elapsed = time.perf_counter() - total_start

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Samples written: {sample_index}")
    print(f"Timing extraction/write: {extraction_elapsed:.2f} s")
    print(f"Timing save/flush: {save_elapsed:.2f} s")
    print(f"Timing total: {total_elapsed:.2f} s")

    print(f"Saved X: {outdir / 'X.npy'}")
    print(f"Saved y: {outdir / 'y.npy'}")
    print(f"Saved metadata: {outdir / 'metadata.csv'}")


def create_memmap_dataset(outdir, n_samples, sample_shape, dtype=np.float32):
    """
    Create memory-mapped dataset file.

    Parameters
    ----------
    outdir : str
        Directory where dataset file is saved.
    n_samples : int
        Number of samples in the dataset.
    sample_shape : tuple of int
        Shape of one VDF sample.
    dtype : data-type, optional
        Desired data type for the array.

    Returns
    -------
    X : numpy.memmap
        Memory mapped array for the VDF.
    y : numpy.memmap
        Memory mapped array for the labels.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.lib.format.open_memmap(
        outdir / "X.npy",
        mode="w+",
        dtype=dtype,
        shape=(int(n_samples), *sample_shape)
    )

    y = np.lib.format.open_memmap(
        outdir / "y.npy",
        mode="w+",
        dtype=np.int64,
        shape=(int(n_samples),),
    )

    return X, y


# =====================================================================
# Old-scheme plot overlays (manual/Hessian search boxes, source-class
# scatter styles). draw_o_point_search_areas and get_o_point_search_flux
# stay live in src/data_proc/plot_tools.py since plot_snapshot_topology
# (current) still uses draw_o_point_search_areas for the flux-contour
# O-selection display toggle.
# =====================================================================

SOURCE_POINT_STYLES = {
    "x_point": {
        "color": "blue",
        "marker": "x",
        "label": "X point",
        "s": 10,
    },
    "o_point": {
        "edgecolor": "blue",
        "facecolor": "none",
        "marker": "o",
        "label": "o point",
        "s": 14,
        "linewidths": 1.0,
    },
    "exhaust": {
        "color": "red",
        "marker": "s",
        "label": "exhaust",
        "s": 10,
    },
    "dayside": {
        "color": "green",
        "marker": "^",
        "label": "dayside",
        "s": 10,
    },
    "other": {
        "color": "red",
        "marker": "s",
        "label": "other",
        "s": 10,
    },
    "lobe": {"color": "blue", "marker": "2", "label": "lobe", "s": 16},
}


def expr_velocity(exprmaps, requestvariables=False):
    """
    Return bulk velocity for Analysator colormap plotting.

    Parameters
    ----------
    exprmaps : dict
        Variables requested from the VLSV file.
    requestvariables : bool, optional
        If true, return the required variable names.

    Returns
    -------
    list[str] or numpy.ndarray
        Required variable names, or the bulk velocity vector field.
    """

    if requestvariables is True:
        return ["rho", "rho_v"]

    rho = exprmaps["rho"][:, :]
    rhov = exprmaps["rho_v"][:, :, :]

    return rhov / rho[:, :, None]


def scatter_all_vdf_cells(ax, reader, boxre=None):
    """
    Scatter all VDF-containing spatial cells on an xz colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where points are drawn.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    boxre : list of float, optional
        Plot box in Earth radii: ``[xmin, xmax, zmin, zmax]``. If provided,
        only VDF cells inside the visible xz range are drawn.
    """

    cellids, coords_re = get_vdf_cells_with_coords_re(reader)
    if len(cellids) == 0:
        return

    if boxre is not None:
        x_min, x_max, z_min, z_max = [float(value) for value in boxre]
        visible = (
            (coords_re[:, 0] >= x_min)
            & (coords_re[:, 0] <= x_max)
            & (coords_re[:, 2] >= z_min)
            & (coords_re[:, 2] <= z_max)
        )
        coords_re = coords_re[visible]

    if len(coords_re) == 0:
        return

    ax.scatter(
        coords_re[:, 0],
        coords_re[:, 2],
        label="VDF cell",
        marker=".",
        s=8,
        color="gold",
        linewidths=0,
        zorder=3,
    )


def draw_point_boxes(ax, metadata_rows, box_config, box_classes):
    """
    Draw configured xz boxes around source X/O point coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where boxes are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    box_config : dict
        Box half-widths in Earth radii.
    box_classes : iterable of str
        Classes whose source coordinates should get boxes.
    """

    if not box_config or not box_classes:
        return

    half_width_x = float(box_config["x_half_width_re"])
    half_width_z = float(box_config["z_half_width_re"])
    box_classes = set(box_classes)
    box_rows = metadata_rows[
        metadata_rows["class_name"].isin(box_classes)
    ].drop_duplicates(["class_name", "x_re", "z_re"])

    for box_index, (_, row) in enumerate(box_rows.iterrows()):
        label = "box" if box_index == 0 else None

        rectangle = Rectangle(
            (row["x_re"] - half_width_x, row["z_re"] - half_width_z),
            2 * half_width_x,
            2 * half_width_z,
            fill=False,
            edgecolor="red",
            linewidth=1.0,
            label=label,
        )
        ax.add_patch(rectangle)


def draw_manual_point_search_boxes(ax, metadata_rows):
    """
    Draw manual fixed search boxes from point-selection metadata.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where boxes are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    """

    required_columns = {
        "selection_method",
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "selection_box_x_half_width_re",
        "selection_box_z_half_width_re",
    }
    if not required_columns.issubset(metadata_rows.columns):
        return

    box_mask = _metadata_method_mask(
        metadata_rows=metadata_rows,
        column="selection_method",
        method="manual",
    )
    box_mask |= _metadata_method_mask(
        metadata_rows=metadata_rows,
        column="plot_selection_method",
        method="manual",
    )
    box_mask |= _metadata_bool_mask(
        metadata_rows=metadata_rows,
        column="manual_selected",
    )

    box_rows = metadata_rows[box_mask].drop_duplicates(
        [
            "point_kind",
            "source_point_x_re",
            "source_point_z_re",
            "selection_box_x_half_width_re",
            "selection_box_z_half_width_re",
        ]
    )
    if box_rows.empty:
        return

    plotted_labels = set()
    colors = {"x": "tab:blue", "o": "tab:blue"}
    combined_methods = {"consensus", "union_physical_priority"}

    for _, row in box_rows.iterrows():
        values = np.asarray(
            [
                row["source_point_x_re"],
                row["source_point_z_re"],
                row["selection_box_x_half_width_re"],
                row["selection_box_z_half_width_re"],
            ],
            dtype=float,
        )
        if np.any(np.isnan(values)):
            continue

        point_kind = row["point_kind"]
        label = None
        if point_kind not in plotted_labels:
            label = f"manual {point_kind.upper()} search box"
            plotted_labels.add(point_kind)

        center_x_re, center_z_re, half_width_x_re, half_width_z_re = values
        color = colors.get(point_kind, "tab:red")
        selection_method = str(row.get("selection_method", ""))
        is_combined = selection_method in combined_methods
        rectangle = Rectangle(
            (center_x_re - half_width_x_re, center_z_re - half_width_z_re),
            2.0 * half_width_x_re,
            2.0 * half_width_z_re,
            facecolor="none" if is_combined else color,
            edgecolor=color,
            alpha=0.9 if is_combined else 0.18,
            linewidth=1.5,
            linestyle="--" if is_combined else "-",
            label=label,
            zorder=2.2,
        )
        ax.add_patch(rectangle)


def _metadata_method_mask(metadata_rows, column, method):
    """
    Return a boolean mask for metadata rows using a named selection method.

    Parameters
    ----------
    metadata_rows : pandas.DataFrame
        Metadata rows to filter.
    column : str
        Metadata column name.
    method : str
        Selection method name to match.

    Returns
    -------
    numpy.ndarray
        Boolean row mask.
    """

    if column not in metadata_rows.columns:
        return np.zeros(len(metadata_rows), dtype=bool)

    return (
        metadata_rows[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
        == str(method).lower()
    )


def _metadata_bool_mask(metadata_rows, column):
    """
    Return a boolean mask from a metadata boolean-like column.

    Parameters
    ----------
    metadata_rows : pandas.DataFrame
        Metadata rows to filter.
    column : str
        Metadata column name.

    Returns
    -------
    numpy.ndarray
        Boolean row mask.
    """

    if column not in metadata_rows.columns:
        return np.zeros(len(metadata_rows), dtype=bool)

    values = metadata_rows[column].fillna(False)
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)

    return (
        values.astype(str)
        .str.lower()
        .isin({"1", "true", "yes"})
        .to_numpy(dtype=bool)
    )


def draw_x_point_search_areas(ax, metadata_rows, x_selection_config=None):
    """
    Draw Hessian-aligned X-point search boxes used for VDF-cell selection.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where X-point search areas are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    x_selection_config : dict, optional
        X-point selection config containing ``half_width_di``.
    """

    required_columns = {
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "di_re",
        "hessian_e0_x",
        "hessian_e0_z",
        "hessian_e1_x",
        "hessian_e1_z",
    }
    if not x_selection_config or not required_columns.issubset(metadata_rows.columns):
        return

    half_width_di = x_selection_config.get("half_width_di", {})
    if not half_width_di:
        return

    half_width_0_di = float(half_width_di["eigenvector_0"])
    half_width_1_di = float(half_width_di["eigenvector_1"])

    x_point_rows = metadata_rows[metadata_rows["point_kind"] == "x"]
    if (
        "selection_method" in x_point_rows.columns
        or "plot_selection_method" in x_point_rows.columns
        or "physical_selected" in x_point_rows.columns
    ):
        x_point_rows = x_point_rows[
            _metadata_method_mask(
                metadata_rows=x_point_rows,
                column="selection_method",
                method="physical",
            )
            | _metadata_method_mask(
                metadata_rows=x_point_rows,
                column="plot_selection_method",
                method="physical",
            )
            | _metadata_bool_mask(
                metadata_rows=x_point_rows,
                column="physical_selected",
            )
        ]
    if x_point_rows.empty:
        return

    box_rows = x_point_rows.drop_duplicates(
        [
            "source_point_x_re",
            "source_point_z_re",
            "di_re",
            "hessian_e0_x",
            "hessian_e0_z",
            "hessian_e1_x",
            "hessian_e1_z",
        ]
    )

    for box_index, (_, row) in enumerate(box_rows.iterrows()):
        values = np.asarray(
            [
                row["source_point_x_re"],
                row["source_point_z_re"],
                row["di_re"],
                row["hessian_e0_x"],
                row["hessian_e0_z"],
                row["hessian_e1_x"],
                row["hessian_e1_z"],
            ],
            dtype=float,
        )
        if np.any(np.isnan(values)):
            continue

        center = values[:2]
        di_re = values[2]
        e0 = values[3:5]
        e1 = values[5:7]
        half_width_0_re = half_width_0_di * di_re
        half_width_1_re = half_width_1_di * di_re

        vertices = np.asarray(
            [
                center + half_width_0_re * e0 + half_width_1_re * e1,
                center - half_width_0_re * e0 + half_width_1_re * e1,
                center - half_width_0_re * e0 - half_width_1_re * e1,
                center + half_width_0_re * e0 - half_width_1_re * e1,
            ],
            dtype=float,
        )

        label = "X search area" if box_index == 0 else None
        polygon = Polygon(
            vertices,
            closed=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.18,
            linewidth=1.5,
            label=label,
            zorder=2.1,
        )
        ax.add_patch(polygon)


def scatter_label_points(ax, reader, metadata_rows):
    """
    Scatter source label points and sampled VDF cells on an xz colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where points are drawn.
    reader : analysator.vlsvfile.VlsvReader
        Reader for the timestep VLSV file.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    """

    marker_rows = metadata_rows.drop_duplicates(["cid"])

    for cell_index, (_, row) in enumerate(marker_rows.iterrows()):
        label = "Used VDF cell" if cell_index == 0 else None
        cell_coord_re = np.asarray(
            reader.get_cell_coordinates(int(row["cid"])),
            dtype=float,
        ) / R_EARTH

        ax.scatter(
            cell_coord_re[0],
            cell_coord_re[2],
            label=label,
            marker=".",
            s=18,
            color="red",
            linewidths=0,
            zorder=5,
        )

    plotted_source_classes = set()
    source_rows = metadata_rows.drop_duplicates(["class_name", "x_re", "z_re"])

    for _, row in source_rows.iterrows():
        class_name = row["class_name"]
        style = SOURCE_POINT_STYLES.get(
            class_name,
            {"color": "black", "marker": ".", "label": class_name},
        )
        label = style["label"] if class_name not in plotted_source_classes else None

        ax.scatter(
            row["x_re"],
            row["z_re"],
            label=label,
            s=style.get("s", 16),
            color=style.get("color"),
            edgecolors=style.get("edgecolor"),
            facecolors=style.get("facecolor"),
            linewidths=style.get("linewidths"),
            marker=style["marker"],
            zorder=6,
        )

        plotted_source_classes.add(class_name)


# =====================================================================
# From plot_dataset.py -- old-scheme dataset plotting (search-box/
# random-class-position colormaps, per-sample and batch VDF xz plots,
# plot xz-slice cache).
# =====================================================================

_MEMMAP_CACHE = {}


def add_dataset_sampling_plot_config(colormap_config, dataset_config_path):
    """
    Add dataset sampling settings needed for colormap overlays.

    The plotting config keeps visual options such as color scale and plot bounds,
    while the dataset-creation config owns the point class names and, for old
    datasets, may also contain the VDF box size used for X-point and O-point
    samples. This helper copies those sampling settings into a colormap config
    dictionary when they are available.

    Parameters
    ----------
    colormap_config : dict
        Colormap plotting options from the plotting config.
    dataset_config_path : str or pathlib.Path
        Path to the dataset-creation YAML config containing ``points`` and
        optionally ``vdf_box`` sections.

    Returns
    -------
    dict
        Copy of ``colormap_config`` with ``box_classes`` and optional
        ``vdf_box`` added.
    """

    dataset_config = load_config(dataset_config_path)
    points_config = dataset_config.get("points", {})
    box_classes = [
        points_config[class_key]
        for class_key in ("x_class_name", "o_class_name")
        if class_key in points_config
    ]

    colormap_config = dict(colormap_config)
    if "vdf_box" in dataset_config:
        colormap_config["vdf_box"] = dataset_config["vdf_box"]
    colormap_config["box_classes"] = box_classes
    colormap_config["file_template_flux"] = dataset_config.get("file_template_flux")
    colormap_config["x_selection"] = points_config.get("x_selection")
    colormap_config["o_core_fraction"] = (
        points_config.get("o_selection", {}).get("core_fraction")
    )

    return colormap_config


def create_colormap_plot_jobs(metadata, output_dir, colormap_config):
    """
    Create plot jobs for one labeled colormap per timestep.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata containing timestep, class, coordinate, and file columns.
    output_dir : pathlib.Path
        Base plot output directory for the dataset.
    colormap_config : dict
        Colormap plotting options. This may include ``boxre``, ``vmin``,
        ``vmax``, ``vdf_box``, and ``box_classes``.

    Returns
    -------
    list of dict
        Keyword argument dictionaries for ``plot_labeled_colormap``.
    """

    colormap_jobs = []

    for frame_index, (_, timestep_metadata) in enumerate(metadata.groupby("timestep")):
        colormap_output_path = output_dir / "colormaps" / f"colormap_{frame_index:04d}.png"

        colormap_jobs.append(
            {
                "metadata_rows": timestep_metadata,
                "output_path": colormap_output_path,
                "boxre": colormap_config.get("boxre", [-40, -1, -6, 6]),
                "vmin": float(colormap_config.get("vmin", -1.5e6)),
                "vmax": float(colormap_config.get("vmax", 1.5e6)),
                "vdf_box_config": colormap_config.get("vdf_box"),
                "box_classes": colormap_config.get("box_classes", []),
                "x_selection_config": colormap_config.get("x_selection"),
                "flux_file_template": colormap_config.get("file_template_flux"),
                "o_core_fraction": colormap_config.get("o_core_fraction"),
            }
        )

    return colormap_jobs


def iter_vdf_plot_jobs(X, y, metadata, output_dir, vdflim, X_plot=None):
    """
    Yield lightweight plot jobs for saved VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF sample array.
    y : numpy.ndarray
        Integer labels for VDF samples.
    metadata : pandas.DataFrame
        Dataset metadata with one row per sample.
    output_dir : pathlib.Path
        Base plot output directory for the dataset.
    vdflim : float
        Velocity axis limit in m/s for VDF plots.
    X_plot : numpy.ndarray, optional
        Memory-mapped cache of physical plot-oriented xz slices with shape
        ``(n_samples, vz, vx)``. If omitted, plots are made from full VDFs in
        ``X``.

    Yields
    ------
    dict
        Keyword argument dictionary for ``plot_vdf_sample_from_dataset``.
    """

    X_path = get_memmap_path(X, "X")
    X_plot_path = (
        None
        if X_plot is None
        else get_memmap_path(X_plot, "X_plot")
    )
    class_frame_counts = {}
    plot_axes_cache = {}
    plot_threshold_cache = {}
    reader_cache = {}
    vdf_shape = tuple(X.shape[1:])

    for sample_index in range(X.shape[0]):
        metadata_row = metadata.iloc[sample_index].to_dict()
        class_name = metadata_row["class_name"]
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])

        if class_name not in class_frame_counts:
            class_frame_counts[class_name] = 0

        class_frame_index = class_frame_counts[class_name]
        file_key = str(file_location)
        extent, dv, threshold = _get_cached_vdf_plot_parameters(
            reader_cache=reader_cache,
            plot_axes_cache=plot_axes_cache,
            plot_threshold_cache=plot_threshold_cache,
            file_location=file_key,
            cid=cid,
            vdf_shape=vdf_shape,
        )

        class_output_dir = output_dir / class_name
        output_path = class_output_dir / f"sample_{class_frame_index:04d}_xz.png"

        yield {
            "X_path": X_path,
            "X_plot_path": X_plot_path,
            "sample_index": int(sample_index),
            "y_label": int(y[sample_index]),
            "metadata_row": metadata_row,
            "extent": extent,
            "output_path": output_path,
            "dv": dv,
            "threshold": threshold,
            "vdflim": vdflim,
        }

        class_frame_counts[class_name] += 1


def run_plot_jobs(plot_function, plot_jobs, n_jobs):
    """
    Run plotting jobs serially or in parallel.

    Parameters
    ----------
    plot_function : callable
        Plotting function called with each job dictionary as keyword arguments.
    plot_jobs : iterable of dict
        Plot job dictionaries. Each dictionary is expanded into keyword
        arguments for ``plot_function``.
    n_jobs : int
        Number of parallel workers. Use 1 for serial plotting.
    """

    if n_jobs == 1:
        for plot_job in plot_jobs:
            plot_function(**plot_job)
        return

    Parallel(n_jobs=n_jobs)(
        delayed(plot_function)(**plot_job)
        for plot_job in plot_jobs
    )


def plot_dataset_search_box(config, metadata, output_path, colormap_config=None, figsize=(10, 9)):
    """
    Colormap of the region(s) configured for sampling (points.regions_re),
    with every VDF-carrying cell in that box marked -- what create_dataset
    could see, not just what it kept as labeled samples. Uses the first
    dataset row's file_location as the representative snapshot.
    """

    file_location = metadata.iloc[0]["file_location"]
    regions_re = config.get("points", {}).get("regions_re", {})

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    boxre = get_regions_re_boxre(regions_re, margin_re=2.0)
    if boxre is None:
        _vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
        boxre = [
            vdf_coords_re[:, 0].min() - 1, vdf_coords_re[:, 0].max() + 1,
            vdf_coords_re[:, 2].min() - 1, vdf_coords_re[:, 2].max() + 1,
        ]

    colormap_config = dict(colormap_config or {"var": "rho"})
    colormap_config.setdefault("boxre", boxre)

    fig, ax = plt.subplots(figsize=figsize)
    pt.plot.plot_colormap(filename=str(file_location), axes=ax, **colormap_config)
    scatter_all_vdf_cells(ax=ax, reader=reader, boxre=boxre)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_labeled_colormap(
        metadata_rows,
        output_path,
        boxre,
        vmin,
        vmax,
        vdf_box_config=None,
        box_classes=None,
        x_selection_config=None,
        flux_file_template=None,
        o_core_fraction=None,
):
    """
    Plot and save a spatial colormap with saved label points overlaid.

    Parameters
    ----------
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    output_path : str
        Output PNG path.
    boxre : list of float
        Plot box in Earth radii: ``[xmin, xmax, zmin, zmax]``.
    vmin : float
        Minimum color scale value for the selected velocity component.
    vmax : float
        Maximum color scale value for the selected velocity component.
    vdf_box_config : dict, optional
        VDF sampling box config from dataset creation. Expected keys are
        ``x_half_width_re`` and ``z_half_width_re``. The boxes are drawn around
        the source X/O point coordinates stored in metadata.
    box_classes : iterable of str, optional
        Class names whose source coordinates should get boxes, normally the
        configured X-point and O-point classes.
    x_selection_config : dict, optional
        X-point selection config used to redraw Hessian-aligned search boxes.
    flux_file_template : str, optional
        Flux file template used to redraw O-point island search contours.
    o_core_fraction : float, optional
        Fraction from O-point flux to boundary flux used for old metadata that
        does not contain ``search_flux``.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_location = metadata_rows.iloc[0]["file_location"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    pt.plot.plot_colormap(
        filename=file_location,
        axes=ax1,
        boxre=boxre,
        expression=expr_velocity,
        operator="x",
        vmin=vmin,
        vmax=vmax,
        streamlines="B",
        streamlinecolor="black",
    )

    reader = pt.vlsvfile.VlsvReader(str(file_location))
    scatter_all_vdf_cells(
        ax=ax1,
        reader=reader,
        boxre=boxre,
    )
    draw_point_boxes(
        ax=ax1,
        metadata_rows=metadata_rows,
        box_config=vdf_box_config,
        box_classes=box_classes,
    )
    draw_manual_point_search_boxes(
        ax=ax1,
        metadata_rows=metadata_rows,
    )
    draw_x_point_search_areas(
        ax=ax1,
        metadata_rows=metadata_rows,
        x_selection_config=x_selection_config,
    )
    draw_o_point_search_areas(
        ax=ax1,
        reader=reader,
        metadata_rows=metadata_rows,
        flux_file_template=flux_file_template,
        o_core_fraction=o_core_fraction,
    )
    scatter_label_points(
        ax=ax1,
        reader=reader,
        metadata_rows=metadata_rows,
    )
    ax1.legend(loc=2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_memmap_path(array, array_name):
    """
    Return the backing filename for a memory-mapped array.

    Parameters
    ----------
    array : numpy.ndarray
        Array expected to be backed by a ``.npy`` memmap.
    array_name : str
        Human-readable array name used in error messages.

    Returns
    -------
    str
        Memmap backing filename.
    """

    filename = getattr(array, "filename", None)
    if filename is None:
        raise ValueError(f"{array_name} must be loaded as a memory-mapped array")

    return str(filename)


def load_memmap_array(array_path):
    """
    Load and cache a read-only memory-mapped array.

    Parameters
    ----------
    array_path : str
        Path to a ``.npy`` array.

    Returns
    -------
    numpy.ndarray
        Read-only memory-mapped array.
    """

    array_path = str(array_path)
    array = _MEMMAP_CACHE.get(array_path)
    if array is None:
        array = np.load(array_path, mmap_mode="r")
        _MEMMAP_CACHE[array_path] = array

    return array


def create_or_load_plot_xz_slice_cache(X, dataset_dir, dataset_id, cache_config):
    """
    Create or load cached physical plot-oriented xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``X.npy``.
    dataset_id : str
        Dataset identifier used in cache path templates.
    cache_config : dict
        Plot cache settings.

    Returns
    -------
    numpy.ndarray or None
        Memory-mapped xz-slice cache, or ``None`` when caching is disabled.
    """

    cache_config = resolve_plot_xz_slice_cache_config(
        cache_config=cache_config,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
    )
    if not cache_config["enabled"]:
        return None

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]

    if (
            not cache_config["rebuild"]
            and is_plot_xz_slice_cache_valid(
                X=X,
                cache_path=cache_path,
                metadata_path=metadata_path,
            )
    ):
        print(f"Using plot xz-slice cache: {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    create_plot_xz_slice_cache(
        X=X,
        cache_config=cache_config,
    )

    return np.load(cache_path, mmap_mode="r")


def resolve_plot_xz_slice_cache_config(cache_config, dataset_dir, dataset_id):
    """
    Resolve plot xz-slice cache paths and settings.

    Parameters
    ----------
    cache_config : dict
        Raw cache settings from plotting config.
    dataset_dir : str or pathlib.Path
        Dataset directory containing ``X.npy``.
    dataset_id : str
        Dataset identifier used in cache path templates.

    Returns
    -------
    dict
        Resolved cache settings.
    """

    cache_config = cache_config or {}
    dataset_dir = Path(dataset_dir)
    enabled = bool(cache_config.get("enabled", False))

    cache_dir_template = cache_config.get("dir")
    if cache_dir_template is None:
        cache_dir = dataset_dir / "cache"
    else:
        cache_dir = create_path(
            path_template=cache_dir_template,
            timestep=dataset_id,
            dataset_id=dataset_id,
        )

    batch_size = int(cache_config.get("batch_size", 128))
    n_jobs = int(cache_config.get("n_jobs", 1))

    if batch_size <= 0:
        raise ValueError("plot.cache.batch_size must be positive")

    if n_jobs == 0:
        raise ValueError("plot.cache.n_jobs must be non-zero")

    return {
        "enabled": enabled,
        "cache_dir": Path(cache_dir),
        "cache_path": Path(cache_dir) / cache_config.get(
            "filename",
            "plot_xz_slice.npy",
        ),
        "metadata_path": Path(cache_dir) / cache_config.get(
            "metadata_filename",
            "plot_xz_slice_metadata.npz",
        ),
        "rebuild": bool(cache_config.get("rebuild", False)),
        "batch_size": batch_size,
        "n_jobs": n_jobs,
    }


def create_plot_xz_slice_cache(X, cache_config):
    """
    Create a memory-mapped cache of physical plot-oriented xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    cache_config : dict
        Resolved cache settings.

    Returns
    -------
    dict
        Cache metadata.
    """

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_shape = (int(X.shape[0]), *infer_plot_xz_slice_shape(X))
    X_plot = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=cache_shape,
    )

    start = time.perf_counter()
    sample_indices = np.arange(int(X.shape[0]), dtype=int)
    batches = list(
        iter_index_batches(
            sample_indices,
            batch_size=cache_config["batch_size"],
        )
    )

    print(f"Creating plot xz-slice cache: {cache_path}")
    print(f"Cache shape: {cache_shape}")
    print(f"Cache jobs: {cache_config['n_jobs']}")

    if cache_config["n_jobs"] == 1:
        for batch_indices in batches:
            write_plot_xz_slice_cache_batch(
                X=X,
                X_plot=X_plot,
                batch_indices=batch_indices,
            )
    else:
        Parallel(
            n_jobs=cache_config["n_jobs"],
            prefer="threads",
            require="sharedmem",
        )(
            delayed(write_plot_xz_slice_cache_batch)(
                X=X,
                X_plot=X_plot,
                batch_indices=batch_indices,
            )
            for batch_indices in batches
        )

    X_plot.flush()
    elapsed = time.perf_counter() - start
    metadata = {
        "enabled": True,
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "raw_vdf_shape": tuple(int(value) for value in X.shape),
        "cache_shape": tuple(int(value) for value in cache_shape),
        "slice": "xz",
        "orientation": "plot",
        "physical_units": "vdf",
        "elapsed_seconds": float(elapsed),
    }
    save_plot_xz_slice_cache_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(f"Created plot xz-slice cache in {elapsed:.2f} s")

    return metadata


def write_plot_xz_slice_cache_batch(X, X_plot, batch_indices):
    """
    Write one batch of physical plot-oriented xz slices into the cache.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    X_plot : numpy.ndarray
        Plot xz-slice cache.
    batch_indices : array-like of int
        Sample indices to write.
    """

    for sample_index in batch_indices:
        X_plot[int(sample_index)] = extract_plot_xz_slice_from_dataset(
            X=X,
            sample_index=int(sample_index),
        )


def extract_plot_xz_slice_from_dataset(X, sample_index):
    """
    Extract the plot-oriented middle xz slice from a saved sample.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    mid_y = X.shape[2] // 2
    return np.asarray(X[int(sample_index), :, mid_y, :].T, dtype=np.float32)


def infer_plot_xz_slice_shape(X):
    """
    Infer plot-oriented xz-slice shape from a VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.

    Returns
    -------
    tuple of int
        Plot-oriented xz-slice shape ``(vz, vx)``.
    """

    return int(X.shape[3]), int(X.shape[1])


def is_plot_xz_slice_cache_valid(X, cache_path, metadata_path):
    """
    Return whether an existing plot xz-slice cache matches the dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    cache_path : str or pathlib.Path
        Cache array path.
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    bool
        Whether the cache can be reused.
    """

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    if not cache_path.exists() or not metadata_path.exists():
        return False

    try:
        X_plot = np.load(cache_path, mmap_mode="r")
        metadata = load_plot_xz_slice_cache_metadata(metadata_path)
    except Exception:
        return False

    expected_shape = (int(X.shape[0]), *infer_plot_xz_slice_shape(X))

    return (
        tuple(X_plot.shape) == expected_shape
        and tuple(metadata.get("raw_vdf_shape", ())) == tuple(X.shape)
        and tuple(metadata.get("cache_shape", ())) == expected_shape
        and metadata.get("slice") == "xz"
        and metadata.get("orientation") == "plot"
        and metadata.get("physical_units") == "vdf"
    )


def save_plot_xz_slice_cache_metadata(metadata_path, metadata):
    """
    Save plot xz-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.
    metadata : dict
        Cache metadata.
    """

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        metadata_path,
        enabled=np.asarray(metadata["enabled"]),
        cache_path=np.asarray(metadata["cache_path"]),
        metadata_path=np.asarray(metadata["metadata_path"]),
        raw_vdf_shape=np.asarray(metadata["raw_vdf_shape"], dtype=int),
        cache_shape=np.asarray(metadata["cache_shape"], dtype=int),
        slice=np.asarray(metadata["slice"]),
        orientation=np.asarray(metadata["orientation"]),
        physical_units=np.asarray(metadata["physical_units"]),
        elapsed_seconds=np.asarray(metadata["elapsed_seconds"]),
    )


def load_plot_xz_slice_cache_metadata(metadata_path):
    """
    Load plot xz-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    dict
        Cache metadata.
    """

    with np.load(metadata_path, allow_pickle=False) as metadata:
        return {
            "enabled": bool(metadata["enabled"].item()),
            "cache_path": str(metadata["cache_path"].item()),
            "metadata_path": str(metadata["metadata_path"].item()),
            "raw_vdf_shape": tuple(
                int(value) for value in metadata["raw_vdf_shape"]
            ),
            "cache_shape": tuple(int(value) for value in metadata["cache_shape"]),
            "slice": str(metadata["slice"].item()),
            "orientation": str(metadata["orientation"].item()),
            "physical_units": str(metadata["physical_units"].item()),
            "elapsed_seconds": float(metadata["elapsed_seconds"].item()),
        }


def plot_vdf_sample_from_dataset(
        X_path,
        X_plot_path,
        sample_index,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
):
    """
    Plot and save one VDF sample loaded from a memory-mapped dataset.

    Parameters
    ----------
    X_path : str
        Path to the saved VDF ``.npy`` array.
    X_plot_path : str or None
        Path to cached plot-oriented xz slices. If ``None``, the full VDF is
        read from ``X_path``.
    sample_index : int
        Sample index to plot.
    y_label : int
        Integer label.
    metadata_row : dict
        Metadata row for the sample.
    extent : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    output_path : str
        Output PNG path.
    dv : float
        Velocity grid cell size in m/s.
    threshold : float
        VDF threshold before multiplying by ``dv``.
    vdflim : float, optional
        Visible velocity limit in m/s.
    """

    if X_plot_path is not None:
        X_plot = load_memmap_array(X_plot_path)
        plot_vdf_xz_slice_from_physical_xz(
            vdf_xz_slice=X_plot[int(sample_index)],
            y_label=y_label,
            metadata_row=metadata_row,
            extent=extent,
            output_path=output_path,
            dv=dv,
            threshold=threshold,
            vdflim=vdflim,
        )
        return

    X = load_memmap_array(X_path)
    plot_vdf_xz_slice(
        vdf=X[int(sample_index)],
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        output_path=output_path,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
    )


def plot_random_class_samples(
        X,
        y,
        metadata,
        output_path,
        vdflim=2e6,
        random_state=None,
        n_columns=2,
):
    """
    Plot one random xz VDF sample from each class in one figure.

    Parameters
    ----------
    X : numpy.ndarray
        Memory-mapped VDF sample array.
    y : numpy.ndarray
        Integer labels for VDF samples.
    metadata : pandas.DataFrame
        Dataset metadata with one row per sample and a ``class_name`` column.
    output_path : str or pathlib.Path
        Output PNG path.
    vdflim : float, optional
        Visible velocity limit in m/s.
    random_state : int, optional
        Random seed for reproducible class sampling.
    n_columns : int, optional
        Number of subplot columns.

    Returns
    -------
    dict
        Mapping from class name to selected sample index.
    """

    if n_columns <= 0:
        raise ValueError("n_columns must be positive")

    selected_indices = _select_random_class_sample_indices(
        metadata=metadata,
        random_state=random_state,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(selected_indices)
    n_rows = (n_samples + n_columns - 1) // n_columns

    plot_axes_cache = {}
    plot_threshold_cache = {}
    reader_cache = {}
    plot_items = []
    vdf_shape = tuple(X.shape[1:])

    for class_name, sample_index in selected_indices.items():
        metadata_row = metadata.iloc[sample_index].to_dict()
        file_location = metadata_row["file_location"]
        cid = int(metadata_row["cid"])
        file_key = str(file_location)
        extent, dv, threshold = _get_cached_vdf_plot_parameters(
            reader_cache=reader_cache,
            plot_axes_cache=plot_axes_cache,
            plot_threshold_cache=plot_threshold_cache,
            file_location=file_key,
            cid=cid,
            vdf_shape=vdf_shape,
        )
        vdf_plot = prepare_vdf_xz_plot(
            vdf=X[sample_index],
            metadata_row=metadata_row,
            dv=dv,
            threshold=threshold,
        )

        plot_items.append(
            {
                "class_name": class_name,
                "sample_index": sample_index,
                "y_label": y[sample_index],
                "metadata_row": metadata_row,
                "extent": extent,
                "vdf_plot": vdf_plot,
            }
        )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5 * n_columns, 5 * n_rows),
        squeeze=False,
    )

    for plot_index, plot_item in enumerate(plot_items):
        ax = axes.flat[plot_index]
        vdf_plot = plot_item["vdf_plot"]

        if vdf_plot is None:
            ax.set_axis_off()
            ax.set_title(str(plot_item["class_name"]), fontsize=11)
            continue

        im = plot_prepared_vdf_xz_slice_on_axis(
            ax=ax,
            vdf_plot=vdf_plot,
            y_label=plot_item["y_label"],
            metadata_row=plot_item["metadata_row"],
            extent=plot_item["extent"],
            vdflim=vdflim,
            title=str(plot_item["class_name"]),
        )
        _add_matching_colorbar(fig=fig, ax=ax, im=im)

    for ax in axes.flat[n_samples:]:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return selected_indices


def plot_dataset_random_class_positions(metadata, output_path, boxre, colormap_config=None, random_state=None, figsize=(10, 9)):
    """
    Colormap with the spatial position of one randomly chosen sample per
    class (see _select_random_class_sample_indices -- the same picker
    plot_random_class_samples uses, so the two plots can use the same seed
    for matching examples), one marker per class so it's easy to eyeball
    where a typical example of each label actually sits. Colors/markers are
    assigned by cycling over sorted(class names) at call time -- no fixed
    per-name palette -- so this adapts automatically if the class list
    changes between runs.
    """

    selected_indices = _select_random_class_sample_indices(
        metadata=metadata, random_state=random_state,
    )
    file_location = metadata.iloc[0]["file_location"]

    colormap_config = dict(colormap_config or {"var": "rho"})
    colormap_config.setdefault("boxre", boxre)

    fig, ax = plt.subplots(figsize=figsize)
    pt.plot.plot_colormap(filename=str(file_location), axes=ax, **colormap_config)

    class_names = sorted(selected_indices)
    colors = (
        plt.cm.tab10(np.linspace(0, 1, len(class_names))) if len(class_names) <= 10
        else plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    )
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

    for index, class_name in enumerate(class_names):
        row = metadata.iloc[selected_indices[class_name]]
        x_re, z_re = float(row["x_re"]), float(row["z_re"])
        ax.scatter(
            [x_re], [z_re], color=[colors[index]], marker=markers[index % len(markers)],
            s=140, edgecolor="black", linewidth=0.8, zorder=5,
            label=f"{class_name} (cid={int(row['cid'])})",
        )
        ax.annotate(
            str(class_name), (x_re, z_re), fontsize=8, color="black",
            xytext=(6, 6), textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, labelspacing=1.2, fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return selected_indices


def _get_cached_vdf_plot_parameters(
        reader_cache,
        plot_axes_cache,
        plot_threshold_cache,
        file_location,
        cid,
        vdf_shape,
):
    """
    Return VDF plot parameters using per-file and per-cell caches.

    Parameters
    ----------
    reader_cache : dict
        Mapping from VLSV file path to open reader.
    plot_axes_cache : dict
        Mapping from ``(file_location, vdf_shape)`` to ``(extent, dv)``.
    plot_threshold_cache : dict
        Mapping from ``(file_location, cid)`` to VDF sparsity threshold.
    file_location : str or pathlib.Path
        Path to the VLSV file.
    cid : int
        Spatial cell ID.
    vdf_shape : tuple of int
        Shape of one saved VDF sample.

    Returns
    -------
    extent : numpy.ndarray
        Velocity mesh extent.
    dv : float
        Velocity cell size.
    threshold : float
        VDF sparsity threshold.
    """

    file_location = str(file_location)
    cid = int(cid)
    axes_key = (file_location, tuple(vdf_shape))
    threshold_key = (file_location, cid)

    reader = _get_cached_vlsv_reader(
        reader_cache=reader_cache,
        file_location=file_location,
    )

    if axes_key not in plot_axes_cache:
        plot_axes_cache[axes_key] = get_vdf_plot_axes_parameters(
            reader=reader,
            vdf_shape=vdf_shape,
        )
    if threshold_key not in plot_threshold_cache:
        plot_threshold_cache[threshold_key] = get_vdf_plot_threshold(
            reader=reader,
            cid=cid,
        )

    extent, dv = plot_axes_cache[axes_key]
    threshold = plot_threshold_cache[threshold_key]

    return extent, dv, threshold


def _get_cached_vlsv_reader(reader_cache, file_location):
    """
    Return an open VLSV reader from a per-call cache.

    Parameters
    ----------
    reader_cache : dict
        Mapping from VLSV file path to open reader.
    file_location : str or pathlib.Path
        Path to the VLSV file.

    Returns
    -------
    analysator.vlsvfile.VlsvReader
        Open reader for ``file_location``.
    """

    file_location = str(file_location)
    reader = reader_cache.get(file_location)
    if reader is None:
        reader = pt.vlsvfile.VlsvReader(file_location)
        reader_cache[file_location] = reader

    return reader


def _select_random_class_sample_indices(metadata, random_state=None):
    """
    Select one random sample index from each class.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Dataset metadata with a ``class_name`` column. If available, the
        ``sample_index`` column is used as the index into ``X`` and ``y``.
    random_state : int, optional
        Random seed for reproducible class sampling.

    Returns
    -------
    dict
        Mapping from class name to selected sample index.
    """

    if "class_name" not in metadata.columns:
        raise ValueError("metadata must contain a class_name column")

    if metadata.empty:
        raise ValueError("metadata must contain at least one sample")

    rng = np.random.default_rng(random_state)
    selected_indices = {}

    for class_name, class_rows in metadata.groupby("class_name", sort=True):
        if "sample_index" in class_rows.columns:
            candidate_indices = class_rows["sample_index"].to_numpy(dtype=int)
        else:
            candidate_indices = class_rows.index.to_numpy(dtype=int)

        selected_indices[str(class_name)] = int(rng.choice(candidate_indices))

    return selected_indices


def _add_matching_colorbar(fig, ax, im):
    """
    Add a colorbar with the same height as its VDF subplot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the subplot.
    ax : matplotlib.axes.Axes
        Axis whose height the colorbar should match.
    im : matplotlib.image.AxesImage
        Image object used for colorbar scaling.
    """

    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="4%", pad=0.05)
    colorbar = fig.colorbar(im, cax=colorbar_ax, label="f(v)")
    colorbar.ax.tick_params(labelsize=8)
