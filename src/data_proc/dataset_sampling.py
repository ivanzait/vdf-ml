import resource
import sys
import time

import numpy as np
import analysator as pt

from src.data_proc.dataset_metadata import (
    OMITTED_DATASET_METADATA_COLUMNS,
    POINT_REFERENCE_METADATA_COLUMNS,
    create_point_reference_metadata_arrays,
    create_vdf_spatial_metadata,
)
from src.data_proc.config import create_timestep_path
from src.data_proc.vdf_extract import VdfExtractor
from src.data_proc.point_selection import (
    create_point_label_data,
    create_point_sample_metadata,
    get_point_selection_result,
    is_point_record,
    iter_labeled_coords,
    unpack_labeled_coord,
)
from src.data_proc.vdf_helpers import (
    DEFAULT_HERMITE_ORDER,
    R_EARTH,
    create_region_mask_re,
    get_b_field,
    get_bulk_velocity,
    get_nearest_vdf_cellid,
    get_rotated_vdf,
    get_vdf_cells_with_coords_re,
    get_vdf_plot_axes_parameters,
    iter_enabled_regions_re,
    vdf_to_hermite_spectra,
)


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
    """Build one metadata.csv row: identity fields + VDF-center/X-O-point spatial fields."""

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
        vdf_coord_re = sample_spec["vdf_coord_re"]
        spatial_metadata = {
            "vdf_x_re": float(vdf_coord_re[0]),
            "vdf_y_re": float(vdf_coord_re[1]),
            "vdf_z_re": float(vdf_coord_re[2]),
            **{
                column: sample_spec[column]
                for column in POINT_REFERENCE_METADATA_COLUMNS
            },
        }
    else:
        spatial_metadata = create_vdf_spatial_metadata(
            vdf_coord_re=sample_spec["vdf_coord_re"],
            point_kind=point_kind,
            source_point_coord_re=source_point_coord_re,
        )
    internal_keys = {
        "file_location",
        "simulation_time",
        "cid",
        "label",
        "class_name",
        "coord_re",
        "vdf_coord_re",
        "neighbor_position",
        "timestep",
        "hermite_order",
        "hermite_enabled",
        "hermite_rotate",
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
