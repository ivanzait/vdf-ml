"""
Snapshot-wide VDF loading and ground-truth labeling: point substances
(x_point/o_point/x_point_o_point from the physical topology detector,
src.data_proc.physics.point_topology; current_layer from the peak-current-
density detector, src.data_proc.physics.current_layer -- toggled per run via
points_config["active_point_substances"], see SCHEMA.md), and
magnetosheath/inner-magnetosphere from a subsolar-anchored Shue magnetopause
model (src.data_proc.physics.magnetopause).

Every ground-truth function here (load_snapshot_vdfs,
find_ground_truth_point_cellids, find_current_layer_cellids,
find_magnetosphere_region_cellids) goes through the shared
get_plasma_vdf_cells: same optional regions_re, applied the same way
(src.data_proc.vdf_tools.create_region_mask_re), so the spatial box the
caller searches/plots is the box every ground-truth category is computed
over, and cells with non-positive density (all-zero placeholder VDFs inside
the simulation's inner boundary) are excluded everywhere rather than
surfacing downstream as a mislabeled/degenerate sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_proc.labeling.point_labels import (
    get_o_point_cellids_by_method,
    get_vdf_cellids_in_b_perp_di_box,
)
from src.data_proc.physics.current_layer import find_current_layer_core_records
from src.data_proc.physics.magnetopause import (
    SHUE_ALPHA_DEFAULT,
    classify_magnetosphere_regions,
    fit_shue_model,
)
from src.data_proc.physics.point_topology import find_point_records
from src.data_proc.physics.vdf_transform import (
    DEFAULT_HERMITE_ORDER,
    compute_density,
    compute_drift_velocity,
    compute_thermal_velocity_components,
    get_rotated_vdf,
    vdf_to_hermite_spectra_log,
)
from src.data_proc.vdf_tools import (
    VdfExtractor,
    create_region_mask_re,
    get_vdf_cells_with_coords_re,
)


def get_plasma_vdf_cells(reader, regions_re=None, density_variable="rho", pop="avgs"):
    """
    VDF-carrying cells that actually hold plasma (density > 0), optionally
    restricted to a union of boxes.

    Some cells are nominally "VDF-carrying" (present in the file's
    CELLSWITHBLOCKS list -- what get_vdf_cells_with_coords_re alone
    returns) but sit inside the simulation's inner boundary, with an
    all-zero velocity distribution and zero density: a placeholder, not
    real data. Every ground-truth function in this module goes through
    this filter so those cells are excluded consistently everywhere,
    instead of surfacing downstream as a mislabeled/degenerate sample.

    regions_re : dict of {name: region_re}, optional
        If given, only cells inside at least one of these boxes are kept
        (see src.data_proc.vdf_tools.create_region_mask_re for the box
        format). If omitted, every plasma-carrying cell in the file is used.

    Returns cellids, coords_re.
    """

    cellids, coords_re = get_vdf_cells_with_coords_re(reader, pop=pop)

    if regions_re:
        mask = np.zeros(len(cellids), dtype=bool)
        for region_re in regions_re.values():
            mask |= create_region_mask_re(coords_re, region_re)
        cellids = cellids[mask]
        coords_re = coords_re[mask]

    if len(cellids) == 0:
        return cellids, coords_re

    densities = np.asarray(reader.read_variable(density_variable, cellids), dtype=float)
    positive_density = densities > 0
    return cellids[positive_density], coords_re[positive_density]


def load_snapshot_vdfs(reader, regions_re=None, density_variable="rho", pop="avgs"):
    """
    Extract every plasma-carrying VDF in a snapshot, optionally restricted
    to a union of boxes (see get_plasma_vdf_cells -- cells with non-positive
    density, e.g. inside the inner boundary, are excluded).

    regions_re : dict of {name: region_re}, optional
        If given, only cells inside at least one of these boxes are kept
        (see src.data_proc.vdf_tools.create_region_mask_re for the box
        format). If omitted, every plasma-carrying cell in the file is used.

    Returns cellids, coords_re, X (dense VDFs, shape (n, vx, vy, vz)).
    """

    cellids, coords_re = get_plasma_vdf_cells(
        reader, regions_re=regions_re, density_variable=density_variable, pop=pop,
    )

    extractor = VdfExtractor(reader=reader, pop=pop)
    X = np.asarray(
        [extractor.extract(cid=int(cid)) for cid in cellids],
        dtype=np.float32,
    )

    return cellids, coords_re, X


def find_ground_truth_point_cellids(reader, flux_file_location, points_config, regions_re=None):
    """
    Physically-detected X/O point cell IDs for one snapshot (no manual-box fallback).

    X points are matched with get_vdf_cellids_in_b_perp_di_box (a rectangular
    box aligned to the local B field at the point: short side along B,
    x_selection.half_width_di_normal * d_i; long side perpendicular to B,
    outflow_aspect_ratio times wider); O points are matched with
    get_o_point_cellids_by_method, which dispatches on
    o_selection.selection_method to either a gyroradius circle (radius
    o_selection.radius_gyroradii * rho_i) or the flux-contour island.

    Point detection itself is already gated by points_config["regions_re"]
    (find_point_records). regions_re here additionally restricts which VDF
    cells are considered as match candidates, matching the same restriction
    load_snapshot_vdfs/find_magnetosphere_region_cellids apply -- with a
    small search box/circle this practically never changes the match, but
    keeps every ground-truth function in this module scoped the same way.
    If omitted, every VDF-carrying cell in the file is a candidate.

    Returns x_point_cellids (set), o_point_cellids (set), x_point_records,
    o_point_records (raw topology-detection output, for reporting/plotting).
    """

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )
    vdf_cellids, vdf_coords_re = get_plasma_vdf_cells(reader, regions_re=regions_re)

    x_point_cellids = set()
    for point_record in x_point_records:
        cellids_by_position = get_vdf_cellids_in_b_perp_di_box(
            reader=reader,
            config=points_config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
        x_point_cellids.update(cellids_by_position.values())

    o_point_cellids = set()
    for point_record in o_point_records:
        cellids_by_position = get_o_point_cellids_by_method(
            config=points_config,
            point_record=point_record,
            vdf_cellids=vdf_cellids,
            vdf_coords_re=vdf_coords_re,
        )
        o_point_cellids.update(cellids_by_position.values())

    return x_point_cellids, o_point_cellids, x_point_records, o_point_records


def find_current_layer_cellids(reader, points_config, regions_re=None):
    """
    Physically-detected current-layer cell IDs for one snapshot: a VDF cell
    is labeled current_layer only if it IS a peak-|J| core cell
    (physics.current_layer.find_current_layer_core_records) -- a direct
    cellid intersection, not a nearest-cell search. Core records are found
    on the dense simulation grid, but their "cellid" is the same global
    cellid namespace the .vlsv file uses everywhere (VDF cellids included),
    so checking membership is exact, no coordinate matching needed.

    No margin/expansion step any more (an earlier version stepped out from
    each core cell by margin_di * d_i along its LMN normal, then
    nearest-matched every stepped point to the closest VDF-carrying cell via
    vdf_tools.get_nearest_vdf_cellid). That nearest-match had no maximum-
    distance cutoff, and on this fixture VDF cells sit on a coarse ~2.35 R_E
    subgrid while margin_di * d_i is only ~0.01-0.1 R_E -- three orders of
    magnitude smaller -- so the "expanded" search points always snapped to
    whichever VDF cell was nearest regardless of how large or small
    margin_di was, which could silently claim a VDF cell well outside the
    physical core (e.g. into inner_magnetosphere/magnetosheath). Direct
    cellid intersection has no such failure mode: a VDF cell only gets
    labeled current_layer if it is itself part of the detected core.

    regions_re : dict of {name: region_re}, optional
        Same union-of-boxes convention as everywhere else in this module.
        If omitted, the whole domain is searched.

    Returns current_layer_cellids (set), current_layer_records (the core
    grid-point records, for reporting/plotting).
    """

    core_records = find_current_layer_core_records(
        reader=reader,
        points_config=points_config,
        regions_re=regions_re,
    )
    if not core_records:
        return set(), core_records

    core_cellids = {record["cellid"] for record in core_records}

    vdf_cellids, _vdf_coords_re = get_plasma_vdf_cells(reader, regions_re=regions_re)
    current_layer_cellids = set(vdf_cellids.tolist()) & core_cellids

    return current_layer_cellids, core_records


def find_magnetosphere_region_cellids(reader, points_config, regions_re=None):
    """
    Region cell IDs for one snapshot, from subsolar-anchored Shue et al.
    (1998)-shaped magnetopause and bow shock surfaces (see
    src.data_proc.physics.magnetopause):
    Fixed priority order (see classify_magnetosphere_regions for the exact
    rules): no_density_data, solar_wind, magnetosheath, inner_magnetosphere,
    lobes, undefined (the catch-all, including the r_mp-to-lobe_r_min_re gap
    when lobe_r_min_re is set larger than r_mp to keep near-Earth nightside
    plasma out of "lobes"). No margin/buffer zone around the magnetopause
    any more -- current_layer's physically-detected peak-|J| core (a
    separate point substance, see SCHEMA.md) already identifies the real
    magnetopause current layer with much better precision than a crude
    geometric buffer here could.

    Config block: points_config["magnetopause"] (x_scan_min_re,
    x_scan_max_re, n_scan_points, density_variable, alpha, lobe_r_min_re).
    The subsolar scan itself (r_mp/r_bs) always runs along the full +x axis
    regardless of regions_re -- only which cells get classified/returned is
    restricted.

    regions_re : dict of {name: region_re}, optional
        If given, only cells inside at least one of these boxes are
        classified (see src.data_proc.vdf_tools.create_region_mask_re for
        the box format) -- the same restriction load_snapshot_vdfs applies,
        so the region ground truth only ever covers VDFs that were actually
        searched/plotted. If omitted, every VDF-carrying cell in the file
        is used.

    Returns cellids_by_region (dict of {"magnetosheath"/"solar_wind"/
    "inner_magnetosphere"/"lobes"/"no_density_data"/"undefined":
    set(cellid)}), shue_fit (r_mp_re/r_bs_re/alpha/subsolar), vdf_cellids,
    vdf_coords_re, region_labels (aligned with vdf_cellids).
    """

    magnetopause_config = (points_config or {}).get("magnetopause", {})
    density_variable = magnetopause_config.get("density_variable", "rho")

    shue_fit = fit_shue_model(
        reader=reader,
        x_scan_min_re=magnetopause_config.get("x_scan_min_re", 5.0),
        x_scan_max_re=magnetopause_config.get("x_scan_max_re", 30.0),
        n_scan_points=magnetopause_config.get("n_scan_points", 300),
        density_variable=density_variable,
        alpha=magnetopause_config.get("alpha", SHUE_ALPHA_DEFAULT),
    )

    vdf_cellids, vdf_coords_re = get_plasma_vdf_cells(
        reader, regions_re=regions_re, density_variable=density_variable,
    )

    densities = reader.read_variable(density_variable, vdf_cellids)
    region_labels = classify_magnetosphere_regions(
        vdf_coords_re=vdf_coords_re,
        densities=densities,
        r_mp_re=shue_fit["r_mp_re"],
        r_bs_re=shue_fit["r_bs_re"],
        alpha=shue_fit["alpha"],
        lobe_r_min_re=magnetopause_config.get("lobe_r_min_re"),
    )

    cellids_by_region = {
        region: set(vdf_cellids[region_labels == region].tolist())
        for region in (
            "magnetosheath", "solar_wind", "inner_magnetosphere",
            "lobes", "no_density_data", "undefined",
        )
    }

    return cellids_by_region, shue_fit, vdf_cellids, vdf_coords_re, region_labels


def combine_ground_truth_labels(vdf_cellids, region_labels, point_substance_cellids_by_label):
    """
    Merge magnetosphere region labels with whichever point substance(s) a
    caller computed into one label per cell in vdf_cellids. Point substances
    take priority over the broader region label (they're the rarer, more
    specific structures): for each (label_name, cellids) in
    point_substance_cellids_by_label, cells matching cellids get label_name,
    overwriting region_labels and any earlier substance in the same call.
    Falls back to region_labels ("magnetosheath"/"solar_wind"/
    "inner_magnetosphere"/"lobes"/"no_density_data"/"undefined") where
    nothing more specific matched.

    point_substance_cellids_by_label : dict of {label_name: set(cellid)}
        Built by the caller from whichever point-substance detector(s) it
        ran (e.g. find_ground_truth_point_cellids for
        x_point/o_point/x_point_o_point, find_current_layer_cellids for
        current_layer) -- this function has no knowledge of which
        substances exist or which are active; see SCHEMA.md.
    """

    labels = np.asarray(region_labels, dtype=object).copy()
    for label_name, cellids in point_substance_cellids_by_label.items():
        labels[np.isin(vdf_cellids, list(cellids))] = label_name
    return labels


def compute_snapshot_ground_truth(reader, points_config, regions_re=None, flux_file_location=None):
    """
    Full ground-truth pipeline for one snapshot: whichever point substances
    are active (points_config["active_point_substances"] -- "x_o_points"
    and/or "current_layer", see SCHEMA.md), Shue-model region
    classification, and the combined per-cell label array. Composes
    find_ground_truth_point_cellids and/or find_current_layer_cellids +
    find_magnetosphere_region_cellids + combine_ground_truth_labels, so
    verify_data.py can never drift out of sync with extract_data.py on how
    a label actually gets assigned. Keeps its own copy of the same toggle
    blocks extract_data.py has, rather than one calling the other (matching
    extract_data.py's existing preference for an inline copy over a shared
    call -- see its own header comment).

    flux_file_location is only needed/read when "x_o_points" is active.

    Returns cellids, coords_re, labels, shue_fit, point_substance_records
    (dict of {substance_name: raw per-substance detector output}, for
    reporting/plotting).
    """

    active_point_substances = (points_config or {}).get("active_point_substances", [])
    point_substance_cellids_by_label = {}
    point_substance_records = {}

    if "x_o_points" in active_point_substances:
        x_point_cellids, o_point_cellids, x_point_records, o_point_records = (
            find_ground_truth_point_cellids(
                reader=reader,
                flux_file_location=flux_file_location,
                points_config=points_config,
                regions_re=regions_re,
            )
        )
        overlap = x_point_cellids & o_point_cellids
        point_substance_cellids_by_label["x_point"] = x_point_cellids - overlap
        point_substance_cellids_by_label["o_point"] = o_point_cellids - overlap
        point_substance_cellids_by_label["x_point_o_point"] = overlap
        point_substance_records["x_o_points"] = (x_point_records, o_point_records)

    if "current_layer" in active_point_substances:
        current_layer_cellids, current_layer_records = find_current_layer_cellids(
            reader=reader,
            points_config=points_config,
            regions_re=regions_re,
        )
        point_substance_cellids_by_label["current_layer"] = current_layer_cellids
        point_substance_records["current_layer"] = current_layer_records

    _cellids_by_region, shue_fit, cellids, coords_re, region_labels = find_magnetosphere_region_cellids(
        reader=reader,
        points_config=points_config,
        regions_re=regions_re,
    )

    labels = combine_ground_truth_labels(
        vdf_cellids=cellids,
        region_labels=region_labels,
        point_substance_cellids_by_label=point_substance_cellids_by_label,
    )

    return cellids, coords_re, labels, shue_fit, point_substance_records


def pick_cluster_representative_cellids(vdf_cellids, labels, random_state=None):
    """
    One representative VDF cell per unique label present, chosen uniformly
    at random from that label's cells (random_state for reproducibility).
    Empty dict if labels is empty.
    """

    vdf_cellids = np.asarray(vdf_cellids)
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)

    return {
        label: int(rng.choice(vdf_cellids[labels == label]))
        for label in sorted(set(labels.tolist()))
    }


def rotate_vdfs_to_b_frame(X, b_field, bulk_velocity, velocity_mesh_extent):
    """
    Rotate every VDF in X into its own local (B, v_perp, B x v_perp) frame
    (physics.vdf_transform.get_rotated_vdf), at full resolution, onto a
    FIXED output grid -- the same shape and velocity_mesh_extent every
    sample already shares -- so every rotated sample stays the same shape
    as its unrotated original and the batch stays one stackable array.
    get_rotated_vdf's default behavior (no fixed grid given) auto-computes
    a bounding box from each call's own B orientation, which would make
    different samples come out different shapes -- not usable here.

    Meant to run once, right after extraction (see
    scripts/data_proc/extract_data.py), producing a second saved array
    (X_rotated.npy, alongside the untouched raw X.npy) rather than
    re-rotating on the fly on every downstream consumer run -- rotation is
    a real per-sample interpolation cost (~3s for a 268^3 VDF on the
    smoke-test fixture), so this is meant to be paid once at extraction
    time, not repeated by every PCA/ML experiment that reads the dataset.

    Falls back to the unrotated VDF for the one degenerate case
    build_rotation_matrix can hit (bulk_velocity exactly parallel to
    b_field, so there's no perpendicular direction to build v_perp_hat
    from) -- rare, but silently producing a wrong rotation would be worse
    than falling back; counted and reported instead of silently swallowed.

    Returns (X_rotated, n_fallback).
    """

    X = np.asarray(X)
    b_field = np.asarray(b_field, dtype=float)
    bulk_velocity = np.asarray(bulk_velocity, dtype=float)
    velocity_mesh_extent = np.asarray(velocity_mesh_extent, dtype=float)
    shape = X.shape[1:]

    X_rotated = np.empty_like(X)
    n_fallback = 0

    for index in range(len(X)):
        try:
            rotated_vdf, _new_shape, _new_v_limits, _rotation_matrix = get_rotated_vdf(
                vdf=X[index],
                shape=shape,
                v_limits=velocity_mesh_extent,
                b_field=b_field[index],
                bulk_velocity=bulk_velocity[index],
                new_v_limits=velocity_mesh_extent,
                new_shape=shape,
            )
            X_rotated[index] = rotated_vdf
        except ValueError:
            X_rotated[index] = X[index]
            n_fallback += 1

    return X_rotated, n_fallback


def compute_hermite_spectra_batch(X, velocity_mesh_extent, sparsity_threshold, order=DEFAULT_HERMITE_ORDER):
    """
    Convert every VDF in X to the Hermite spectra of its log10
    (physics.vdf_transform.vdf_to_hermite_spectra_log), a compact
    (order, order, order) basis.

X should be rotated into the local B frame already (X_rotated.npy, see
    rotate_vdfs_to_b_frame) -- combines both fixes: rotation removes local
    B-orientation as a confound (the same physical distribution otherwise
    looks different depending on where in the domain it sits), and the
    log10 step (see vdf_to_hermite_spectra_log) keeps a VDF's wings/tails
    -- often where non-Maxwellian structure lives -- from being swamped by
    its peak the way a raw linear-scale projection would swamp them. u/vth
    normalization (drift velocity/thermal speed, computed internally per
    sample) further keeps "where's the peak"/"how wide is it" from being
    left as pixel position/spread for PCA to contend with, the way a raw
    fixed-grid pixel slice would leave them. This function itself is
    representation-agnostic (X.npy works too), but extract_data.py always
    passes the rotated array -- see scripts/ml_models/
    rebuild_hermite_dataset.py for rebuilding X_hermite.npy at a new order
    from an already-saved X_rotated.npy, without re-paying rotation's cost.

    sparsity_threshold : array-like, shape (n,)
        Each sample's own sparsity floor ("MinValue" -- see
        vdf_tools.get_vdf_plot_threshold), passed through to
        vdf_to_hermite_spectra_log per sample.

    Meant to run once, at extraction time (see scripts/data_proc/
    extract_data.py), producing a saved array (X_hermite.npy) rather than
    repeating the transform on every downstream PCA/ML run -- cheap per
    sample (~0.2s for a 268^3 VDF at the default order on the smoke-test
    fixture) but still real, avoidable, repeated cost otherwise.

    Returns X_hermite, shape (len(X), order, order, order).
    """

    X = np.asarray(X)
    velocity_mesh_extent = np.asarray(velocity_mesh_extent, dtype=float)
    sparsity_threshold = np.asarray(sparsity_threshold, dtype=float)
    shape = X.shape[1:]

    X_hermite = np.empty((len(X), order, order, order), dtype=np.float32)
    for index in range(len(X)):
        X_hermite[index] = vdf_to_hermite_spectra_log(
            vdf=X[index], shape=shape, v_limits=velocity_mesh_extent,
            sparsity_threshold=sparsity_threshold[index], order=order,
        )

    return X_hermite


MOMENT_FEATURE_COLUMNS = [
    "hermite_density_m3",
    "hermite_ux_ms", "hermite_uy_ms", "hermite_uz_ms",
    "hermite_vthx_ms", "hermite_vthy_ms", "hermite_vthz_ms",
]


def compute_moment_features_batch(X, velocity_mesh_extent):
    """
    Lower-order physical moments per VDF in X: density (zeroth), bulk
    velocity (first, 3 components), and the per-axis/anisotropic thermal
    velocity (second, 3 components) -- physics.vdf_transform.compute_density/
    compute_drift_velocity/compute_thermal_velocity_components.

    Meant as a companion to compute_hermite_spectra_batch: the Hermite
    spectra capture shape (normalized away from density/position/width, see
    vdf_to_hermite_spectra_log), so these moments -- density and velocity
    especially -- are exactly the absolute-scale information that
    normalization deliberately throws away. Concatenating both back
    together for PCA (see scripts/ml_models/run_snapshot_pca.py's
    PCA_CONFIG["include_moment_features"]) lets shape and scale each
    contribute, rather than picking one.

    X should be the same rotated array Hermite spectra are built from
    (X_rotated.npy) -- then ux/uy/uz and vthx/vthy/vthz come out in the
    (B, v_perp, B x v_perp) frame (parallel to B, then two perpendicular
    directions), a real physical anisotropy, not an artifact of whichever
    way the simulation's coordinate axes happen to point.

    Returns an (len(X), 7) array, columns MOMENT_FEATURE_COLUMNS.
    """

    X = np.asarray(X)
    velocity_mesh_extent = np.asarray(velocity_mesh_extent, dtype=float)
    shape = X.shape[1:]

    moments = np.empty((len(X), 7), dtype=np.float64)
    for index in range(len(X)):
        vdf = X[index]
        density = compute_density(vdf, shape, velocity_mesh_extent)
        u = compute_drift_velocity(vdf, shape, velocity_mesh_extent)
        vth_components = compute_thermal_velocity_components(vdf, shape, velocity_mesh_extent, u)
        moments[index] = [density, u[0], u[1], u[2], vth_components[0], vth_components[1], vth_components[2]]

    return moments


def save_labeled_vdfs(
    output_dir, cellids, coords_re, X, labels, timestep, file_location,
    b_field, bulk_velocity, velocity_mesh_extent, sparsity_threshold,
    X_rotated=None, X_hermite=None,
):
    """
    Save extracted VDFs and their ground-truth labels (e.g. from
    combine_ground_truth_labels) to output_dir: X.npy (VDF arrays, shape
    (n, vx, vy, vz), whatever representation X already is -- raw phase-space
    density unless the caller transformed it) and metadata.csv
    (sample_index, cid, x_re/y_re/z_re, label, timestep, file_location,
    bx_t/by_t/bz_t, vx_ms/vy_ms/vz_ms, vspace_*min_ms/vspace_*max_ms,
    min_value).

    No Hermite transform or coordinate-axis rotation happens to X here --
    this only saves what it's given. Apply those beforehand if wanted;
    leaving them out entirely is the fast path when compute is constrained.

    b_field, bulk_velocity : array-like, shape (n, 3)
        Local B [T] / bulk flow V [m/s] at each cell. Saved regardless of
        whether X_rotated is given, so a downstream consumer that never
        reopens the .vlsv file can still rotate X on its own if wanted.
    velocity_mesh_extent : array-like, shape (6,)
        [vxmin, vymin, vzmin, vxmax, vymax, vzmax] [m/s] -- the same for
        every cell in one snapshot (one global velocity mesh per
        population), saved once per row (same constant-column convention as
        timestep/file_location above). This is the v_limits
        physics.vdf_transform.get_rotated_vdf needs.
    sparsity_threshold : array-like, shape (n,)
        Each cell's own sparsity floor ("MinValue" -- see
        vdf_tools.get_vdf_plot_threshold). Saved so a downstream consumer
        can reproduce/redo a log-space representation
        (compute_hermite_spectra_batch) without reopening the .vlsv file.
    X_rotated : numpy.ndarray, optional
        Same shape as X -- each sample rotated into its own local
        (B, v_perp, B x v_perp) frame (see
        labeling.snapshot_labeling.rotate_vdfs_to_b_frame), computed once
        here at extraction time rather than repeated by every downstream
        consumer. Saved as a second array, X_rotated.npy, alongside the
        untouched raw X.npy -- not merged into it, so consumers that want
        the real simulation-frame VDF (CNN training, verify_data.py's
        example plots) are unaffected. Omit to skip building/saving it.
    X_hermite : numpy.ndarray, optional
        Shape (n, order, order, order) -- each sample's Hermite spectra
        (see labeling.snapshot_labeling.compute_hermite_spectra_batch),
        computed once here rather than repeated by every downstream
        consumer. Saved as a third array, X_hermite.npy. Omit to skip
        building/saving it.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cellids = np.asarray(cellids)
    coords_re = np.asarray(coords_re)
    labels = np.asarray(labels)
    b_field = np.asarray(b_field, dtype=float)
    bulk_velocity = np.asarray(bulk_velocity, dtype=float)
    velocity_mesh_extent = np.asarray(velocity_mesh_extent, dtype=float)
    sparsity_threshold = np.asarray(sparsity_threshold, dtype=float)

    np.save(output_dir / "X.npy", X)
    if X_rotated is not None:
        np.save(output_dir / "X_rotated.npy", X_rotated)
    if X_hermite is not None:
        np.save(output_dir / "X_hermite.npy", X_hermite)

    metadata = pd.DataFrame(
        {
            "sample_index": np.arange(len(cellids)),
            "cid": cellids.astype(int),
            "x_re": coords_re[:, 0],
            "y_re": coords_re[:, 1],
            "z_re": coords_re[:, 2],
            "label": labels,
            "timestep": timestep,
            "file_location": str(file_location),
            "bx_t": b_field[:, 0],
            "by_t": b_field[:, 1],
            "bz_t": b_field[:, 2],
            "vx_ms": bulk_velocity[:, 0],
            "vy_ms": bulk_velocity[:, 1],
            "vz_ms": bulk_velocity[:, 2],
            "vspace_xmin_ms": velocity_mesh_extent[0],
            "vspace_ymin_ms": velocity_mesh_extent[1],
            "vspace_zmin_ms": velocity_mesh_extent[2],
            "vspace_xmax_ms": velocity_mesh_extent[3],
            "vspace_ymax_ms": velocity_mesh_extent[4],
            "vspace_zmax_ms": velocity_mesh_extent[5],
            "min_value": sparsity_threshold,
        }
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Saved X {X.shape} to {output_dir / 'X.npy'}")
    if X_rotated is not None:
        print(f"Saved X_rotated {X_rotated.shape} to {output_dir / 'X_rotated.npy'}")
    if X_hermite is not None:
        print(f"Saved X_hermite {X_hermite.shape} to {output_dir / 'X_hermite.npy'}")
    print(f"Saved metadata ({len(metadata)} rows) to {output_dir / 'metadata.csv'}")

    return metadata
