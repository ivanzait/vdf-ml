#python scripts/data_proc/extract_data.py
#
# Edit src/data_proc/pipeline_config.py to change the snapshot/search-box/
# detector settings -- extract_data.py and verify_data.py share that one
# config so they can never disagree about what "this run" extracted and
# labeled.
#
# The current main entry point for VDF data extraction + labeling: extracts
# every VDF inside SPATIAL_BOXRE for one snapshot, labels each cell (Shue-
# model magnetosphere regions, plus whichever "point-like" substance(s) are
# toggled on in POINTS_CONFIG["active_point_substances"] -- current_layer
# [peak-|J| core, exact cellid match] and/or x_o_points [Hessian
# critical-point detector], see SCHEMA.md), saves X.npy + metadata.csv, prints VDF value
# statistics (absorbed from the old inspect_dataset.py -- reuses the X
# already in memory, no reload), then plots what was extracted (every cell
# colored by its label, magnetopause/bow-shock drawn). No Hermite transform
# or coordinate-axis rotation -- raw VDFs only, kept fast on purpose.
#
# This replaced the older create_dataset.py pipeline (a different,
# now-outdated label scheme: lobe/exhaust/o_point/x_point/dayside). That
# pipeline's scripts were removed as unused; its code lives on, unreferenced,
# in src/deprecated.py -- see docs/DEPRECATED.md.
#
# See also verify_data.py: one representative VDF per label, mapped
# spatially plus its three velocity-space cuts.
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import analysator as pt
except ImportError:
    # Falls back to a local analysator checkout when it isn't installed as
    # a package (e.g. a plain git clone on a laptop instead of a cluster
    # module/PYTHONPATH setup). Override with the ANALYSATOR_PATH env var.
    analysator_path = os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    )
    sys.path.append(analysator_path)
    import analysator as pt

from src.data_proc import pipeline_config as config
from src.data_proc.dataset_io import print_vdf_statistics
from src.data_proc.physics.magnetopause import (
    SHUE_ALPHA_DEFAULT,
    classify_magnetosphere_regions,
    fit_shue_model,
)
from src.data_proc.labeling.snapshot_labeling import (
    combine_ground_truth_labels,
    compute_hermite_spectra_batch,
    find_current_layer_cellids,
    find_ground_truth_point_cellids,
    load_snapshot_vdfs,
    rotate_vdfs_to_b_frame,
    save_labeled_vdfs,
)
from src.data_proc.plot_tools import plot_combined_clusters
from src.data_proc.vdf_tools import get_vdf_plot_axes_parameters

DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "snapshot_vdfs" / config.RUN_ID
PLOT_OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "extract_data" / config.RUN_ID


def main():
    reader = pt.vlsvfile.VlsvReader(config.FILE_LOCATION)

    print("Loading VDFs...")
    cellids, coords_re, X = load_snapshot_vdfs(reader, regions_re=config.REGIONS_RE)
    print(f"Loaded {len(cellids)} VDFs")

    point_substance_cellids_by_label = {}
    current_layer_records = None

    if "x_o_points" in config.POINTS_CONFIG["active_point_substances"]:
        print("Running physical topology detector for X/O-point ground truth...")
        x_point_cellids, o_point_cellids, x_point_records, o_point_records = (
            find_ground_truth_point_cellids(
                reader=reader,
                flux_file_location=config.FLUX_FILE_LOCATION,
                points_config=config.POINTS_CONFIG,
                regions_re=config.REGIONS_RE,
            )
        )
        print(
            f"{len(x_point_records)} X points -> {len(x_point_cellids)} cells, "
            f"{len(o_point_records)} O points -> {len(o_point_cellids)} cells"
        )
        overlap = x_point_cellids & o_point_cellids
        point_substance_cellids_by_label["x_point"] = x_point_cellids - overlap
        point_substance_cellids_by_label["o_point"] = o_point_cellids - overlap
        point_substance_cellids_by_label["x_point_o_point"] = overlap

    if "current_layer" in config.POINTS_CONFIG["active_point_substances"]:
        print("Searching for the current layer (peak |J| core, exact cellid match)...")
        current_layer_cellids, current_layer_records = find_current_layer_cellids(
            reader=reader,
            points_config=config.POINTS_CONFIG,
            regions_re=config.REGIONS_RE,
        )
        print(
            f"{len(current_layer_records)} current-layer core points -> "
            f"{len(current_layer_cellids)} cells"
        )
        point_substance_cellids_by_label["current_layer"] = current_layer_cellids

    print("Fitting subsolar Shue magnetopause model for region ground truth...")
    density_variable = config.MAGNETOPAUSE_CONFIG["density_variable"]
    shue_fit = fit_shue_model(
        reader=reader,
        x_scan_min_re=config.MAGNETOPAUSE_CONFIG["x_scan_min_re"],
        x_scan_max_re=config.MAGNETOPAUSE_CONFIG["x_scan_max_re"],
        n_scan_points=config.MAGNETOPAUSE_CONFIG["n_scan_points"],
        density_variable=density_variable,
        alpha=config.MAGNETOPAUSE_CONFIG.get("alpha", SHUE_ALPHA_DEFAULT),
    )
    densities = reader.read_variable(density_variable, cellids)
    region_labels = classify_magnetosphere_regions(
        vdf_coords_re=coords_re,
        densities=densities,
        r_mp_re=shue_fit["r_mp_re"],
        r_bs_re=shue_fit["r_bs_re"],
        alpha=shue_fit["alpha"],
        lobe_r_min_re=config.MAGNETOPAUSE_CONFIG.get("lobe_r_min_re"),
    )

    labels = combine_ground_truth_labels(
        vdf_cellids=cellids,
        region_labels=region_labels,
        point_substance_cellids_by_label=point_substance_cellids_by_label,
    )
    print(f"Label counts: {dict(Counter(labels.tolist()))}")

    # B/bulk V/sparsity-threshold per cell + the snapshot's velocity-mesh
    # extent: saved to metadata.csv (not derived from X.npy) so a
    # downstream consumer that never reopens the reader -- e.g.
    # run_snapshot_pca.py -- can still rotate a raw VDF into its local
    # (B, v_perp, B x v_perp) frame, or redo a log-space representation,
    # later.
    b_field = reader.read_variable("B", cellids)
    bulk_velocity = reader.read_variable("V", cellids)
    sparsity_threshold = reader.read_variable("MinValue", cellids)
    velocity_mesh_extent, _dv = get_vdf_plot_axes_parameters(reader=reader, vdf_shape=X.shape[1:])

    # Hermite spectra are computed from the rotated VDF (see
    # compute_hermite_spectra_batch), so rotation runs whenever either
    # BUILD_ROTATED_DATASET or BUILD_HERMITE_DATASET wants its output --
    # X_rotated.npy itself is only saved if BUILD_ROTATED_DATASET is on.
    # Rotation doesn't depend on HERMITE_ORDER at all, so re-running this
    # whole script just to retune order re-pays its ~3s/VDF cost for
    # nothing -- once X_rotated.npy is saved (BUILD_ROTATED_DATASET=True
    # here once), scripts/ml_models/rebuild_hermite_dataset.py rebuilds
    # X_hermite.npy at a new order directly from that saved array, no
    # re-extraction/re-rotation needed.
    X_rotated = None
    if config.BUILD_ROTATED_DATASET or config.BUILD_HERMITE_DATASET:
        print("Rotating every VDF into its local (B, v_perp, B x v_perp) frame...")
        X_rotated, n_fallback = rotate_vdfs_to_b_frame(
            X=X, b_field=b_field, bulk_velocity=bulk_velocity, velocity_mesh_extent=velocity_mesh_extent,
        )
        if n_fallback:
            print(f"{n_fallback} VDF(s) kept unrotated (bulk V exactly parallel to B)")

    X_hermite = None
    if config.BUILD_HERMITE_DATASET:
        print(f"Computing log-space Hermite spectra (order={config.HERMITE_ORDER}) for every rotated VDF...")
        X_hermite = compute_hermite_spectra_batch(
            X=X_rotated, velocity_mesh_extent=velocity_mesh_extent,
            sparsity_threshold=sparsity_threshold, order=config.HERMITE_ORDER,
        )

    save_labeled_vdfs(
        output_dir=DATA_OUTPUT_DIR,
        cellids=cellids,
        coords_re=coords_re,
        X=X,
        labels=labels,
        timestep=config.TIMESTEP,
        file_location=config.FILE_LOCATION,
        b_field=b_field,
        bulk_velocity=bulk_velocity,
        velocity_mesh_extent=velocity_mesh_extent,
        sparsity_threshold=sparsity_threshold,
        X_rotated=X_rotated if config.BUILD_ROTATED_DATASET else None,
        X_hermite=X_hermite,
    )

    print_vdf_statistics(X)

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_output_path = PLOT_OUTPUT_DIR / "all_vdfs.png"
    plot_combined_clusters(
        file_location=config.FILE_LOCATION,
        vdf_coords_re=coords_re,
        combined_labels=labels,
        shue_fit=shue_fit,
        output_path=plot_output_path,
        boxre=config.PLOT_BOXRE,
        current_layer_records=current_layer_records,
        lobe_r_min_re=config.MAGNETOPAUSE_CONFIG.get("lobe_r_min_re"),
    )
    print(f"Saved plot to: {plot_output_path}")


if __name__ == "__main__":
    main()
