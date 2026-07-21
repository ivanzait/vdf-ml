#python scripts/ml_models/rebuild_hermite_dataset.py
#
# Edit src/data_proc/pipeline_config.py (HERMITE_ORDER, RUN_ID) to change
# settings -- shares config with extract_data.py/run_snapshot_pca.py.
#
# Rebuilds X_hermite.npy from an already-saved X_rotated.npy at the
# current HERMITE_ORDER, without re-extracting or re-rotating. Rotation is
# the expensive step (~3s/VDF on the smoke-test fixture's 268^3 grid);
# this script only pays Hermite's own much smaller per-sample cost, so
# retuning HERMITE_ORDER doesn't require repeating extract_data.py's whole
# pipeline (VDF extraction, labeling, region classification, rotation --
# none of which depend on HERMITE_ORDER at all). Needs extract_data.py to
# have been run once already with BUILD_ROTATED_DATASET = True for this
# RUN_ID; raises a clear FileNotFoundError otherwise.
#
# Also (re)computes the lower-order moment features (density, bulk
# velocity, anisotropic thermal velocity -- see
# labeling.snapshot_labeling.compute_moment_features_batch) from the same
# X_rotated.npy, and adds/overwrites their columns in metadata.csv --
# these don't depend on HERMITE_ORDER either, so it's cheap to keep them
# in sync here rather than a separate script.
import importlib.util
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

if importlib.util.find_spec("analysator") is None:
    # vdf_tools.py imports analysator at module load, even though this
    # script never opens a VLSV reader -- make sure it's importable.
    sys.path.append(os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    ))

import numpy as np

from src.data_proc import pipeline_config as config
from src.data_proc.dataset_io import load_labeled_vdfs
from src.data_proc.labeling.snapshot_labeling import (
    compute_hermite_spectra_batch,
    compute_moment_features_batch,
    MOMENT_FEATURE_COLUMNS,
)

DATASET_DIR = PROJECT_ROOT / "data" / "snapshot_vdfs" / config.RUN_ID


def main():
    print(f"Loading X_rotated.npy from {DATASET_DIR}...")
    X_rotated, metadata = load_labeled_vdfs(DATASET_DIR, mmap=True, x_filename="X_rotated.npy")
    print(f"Loaded {len(metadata)} rotated VDFs")

    velocity_mesh_extent = metadata[
        ["vspace_xmin_ms", "vspace_ymin_ms", "vspace_zmin_ms", "vspace_xmax_ms", "vspace_ymax_ms", "vspace_zmax_ms"]
    ].iloc[0].to_numpy()
    sparsity_threshold = metadata["min_value"].to_numpy()

    print(f"Computing log-space Hermite spectra (order={config.HERMITE_ORDER}) for every rotated VDF...")
    X_hermite = compute_hermite_spectra_batch(
        X=X_rotated,
        velocity_mesh_extent=velocity_mesh_extent,
        sparsity_threshold=sparsity_threshold,
        order=config.HERMITE_ORDER,
    )

    output_path = DATASET_DIR / "X_hermite.npy"
    np.save(output_path, X_hermite)
    print(f"Saved X_hermite {X_hermite.shape} to {output_path}")

    print("Computing lower-order moment features (density, velocity, anisotropic thermal velocity)...")
    moments = compute_moment_features_batch(X=X_rotated, velocity_mesh_extent=velocity_mesh_extent)
    metadata[MOMENT_FEATURE_COLUMNS] = moments
    metadata_path = DATASET_DIR / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    print(f"Saved moment feature columns {MOMENT_FEATURE_COLUMNS} to {metadata_path}")


if __name__ == "__main__":
    main()
