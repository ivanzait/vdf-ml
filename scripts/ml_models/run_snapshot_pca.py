#python scripts/ml_models/run_snapshot_pca.py
#
# Edit src/data_proc/pipeline_config.py (PCA_CONFIG, RUN_ID) to change
# settings -- shared with extract_data.py/verify_data.py so the PCA run
# always scores against the exact ground truth those scripts saved.
#
# Blind PCA + KMeans clustering (auto-k via silhouette score) on the
# dataset extract_data.py already saved to data/snapshot_vdfs/<RUN_ID>/:
# no VLSV reader is opened here, ground truth comes straight from
# metadata.csv's label column. Saves pca_results.npz + cluster_summary.csv
# to data/pca/<RUN_ID>/ for plot_snapshot_pca.py to visualize separately.
import importlib.util
import os
import sys
from pathlib import Path

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
import pandas as pd

from src.data_proc import pipeline_config as config
from src.data_proc.dataset_io import load_labeled_vdfs
from src.data_proc.labeling.snapshot_labeling import MOMENT_FEATURE_COLUMNS
from src.ml_models.features import create_features, normalize_feature_samples
from src.ml_models.vdf_snapshot_clustering import (
    fit_pca_clusters,
    summarize_clusters_against_ground_truth,
)

DATASET_DIR = PROJECT_ROOT / "data" / "snapshot_vdfs" / config.RUN_ID
OUTPUT_DIR = PROJECT_ROOT / "data" / "pca" / config.RUN_ID

X_FILENAME_BY_REPRESENTATION = {
    "raw": "X.npy",
    "rotated": "X_rotated.npy",
    "hermite": "X_hermite.npy",
}


def ground_truth_cellids_by_category(metadata):
    """{label: set(cid)} for every unique label extract_data.py saved to metadata.csv."""

    return {
        label: set(metadata.loc[metadata["label"] == label, "cid"].tolist())
        for label in sorted(metadata["label"].unique())
    }


def main():
    pca_config = config.PCA_CONFIG
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    representation = pca_config.get("feature_representation", "raw")
    x_filename = X_FILENAME_BY_REPRESENTATION[representation]
    print(f"Loading dataset from {DATASET_DIR} ({x_filename})...")
    X, metadata = load_labeled_vdfs(DATASET_DIR, mmap=True, x_filename=x_filename)
    cellids = metadata["cid"].to_numpy()
    print(f"Loaded {len(cellids)} VDFs")

    print("Building features and fitting PCA/KMeans...")
    n_moment_features = 0
    if representation == "hermite":
        # Already a compact (order, order, order) basis, not a pixel grid --
        # no slice/log/downsample step, just flatten each sample's spectra
        # into one feature row.
        features = np.asarray(X).reshape(len(X), -1)
        sample_normalization = pca_config.get("sample_normalization", "none")
        if sample_normalization != "none":
            # Physically complex/structured populations (current_layer,
            # magnetosheath) have spectra ~6-9x larger in magnitude than
            # quieter ones (lobes/solar_wind/inner_magnetosphere/undefined)
            # -- same shape, different overall scale (see TESTING.md).
            # StandardScaler alone (per-FEATURE column scaling) doesn't
            # remove that -- it leaves each sample's own scale intact, so
            # the two amplitude tiers still dominate variance ahead of any
            # shape difference. Normalizing each sample's row first (mean-
            # center + divide by its own std) removes that per-sample scale
            # before StandardScaler ever runs, letting shape compete with
            # shape instead of magnitude.
            print(f"Applying per-sample '{sample_normalization}' normalization before PCA...")
            features = normalize_feature_samples(
                features=features,
                sample_normalization=sample_normalization,
                sample_norm_eps=pca_config.get("sample_norm_eps", 1e-6),
            )
        if pca_config.get("include_moment_features", False):
            # Concatenated AFTER any per-sample normalization above, and
            # NOT normalized themselves -- these are the absolute-scale
            # physical quantities (density, velocity, anisotropic thermal
            # velocity) vdf_to_hermite_spectra_log's own u/vth
            # normalization deliberately strips out of the spectra; per-
            # sample-normalizing them too would throw away the exact
            # inter-sample scale differences they're meant to add back.
            # StandardScaler downstream (per-FEATURE column scaling) still
            # puts them on a comparable footing with the Hermite columns
            # for PCA, without touching their relative values across
            # samples. See labeling.snapshot_labeling.
            # compute_moment_features_batch/MOMENT_FEATURE_COLUMNS.
            missing = [column for column in MOMENT_FEATURE_COLUMNS if column not in metadata.columns]
            if missing:
                raise ValueError(
                    f"metadata.csv is missing moment feature columns {missing} -- "
                    f"re-run scripts/ml_models/rebuild_hermite_dataset.py for this RUN_ID."
                )
            moments = metadata[MOMENT_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            n_moment_features = moments.shape[1]
            print(f"Concatenating {n_moment_features} moment features ({MOMENT_FEATURE_COLUMNS})...")
            features = np.concatenate([features, moments], axis=1)
    else:
        features = create_features(
            X,
            downsample_factor=pca_config["downsample_factor"],
            log_eps=pca_config["log_eps"],
            n_jobs=pca_config["n_jobs"],
        )

    feature_weights = None
    if n_moment_features > 0:
        # The 7 moment columns are vastly outnumbered by the Hermite
        # spectra columns (e.g. 7 of 2751 at order=14) -- after
        # StandardScaler, every column has equal (unit) variance, but their
        # AGGREGATE contribution to total variance is still diluted purely
        # by being outnumbered ~390:1, independent of how informative they
        # actually are. moment_feature_weight boosts them post-scaling (see
        # fit_pca_clusters) so they can compete for the leading PCs instead
        # of being swamped by column count alone. 1.0 = no boost (the
        # column-count-diluted behavior from before this weighting existed).
        moment_feature_weight = pca_config.get("moment_feature_weight", 1.0)
        n_hermite_features = features.shape[1] - n_moment_features
        feature_weights = np.concatenate([
            np.ones(n_hermite_features),
            np.full(n_moment_features, moment_feature_weight),
        ])
        print(f"Weighting moment features by {moment_feature_weight}x (post-StandardScaler) before PCA...")

    result = fit_pca_clusters(
        features,
        k_range=pca_config["k_range"],
        n_components=pca_config["n_components"],
        random_state=pca_config["random_state"],
        feature_weights=feature_weights,
    )
    print(f"Silhouette scores by k: {dict(zip(result['k_range'], result['silhouette_scores']))}")
    print(f"Best k: {result['best_k']}")

    categories = sorted(metadata["label"].unique())
    summary = summarize_clusters_against_ground_truth(
        cellids=cellids,
        labels=result["labels"],
        ground_truth_cellids_by_category=ground_truth_cellids_by_category(metadata),
    )
    print("\nClusters, smallest first:")
    for row in summary:
        category_summary = " ".join(
            f"{category}={row[f'n_{category}']} ({row[f'{category}_fraction']:.0%})"
            for category in categories
        )
        print(f"  cluster {row['cluster']}: size={row['size']} {category_summary}")

    np.savez(
        OUTPUT_DIR / "pca_results.npz",
        sample_index=metadata["sample_index"].to_numpy(),
        pca_scores=result["pca_scores"].astype(np.float32),
        cluster_labels=result["labels"],
        k_range=np.asarray(result["k_range"]),
        silhouette_scores=np.asarray(result["silhouette_scores"]),
        best_k=result["best_k"],
    )
    print(f"Saved PCA scores to: {OUTPUT_DIR / 'pca_results.npz'}")

    summary_df = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "cellids"} for row in summary]
    )
    summary_df.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    print(f"Saved cluster summary to: {OUTPUT_DIR / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
