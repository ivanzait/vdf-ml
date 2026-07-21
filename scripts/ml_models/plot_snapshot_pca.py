#python scripts/ml_models/plot_snapshot_pca.py
#
# Edit src/data_proc/pipeline_config.py (PCA_CONFIG, RUN_ID) to change
# settings -- shared with run_snapshot_pca.py, which must be run first.
#
# Visualizes the PCA/KMeans results run_snapshot_pca.py saved to
# data/pca/<RUN_ID>/pca_results.npz: silhouette-vs-k, a PC1-vs-PC2 scatter
# with two colorbars (cluster_phys, the physical ground-truth region label,
# and cluster_ml, the blind KMeans cluster -- see README's "Terminology"
# section), where the smallest (rarest, most interesting) blind clusters
# sit spatially, and one example VDF per blind cluster. Only this script
# opens the VLSV file (for plot backgrounds and VDF re-extraction) --
# run_snapshot_pca.py never touches it. Doesn't re-draw the ground-truth
# region/combined-cluster maps -- those are already saved by
# extract_data.py against the exact same labels.
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import analysator as pt
except ImportError:
    analysator_path = os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    )
    sys.path.append(analysator_path)
    import analysator as pt

import numpy as np
import pandas as pd

from src.data_proc import pipeline_config as config
from src.data_proc.dataset_io import load_labeled_vdfs
from src.data_proc.labeling.snapshot_labeling import pick_cluster_representative_cellids
from src.data_proc.plot_tools import (
    plot_cluster_hermite_spectra,
    plot_cluster_vdf_examples,
    plot_cluster_vdf_positions,
    plot_colormap_with_vdf_markers,
    plot_pca_scatter,
    plot_silhouette_scores,
)
from src.ml_models.vdf_snapshot_clustering import summarize_clusters_against_ground_truth

DATASET_DIR = PROJECT_ROOT / "data" / "snapshot_vdfs" / config.RUN_ID
PCA_DIR = PROJECT_ROOT / "data" / "pca" / config.RUN_ID
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "pca_snapshot" / config.RUN_ID


def load_pca_results():
    metadata = pd.read_csv(DATASET_DIR / "metadata.csv")
    pca_results = np.load(PCA_DIR / "pca_results.npz")

    if not np.array_equal(pca_results["sample_index"], metadata["sample_index"].to_numpy()):
        raise ValueError(
            f"{PCA_DIR / 'pca_results.npz'} sample_index doesn't match "
            f"{DATASET_DIR / 'metadata.csv'} -- re-run run_snapshot_pca.py"
        )

    return metadata, pca_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = pt.vlsvfile.VlsvReader(str(config.FILE_LOCATION))

    print(f"Loading PCA results from {PCA_DIR}...")
    metadata, pca_results = load_pca_results()
    cellids = metadata["cid"].to_numpy()
    coords_re = metadata[["x_re", "y_re", "z_re"]].to_numpy()
    cluster_labels = pca_results["cluster_labels"]

    result = {
        "k_range": pca_results["k_range"],
        "silhouette_scores": pca_results["silhouette_scores"],
        "best_k": int(pca_results["best_k"]),
        "pca_scores": pca_results["pca_scores"],
        "labels": cluster_labels,
    }
    def save(name, fn, *args, **kwargs):
        output_path = OUTPUT_DIR / name
        fn(*args, output_path=output_path, **kwargs)
        print(f"Saved {name} to: {output_path}")

    save("silhouette_by_k.png", plot_silhouette_scores, result)
    save(
        "pca_scatter.png", plot_pca_scatter,
        result=result, cellids=cellids, phys_labels=metadata["label"].to_numpy(),
    )

    categories = sorted(metadata["label"].unique())
    ground_truth_cellids_by_category = {
        category: set(metadata.loc[metadata["label"] == category, "cid"])
        for category in categories
    }
    summary = summarize_clusters_against_ground_truth(
        cellids=cellids,
        labels=cluster_labels,
        ground_truth_cellids_by_category=ground_truth_cellids_by_category,
    )
    print("\nClusters, smallest first:")
    for row in summary:
        category_summary = " ".join(
            f"{category}={row[f'n_{category}']} ({row[f'{category}_fraction']:.0%})"
            for category in categories
        )
        print(f"  cluster {row['cluster']}: size={row['size']} {category_summary}")

    n_smallest = config.PCA_CONFIG["n_smallest_clusters_to_plot"]
    smallest_cluster_coords_re = coords_re[
        np.isin(
            cellids,
            np.concatenate([row["cellids"] for row in summary[:n_smallest]]),
        )
    ]
    save(
        "spatial_smallest_clusters.png", plot_colormap_with_vdf_markers,
        file_location=config.FILE_LOCATION,
        colormap_config={"var": "rho", "boxre": config.PLOT_BOXRE},
        all_coords_re=coords_re, selected_coords_re=smallest_cluster_coords_re,
    )

    representative_cellids = pick_cluster_representative_cellids(
        cellids, cluster_labels, random_state=config.PCA_CONFIG["random_state"],
    )
    representative_cellids = {
        f"cluster_{label:02d}": cid for label, cid in representative_cellids.items()
    }
    save(
        "cluster_vdf_positions.png", plot_cluster_vdf_positions,
        file_location=config.FILE_LOCATION, reader=reader,
        representative_cellids=representative_cellids, boxre=config.PLOT_BOXRE,
    )
    save(
        "cluster_vdf_examples.png", plot_cluster_vdf_examples,
        reader=reader, representative_cellids=representative_cellids,
        pop=config.POP, vdflim=config.VDFLIM,
    )

    if config.PCA_CONFIG.get("feature_representation") == "hermite":
        # Verification: the actual flattened feature vectors PCA saw,
        # straight from X_hermite.npy -- see plot_cluster_hermite_spectra.
        X_hermite, _hermite_metadata = load_labeled_vdfs(DATASET_DIR, mmap=True, x_filename="X_hermite.npy")

        print("\nHermite spectra per PCA cluster representative:")
        save(
            "cluster_hermite_spectra.png", plot_cluster_hermite_spectra,
            cellids=cellids, X_hermite=X_hermite, representative_cellids=representative_cellids,
        )

        # Broader check: one representative per PHYSICAL label (cluster_phys,
        # not the blind PCA cluster_ml) -- confirms rotation+Hermite behaves
        # sanely across every real category, not just whichever two PCA
        # clusters happened to fall out this run.
        phys_representative_cellids = pick_cluster_representative_cellids(
            cellids, metadata["label"].to_numpy(), random_state=config.PCA_CONFIG["random_state"],
        )
        print("\nHermite spectra per physical label representative:")
        save(
            "phys_cluster_hermite_spectra.png", plot_cluster_hermite_spectra,
            cellids=cellids, X_hermite=X_hermite, representative_cellids=phys_representative_cellids,
        )


if __name__ == "__main__":
    main()
