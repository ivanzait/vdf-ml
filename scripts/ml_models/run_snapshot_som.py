#python scripts/ml_models/run_snapshot_som.py
#
# Edit src/data_proc/pipeline_config.py (SOM_CONFIG, RUN_ID) to change
# settings -- shared with run_snapshot_pca.py, which must be run first.
#
# One Self-Organizing Map per blind PCA/KMeans cluster: PCA's k=2 split
# separates calm from disturbed plasma well (see cluster_summary.csv), so
# each SOM maps the FINER structure within one of those tiers, fit on the
# saved pca_scores of just that cluster's samples (the same space the
# split was found in -- no features are rebuilt and no VLSV file is
# opened). Samples are painted onto each map by metadata.csv's expert
# label, and the trained codebook is optionally KMeans-partitioned --
# together the SOM analogue of scoring cluster_ml against cluster_phys.
import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

if importlib.util.find_spec("analysator") is None:
    # plot_tools.py imports vdf_tools.py, which imports analysator at
    # module load, even though this script never opens a VLSV reader.
    sys.path.append(os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    ))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.data_proc import pipeline_config as config
from src.data_proc.plot_tools import plot_som_label_maps
from src.ml_models.vdf_snapshot_clustering import fit_som_map, summarize_clusters_against_ground_truth

PCA_RESULTS_PATH = PROJECT_ROOT / "data" / "pca" / config.RUN_ID / "pca_results.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "snapshot_vdfs" / config.RUN_ID / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "som_snapshot" / config.RUN_ID


def main():
    som_config = config.SOM_CONFIG
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PCA results from {PCA_RESULTS_PATH}...")
    pca_results = np.load(PCA_RESULTS_PATH)
    pca_scores = pca_results["pca_scores"]
    cluster_labels = pca_results["cluster_labels"]
    sample_index = pca_results["sample_index"]

    metadata_by_sample = pd.read_csv(METADATA_PATH).set_index("sample_index").loc[sample_index]
    phys_labels = metadata_by_sample["label"].to_numpy()
    cellids = metadata_by_sample["cid"].to_numpy()
    print(f"Loaded {len(sample_index)} samples, best_k={int(pca_results['best_k'])}")

    som_maps = {}
    for cluster in sorted(set(cluster_labels.tolist())):
        mask = cluster_labels == cluster
        n_samples = int(mask.sum())
        if n_samples < som_config["min_cluster_size"]:
            print(f"cluster {cluster}: {n_samples} samples < min_cluster_size, skipping")
            continue

        # Per-tier PCA refit (see SOM_CONFIG["n_components"]): rerank this
        # cluster's samples by their own within-tier variance and truncate,
        # so the SOM's metric isn't spent on directions that only encoded
        # the between-tier split. Deliberately inline and visible -- three
        # lines, and the truncation choice should be auditable right where
        # the SOM is fit.
        features = pca_scores[mask]
        n_components = som_config.get("n_components")
        if n_components is not None:
            tier_pca = PCA(
                n_components=min(n_components, n_samples - 1, features.shape[1]),
                random_state=som_config["random_state"],
            )
            features = tier_pca.fit_transform(features)
            print(
                f"\ncluster {cluster} ({n_samples} samples): per-tier PCA refit to "
                f"{features.shape[1]} components ({tier_pca.explained_variance_ratio_.sum():.0%} "
                f"of within-tier variance), fitting {som_config['som_shape']} SOM..."
            )
        else:
            print(f"\ncluster {cluster} ({n_samples} samples): fitting {som_config['som_shape']} SOM on global pca_scores...")

        result = fit_som_map(
            features=features,
            som_shape=som_config["som_shape"],
            sigma=som_config["sigma"],
            learning_rate=som_config["learning_rate"],
            n_iterations=som_config["n_iterations"],
            random_state=som_config["random_state"],
            n_node_clusters=som_config["n_node_clusters"],
            node_cluster_method=som_config["node_cluster_method"],
            dbscan_eps=som_config["dbscan_eps"],
            dbscan_min_samples=som_config["dbscan_min_samples"],
        )
        print(f"  quantization error: {result['quantization_error']:.3f}")

        cluster_phys_labels = phys_labels[mask]
        cluster_cellids = cellids[mask]
        if result["node_cluster_grid"] is not None:
            # Score the blind "clusters within the cluster" against the
            # expert labels, with the same summary run_snapshot_pca.py uses
            # for the top-level KMeans clusters, plus one number (adjusted
            # Rand index: 1 = node-clusters reproduce the expert partition
            # exactly, 0 = no better than chance) so tuning SOM_CONFIG is
            # a measured comparison, not a visual impression.
            node_clusters_per_sample = result["node_cluster_grid"][
                result["bmu_coords"][:, 0], result["bmu_coords"][:, 1]
            ]
            ari = adjusted_rand_score(cluster_phys_labels, node_clusters_per_sample)
            print(f"  adjusted Rand index (node-clusters vs cluster_phys): {ari:.3f}")

            categories = sorted(set(cluster_phys_labels.tolist()))
            summary = summarize_clusters_against_ground_truth(
                cellids=cluster_cellids,
                labels=node_clusters_per_sample,
                ground_truth_cellids_by_category={
                    label: set(cluster_cellids[cluster_phys_labels == label].tolist())
                    for label in categories
                },
            )
            for row in summary:
                composition = " ".join(
                    f"{category}={row[f'n_{category}']} ({row[f'{category}_fraction']:.0%})"
                    for category in categories if row[f"n_{category}"]
                )
                # DBSCAN's -1 means "not part of any cluster" -- not a
                # cluster with id -1, so label it, don't let it read like
                # an ordinary (if oddly-numbered) node-cluster.
                cluster_name = "noise (-1)" if row["cluster"] == -1 else str(row["cluster"])
                print(f"  node-cluster {cluster_name}: size={row['size']} {composition}")

        som_maps[f"cluster {cluster} (n={n_samples})"] = {
            "bmu_coords": result["bmu_coords"],
            "u_matrix": result["u_matrix"],
            "node_cluster_grid": result["node_cluster_grid"],
            "phys_labels": cluster_phys_labels,
        }

    output_path = OUTPUT_DIR / "som_label_maps.png"
    plot_som_label_maps(som_maps, output_path=output_path)
    print(f"\nSaved SOM label maps to: {output_path}")


if __name__ == "__main__":
    main()
