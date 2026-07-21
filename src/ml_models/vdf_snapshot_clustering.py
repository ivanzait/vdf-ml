"""
PCA/KMeans blind clustering of VDFs, and scoring clusters against
externally-supplied ground truth.

Loads every VDF in configured spatial boxes (no topology knowledge used),
reduces them with PCA, clusters the scores with KMeans (auto-selecting k via
silhouette score), and checks whether the smallest/most-separated clusters
line up with ground truth computed independently by
src.data_proc.labeling.snapshot_labeling (X/O points from the physical
topology detector, magnetosheath/inner-magnetosphere from a subsolar-anchored
Shue magnetopause model). X/O points are expected to be rare and visually
distinct, so they should show up as small, well-separated clusters if the
blind approach works.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca_clusters(features, k_range, n_components=20, random_state=1234, feature_weights=None):
    """
    PCA-reduce features, then KMeans-cluster the scores with k chosen by silhouette score.

    n_components is clamped to be valid for the sample count. k_range values
    that aren't feasible for the sample count (>= n_samples) are skipped.

    feature_weights : array-like, shape (n_features,), optional
        Per-feature multiplier applied to the STANDARDIZED data (after
        StandardScaler, before PCA) -- not before, since StandardScaler
        always resets every column back to unit variance regardless of its
        input scale, so any weighting applied earlier would just be undone.
        Lets a feature GROUP that's vastly outnumbered by another (e.g. 7
        physical-moment columns concatenated onto ~2700 Hermite-spectra
        columns, see scripts/ml_models/run_snapshot_pca.py) carry more
        influence over the leading PCs than its raw column count alone
        would give it -- otherwise its aggregate contribution to total
        variance is diluted simply by being outnumbered, independent of
        how informative it actually is. None (default) applies no extra
        weighting (every column stays unit variance, the original
        behavior).

    Returns a dict: scaler, pca, pca_scores, k_range, silhouette_scores
    (aligned with k_range, NaN where a k was skipped), best_k, kmeans, labels.
    """

    n_samples, n_features = features.shape
    n_components = max(1, min(n_components, n_samples - 1, n_features))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    if feature_weights is not None:
        feature_weights = np.asarray(feature_weights, dtype=float)
        scaled = scaled * feature_weights[None, :]

    pca = PCA(n_components=n_components, random_state=random_state)
    pca_scores = pca.fit_transform(scaled)

    k_range = [k for k in k_range if 2 <= k < n_samples]
    if not k_range:
        raise ValueError(
            f"No usable k in the requested range for {n_samples} samples"
        )

    silhouette_scores = []
    models_by_k = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pca_scores)
        silhouette_scores.append(silhouette_score(pca_scores, labels))
        models_by_k[k] = kmeans

    best_k = k_range[int(np.argmax(silhouette_scores))]
    best_model = models_by_k[best_k]

    return {
        "scaler": scaler,
        "pca": pca,
        "pca_scores": pca_scores,
        "k_range": k_range,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "kmeans": best_model,
        "labels": best_model.labels_,
    }


def summarize_clusters_against_ground_truth(
    cellids,
    labels,
    ground_truth_cellids_by_category,
):
    """
    Per-cluster size and ground-truth-category overlap, sorted smallest
    cluster first.

    ground_truth_cellids_by_category : dict of {category_name: set(cellid)}
        E.g. {"x_point": ..., "o_point": ..., "magnetosheath": ...,
        "inner_magnetosphere": ...}.

    Returns a list of dicts: cluster, size, cellids, and per category
    n_<category>/<category>_fraction.
    """

    cellids = np.asarray(cellids)
    labels = np.asarray(labels)

    rows = []
    for cluster in sorted(set(labels.tolist())):
        cluster_cellids = cellids[labels == cluster]
        size = len(cluster_cellids)
        row = {"cluster": int(cluster), "size": size, "cellids": cluster_cellids}
        for category, category_cellids in ground_truth_cellids_by_category.items():
            n_category = sum(1 for cid in cluster_cellids if int(cid) in category_cellids)
            row[f"n_{category}"] = n_category
            row[f"{category}_fraction"] = n_category / size if size else 0.0
        rows.append(row)

    rows.sort(key=lambda row: row["size"])
    return rows
