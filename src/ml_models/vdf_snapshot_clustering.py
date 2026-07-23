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
from minisom import MiniSom
from sklearn.cluster import DBSCAN, KMeans
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


def fit_som_map(
    features, som_shape=(6, 6), sigma=1.5, learning_rate=0.5, n_iterations=5000, random_state=1234,
    n_node_clusters=0, node_cluster_method="kmeans", dbscan_eps=1.0, dbscan_min_samples=2,
):
    """
    Train a Self-Organizing Map on features (typically the PCA scores of
    one blind cluster's samples -- see scripts/ml_models/run_snapshot_som.py)
    and locate every sample's best-matching unit (BMU).

    A SOM lays its nodes out on a fixed 2D grid but fits their codebook
    vectors in the FULL feature space, preserving topology: samples that
    are close in feature space land on nearby nodes. Painting the samples'
    ground-truth labels onto that grid then shows whether expert-based
    categories occupy distinct regions of the map -- a finer-grained,
    nonlinear complement to reading the same labels off a 2-component PCA
    scatter.

    n_node_clusters > 0 (KMeans) or node_cluster_method="dbscan" (n_node_clusters
    ignored -- DBSCAN discovers the count) additionally clusters the trained
    codebook vectors, giving each NODE a cluster id -- coarse structure of
    the map itself, independent of any ground-truth label. Both methods are
    weighted by each node's hit count (how many samples had it as their
    BMU): with only a few dozen samples on e.g. a 4x4=16-node map, several
    nodes are typically empty or singletons, and their codebook vectors are
    shaped only by neighborhood pull during training rather than by any
    real sample -- unweighted clustering gives them an equal vote against
    nodes several samples agreed on. Zero-weight (empty) nodes still get
    assigned a cluster label under KMeans (nearest final centroid); DBSCAN
    can legitimately call a node -1 ("noise", not part of any cluster) --
    at codebook sizes this small (~16-36 points), that's expected for
    ordinary boundary nodes, not necessarily a diagnostic problem, since
    DBSCAN has much less to estimate density from than it's designed for.
    KMeans assumes convex (roughly spherical) node-clusters in codebook
    space; DBSCAN doesn't, so it's worth trying if the map's structure
    looks like a curved/elongated ridge rather than blobs -- but compare
    the resulting ARI against KMeans's, don't assume better shape-fitting
    per se means better recovery of the expert labels at this sample size.

    Returns a dict: som, bmu_coords (n_samples, 2) integer node coordinates
    per sample, u_matrix (som_shape, mean codebook distance to neighbors),
    node_cluster_grid (som_shape int array with -1 for DBSCAN noise, or
    None), quantization_error.
    """

    features = np.asarray(features, dtype=float)
    n_rows, n_cols = som_shape

    som = MiniSom(
        n_rows, n_cols, features.shape[1],
        sigma=sigma, learning_rate=learning_rate, random_seed=random_state,
    )
    som.pca_weights_init(features)
    som.train(features, n_iterations, verbose=False)

    bmu_coords = np.array([som.winner(sample) for sample in features], dtype=int)
    u_matrix = som.distance_map()

    node_cluster_grid = None
    if node_cluster_method == "dbscan" or n_node_clusters > 0:
        codebook = som.get_weights().reshape(n_rows * n_cols, features.shape[1])
        hit_counts = np.zeros((n_rows, n_cols), dtype=float)
        for row, col in bmu_coords:
            hit_counts[row, col] += 1
        weights = hit_counts.reshape(n_rows * n_cols)

        if node_cluster_method == "dbscan":
            # DBSCAN's sample_weight makes a node count as `weight` copies
            # of itself for both the min_samples core-point test and its
            # neighbors' counts -- the density-based analogue of KMeans's
            # weighted centroids above. Zero-weight (empty) nodes can still
            # anchor a DBSCAN cluster geometrically (they're not excluded
            # the way a zero centroid-weight excludes a KMeans node from
            # shaping centroids) -- a real difference between the two
            # methods' handling of empty nodes, not just a style choice.
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            node_cluster_grid = dbscan.fit_predict(codebook, sample_weight=weights).reshape(n_rows, n_cols)
        else:
            kmeans = KMeans(n_clusters=n_node_clusters, random_state=random_state, n_init=10)
            node_cluster_grid = kmeans.fit_predict(codebook, sample_weight=weights).reshape(n_rows, n_cols)

    return {
        "som": som,
        "bmu_coords": bmu_coords,
        "u_matrix": u_matrix,
        "node_cluster_grid": node_cluster_grid,
        "quantization_error": float(som.quantization_error(features)),
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
