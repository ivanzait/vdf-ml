# PCA Guide

This repo has **two independent PCA pipelines** that don't share code or
purpose — don't conflate them:

1. **Snapshot clustering PCA** (current, actively developed) — blind
   PCA+KMeans clustering of every VDF in one Vlasiator snapshot, scored
   against the physics-derived ground truth `extract_data.py` computes.
   Lives in `scripts/ml_models/{run,plot}_snapshot_pca.py` +
   `src/ml_models/vdf_snapshot_clustering.py`.
2. **Legacy CNN-dataset PCA** — a much older, much larger
   (`src/ml_models/dataset_pca.py`, ~4900 lines) diagnostic/preview tool
   built around the old static-label dataset format. Its CNN
   training-filter integration has been **removed**; only its standalone
   visualization path (`scripts/data_proc/plot_dataset_pca.py`) still runs.

See [`schema.md`](schema.md) and README's "Terminology" section for the
`cluster_phys`/`cluster_ml` distinction both pipelines use.

## 1. Snapshot clustering PCA (current)

### Role


Sanity-checks the physics-driven labels (`cluster_phys`, from
`extract_data.py`) by asking: does an unsupervised, label-blind PCA+KMeans
clustering (`cluster_ml`) on the same VDFs recover similar structure? If
the rarest/most physically distinctive substances (e.g. `current_layer`,
`magnetosheath`) come out as their own clean blind clusters, that's
independent evidence the physical labels are capturing something real.

Like the legacy pipeline, PCA is refit from scratch every run — nothing is
pickled/saved, only the output (`pca_results.npz`, plots) is persisted.

### Pipeline

```
python scripts/data_proc/extract_data.py             # saves X.npy (+ optionally X_rotated.npy/X_hermite.npy)
python scripts/ml_models/rebuild_hermite_dataset.py   # only if using "hermite" -- see below
python scripts/ml_models/run_snapshot_pca.py          # fit PCA+KMeans, save pca_results.npz + cluster_summary.csv
python scripts/ml_models/plot_snapshot_pca.py          # silhouette/PC1-PC2/spatial/example-VDF/Hermite-verification plots
```

All four share `src/data_proc/pipeline_config.py` (`RUN_ID`, `PCA_CONFIG`,
`HERMITE_ORDER`, `BUILD_ROTATED_DATASET`, `BUILD_HERMITE_DATASET`) — one
config so every stage agrees on what "this run" means.

### Feature representations — `PCA_CONFIG["feature_representation"]`

| Value | Array clustered | Built by |
|---|---|---|
| `"raw"` (default) | `X.npy` — log10-scaled, downsampled `vy=mid` xz-slice of the raw VDF (`ml_models.features.create_feature`) | always (`extract_data.py`) |
| `"rotated"` | `X_rotated.npy` — same xz-slice/downsample, but every VDF first rotated into its own local `(B, v_perp, B×v_perp)` frame at full resolution (`labeling.snapshot_labeling.rotate_vdfs_to_b_frame`, `physics.vdf_transform.get_rotated_vdf`) | `extract_data.py`, needs `BUILD_ROTATED_DATASET = True` |
| `"hermite"` | `X_hermite.npy` — the rotated VDF's Hermite spectra, `log10(f/sparsity_threshold)` projected onto a truncated `(order, order, order)` basis (`physics.vdf_transform.vdf_to_hermite_spectra_log`), flattened directly (no slice/downsample — it's already compact) | `extract_data.py` (needs `BUILD_HERMITE_DATASET = True`, which also runs rotation as a prerequisite even if `BUILD_ROTATED_DATASET` is off) — or cheaply **rebuilt at a new `HERMITE_ORDER`** via `rebuild_hermite_dataset.py`, which reuses an already-saved `X_rotated.npy` instead of re-rotating (rotation costs ~3s/VDF; Hermite itself is much cheaper) |

`"raw"` only intersects a single `vy=mid` plane of the VDF before
downsampling — a real, pre-existing limitation discovered mid-session (see
TESTING.md): most of the 3D VDF's information is discarded for that
representation. `"hermite"` doesn't have this problem — it projects the
full 3D grid.

`run_snapshot_pca.py` raises a clear `FileNotFoundError` if the array a
chosen mode needs wasn't built for this `RUN_ID`.

### `"hermite"`-only extras

Only meaningful (and only wired up) when `feature_representation ==
"hermite"`:

- **`PCA_CONFIG["sample_normalization"]`** (`"none"` / `"standard"`) — applied to the flattened Hermite features BEFORE `StandardScaler`+PCA (`ml_models.features.normalize_feature_samples`, mean-center + divide by each sample's own std). Fixes a real failure mode: physically complex/structured populations (`current_layer`, `magnetosheath`) have Hermite spectra ~6-9x larger in magnitude than quieter ones (`lobes`/`solar_wind`/`inner_magnetosphere`/`undefined`) at the *same shape* — `StandardScaler` alone (per-feature/column scaling) doesn't remove that per-sample scale, so the two amplitude tiers dominate variance ahead of any shape difference, collapsing KMeans to a degenerate "one outlier vs. everyone" split regardless of `k`. Normalizing per-sample first fixed this (see TESTING.md for the full debugging story).
- **`PCA_CONFIG["include_moment_features"]`** (bool) — concatenates 7 lower-order physical moments (density, 3 bulk-velocity components, 3 anisotropic thermal-velocity components — `labeling.snapshot_labeling.compute_moment_features_batch`/`MOMENT_FEATURE_COLUMNS`) onto the flattened Hermite features, computed from the same rotated frame. The Hermite spectra deliberately normalize position/width/density away to isolate shape (`vdf_to_hermite_spectra_log`'s u/vth normalization); these moments are exactly that discarded absolute-scale information, added back explicitly. **Not** per-sample normalized themselves (that would erase the real inter-sample scale differences they exist to add back) — `StandardScaler` still puts them on a comparable per-column footing with the Hermite columns before PCA. Requires `rebuild_hermite_dataset.py` to have been run (adds the moment columns to `metadata.csv`); raises a clear `ValueError` otherwise.
- **`PCA_CONFIG["moment_feature_weight"]`** — post-`StandardScaler` multiplier on just the 7 moment columns (`ml_models.vdf_snapshot_clustering.fit_pca_clusters`'s `feature_weights` param). With only 7 moment columns against ~2700+ Hermite columns (at `HERMITE_ORDER=14`), their aggregate contribution to total variance is diluted by column count alone even though each one individually has unit variance after scaling — weighting compensates. `1.0` = no boost (weighting applied before `StandardScaler` would be pointless: it always resets every column back to unit variance regardless of input scale). `~sqrt(n_hermite_features / 7)` (≈20 at order=14) gives the moments as a group roughly equal total variance to the entire Hermite block. Verified end-to-end: `1.0` exactly reproduces the unweighted result (true no-op); `10.0` measurably improved cluster purity (`magnetosheath`/`current_layer` reached 100% purity, silhouette at k=2 rose from 0.252 to 0.314).

### Verification tooling

`plot_snapshot_pca.py` always produces: `silhouette_by_k.png`,
`pca_scatter.png` (PC1-vs-PC2, two colorbars: `cluster_phys` and
`cluster_ml`), `spatial_smallest_clusters.png`, and
`cluster_vdf_examples.png` (one representative VDF per blind cluster --
a combined figure with a spatial-position header row plus each
representative's velocity-space cuts below, see
`plot_tools.plot_cluster_vdf_examples`).

When `feature_representation == "hermite"`, it additionally saves (see
`plot_tools.plot_cluster_hermite_spectra`):
- `cluster_hermite_spectra.png` — one representative per **PCA** cluster.
- `phys_cluster_hermite_spectra.png` — one representative per **physical**
  label (`cluster_phys`), a broader check that rotation+Hermite behaves
  sanely across every real category, not just whichever clusters a given
  run happened to produce.

Both show the actual `(order, order, order)` feature array PCA sees (not a
reconstruction), reduced to 2D per representative by integrating in
quadrature over the second perpendicular axis (`sqrt(sum(spectra**2,
axis=2))`) — `spectra[n, m, l]` indexes `(parallel=B, perp1, perp2)`, so
this collapses to a `(parallel order n, perp order m)` image, the direct
Hermite-space analogue of reducing a 3D VDF to `(v_parallel, v_perp)` by
integrating over gyrophase. Each panel is normalized to its own max (a
shared scale would hide the quiet categories entirely under
`current_layer`/`magnetosheath`'s larger coefficients).

**How this caught real bugs** (see TESTING.md for full details): plotting
the actual flattened spectra (not a summary statistic) revealed that an
early "floor sub-threshold cells at `log10(threshold)`" implementation
integrated a large constant background over the *entire* velocity-space
volume (99%+ empty on the fixture), producing 1e9-1e10-magnitude spectra
dominated by each sample's own basis-function normalization rather than
its physical shape — fixed by using `log10(vdf/threshold)` instead (zero
exactly where the VDF has no signal, restoring compact support). The same
plots later showed that raising `HERMITE_ORDER` alone didn't fix a
persistent single-outlier clustering split — the per-physical-label
version showed *why*: `current_layer` and `magnetosheath` genuinely have
~6-9x larger-magnitude spectra than the other four categories, a real
physical difference (structured/non-Maxwellian populations vs.
quiet/near-Maxwellian ones), not a numerical artifact — which is what
motivated `sample_normalization` and `moment_feature_weight` above.

### Second stage: SOM within each blind cluster (`run_snapshot_som.py`)

The k=2 KMeans split separates calm from disturbed plasma with perfect
recall of both disturbed substances, but silhouette scores show no clean
second elbow — the substances *within* each tier don't separate at the
KMeans level. `scripts/ml_models/run_snapshot_som.py` probes exactly that
gap: one Self-Organizing Map per blind cluster (`SOM_CONFIG` in
`pipeline_config.py`), fit on the saved `pca_scores` of just that
cluster's samples — the same space the split was found in, so no features
are rebuilt and no VLSV file is opened (chains off `pca_results.npz`,
like `plot_snapshot_pca.py`).

Before each SOM is fit, that cluster's samples get a per-tier PCA refit
truncated to `SOM_CONFIG["n_components"]` — the global PCA's leading
components are partly spent encoding the calm-vs-disturbed split itself,
so within one tier the structure of interest hides in later components.
The truncation is the active ingredient (centering + rotation alone
leave Euclidean distances, and hence the SOM, unchanged).

Each map is painted two ways in `som_label_maps.png`
(`plot_tools.plot_som_label_maps`): samples on their best-matching unit
colored by `cluster_phys` over the U-matrix (dark ridges = boundaries in
codebook space), and the same samples over a KMeans partition of the
trained codebook vectors (`n_node_clusters`) — blind "clusters within
the cluster", weighted by each node's hit count (see `fit_som_map`'s
docstring: at a few dozen samples spread over dozens of nodes, most
nodes are empty or singleton-occupied, and an unweighted KMeans lets
those outvote nodes many samples agreed on). The script scores those
node-clusters against the expert labels the same way
`cluster_summary.csv` scores the top-level KMeans clusters, plus an
adjusted Rand index per tier (1 = expert partition reproduced exactly,
0 = chance) so `SOM_CONFIG` tuning is a measured comparison rather than
a visual impression.

On the fixture, at the current defaults (`n_components = 8`,
`som_shape = (4, 4)`) this recovers structure the flat KMeans `k`-scan
missed: `solar_wind` forms its own 100%-pure node-cluster in the calm
tier (ARI 0.40), and `current_layer` concentrates strongly (60-75%
purity depending on run) in a small node-cluster within the disturbed
tier (ARI 0.31). Both the hit-weighting and the `(4, 4)` map size were
verified against actual node-occupancy counts, not chosen by best score
alone — but the disturbed tier (42 samples) still swings noticeably
between single-seed configurations (e.g. `som_shape` scanned 0.17-0.31),
so treat these as the current best-understood defaults for this
fixture, not a converged optimum; re-validate on a production snapshot.

**Codebook clustering method** (`SOM_CONFIG["node_cluster_method"]`):
KMeans (default) assumes convex node-clusters in codebook space; DBSCAN
(`fit_som_map`'s other branch, same hit-count weighting via
`sample_weight`) doesn't, and can legitimately mark a node `-1` ("noise")
instead of forcing it into a cluster — worth trying when the map's
structure looks like a curved ridge rather than blobs. On the fixture,
hit-weighted DBSCAN at `dbscan_eps = 18` beats the tuned KMeans baseline
on both tiers (calm ARI 0.430 vs 0.404, disturbed ARI 0.330 vs 0.306).
`dbscan_eps` must be calibrated against the codebook's own pairwise-
distance scale, not guessed — an initial scan at eps 0.5-3.0 gave
identical, misleadingly-fragmented output (one cluster per node) because
those values sat far below this fixture's actual minimum pairwise
codebook distance (~4.7-8.9, checked via `scipy.spatial.distance.pdist`
on `som.get_weights()`); eps values that small can never connect two
nodes at all. KMeans stays the default in `SOM_CONFIG` pending a decision
on whether to switch it — this DBSCAN result is measured and reproducible
on the fixture, not yet validated on a production snapshot.

## 2. Legacy CNN-dataset PCA (`src/ml_models/dataset_pca.py`)

Built around the old static-label dataset format
(`lobe`/`exhaust`/`o_point`/`x_point`/`dayside`), not the current
`extract_data.py` pipeline. Two original uses, only one still wired up:

1. ~~Training filter for the CNN~~ — **removed**. The old
   `training_filter` config block, `_run_training_filter_pca`, and
   `_apply_training_filter` no longer exist anywhere in
   `src/ml_models/pytorch_cnn.py` (confirmed by direct search — zero
   matches). CNN training no longer calls into `dataset_pca.py` at all.
2. **Standalone diagnostic/visualization** — still functional:
   ```
   python scripts/data_proc/plot_dataset_pca.py --config configs/plot_dataset_pca.yaml --timestep 3408_100 --pca-id v0.01
   ```
   The filter-preview *metrics and plots* (`add_filter_preview_metrics`,
   `plot_pca_filter_preview`, neighbor-purity computation, etc.) are all
   still present in `dataset_pca.py` and still run here — only the "apply
   this filter to actually drop CNN training samples" wiring was removed.
   Useful for eyeballing class separability in the old label scheme; not
   connected to anything else in the current pipeline.

PCA is refit from scratch every run here too (`sklearn.IncrementalPCA` or
`torch.pca_lowrank`/`torch.linalg.svd`, via `pca.backend`) — see
`dataset_pca.py`'s own docstrings for backend/config details, which are
otherwise unchanged from before this session's refactor.

**Feature source**: log10-scaled xz-slice cache
(`src/ml_models/feature_cache.py`'s `create_or_load_log_slice_cache`),
same representation as the CNN's `raw_vdf` path — same "only intersects
`vy=mid`" limitation as the new pipeline's `"raw"` mode above, for the
same underlying reason.
