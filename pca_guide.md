# PCA Guide

How PCA is used across this repo. Everything lives in [src/dataset_pca.py](src/dataset_pca.py) (~4900 lines) except two unrelated, unconnected uses noted at the bottom.

## Role

PCA is **not** a standalone model used at inference time. It has two uses:

1. **Training filter for the CNN** — flags and removes noisy/borderline training samples before `train_pytorch_convolutional_neural_network_classifier` runs (`training_filter.source: pca`).
2. **Standalone diagnostic/visualization** — `scripts/plot_dataset_pca.py`, used to inspect class separability and tune filter thresholds before wiring them into training.

PCA is **refit from scratch on every run** — no fitted model is ever pickled/saved (`torch.save`/`joblib.dump`). Only the *output* (scores, metrics, plots) is persisted to disk.

## Feature source

PCA does **not** operate on raw 3D VDFs or Hermite spectra. It runs on the same representation as the `raw_vdf` CNN path: the log10-scaled **xz-slice** of each VDF (`create_pca_feature_cache_input_config`, [src/dataset_pca.py:591](src/dataset_pca.py)), built via the shared cache in [src/autoencoder_data.py](src/autoencoder_data.py) (`create_or_load_log_slice_cache`). Optional `downsample_factor` shrinks the slice before PCA.

**Consequence:** `training_filter.source: pca` is incompatible with `features.representation: hermite` — enforced by an explicit `ValueError` in [src/pytorch_cnn.py:1256](src/pytorch_cnn.py).

## Backends

Controlled by `pca.backend`:

| Backend | Implementation |
|---|---|
| `sklearn` (`incremental`) | `sklearn.decomposition.IncrementalPCA`, `partial_fit` in batches over a `StandardScaler`-scaled matrix |
| `torch` (default) | `algorithm: lowrank` → `torch.pca_lowrank(x, q=n_components+oversamples, niter=…)`; `algorithm: exact` → `torch.linalg.svd` |

`torch` backend has an optional multi-GPU path (manual sharding + distributed matmuls, `fit_multi_device_lowrank_pca`), with automatic CPU fallback on CUDA OOM.

## CNN training filter — how it decides what to remove

Call chain in `train_pytorch_convolutional_neural_network_classifier` ([src/pytorch_cnn.py:1188](src/pytorch_cnn.py)):

1. **`_run_training_filter_pca`** runs the full `plot_dataset_pca` pipeline (config from `training_filter.pca.*`) before CNN data loading, saving `pca_sample_metrics.csv` under `<model_output_dir>/pca/`.
2. **`_apply_training_filter`** (source=`pca`) reads that CSV back, joins it against the CNN's actual train split, and applies the filter.

**Rule** (from `add_filter_preview_metrics` in `dataset_pca.py`): for each training sample of a *candidate* class (default `exhaust`, `dayside`), look at its `k_neighbors` (25) nearest neighbors in PCA space (fit on a class-balanced training subset). Flag it for removal if:

- fraction of neighbors from the *point* classes (`x_point`, `o_point`) ≥ `min_point_neighbor_fraction` (0.5, overridable per class — `exhaust: 0.4`, `dayside: 0.6`), **and**
- fraction of same-class neighbors (`same_class_fraction`) ≤ `max_same_class_fraction` (0.60).

In plain terms: *an exhaust/dayside sample that looks more like x_point/o_point than its own class in PCA space is probably a noisy/borderline label — drop it from training.*

- `protected_classes` (`x_point`, `o_point` by default) are never removed.
- Removal per class is capped at `max_removed_fraction_per_class` (0.3); among eligible candidates, the worst are removed first (`_rank_pca_filter_candidates`: sort by `point_neighbor_fraction` desc, then `same_class_fraction` asc).
- `dry_run: true` computes everything but skips the actual removal.

## Standalone visualization

```
python scripts/plot_dataset_pca.py --config configs/plot_dataset_pca.yaml --timestep 3408_100 --pca-id v0.01
```

Independent of CNN training. Outputs under `<output_dir>/pca[_v<id>]/`:
- `pca_sample_metrics.csv`, `pca_scores.npz`, `pca_metrics.txt`
- `pca_train_by_class.png`, `pca_train_neighbor_purity.png`, `pca_filter_preview.png`, `pca_tsne_by_class.png` (t-SNE/UMAP on PCA space)

Used to eyeball class separability and calibrate `filter_preview` thresholds before enabling the CNN filter.

## Key config blocks

Same schema in [configs/plot_dataset_pca.yaml](configs/plot_dataset_pca.yaml) (top-level) and [configs/train_pytorch_convolutional_neural_network_classifier.yaml](configs/train_pytorch_convolutional_neural_network_classifier.yaml) (nested under `training_filter.pca.*`):

- `features.*` — xz-slice cache settings (downsample, log_eps, cache dir).
- `pca.*` — backend/algorithm/n_components/device.
- `pca_fit.*` — class-balanced subsampling used only to *fit* PCA (`class_names` excludes `lobe` by default so the majority class doesn't dominate the axes).
- `neighbor_metrics.*` — `k_neighbors`, batch size, purity-bucket thresholds for reporting.
- `filter_preview.*` — the actual filter rule: `candidate_classes`, `point_neighbor_classes`, `protected_classes`, `min_point_neighbor_fraction[_by_class]`, `max_same_class_fraction[_by_class]`.
- `plot.*` / `embedding_plot.*` — visualization toggles (usually off in the CNN training config for speed).

`training_filter.*` (CNN config, outside the nested `pca` block) additionally has: `enabled`, `source` (`pca` or `cnn_embedding_knn`), `dry_run`, `candidate_classes`, `point_neighbor_classes`, `protected_classes`, `max_removed_fraction_per_class`.

## Alternative: `cnn_embedding_knn` filter

`training_filter.source` is either `pca` or `cnn_embedding_knn` (mutually exclusive). The embedding-kNN path applies the same neighbor-purity idea, but on a trained CNN's penultimate-layer embeddings instead of PCA scores — it's a **separate, independently reimplemented** code path in `pytorch_cnn.py` (not routed through `dataset_pca.py`), requiring a model to already exist (or training an extra "source" model via `run_source_model: true`).

**Shared code between the two filters:** only the final candidate-selection/ranking (`_select_pca_filter_candidates`, `_rank_pca_filter_candidates`) and row-removal (`_remove_training_samples`) functions, both defined once in `pytorch_cnn.py`.

**Conceptual difference:** PCA filter = cheap, unsupervised, pre-training check in a linear subspace of raw VDF features. Embedding filter = same idea, but in the task-supervised space the CNN itself learned.

## Unrelated PCA usages (not part of this pipeline)

- [src/autoencoder_plot.py:374](src/autoencoder_plot.py) `plot_latent_pca` — manual `numpy.linalg.svd` 2D projection of the **autoencoder's** latent space, for `latent_pca.png` only.
- [src/pytorch_cnn.py:3055](src/pytorch_cnn.py) — a throwaway `sklearn.decomposition.PCA(n_components=2)` used purely to get 2D scatter-plot coordinates for CNN embeddings, unrelated to the main PCA pipeline.
