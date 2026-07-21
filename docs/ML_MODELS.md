# ML models — `src/ml_models/`

Training data loading, model classes, training loops, prediction, model
checkpoint I/O, and the standalone PCA diagnostic tool. May depend on all of
`src/data_proc/` (including `physics/` and `labeling/`); nothing under
`src/data_proc/` may depend back on this package.

**When adding a function here**: add a one-line entry below. For
`dataset_pca.py`, only the small set of genuinely public entry points are
listed individually — it's a large (~4900-line) standalone diagnostic tool
with many private `resolve_*`/`create_*`/`fit_*` helpers; grep the file for
the full internal call graph rather than expecting every helper listed here.

## `coordinate_prediction.py` — CNN/classifier prediction pipeline (live, production)

- `create_model_features(vdfs, cids, model, reader, downsample_factor, ...)` — build the feature matrix a trained model expects from raw dense VDFs, mirroring the training-time representation (raw xz-slice vs. rotated Hermite spectra via `src.data_proc.physics.vdf_transform`).
- `predict_coordinate(...)` — predict a class/coordinate for VDFs at given cellids.
- `predict_region_timesteps(...)` / `predict_region(config, timestep, model_id, load_model, file_source=None)` — predict region labels across a spatial box for one or more timesteps.
- `predict_batch_with_scores(model, features, prediction_kwargs=None)` — batched prediction with confidence scores.
- `resolve_file_source_and_template(config, file_source=None)` — resolve which bulk-file template/source a prediction run uses.
- `create_class_probability_fields(class_labels, label_to_class)` — per-class probability columns for output rows.
- `get_batch_prediction_scores(...)` / `get_batch_decision_scores(...)` / `get_prediction_score(...)` — per-sample confidence/decision-function scores.
- `create_region_prediction_rows(...)` — assemble output rows for a region-prediction run.

## `model_io.py` — checkpoint loading

- `load_perceptron_model(model_dir)` / `load_logistic_regression_model(model_dir)` / `load_multilayer_perceptron_classifier_model(model_dir)` / `load_pytorch_convolutional_neural_network_classifier_model(model_dir)` — model-type-specific checkpoint loaders.
- `load_model(model_dir, model_filename)` — generic joblib/pickle checkpoint load used by the loaders above.

## `model_split.py`

- `split_by_timestep(...)` — train/val/test split by timestep (not by sample) to avoid leakage across nearby samples in the same snapshot.
- `_indices_for_timesteps(metadata, timesteps)` — row indices belonging to a set of timesteps.

## `model_evaluation.py`

- `create_lobe_vs_rest_labels(metadata, positive_class_name="lobe")` — binary one-vs-rest label array for a given class.
- `create_predictions_dataframe(metadata, indices, y_true, y_pred, split_name)` — tidy predictions table for evaluation/plotting.

## `features.py` — xz-slice + log-scaling feature extraction (non-CNN models)

- `create_features(X, downsample_factor=8, log_eps=1e-30, n_jobs=1)` — full feature matrix from a VDF dataset.
- `create_features_chunk(...)` / `create_feature(vdf, downsample_factor=8, log_eps=1e-30)` — per-chunk / per-sample feature extraction.
- `downsample_2d(array, factor)` / `downsample_2d_batch(arrays, factor)` — spatial downsampling.
- `create_features_in_batches(...)` — batched, optionally parallel feature extraction.
- `create_features_from_log_slice_cache(...)` / `create_log_slice_cache_feature_batch(...)` — build features from a precomputed log-slice cache (see `feature_cache.py`).
- `normalize_feature_samples(features, sample_normalization, sample_norm_eps)` — per-sample feature normalization.

## `feature_cache.py` — log-scaled xz-slice cache shared by `training.py` and `dataset_pca.py`

- `resolve_cache_config(...)` — resolve cache paths/settings from config.
- `create_or_load_log_slice_cache(X, input_config, cache_config)` / `create_log_slice_cache(...)` / `write_log_slice_cache_batch(...)` — create-or-reuse and batched-write logic.
- `is_log_slice_cache_valid(...)` / `save_log_slice_cache_metadata(...)` / `load_log_slice_cache_metadata(...)` — cache validity check + metadata I/O.
- `infer_plot_xz_slice_shape(X)` / `extract_plot_xz_slice_from_dataset(X, sample_index)` — memmap-friendly xz-slice shape/extraction for batched arrays (intentionally separate from `plot_tools.extract_plot_xz_slice`, which operates on one dense VDF at a time — see that file's docstring).
- `create_log_plot_xz_slice_from_dataset(...)` — log-scaled xz slice from a saved sample.
- `create_log_slice(physical_slice, log_eps=1e-30, clip_negative_to_zero=True)` / `normalize_log_slice(...)` / `denormalize_log_slice(...)` / `log_slice_to_physical(...)` / `normalized_log_slice_to_physical(...)` — log-scale transform and its inverses.

## `pytorch_cnn.py` — CNN classifier (raw-VDF or Hermite-spectra input)

- `apply_pytorch_cnn_training_overrides(...)` — apply CLI/config overrides to CNN training config.
- `class PyTorchCNNClassifier(nn.Module)` — the CNN model: normalize → conv stack (2D for raw-VDF xz-slice, or 3D for Hermite spectra, depending on `representation`/`volume_shape`/`hermite_rotate`) → classifier head.
- `train_pytorch_convolutional_neural_network_classifier(...)` — main training entry point.
- `load_pytorch_cnn_checkpoint(...)` / `save_pytorch_cnn_checkpoint(model, checkpoint_path)` — checkpoint I/O.
- `_train_cnn_model_for_data(...)` / `_fit_model(...)` — internal training loop pieces.
- `_plot_failure_cases(...)` — failure-case xz-slice plots (uses `plot_tools.plot_vdf_xz_slice`).
- Remaining `_resolve_*`/`_create_*` functions are private training-config/device/seed/class-weight helpers.

## `training.py` — non-CNN model training (logistic regression, perceptron, MLP)

- `train_logistic_regression(config, dataset_id, model_id)` / `train_perceptron(...)` / `train_multilayer_perceptron_classifier(...)` — per-model-type training entry points.
- `_fit_sklearn_mlp_with_timestep_validation(...)` — MLP fit with a timestep-aware validation split.
- `train_binary_classifier(...)` — shared binary-classifier training path.
- `load_training_data(config, dataset_id, model_id, target_kind)` — load features/labels for training, dispatching on raw-vdf vs. hermite representation.
- `_create_feature_cache_input_config(...)` / `_create_training_feature_matrix(...)` / `_create_hermite_feature_matrix(...)` — feature-matrix construction helpers.
- `evaluate_model(...)` / `create_predictions(...)` — evaluation + prediction-table construction.
- `save_training_outputs(...)` / `save_training_artifacts(...)` / `create_metrics_text(...)` — save checkpoints/metrics/plots.

## `vdf_snapshot_clustering.py` — PCA/KMeans blind clustering (genuine ML)

Used by `scripts/ml_models/run_snapshot_pca.py` (fit + score) and
`scripts/ml_models/plot_snapshot_pca.py` (re-scores cheaply from saved
results to pick the smallest clusters to plot -- see `docs/DATA_PROC.md`'s
cluster-plotting pipeline section).

- `fit_pca_clusters(features, k_range, n_components=20, random_state=1234)` — PCA-reduce then KMeans-cluster, auto-selecting `k` by silhouette score.
- `summarize_clusters_against_ground_truth(cellids, labels, ground_truth_cellids_by_category)` — per-cluster size + ground-truth-category overlap, scored against `src.data_proc.labeling.snapshot_labeling`'s output (read back from `extract_data.py`'s saved `metadata.csv`, not recomputed).

(All plotting for this module -- silhouette/PCA-scatter diagnostics and the 3-step cluster-visualization pipeline -- lives in `src/data_proc/plot_tools.py`, not here; see `docs/DATA_PROC.md`. It moved to `data_proc` because it only ever consumed plain cellids/labels, never any PCA/KMeans internals.)

## `dataset_pca.py` — standalone PCA diagnostic tool (`scripts/plot_dataset_pca.py`)

Key entry points:
- `plot_dataset_pca(config, timestep, pca_id=None)` — the main entry point: fit PCA (CPU/incremental or multi-GPU torch path), score neighbor purity against ground truth, save plots + metrics text.
- `apply_dataset_pca_preset(config, pca_preset=None)` — apply a named config preset.
- `create_pca_analysis_features(...)` — build the feature matrix (VDF + optional spatial branches) PCA runs on.
- `fit_incremental_pca(...)` / `fit_transform_torch_pca(...)` / `fit_multi_device_lowrank_pca(...)` — CPU vs. single/multi-GPU PCA fit paths.
- `compute_neighbor_purity(...)` / `compute_point_neighbor_metrics(...)` — the neighbor-purity ground-truth scoring this tool is built around.
- `save_pca_plots(...)` / `save_pca_outputs(...)` — save the diagnostic plot set + metrics.
- Everything else (`resolve_*_config`, `fit_transform_*`, `create_*_metric_lines`, `format_*`, `plot_pca_*`) is internal to the above; see `pca_guide.md` at the repo root for the tool's usage guide.
