# vdf-ml

Extracts labeled velocity distribution function (VDF) samples from Vlasiator
`.vlsv` simulation output, and trains/evaluates ML classifiers (logistic
regression, perceptron, MLP, CNN) that label VDFs.

The **current** production labeling path is physics-driven: magnetosphere
regions from a subsolar-anchored Shue (1998) model, plus whichever
"point-like" substance(s) are toggled on in `pipeline_config.py` --
`current_layer` (peak-current-density core, exact cellid match, the
default) and/or `x_o_points` (a Hessian critical-point detector) -- via
`scripts/data_proc/extract_data.py`. See [`schema.md`](schema.md) for the
substance concept and how to add a new one. An **older**, Santtu-authored
label scheme (static `lobe`/`exhaust`/`o_point`/`x_point`/`dayside` classes)
has its scripts removed; the code it depended on is consolidated (but no
longer called by anything) in `src/deprecated.py` — see `docs/DEPRECATED.md`.

**Known gap**: `extract_data.py` saves `X.npy` + `metadata.csv`
(labels as a metadata column), not the `X.npy`/`y.npy`/`metadata.csv` triple
`train_*.py`/`predict_*.py` and the remaining old-format tools
(`plot_dataset_sample.py`, `plot_dataset_pca.py`)
expect via `dataset_io.load_dataset`. Training/prediction currently only run
against datasets already saved in the old format (e.g. `data/local_smoke_test/`)
— there is no current script producing a new legacy-format dataset, and no
bridge yet from the new pipeline's output to training. Fixing `load_dataset`/
`save_labeled_vdfs` to agree on one format is the next step before training
can run against the new labels.

## Terminology: `cluster_phys` vs `cluster_ml`

This repo has two different, deliberately separate notions of "cluster" —
don't conflate them:

- **`cluster_phys`** — a physical region label: which magnetospheric
  structure a VDF cell physically belongs to. Made of base region
  substances (`magnetosheath`, `solar_wind`, `inner_magnetosphere`,
  `lobes`, `no_density_data`, `undefined` -- always computed, in a fixed
  priority order, see [`schema.md`](schema.md)) plus whichever point
  substance(s) are toggled on
  (`current_layer`; or `x_point`/`o_point`/`x_point_o_point`; or both --
  see `pipeline_config.py`'s `POINTS_CONFIG["active_point_substances"]` and
  [`schema.md`](schema.md)). This is ground truth, not discovered — it's
  what `extract_data.py` computes and saves as `metadata.csv`'s `label`
  column, and what `verify_data.py` visualizes one representative VDF per
  label for.
- **`cluster_ml`** — a blind, purely statistical grouping discovered by
  unsupervised ML (currently PCA + KMeans, auto-`k` via silhouette score)
  run on the same VDF features, with no knowledge of `cluster_phys`. Its
  purpose is to sanity-check whether the physical structure is also
  recoverable from the data alone. Computed by `run_snapshot_pca.py`,
  visualized by `plot_snapshot_pca.py`.

`run_snapshot_pca.py`/`plot_snapshot_pca.py` score `cluster_ml` against
`cluster_phys` (do the blind clusters line up with the physical labels).
Its PCA scatter plot (`pca_scatter.png`) draws both on the same PC1-vs-PC2
points with two separate colorbars — a large translucent ring per point
colored by `cluster_phys`, a small solid dot on top colored by
`cluster_ml` — see "Pipeline" step 4 below.

## Layout

```
src/data_proc/            Technical tools: VLSV reading, region/box masks,
                           dataset I/O, plotting. Scheme-agnostic.
  physics/                 Physics calculations: X/O critical-point detection,
                            current-layer (peak-|J| core) detection, Shue
                            magnetopause model, VDF rotation + Hermite
                            transform (incl. an LMN/MDD-MGA boundary-normal
                            basis builder, kept as a future rotation-frame
                            option, currently unused). No dataset/label-scheme
                            assumptions.
  labeling/                Current label-assignment subsystem: turns detected
                            point substances + regions into per-VDF-cell
                            labels (see schema.md).
src/ml_models/             Training-data loading, model classes, training
                           loops, coordinate/region prediction, model
                           checkpoint I/O, the standalone PCA diagnostic tool.
src/deprecated.py          The old label scheme, consolidated, kept working
                           but frozen -- do not build on top of it.
scripts/                   Thin CLI entry points, mirroring data_proc/ml_models
                           (scripts/data_proc/, scripts/ml_models/). This is
                           what you actually run.
configs/                   One YAML config per script (`--config path/to/x.yaml`).
docs/                      Per-area function-name index -- see docs/README.md.
```

**Layering**: `data_proc/physics/` depends on nothing else under `src/`
(besides the low-level VLSV-reading primitives in `data_proc/vdf_tools.py`
needed to read field data off a reader). The rest of `data_proc/` (including
`labeling/`) may depend on `physics/`. `ml_models/` may depend on all of
`data_proc/`. **The reverse never happens** — no file under `src/data_proc/`
may import anything from `src/ml_models/`. `src/deprecated.py` may depend on
`data_proc/`; nothing depends on `deprecated.py`.

### Function-level docs

Full per-function indexes live in [`docs/`](docs/README.md):
[`PHYSICS.md`](docs/PHYSICS.md), [`DATA_PROC.md`](docs/DATA_PROC.md),
[`LABELING.md`](docs/LABELING.md), [`ML_MODELS.md`](docs/ML_MODELS.md),
[`DEPRECATED.md`](docs/DEPRECATED.md) — check there before grepping the
whole tree for whether a helper already exists, and update the relevant file
whenever you add, rename, or remove a function.

## Setup

```
source .venv/bin/activate
export PTNOLATEX=1                                  # analysator plots need this locally (no LaTeX binary)
export PYTHONPATH=/path/to/your/analysator/checkout  # if analysator isn't pip-installed
```

## Conventions

See [`PIPELINE.md`](PIPELINE.md) for the rules governing any change to this
codebase — two ground rules (this is physicists' code; this is multi-user
code) plus the concrete conventions (no redundant functions, one shared
config per pipeline, layering, docstring style, verify-by-running). Read it
before making a nontrivial change, human or agent.

## Pipeline

### Extraction + labeling (current)

`extract_data.py` and `verify_data.py` both read
`src/data_proc/pipeline_config.py` — one shared file for the
snapshot/search-box/detector settings, so extraction and every
verification plot always agree on what "this run" means. Edit
`pipeline_config.py`, not the scripts, to change the run.

1. **Extract, label, and plot the overview** in one step:
   `python scripts/data_proc/extract_data.py`. Runs Shue-model region
   classification plus whichever point substance(s)
   `POINTS_CONFIG["active_point_substances"]` selects (`current_layer`
   and/or `x_o_points`, see [`schema.md`](schema.md)), saves `X.npy` +
   `metadata.csv` (per-cell `B`/bulk `V` and the velocity-mesh extent are
   saved too, so a representation choice like rotation can be made
   downstream without reopening the `.vlsv` file; no `y.npy` -- see the
   known gap above), then immediately plots every extracted cell colored by
   its label with the magnetopause/bow-shock boundaries drawn on top --
   using the labels already computed, no second snapshot pass. If
   `pipeline_config.BUILD_ROTATED_DATASET = True`, also saves
   `X_rotated.npy` -- every VDF rotated into its own local
   `(B, v_perp, B x v_perp)` frame at full resolution (off by default, a
   real per-sample interpolation cost). If `BUILD_HERMITE_DATASET = True`,
   also saves `X_hermite.npy` -- every VDF's log-space Hermite spectra
   (rotation runs as a prerequisite either way, whether or not
   `BUILD_ROTATED_DATASET` itself is on); see
   `PCA_CONFIG["feature_representation"]` in step 4 below.

2. **Verify per-label VDFs**: `python scripts/data_proc/verify_data.py`.
   Picks one representative VDF per label (same `pipeline_config.py`, same
   ground truth) and produces two plots: where each one sits spatially, and
   its three velocity-space cuts (vx-vy, vx-vz, vy-vz) sliced through its
   own peak.

3. **Other verification angles** (independent parameters, not wired into
   `pipeline_config.py` -- see `PIPELINE.md`'s "one shared config per
   pipeline" rule):
   ```
   python scripts/data_proc/plot_nulls.py        # sanity-check the X/O detector + search boxes
   python scripts/data_proc/plot_vdf_hermite.py  # ad-hoc VDF/Hermite/rotation exploration
   ```

4. **(Optional) compute `cluster_ml`** on the extracted dataset and score it
   against `cluster_phys` (see "Terminology" above):
   ```
   python scripts/ml_models/run_snapshot_pca.py    # PCA + KMeans (auto-k), scored against metadata.csv's cluster_phys (label column)
   python scripts/ml_models/plot_snapshot_pca.py    # silhouette/PC1-PC2/smallest-cluster/representative-VDF plots
   ```
   `PCA_CONFIG["feature_representation"]` switches which array gets
   clustered: `"raw"` (`X.npy`, simulation-frame VDFs, the default),
   `"rotated"` (`X_rotated.npy`, each VDF in its own local
   `(B, v_perp, B x v_perp)` frame -- removes local-field-orientation as a
   confound, since the same physical distribution can otherwise look very
   different depending on where in the domain it sits; needs
   `BUILD_ROTATED_DATASET = True` on the `extract_data.py` run that
   produced this dataset), or `"hermite"` (`X_hermite.npy`, flattened
   directly -- log-space Hermite spectra of the rotated VDF, u/vth- and
   log-scale-normalized so bulk-speed/temperature/orientation differences
   don't dominate the representation the way they do for a raw pixel
   slice; needs `BUILD_HERMITE_DATASET = True`).
   `run_snapshot_pca.py` needs `extract_data.py` to have already run for
   this `RUN_ID` (it loads `data/snapshot_vdfs/<RUN_ID>/`) and never opens
   the VLSV file itself; `plot_snapshot_pca.py` reads its saved
   `data/pca/<RUN_ID>/pca_results.npz` and only then opens the VLSV file,
   for plot backgrounds and VDF re-extraction.

### Training + prediction (currently only against old-format datasets)

`--dataset-id` / `--model-id` are your own labels for a given run (e.g.
`3408_100`, `v1.0`) used to name output directories. These scripts read
datasets via `dataset_io.load_dataset`, which requires an `X.npy`/`y.npy`/
`metadata.csv` triple — not what `extract_data.py` currently
produces (see the known gap above). They still work against any
already-saved dataset in that older format (e.g. `data/local_smoke_test/`).

```
python scripts/ml_models/train_cnn.py --config configs/train_pytorch_convolutional_neural_network_classifier.yaml \
    --dataset-id 3408_100 --model-id v1.0

python scripts/ml_models/train_classifier.py --model-type logistic_regression \
    --config configs/train_logistic_regression.yaml --dataset-id 3408_100 --model-id v1.0

python scripts/ml_models/predict_coordinate.py --model-type cnn \
    --config configs/predict_coordinate_pytorch_convolutional_neural_network_classifier.yaml \
    --timestep 4000 --model-id v1.0 --coord-re -12 0 0

python scripts/ml_models/predict_region.py --model-type cnn \
    --config configs/predict_region_pytorch_convolutional_neural_network_classifier.yaml \
    --start-timestep 1300 --n-timesteps 100 --model-id v1.0
```

`--model-type` is one of `logistic_regression`, `perceptron`,
`multilayer_perceptron_classifier` (via `train_classifier.py`/
`predict_coordinate.py`/`predict_region.py`), or `cnn` (`train_cnn.py`, its
own script since it takes a different config shape --
`features.representation: raw_vdf|hermite`, conv `channels`, etc.).

Also still usable against an old-format dataset: `plot_dataset_sample.py`,
`plot_dataset_pca.py`. (`inspect_dataset.py`
was removed -- `extract_data.py` now prints VDF statistics itself right
after saving, so a separate load-and-inspect step is no longer needed for
the current pipeline.)

## Other tools

- `TESTING.md` — how to validate a change to `src/data_proc/` (no automated
  test suite; a stage-by-stage manual verification guide against a local
  fixture snapshot, plus a table of failure modes caught during development
  so they don't silently regress).
- `pca_guide.md` — the standalone PCA diagnostic tool
  (`scripts/data_proc/plot_dataset_pca.py`): class-separability plots and
  neighbor-purity metrics on a dataset's VDF features. Not part of the CNN
  training path.
- `scripts/data_proc/plot_nulls.py` — distinct from `plot_vdf_hermite.py`:
  this one runs the actual X/O detector and shows its selection geometry
  (search boxes/contours + match counts), rather than letting you pick
  cells manually. See "Pipeline" above for all three plotting scripts.
- `scripts/ml_models/run_snapshot_pca.py`/`plot_snapshot_pca.py` — the
  closest thing this repo has to an integration test for `data_proc`'s
  topology/labeling/magnetopause code (checks `cluster_ml` against
  `cluster_phys`, see "Terminology" above). See "Pipeline" above.
