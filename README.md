# vdf-ml

Extracts labeled velocity distribution function (VDF) samples from Vlasiator
`.vlsv` simulation output, and trains/evaluates ML classifiers (logistic
regression, perceptron, MLP, CNN) that label VDFs by magnetospheric region
(lobe, exhaust, X-point, O-point, dayside).

## Layout

```
src/data_proc/   VLSV reading, VDF extraction, Hermite transform + rotation,
                 X/O-point topology detection/selection/labeling, dataset
                 I/O, all plotting, YAML config loading.
src/ml_models/   Training-data loading, model classes, training loops,
                 coordinate/region prediction, model checkpoint I/O, the
                 standalone PCA diagnostic tool.
scripts/         Thin CLI entry points, mirroring the two folders above
                 (scripts/data_proc/, scripts/ml_models/). This is what you
                 actually run.
configs/         One YAML config per script (`--config path/to/x.yaml`).
```

`ml_models` depends on `data_proc`; the reverse never happens.

## Setup

```
source .venv/bin/activate
export PTNOLATEX=1                                  # analysator plots need this locally (no LaTeX binary)
export PYTHONPATH=/path/to/your/analysator/checkout  # if analysator isn't pip-installed
```

## Pipeline

Run these in order. `--dataset-id` / `--model-id` are your own labels for a
given run (e.g. `3408_100`, `v1.0`) used to name output directories.

1. **Create a dataset** from `.vlsv` files (+ flux files for X/O-point
   detection):
   ```
   python scripts/data_proc/create_dataset.py --config configs/create_dataset.yaml \
       --start-timestep 3408 --n-timesteps 100 --dataset-kind train
   ```
   Set `hermite.enabled: true` in the config to store Hermite spectra
   instead of raw VDFs (see `hermite.order`, `hermite.rotate`).

2. **(Optional) backfill metadata** on an existing dataset (e.g. after
   changing point-topology settings):
   ```
   python scripts/data_proc/backfill_dataset_metadata.py --config configs/create_dataset.yaml \
       --dataset-id 3408_100
   ```

3. **Train** a classifier:
   ```
   python scripts/ml_models/train_cnn.py --config configs/train_pytorch_convolutional_neural_network_classifier.yaml \
       --dataset-id 3408_100 --model-id v1.0

   python scripts/ml_models/train_classifier.py --model-type logistic_regression \
       --config configs/train_logistic_regression.yaml --dataset-id 3408_100 --model-id v1.0
   ```
   `--model-type` is one of `logistic_regression`, `perceptron`,
   `multilayer_perceptron_classifier`. The CNN has its own script
   (`train_cnn.py`) since it takes a different config shape (`features.representation: raw_vdf|hermite`, conv `channels`, etc.).

4. **Predict** with a trained model, against a fresh `.vlsv` file:
   ```
   python scripts/ml_models/predict_coordinate.py --model-type cnn \
       --config configs/predict_coordinate_pytorch_convolutional_neural_network_classifier.yaml \
       --timestep 4000 --model-id v1.0 --coord-re -12 0 0

   python scripts/ml_models/predict_region.py --model-type cnn \
       --config configs/predict_region_pytorch_convolutional_neural_network_classifier.yaml \
       --start-timestep 1300 --n-timesteps 100 --model-id v1.0
   ```
   `--model-type` is one of `logistic_regression`, `perceptron`,
   `multilayer_perceptron_classifier`, `cnn`.

5. **Verify visually**: overlay the sampled/labeled points on a colormap, or
   inspect a single VDF:
   ```
   python scripts/data_proc/plot_dataset_labels.py    # edit its PARAMETERS block first
   python scripts/data_proc/plot2Dmap.py              # ad-hoc single-file exploration + Hermite/rotation QA
   python scripts/data_proc/plot_dataset_sample.py --config configs/plot_dataset_sample.yaml \
       --timestep 3408_100 --sample-index 11
   python scripts/data_proc/inspect_dataset.py <dataset_dir>
   ```

`plot_dataset_labels.py` and `plot2Dmap.py` take their parameters as plain
Python variables at the top of the file instead of a YAML config — edit and
rerun. Everything else uses `--config`.

## Other tools

- `pca_guide.md` — the standalone PCA diagnostic tool
  (`scripts/data_proc/plot_dataset_pca.py`): class-separability plots and
  neighbor-purity metrics on a dataset's VDF features. Not part of the CNN
  training path.
- `scripts/data_proc/animate_dataset_by_class.py` — stitches per-class plot
  frames into a GIF/animation.
- `scripts/data_proc/create_log_slice_cache.py` — precomputes the log-scaled
  xz-slice feature cache used by the `raw_vdf` CNN/logreg/perceptron/MLP
  feature path (not needed for `hermite` representation).
