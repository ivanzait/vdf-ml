# Project Context: vdf-ml

## Overview

**vdf-ml** is a Python-based scientific computing project for analyzing velocity distribution functions (VDFs) from Vlasiator plasma simulations. It extracts labeled VDF samples from `.vlsv` simulation output and trains/evaluates ML classifiers (logistic regression, perceptron, MLP, CNN) to label VDFs.

## Architecture

### Layer Separation
- **data_proc/physics/**: Physics calculations (X/O critical-point detection, current-layer detection, Shue magnetopause model, VDF rotation + Hermite transform)
- **data_proc/labeling/**: Label assignment subsystem (turns detected point substances + regions into per-VDF-cell labels)
- **ml_models/**: Training-data loading, model classes, training loops, coordinate/region prediction, model checkpoint I/O, PCA diagnostic tool
- **deprecated.py**: Old label scheme (frozen, not built upon)

**Key constraint**: `data_proc/physics/` depends on nothing else under `src/`. The rest of `data_proc/` may depend on `physics/`. `ml_models/` may depend on all of `data_proc/`. The reverse never happens.

### Key Concepts

#### Two clustering notions:
- **cluster_phys**: Physical region label (ground truth from physics detection)
- **cluster_ml**: Blind statistical grouping discovered by unsupervised ML (PCA + KMeans)

#### Substance taxonomy:
- Base regions: `magnetosheath`, `solar_wind`, `inner_magnetosphere`, `lobes`, `no_density_data`, `undefined`
- Point substances: `current_layer` (peak-current-density core), `x_o_points` (Hessian critical-point detector)

## Pipeline

### Current production labeling path:
1. **Extract, label, plot overview**: `scripts/data_proc/extract_data.py`
2. **Verify per-label VDFs**: `scripts/data_proc/verify_data.py`
3. **Other verification angles**: `plot_nulls.py`, `plot_vdf_hermite.py`
4. **Compute cluster_ml**: `run_snapshot_pca.py`, `plot_snapshot_pca.py`

### Training/prediction (legacy format):
Currently only works against old-format datasets (`X.npy`/`y.npy`/`metadata.csv` triple), not the new pipeline's output.

## Configuration

- **Shared config**: `src/data_proc/pipeline_config.py` (one file for snapshot/search-box/detector settings)
- **Script-specific configs**: `configs/` directory (YAML files per script)
- **Runtime ID**: `RUN_ID` used to name output directories

## Dependencies

Core dependencies (from code inspection):
- `analysator` (Vlasiator I/O)
- `numpy`
- `scikit-learn`
- `pytorch`
- `matplotlib`
- `scipy`
- `pandas`

## Development Conventions

See `PIPELINE.md` for rules:
1. **Physics code rules**: This is physicists' code
2. **Multi-user rules**: This is multi-user code
3. **No redundant functions**
4. **One shared config per pipeline**
5. **Layering** (no circular dependencies)
6. **Docstring style**
7. **Verify by running**

## Testing Approach

**No automated test suite** - physics-derived logic is easier to validate by eye against real snapshots than to unit-test against synthetic fixtures. Testing means running pipeline stages against the fixture and checking printed diagnostics/saved plots against "what good looks like" notes.

## File Structure

```
├── src/
│   ├── data_proc/
│   │   ├── physics/          # Physics calculations
│   │   ├── labeling/         # Label assignment
│   │   └── *.py              # Technical tools
│   ├── ml_models/            # ML models and training
│   └── deprecated.py         # Old label scheme (frozen)
├── scripts/
│   ├── data_proc/            # CLI entry points for data processing
│   └── ml_models/            # CLI entry points for ML
├── configs/                  # YAML configs per script
├── docs/                     # Per-function indexes
├── notebooks/                # Jupyter notebooks
└── data/                     # Generated datasets and plots
```

## Known Gaps

1. **Format mismatch**: `extract_data.py` saves `X.npy` + `metadata.csv` (labels as column), but `train_*.py`/`predict_*.py` expect `X.npy`/`y.npy`/`metadata.csv` triple
2. **No bridge**: No script produces legacy-format dataset, no bridge from new pipeline output to training
3. **CNN training**: Not yet wired to current pipeline's output

## Skills

Project-specific skills:
- `load_ml_knowledge`: ML/PCA pipelines, feature representations, CNN training
- `load_physicist_knowledge`: Plasma physics, magnetosphere detection, labeling
- `orchestrate`: Route tasks to correct domain knowledge