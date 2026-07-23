# Key Concepts Cheat Sheet for AI Agents

**Purpose**: Essential terms and relationships for understanding vdf-ml.

## Core Terminology

### 1. VDF (Velocity Distribution Function)
- **What**: Particle velocity distribution in space plasma
- **Source**: Vlasiator `.vlsv` simulation output
- **Processing**: Extracted as 3D arrays, potentially rotated/transformed

### 2. `cluster_phys` vs `cluster_ml`
- **`cluster_phys`**: Physical region label (ground truth)
  - Base substances + active point substances
  - Computed by physics detection (Shue model + point detectors)
  - Saved in `metadata.csv` as `label` column

- **`cluster_ml`**: Statistical grouping (discovered)
  - Found by unsupervised ML (PCA + KMeans)
  - No knowledge of `cluster_phys`
  - Purpose: Sanity check if physics structure is statistically recoverable

**CRITICAL**: Don't conflate these! Different purposes, different computations.

### 3. Substance Taxonomy
```
Base substances (always computed):
├── magnetosheath
├── solar_wind  
├── inner_magnetosphere
├── lobes
├── no_density_data
└── undefined

Point substances (configurable):
├── current_layer (default) - peak current density cores
├── x_point - Hessian critical points
├── o_point - Hessian critical points
└── x_point_o_point - both X and O points

cluster_phys = base substance + active point substances
```

### 4. Two PCA Pipelines
```
Snapshot PCA (current):
├── Script: run_snapshot_pca.py / plot_snapshot_pca.py
├── Input: A saved snapshot dataset (extract_data.py's output)
├── Output: cluster_ml for that snapshot
└── Compare: With cluster_phys for sanity check

Dataset PCA (legacy):
├── Script: src/ml_models/dataset_pca.py
├── Input: Old-format (X.npy/y.npy/metadata.csv) dataset
├── Output: Class-separability plots, neighbor-purity metrics
└── Status: Standalone diagnostic tool, not deprecated -- just no longer
    wired into CNN training filtering (see docs/PCA_GUIDE.md)
```

### 5. Two Dataset Formats (Intentional, Not a Mismatch to Fix)
```
New pipeline (extract_data.py):
├── X.npy - VDF features
├── metadata.csv - labels in 'label' column
└── No y.npy file

Old tools (train_*.py, predict_*.py):
├── Expect: X.npy, y.npy, metadata.csv triple
└── Current: Only work with old-format datasets

These are two independent, permanently separate pipelines by design (see
README's "known gap" and docs/PCA_GUIDE.md) -- not a bridge that's missing
and needs building. Don't propose one unless the user explicitly asks.
```

## Architectural Relationships

### Layer Separation
```
data_proc/ (physics)          ml_models/ (ML)
├── physics/                  ├── features/
├── labeling/                 ├── training/
├── pipeline_config.py       ├── prediction/
└── NEVER imports ml_models  └── Separate pipeline
```

**RULE**: `data_proc` never imports `ml_models` (enforced by grep check)

### Configuration Files
```
Shared physics config:
└── pipeline_config.py
   ├── Used by: extract_data.py, verify_data.py
   ├── Used by: run_snapshot_pca.py, plot_snapshot_pca.py
   └── Ensures: All scripts agree on "one run"

ML training configs:
└── configs/*.yaml
   ├── Used by: train_*.py, predict_*.py
   └── Separate: From physics pipeline
```

### Feature Representations
```
Raw VDF → Rotation transform → Hermite spectra
├── Raw: Direct 3D VDF array
├── Rotated: Local (B, v_perp, B x v_perp) frame  
└── Hermite: Log-space spectral representation

Config control:
├── BUILD_ROTATED_DATASET: Save rotated VDFs
├── BUILD_HERMITE_DATASET: Save Hermite features
└── PCA_CONFIG["feature_representation"]: Which to use
```

## Development Rules (from docs/PIPELINE.md)

### 1. No Redundant Functions
- **Check first**: `grep -rn "def .*<verb>" src/`
- **Common locations**: `vdf_tools.py`, `batches.py`, `config.py`
- **Example**: `iter_index_batches` consolidated from 4 implementations

### 2. One Shared Config Per Pipeline
- **Physics pipeline**: Single `pipeline_config.py`
- **ML pipeline**: Separate `configs/*.yaml` system
- **Don't convert**: Between styles without discussion

### 3. Docstring Philosophy
- **Minimal**: One line unless non-obvious WHY
- **Physical focus**: Explain subtlety, convention, gotcha
- **Not**: Restating what function name shows

### 4. Multi-User Awareness
- **Check**: `git status`, `git log -3` before assuming state
- **Prefer**: Small, independently-committable changes
- **Diff**: If file looks different, compare before overwriting

## Common AI Confusion Points

### 1. "Why two PCA pipelines?"
1. **Snapshot PCA**: Current production, compares `cluster_ml` to `cluster_phys`
2. **Dataset PCA**: Legacy approach, whole dataset analysis, still a live
   standalone diagnostic tool
3. **Not redundant**: different inputs (single snapshot vs. whole old-format
   dataset), different purposes -- see `docs/PCA_GUIDE.md`

### 2. "How to add new substance?"
1. **Define in `docs/SCHEMA.md`**: Substance concept and hierarchy
2. **Add detection in physics/**: Implementation
3. **Configure in `pipeline_config.py`**: Activation
4. **Test with `verify_data.py`**: Visualization

### 3. "Should I bridge the two dataset formats?"
- **Current state**: New extraction doesn't produce old-format `y.npy`
- **This is intentional**: two permanently separate pipelines, not a gap
  (see README's "known gap" and `docs/PCA_GUIDE.md`)
- **Don't** build a bridge unless the user explicitly asks for one

### 4. "Physics vs ML boundary?"
- **Physics**: Detection, labeling, transformation
- **ML**: Analysis, clustering, prediction
- **Interface**: Data format, labels, features
- **Separation**: `data_proc` ≠ `ml_models` (no cross-import)

## Quick Reference

### For Physics Tasks
- **Key terms**: magnetopause, current_layer, rotation, hermite, substance
- **Key files**: `PHYSICS.md`, `pipeline_config.py`, `src/data_proc/physics/`
- **Output**: `cluster_phys`, `metadata.csv`, potentially transformed features

### For ML Tasks
- **Key terms**: PCA, clustering, features, training, prediction, `cluster_ml`
- **Key files**: `ML_MODELS.md`, `configs/*.yaml`, `src/ml_models/`
- **Output**: Models, predictions, discovered clusters, evaluations

### For Both Domains
- **Coordination**: Label access, feature compatibility
- **Skills**: Use `orchestrate` for auto-routing
- **Focus**: Interface between physics detection and ML analysis

### Repository Evolution
- **Current**: Snapshot PCA, new extraction format
- **Legacy**: Dataset PCA, old training format
- **Relationship**: Two permanently separate pipelines, not an in-progress
  transition -- no bridge is planned or needed
- **Deprecated**: Check `DEPRECATED.md` before implementing