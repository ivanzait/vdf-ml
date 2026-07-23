# Decision Routing for AI Agents

**Purpose**: Help AI agents determine if a task is physics, ML, or both.

## Quick Decision Tree

```
Is the task about: 
├── Magnetosphere detection? → PHYSICS
├── Current layers? → PHYSICS  
├── Rotation transforms? → PHYSICS
├── VDF labeling? → PHYSICS
├── Substance classification? → PHYSICS
├── PCA/clustering? → ML
├── Feature engineering? → ML
├── Model training? → ML
├── Prediction/evaluation? → ML
├── Data extraction + ML? → BOTH
├── Label verification + PCA? → BOTH
└── Feature analysis of physics data? → BOTH
```

## Detailed Routing Guide

### PHYSICS Tasks (use `load_physicist_knowledge`)
- **Magnetopause detection**: Shue model, boundary detection
- **Current layer identification**: Peak current density cores
- **Rotation transforms**: VDF rotation into local frame
- **Hermite spectra**: Log-space Hermite transformation
- **Point detection**: X/O critical points via Hessian
- **Substance labeling**: Base + point substance classification
- **Data extraction**: From `.vlsv` files with physics labels
- **Verification plots**: Physics label validation

**Key files**: `src/data_proc/physics/`, `pipeline_config.py`, `PHYSICS.md`

### ML Tasks (use `load_ml_knowledge`)
- **PCA analysis**: Dimensionality reduction, clustering
- **Feature engineering**: Raw vs Hermite representations
- **Model training**: Logistic regression, perceptron, MLP, CNN
- **Model prediction**: Using trained classifiers
- **Feature caching**: Optimization of feature computation
- **Dataset PCA**: Legacy pipeline (whole dataset)
- **Snapshot PCA**: Current pipeline (single snapshot)
- **Model evaluation**: Performance metrics, confusion matrices

**Key files**: `src/ml_models/`, `configs/*.yaml`, `ML_MODELS.md`

### BOTH Tasks (use `orchestrate` skill)
- **Extract data for ML training**: Physics detection → ML dataset
- **Verify physics labels with PCA**: Compare `cluster_phys` vs `cluster_ml`
- **Analyze feature representations**: Physics transforms for ML features
- **End-to-end workflows**: Extraction → labeling → PCA → training
- **Substance impact on ML**: How physics labels affect model performance

Note: the new snapshot pipeline and the legacy train/predict pipeline use
different dataset formats *on purpose* (two independent, permanently
separate pipelines -- see `docs/PCA_GUIDE.md`), not a gap to bridge.

**Key files**: Both domains, focus on interface: data format, labels, features

## Common Ambiguous Cases

### Case 1: "Run PCA on VDF data"
**Clarify**:
- **Snapshot PCA** (single file) → ML (but uses physics labels for comparison)
- **Dataset PCA** (whole dataset) → ML
- **Purpose**: Discovering `cluster_ml` vs comparing to `cluster_phys`
- **Decision**: ML task, but understand physics labels for comparison

### Case 2: "Analyze feature representations"
**Clarify**:
- **Raw VDF features** → ML (but data comes from physics extraction)
- **Hermite features** → ML (requires physics rotation transform)
- **Rotation transforms** → Physics (prerequisite for Hermite)
- **Decision**: BOTH - coordinate physics prerequisite with ML analysis

### Case 3: "Improve labeling accuracy"
**Clarify**:
- **Physics detection improvement** → PHYSICS (better magnetopause model)
- **ML classifier improvement** → ML (better model training)
- **Decision**: Specify which aspect - detection or classification

### Case 4: "Create dataset for training"
**Clarify**:
- **Extract from `.vlsv`** → PHYSICS (detection, labeling)
- **Which training pipeline?** → the legacy train/predict tools need the
  old-format dataset, not the new snapshot pipeline's output -- confirm
  which one before assuming a conversion step is needed
- **Decision**: BOTH - physics extraction with ML format considerations

## AI Routing Checklist

### Before Starting Work
1. **What is the core domain?** Physics detection or ML analysis?
2. **What are the inputs/outputs?** `.vlsv` files or existing datasets?
3. **What is the purpose?** Discovery (`cluster_ml`) or ground truth (`cluster_phys`)?
4. **Which pipeline?** The new snapshot pipeline and the legacy train/predict
   pipeline are separate -- confirm which one the task means.

### Skill Selection
```
IF task contains:
  "magnetopause", "current_layer", "rotation", "hermite", 
  "Shue", "detection", "extract", "label", "verify"
THEN load_physicist_knowledge

IF task contains:
  "PCA", "clustering", "feature", "train", "predict", 
  "model", "classifier", "KMeans", "silhouette"
THEN load_ml_knowledge

IF task contains BOTH domains OR is ambiguous
THEN use orchestrate skill (auto-routes)
```

### File Priority by Domain

**Physics-first reading order**:
1. `PHYSICS.md`
2. `pipeline_config.py`
3. `src/data_proc/physics/` relevant files
4. `LABELING.md` (if labeling involved)
5. `DATA_PROC.md` (if data processing involved)

**ML-first reading order**:
1. `ML_MODELS.md`
2. Relevant `configs/*.yaml`
3. `src/ml_models/` relevant files
4. Confirm which pipeline (snapshot vs. legacy train/predict) the task needs
5. `feature_cache.py` (if performance optimization needed)

**Both domains reading order**:
1. `orchestrate` skill decision
2. Both `PHYSICS.md` and `ML_MODELS.md`
3. Interface: `pipeline_config.py` and data format

## Error Prevention in Routing

### Common Routing Mistakes
1. **Treating PCA as physics**: PCA discovers `cluster_ml`, not `cluster_phys`
2. **Assuming a format bridge exists or should be built**: the two dataset
   formats are intentionally separate (see above)
3. **Cross-layer import**: `data_proc` importing `ml_models` (violates rule)
4. **Redundant implementation**: Not checking `DEPRECATED.md` first
5. **Skill mismatch**: Physics task with ML knowledge or vice versa

### Verification Steps
1. **Check DEPRECATED.md**: Before implementing functionality
2. **Verify layer separation**: No cross-import between `data_proc` and `ml_models`
3. **Confirm which pipeline**: Snapshot pipeline or legacy train/predict?
4. **Consult docs/PIPELINE.md**: Follow repository development rules
5. **Test skill loading**: Use correct domain knowledge skill

## Examples

### Example 1: "Add new substance detection"
- **Domain**: PHYSICS
- **Skill**: `load_physicist_knowledge`
- **Key files**: `docs/SCHEMA.md`, `pipeline_config.py`, `src/data_proc/physics/`
- **Check**: `DEPRECATED.md` for existing detection patterns

### Example 2: "Train CNN on Hermite features"
- **Domain**: BOTH
- **Skill**: `orchestrate` (auto-routes to both)
- **Key files**: Physics: rotation transforms; ML: training configs, CNN implementation
- **Check**: `train_cnn.py` currently targets the legacy dataset format, not
  Hermite features from the new pipeline -- confirm scope before assuming
  this already works end to end

### Example 3: "Plot PCA results colored by physics labels"
- **Domain**: BOTH
- **Skill**: `orchestrate`
- **Key files**: ML: PCA code; Physics: label data for coloring
- **Check**: Label access in the PCA workflow (`metadata.csv`'s `label` column)

### Example 4: "Optimize batch iteration for large datasets"
- **Domain**: NEUTRAL (infrastructure)
- **Skill**: Check current implementation in `batches.py`
- **Key files**: `batches.py`, `vdf_tools.py`, `config.py`
- **Check**: `DEPRECATED.md` for consolidation history