# AI Agent Guide for vdf-ml

**Purpose**: Help AI agents (LLMs) work effectively with this physics/ML repository.

## Core Confusion Points for AI Agents

### 1. Domain Ambiguity
- **Physics tasks**: magnetosphere detection, current layers, rotation transforms, labeling
1. **ML tasks**: PCA clustering, feature engineering, model training, prediction
2. **Cross-domain tasks**: Both physics detection AND ML analysis

**AI Tip**: Use the `orchestrate` skill to route tasks correctly.

### 2. Two PCA Pipelines (Common Mistake)
- **Snapshot PCA**: `run_snapshot_pca.py` - analyzes single snapshot, finds `cluster_ml`
- **Dataset PCA**: Legacy pipeline - analyzes entire dataset
- **Don't confuse**: `cluster_phys` (ground truth) vs `cluster_ml` (discovered)

### 3. Substance Taxonomy
1. **Base substances**: `magnetosheath`, `solar_wind`, `inner_magnetosphere`, `lobes`, `no_density_data`, `undefined`
2. **Point substances**: `current_layer`, `x_point`, `o_point`, `x_point_o_point`
3. **Combined**: `cluster_phys` = base + active point substances

### 4. Two Dataset Formats (Intentional, Not a Gap)
- **New pipeline**: `X.npy` + `metadata.csv` (labels in metadata column)
- **Old tools**: Expect `X.npy`/`y.npy`/`metadata.csv` triple
- **This is deliberate**: the new snapshot-extraction pipeline and the old
  train/predict tools are two independent, permanently separate pipelines
  (see README's "known gap" and `docs/PCA_GUIDE.md`) — not an accidental
  incompatibility awaiting a bridge. Don't propose building one unless the
  user explicitly asks for it.

## Repository Structure

### Core Directories
```
src/data_proc/physics/    # Magnetopause, current layers, rotation
src/data_proc/labeling/   # Snapshot labeling, point labels  
src/ml_models/            # PCA, features, model training
configs/                  # YAML configs for ML training
scripts/                  # Main executable scripts
docs/                     # Function indices and documentation
```

### Key Configuration Files
1. `src/data_proc/pipeline_config.py` - Shared physics pipeline config
2. `configs/*.yaml` - ML training/prediction configs
3. `pipeline_config.py` settings control what runs

## AI Workflow Patterns

### When Starting a Task
1. **Determine domain**: Physics, ML, or both?
2. **Check DEPRECATED.md**: Avoid using removed/renamed functions
3. **Read relevant docs**: PHYSICS.md, ML_MODELS.md, DATA_PROC.md, LABELING.md
4. **Use correct skill**: `load_physicist_knowledge` or `load_ml_knowledge`

### Common Task Templates
1. **Extract and label data**: `scripts/data_proc/extract_data.py`
2. **Verify physics labels**: `scripts/data_proc/verify_data.py`
3. **Run PCA clustering**: `scripts/ml_models/run_snapshot_pca.py`
4. **Plot PCA results**: `scripts/ml_models/plot_snapshot_pca.py`
5. **Train ML model**: `scripts/ml_models/train_*.py`
6. **Predict with model**: `scripts/ml_models/predict_*.py`

## AI-Specific Guidelines

### Context Optimization
- **For physics tasks**: Read PHYSICS.md, pipeline_config.py, relevant physics files
- **For ML tasks**: Read ML_MODELS.md, feature files, training configs
- **For cross-domain**: Read both, focus on interface (data format, labels)

### Error Prevention
1. **Don't import across layers**: `data_proc` never imports `ml_models`
2. **Don't assume the two dataset formats need bridging**: that split is
   intentional (see above), not a bug
3. **Verify skill loading**: Use orchestrate skill for domain routing
4. **Consult DEPRECATED.md**: Before reimplementing functionality

### Token Efficiency
- **Skip comprehensive docstrings**: This repo uses minimal docstrings (docs/PIPELINE.md rule)
- **Focus on physical WHY**: Not code restatement
- **Use function indices**: docs/*.md files catalog available functions

## Development Rules (docs/PIPELINE.md)
1. **No redundant functions**: Grep before implementing
2. **One shared config per pipeline**: Don't create drift
3. **Layer separation**: `data_proc` ≠ `ml_models`
4. **Multi-user aware**: Check git status before assuming state

## Quick Start for AI Agents

### Physics Task
```
1. Load load_physicist_knowledge skill
2. Read PHYSICS.md and pipeline_config.py
3. Check DEPRECATED.md for removed functions
4. Implement using src/data_proc/physics/ tools
```

### ML Task  
```
1. Load load_ml_knowledge skill
2. Read ML_MODELS.md and configs/*.yaml
3. Confirm whether the task needs the current snapshot pipeline or the
   legacy train/predict pipeline -- they're separate, not interchangeable
4. Implement using src/ml_models/ tools
```

### Cross-Domain Task
```
1. Load orchestrate skill (routes automatically)
2. Read both PHYSICS.md and ML_MODELS.md
3. Focus on interface: labels, data format, features
4. Coordinate physics detection with ML analysis
```

## Cleanup Candidates Identified

### Potential Redundancy
1. **Multiple batch iteration implementations** - Check `batches.py` consolidation
2. **Deprecated functions in src/deprecated.py** - Verify nothing uses them
3. **Config duplication risk** - Ensure pipeline_config.py is single source

Not a cleanup candidate: the old vs. new dataset format split is intentional
(see above) — don't queue up "unify the formats" as work.

### AI Confusion Sources
1. **Two PCA pipelines** - Clear documentation of purpose (see `docs/PCA_GUIDE.md`)
2. **Substance taxonomy complexity** - Simplified explanation (see `docs/SCHEMA.md`)
3. **Domain boundary ambiguity** - Clearer task routing (see the `orchestrate` skill)

## Next Steps for AI Agents
1. **Always route through orchestrate skill** for domain classification
2. **Check DEPRECATED.md first** before implementing
3. **Respect layer separation** (data_proc vs ml_models)
4. **Optimize for physicist readability** over code abstraction