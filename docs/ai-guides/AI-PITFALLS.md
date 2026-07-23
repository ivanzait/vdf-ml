# AI Pitfalls and Prevention Guide

**Purpose**: Common mistakes AI agents make with vdf-ml and how to avoid them.

## Critical Pitfalls

### Pitfall 1: Conflating `cluster_phys` and `cluster_ml`

**Mistake**: Treating them as interchangeable or assuming one derives from the other.

**Reality**:
- `cluster_phys`: Physics ground truth from detection algorithms
- `cluster_ml`: Statistical discovery from unsupervised ML
- **Purpose**: Compare to see if physics structure is statistically recoverable

**Prevention**:
- Always clarify which "cluster" is referenced
- For PCA tasks: output is `cluster_ml` (discovered)
- For physics tasks: output is `cluster_phys` (detected)
- When comparing: explicitly state comparison purpose

### Pitfall 2: Assuming the Two Dataset Formats Should Be Bridged

**Mistake**: Assuming new extraction works with old training tools, or
proposing to build a converter between them.

**Reality**:
- New pipeline: `X.npy` + `metadata.csv` (labels in metadata column)
- Old tools: Expect `X.npy`/`y.npy`/`metadata.csv` triple
- **Intentional split**: two independent, permanently separate pipelines
  (see README's "known gap" and `docs/PCA_GUIDE.md`), not a bug

**Prevention**:
- For training tasks: verify dataset has `y.npy` file
- For extraction tasks: note this output format is by design, not a limitation
- When designing workflows: don't assume a conversion step exists or should
  be built
- Use an existing old-format dataset for training, or build one via
  `src/deprecated.py`'s `create_dataset`

### Pitfall 3: Cross-Layer Import Violation

**Mistake**: `data_proc` importing from `ml_models` or vice versa.

**Reality**:
- **Rule**: `data_proc` never imports `ml_models` (`docs/PIPELINE.md`)
- **Enforcement**: `grep -rn "from src.ml_models" src/data_proc/` should return nothing
- **Architecture**: Deliberate separation of physics and ML concerns

**Prevention**:
- Check imports before adding new ones
- If coordination needed: use interface patterns (data files, configs)
- Respect the layer separation design decision
- Verify with grep check after changes

### Pitfall 4: Redundant Implementation

**Mistake**: Creating new function when similar exists.

**Reality**:
- **Rule**: No redundant functions (`docs/PIPELINE.md`)
- **History**: `iter_index_batches` consolidated from 4 implementations
- **Common locations**: `vdf_tools.py`, `batches.py`, `config.py`

**Prevention**:
1. **Always grep first**: `grep -rn "def .*<verb>" src/`
2. **Check DEPRECATED.md**: For removed/consolidated functions
3. **Search common files**: Look in consolidation locations
4. **If duplicate found**: Consolidate, update callers

### Pitfall 5: Configuration Drift

**Mistake**: Different scripts using different parameter values.

**Reality**:
- **Physics pipeline**: Single `pipeline_config.py` shared by all scripts
- **ML pipeline**: Separate `configs/*.yaml` system
- **Don't convert**: Between styles without discussion

**Prevention**:
- For physics scripts: always use `pipeline_config.py`
- For ML scripts: use YAML configs, not hardcoded parameters
- Don't create script-specific parameter blocks
- If changing config style: discuss impact first

### Pitfall 6: Skill Mismatch

**Mistake**: Physics task with ML knowledge or vice versa.

**Reality**:
- **Physics domain**: magnetosphere, current layers, rotation, detection
- **ML domain**: PCA, features, training, prediction, clustering
- **Cross-domain**: Need both, use `orchestrate` for routing

**Prevention**:
1. **Use `orchestrate` skill**: Auto-routes to correct domain
2. **Check task keywords**: Physics vs ML terminology
3. **Verify skill loading**: Confirm correct domain knowledge
4. **When unsure**: Start with `orchestrate`, let it route

### Pitfall 7: Missing Multi-User Awareness

**Mistake**: Assuming exclusive ownership of repository state.

**Reality**:
- **Rule**: This is multi-user code (`docs/PIPELINE.md`)
- **Multiple agents/users**: Work concurrently on same tree
- **Need to check**: git status before assuming file state

**Prevention**:
- **Before editing**: `git status`, `git log -3`
- **After changes**: Re-grep for stale references
- **If file looks different**: Diff before overwriting
- **Prefer**: Small, independently-committable changes

## Domain-Specific Pitfalls

### Physics Domain Pitfalls

**P1: Misunderstanding substance taxonomy**
- **Mistake**: Confusing base vs point substances, or their combination.
- **Prevention**:
  - Base substances: always computed, fixed priority order
  - Point substances: configurable via `pipeline_config.py`
  - `cluster_phys`: base + active point substances
  - Consult `docs/SCHEMA.md` for substance concept

**P2: Ignoring rotation prerequisite**
- **Mistake**: Assuming Hermite features without rotation.
- **Prevention**:
  - Hermite requires rotation into local frame
  - `BUILD_ROTATED_DATASET` controls saving rotated VDFs
  - `BUILD_HERMITE_DATASET` requires rotation (saves spectra)
  - Check `pipeline_config.py` settings

**P3: Overlooking verification integration**
- **Mistake**: Not using built-in verification.
- **Prevention**:
  - `extract_data.py` auto-plots verification
  - `verify_data.py` for standalone verification
  - Visual verification is key (physicists' code)
  - Always include verification in physics workflows

### ML Domain Pitfalls

**M1: Assuming new-format dataset compatibility**
- **Mistake**: Trying to train on newly extracted data, or assuming a
  conversion step exists.
- **Prevention**:
  - Check for `y.npy` file existence
  - This is a permanent pipeline split, not a limitation to fix
  - Use an existing old-format dataset for training
  - Don't propose implementing a format bridge unless explicitly asked

**M2: Confusing PCA pipelines**
- **Mistake**: Using legacy dataset PCA instead of snapshot PCA.
- **Prevention**:
  - Snapshot PCA: `run_snapshot_pca.py` (current production)
  - Dataset PCA: `src/ml_models/dataset_pca.py` (standalone diagnostic,
    not deprecated, just not wired into CNN training filtering)
  - Clarify which is needed for task
  - Default to snapshot PCA unless specified

**M3: Ignoring feature representation choice**
- **Mistake**: Not considering raw vs rotated vs Hermite features.
- **Prevention**:
  - Check `PCA_CONFIG["feature_representation"]`
  - Consider physics implications of feature choice
  - Raw: direct VDF, Rotated: local frame, Hermite: spectra
  - Choice affects ML results and interpretability

## Cross-Domain Pitfalls

**CD1: Assuming a format bridge should exist**
- **Mistake**: Designing workflows that assume the new pipeline feeds
  directly into the legacy train/predict tools.
- **Prevention**:
  - Recognize the split as intentional (see Pitfall 2 above)
  - Design cross-domain workflows within the snapshot pipeline instead
    (extraction → snapshot PCA, comparing `cluster_ml`/`cluster_phys`)
  - Use existing old-format datasets for anything that needs training
  - Note the split explicitly in workflow descriptions, don't paper over it

**CD2: Poor physics-ML interface design**
- **Mistake**: Tight coupling between detection and analysis.
- **Prevention**:
  - Use data files as interface, not direct calls
  - Respect layer separation (no cross-import)
  - Design clean interfaces: labels, features, formats
  - Follow existing patterns in codebase

**CD3: Missing end-to-end testing**
- **Mistake**: Not testing the full extraction → labelling → PCA workflow.
- **Prevention**:
  - Test extraction → labelling → snapshot PCA end to end
  - Verify physics labels propagate through to `cluster_phys` in PCA output
  - Check feature compatibility across steps
  - Include verification at each stage (see `docs/TESTING.md`)

## Prevention Checklist for AI Agents

### Before Starting Work
1. [ ] **Route correctly**: Use `orchestrate` skill for domain classification
2. [ ] **Check DEPRECATED.md**: Avoid removed/renamed functionality
3. [ ] **Confirm which pipeline**: Snapshot pipeline or legacy train/predict --
   don't assume they're interchangeable
4. [ ] **Grep for duplicates**: `grep -rn "def .*<verb>" src/`
5. [ ] **Check git status**: Multi-user awareness

### During Implementation
1. [ ] **Respect layer separation**: No `data_proc` → `ml_models` imports
2. [ ] **Use shared configs**: `pipeline_config.py` or YAML configs
3. [ ] **Follow substance taxonomy**: `docs/SCHEMA.md` definitions
4. [ ] **Include verification**: Visual verification for physics
5. [ ] **Don't build a format bridge**: unless the user explicitly asks for one

### After Completion
1. [ ] **Re-grep for references**: After renames/consolidations
2. [ ] **Update DEPRECATED.md**: If removing/consolidating functions
3. [ ] **Test cross-domain flows**: If affecting both physics and ML
4. [ ] **Verify skill integration**: Works with domain knowledge skills
5. [ ] **Check multi-user impact**: Small, committable changes

## Quick Reference: Pitfall → Prevention

| Pitfall | Prevention Action |
|---------|-------------------|
| `cluster_phys`/`cluster_ml` confusion | Always clarify which "cluster", state purpose |
| Assuming the two dataset formats need bridging | They're intentionally separate -- use an existing old-format dataset instead |
| Cross-layer import | grep check, use interface patterns |
| Redundant implementation | grep first, check DEPRECATED.md |
| Configuration drift | Use shared configs, don't create duplicates |
| Skill mismatch | Use `orchestrate` skill for routing |
| Multi-user unawareness | Check git status before assuming state |
| Substance taxonomy error | Consult `docs/SCHEMA.md`, understand base vs point |
| Missing rotation prerequisite | Check `pipeline_config.py` rotation/Hermite settings |
| PCA pipeline confusion | Default to snapshot PCA, clarify if legacy needed |

## Emergency Recovery

### If you made one of these mistakes:
1. **Stop immediately**: Don't compound the error
2. **Identify which pitfall**: Reference this guide
3. **Check current state**: `git status`, affected files
4. **Apply prevention**: Follow prevention steps for that pitfall
5. **Test correction**: Verify fix addresses the root cause
6. **Document learning**: Consider adding to this guide if new pitfall

### Most Common Recovery Scenarios:
1. **Wrong data format**: Switch to a compatible (old-format) dataset --
   don't implement a bridge
2. **Cross-layer import**: Refactor to use interface pattern (data files)
3. **Redundant function**: Consolidate with existing, update callers
4. **Skill mismatch**: Restart with correct skill or use `orchestrate`
5. **Configuration duplicate**: Move to shared config, update all users