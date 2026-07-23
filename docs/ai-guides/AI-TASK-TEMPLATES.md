# AI Task Templates for vdf-ml

**Purpose**: Common operation patterns with examples for AI agents.

## Template 1: Extract and Label Data (Physics)

### When to use
- Starting new analysis from `.vlsv` files
- Need physics-labeled dataset
- Want to verify detection visually

### Steps
```
1. Configure pipeline_config.py:
   - Set search bounds, snapshot file
   - Choose active_point_substances
   - Set BUILD_ROTATED_DATASET / BUILD_HERMITE_DATASET

2. Run extraction:
   python scripts/data_proc/extract_data.py

3. Verify labels:
   python scripts/data_proc/verify_data.py
```

### AI Considerations
- **Output format**: `X.npy` + `metadata.csv` (no `y.npy`)
- **Visual verification**: `verify_data.py` shows labels on plot
- **Feature options**: Raw, rotated, or Hermite features
- **Skill**: `load_physicist_knowledge`

### Example Task
"Extract VDFs from snapshot X with current_layer detection and save Hermite features"

## Template 2: Run Snapshot PCA (ML with Physics Context)

### When to use
- Want to discover `cluster_ml` from physics data
- Compare statistical clustering to ground truth
- Sanity check physics structure recoverability

### Steps
```
1. Ensure extraction done (Template 1)
2. Configure PCA in pipeline_config.py:
   - PCA_CONFIG["feature_representation"]
   - PCA_CONFIG["n_components"]
   - PCA_CONFIG["k_range"]

3. Run PCA:
   python scripts/ml_models/run_snapshot_pca.py

4. Plot results:
   python scripts/ml_models/plot_snapshot_pca.py
```

### AI Considerations
- **Comparison**: `cluster_ml` (discovered) vs `cluster_phys` (ground truth)
- **Feature choice**: Raw, rotated, or Hermite (configurable)
- **Auto-k**: KMeans with silhouette score optimization
- **Skill**: `load_ml_knowledge` (but understand physics labels)

### Example Task  
"Run PCA on Hermite features from latest extraction and plot comparison to physics labels"

## Template 3: Train ML Model (Pure ML)

### When to use
- Train classifier on existing dataset
- Need prediction capability
- Compare model performance

### Steps
```
1. Ensure dataset is in the old format (X.npy, y.npy, metadata.csv)
   - The new snapshot pipeline (extract_data.py) does not produce this --
     that's by design, not a gap. Use an existing old-format dataset, or
     build one via src/deprecated.py's create_dataset (see README's
     "Legacy path").

2. Choose config from configs/:
   - train_logistic_regression.yaml
   - train_perceptron.yaml  
   - train_multilayer_perceptron_classifier.yaml
   - train_pytorch_convolutional_neural_network_classifier.yaml

3. Train model:
   python scripts/ml_models/train_*.py --config path/to/config.yaml

4. (Optional) Predict with model:
   python scripts/ml_models/predict_*.py --config path/to/config.yaml
```

### AI Considerations
- **Format limitation**: Only works with old-format datasets -- this is a
  separate pipeline from the new snapshot extraction, not a temporary gap
- **Config-driven**: YAML files control training parameters
- **Model variety**: Logistic regression, perceptron, MLP, CNN
- **Skill**: `load_ml_knowledge`

### Example Task
"Train CNN classifier on existing old-format dataset using configs/train_pytorch_convolutional_neural_network_classifier.yaml"

## Template 4: Cross-Domain Analysis (Physics + ML)

### When to use
- End-to-end workflow comparing physics detection to blind ML clustering
- Feature analysis across representations (raw/rotated/Hermite)
- The actual working cross-domain path today -- not CNN training, which is
  a separate, not-yet-connected pipeline (see Template 3)

### Steps
```
1. Extract and label (Template 1), with BUILD_ROTATED_DATASET/
   BUILD_HERMITE_DATASET on if the feature representation needs them
2. Run snapshot PCA (Template 2) -- compares cluster_ml (discovered) to
   cluster_phys (ground truth) on the same extracted data
3. Analyze physics-ML relationship via plot_snapshot_pca.py's outputs
```

### AI Considerations
- **No format conversion needed**: this workflow stays entirely within the
  new snapshot pipeline (extract_data.py → run_snapshot_pca.py)
- **Coordination**: Physics detection quality affects ML results
- **Feature impact**: Different representations (raw/rotated/Hermite) affect ML
- **Skill**: `orchestrate` (auto-routes to both domains)
- **Out of scope for this template**: training a classifier on physics-
  detected substances -- that needs the legacy old-format pipeline
  (Template 3), which isn't wired to this one

### Example Task
"Run PCA on Hermite features from a current_layer extraction and check whether cluster_ml recovers cluster_phys"

## Template 5: Add New Substance Detection (Physics Extension)

### When to use
- Extending physics detection capabilities
- Adding new point substance or refining existing
- Following repository evolution pattern

### Steps
```
1. Define the substance in docs/SCHEMA.md:
   - name, existence predicate (see docs/SCHEMA.md's own convention)

2. Implement a find_<name>_cellids(reader, points_config, regions_re=None)
   function in src/data_proc/physics/ + src/data_proc/labeling/, following
   the existing pattern (find_current_layer_cellids is the reference
   example) -- returns a cellid set, no central registry/dispatcher

3. Add a literal `if "<name>" in active_point_substances:` toggle block at
   each call site (extract_data.py and compute_snapshot_ground_truth), and
   add "<name>" to pipeline_config.py's active_point_substances list

4. Test with extract_data.py + verify_data.py
```

### AI Considerations
- **Schema compliance**: Follow `docs/SCHEMA.md`'s substance concept
- **Pattern reuse**: Mirror `find_current_layer_cellids`, not a new abstraction
- **No dispatcher**: Each call site keeps its own literal toggle block, by
  design (see docs/SCHEMA.md) -- don't introduce a registry
- **Visual verification**: Test with `verify_data.py`
- **Skill**: `load_physicist_knowledge`

### Example Task
"Add detection for bow shock precursor signatures as new point substance"

## Template 6: Code Cleanup / Consolidation

### When to use
- Suspect redundant functionality
- Following docs/PIPELINE.md "no redundant functions" rule
- Repository maintenance

### Steps
```
1. Check DEPRECATED.md first:
   - See what's already removed/consolidated

2. Search for similar functionality:
   grep -rn "def .*<verb>" src/
   grep -rn "operation_pattern" src/

3. Check common consolidation locations:
   - vdf_tools.py (general VDF operations)
   - batches.py (iteration patterns)
   - config.py (configuration helpers)

4. If found duplicate:
   - Consolidate to existing implementation
   - Update all callers
   - Add to DEPRECATED.md if removing

5. If new but similar:
   - Consider extending existing vs new
   - Follow "physicists' code" principle
```

### AI Considerations
- **Repository history**: Check `DEPRECATED.md` for past consolidations
- **Common patterns**: Batch iteration, configuration, VDF operations
- **Multi-user**: Changes affect other users/agents
- **Rule compliance**: `docs/PIPELINE.md` "no redundant functions"
- **Skill**: Neutral (infrastructure)

### Example Task
"Check for duplicate batch iteration implementations and consolidate to batches.py"

## Template 7: Verification and Validation

### When to use
- Ensuring physics detection correctness
- Validating ML results against physics ground truth
- Repository quality maintenance

### Steps
```
Physics verification:
1. extract_data.py (auto-runs verification plot)
2. verify_data.py (standalone verification)
3. plot_nulls.py (X/O point verification)
4. plot_vdf_hermite.py (Hermite transform verification)

ML verification:
1. plot_snapshot_pca.py (PCA results visualization)
2. plot_dataset_sample.py (dataset sampling visualization)
3. plot_dataset_pca.py (legacy PCA visualization)
```

### AI Considerations
- **Integrated verification**: `extract_data.py` auto-plots verification
- **Standalone tools**: For focused verification tasks
- **Visual focus**: Physicists' code → plots over metrics
- **Skill**: Domain-specific or `orchestrate` for cross-domain

### Example Task
"Verify current_layer detection matches peak current density cores visually"

## Common Pitfalls and Solutions

### Pitfall 1: Wrong data format for ML tools
- **Symptom**: Training fails because no `y.npy`
- **Cause**: New extraction format vs. old tool expectation -- these are
  two separate pipelines on purpose
- **Solution**: Use an existing old-format dataset (or build one via
  `src/deprecated.py`'s `create_dataset`). Don't build a format bridge
  unless the user explicitly asks for one.

### Pitfall 2: Cross-layer import violation
- **Symptom**: `data_proc` importing `ml_models`
- **Cause**: Not respecting layer separation
- **Solution**: Refactor to keep layers separate, use interface patterns

### Pitfall 3: Redundant implementation
- **Symptom**: New function similar to existing
- **Cause**: Not checking `DEPRECATED.md` or grepping first
- **Solution**: Consolidate, update callers, document in `DEPRECATED.md`

### Pitfall 4: Configuration drift
- **Symptom**: Different scripts using different parameters
- **Cause**: Not using shared `pipeline_config.py`
- **Solution**: Centralize configuration, update all scripts

### Pitfall 5: Skill mismatch
- **Symptom**: Physics task with ML knowledge or vice versa
- **Cause**: Not using `orchestrate` for routing
- **Solution**: Use `orchestrate` skill for domain classification