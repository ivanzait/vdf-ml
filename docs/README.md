# Function-index docs

One file per area, each a per-function name + one-line description + file
path, so a new session can find what already exists without reading the
whole tree.

- [`PHYSICS.md`](PHYSICS.md) — `src/data_proc/physics/`: X/O critical-point detection, Shue magnetopause model, VDF rotation + Hermite transform.
- [`DATA_PROC.md`](DATA_PROC.md) — `src/data_proc/`: VLSV I/O, region/box masks, dataset I/O, plotting.
- [`LABELING.md`](LABELING.md) — `src/data_proc/labeling/`: the current (physics-driven) ground-truth/label-assignment subsystem.
- [`ML_MODELS.md`](ML_MODELS.md) — `src/ml_models/`: training, prediction, model I/O, the PCA diagnostic tool.
- [`DEPRECATED.md`](DEPRECATED.md) — `src/deprecated.py`: the old label-scheme pipeline, kept working but not to be extended.

**Keep these in sync.** When you add, rename, or remove a function in one of
these areas, update the corresponding `.md` file in the same change — that's
the entire point of this index.

## Other reference docs

Not function indexes — concept/rules/usage guides, kept alongside the
function indices for the same reason (a new session shouldn't have to
re-derive them from code or from prior conversations):

- [`SCHEMA.md`](SCHEMA.md) — defines "substance" (the `cluster_phys`
  category concept: a name + an existence predicate), documents the base
  region substances and point substances, and the convention for adding a
  new one (a `find_<name>_cellids` function + a toggle block per call site,
  no central registry).
- [`PCA_GUIDE.md`](PCA_GUIDE.md) — usage guide for the two independent PCA
  pipelines (current snapshot-clustering vs. legacy CNN-dataset): feature
  representations, per-sample normalization, moment-feature weighting, and
  the debugging history behind each of those choices.
- [`PIPELINE.md`](PIPELINE.md) — repository-wide development rules (layer
  separation, no redundant functions, docstring style, verification).
- [`TESTING.md`](TESTING.md) — how to validate a change to `src/data_proc/`/
  `src/ml_models/`: per-stage manual verification against a local fixture,
  a processing-time cheat sheet, and a table of failure modes caught during
  development.
