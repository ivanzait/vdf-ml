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
