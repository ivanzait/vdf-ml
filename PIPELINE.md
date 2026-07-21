# PIPELINE.md — rules for developing this codebase

## Two ground rules, before anything else

1. **This is physicists' code.** Readers and maintainers are domain
   scientists first, software engineers second. Optimize for physical
   correctness and readability by someone reasoning about VDFs, fields, and
   critical points — not for generic software abstraction. A docstring
   should explain a physical or numerical WHY (a subtlety, a convention, a
   known gotcha), never restate what a well-named function already shows.
   Don't build an abstraction the physics doesn't need yet.

2. **This is multi-user code.** More than one person (or agent) works on
   this repo concurrently, in separate sessions against the same tree.
   Never assume exclusive ownership of a file or a moment in time: check
   `git log -3` / `git status` before trusting your own memory of the
   layout, re-grep for stale references after any rename/merge/delete even
   when you're confident you tracked every caller, and prefer small,
   independently-committable changes over sweeping ones that are hard for
   someone else to review or reconcile mid-flight. If a file looks
   unexpectedly different from what you last wrote, don't assume it's
   wrong — diff it against what you expected and ask before overwriting.

## Rules

- **No redundant functions.** `iter_index_batches` alone was independently
  reimplemented four times across this codebase before being consolidated
  into `batches.py`. Before writing a new helper, grep for the operation
  you're about to implement (`grep -rn "def .*<verb>" src/`) — there is
  likely already one in `vdf_tools.py`, `batches.py`, or `config.py`.
- **One shared config per pipeline.** Scripts that operate on the same
  run/dataset import one shared config, never keep their own
  independently-edited parameter block — that's how settings silently
  drift between pipeline steps. Concretely: `extract_data.py`,
  `verify_data.py`, `run_snapshot_pca.py`, and `plot_snapshot_pca.py` all
  share `src/data_proc/pipeline_config.py` (plain Python, not YAML) so
  none of them can disagree about what one run means.
  `train_*.py`/`predict_*.py` are a separate pipeline and keep their own
  `--config path/to/x.yaml` style. Standalone ad-hoc tools
  (`plot_vdf_hermite.py`, `plot_nulls.py`) deliberately keep independent
  parameters, since they're for tuning/exploration, not reproducing a
  specific run. Don't convert one style into another without discussing it
  first.
- **`data_proc` never imports `ml_models`.** Verify with
  `grep -rn "from src.ml_models" src/data_proc/` — it should return
  nothing.
- **Docstrings are one line unless there's a non-obvious WHY.** Most
  functions here need no more than
  `"""Middle xz slice of a 3D VDF array."""`. Multi-line docstrings are
  reserved for things a reader would otherwise get wrong — see
  `build_rotation_matrix`/`get_rotated_vdf` in `physics/vdf_transform.py`
  for the bar to clear (a real numerical subtlety, not a restated
  parameter list).
- **Verify a refactor by running it, not just importing it.** `py_compile`
  catches syntax errors, not wiring mistakes. Re-run the actual pipeline
  step against the local fixture (see `TESTING.md`) and diff its output
  against a pre-change baseline.
- **Keep `docs/*.md` in sync.** Every function-name index in `docs/` exists
  so nobody has to re-read the whole tree to know what's there — update the
  relevant file whenever you add, rename, or remove a function.
- **New `cluster_phys` categories follow [`schema.md`](schema.md).** It
  defines what a substance is (name + existence predicate) and the exact
  steps for adding one — read it before adding or changing a label.

## Two verification tiers — local vs. orchestrator

Repo-wide `py_compile` + `pyflakes` + a full pipeline re-run on every small
edit is expensive and noisy for what it's checking. Match the check to the
size of the change:

- **Local (default for a scoped edit — a doc tweak, one function, one
  script's parameters):** `py_compile` on just the file(s) you touched, plus
  running the one script actually affected, if there's an obvious cheap way
  to. Nothing repo-wide. This is the common case — most edits in one
  session are this size.
- **Orchestrator (global, reserved for finishing a multi-step task or
  before a nontrivial commit):** the full sweep —
  `py_compile $(find src scripts -name "*.py" -not -path "*/__pycache__/*")`,
  `pyflakes src/ scripts/`, and an actual end-to-end run of every pipeline
  stage the task touched, diffed against a pre-change baseline (see
  `TESTING.md`). This is the one pass that has to certify the whole body of
  work hangs together — run it once at the end, not after every
  intermediate step along the way.

Don't run the orchestrator pass reflexively after each small edit; don't
skip it entirely when a task is actually done. If it's unclear which tier a
change needs, ask rather than defaulting to the expensive one.
