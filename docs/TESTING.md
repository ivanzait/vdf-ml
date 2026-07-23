# Testing `src/data_proc/` and `src/ml_models/`

No automated test suite — the logic is physics-derived (critical-point
detection, magnetopause fitting, VDF geometry, spectral transforms) and
easier to validate by eye against a real snapshot than to unit-test against
synthetic fixtures. "Testing" means: run the pipeline stage against the
fixture below, and check the printed diagnostics / saved plots against each
section's "what good looks like" notes — so a fresh session doesn't have to
re-derive the checks, and regressions once caught by eye stay caught.

## Pipeline stages

Production use of this repo moves through six stages. The first four live
under `src/data_proc/`; PCA analysis lives under `src/ml_models/`; CNN
training/recognition is future work, not yet wired to this pipeline's
output.

| # | Stage | What it does | Driven by |
|---|---|---|---|
| 1 | **Data extraction** | Read a `.vlsv` snapshot, pull the raw VDF for each VDF-carrying cell | `vdf_tools.py` |
| 2 | **Processing** | Rotate each VDF into its local `(B, v_perp, B×v_perp)` frame, project onto a Hermite basis in log-space, compute lower-order physical moments | `physics/vdf_transform.py`, `labeling/snapshot_labeling.py` |
| 3 | **Labelling** | Assign each VDF cell a ground-truth substance: point substances (`x_o_points`/`current_layer`) + base region classification (Shue-model magnetopause/bow-shock) | `physics/point_topology.py`, `physics/current_layer.py`, `physics/magnetopause.py`, `labeling/snapshot_labeling.py` |
| 4 | **Verification** | Sanity-check labelling by eye: one representative VDF per label, mapped spatially and cut three ways | `scripts/data_proc/verify_data.py` |
| 5 | **PCA analysis** | Blind PCA+KMeans clustering on a saved dataset, scored against the physical labels | `src/ml_models/vdf_snapshot_clustering.py`, `scripts/ml_models/{run,plot}_snapshot_pca.py` |
| 6 | *(future)* **CNN training / cluster recognition** | Train a classifier to recognize clusters/substances directly from a VDF | not yet built against this pipeline — see the note at the end of this file |

Stages 1-3 are usually run together and saved in one shot by
`scripts/data_proc/extract_data.py` (see "Assembling the dataset" below) —
but each is independently testable without paying for the other two, which
is the key lever for keeping iteration fast (see the cheat sheet next).

### Processing-time cheat sheet

The expensive step in this whole pipeline is **rotation** (~3s/VDF on this
fixture's 268³ grid, ~6 minutes for all ~116 VDFs). Everything else —
extraction, labelling, Hermite transform, moment features, PCA/KMeans
fitting — is seconds or less. Before re-running anything, check whether a
cheaper option below already covers the change:

| Testing a change to... | Cheapest way to check it | Only re-run the full thing when... |
|---|---|---|
| Extraction (`vdf_tools.py`) | Stage 1's snippet below, on 1-2 cells | The extraction *loop*/CLI wiring in `extract_data.py` itself changed |
| Rotation math (`get_rotated_vdf`) | `plot_vdf_hermite.py` on an explicit 1-3 cell list (Stage 2) | Confirming rotation looks right across the *whole* sample set, not just a few cells |
| Hermite order / moment features, given rotation is unchanged | `rebuild_hermite_dataset.py` (reuses a saved `X_rotated.npy`, ~38s-1m18s vs. ~7min) | `X_rotated.npy` doesn't exist yet for this `RUN_ID` |
| Point substance / region classification logic | `plot_nulls.py` (x_o_points) or a tiny synthetic-array snippet (regions, Stage 3) — neither touches VDF extraction at all | Checking label counts against real fixture geometry (`verify_data.py`, next row) |
| Labelling as a whole, or any change to Stage 3 | `verify_data.py` — recomputes ground truth from the reader directly and only extracts a *handful* of representative VDFs (one per label), never all ~116, never rotation/Hermite | N/A — this already is the fast path; prefer it over `extract_data.py` for labelling-only changes |
| PCA config (`feature_representation`, `k_range`, weights, `sample_normalization`) | Re-run `run_snapshot_pca.py`/`plot_snapshot_pca.py` against the existing saved dataset — the PCA/KMeans fit itself is sub-second at ~116 samples | The feature array it needs (`X.npy`/`X_rotated.npy`/`X_hermite.npy`) doesn't exist yet for this `RUN_ID` |
| SOM config (`SOM_CONFIG`: map shape, iterations, node clusters) | Re-run `run_snapshot_som.py` against the existing `pca_results.npz` (~15s, no VLSV file opened) | `pca_results.npz` doesn't exist yet, or the PCA config changed (the SOMs are fit in that run's score space) |
| Anything, before calling a change done | Full `extract_data.py` run with `BUILD_ROTATED_DATASET`/`BUILD_HERMITE_DATASET` matching what downstream stages need, per PIPELINE.md's orchestrator tier (see "Regression checklist" below) | Always, as the final check — the cheap paths above are for iterating, not for signing off |

`extract_data.py` itself also has a cheap mode: leave
`BUILD_ROTATED_DATASET = False` and `BUILD_HERMITE_DATASET = False` when
the change under test doesn't touch rotation or Hermite — extraction +
labelling alone finishes in seconds, not minutes, on this fixture.

## Test fixture

One local Vlasiator timestep, used throughout development:

```
FILE_LOCATION      = "/Users/ivanzait/Downloads/bulk.0003408.vlsv"
FLUX_FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.bin"   # flux function, for X/O detection
```

Setup (venv, `PTNOLATEX`, `ANALYSATOR_PATH`) is in the README's **Setup**
section.

Sparse and coarse on purpose (~2880 VDF-carrying cells over the whole
domain, ~2.35 R_E spacing) — this exercises edge cases a dense production
run wouldn't (e.g. an X-point box needing to be *wider* than local VDF
spacing to catch anything). Don't extrapolate absolute cell counts to a real
run; only relative behavior transfers. Any other small `.vlsv` + matching
flux `.bin` works if this one becomes unavailable — numbers below are this
fixture's baseline, not universal constants.

## Stage 1 — Data extraction (VLSV reading + raw VDF pull)

File: `vdf_tools.py` (`VdfExtractor`, `get_vdf_cells_with_coords_re`,
region masks, plot-axis parameters).

```python
import analysator as pt
from src.data_proc.vdf_tools import VdfExtractor, get_vdf_cells_with_coords_re

reader = pt.vlsvfile.VlsvReader(FILE_LOCATION)
vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)
extractor = VdfExtractor(reader=reader)
vdf = extractor.extract(cid=int(vdf_cellids[0]))
```

This is the cheapest possible test of extraction logic — one cell, no
labelling, no rotation, no saving. Runs in well under a second.

**What good looks like:**
- `vdf_cellids`/`vdf_coords_re` come from the file's own `CELLSWITHBLOCKS`
  tag (ground truth, not a derived guess). On the fixture: 2880 cells, `y`
  all `0.0` (2D xz run), `x` roughly `[-94, 45]` R_E, `z` roughly
  `[-56, 54]` R_E.
- `vdf.shape` is `(4*nx, 4*ny, 4*nz)` in axis order `(vx, vy, vz)` — if you
  ever touch `VdfExtractor.extract`'s `swapaxes` call, re-verify this order,
  since a silent axis swap corrupts every downstream slice without raising.
  `vdf.sum() > 0` for any real VDF-carrying cell.
- `vdf.max()`'s index should land away from every array edge for a
  well-formed distribution — an edge index usually means the mesh size
  (`reader.get_velocity_mesh_size`) was read wrong, or the cell doesn't
  actually carry a VDF (check `cid` is in `vdf_cellids`).

## Stage 2 — Processing (rotation, Hermite transform, moment features)

Files: `physics/vdf_transform.py` (`get_rotated_vdf`,
`vdf_to_hermite_spectra_log`, `compute_density`,
`compute_thermal_velocity_components`), `labeling/snapshot_labeling.py`
(`rotate_vdfs_to_b_frame`, `compute_hermite_spectra_batch`,
`compute_moment_features_batch`).

This stage only matters for the `"rotated"`/`"hermite"` PCA feature
representations (Stage 5) — skip it entirely if a change only touches
extraction or labelling.

### Fast test: rotation + Hermite on a handful of cells

`scripts/data_proc/plot_vdf_hermite.py` — edit its `PARAMETERS` block to
select 1-3 cells (by spatial box or explicit coordinates) and run it. Draws
a 2D colormap with the selected cells marked, plus one row per cell of the
production representation: raw VDF | rotated into the local B frame |
log-space Hermite spectra (`plot_tools.plot_vdf_rotation_hermite_grid`,
the same drawing `plot_vdf_rotation_hermite.py` uses for a label's
representative) — pays rotation's ~3s/VDF cost for only the cells you
asked for, not the whole fixture.

```
python scripts/data_proc/plot_vdf_hermite.py
```

**What good looks like:**
- Raw vs. rotated panels: the rotated VDF's bulk-flow direction should
  align with the marked `v_parallel` axis — a rotation that doesn't
  visibly straighten the flow means `build_rotation_matrix`/
  `get_rotated_vdf` regressed. The rotated panel's title prints the
  density change from rotation — near 0% is healthy; a large shift
  signals a rotation/interpolation bug.
- The Hermite-spectrum panel should be smooth/decaying with increasing
  order for a near-Maxwellian population (`solar_wind`/`lobes`), and
  visibly more structured (checkerboard-like) for `current_layer`/
  `magnetosheath` — see the two-tier amplitude note in the failure-mode
  table below; a *flat* or *noisy-at-every-order* spectrum for any
  population usually means the log-space patch broke (see below).

### Fast test: retuning `HERMITE_ORDER` or moment features

`scripts/ml_models/rebuild_hermite_dataset.py` — reuses an already-saved
`X_rotated.npy` for `pipeline_config.RUN_ID` and rebuilds `X_hermite.npy`
at the current `HERMITE_ORDER`, plus recomputes the moment-feature columns
in `metadata.csv`. Needs one prior `extract_data.py` run with
`BUILD_ROTATED_DATASET = True` for this `RUN_ID`; raises a clear
`FileNotFoundError` otherwise instead of silently re-rotating.

```
python scripts/ml_models/rebuild_hermite_dataset.py
```

**What good looks like:**
- Runtime is seconds to ~1-2 minutes on the fixture (~116 samples),
  *not* minutes-per-sample — if it's taking rotation-scale time, it's
  accidentally re-rotating instead of reading `X_rotated.npy`.
- `X_hermite.npy`'s shape is `(n_samples, order, order, order)` matching
  the just-edited `HERMITE_ORDER`.
- Never re-run `extract_data.py` (which would re-pay the ~6-minute
  rotation cost) just to retune `HERMITE_ORDER` or the moment features —
  this script exists specifically so that isn't necessary.

### Correctness check: the compact-support / log-space patch

The single most important invariant in this stage: background (below-
`sparsity_threshold`) cells must map to exactly `0` in the quantity
projected onto the Hermite basis, not a large floor constant — see the
"log-floor background-integration bug" entry in the failure-mode table
below for what breaks if this regresses. `vdf_to_hermite_spectra_log`
computes `log10(vdf/sparsity_threshold)` (zero for background), never
`log10(vdf)` floored at `log10(sparsity_threshold)` (a huge, sample-
dependent constant integrated over the ~99.85%-empty grid).

Visual check: `plot_cluster_hermite_spectra` (called from
`plot_snapshot_pca.py`, Stage 5, when `feature_representation == "hermite"`)
plots the literal *flattened* feature vector PCA consumes, reduced to a 2D
`(parallel order, perp order)` heatmap, one panel per PCA-cluster and per
physical-label representative. A spectrum that's enormous in magnitude
(~1e9-1e10) relative to the others is the background-integration bug, not
real physics; a spectrum that's ~6-9x larger than others but the *same
qualitative shape* (checkerboard decay) is real — `current_layer`/
`magnetosheath` are genuinely more structured/non-Maxwellian than
`solar_wind`/`lobes`.

## Stage 3 — Labelling (point substances + region classification)

Both point substances (see [`SCHEMA.md`](SCHEMA.md)) and base-region
classification feed into one label per VDF cell
(`labeling.snapshot_labeling.combine_ground_truth_labels`). Which point
substance(s) run is a toggle, `POINTS_CONFIG["active_point_substances"]`
in `pipeline_config.py` — not mutually exclusive, both are independently
testable. None of this stage touches VDF arrays directly (it only reads
`B`, density, and cell coordinates) — it's fast (seconds) even at full
fixture scale, and is the cheapest stage to iterate on.

### Point substances: `x_o_points`

Files: `physics/point_topology.py` (`find_point_records`),
`labeling/point_labels.py` (box/circle selection).

Primary tool: `scripts/data_proc/plot_nulls.py` — edit its `PARAMETERS`
block (`SHOW_X_POINTS`/`SHOW_O_POINTS`, `PLOT_BOXRE`, `REGIONS_RE`,
`POINTS_CONFIG`) and run it. Draws detected X/O points, their search
box/circle, and matched-cell counts on a density colormap in one shot.
Independent of `active_point_substances` — always exercises the detector
directly, and never extracts a single VDF (fastest available test for this
detector).

```
python scripts/data_proc/plot_nulls.py
```

**What good looks like:**
- `X points: N detected, M matched a VDF cell` (same for O) — `N` a handful
  (5-15 on the fixture), not hundreds; hundreds means a broken
  zero-contour-intersection or Hessian-sign step.
- `d_i`/`rho_i` annotations are physically plausible (`0.01-1 R_E` near the
  current sheet here); `0`/`NaN` means the density read failed — check
  `density_variable` against `reader.get_all_variables()`.
- X-point box is a **rectangle**: narrow perpendicular to B
  (`half_width_di_normal`), long along B (`outflow_aspect_ratio`, default
  1:10). A square means `outflow_aspect_ratio` isn't applied — check
  `get_vdf_cellids_in_b_perp_di_box`.
- O-point area is a **circle** (`radius_gyroradii × rho_i`, method
  `"gyroradius"`) or a flux-contour blob (`"flux_contour"`). Toggling
  `selection_method` should visibly change both shape and matched count, or
  the dispatch in `get_o_point_cellids_by_method` isn't being reached.
- A box/circle that matches zero cells no matter how wide `regions_re` is
  set is a VDF-density problem, not a selection bug — compare the physical
  half-width against the spacing between neighboring `vdf_coords_re` points.
- `src/deprecated.py`'s `get_physical_point_cellids_by_position` (old
  Hessian-eigenvector X-box + flux-contour O-selection) is intentionally
  different from the current B-perpendicular/gyroradius path — don't make
  them match if you're ever reading it for reference.

### Point substances: `current_layer`

File: `physics/current_layer.py` (`find_current_layer_core_records`); the
final VDF-cell selection itself is `labeling/snapshot_labeling.py`'s
`find_current_layer_cellids`.

No standalone diagnostic script exists for this one yet (unlike
`plot_nulls.py` for X/O) — that's a documented gap, not a design choice.
Until one exists, the fastest check is calling
`find_current_layer_core_records`/`find_current_layer_cellids` directly
against just a `VlsvReader` in a snippet (seconds, no VDF extraction, no
saving) rather than running the full `extract_data.py`; watch its console
output and `all_vdfs.png` either way.

```
python scripts/data_proc/extract_data.py
```

A VDF cell is labeled `current_layer` only if its own cellid IS one of the
peak-`|J|` core cells found by `find_current_layer_core_records` — a direct
cellid intersection, not a search. An earlier version instead expanded each
core cell by `margin_di * d_i` along its local LMN normal and nearest-
matched every expanded point to the closest VDF cell
(`vdf_tools.get_nearest_vdf_cellid`); that nearest-match had no maximum-
distance cutoff, and on this fixture VDF cells sit on a coarse ~2.35 R_E
subgrid while `margin_di * d_i` was only ~0.01-0.1 R_E (three orders of
magnitude smaller), so the "expansion" never changed which VDF cell got
picked and could silently claim one well outside the real core (e.g. into
`inner_magnetosphere`/`magnetosheath`). Removed entirely; `core_fraction`
is now the only lever on how large the `current_layer` selection is.

**What good looks like:**
- Console prints `N current-layer core points -> M cells`. `N` (core grid
  points, on the dense `physics/current_layer.py` grid, not VDF cells)
  should be small relative to the search box's total grid-point count —
  `core_fraction` means only the top `(1 - core_fraction)` fraction of `|J|`
  values qualify *within each named `search_regions_re` box independently*
  (dayside vs. tail, by default — a single global threshold would let the
  stronger dayside magnetopause current swallow the weaker tail sheet
  entirely), after excluding anything within `min_r_re` of Earth. `M` (the
  actual `current_layer` VDF-cell count) is now typically much smaller than
  `N`, since a VDF cell only counts if it happens to coincide exactly with
  a core grid point — on the fixture with `core_fraction = 0.4`:
  `2014 current-layer core points -> 7 cells`. Raising `core_fraction`
  (tighter core) shrinks both `N` and `M`; the *shape* of the `N`-point
  cloud (two coherent curves along the dayside magnetopause and tail
  neutral sheet, no near-Earth scatter) is what to check, not the raw
  count. If `N` is huge (most of a `search_regions_re` box), `core_fraction`
  is too permissive or `|J|` computed near-uniform (check `B` isn't
  accidentally all zero -- an all-zero `B` grid gives an all-zero Jacobian,
  so `Jmag` is zero everywhere and `peak_j <= 0` short-circuits that
  sub-region to zero core records, not a crash). If a real
  structure (e.g. the tail sheet) is missing entirely, check
  `search_regions_re`'s boxes actually cover it, and that the peak `|J|` in
  that box isn't itself sitting on a zero-density (vacuum/inner-boundary)
  cell -- `find_current_layer_core_records` already restricts the peak
  search to cells with `density > 0` for exactly this reason (a known
  failure mode caught during development, see the table below). If `M` is
  `0` even though `N` is a sensible-looking band, that's expected when no
  VDF cell happens to sit inside the core this snapshot -- VDF cells are
  much sparser than the dense grid; raise `core_fraction`'s search box or
  lower `core_fraction` slightly rather than treating a zero as a bug.
- `all_vdfs.png` draws two current-layer overlays now: a scattered pink dot
  cloud (`draw_current_layer_region`, one dot per core grid point -- the
  raw physical detection) and diamond markers (`deeppink`, `region_styles`
  -- the actual selected VDF cells). The diamonds should sit visibly inside
  the dot cloud, never off to the side of it; a diamond outside the cloud
  would mean selection and detection have drifted apart.
- The dayside dot cloud should sit right on `all_vdfs.png`'s black
  magnetopause curve — a strong cross-check, since the two are computed by
  completely independent methods (peak current-density vs. the Shue
  density-scan fit) and should still agree spatially.

### Magnetosphere region classification

File: `physics/magnetopause.py` (`find_subsolar_point`, `fit_shue_model`,
`shue_boundary_r_re`, `classify_magnetosphere_regions`).

`classify_magnetosphere_regions` is a pure function of `coords_re` +
density (no reader/VDF needed at all once you have those two arrays) —
the fastest way to smoke-test it after a rule change is a tiny synthetic
`coords_re`/density array in a snippet, checking the returned labels match
by hand for a few constructed points (e.g. one point just inside `r_mp`, one
just outside `lobe_r_min_re`). Against the real fixture, it's exercised by
`extract_data.py`/`verify_data.py` ("Assembling the dataset" and Stage 4
below) — its `Label counts:` dict and `all_vdfs.png` plot are built from
the same Shue fit computed here.

**What good looks like:**
- The fitted Shue model, independent of the search box (it always scans the
  full +x axis): on the fixture, `r_mp ≈ 7.97 R_E` (magnetopause, density
  `1.13e6 → 1.85e5`), `r_bs ≈ 19.34 R_E` (bow shock, `1.10e6 → 2.86e6`, a
  3-4x compression — the theoretical max for a perpendicular MHD shock; far
  outside that range means the bow-shock index picked the wrong extremum).
  `r_bs == None` means the scan didn't reach undisturbed solar wind — check
  `x_scan_max_re` before trusting `magnetosheath` counts.
- `Label counts:` region tallies should track the geometry, not be 100% one
  region. On the fixture (`SPATIAL_BOXRE`), with the default
  `active_point_substances = ["current_layer"]`:
  `{'lobes': 35, 'undefined': 5, 'inner_magnetosphere': 22, 'current_layer': 7,
  'magnetosheath': 27, 'solar_wind': 20}` (`current_layer` is a point
  substance and overrides whichever base region a cell would otherwise get,
  so it doesn't add to the total the way an X/O-style rare point-count
  would — see the `current_layer` subsection above). With
  `active_point_substances = ["x_o_points"]` instead:
  `{'lobes': 40, 'undefined': 5, 'inner_magnetosphere': 22, 'magnetosheath': 27,
  'solar_wind': 20, 'x_point_o_point': 1, 'x_point': 1}`. Either way, a
  `magnetosheath` count that balloons to most of the box means the bow
  shock isn't being applied (see failure-mode table below).
- `inner_magnetosphere`/`lobes` split at `MAGNETOPAUSE_CONFIG["lobe_r_min_re"]`
  (default `10.0 R_E`), *not* `r_mp` (~`7.97 R_E` on the fixture) — `r_mp` is a
  dayside-only standoff distance; applying it as a uniform-angle sphere
  would mislabel near-Earth nightside plasma as `lobes` (see failure-mode
  table below). The band between the two circles (`r_mp <= R < lobe_r_min_re`)
  is labeled `undefined` rather than assigned to either — 5 cells on the
  fixture, regardless of which point substance is active (it's a base
  region rule, computed before any point substance is applied).
- `all_vdfs.png` draws two separate circles, never one standing in for the
  other: the blue dashed `R = r_mp` circle (magnetopause standoff, ~`7.97
  R_E`) and a cyan dotted `R = <lobe_r_min_re>` circle (`10.0 R_E` by
  default) -- lobes fill the tail outside the *cyan* circle, inner
  magnetosphere is inside it; the blue `r_mp` circle sits inside that and
  isn't the classification boundary for either label.

## Assembling the dataset (`extract_data.py`)

Files: `scripts/data_proc/extract_data.py` (see README "Pipeline"). This
script runs Stages 1-3 together and saves the result — it's the thing
downstream verification/PCA stages actually consume. Edit
`src/data_proc/pipeline_config.py` (shared by every downstream script) to
point at the Stage 1 fixture, then:

```
python scripts/data_proc/extract_data.py
```

Leave `BUILD_ROTATED_DATASET = False` and `BUILD_HERMITE_DATASET = False`
unless the change under test specifically needs Stage 2's output — this
keeps a full run to seconds instead of ~6-7 minutes (see the cheat sheet
above).

**What good looks like:**
- Prints `Loaded N VDFs`, a count line per active point substance
  (`active_point_substances`), and a `Label counts:` dict that isn't 100%
  one region — all-one-region usually means `REGIONS_RE`/`POINTS_CONFIG`
  don't overlap any real structure, or `SPATIAL_BOXRE` is too
  small/misplaced.
- `X.npy` shape is `(n_samples, vx, vy, vz)`, `n_samples` matches
  `metadata.csv`'s row count; `metadata.csv` has one `label` column (no
  separate `y.npy` — see README's "known gap"), plus per-cell `bx_t`/`by_t`/
  `bz_t`, `vx_ms`/`vy_ms`/`vz_ms`, and the snapshot-constant
  `vspace_*min_ms`/`vspace_*max_ms` velocity-mesh extent (same
  constant-per-row convention as `timestep`). If `BUILD_ROTATED_DATASET`
  was `True` for this run, a second array `X_rotated.npy` (same shape as
  `X.npy`) is also saved — see Stage 2 above for its cost. If
  `BUILD_HERMITE_DATASET` was also `True`, `X_hermite.npy` and the moment
  feature columns (`hermite_density_m3`, `hermite_ux_ms`/`uy_ms`/`uz_ms`,
  `hermite_vthx_ms`/`vthy_ms`/`vthz_ms`) land in `metadata.csv` too.
- `VDF statistic:` block (`min`/`max`/`mean`/`std`, from
  `print_vdf_statistics`): a tiny negative `min` (e.g. `-1e-20` vs. a `max`
  around `1e-10`) is normal float32 noise near zero. What's actually
  diagnostic: `max`/`std` of exactly `0`, or `min`/`max` the same magnitude
  as `mean` — either means extraction returned empty/constant arrays.
- `data/plots/extract_data/<RUN_ID>/all_vdfs.png` and `metadata.csv` are
  built from the same in-memory `labels` array, so they can never disagree
  — a visual mismatch is a plotting bug, not a ground-truth one.
- Any single-VDF slice plot should show a contiguous blob near the
  annotated bulk velocity, not scattered noise or an empty axes. Multi-panel
  grids (one VDF per cluster) must slice through each VDF's own peak index,
  not the mesh's geometric center (see failure-mode table below).
- **After any refactor touching this stage**, don't stop at
  `py_compile`/import success — re-run `extract_data.py` against the same
  fixture and diff `X.npy` (`np.array_equal`) and `metadata.csv`'s `label`
  counts against a pre-change copy.

**Legacy path** (`src/deprecated.py`'s `create_dataset`, old static label
scheme, no script wraps it anymore):
```python
from src.data_proc.config import load_config
from src.deprecated import create_dataset
create_dataset(config=load_config("configs/create_dataset_local_smoke.yaml"),
               start_timestep=3408, n_timesteps=1, dataset_kind="train")
```
`plot_dataset_sample.py`/`plot_dataset_pca.py` still expect that legacy
`X.npy`/`y.npy`/`metadata.csv` triple; `plot_dataset_sample.py`'s
`--timestep` templates into `configs/plot_dataset_sample.yaml`'s
`dataset_dir` (`data/train/timesteps_{timestep}`) — point `--config` at a
matching YAML if you used the smoke config to create the dataset. This is
also the dataset format `train_cnn.py` (Stage 6, future) currently expects
— see the note at the end of this file.

## Stage 4 — Verification (`verify_data.py`)

File: `scripts/data_proc/verify_data.py`. Recomputes ground truth directly
from the reader (same call as Stage 3's `compute_snapshot_ground_truth`) —
it does **not** need `extract_data.py` to have been run first, and only
extracts a handful of representative VDFs (one per label), never all
~116 — cheaper than a full `extract_data.py` run, and the preferred check
for any labelling-only change.

```
python scripts/data_proc/verify_data.py
```

**What good looks like:**
- Prints `M VDF cells across K labels`, then `Representative cells: [...]`
  — `K` should match the number of distinct labels active this run (base
  regions + whichever point substance(s) `active_point_substances` selects).
- `vdf_examples.png` is one combined figure now (`plot_cluster_vdf_examples`,
  header row via `_draw_cluster_positions`): a standalone `vdf_positions.png`
  is no longer produced by this script, though `plot_cluster_vdf_positions`
  is still directly callable if the positions plot alone is ever needed.
  - Header row: each representative's marker sits inside the region its
    label implies (e.g. a `magnetosheath` marker between the two Shue
    circles, a `current_layer` marker on the dot cloud from Stage 3) — a
    marker sitting somewhere that contradicts its own label means Stage 3's
    labelling and this plot's cellid lookup have drifted apart.
  - Rows below: each representative's three velocity-space cuts (vx-vy,
    vx-vz, vy-vz) show a contiguous blob, not scattered noise or an empty
    panel — same peak-slicing caveat as "Assembling the dataset" above
    (fast-flowing populations need their own peak index, not the mesh
    center).

See also `plot_nulls.py` (Stage 3, X/O sanity) and `plot_vdf_hermite.py`
(Stage 2, manual VDF/Hermite/rotation drill-down) for other verification
angles this script doesn't cover.

## Stage 5 — PCA analysis

Files: `src/ml_models/vdf_snapshot_clustering.py`,
`scripts/ml_models/run_snapshot_pca.py`/`plot_snapshot_pca.py`. Needs a
saved dataset from "Assembling the dataset" above for this `RUN_ID` — the
PCA/KMeans fit itself is sub-second at ~116 samples, so iterating on
`PCA_CONFIG` never needs a re-extraction, only a re-run of these two
scripts. See [`PCA_GUIDE.md`](PCA_GUIDE.md) for the full pipeline
(feature representations, per-sample normalization, moment-feature
weighting) — this section is the testing checklist, not the design doc.

```
python scripts/ml_models/run_snapshot_pca.py
python scripts/ml_models/plot_snapshot_pca.py
```

**What good looks like:**
- Blind PCA+KMeans clustering on the saved dataset, scored against
  `metadata.csv`'s `label` column — good result: the smallest clusters
  overlap heavily with the rarest, most physically distinct substances
  active that run — on the fixture with the default
  `active_point_substances = ["current_layer"]`, that's `magnetosheath`
  and `current_layer`, not `x_point`/`o_point`, which only show up when
  `active_point_substances = ["x_o_points"]` instead.
- `PCA_CONFIG["feature_representation"]`: `"raw"` uses `X.npy` (always
  present); `"rotated"` needs `BUILD_ROTATED_DATASET = True` on the
  `extract_data.py` run that produced this dataset; `"hermite"` needs
  `BUILD_HERMITE_DATASET = True` (this always pulls in rotation as a
  prerequisite even if `BUILD_ROTATED_DATASET` itself was left off). A
  clear `FileNotFoundError` instead of a silent fallback means the array
  that mode needs wasn't built — re-run "assembling the dataset" (or, for
  `"hermite"` alone, just `rebuild_hermite_dataset.py`, Stage 2) with the
  right flags rather than treating this as a bug.
- For `"hermite"`, watch out for a degenerate result: if `Best k` collapses
  to `2` with one giant cluster (~all samples) and one or two singleton
  outliers despite a deceptively *high* silhouette score, that usually
  means the representation is dominated by an overall scale factor
  (density, or a too-high Hermite `order` letting numerically-noisy
  high-order coefficients get amplified by `StandardScaler`) rather than
  shape — this was caught once during development, see the failure-mode
  table below. `include_moment_features`/`moment_feature_weight` and
  `sample_normalization` (Stage 2's per-sample scale removal) are the two
  levers that fixed this in this project's history — don't reach for a
  higher `HERMITE_ORDER` first, that's a different knob (basis truncation,
  not scale).
- `cluster_hermite_spectra.png`/`phys_cluster_hermite_spectra.png` (only
  produced when `feature_representation == "hermite"`): visual check
  described in Stage 2's "Correctness check" above, applied to whichever
  clusters/labels this run actually produced.

### Second stage: SOM within each blind cluster

`scripts/ml_models/run_snapshot_som.py` — one Self-Organizing Map per
blind cluster, fit on `pca_results.npz`'s saved `pca_scores` (needs
`run_snapshot_pca.py` to have run for this `RUN_ID`; opens no VLSV file,
~15s on the fixture, so iterating on `SOM_CONFIG` is cheap).

```
python scripts/ml_models/run_snapshot_som.py
```

**What good looks like:**
- Console prints one block per blind cluster (clusters below
  `min_cluster_size` are skipped with a message, not an error): the
  per-tier PCA refit line (component count + % of within-tier variance
  kept — on the fixture, 8 components keep 92% for the calm tier but
  only ~56% for the more heterogeneous disturbed one), quantization
  error, and, if `n_node_clusters > 0`, an **adjusted Rand index** plus
  per-node-cluster expert-label compositions. The ARI is the tuning
  metric: fixture baseline is ~0.40 (calm) / ~0.31 (disturbed) at
  `n_components = 8`, `som_shape = (4, 4)`, hit-weighted codebook KMeans
  (see `fit_som_map`'s docstring and `SOM_CONFIG`'s comment for why —
  small tiers leave most SOM nodes empty or singleton-occupied, so an
  unweighted codebook KMeans lets those nodes outvote nodes dozens of
  samples agreed on; weighting by hit count fixes that, and a smaller
  map reduces how many nodes are that starved in the first place). Both
  levers were verified against node-occupancy counts, not picked by
  best score alone — but the disturbed tier (42 samples) still swings
  noticeably between single-seed configurations, so treat a clear ARI
  drop after a change as a regression signal and don't chase
  per-node-cluster composition anecdotes (they reshuffle between
  configurations more than the ARI does).
- `som_label_maps.png`: expert labels should occupy coherent *regions* of
  each map, not be scattered uniformly — on the fixture's calm cluster,
  `solar_wind` sits in one isolated corner and `lobes`/
  `inner_magnetosphere` hold largely distinct territories. Uniformly
  mixed labels across the whole map would mean the SOM (or the PCA space
  it's fit in) carries no substance information at this granularity.
- The same sample must sit at the same jittered position in the U-matrix
  row and the codebook-partition row — a mismatch means the fixed-seed
  jitter regressed (see `plot_som_label_maps`'s comment).

**DBSCAN codebook clustering** (`SOM_CONFIG["node_cluster_method"] =
"dbscan"`, alternative to the default `"kmeans"`): same hit-weighting,
but doesn't assume convex node-clusters and can mark a node `-1` ("noise")
instead of forcing it into a cluster — rendered as a white/unfilled cell
in `som_label_maps.png`'s codebook-partition row, not a solid color.
`dbscan_eps` needs calibrating against the codebook's actual pairwise-
distance scale first (`scipy.spatial.distance.pdist` on
`som.get_weights()`) — an uncalibrated eps below that scale silently
produces one cluster per node on every value tried (identical,
eps-invariant output is the tell). On the fixture, `dbscan_eps = 18`
beats tuned KMeans on both tiers (ARI 0.430 calm / 0.330 disturbed vs.
0.404 / 0.306) — see `docs/PCA_GUIDE.md` for the full comparison; not yet
adopted as the default pending validation on a production snapshot.

## Stage 6 (future) — CNN training / cluster recognition

Not yet built against this pipeline. `scripts/ml_models/train_cnn.py`/
`predict_region.py`/`predict_coordinate.py` exist but currently target the
**legacy** dataset format (`src/deprecated.py`'s `create_dataset`, old
static label scheme — see "Legacy path" above), not the substance taxonomy
`extract_data.py`/PCA analysis use now. There is no current test coverage
for training a CNN on Stage 5's clusters or the current label set — treat
any such request as new-feature work, not a regression check, until this
stage is actually wired up. When it is, extend this section with the same
pattern as Stage 5: a "what good looks like" checklist, and a cheat-sheet
entry for the fastest way to test a training-only change without
re-running extraction/rotation/Hermite.

## Known failure modes (caught during development — check these don't regress)

These were each found by actually looking at a plot, not by reading code.
If a change touches the area named, re-check the specific symptom.

| Symptom | Area | Cause / fix |
|---|---|---|
| X-point ground truth matches cells tens of R_E away from the source point | `labeling/point_labels.py` (X-box) | Only one in-plane axis of the search box was bounded, turning it into an infinite strip. Both axes must be bounded. |
| X-point search box renders as a plain square regardless of aspect ratio | `labeling/point_labels.py`/`plot_tools.py` | `outflow_aspect_ratio` not applied to the perpendicular-to-B axis, or the drawn `Polygon` and the actual selection box use different half-widths — they must match. |
| O-point search circle radius looks right in text but never intersects any VDF cell | `magnetopause.py` scan range vs. `di`/`rho_i` scale | Not a bug per se — a genuinely tiny physical scale (`d_i`/`rho_i` ~0.01-0.1 R_E) vs. sparse VDF sampling (~1+ R_E spacing on this fixture). Widen the multiplier to test the *shape* logic; don't assume 0 matches means broken code. |
| `magnetosheath` cell count is an order of magnitude too large, extending tens of R_E past any real magnetosheath | `magnetopause.py` | The Shue model only detects the magnetopause (inner boundary); without also detecting the bow shock (outer boundary), everything sunward of the magnetopause — including undisturbed solar wind far upstream — gets counted as sheath. Fixed by also scanning for the density-rise crossing (bow shock) sunward of the magnetopause drop, and adding a `solar_wind` label beyond it. |
| One named cluster (e.g. an O-point) silently vanishes from a combined-label plot even though its ground-truth count is nonzero | Any code merging multiple label sources by overwrite priority | Two independently-detected structures (an X-point and an O-point) can legitimately match the *same* VDF cell on a sparse grid. A priority overwrite (`labels[is_x] = "x_point"` after `labels[is_o] = "o_point"`) erases the loser instead of representing the overlap. Give the overlap its own label instead of silently picking a winner. |
| A per-cluster VDF grid shows a real distribution in one velocity-plane column but a blank/empty panel in another, for fast-flowing clusters specifically (solar wind, magnetosheath) | Any VDF slicing that fixes an axis at the mesh's geometric center | A beam offset far from `v=0` along one axis (e.g. `vx ≈ -750 km/s` for solar wind) isn't present at that axis's center index, so a plane that fixes that axis at center (e.g. `vy-vz` fixing `vx`) samples empty space. Slice through the VDF's own peak index instead (`np.unravel_index(np.argmax(vdf), vdf.shape)`). |
| `current_layer` core search returns 0 records even though `|J|` clearly peaks somewhere in the search box | `physics/current_layer.py` (`find_current_layer_core_records`) | The dense grid's peak `|J|` was being searched over the *whole* box, including the simulation's inner boundary/vacuum region — the dipole field's strong curvature there produces a numerically large `curl(B)` with no physical current behind it (density is exactly `0`), so it silently won every peak search and then every one of its "core" cells failed the `density > 0` check, leaving nothing. Fixed by restricting the peak-`|J|` search itself to `density > 0` cells, not just filtering after the fact. |
| `current_layer` finds the dayside magnetopause but never the tail current sheet (or vice versa) even with a search box that spans both | `physics/current_layer.py` (`find_current_layer_core_records`) | A single global `core_fraction * peak(|J|)` threshold lets the strongest structure in the box (usually the dayside magnetopause) swallow the whole search — a real but weaker structure (the tail sheet) never gets close to that peak. Fixed by searching each named box in `current_layer_selection["search_regions_re"]` (e.g. `"dayside"`/`"tail"`) independently, each against its own local peak, then unioning the results. |
| Lowering `core_fraction` to catch a weak real structure (e.g. the tail sheet) also pulls in a couple of points right next to the inner boundary, at high `|z|` relative to `|x|` | `physics/current_layer.py` (`find_current_layer_core_records`) | Field-aligned currents near the inner simulation boundary are a different physical structure (mapped along B to the ionosphere, not a cross-field current sheet), but a `dayside`/`tail` x-only split doesn't exclude them — they can sit inside either box and, once the threshold is lowered enough, compete with the box's real peak. Recognizable by `R = sqrt(x_re**2 + z_re**2)` a few R_E (on the fixture, `R ~ 4.6 R_E` vs. `R >= 8.2 R_E` for real detections) and by a suspiciously exact, uniform density across all of them (a boundary-condition fill value, not physically-varying plasma). Fixed by `current_layer_selection["min_r_re"]`, excluding cells within that radius of Earth from every sub-region's search before any peak is computed. |
| `all_vdfs.png`'s only cutoff circle is labeled/positioned at `r_mp` even though `inner_magnetosphere`/`lobes` were classified against a different radius (`lobe_r_min_re`) | `plot_tools.py` (`draw_shue_boundaries`) | An early version repurposed the single `show_r_mp_circle` toggle to draw `lobe_r_min_re` *instead of* `r_mp` when the two differed -- conflating two genuinely different physical quantities (the fitted magnetopause standoff vs. an independently-chosen classification radius) into one circle. Fixed by drawing them as two separate, independently-toggled circles (`show_r_mp_circle` always draws `r_mp`; a new `lobe_r_min_re` param draws its own circle, distinct style, only if given), so neither is ever silently swapped for the other. |
| `feature_representation = "hermite"` PCA collapses to `Best k = 2`, one giant cluster (nearly all samples) plus one or two singleton outliers, despite a *higher* silhouette score than the raw-pixel baseline | `ml_models/vdf_snapshot_clustering.py` (`fit_pca_clusters`'s `StandardScaler`) vs. `physics/vdf_transform.py` (`vdf_to_hermite_spectra`) | The raw (non-log) Hermite spectra of a VDF scale ~linearly with that VDF's own density (the `(0,0,0)` coefficient IS, up to a constant, the density) -- u/vth normalization removes position/width as confounds but not overall amplitude. `StandardScaler` treats every coefficient as equally informative, so the one or two samples with genuinely unusual density dominate variance and every other coefficient (the real shape information) gets drowned out -- the "scale, not shape" confound this representation was meant to fix, just showing up through density instead of pixel position. Fixed by projecting `log10(vdf)` instead of the raw linear `vdf` (`vdf_to_hermite_spectra_log`, log-compresses the dynamic range the same way the raw-pixel feature path already does) rather than trying to normalize the raw spectra after the fact. |
| Log-space Hermite spectra have magnitude ~1e9-1e10, dwarfing every other sample, even after the log fix above | `physics/vdf_transform.py` (`vdf_to_hermite_spectra_log`) | Substituting `log10(sparsity_threshold)` as a literal floor value for background cells (instead of subtracting it) integrates a huge sample-independent constant against every Hermite basis function over the ~99.85%-empty grid — the result is dominated by each sample's own basis-normalization overlap with that constant, not real VDF shape. Fixed by computing `log10(vdf/sparsity_threshold)` so background cells map to exactly `0` (compact support preserved), not a large floor value. |
| Two independent PCA runs on the same physical clusters give visibly different-scale results with no config change, tracing back to `current_layer`/`magnetosheath` specifically | `ml_models/vdf_snapshot_clustering.py` / feature engineering | Not a bug — a genuine two-tier physical amplitude difference: `current_layer`/`magnetosheath` (structured, non-Maxwellian) have Hermite spectra ~6-9x larger in magnitude than `lobes`/`solar_wind`/`inner_magnetosphere`/`undefined` (quiet, near-Maxwellian), confirmed via `plot_cluster_hermite_spectra` to be the *same qualitative shape*, just different scale. Fixed (where it caused degenerate clustering) via per-sample normalization (`sample_normalization = "standard"`) before the per-feature `StandardScaler`. |

## Regression checklist after refactoring `data_proc`/`ml_models`

This is the **local** verification tier — see `PIPELINE.md`'s "Two
verification tiers" for when this is enough vs. when a full repo-wide
orchestrator pass is warranted instead. Use the cheat sheet near the top of
this file to pick the cheapest command that actually exercises the stage(s)
touched — don't default to a full `extract_data.py` run while iterating.

1. `python -m py_compile` every touched file — catches syntax/import errors
   only, not wiring mistakes.
2. Re-run the actual stage(s) touched, using the cheapest tool from the
   cheat sheet above (a single-cell snippet, `plot_nulls.py`,
   `plot_vdf_hermite.py`, `rebuild_hermite_dataset.py`, or `verify_data.py`
   — not necessarily the full pipeline) while iterating.
3. Before considering the change done: run the **full** `extract_data.py`
   (with `BUILD_ROTATED_DATASET`/`BUILD_HERMITE_DATASET` matching what
   downstream stages need) against the fixture once, as the orchestrator-
   tier sign-off.
4. If the change could affect dataset output: diff `X.npy`/`X_rotated.npy`/
   `X_hermite.npy` (`np.array_equal`) and `metadata.csv`'s `label` counts
   against a pre-change copy.
5. If the change could affect a plot: actually open the saved PNG and look
   at it. Every failure mode in the table above was caught this way, not by
   reading the diff.
