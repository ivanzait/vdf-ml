# Testing `src/data_proc/`

No automated test suite — the logic is physics-derived (critical-point
detection, magnetopause fitting, VDF geometry) and easier to validate by eye
against a real snapshot than to unit-test against synthetic fixtures.
"Testing" means: run the pipeline stage against the fixture below, and check
the printed diagnostics / saved plots against each section's "what good
looks like" notes — so a fresh session doesn't have to re-derive the checks,
and regressions once caught by eye stay caught.

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

## Stage 1 — VLSV reading + VDF extraction

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

## Stage 2 — Point substances: `x_o_points` and `current_layer`

Both are point substances (see [`schema.md`](schema.md)); which one(s)
`extract_data.py`/`verify_data.py` actually compute is a toggle,
`POINTS_CONFIG["active_point_substances"]` in `pipeline_config.py` — not
mutually exclusive, both are independently testable.

### `x_o_points`

Files: `physics/point_topology.py` (`find_point_records`),
`labeling/point_labels.py` (box/circle selection).

Primary tool: `scripts/data_proc/plot_nulls.py` — edit its `PARAMETERS`
block (`SHOW_X_POINTS`/`SHOW_O_POINTS`, `PLOT_BOXRE`, `REGIONS_RE`,
`POINTS_CONFIG`) and run it. Draws detected X/O points, their search
box/circle, and matched-cell counts on a density colormap in one shot.
Independent of `active_point_substances` — always exercises the detector
directly.

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

### `current_layer`

File: `physics/current_layer.py` (`find_current_layer_core_records`); the
final VDF-cell selection itself is `labeling/snapshot_labeling.py`'s
`find_current_layer_cellids`.

Exercised via `extract_data.py` (Stage 4 below, with `"current_layer"` in
`active_point_substances`, the default) — no standalone diagnostic script
exists for this one yet (unlike `plot_nulls.py` for X/O), so watch its
console output and `all_vdfs.png`.

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

## Stage 3 — Magnetosphere region classification

File: `physics/magnetopause.py` (`find_subsolar_point`, `fit_shue_model`,
`shue_boundary_r_re`, `classify_magnetosphere_regions`).

Exercised by `extract_data.py` (Stage 4 below) — its `Label counts:` dict
and `all_vdfs.png` plot are built from the same Shue fit computed here.

**What good looks like:**
- The fitted Shue model, independent of the search box (it always scans the
  full +x axis): on the fixture, `r0 ≈ 7.97 R_E` (magnetopause, density
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
  would — see Stage 2's `current_layer` subsection). With
  `active_point_substances = ["x_o_points"]` instead:
  `{'lobes': 40, 'undefined': 5, 'inner_magnetosphere': 22, 'magnetosheath': 27,
  'solar_wind': 20, 'x_point_o_point': 1, 'x_point': 1}`. Either way, a
  `magnetosheath` count that balloons to most of the box means the bow
  shock isn't being applied (see failure-mode table below).
- `inner_magnetosphere`/`lobes` split at `MAGNETOPAUSE_CONFIG["lobe_r_min_re"]`
  (default `10.0 R_E`), *not* `r0` (~`7.97 R_E` on the fixture) — `r0` is a
  dayside-only standoff distance; applying it as a uniform-angle sphere
  would mislabel near-Earth nightside plasma as `lobes` (see failure-mode
  table below). The band between the two circles (`r0 <= R < lobe_r_min_re`)
  is labeled `undefined` rather than assigned to either — 5 cells on the
  fixture, regardless of which point substance is active (it's a base
  region rule, computed before any point substance is applied).
- `all_vdfs.png` draws two separate circles, never one standing in for the
  other: the blue dashed `R = r0` circle (magnetopause standoff, ~`7.97
  R_E`) and a cyan dotted `R = <lobe_r_min_re>` circle (`10.0 R_E` by
  default) -- lobes fill the tail outside the *cyan* circle, inner
  magnetosphere is inside it; the blue `r0` circle sits inside that and
  isn't the classification boundary for either label.

## Stage 4 — Dataset creation + verification plots

Files: `scripts/data_proc/extract_data.py` and `verify_data.py` (see
README "Pipeline"). Edit `src/data_proc/pipeline_config.py` (shared by
both) to point at the Stage 1 fixture, then:

```
python scripts/data_proc/extract_data.py   # extract + label + stage-1 overview plot
python scripts/data_proc/verify_data.py    # one representative VDF per label, mapped + cut three ways
```

**What good looks like:**
- `extract_data.py` prints `Loaded N VDFs`, a count line per active point
  substance (`active_point_substances`), and a `Label counts:` dict that
  isn't 100% one region — all-one-region usually means
  `REGIONS_RE`/`POINTS_CONFIG` don't overlap any real structure, or
  `SPATIAL_BOXRE` is too small/misplaced.
- `X.npy` shape is `(n_samples, vx, vy, vz)`, `n_samples` matches
  `metadata.csv`'s row count; `metadata.csv` has one `label` column (no
  separate `y.npy` — see README's "known gap"), plus per-cell `bx_t`/`by_t`/
  `bz_t`, `vx_ms`/`vy_ms`/`vz_ms`, and the snapshot-constant
  `vspace_*min_ms`/`vspace_*max_ms` velocity-mesh extent (same
  constant-per-row convention as `timestep`). If
  `pipeline_config.BUILD_ROTATED_DATASET` was `True` for this run, a second
  array `X_rotated.npy` (same shape as `X.npy`) is also saved — every VDF
  rotated into its own local `(B, v_perp, B x v_perp)` frame (see
  `labeling.snapshot_labeling.rotate_vdfs_to_b_frame`); off by default since
  it costs real per-sample interpolation time (~3s/VDF on this fixture's
  268^3 grid, so ~6 minutes for the full smoke-test snapshot).
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

**Other verification tools** (see README "Pipeline" for the full list):
`plot_nulls.py` (Stage 2), `plot_vdf_hermite.py` (ad-hoc single-file
exploration), `run_snapshot_pca.py`/`plot_snapshot_pca.py` (blind PCA+KMeans
clustering on the saved dataset, scored against `metadata.csv`'s `label`
column — needs `extract_data.py` to have run first for this `RUN_ID`; good
result: the smallest clusters overlap heavily with the rarest, most
physically distinct substances active that run — on the fixture with the
default `active_point_substances = ["current_layer"]`, that's
`magnetosheath` and `current_layer`, not `x_point`/`o_point`, which only
show up when `active_point_substances = ["x_o_points"]` instead. Set
`PCA_CONFIG["feature_representation"]` to `"rotated"` (needs
`BUILD_ROTATED_DATASET = True` on the `extract_data.py` run that produced
this dataset) or `"hermite"` (needs `BUILD_HERMITE_DATASET = True`; note
this always pulls in rotation as a prerequisite even if
`BUILD_ROTATED_DATASET` itself was left off) to cluster `X_rotated.npy`/
`X_hermite.npy` instead of the default `X.npy` — a clear
`FileNotFoundError` instead of a silent fallback means the array that mode
needs wasn't built. For `"hermite"`, watch out for a degenerate result: if
`Best k` collapses to `2` with one giant cluster (~all samples) and one or
two singleton outliers despite a deceptively *high* silhouette score, that
usually means the representation is dominated by an overall scale factor
(density, or a too-high Hermite `order` letting numerically-noisy
high-order coefficients get amplified by `StandardScaler`) rather than
shape — this was caught once during development, see the table below.).

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
matching YAML if you used the smoke config to create the dataset.

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
| `all_vdfs.png`'s only cutoff circle is labeled/positioned at `r0` even though `inner_magnetosphere`/`lobes` were classified against a different radius (`lobe_r_min_re`) | `plot_tools.py` (`draw_shue_boundaries`) | An early version repurposed the single `show_r0_circle` toggle to draw `lobe_r_min_re` *instead of* `r0` when the two differed -- conflating two genuinely different physical quantities (the fitted magnetopause standoff vs. an independently-chosen classification radius) into one circle. Fixed by drawing them as two separate, independently-toggled circles (`show_r0_circle` always draws `r0`; a new `lobe_r_min_re` param draws its own circle, distinct style, only if given), so neither is ever silently swapped for the other. |
| `feature_representation = "hermite"` PCA collapses to `Best k = 2`, one giant cluster (nearly all samples) plus one or two singleton outliers, despite a *higher* silhouette score than the raw-pixel baseline | `ml_models/vdf_snapshot_clustering.py` (`fit_pca_clusters`'s `StandardScaler`) vs. `physics/vdf_transform.py` (`vdf_to_hermite_spectra`) | The raw (non-log) Hermite spectra of a VDF scale ~linearly with that VDF's own density (the `(0,0,0)` coefficient IS, up to a constant, the density) -- u/vth normalization removes position/width as confounds but not overall amplitude. `StandardScaler` treats every coefficient as equally informative, so the one or two samples with genuinely unusual density dominate variance and every other coefficient (the real shape information) gets drowned out -- the "scale, not shape" confound this representation was meant to fix, just showing up through density instead of pixel position. Fixed by projecting `log10(vdf)` instead of the raw linear `vdf` (`vdf_to_hermite_spectra_log`, log-compresses the dynamic range the same way the raw-pixel feature path already does) rather than trying to normalize the raw spectra after the fact. |

## Regression checklist after refactoring `data_proc`

This is the **local** verification tier — see `PIPELINE.md`'s "Two
verification tiers" for when this is enough vs. when a full repo-wide
orchestrator pass is warranted instead.

1. `python -m py_compile` every touched file — catches syntax/import errors
   only, not wiring mistakes.
2. Re-run the actual stage(s) touched against the Stage 1 fixture (this
   file's per-stage commands above).
3. If the change could affect dataset output: diff `X.npy`/`y.npy`
   (`np.array_equal`) and `metadata.csv` against a pre-change copy (Stage 4).
4. If the change could affect a plot: actually open the saved PNG and look
   at it. Every failure mode in the table above was caught this way, not by
   reading the diff.
