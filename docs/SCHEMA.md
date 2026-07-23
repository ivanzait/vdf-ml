# SCHEMA.md — rules for `cluster_phys` substances

`cluster_phys` (see README's "Terminology" section) is built from a set of
**substances**. This file defines what a substance is and how to add one,
so future additions stay consistent instead of ad hoc.

## What a substance is

A substance is one possible value of `metadata.csv`'s `label` column. Every
substance has exactly two required properties:

1. **Name** — a unique string identifier. The value that ends up in
   `metadata.csv`.
2. **Existence predicate** — a rule that decides, for a given VDF cell,
   whether it belongs to this substance. Not necessarily a single formula:
   `magnetosheath`'s predicate is a Shue-surface distance test;
   `current_layer`'s is "exact cellid match against the peak-|J| core"
   (`core_fraction` is the only lever — no margin/expansion step). What
   matters is that it's a well-defined
   set-membership test over VDF cells, computable from one snapshot.

Nothing else is required. A substance doesn't need its own plot style, its
own config block, or its own detector shape — those are implementation
conveniences the existing substances happen to have, not part of the
contract.

## Two kinds of substance

**Base region substances** — `solar_wind`, `magnetosheath`,
`inner_magnetosphere`, `lobes`, `no_density_data`, `undefined`. Always
computed, every run, by `classify_magnetosphere_regions`
(`src/data_proc/physics/magnetopause.py`), in this fixed priority order:

1. `no_density_data` — density <= 0 (e.g. the simulation's inner-boundary
   vacuum region). Checked first: this is a data-quality flag, not a
   spatial category, so it's kept distinct even from `undefined`.
2. `solar_wind` — sunward of the (flared) bow shock surface, if a bow
   shock was fit.
3. `magnetosheath` — sunward of the magnetopause (Shue surface) but not
   `solar_wind` — everything between the bow shock and the magnetopause.
4. `inner_magnetosphere` — earthward of the magnetopause AND R < `r_mp` (R =
   distance from Earth center, `r_mp` = the fitted Shue standoff distance) —
   a simple spherical cutoff near Earth, not the flared Shue shape.
5. `lobes` — earthward of the magnetopause AND R >= `lobe_r_min_re`
   (`MAGNETOPAUSE_CONFIG["lobe_r_min_re"]`, default `10.0`) — tail
   lobes/plasma sheet.
6. `undefined` — everything else: earthward of the magnetopause but
   neither `inner_magnetosphere` nor `lobes` (the `r_mp`-to-`lobe_r_min_re`
   gap), plus any other cell that doesn't satisfy any rule above.
   `lobe_r_min_re` is deliberately larger than `r_mp` (`r_mp` is a
   *dayside-only* standoff distance; applying it as a uniform-angle sphere
   would pull near-Earth nightside plasma — inner magnetosphere/ring
   current — into `lobes`), so the band between them genuinely isn't
   confidently either category rather than being arbitrarily assigned.
   On the fixture snapshot this band held 5/116 cells (`undefined`), vs.
   19 `inner_magnetosphere` and 37 `lobes` — small, as expected for a gap
   band, not dominant.

There is no `boundary_layer` substance any more — its old role (a
`margin_di × d_i` buffer around the Shue magnetopause surface) is
superseded by `current_layer`'s physically-detected peak-`|J|` core (see
below), which identifies the real magnetopause current layer with much
better precision than a crude geometric buffer.

**Point substances** — `current_layer`; `x_point`/`o_point`/
`x_point_o_point` as one family (see "Substance families" below). Optional:
which one(s) actually run for a given snapshot is a per-run toggle,
`POINTS_CONFIG["active_point_substances"]` (a list, e.g.
`["current_layer"]`, `["x_o_points"]`, or both) in
`src/data_proc/pipeline_config.py`. A point substance's label always
overrides the base region label on that cell — that's what makes it a
*point* substance rather than a region one: rarer, more specific, wins on
overlap. If two active point substances both match the same cell, the one
listed later in `active_point_substances` wins (see
`combine_ground_truth_labels`, `src/data_proc/labeling/snapshot_labeling.py`).

Point substances always override whichever base region label a cell would
otherwise get — including `undefined` and `no_density_data` — since
`combine_ground_truth_labels` applies them after
`classify_magnetosphere_regions` runs. This is what implements
`current_layer`'s top classification priority: a cell inside the current
layer gets labeled `current_layer` regardless of what the base region
rules above would have assigned it, with no special-case code needed.

## Substance families

A "family" is a group of substances produced by one detector call because
they share detection machinery, not because they're conceptually one
thing. `x_point`/`o_point`/`x_point_o_point` is the current example:
`find_ground_truth_point_cellids` (`src/data_proc/labeling/snapshot_labeling.py`)
detects X- and O-points together (they share the same flux-function read),
and a cell matched by both searches gets its own `x_point_o_point` label
rather than one silently overwriting the other. A family still only
appears in `active_point_substances` as one entry (`"x_o_points"`), even
though it can produce more than one label.

## Adding a new point substance

1. Write `find_<name>_cellids(reader, points_config, regions_re=None)` in
   `src/data_proc/labeling/snapshot_labeling.py`, returning a cellid set
   (or a `{label: cellids}` dict if it's a family producing more than one
   label). Mirror `find_ground_truth_point_cellids`'s shape (dense-grid
   detection → physical-unit expansion → nearest-VDF-cell matching via
   `vdf_tools.get_nearest_vdf_cellid`) or `find_current_layer_cellids`'s
   (dense-grid detection → exact cellid match, no expansion) — not a hard
   requirement, just the two shapes every substance here happens to use.
2. Add a matching `if "<name>" in active_point_substances:` block at each
   of the two call sites that build `point_substance_cellids_by_label` —
   `scripts/data_proc/extract_data.py`'s `main()` and
   `compute_snapshot_ground_truth` in `snapshot_labeling.py`. Deliberately
   **no central registry/dispatcher** — each call site keeps its own
   literal, visible toggle blocks, the same way `extract_data.py` already
   keeps its own inline ground-truth computation instead of calling
   `compute_snapshot_ground_truth` (see that script's header comment).
3. Add `"<name>"` to `POINTS_CONFIG["active_point_substances"]` in
   `pipeline_config.py` to turn it on for a run.
4. If it needs its own config, add a
   `POINTS_CONFIG["<name>_selection"]` block (see `current_layer_selection`
   for the pattern) — don't repurpose another substance's config block.
5. If it should render on `plot_combined_clusters`'s overview plot, add an
   entry to that function's `region_styles` dict
   (`src/data_proc/plot_tools.py`).

No other file needs to change: `run_snapshot_pca.py`/`plot_snapshot_pca.py`
build their category set dynamically from whatever's in `metadata.csv`'s
`label` column, so a new substance shows up in `cluster_ml`-vs-`cluster_phys`
scoring and the PCA scatter plot automatically.

See also [`PIPELINE.md`](PIPELINE.md) for the general development rules
this file's conventions sit inside of.
