# Data proc (tech tools) — `src/data_proc/`

VLSV I/O, generic array/config utilities, dataset I/O, and plotting
infrastructure. Scheme-agnostic: nothing here assumes a particular label
scheme. May depend on `src/data_proc/physics/`; must not depend on
`src/data_proc/labeling/` or `src/ml_models/`.

**When adding a function here**: add a one-line entry below.

## `config.py`

- `load_config(config_path)` — load a YAML config file.
- `create_path(path_template, **values)` — format a path template with arbitrary keyword values.
- `create_timestep_path(path_template, timestep)` — format a `{timestep:04d}`-style bulk/flux file path template.

## `pipeline_config.py` — shared parameter values, not functions

Not a library module -- plain Python constants (`FILE_LOCATION`, `SPATIAL_BOXRE`, `REGIONS_RE`, `POINTS_CONFIG`, `MAGNETOPAUSE_CONFIG`, `PLOT_BOXRE`, `RANDOM_STATE`, `POP`, `VDFLIM`, `PCA_CONFIG`, etc.) imported by `scripts/data_proc/extract_data.py`, `scripts/data_proc/verify_data.py`, and `scripts/ml_models/run_snapshot_pca.py`/`plot_snapshot_pca.py` (`from src.data_proc import pipeline_config as config`) so they can never disagree about what one run means. `plot_nulls.py`/`plot_vdf_hermite.py` intentionally keep independent parameters instead (see the module's own header comment).

## `vdf_tools.py` — coordinate/region helpers, cellid lookups, VDF-plot-parameter readers, dense VDF extraction

Organized into four sections, in file order, each independently extractable
into its own module later if this file grows again:

### Masking — coordinate/region matching, box membership

- `coord_re_to_m(coord_re)` — convert a coordinate in Earth radii to meters.
- `create_coordinate_name(coord_re)` — filesystem-safe coordinate name (e.g. `xm12_y0_z0p5`).
- `iter_enabled_regions_re(points_config, names_key, default_region_name="tail")` — yield `(region_name, region_re)` for configured regions.
- `find_matching_region_name_re(coord_re, points_config, names_key)` — first configured region containing a coordinate.
- `is_coord_in_region_re(coord_re, region_re)` — whether one coordinate falls inside a `region_re` box.
- `create_region_mask_re(coords_re, region_re)` — vectorized version of the above over an array of coordinates.
- `get_region_axis_bounds_re(region_re, axis_name)` — `(lower, upper)` bounds for one axis from `*_between`/`*_abs_max`/`*_min`/`*_max` keys.

### Primitives — cellid/coordinate/field lookups against an open reader

- `get_cellid_with_vdf(reader, coord_re, pop="avgs")` — spatial cell ID with a VDF nearest to a coordinate.
- `get_vdf_cellid_set(reader, pop="avgs")` — set of every VDF-carrying cell ID for a population.
- `get_vdf_cells_with_coords_re(reader, pop="avgs")` — same, with each cell's coordinates.
- `get_nearest_vdf_cellid(coord_re, vdf_cellids, vdf_coords_re)` — nearest VDF cell to a coordinate, from an already-loaded cellid/coords pair.
- `get_b_field(reader, cid)` / `get_bulk_velocity(reader, cid)` — read `B`/`V` at a cell.

### VDF plot parameters — velocity-mesh extent/spacing/threshold, xz slicing

- `get_velocity_cell_size_from_extent(extent, vdf_shape, axis="vy")` — velocity cell size `dv` along one axis.
- `get_vdf_plot_parameters(reader, cid, vdf_shape, pop="avgs")` — `(extent, dv, threshold)` for a VDF sample, from an open reader.
- `get_vdf_plot_axes_parameters(reader, vdf_shape, pop="avgs")` — `(extent, dv)` only.
- `get_vdf_plot_threshold(reader, cid)` — sparsity threshold (`MinValue`) below which a cell's VDF values are noise.
- `get_vdf_plot_parameters_from_file(file_location, cid, vdf_shape, pop="avgs")` — `get_vdf_plot_parameters`, opening the VLSV file directly.
- `create_xz_slice(vdf)` — middle xz slice of a dense 3D VDF, `(vx, vy, vz) -> (vx, vz)`.

### Extractor — dense VDF extraction from sparse VLSV velocity-space data

- `class VdfExtractor` — open-reader-scoped VDF extraction helper; caches the sorted velocity-cell ordering once per population (`.extract(cid, box=-1)` per cell after that).
- `create_sorted_velocity_indices(reader, n_velocity_cells, pop="avgs")` — precompute the index order needed to densify a VLSV velocity block layout.
- `extract_vdf(file_location, cid, box=-1, pop="avgs")` — extract one dense VDF array directly from a file path (opens its own reader); prefer `VdfExtractor` when extracting many cells from the same file.

(`cell_has_vdf`, `get_spatial_index_range`, `get_vdf_cellids_in_box` were removed — confirmed zero callers anywhere, live or deprecated.)

## `batches.py`

- `iter_array_batches(X, indices=None, batch_size=64)` / `iter_index_batches(indices, batch_size)` — batch iteration over an array or an index list.
- `get_array_batch(X, batch_indices)` — fetch one batch, using a contiguous slice when possible.
- `create_contiguous_slice(indices)` — detect whether a sorted index list is a contiguous range.

## `dataset_io.py` — generic, label-scheme-agnostic saved-dataset I/O

- `save_metadata(outdir, metadata)` — write `metadata.csv`.
- `load_dataset(dataset_dir, mmap=True)` — load the legacy `X.npy`/`y.npy`/`metadata.csv` triple.
- `load_labeled_vdfs(dataset_dir, mmap=True, x_filename="X.npy")` — load `X.npy`/`metadata.csv` saved by `labeling.snapshot_labeling.save_labeled_vdfs` (no `y.npy` -- label lives in metadata's `label` column); used by `scripts/ml_models/run_snapshot_pca.py`. Pass `x_filename="X_rotated.npy"` to load the B-frame-rotated array instead (see `PCA_CONFIG["use_rotated_vdfs"]`) -- raises `FileNotFoundError` with a clear message if that run's `extract_data.py` didn't build it.
- `print_vdf_statistics(X, batch_size=64)` — batched min/max/mean/std over a VDF array; called by `extract_data.py` right after saving (X is already in memory, no reload) -- the fast way to catch a broken/all-zero extraction.

## `plot_tools.py` — the plotting library

One file, all live plotting: VDF/Hermite exploration, xz-slice plotting,
O-point overlays, topology diagnostics, and the cluster-plotting pipeline
(driven by `scripts/ml_models/run_snapshot_pca.py`/`plot_snapshot_pca.py`
and `scripts/data_proc/verify_data.py`). Absorbed the old `plot_helpers.py`
and `src/ml_models/vdf_snapshot_clustering_plots.py` — the latter moved
from `ml_models` to `data_proc` since it only ever consumed plain
cellids/labels, not any PCA/KMeans internals. Organized into sections,
in file order:

### Ad-hoc exploration — `scripts/data_proc/plot_vdf_hermite.py`

- `select_vdf_points(cellids, coords_re, points_re=None, x_range=None, y_range=None, z_range=None)` — nearest-cell (via `vdf_tools.get_nearest_vdf_cellid`) or axis-range VDF-cell selection.
- `plot_colormap_with_vdf_markers(...)` — **pipeline step 1**: 2D colormap with VDF-cell markers overlaid; also used by `plot_snapshot_pca.py`.
- `plot_vdf_and_hermite_grid(...)` / `plot_vdf_rotation_comparison(...)` — ad-hoc VDF/Hermite-spectrum and rotation-comparison panels.

### Single-VDF xz-slice plotting — shared by ad-hoc scripts, `src/deprecated.py`, and model failure-case plots

- `extract_plot_xz_slice(vdf)` — plot-oriented middle xz slice of one dense VDF (canonical implementation; `src/ml_models/feature_cache.py` has its own memmap-friendly `extract_plot_xz_slice_from_dataset` variant for batched arrays, kept separate on purpose).
- `prepare_physical_xz_plot(...)` / `prepare_vdf_xz_plot(...)` — threshold/scale a slice into a plot-ready masked array.
- `plot_prepared_vdf_xz_slice_on_axis(...)` / `plot_vdf_xz_slice_on_axis(...)` / `plot_physical_xz_slice_on_axis(...)` — draw a prepared/raw/physical xz slice on a given axis.
- `plot_vdf_xz_slice(...)` / `plot_vdf_xz_slice_from_physical_xz(...)` — save a single-VDF xz-slice figure.

### Region/box helpers

- `get_regions_re_boxre(regions_re, margin_re=2.0)` — plot bounding box from a `regions_re` config, with margin.

### O-point search-area overlay

- `draw_o_point_search_areas(ax, reader, metadata_rows, flux_file_template, o_core_fraction=None)` — draw O-point closed flux-island search contours; used by `plot_snapshot_topology`'s flux-contour display toggle.
- `get_o_point_search_flux(row, o_core_fraction=None)` — contour flux from metadata `search_flux`, or a fallback interpolation (intentionally mirrors `physics.point_topology.find_island_boundary_contour`'s formula) for older metadata rows.
- `_metadata_method_mask(...)` / `_metadata_bool_mask(...)` — private boolean-mask helpers over metadata columns.

### Topology diagnostics

- `plot_snapshot_topology(...)` — sanity-check plot of detected X/O points and their physical search boxes/contours against actual VDF cells in a snapshot; the current topology-diagnostics tool.

### Snapshot colormap + Shue boundaries — shared backdrop for the cluster plots below

- `draw_snapshot_colormap(ax, file_location, boxre, var="rho")` — density colormap background.
- `draw_shue_boundaries(ax, shue_fit, show_r0_circle=True, show_subsolar_marker=False, lobe_r_min_re=None)` — magnetopause/bow-shock curves, plus two independent optional circles that are never conflated: `show_r0_circle` draws `R = r0` (the fitted Shue *magnetopause standoff distance*, always `shue_fit["r0_re"]`), and `lobe_r_min_re` (only drawn if given) draws the separate *inner_magnetosphere/lobes classification cutoff* -- pass the same value `classify_magnetosphere_regions` was actually called with, or the circle silently disagrees with the real split. Plus optional subsolar marker (at `r0`).
- `draw_current_layer_region(ax, current_layer_records, color="deeppink", alpha=0.35, s=6)` — small dot at each current-layer core point's own dense-grid position, showing the detected core's actual physical shape/extent -- for visual comparison against the much sparser VDF cells that actually get labeled `current_layer` (a direct cellid match now, see `find_current_layer_cellids`, no search radius).
- `plot_combined_clusters(...)` — spatial colormap of regions + whichever point substance(s) are active combined (x_point/o_point/x_point_o_point, current_layer -- see `schema.md`); optionally plots the current layer core's dense-grid extent via `draw_current_layer_region` above (pass `current_layer_records`), and passes `lobe_r_min_re` through to `draw_shue_boundaries` so the drawn lobes-cutoff circle (separate from the `r0` circle) matches the real `inner_magnetosphere`/`lobes` split. Backs `scripts/data_proc/extract_data.py`'s post-extraction overview plot.
- `plot_silhouette_scores(...)` — silhouette-vs-k diagnostic plot; backs `scripts/ml_models/plot_snapshot_pca.py`.
- `plot_pca_scatter(...)` — PC1-vs-PC2 scatter with two colorbars on the same points: `cluster_phys` (physical ground-truth region label, large translucent ring) and `cluster_ml` (blind KMeans cluster, small solid dot) -- see README's "Terminology" section. Backs `scripts/ml_models/plot_snapshot_pca.py`.

### Cluster-plotting pipeline — `scripts/ml_models/run_snapshot_pca.py` + `plot_snapshot_pca.py`

Step 1 is `plot_colormap_with_vdf_markers` above; steps 2 and 3 close out the file. The same two functions also back `scripts/data_proc/verify_data.py`, scored against real extraction ground truth instead of blind clusters:

- `plot_cluster_vdf_positions(...)` — **pipeline step 2**: colormap with the spatial position of each cluster's representative VDF. Optionally plots the current layer core's dense-grid extent too (pass `current_layer_records`, see `draw_current_layer_region`) -- `verify_data.py` passes this whenever `"current_layer"` is a key in `compute_snapshot_ground_truth`'s returned `point_substance_records`, so it reacts to whatever `active_point_substances` actually ran, no hardcoding.
- `plot_cluster_vdf_examples(...)` — **pipeline step 3**: example VDFs per cluster, three velocity-space projections per row, sliced through each VDF's own peak.

## `animation.py`

- `create_animation(...)` — assemble a sequence of frame PNGs into a video via ffmpeg.
- `build_ffmpeg_command(...)` — construct the ffmpeg CLI invocation.
