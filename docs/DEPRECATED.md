# Deprecated — `src/deprecated.py`

The old student-authored label scheme (`lobe`/`exhaust`/`o_point`/`x_point`/`dayside`,
driven by static `class_coords_re` plus manual/Hessian point selection),
consolidated into one file. Superseded by `src/data_proc/labeling/` (see
`docs/LABELING.md`). **Do not build new functionality on top of this file.**

**No longer has any script wrapper**: `create_dataset.py`,
`backfill_dataset_metadata.py`, `plot_dataset_labels.py`,
`plot_dataset_random_class_samples.py`, `plot_dataset_all.py`, and
`animate_dataset_by_class.py` (the scripts that used to call into this file)
were removed as unused now that `scripts/data_proc/extract_data.py`
is the production path. This file itself is unreferenced by any live code
as of that removal — it's kept only as a readable historical reference, not
because anything still calls it. Worth deleting outright in a future pass
if nobody needs to consult it as reference.

Sections below mirror the `# ==== ... ====` dividers in the file itself.

## X/O-point distance/vector metadata (from `dataset_metadata.py`)

- `create_point_reference_metadata_arrays(...)` — per-sample distance/vector to nearest (or own source) X and O point.
- `create_vdf_spatial_metadata(vdf_coord_re, point_kind=None, source_point_coord_re=None)` — single-sample version of the above.
- `backfill_dataset_spatial_metadata(dataset_dir, config, n_jobs=4)` — recompute the above for an existing dataset's `metadata.csv`, in place, atomically.
- `_normalize_point_coords_re(...)`, `_find_nearest_point_geometry(...)`, `_read_timestep_spatial_data(...)`, `_format_metadata_value(...)` — private helpers for the two functions above.

## Old-scheme point selection (from `point_selection.py`)

- `iter_labeled_coords(config)` — yield static `(class_name, label, coord_re)` tuples from `config.class_coords_re`.
- `create_point_label_data(config, timestep, reader=None)` — detect X/O points and attach class_name/label (old scheme).
- `remove_shared_cellid_points(...)` / `group_point_records_by_cellid(...)` — drop X/O records sharing a VDF cell.
- `is_point_record(...)` / `unpack_labeled_coord(...)` — distinguish detected-point records from static coordinate tuples.
- `get_point_selection_result(...)` — union of physical (Hessian box / flux contour) and manual-box selection, with agreement metadata.
- `get_physical_point_cellids_by_position(...)`, `create_point_selection_result(...)`, `invert_cellids_by_position(...)`, `create_combined_selection_metadata(...)` — selection-result plumbing.
- `get_vdf_cellids_in_manual(...)` — fixed axis-aligned box selection.
- `get_vdf_cellids_in_hessian_di_box(...)` — Hessian-eigenvector-aligned X-point box (old scheme's physical X selector; current scheme uses `labeling.point_labels.get_vdf_cellids_in_b_perp_di_box` instead).

## Sample-spec planning/extraction (from `dataset_sampling.py`)

- `create_timestep_sample_specs_for_timestep(config, timestep)` — plan sample specs for one timestep.
- `create_timestep_sample_specs(...)` — build VDF sample specs: point classes + background-region fill, conflict resolution.
- `add_point_reference_metadata(...)` — attach source-or-nearest X/O distance/vector fields to sample specs.
- `create_background_region_sample_specs(...)`, `get_region_background_class_name(...)`, `get_vdf_cellids_in_region_re(...)` — background-class sample filling.
- `find_conflicting_cellids(...)` / `remove_conflicting_cellids(...)` — drop cells claimed by more than one class.
- `iter_timestep_sample_specs(sample_specs)` — extract each planned sample's VDF (raw, or Hermite spectra via `physics.vdf_transform`) + metadata.
- `create_sample_metadata_row(...)` — build one `metadata.csv` row.
- `write_timestep_samples(X, y, metadata, timestep_samples, sample_index)` — write extracted samples into output arrays.
- `iter_chunks(...)`, `get_point_class_names(...)`, `print_memory_usage(...)`, `get_memory_usage_mb(...)` — small utilities.

## Memmap-batched extraction (from `dataset_extraction.py`)

- `release_memmap_pages(...)`, `flush_and_release_memmaps(...)` — memmap cleanup.
- `get_worker_count(n_jobs, config_name)` — resolve a `n_jobs` config value to a worker count.
- `plan_dataset_sample_specs(config, timesteps, planning_n_jobs)` — plan sample specs across timesteps, serial or parallel.
- `find_first_nonempty_timestep(...)` / `extract_first_sample_from_specs(...)` — bootstrap the first sample (needed to infer array shape before allocating memmaps).
- `write_remaining_timesteps(...)` — dispatch to serial or parallel timestep writing.
- `write_timesteps_serial(...)` / `write_timesteps_parallel(...)` — the two extraction paths.
- `extract_timestep_samples_to_temp(...)`, `write_extracted_timestep(...)`, `extract_timestep_chunk_parallel(...)` — parallel-path temp-file plumbing.

## Dataset creation orchestration (from `dataset_creation.py`)

- `create_dataset(config, start_timestep, n_timesteps, dataset_kind)` — main old-pipeline entry point; no script calls it anymore (see file header above).
- `create_memmap_dataset(outdir, n_samples, sample_shape, dtype=np.float32)` — allocate the output `X.npy`/`y.npy` memmaps.

(`load_dataset`/`save_metadata`/`print_vdf_statistics` are generic and live on in `src/data_proc/dataset_io.py`, not here. `print_dataset_info` was removed -- it required a separate `y.npy`, its only caller was `inspect_dataset.py`, and both are gone now that `extract_data.py` prints VDF statistics itself right after saving.)

## Old-scheme plot overlays (originally from `plot_helpers.py`, before it merged into `src/data_proc/plot_tools.py`)

- `SOURCE_POINT_STYLES` — per-class-name marker/color styles.
- `expr_velocity(exprmaps, requestvariables=False)` — Analysator bulk-velocity expression for colormap plotting.
- `scatter_all_vdf_cells(ax, reader, boxre=None)` — scatter all VDF cells on a colormap.
- `draw_point_boxes(...)`, `draw_manual_point_search_boxes(...)`, `draw_x_point_search_areas(...)` — old-scheme search-box overlays.
- `scatter_label_points(ax, reader, metadata_rows)` — scatter labeled sample points, styled by `SOURCE_POINT_STYLES`.
- `_metadata_method_mask(...)` / `_metadata_bool_mask(...)` — private metadata-column mask helpers (small duplicate of the same-named private helpers kept in the live `src/data_proc/plot_tools.py`, intentionally not shared across the live/deprecated boundary).

(`draw_o_point_search_areas`/`get_o_point_search_flux` stay live in `src/data_proc/plot_tools.py` — see `docs/DATA_PROC.md`.)

## Old-scheme dataset plotting (from `plot_dataset.py`)

- `add_dataset_sampling_plot_config(...)` — merge dataset-creation config into a colormap plot config.
- `create_colormap_plot_jobs(...)` / `plot_labeled_colormap(...)` — per-timestep labeled colormap plots.
- `iter_vdf_plot_jobs(...)` / `run_plot_jobs(...)` — per-sample VDF plot job generation/execution.
- `plot_dataset_search_box(...)` — search-box-with-marked-cells colormap.
- `get_memmap_path(...)` / `load_memmap_array(...)` — memmap path/cache helpers.
- `create_or_load_plot_xz_slice_cache(...)`, `resolve_plot_xz_slice_cache_config(...)`, `create_plot_xz_slice_cache(...)`, `write_plot_xz_slice_cache_batch(...)`, `extract_plot_xz_slice_from_dataset(...)`, `infer_plot_xz_slice_shape(...)`, `is_plot_xz_slice_cache_valid(...)`, `save_plot_xz_slice_cache_metadata(...)`, `load_plot_xz_slice_cache_metadata(...)` — the plot xz-slice cache (mirrors `src/ml_models/feature_cache.py`'s cache, kept separate on purpose).
- `plot_vdf_sample_from_dataset(...)` — plot one saved VDF sample.
- `plot_random_class_samples(...)` — one random xz VDF sample per class in one figure.
- `plot_dataset_random_class_positions(...)` — spatial position of one random sample per class.
- `_get_cached_vdf_plot_parameters(...)`, `_get_cached_vlsv_reader(...)`, `_select_random_class_sample_indices(...)`, `_add_matching_colorbar(...)` — private plotting helpers.
