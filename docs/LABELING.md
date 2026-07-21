# Labeling — `src/data_proc/labeling/`

Modular ground-truth/label-assignment subsystem for the current (physics-driven)
label scheme: turns detected point substances (see `schema.md`) + Shue-model
magnetosphere regions into per-VDF-cell labels. Used by
`scripts/data_proc/extract_data.py`/`verify_data.py` (main production path).
`scripts/ml_models/run_snapshot_pca.py` (blind PCA/KMeans clustering) scores
against this ground truth too, but reads it back out of `extract_data.py`'s
saved `metadata.csv` rather than recomputing it -- `plot_snapshot_pca.py`
still calls `pick_cluster_representative_cellids` from this module directly,
for its one-VDF-per-blind-cluster plot.

**When adding a function here**: add a one-line entry below.

## `point_labels.py` — VDF-cell selection around one detected X/O point

- `get_point_selection_config(config, point_kind)` — return the `x_selection`/`o_selection` config block for one point kind.
- `get_manual_config_re(config, point_kind)` — manual point-box half-widths in Re.
- `compute_b_perp_di_box_geometry(reader, point_record, half_width_di_normal, outflow_aspect_ratio)` — shared `(b_hat, perp_hat, half_width_normal_m, half_width_outflow_m)` geometry, factored out so plotting code (`plot_tools.plot_snapshot_topology`) draws exactly the box that was actually selected.
- `get_vdf_cellids_in_b_perp_di_box(reader, config, point_record, vdf_cellids, vdf_coords_re)` — X-point ion-diffusion-region proxy box, aligned perpendicular/parallel to local B.
- `get_vdf_cellids_in_flux_contour(config, point_record, vdf_cellids, vdf_coords_re)` — O-point closed-flux-island selection.
- `get_vdf_cellids_in_gyroradius_circle(config, point_record, vdf_cellids, vdf_coords_re)` — O-point round-area selection sized by local thermal ion gyroradius.
- `get_o_point_cellids_by_method(config, point_record, vdf_cellids, vdf_coords_re)` — dispatches to gyroradius-circle or flux-contour selection based on `o_selection.selection_method`.
- `create_point_sample_metadata(config, point_record)` — metadata fields describing a point-selected sample (source point, selection box, hessian/flux info).
- `optional_float(value)` — `float(value)`, or `nan` if `value` is `None`.

## `snapshot_labeling.py` — snapshot-wide VDF loading + ground-truth labeling

- `get_plasma_vdf_cells(reader, regions_re=None, density_variable="rho", pop="avgs")` — VDF-carrying cells with actual plasma (density > 0), optionally box-restricted; the shared filter every function below goes through.
- `load_snapshot_vdfs(reader, regions_re=None, density_variable="rho", pop="avgs")` — extract every plasma-carrying VDF in a snapshot. Returns `(cellids, coords_re, X)`.
- `find_ground_truth_point_cellids(reader, flux_file_location, points_config, regions_re=None)` — physically-detected X/O point cell IDs for one snapshot (the `x_o_points` substance family).
- `find_current_layer_cellids(reader, points_config, regions_re=None)` — physically-detected current-layer cell IDs: a VDF cell is `current_layer` only if its own cellid IS a peak-`|J|` core cell (`src.data_proc.physics.current_layer`) -- exact cellid intersection, no margin/expansion or nearest-cell search. A peer to `find_ground_truth_point_cellids`, not a replacement -- see `schema.md`.
- `find_magnetosphere_region_cellids(reader, points_config, regions_re=None)` — Shue-model region cell IDs (solar_wind/magnetosheath/inner_magnetosphere/lobes/no_density_data/undefined, see `schema.md` for the priority order).
- `combine_ground_truth_labels(vdf_cellids, region_labels, point_substance_cellids_by_label)` — merge region labels with whichever point substance(s) a caller computed into one label per cell (point substances take priority, in dict order). Generic over how many/which substances exist -- see `schema.md`.
- `compute_snapshot_ground_truth(reader, points_config, regions_re=None, flux_file_location=None)` — composes whichever of `find_ground_truth_point_cellids`/`find_current_layer_cellids` are active (`points_config["active_point_substances"]`) + `find_magnetosphere_region_cellids` + `combine_ground_truth_labels` into the full ground-truth pipeline (no VDF-array extraction). Keeps its own copy of the same toggle blocks `extract_data.py` has, so `verify_data.py` can't drift out of sync with it. Returns `(cellids, coords_re, labels, shue_fit, point_substance_records)`.
- `pick_cluster_representative_cellids(vdf_cellids, labels, random_state=None)` — one random representative cell per unique label.
- `rotate_vdfs_to_b_frame(X, b_field, bulk_velocity, velocity_mesh_extent)` — rotate every VDF in `X` into its own local `(B, v_perp, B x v_perp)` frame (`physics.vdf_transform.get_rotated_vdf`), at full resolution, onto a fixed output grid (same shape/extent every sample already shares) so the batch stays one stackable array. Falls back to the unrotated VDF for the rare case `build_rotation_matrix` can't build a frame (bulk `V` exactly parallel to `B`). Meant to run once at extraction time (`extract_data.py`, gated by `pipeline_config.BUILD_ROTATED_DATASET`, or as a prerequisite of `BUILD_HERMITE_DATASET` -- see below), not repeated by every downstream PCA/ML run. Returns `(X_rotated, n_fallback)`.
- `compute_hermite_spectra_batch(X, velocity_mesh_extent, sparsity_threshold, order=DEFAULT_HERMITE_ORDER)` — convert every VDF in `X` to the Hermite spectra of its `log10` (`physics.vdf_transform.vdf_to_hermite_spectra_log`). `X` should be `rotate_vdfs_to_b_frame`'s output (`extract_data.py` always passes the rotated array) -- combines B-frame rotation (removes local field orientation as a confound) with the log10 step (keeps a VDF's wings/tails from being swamped by its peak, same reason `ml_models.features` log-scales the raw-pixel representation) and u/vth normalization (drift velocity/thermal speed factored out, not left as pixel position/spread). `sparsity_threshold`: each sample's own "MinValue" (see `vdf_tools.get_vdf_plot_threshold`), passed through per sample. Gated by `pipeline_config.BUILD_HERMITE_DATASET`. Returns `X_hermite`, shape `(len(X), order, order, order)`.
- `save_labeled_vdfs(output_dir, cellids, coords_re, X, labels, timestep, file_location, b_field, bulk_velocity, velocity_mesh_extent, sparsity_threshold, X_rotated=None, X_hermite=None)` — save `X.npy` + `metadata.csv` (now including per-cell `bx_t/by_t/bz_t`, `vx_ms/vy_ms/vz_ms`, `min_value`, and the snapshot-constant `vspace_*min_ms`/`vspace_*max_ms` velocity-mesh extent -- saved regardless of `X_rotated`/`X_hermite`, so a downstream consumer can always rebuild either representation itself if wanted). If `X_rotated` is given (see `rotate_vdfs_to_b_frame` above), also saves it as a second array, `X_rotated.npy`. If `X_hermite` is given (see `compute_hermite_spectra_batch` above), also saves it as a third array, `X_hermite.npy`. Neither is merged into `X.npy` itself, so consumers that want the real simulation-frame VDF (CNN training, `verify_data.py`'s example plots) are unaffected.
