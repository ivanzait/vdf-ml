# Physics — `src/data_proc/physics/`

Pure physical models and calculations: X/O critical-point detection,
current-layer (peak-|J| core) detection, the Shue magnetopause/bow-shock
model, and the VDF rotation + Hermite spectral transform. No dataset/
label-scheme assumptions live here. Depends on nothing else under `src/`
except the low-level VLSV-reading primitives in `src/data_proc/vdf_tools.py`
(reading B, region matching) needed to pull field data off a reader (and,
for `current_layer.py`, `point_topology.py`'s `MU0`/`compute_ion_inertial_length`
— physics submodules import from each other, see `magnetopause.py`'s import
of `compute_ion_inertial_length` for the existing precedent).

**When adding a function here**: add a one-line entry below. Keep this file
in sync — it's the fast way for a new session to know what physics already
exists before re-deriving it.

## `point_topology.py` — X/O critical-point (Hessian) detection, flux islands

- `find_point_records(reader, flux_file_location, points_config=None)` — main entry point: detect X/O points per configured region, return `(x_point_records, o_point_records)`.
- `read_smoothed_flux_grid(reader, flux_file_location, kernel_size=5)` — load and Gaussian-smooth the poloidal flux function grid from a `.bin` flux file.
- `get_contour_paths(contour)` — extract vertex paths from a matplotlib contour object.
- `find_intersection(v1, v2)` — line-segment intersection helper for contour tracing.
- `intersection_to_points(intersection)` — normalize a shapely intersection result to a point list.
- `calculate_hessian(flux_function_xz, i, j, dx)` — 2D Hessian matrix of the flux function at grid index `(i, j)`.
- `interpolate_flux(flux_function_xz, x, z, xmin, zmin, dx)` — bilinear flux value at an arbitrary `(x, z)`.
- `create_point_record(...)` — build one X/O point record dict (coord, cellid, eigvecs/di for X; flux/contour info for O).
- `add_ion_inertial_length(reader, point_record, points_config)` — attach `di_m`/`di_re`/`rho` to an X-point record.
- `compute_ion_inertial_length(number_density)` — ion inertial length `d_i` from number density.
- `add_thermal_gyroradius(reader, point_record, points_config)` — attach `rho_i_re`/`temperature_k` to an O-point record.
- `compute_thermal_gyroradius(temperature_k, b_magnitude_t)` — thermal ion gyroradius from temperature and |B|.
- `add_o_point_island_contours(...)` — attach closed-flux-island contour info to O-point records.
- `find_island_boundary_contour(...)` — the O-point flux island's closed contour at `core_fraction` between O-point and boundary flux.
- `find_smallest_closed_contour(x_array, z_array, flux_function_zx, contour_flux, point_xz)` — smallest closed contour at a given flux level containing `point_xz`; also used directly by `plot_tools.draw_o_point_search_areas` to redraw the O-point search area.
- `polygon_area(vertices)` — shoelace-formula polygon area.

## `current_layer.py` — current-density peak detection

Live consumer: `labeling.snapshot_labeling.find_current_layer_cellids`,
toggled on via `points_config["active_point_substances"]` — see `schema.md`.

- `read_b_field_grid(reader)` — dense `(z_cells, x_cells)` grid of `B`, read natively from the `.vlsv` file (no external flux file needed); mirrors `point_topology.read_smoothed_flux_grid`'s index convention.
- `compute_b_jacobian_grid(Bx_zx, By_zx, Bz_zx, dx)` — the six in-plane partial derivatives of `B` via `np.gradient`.
- `compute_current_density_grid(jacobian_grid)` — `J = curl(B) / MU0`, reduced for this 2D (y-invariant) domain; returns `(Jx_zx, Jy_zx, Jz_zx, Jmag_zx)`.
- `find_current_layer_core_records(reader, points_config, regions_re=None)` — grid points with `|J| >= core_fraction * peak(|J|)`, peak found independently within each named box in `current_layer_selection["search_regions_re"]` (e.g. dayside vs. tail, so the stronger structure doesn't swallow the threshold for the weaker one), each with its `d_i`. Points within `current_layer_selection["min_r_re"]` of Earth are excluded from every sub-region before any peak is computed, to keep field-aligned currents near the inner boundary out of the search. These `cellid`s are in the same global cellid namespace as the rest of the `.vlsv` file, so `find_current_layer_cellids` (`labeling/snapshot_labeling.py`) can match them against VDF cellids by direct set intersection -- no margin/expansion step, no nearest-cell search (an earlier version stepped out from each core cell along a per-point LMN boundary normal and nearest-matched to the closest VDF cell, removed: on this fixture VDF cells sit on a ~2.35 R_E subgrid while the margin was only ~0.01-0.1 R_E, so the "expansion" never changed which VDF cell a nearest-match picked, and could silently claim a VDF cell well outside the real core). The LMN basis builder that margin step used, `compute_local_lmn_basis`, moved to `vdf_transform.py` -- see below.

## `magnetopause.py` — subsolar-anchored Shue (1998) magnetopause + bow shock

- `find_subsolar_point(reader, x_scan_min_re=5.0, x_scan_max_re=30.0, n_scan_points=300, density_variable="rho")` — locate the subsolar magnetopause/bow-shock crossings via a density scan along +x.
- `fit_shue_model(reader, x_scan_min_re=5.0, x_scan_max_re=30.0, n_scan_points=300, density_variable="rho", alpha=SHUE_ALPHA_DEFAULT)` — fit `r0`/`r_bs`/`alpha` from the subsolar scan; returns the full shue_fit dict.
- `shue_boundary_r_re(x_re, z_re, r0_re, alpha=SHUE_ALPHA_DEFAULT)` — Shue-model boundary radius at a given angle from the subsolar point.
- `classify_magnetosphere_regions(vdf_coords_re, densities, r0_re, r_bs_re=None, alpha=SHUE_ALPHA_DEFAULT, lobe_r_min_re=None)` — classify cells into no_density_data/solar_wind/magnetosheath/inner_magnetosphere/lobes/undefined, in that fixed priority order (see `schema.md`). `lobe_r_min_re` (default `r0_re`) is the inner_magnetosphere/lobes split radius -- set larger than `r0` (a dayside-only standoff distance) to keep near-Earth nightside plasma out of `lobes`; the gap between `r0` and `lobe_r_min_re` (and any other unclassified cell) is labeled `undefined` rather than assigned to either. No `boundary_layer`/`margin_di` any more -- superseded by `current_layer`'s physically-detected peak-`|J|` core (`physics/current_layer.py`).

## `vdf_transform.py` — rotation into `(B, v_perp, B×v_perp)` frame + Hermite spectra

Live consumers: `plot_tools.py` (ad-hoc VDF/Hermite plots), `src/ml_models/coordinate_prediction.py` (CNN prediction feature pipeline). `src/deprecated.py` also depends on this (old pipeline's optional Hermite path).

- `unit_vector(vector)` — normalize a vector; raises on zero vector.
- `build_rotation_matrix(b_field, bulk_velocity)` — 3×3 orthonormal rotation matrix with rows `(b_hat, v_perp_hat, b_hat × v_perp_hat)`.
- `compute_local_lmn_basis(dBx_dx, dBx_dz, dBy_dx, dBy_dz, dBz_dx, dBz_dz, j_vector)` — LMN boundary-normal basis at one point via Minimum Gradient Analysis / Minimum Directional Derivative Analysis on the Jacobian `G = grad(B)` (Alho et al. 2024, *Ann. Geophys.* 42, 145, https://doi.org/10.5194/angeo-42-145-2024); returns `(l_hat, m_hat, n_hat)` or `None`. Moved here from `current_layer.py`, which used it for a margin-expansion step `find_current_layer_cellids` no longer does (see above) -- currently has no caller, kept as a future alternative rotation frame (an (L, M, N)-aligned option alongside `build_rotation_matrix`'s `(B, v_perp, B×v_perp)`).
- `rotate_bounds(v_limits, rotation_matrix)` — axis-aligned bounding box of a rotated velocity-space cuboid.
- `compute_new_shape(v_limits, shape, new_v_limits)` — rotated-frame grid shape preserving the original cell spacing.
- `get_rotated_vdf(vdf, shape, v_limits, b_field, bulk_velocity, new_v_limits=None, new_shape=None)` — trilinearly interpolate a VDF into the rotated frame. By default the output bounding box/shape is auto-computed from `b_field`'s orientation (`rotate_bounds`/`compute_new_shape`) and varies per call -- fine for one VDF at a time (e.g. before `vdf_to_hermite_spectra`, whose output size depends only on Hermite order). Pass `new_v_limits`/`new_shape` explicitly to force a fixed output grid instead, needed when rotating many VDFs that must all come out the same shape (see `labeling.snapshot_labeling.rotate_vdfs_to_b_frame`). Returns `(rotated_vdf, new_shape, new_v_limits, rotation_matrix)`.
- `hermite_polynomials(v_axis, order)` — Gaussian-weighted physicists' Hermite polynomials, orders `0..order-1`.
- `normalized_hermite_basis(shape, v_limits, order, vth, u, axis)` — normalized Hermite basis along one velocity axis.
- `compute_drift_velocity(vdf, shape, v_limits)` — bulk drift velocity `[ux, uy, uz]` of a dense VDF.
- `compute_thermal_velocity(vdf, shape, v_limits, u)` — isotropic thermal velocity around drift velocity `u`.
- `compute_hermite_spectra(vdf, shape, v_limits, order, vth, u)` — project a VDF onto the separable Hermite basis.
- `vdf_to_hermite_spectra(vdf, shape, v_limits, order=DEFAULT_HERMITE_ORDER)` — convenience wrapper: compute drift/thermal velocity, then the Hermite spectra of the raw (linear-scale) VDF.
- `vdf_to_hermite_spectra_log(vdf, shape, v_limits, sparsity_threshold, order=DEFAULT_HERMITE_ORDER)` — like `vdf_to_hermite_spectra`, but projects `log10(vdf)` instead of `vdf` itself (cells at/below `sparsity_threshold` -- the VDF's own "MinValue" sparsity floor, see `vdf_tools.get_vdf_plot_threshold` -- get `log10(sparsity_threshold)` instead of `log10(0)`). `u`/`vth` (which set the basis functions' center/width) are still computed from the raw linear `vdf` -- those are genuine physical moments, not something a log-density-weighted average would mean anything for. Live consumer: `labeling.snapshot_labeling.compute_hermite_spectra_batch` (see `docs/LABELING.md`).
