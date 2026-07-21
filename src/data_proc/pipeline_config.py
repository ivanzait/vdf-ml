# Shared parameters for the extract -> verify data_proc pipeline
# (extract_data.py, verify_data.py). Edit this file, not the scripts, to
# change which snapshot/search-box/detector settings the pipeline uses --
# one shared copy is what guarantees extraction and every verification plot
# agree on what "this run" means, instead of drifting apart across
# independently-edited PARAMETERS blocks.
#
# plot_nulls.py and plot_vdf_hermite.py intentionally do NOT import this --
# they're standalone diagnostic/exploration tools (tuning detector settings,
# picking arbitrary cells) that often want different settings than the
# current production run.

FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.vlsv"
FLUX_FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.bin"
TIMESTEP = 3408
RUN_ID = "smoke_test"

# Whether extract_data.py also builds+saves X_rotated.npy: every VDF
# rotated into its own local (B, v_perp, B x v_perp) frame, at full
# resolution (labeling.snapshot_labeling.rotate_vdfs_to_b_frame). Off by
# default -- it's a real per-sample interpolation cost (~3s/VDF on the
# smoke-test fixture's 268^3 grid) most extraction runs don't need; turn on
# when testing whether B-frame-aligned features improve blind PCA
# clustering (see run_snapshot_pca.py's PCA_CONFIG["feature_representation"]).
# Note rotation itself still RUNS (but isn't saved) whenever
# BUILD_HERMITE_DATASET is on below, since that representation is built
# from the rotated VDF -- this flag only controls whether X_rotated.npy
# also gets written out as its own array.
BUILD_ROTATED_DATASET = False

# Whether extract_data.py also builds+saves X_hermite.npy: every rotated
# VDF's log-space Hermite spectra, shape (n, order, order, order)
# (labeling.snapshot_labeling.compute_hermite_spectra_batch -- log10 first,
# floored at each cell's own sparsity threshold, then projected onto the
# Hermite basis; u/vth-normalized per sample, so unlike a raw pixel slice,
# bulk-speed/temperature differences don't dominate the representation).
# Off by default -- pulls in BUILD_ROTATED_DATASET's rotation cost as a
# prerequisite even when BUILD_ROTATED_DATASET itself is False (see above),
# plus its own smaller per-sample cost (~0.2s/VDF on the smoke-test
# fixture). See run_snapshot_pca.py's PCA_CONFIG["feature_representation"].
BUILD_HERMITE_DATASET = True
HERMITE_ORDER = 14

# VDF search box [xmin, xmax, zmin, zmax] in Re. Only VDFs inside this box
# are extracted, labeled, and plotted.
SPATIAL_BOXRE = [-30, 30, -5, 5]
REGIONS_RE = {
    "search_box": {
        "x_between": [SPATIAL_BOXRE[0], SPATIAL_BOXRE[1]],
        "z_between": [SPATIAL_BOXRE[2], SPATIAL_BOXRE[3]],
    },
}

# Physical X/O-point + current-layer detector settings.
POINTS_CONFIG = {
    "point_region_names": list(REGIONS_RE),
    "regions_re": REGIONS_RE,
    # Which point-like substance detector(s) actually run and get labeled --
    # a toggle, not a hard choice: "current_layer" (peak-|J| core, exact
    # cellid match, no flux file needed), "x_o_points" (the Hessian
    # critical-point detector below, needs FLUX_FILE_LOCATION), or both
    # together (later entries win where they overlap). Switching is a
    # one-line edit here, no code changes in extract_data.py/verify_data.py.
    # See schema.md.
    "active_point_substances": ["current_layer"],
    "x_selection": {
        "density_variable": "rho",
        # Ion diffusion region proxy: rectangular box perpendicular/parallel
        # to the local B field at the X point. Short side along B (normal to
        # the current layer), half_width_di_normal * d_i; long side
        # perpendicular to B (outflow direction), outflow_aspect_ratio times
        # wider (0.5, 10.0 -> 1 d_i normal x 10 d_i outflow).
        "half_width_di_normal": 0.5,
        "outflow_aspect_ratio": 10.0,
        "y_half_width_re": 0.0,
        "manual_re": {"x_half_width_re": 0.4, "y_half_width_re": 0.0, "z_half_width_re": 0.3},
    },
    "o_selection": {
        # "gyroradius" -> round area, radius_gyroradii * rho_i (local thermal
        # ion gyroradius, from PTensorDiagonal/rho/B at the O point).
        # "flux_contour" -> the O point's closed flux island (core_fraction
        # from O-point flux to boundary flux).
        "selection_method": "gyroradius",
        "core_fraction": 0.7,
        "radius_gyroradii": 1.0,
        "y_half_width_re": 0.0,
        "manual_re": {"x_half_width_re": 0.4, "y_half_width_re": 0.0, "z_half_width_re": 0.3},
    },
    "current_layer_selection": {
        "density_variable": "rho",
        # "Core" = grid cells with |J| >= core_fraction * peak(|J|), found
        # independently within each named box below (not one global peak --
        # the dayside magnetopause current is typically much stronger than
        # the tail current sheet, so a single global threshold would find
        # only the former). A VDF cell is labeled current_layer only if it
        # IS a core cell (exact cellid match, see find_current_layer_cellids)
        # -- no margin/expansion step any more. An earlier version expanded
        # the core by margin_di * d_i and nearest-matched to the closest VDF
        # cell, but VDF cells sit on a much coarser subgrid than d_i (see
        # find_current_layer_cellids' docstring), so that expansion never
        # actually changed which VDF cell got matched -- it was inert, and
        # could silently claim a VDF cell outside the real core. core_fraction
        # is the only real lever on how large the current_layer selection is.
        "core_fraction": 0.2,
        "search_regions_re": {
            "dayside": {"x_between": [0, SPATIAL_BOXRE[1]], "z_between": [SPATIAL_BOXRE[2], SPATIAL_BOXRE[3]]},
            "tail": {"x_between": [SPATIAL_BOXRE[0], 0], "z_between": [SPATIAL_BOXRE[2], SPATIAL_BOXRE[3]]},
        },
        # Exclude cells within min_r_re of Earth from the search entirely,
        # before any peak is computed -- field-aligned currents near the
        # inner simulation boundary are a different physical structure
        # (mapped along B to the ionosphere, not a cross-field current
        # sheet) and can otherwise compete with a sub-region's real peak
        # once core_fraction is lowered. On the fixture, real detections
        # sit at R >= 8.2 R_E; the FAC-like artifact was at R ~ 4.6 R_E.
        "min_r_re": 6.0,
    },
}

# Subsolar-anchored Shue et al. (1998)-shaped magnetopause AND bow shock.
# No margin/buffer setting here any more -- the old boundary_layer margin
# zone is superseded by current_layer's own physically-detected peak-|J|
# core (POINTS_CONFIG["current_layer_selection"] above).
MAGNETOPAUSE_CONFIG = {
    "density_variable": "rho",
    "x_scan_min_re": 5.0,
    "x_scan_max_re": 30.0,
    "n_scan_points": 300,
    # inner_magnetosphere/lobes split uses this radius instead of r0 (the
    # *dayside* standoff distance, ~8 R_E) -- r0 alone, applied as a
    # uniform-angle sphere, pulls near-Earth nightside plasma (inner
    # magnetosphere/ring current, not real lobe plasma) into "lobes". The
    # gap between r0 and lobe_r_min_re is labeled "undefined" rather than
    # arbitrarily assigned to either -- see classify_magnetosphere_regions.
    "lobe_r_min_re": 10.0,
}
POINTS_CONFIG["magnetopause"] = MAGNETOPAUSE_CONFIG

# Plot zoom box [xmin, xmax, zmin, zmax] in Re for the verification figures,
# independent of SPATIAL_BOXRE (the extraction search box).
PLOT_BOXRE = SPATIAL_BOXRE

# Seed for the "one random VDF per label" pick in verify_data.py, so its two
# plots (positions, examples) always show the same representative cells.
RANDOM_STATE = 1234

POP = "avgs"
VDFLIM = 2e6

# Shared by run_snapshot_pca.py/plot_snapshot_pca.py: blind PCA+KMeans
# clustering on an already-extracted dataset (data/snapshot_vdfs/<RUN_ID>/),
# scored against the same ground truth extract_data.py saved to metadata.csv.
PCA_CONFIG = {
    "downsample_factor": 8,
    "log_eps": 1e-30,
    "n_jobs": 1,
    "n_components": 40,
    "k_range": list(range(2, 15)),
    "random_state": RANDOM_STATE,
    "n_smallest_clusters_to_plot": 3,
    # Which saved array run_snapshot_pca.py clusters:
    # "raw" -> X.npy (log-scaled downsampled xz slice, the default),
    # "rotated" -> X_rotated.npy (same slice/downsample, but every VDF
    #   first rotated into its own local (B, v_perp, B x v_perp) frame --
    #   needs BUILD_ROTATED_DATASET = True on the extract_data.py run that
    #   produced this dataset),
    # "hermite" -> X_hermite.npy, flattened directly (no slice/downsample --
    #   it's already a compact (order, order, order) basis, not a pixel
    #   grid) -- needs BUILD_HERMITE_DATASET = True on that same run.
    # run_snapshot_pca.py raises a clear FileNotFoundError if the array the
    # chosen mode needs wasn't built.
    "feature_representation": "hermite",
    # "hermite" representation only: per-sample (row) normalization applied
    # BEFORE StandardScaler+PCA, via ml_models.features.normalize_feature_samples.
    # "none" (default) -- leaves each sample's own scale intact; "standard"
    # -- mean-centers and divides each sample's flattened spectra by its
    # own std first. Physically complex/structured populations
    # (current_layer, magnetosheath) have spectra ~6-9x larger in magnitude
    # than quieter ones at the SAME shape -- StandardScaler alone (per-
    # feature column scaling) doesn't remove that per-sample scale, so it
    # dominates variance ahead of any shape difference (see TESTING.md).
    "sample_normalization": "standard",
    "sample_norm_eps": 1e-6,
    # "hermite" representation only: concatenate density/velocity/
    # anisotropic-thermal-velocity (7 columns, see labeling.
    # snapshot_labeling.compute_moment_features_batch/
    # MOMENT_FEATURE_COLUMNS) onto the flattened Hermite spectra before
    # StandardScaler+PCA. The Hermite spectra deliberately normalize away
    # position/width/density (see vdf_to_hermite_spectra_log) to isolate
    # shape -- these moments are exactly that discarded absolute-scale
    # information, added back as explicit features. Requires
    # scripts/ml_models/rebuild_hermite_dataset.py to have been run for
    # this RUN_ID (adds the moment columns to metadata.csv); raises a
    # clear ValueError otherwise.
    "include_moment_features": True,
    # Post-StandardScaler weight multiplier on the 7 moment columns (see
    # ml_models.vdf_snapshot_clustering.fit_pca_clusters's feature_weights).
    # With only 7 moment columns against ~2700+ Hermite spectra columns,
    # their aggregate contribution to total variance is diluted by column
    # count alone even though StandardScaler already gives each one unit
    # variance -- 1.0 leaves that as-is (no boost); ~sqrt(n_hermite_features
    # / 7) would give the moments as a group roughly EQUAL total variance
    # to the entire Hermite block (~20 at HERMITE_ORDER=14, since
    # 14**3 - 7 ~= 2737 hermite features -- sqrt(2737/7) ~= 19.8).
    "moment_feature_weight": 10.0,
}
