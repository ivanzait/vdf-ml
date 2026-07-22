"""
Plotting library for VLSV snapshots, VDF/Hermite diagnostics, and the
cluster-plotting pipeline shared by three scripts, each scoring a different
label source against the same three plot functions:
  1. search area with every VDF cell marked -- plot_colormap_with_vdf_markers
  2. one representative VDF per cluster, mapped spatially -- plot_cluster_vdf_positions
  3. example VDFs for each cluster -- plot_cluster_vdf_examples

plot_combined_clusters (in the same section as step 2/3 above) also backs
scripts/data_proc/extract_data.py's post-extraction overview plot: every
VDF cell colored by that script's own ground truth, with Shue-model
boundaries drawn on top. Its steps-2/3 counterpart,
plot_cluster_vdf_positions/plot_cluster_vdf_examples, backs
scripts/data_proc/verify_data.py the same way (real extraction ground
truth) and scripts/ml_models/plot_snapshot_pca.py the same way again (blind
PCA/KMeans clusters instead of ground truth).

Sections, in file order: ad-hoc exploration, single-VDF xz-slice plotting,
region/box helpers, O-point search-area overlay, topology diagnostics,
snapshot colormap + Shue-boundary drawing, and the cluster-plotting pipeline
above.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon

from src.data_proc.config import create_timestep_path
from src.data_proc.labeling.point_labels import (
    compute_b_perp_di_box_geometry,
    create_point_sample_metadata,
    get_o_point_cellids_by_method,
    get_vdf_cellids_in_b_perp_di_box,
)
from src.data_proc.physics.magnetopause import shue_boundary_r_re
from src.data_proc.physics.point_topology import (
    find_point_records,
    find_smallest_closed_contour,
    read_smoothed_flux_grid,
)
from src.data_proc.physics.vdf_transform import (
    DEFAULT_HERMITE_ORDER,
    get_rotated_vdf,
    vdf_to_hermite_spectra,
)
from src.data_proc.vdf_tools import (
    R_EARTH,
    VdfExtractor,
    create_xz_slice,
    get_b_field,
    get_bulk_velocity,
    get_nearest_vdf_cellid,
    get_region_axis_bounds_re,
    get_vdf_cells_with_coords_re,
    get_vdf_plot_axes_parameters,
    get_vdf_plot_parameters,
)

COLORMAP_MARKER_KEYS = (
    "show_all_vdf_cells",
    "all_points_style",
    "selected_points_style",
)

DEFAULT_ALL_POINTS_STYLE = {
    "marker": "o",
    "s": 6,
    "color": "black",
    "alpha": 0.6,
}

DEFAULT_SELECTED_POINTS_STYLE = {
    "marker": "*",
    "s": 300,
    "edgecolor": "black",
    "facecolor": "none",
    "linewidth": 1.5,
}


# =====================================================================
# Ad-hoc exploration -- used by scripts/data_proc/plot_vdf_hermite.py
# =====================================================================

def select_vdf_points(
    cellids,
    coords_re,
    x_range=None,
    y_range=None,
    z_range=None,
    points_re=None,
):
    """Select VDF-carrying cells by spatial box, or by nearest-cell match to explicit points_re coordinates."""

    coords_re = np.asarray(coords_re, dtype=float)

    if points_re:
        nearest_cellids = {
            get_nearest_vdf_cellid(point, cellids, coords_re)
            for point in points_re
        }
        indices = np.sort(np.nonzero(np.isin(cellids, list(nearest_cellids)))[0])
    else:
        mask = np.ones(len(coords_re), dtype=bool)
        for axis_index, axis_range in enumerate((x_range, y_range, z_range)):
            if axis_range is None:
                continue
            axis_min, axis_max = axis_range
            mask &= (coords_re[:, axis_index] >= axis_min) & (
                coords_re[:, axis_index] <= axis_max
            )
        indices = np.nonzero(mask)[0]

    return cellids[indices], coords_re[indices]


def plot_colormap_with_vdf_markers(
    file_location,
    colormap_config,
    all_coords_re=None,
    selected_coords_re=None,
    output_path=None,
    figsize=(12, 12),
):
    """Plot an Analysator 2D colormap with all/selected VDF-cell markers overlaid; returns the created figure."""

    import analysator as pt

    plot_kwargs = {
        key: value
        for key, value in colormap_config.items()
        if key not in COLORMAP_MARKER_KEYS
    }

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot([0.1, 0.1, 0.8, 0.8])

    pt.plot.plot_colormap(filename=str(file_location), axes=ax, **plot_kwargs)

    if colormap_config.get("show_all_vdf_cells", True) and all_coords_re is not None and len(all_coords_re):
        style = colormap_config.get("all_points_style", DEFAULT_ALL_POINTS_STYLE)
        ax.scatter(all_coords_re[:, 0], all_coords_re[:, 2], **style)

    if selected_coords_re is not None and len(selected_coords_re):
        style = colormap_config.get("selected_points_style", DEFAULT_SELECTED_POINTS_STYLE)
        ax.scatter(selected_coords_re[:, 0], selected_coords_re[:, 2], **style)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_vdf_and_hermite_grid(
    reader,
    cellids,
    coords_re,
    order=DEFAULT_HERMITE_ORDER,
    pop="avgs",
    vdf_cmap="viridis",
    hermite_cmap="RdBu_r",
    output_path=None,
):
    """Plot side-by-side VDF xz-slices and Hermite-spectra slices, one row per cell."""

    cellids = np.asarray(cellids)
    coords_re = np.asarray(coords_re, dtype=float)
    n_points = len(cellids)
    if n_points == 0:
        raise ValueError("No VDF points were selected for the detail plot")

    extractor = VdfExtractor(reader=reader, pop=pop)

    fig, axes = plt.subplots(n_points, 2, figsize=(11, 5 * n_points))
    axes = np.atleast_2d(axes)

    for row, (cid, coord) in enumerate(zip(cellids, coords_re)):
        vdf = extractor.extract(cid=int(cid), box=-1)
        v_limits, _dv = get_vdf_plot_axes_parameters(reader=reader, vdf_shape=vdf.shape)
        spectra = vdf_to_hermite_spectra(vdf=vdf, shape=vdf.shape, v_limits=v_limits, order=order)

        vdf_slice = create_xz_slice(vdf)
        vdf_slice_plot = np.where(vdf_slice > 0, vdf_slice, np.nan)
        extent_km = np.asarray(v_limits) / 1000.0

        ax_vdf = axes[row, 0]
        im0 = ax_vdf.imshow(
            vdf_slice_plot.T,
            origin="lower",
            extent=[extent_km[0], extent_km[3], extent_km[2], extent_km[5]],
            norm=LogNorm(),
            cmap=vdf_cmap,
        )
        ax_vdf.set_title(
            f"VDF xz-slice  cid={int(cid)}  coord_re={np.round(coord, 2).tolist()}"
        )
        ax_vdf.set_xlabel("vx [km/s]")
        ax_vdf.set_ylabel("vz [km/s]")
        fig.colorbar(im0, ax=ax_vdf, label="f(v)")

        hermite_slice = spectra[:, 0, :]
        vmax = float(np.abs(hermite_slice).max()) or 1.0
        ax_h = axes[row, 1]
        im1 = ax_h.imshow(
            hermite_slice.T,
            origin="lower",
            cmap=hermite_cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        ax_h.set_title(f"Hermite spectra  ny=0 slice  order={order}")
        ax_h.set_xlabel("nx")
        ax_h.set_ylabel("nz")
        fig.colorbar(im1, ax=ax_h, label="coefficient")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_vdf_rotation_comparison(
    reader,
    cellids,
    coords_re,
    pop="avgs",
    vdf_cmap="viridis",
    output_path=None,
):
    """Plot VDF xz-slices before vs. after (B, v_perp, B x v_perp) rotation, one row per cell -- density should barely change; a large shift signals a rotation bug."""

    cellids = np.asarray(cellids)
    coords_re = np.asarray(coords_re, dtype=float)
    n_points = len(cellids)
    if n_points == 0:
        raise ValueError("No VDF points were selected for the rotation comparison")

    extractor = VdfExtractor(reader=reader, pop=pop)

    fig, axes = plt.subplots(n_points, 2, figsize=(11, 5 * n_points))
    axes = np.atleast_2d(axes)

    for row, (cid, coord) in enumerate(zip(cellids, coords_re)):
        vdf = extractor.extract(cid=int(cid), box=-1)
        v_limits, _dv = get_vdf_plot_axes_parameters(reader=reader, vdf_shape=vdf.shape)

        b_field = get_b_field(reader=reader, cid=int(cid))
        bulk_velocity = get_bulk_velocity(reader=reader, cid=int(cid))
        angle_deg = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(b_field, bulk_velocity)
                    / (np.linalg.norm(b_field) * np.linalg.norm(bulk_velocity)),
                    -1.0,
                    1.0,
                )
            )
        )

        rotated_vdf, new_shape, new_v_limits, _rotation_matrix = get_rotated_vdf(
            vdf=vdf,
            shape=vdf.shape,
            v_limits=v_limits,
            b_field=b_field,
            bulk_velocity=bulk_velocity,
        )
        density_change_pct = 100.0 * (rotated_vdf.sum() - vdf.sum()) / vdf.sum()

        original_slice = create_xz_slice(vdf)
        original_slice_plot = np.where(original_slice > 0, original_slice, np.nan)
        extent_km_original = np.asarray(v_limits) / 1000.0

        ax_original = axes[row, 0]
        im0 = ax_original.imshow(
            original_slice_plot.T,
            origin="lower",
            extent=[
                extent_km_original[0],
                extent_km_original[3],
                extent_km_original[2],
                extent_km_original[5],
            ],
            norm=LogNorm(),
            cmap=vdf_cmap,
        )
        ax_original.set_title(
            f"Original xz-slice  cid={int(cid)}  coord_re={np.round(coord, 2).tolist()}\n"
            f"B={np.round(b_field * 1e9, 2).tolist()} nT  "
            f"V={np.round(bulk_velocity / 1000.0, 1).tolist()} km/s  "
            f"angle(B,V)={angle_deg:.1f} deg"
        )
        ax_original.set_xlabel("vx [km/s]")
        ax_original.set_ylabel("vz [km/s]")
        fig.colorbar(im0, ax=ax_original, label="f(v)")

        mid_perp = new_shape[1] // 2
        rotated_slice = rotated_vdf[:, mid_perp, :]
        rotated_slice_plot = np.where(rotated_slice > 0, rotated_slice, np.nan)
        extent_km_rotated = np.asarray(new_v_limits) / 1000.0

        ax_rotated = axes[row, 1]
        im1 = ax_rotated.imshow(
            rotated_slice_plot.T,
            origin="lower",
            extent=[
                extent_km_rotated[0],
                extent_km_rotated[3],
                extent_km_rotated[2],
                extent_km_rotated[5],
            ],
            norm=LogNorm(),
            cmap=vdf_cmap,
        )
        ax_rotated.set_title(
            f"Rotated slice (fixed v_perp)  shape={new_shape}\n"
            f"density change from rotation: {density_change_pct:+.2f}%"
        )
        ax_rotated.set_xlabel("v_parallel (B) [km/s]")
        ax_rotated.set_ylabel("v_perp2 (B x v_perp) [km/s]")
        fig.colorbar(im1, ax=ax_rotated, label="f(v)")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# =====================================================================
# Single-VDF xz-slice plotting -- shared by ad-hoc scripts, the old
# plot_dataset pipeline (src/deprecated.py), and model training/prediction
# failure-case plots
# =====================================================================

def extract_plot_xz_slice(vdf):
    """Plot-oriented middle xz slice from a dense VDF with shape (vx, vy, vz). Returns shape (vz, vx)."""

    return np.asarray(create_xz_slice(vdf).T, dtype=np.float32)


def prepare_physical_xz_plot(vdf_xz_slice, metadata_row, dv, threshold):
    """Threshold and scale a plot-oriented xz slice by dv. Returns a masked array, or None if empty."""

    vdf_plot_raw = np.asarray(vdf_xz_slice, dtype=np.float32) * dv
    vdf_plot = np.where(vdf_plot_raw < threshold * dv, 0, vdf_plot_raw)
    vdf_plot = np.ma.masked_less_equal(vdf_plot, 0)

    if vdf_plot.count() == 0:
        print(
            "Using unthresholded VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        vdf_plot = np.ma.masked_less_equal(vdf_plot_raw, 0)

    if vdf_plot.count() == 0:
        print(
            "Skipping empty VDF plot: "
            f"timestep={metadata_row.get('timestep', 'unknown')}, "
            f"cid={metadata_row.get('cid', 'unknown')}, "
            f"class={metadata_row.get('class_name', 'unknown')}"
        )
        return None

    return vdf_plot


def prepare_vdf_xz_plot(vdf, metadata_row, dv, threshold):
    """Threshold and scale the middle xz slice of a dense VDF. Returns a masked array, or None if empty."""

    return prepare_physical_xz_plot(
        vdf_xz_slice=extract_plot_xz_slice(vdf),
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )


def plot_prepared_vdf_xz_slice_on_axis(
        ax,
        vdf_plot,
        y_label,
        metadata_row,
        extent,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
        sample_index=None,
        title=None,
):
    """Draw a prepared (thresholded) xz VDF slice on an existing matplotlib axis. Returns the AxesImage."""

    extent = np.asarray(extent)

    vxmin = extent[0]
    vzmin = extent[2]
    vxmax = extent[3]
    vzmax = extent[5]

    im = ax.imshow(
        vdf_plot,
        origin="lower",
        extent=[
        vxmin / 1000, vxmax / 1000,
        vzmin / 1000, vzmax / 1000
    ],

        norm="log",
        cmap="nipy_spectral",
    )

    ax.grid(color="gray", axis="both")
    ax.set_xlim(-vdflim / 1000, vdflim / 1000)
    ax.set_ylim(-vdflim / 1000, vdflim / 1000)
    ax.set_xlabel("v_x")
    ax.set_ylabel("v_z")

    if title is not None:
        ax.set_title(title)
        return im

    title_parts = [
        f"timestep={metadata_row.get('timestep', 'unknown')}",
        f"cid={metadata_row.get('cid', 'unknown')}",
    ]

    if sample_index is not None:
        title_parts.insert(0, f"sample={int(sample_index)}")

    class_name = metadata_row.get("class_name")
    if class_name is not None:
        title_parts.append(f"class={class_name}")

    if y_label is not None:
        title_parts.append(f"true={int(y_label)}")

    if predicted_class_name is not None:
        title_parts.append(f"pred={predicted_class_name}")

    if decision_score is not None:
        title_parts.append(f"score={decision_score:.3g}")

    ax.set_title(", ".join(title_parts))

    return im


def plot_vdf_xz_slice_on_axis(
        ax,
        vdf,
        y_label,
        metadata_row,
        extent,
        dv,
        threshold,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
        sample_index=None,
):
    """Draw one xz VDF slice (from a dense VDF) on an existing matplotlib axis. Returns the AxesImage, or None if empty."""

    vdf_plot = prepare_vdf_xz_plot(
        vdf=vdf,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )

    if vdf_plot is None:
        return None

    return plot_prepared_vdf_xz_slice_on_axis(
        ax=ax,
        vdf_plot=vdf_plot,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        vdflim=vdflim,
        decision_score=decision_score,
        predicted_class_name=predicted_class_name,
        sample_index=sample_index,
    )


def plot_physical_xz_slice_on_axis(
        ax,
        vdf_xz_slice,
        y_label,
        metadata_row,
        extent,
        dv,
        threshold,
        vdflim=2e6,
):
    """Draw one cached plot-oriented xz VDF slice on an existing matplotlib axis. Returns the AxesImage, or None if empty."""

    vdf_plot = prepare_physical_xz_plot(
        vdf_xz_slice=vdf_xz_slice,
        metadata_row=metadata_row,
        dv=dv,
        threshold=threshold,
    )

    if vdf_plot is None:
        return None

    return plot_prepared_vdf_xz_slice_on_axis(
        ax=ax,
        vdf_plot=vdf_plot,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        vdflim=vdflim,
    )


def plot_vdf_xz_slice(
        vdf,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
        decision_score=None,
        predicted_class_name=None,
):
    """Plot and save an xz VDF slice from one dense VDF sample."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 6))
    im = plot_vdf_xz_slice_on_axis(
        ax=ax1,
        vdf=vdf,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
        decision_score=decision_score,
        predicted_class_name=predicted_class_name,
    )

    if im is None:
        plt.close(fig)
        return

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_vdf_xz_slice_from_physical_xz(
        vdf_xz_slice,
        y_label,
        metadata_row,
        extent,
        output_path,
        dv,
        threshold,
        vdflim=2e6,
):
    """Plot and save one cached plot-oriented xz VDF slice."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 6))
    im = plot_physical_xz_slice_on_axis(
        ax=ax1,
        vdf_xz_slice=vdf_xz_slice,
        y_label=y_label,
        metadata_row=metadata_row,
        extent=extent,
        dv=dv,
        threshold=threshold,
        vdflim=vdflim,
    )

    if im is None:
        plt.close(fig)
        return

    fig.colorbar(im, ax=ax1, label="f(v)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Region/box helpers
# =====================================================================

def get_regions_re_boxre(regions_re, margin_re=2.0):
    """xz plot box [xmin, xmax, zmin, zmax] covering the union of regions_re boxes, with a margin. None if regions_re is empty."""

    if not regions_re:
        return None

    x_lower, x_upper, z_lower, z_upper = [], [], [], []
    for region_re in regions_re.values():
        lower, upper = get_region_axis_bounds_re(region_re, "x")
        if lower is not None:
            x_lower.append(lower)
        if upper is not None:
            x_upper.append(upper)
        lower, upper = get_region_axis_bounds_re(region_re, "z")
        if lower is not None:
            z_lower.append(lower)
        if upper is not None:
            z_upper.append(upper)

    if not (x_lower and x_upper and z_lower and z_upper):
        return None

    return [
        min(x_lower) - margin_re, max(x_upper) + margin_re,
        min(z_lower) - margin_re, max(z_upper) + margin_re,
    ]


# =====================================================================
# O-point search-area overlay
# =====================================================================

def draw_o_point_search_areas(ax, reader, metadata_rows, flux_file_template, o_core_fraction=None):
    """Draw closed O-point flux contours used for VDF-cell selection."""

    required_columns = {
        "point_kind",
        "source_point_x_re",
        "source_point_z_re",
        "source_point_flux",
        "boundary_flux",
        "timestep",
    }
    if not flux_file_template or not required_columns.issubset(metadata_rows.columns):
        return

    o_point_rows = metadata_rows[metadata_rows["point_kind"] == "o"]
    if (
        "selection_method" in o_point_rows.columns
        or "plot_selection_method" in o_point_rows.columns
        or "physical_selected" in o_point_rows.columns
    ):
        o_point_rows = o_point_rows[
            _metadata_method_mask(
                metadata_rows=o_point_rows,
                column="selection_method",
                method="physical",
            )
            | _metadata_method_mask(
                metadata_rows=o_point_rows,
                column="plot_selection_method",
                method="physical",
            )
            | _metadata_bool_mask(
                metadata_rows=o_point_rows,
                column="physical_selected",
            )
        ]
    if o_point_rows.empty:
        return

    contour_key_columns = ["source_point_x_re", "source_point_z_re", "boundary_flux"]
    if "search_flux" in o_point_rows.columns:
        contour_key_columns.append("search_flux")
    contour_rows = o_point_rows.drop_duplicates(contour_key_columns)

    timestep = int(contour_rows.iloc[0]["timestep"])
    flux_file_location = create_timestep_path(
        path_template=flux_file_template,
        timestep=timestep,
    )
    x_array_m, z_array_m, flux_function_zx = read_smoothed_flux_grid(
        reader=reader,
        flux_file_location=flux_file_location,
    )
    x_array_re = x_array_m / R_EARTH
    z_array_re = z_array_m / R_EARTH

    for contour_index, (_, row) in enumerate(contour_rows.iterrows()):
        search_flux = get_o_point_search_flux(
            row=row,
            o_core_fraction=o_core_fraction,
        )
        contour = find_smallest_closed_contour(
            x_array=x_array_re,
            z_array=z_array_re,
            flux_function_zx=flux_function_zx,
            contour_flux=search_flux,
            point_xz=(
                float(row["source_point_x_re"]),
                float(row["source_point_z_re"]),
            ),
        )
        if contour is None:
            continue

        _, vertices_re = contour

        label = "O search area" if contour_index == 0 else None
        polygon = Polygon(
            vertices_re,
            closed=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.18,
            linewidth=1.5,
            label=label,
            zorder=2,
        )
        ax.add_patch(polygon)


def get_o_point_search_flux(row, o_core_fraction=None):
    """Return O-point contour flux from metadata search_flux, or a fallback mirroring find_island_boundary_contour's interpolation for older metadata rows lacking that column."""

    if "search_flux" in row:
        try:
            search_flux = float(row["search_flux"])
        except (TypeError, ValueError):
            search_flux = float("nan")

        if not np.isnan(search_flux):
            return search_flux

    if o_core_fraction is None:
        o_core_fraction = 1.0

    o_core_fraction = min(1.0, max(0.0, float(o_core_fraction)))
    source_flux = float(row["source_point_flux"])
    boundary_flux = float(row["boundary_flux"])

    return source_flux + o_core_fraction * (boundary_flux - source_flux)


def _metadata_method_mask(metadata_rows, column, method):
    """Boolean mask for metadata rows using a named selection method."""

    if column not in metadata_rows.columns:
        return np.zeros(len(metadata_rows), dtype=bool)

    return (
        metadata_rows[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
        == str(method).lower()
    )


def _metadata_bool_mask(metadata_rows, column):
    """Boolean mask from a metadata boolean-like column."""

    if column not in metadata_rows.columns:
        return np.zeros(len(metadata_rows), dtype=bool)

    values = metadata_rows[column].fillna(False)
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)

    return (
        values.astype(str)
        .str.lower()
        .isin({"1", "true", "yes"})
        .to_numpy(dtype=bool)
    )


# =====================================================================
# Topology diagnostics
# =====================================================================

def plot_snapshot_topology(
    file_location,
    flux_file_location,
    points_config,
    output_path=None,
    colormap_config=None,
    figsize=(10, 9),
    show_x_points=True,
    show_o_points=True,
    plot_boxre=None,
):
    """Colormap of one snapshot with detected X/O points, their physical search boxes/contours, and d_i/rho_i annotated -- the sanity check for a labeling/topology change; run before trusting ground truth built elsewhere."""

    import analysator as pt

    file_location = str(file_location)
    reader = pt.vlsvfile.VlsvReader(file_location)

    x_point_records, o_point_records = find_point_records(
        reader=reader,
        flux_file_location=flux_file_location,
        points_config=points_config,
    )
    if not show_x_points:
        x_point_records = []
    if not show_o_points:
        o_point_records = []
    vdf_cellids, vdf_coords_re = get_vdf_cells_with_coords_re(reader)

    metadata_rows = []
    for point_record in x_point_records + o_point_records:
        row = create_point_sample_metadata(
            config=points_config,
            point_record=point_record,
        )
        if point_record["point_kind"] == "x":
            cellids_by_position = get_vdf_cellids_in_b_perp_di_box(
                reader=reader,
                config=points_config,
                point_record=point_record,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
        else:
            cellids_by_position = get_o_point_cellids_by_method(
                config=points_config,
                point_record=point_record,
                vdf_cellids=vdf_cellids,
                vdf_coords_re=vdf_coords_re,
            )
        row["timestep"] = 0
        row["plot_selection_method"] = "physical"
        row["physical_selected"] = True
        matched_cid = next(iter(cellids_by_position.values()), None)
        if matched_cid is not None:
            row["cid"] = int(matched_cid)
        metadata_rows.append(row)

    metadata_rows = (
        pd.DataFrame(metadata_rows)
        if metadata_rows
        else pd.DataFrame(columns=["point_kind", "source_point_x_re", "source_point_z_re"])
    )

    n_x, n_o = len(x_point_records), len(o_point_records)
    n_x_matched = (
        metadata_rows.loc[metadata_rows["point_kind"] == "x", "cid"].notna().sum()
        if "cid" in metadata_rows else 0
    )
    n_o_matched = (
        metadata_rows.loc[metadata_rows["point_kind"] == "o", "cid"].notna().sum()
        if "cid" in metadata_rows else 0
    )
    if show_x_points:
        print(
            f"X points: {n_x} detected, {n_x_matched} matched a VDF cell "
            f"(half_width_di_normal={points_config.get('x_selection', {}).get('half_width_di_normal')}, "
            f"outflow_aspect_ratio={points_config.get('x_selection', {}).get('outflow_aspect_ratio', 10.0)})"
        )
    else:
        print("X points: skipped (show_x_points=False)")
    o_selection_method = points_config.get("o_selection", {}).get("selection_method", "gyroradius")
    if show_o_points:
        if o_selection_method == "gyroradius":
            print(
                f"O points: {n_o} detected, {n_o_matched} matched a VDF cell "
                f"(selection_method=gyroradius, radius_gyroradii={points_config.get('o_selection', {}).get('radius_gyroradii', 1.0)})"
            )
        else:
            print(
                f"O points: {n_o} detected, {n_o_matched} matched a VDF cell "
                f"(selection_method=flux_contour, core_fraction={points_config.get('o_selection', {}).get('core_fraction')})"
            )
    else:
        print("O points: skipped (show_o_points=False)")

    colormap_config = dict(colormap_config or {"var": "rho"})
    boxre = plot_boxre if plot_boxre is not None else get_regions_re_boxre(points_config.get("regions_re"), margin_re=2.0)
    if boxre is None:
        boxre = [
            vdf_coords_re[:, 0].min() - 1, vdf_coords_re[:, 0].max() + 1,
            vdf_coords_re[:, 2].min() - 1, vdf_coords_re[:, 2].max() + 1,
        ]
    colormap_config.setdefault("boxre", boxre)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    pt.plot.plot_colormap(filename=file_location, axes=ax, **colormap_config)

    ax.scatter(
        vdf_coords_re[:, 0], vdf_coords_re[:, 2],
        marker="o", s=18, facecolor="white", edgecolor="black", linewidth=0.6,
        alpha=0.9, zorder=3, label=f"VDF cell ({len(vdf_coords_re)})",
    )

    half_width_di_normal = points_config.get("x_selection", {}).get("half_width_di_normal")
    outflow_aspect_ratio = points_config.get("x_selection", {}).get("outflow_aspect_ratio", 10.0)
    if half_width_di_normal is not None:
        for box_index, point_record in enumerate(x_point_records):
            if point_record.get("di_m") is None:
                continue
            geometry = compute_b_perp_di_box_geometry(
                reader=reader,
                point_record=point_record,
                half_width_di_normal=half_width_di_normal,
                outflow_aspect_ratio=outflow_aspect_ratio,
            )
            if geometry is None:
                continue
            b_hat_inplane, perp_hat_inplane, half_width_normal_m, half_width_outflow_m = geometry
            half_width_normal_re = half_width_normal_m / R_EARTH
            half_width_outflow_re = half_width_outflow_m / R_EARTH
            center_re = np.array([point_record["coord_re"][0], point_record["coord_re"][2]])
            corners = np.array(
                [
                    center_re + sign_along * half_width_outflow_re * b_hat_inplane
                    + sign_perp * half_width_normal_re * perp_hat_inplane
                    for sign_along, sign_perp in [(1, 1), (1, -1), (-1, -1), (-1, 1)]
                ]
            )
            polygon = Polygon(
                corners, closed=True, facecolor="tab:blue", edgecolor="tab:blue",
                alpha=0.18, linewidth=1.5, zorder=2,
                label="X search box (perp to B, 1:10)" if box_index == 0 else None,
            )
            ax.add_patch(polygon)

    if show_o_points and o_selection_method == "gyroradius":
        radius_gyroradii = points_config.get("o_selection", {}).get("radius_gyroradii", 1.0)
        for circle_index, point_record in enumerate(o_point_records):
            if point_record.get("rho_i_re") is None:
                continue
            radius_re = float(radius_gyroradii) * float(point_record["rho_i_re"])
            center_re = (point_record["coord_re"][0], point_record["coord_re"][2])
            circle = Circle(
                center_re, radius_re, facecolor="tab:green", edgecolor="tab:green",
                alpha=0.18, linewidth=1.5, zorder=2,
                label="O search circle (gyroradius)" if circle_index == 0 else None,
            )
            ax.add_patch(circle)
    elif show_o_points:
        draw_o_point_search_areas(
            ax=ax,
            reader=reader,
            metadata_rows=metadata_rows,
            flux_file_template=str(flux_file_location),
            o_core_fraction=points_config.get("o_selection", {}).get("core_fraction"),
        )
    if "cid" in metadata_rows.columns:
        matched_rows = metadata_rows[metadata_rows["cid"].notna()]
        if not matched_rows.empty:
            matched_coords_re = np.asarray(
                [
                    reader.get_cell_coordinates(int(cid)) / R_EARTH
                    for cid in matched_rows["cid"]
                ]
            )
            ax.scatter(
                matched_coords_re[:, 0], matched_coords_re[:, 2],
                marker=".", s=40, color="red", zorder=5, label="Matched VDF cell",
            )

    if show_x_points:
        x_rows = metadata_rows[metadata_rows["point_kind"] == "x"]
        ax.scatter(
            x_rows["source_point_x_re"], x_rows["source_point_z_re"],
            marker="x", s=80, color="blue", linewidths=2, label="X point (detected)",
        )
        if "di_re" in x_rows.columns:
            for _, row in x_rows.iterrows():
                ax.annotate(
                    f"d_i={row['di_re']:.3f} Re",
                    (row["source_point_x_re"], row["source_point_z_re"]),
                    fontsize=7, color="blue", xytext=(4, 4), textcoords="offset points",
                )

    if show_o_points:
        o_rows = metadata_rows[metadata_rows["point_kind"] == "o"]
        ax.scatter(
            o_rows["source_point_x_re"], o_rows["source_point_z_re"],
            marker="o", s=80, facecolor="none", edgecolor="green", linewidths=2, label="O point (detected)",
        )
        if "rho_i_re" in o_rows.columns:
            for _, row in o_rows.iterrows():
                ax.annotate(
                    f"rho_i={row['rho_i_re']:.3f} Re",
                    (row["source_point_x_re"], row["source_point_z_re"]),
                    fontsize=7, color="green", xytext=(4, 4), textcoords="offset points",
                )

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# =====================================================================
# Snapshot colormap backdrop + Shue boundaries -- shared by every plot in
# the clustering pipeline below, so the curves can't drift out of sync
# =====================================================================

def draw_snapshot_colormap(ax, file_location, boxre, var="rho"):
    """Density colormap background shared by every spatial cluster plot below."""

    import analysator as pt

    pt.plot.plot_colormap(filename=str(file_location), axes=ax, var=var, boxre=list(boxre))


def draw_shue_boundaries(ax, shue_fit, show_r0_circle=True, show_subsolar_marker=False, lobe_r_min_re=None):
    """
    Draw the magnetopause/bow-shock Shue curves, plus two independent
    optional circles -- these are two different physical quantities, drawn
    separately, never conflated:
    - show_r0_circle: R = r0, the fitted Shue *magnetopause standoff
      distance* (shue_fit["r0_re"]) -- always this value, regardless of
      lobe_r_min_re.
    - lobe_r_min_re: R = lobe_r_min_re, the separate *inner_magnetosphere/
      lobes classification cutoff* (see MAGNETOPAUSE_CONFIG["lobe_r_min_re"],
      classify_magnetosphere_regions) -- only drawn if explicitly given, in
      a distinct style, since it generally differs from r0 (r0 is a
      dayside-only standoff distance; lobe_r_min_re is deliberately a
      separate, larger radius so nightside plasma near Earth isn't
      mislabeled "lobes" -- see schema.md).
    Also optionally the subsolar marker (always at r0).
    """

    theta = np.linspace(0, 2.5, 300)
    r_shue_re = shue_boundary_r_re(
        x_re=np.cos(theta), z_re=np.sin(theta), r0_re=shue_fit["r0_re"], alpha=shue_fit["alpha"],
    )
    ax.plot(r_shue_re * np.cos(theta), r_shue_re * np.sin(theta), color="black", linewidth=1.5, label="Magnetopause")
    ax.plot(r_shue_re * np.cos(theta), -r_shue_re * np.sin(theta), color="black", linewidth=1.5)

    if shue_fit["r_bs_re"] is not None:
        r_bs_re = shue_boundary_r_re(
            x_re=np.cos(theta), z_re=np.sin(theta), r0_re=shue_fit["r_bs_re"], alpha=shue_fit["alpha"],
        )
        ax.plot(r_bs_re * np.cos(theta), r_bs_re * np.sin(theta), color="tab:gray", linewidth=1.5, label="Bow shock")
        ax.plot(r_bs_re * np.cos(theta), -r_bs_re * np.sin(theta), color="tab:gray", linewidth=1.5)

    full_circle_theta = np.linspace(0, 2 * np.pi, 300)

    if show_r0_circle:
        ax.plot(
            shue_fit["r0_re"] * np.cos(full_circle_theta), shue_fit["r0_re"] * np.sin(full_circle_theta),
            color="tab:blue", linewidth=1.5, linestyle="--",
            label=f"R = r0 ({shue_fit['r0_re']:.2f} Re, magnetopause standoff)",
        )

    if lobe_r_min_re is not None:
        ax.plot(
            lobe_r_min_re * np.cos(full_circle_theta), lobe_r_min_re * np.sin(full_circle_theta),
            color="tab:cyan", linewidth=1.5, linestyle=":",
            label=f"R = {lobe_r_min_re:.2f} (inner magnetosphere/lobes cutoff)",
        )

    if show_subsolar_marker:
        ax.scatter(
            [shue_fit["r0_re"]], [0], marker="*", s=200, color="black", zorder=5,
            label=f"subsolar (r0={shue_fit['r0_re']:.2f} Re)",
        )


def draw_current_layer_region(ax, current_layer_records, color="deeppink", alpha=0.35, s=6):
    """
    Small dot at each current-layer core point's own dense-grid position --
    shows the detected core's actual physical shape/extent (on the fine
    simulation grid), for visual comparison against which -- much sparser --
    VDF cells actually got labeled current_layer (see
    labeling.snapshot_labeling.find_current_layer_cellids: a VDF cell is
    current_layer only if its cellid IS one of these core points, a direct
    match, not a search-radius/nearest-cell step).

    current_layer_records : list of dict
        Core-point records from
        labeling.snapshot_labeling.find_current_layer_cellids /
        physics.current_layer.find_current_layer_core_records (needs
        coord_re per record).
    """

    coords_re = np.asarray([record["coord_re"] for record in current_layer_records], dtype=float)
    ax.scatter(
        coords_re[:, 0], coords_re[:, 2], color=color, alpha=alpha, s=s,
        edgecolor="none", zorder=3,
    )


def plot_combined_clusters(
    file_location, vdf_coords_re, combined_labels, shue_fit, output_path,
    boxre=(-30, 15, -10, 10), figsize=(10, 10.5),
    current_layer_records=None, lobe_r_min_re=None,
):
    """All named clusters on one colormap: Shue-fit background regions plus whichever point substance(s) are active (x_point/o_point/x_point_o_point and/or current_layer, see schema.md) on top; no_density_data cells are left unplotted (invalid density, not a real spatial category). "undefined" (the r0-to-lobe_r_min_re gap and any other unclassified cell -- see classify_magnetosphere_regions) is drawn, deliberately visible, so it's easy to check how big that band actually is. Pass current_layer_records (see find_current_layer_cellids) to additionally show the current layer core's actual dense-grid extent -- see draw_current_layer_region. Pass lobe_r_min_re (see MAGNETOPAUSE_CONFIG) so the drawn inner-magnetosphere/lobes cutoff circle matches what classify_magnetosphere_regions actually used, not r0."""

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    draw_snapshot_colormap(ax, file_location, boxre)

    if current_layer_records:
        draw_current_layer_region(ax, current_layer_records)

    region_styles = {
        "solar_wind": {"label": "Solar wind", "color": "tab:gray", "marker": ".", "s": 22, "zorder": 2},
        "magnetosheath": {"label": "Magnetosheath", "color": "tab:orange", "marker": ".", "s": 22, "zorder": 2},
        "lobes": {"label": "Lobes", "color": "tab:purple", "marker": ".", "s": 22, "zorder": 2},
        "inner_magnetosphere": {"label": "Inner magnetosphere", "color": "tab:blue", "marker": ".", "s": 22, "zorder": 2},
        "undefined": {"label": "Undefined", "color": "dimgray", "marker": "s", "s": 30, "zorder": 3},
        "x_point": {"label": "X-points (IDR)", "color": "red", "marker": "x", "s": 130, "zorder": 5},
        "o_point": {"label": "O-points (dipolarization front)", "color": "gold", "marker": "o", "s": 100, "zorder": 5},
        "x_point_o_point": {"label": "X+O shared cell", "color": "black", "marker": "*", "s": 220, "zorder": 6},
        "current_layer": {"label": "Current layer", "color": "deeppink", "marker": "D", "s": 60, "zorder": 5},
    }
    for region, style in region_styles.items():
        mask = combined_labels == region
        if not mask.any():
            continue
        ax.scatter(
            vdf_coords_re[mask, 0], vdf_coords_re[mask, 2],
            alpha=0.85, edgecolor="black" if region == "o_point" else "none",
            linewidth=0.6, label=f"{style['label']} ({mask.sum()})",
            color=style["color"], marker=style["marker"], s=style["s"], zorder=style["zorder"],
        )

    draw_shue_boundaries(ax, shue_fit, show_r0_circle=True, show_subsolar_marker=False, lobe_r_min_re=lobe_r_min_re)

    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.tick_params(axis="both", labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    if current_layer_records:
        core_handle = ax.scatter([], [], color="deeppink", alpha=0.35, s=20, edgecolor="none")
        handles.append(core_handle)
        labels.append(f"Current layer core ({len(current_layer_records)} dense-grid points)")
    # fig.legend (figure-fraction coords), not ax.legend (axes-fraction) --
    # draw_snapshot_colormap's analysator call fixes the axes to an equal
    # aspect ratio, which can leave the axes box much shorter than the
    # figure for a wide/flat boxre. Pinning the legend to the figure's
    # bottom edge then leaves a dead gap between the (short, vertically
    # centered) axes and the legend. Force a draw so get_tightbbox reflects
    # that aspect-corrected box, then anchor the legend just below the
    # axes' full rendered extent -- tick labels and the x-axis label
    # included, not just the bare plotting box (ax.get_position() alone
    # excludes those and made the legend overlap the x-axis label) -- so it
    # sits in the gap instead of below it.
    fig.tight_layout()
    fig.canvas.draw()
    axes_bottom_fig_frac = ax.get_tightbbox().transformed(fig.transFigure.inverted()).y0
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, axes_bottom_fig_frac - 0.01),
        borderaxespad=0.5, fontsize=9, ncol=4,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_scores(result, output_path):
    """Silhouette score vs k, with the chosen best_k marked."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["k_range"], result["silhouette_scores"], marker="o")
    ax.axvline(result["best_k"], color="tab:red", linestyle="--", label=f"best k={result['best_k']}")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_scatter(result, cellids, phys_labels, output_path):
    """PC1 vs PC2 with two colorbars on the same points: cluster_phys (physical ground-truth region label, large translucent ring) and cluster_ml (blind KMeans cluster, small solid dot on top) -- see README's "Terminology" section for the cluster_phys/cluster_ml distinction."""

    scores = result["pca_scores"]
    cluster_ml = np.asarray(result["labels"])
    phys_labels = np.asarray(phys_labels)

    phys_categories = sorted(set(phys_labels.tolist()))
    phys_codes = np.array([phys_categories.index(label) for label in phys_labels])
    ml_categories = sorted(set(cluster_ml.tolist()))

    fig, ax = plt.subplots(figsize=(9, 6))

    phys_scatter = ax.scatter(
        scores[:, 0], scores[:, 1],
        c=phys_codes, cmap="tab10", vmin=-0.5, vmax=len(phys_categories) - 0.5,
        s=220, alpha=0.35, edgecolors="none", zorder=1,
    )
    ml_scatter = ax.scatter(
        scores[:, 0], scores[:, 1],
        c=cluster_ml, cmap="tab20", vmin=-0.5, vmax=len(ml_categories) - 0.5,
        s=30, edgecolors="black", linewidth=0.3, zorder=2,
    )

    phys_cbar = fig.colorbar(phys_scatter, ax=ax, location="left", pad=0.15, fraction=0.05)
    phys_cbar.set_ticks(range(len(phys_categories)))
    phys_cbar.set_ticklabels(phys_categories)
    phys_cbar.set_label("cluster_phys")

    ml_cbar = fig.colorbar(ml_scatter, ax=ax, location="right", fraction=0.05)
    ml_cbar.set_ticks(ml_categories)
    ml_cbar.set_label("cluster_ml")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Cluster-plotting pipeline (scripts/ml_models/run_snapshot_pca.py +
# plot_snapshot_pca.py, and scripts/data_proc/verify_data.py): step 1 is
# plot_colormap_with_vdf_markers above; steps 2 and 3 below.
# =====================================================================

def plot_cluster_vdf_positions(
    file_location, reader, representative_cellids, output_path,
    boxre=(-30, 15, -10, 10), figsize=(10, 9),
    current_layer_records=None,
):
    """Step 2: colormap with the spatial position of each cluster's representative VDF (same cells as plot_cluster_vdf_examples), one marker per label so the two plots always match. Pass current_layer_records (see find_current_layer_cellids) to additionally show the current layer core's actual dense-grid extent -- see draw_current_layer_region."""

    if not representative_cellids:
        print("No cluster positions to plot.")
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    draw_snapshot_colormap(ax, file_location, boxre)

    if current_layer_records:
        draw_current_layer_region(ax, current_layer_records)

    labels = sorted(representative_cellids)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels))) if len(labels) <= 10 else plt.cm.tab20(np.linspace(0, 1, len(labels)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

    for index, label in enumerate(labels):
        cid = representative_cellids[label]
        x_re, _y_re, z_re = np.asarray(reader.get_cell_coordinates(int(cid)), dtype=float) / R_EARTH
        ax.scatter(
            [x_re], [z_re], color=[colors[index]], marker=markers[index % len(markers)],
            s=140, edgecolor="black", linewidth=0.8, zorder=5, label=f"{label} (cid={cid})",
        )
        ax.annotate(
            label, (x_re, z_re), fontsize=8, color="black",
            xytext=(6, 6), textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    handles, plot_labels = ax.get_legend_handles_labels()
    if current_layer_records:
        core_handle = ax.scatter([], [], color="deeppink", alpha=0.35, s=20, edgecolor="none")
        handles.append(core_handle)
        plot_labels.append(f"Current layer core ({len(current_layer_records)} dense-grid points)")
    ax.legend(
        handles, plot_labels, loc="upper left", bbox_to_anchor=(1.22, 1),
        borderaxespad=0, fontsize=7, labelspacing=1.2,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_vdf_examples(reader, representative_cellids, output_path, pop="avgs", vdflim=2e6, row_figsize=(12, 3.6)):
    """Step 3: one example VDF per cluster label in representative_cellids (from labeling.snapshot_labeling.pick_cluster_representative_cellids), three velocity-space projections per row sliced through each VDF's own peak (not the mesh center, since fast populations sit far from vx=0)."""

    if not representative_cellids:
        print("No clusters to plot VDF examples for.")
        return

    extractor = VdfExtractor(reader=reader, pop=pop)
    extent, dv, _threshold = get_vdf_plot_parameters(
        reader=reader, cid=next(iter(representative_cellids.values())),
        vdf_shape=extractor.vdf_shape, pop=pop,
    )
    vxmin, vymin, vzmin, vxmax, vymax, vzmax = extent / 1000.0  # m/s -> km/s

    plane_axes = [
        ("vx-vy", "v_x [km/s]", "v_y [km/s]", [vxmin, vxmax, vymin, vymax]),
        ("vx-vz", "v_x [km/s]", "v_z [km/s]", [vxmin, vxmax, vzmin, vzmax]),
        ("vy-vz", "v_y [km/s]", "v_z [km/s]", [vymin, vymax, vzmin, vzmax]),
    ]

    n_rows = len(representative_cellids)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(row_figsize[0], row_figsize[1] * n_rows), squeeze=False,
    )

    for row, (label, cid) in enumerate(representative_cellids.items()):
        vdf = extractor.extract(cid=cid)
        threshold = float(reader.read_variable("MinValue", int(cid)))
        peak_x, peak_y, peak_z = np.unravel_index(np.argmax(vdf), vdf.shape)
        plane_slices = [vdf[:, :, peak_z], vdf[:, peak_y, :], vdf[peak_x, :, :]]

        for col, ((plane_name, xlabel, ylabel, plane_extent), plane_slice) in enumerate(zip(plane_axes, plane_slices)):
            ax = axes[row, col]
            vdf_plot_raw = plane_slice.T.astype(np.float32) * dv
            vdf_plot = np.ma.masked_less_equal(
                np.where(vdf_plot_raw < threshold * dv, 0, vdf_plot_raw), 0,
            )
            if vdf_plot.count() == 0:
                vdf_plot = np.ma.masked_less_equal(vdf_plot_raw, 0)

            if vdf_plot.count() > 0:
                im = ax.imshow(vdf_plot, origin="lower", extent=plane_extent, norm="log", cmap="nipy_spectral")
                fig.colorbar(im, ax=ax, label="f(v)", fraction=0.046, pad=0.04)
            ax.set_xlim(-vdflim / 1000, vdflim / 1000)
            ax.set_ylim(-vdflim / 1000, vdflim / 1000)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label} (cid={cid}) -- {plane_name}" if col == 0 else plane_name, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_hermite_spectra(cellids, X_hermite, representative_cellids, output_path=None):
    """
    Verification plot for PCA_CONFIG["feature_representation"] == "hermite":
    the actual (order, order, order) feature array run_snapshot_pca.py
    flattens and feeds to StandardScaler+PCA (X_hermite.npy, indexed by
    cellids to match representative_cellids' cid values) -- not a
    reconstruction back into velocity space, the literal numbers PCA sees,
    just not flattened for display.

    Reduced to 2D per representative: spectra[n, m, l] indexes
    (parallel=B, perp1, perp2) in that order (matches the rotation frame
    build_rotation_matrix/get_rotated_vdf build, rows (b_hat, v_perp_hat,
    b_hat x v_perp_hat), and compute_hermite_spectra's own "ijk,ni,mj,lk
    ->nml" axis order). Integrating (in quadrature, sqrt(sum(spectra**2,
    axis=2))) over l -- the second perpendicular direction -- collapses
    that one axis, leaving a (parallel order n, perp order m) image per
    representative: the direct Hermite-space analogue of reducing a 3D
    VDF to (v_parallel, v_perp) by integrating over gyrophase (assuming
    gyrotropy around B). One panel per representative (shared color scale
    would hide the small categories entirely under current_layer/
    magnetosheath's much larger coefficients -- see TESTING.md -- so each
    panel is normalized to its own max instead).
    """

    if not representative_cellids:
        print("No clusters to plot Hermite spectra for.")
        return

    cellids = np.asarray(cellids)
    n_panels = len(representative_cellids)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows), squeeze=False)

    for index, (label, cid) in enumerate(representative_cellids.items()):
        row, col = divmod(index, n_cols)
        ax = axes[row, col]

        matches = np.flatnonzero(cellids == int(cid))
        if matches.size == 0:
            print(f"  {label} (cid={cid}): not found in cellids, skipping")
            ax.axis("off")
            continue

        spectra = np.asarray(X_hermite[matches[0]], dtype=float)
        par_perp = np.sqrt(np.sum(spectra**2, axis=2))
        vmax = float(par_perp.max()) or 1.0
        print(f"  {label} (cid={cid}): par-perp max={vmax:.3g}, full L2 norm={np.linalg.norm(spectra):.3g}")

        im = ax.imshow(par_perp.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{label} (cid={cid})\nmax={vmax:.2g}", fontsize=9)
        ax.set_xlabel("parallel order n")
        ax.set_ylabel("perp order m")
        fig.colorbar(im, ax=ax, label="|coeff| (L2 over 2nd perp)", fraction=0.046, pad=0.04)

    for index in range(n_panels, n_rows * n_cols):
        row, col = divmod(index, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("Hermite spectra per cluster representative, integrated over the 2nd perpendicular axis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
