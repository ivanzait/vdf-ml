"""
Plotting library for Vlasiator simulation and VDF/Hermite ML results.

Calling scripts (e.g. ``scripts/plot2Dmap.py``) define file paths, point
selections, and styling as plain Python variables and call into these
functions directly -- no YAML config files.

Sections
--------
Simulation plots: ``plot_colormap_with_vdf_markers``, ``select_vdf_points``
ML / VDF plots: ``plot_vdf_and_hermite_grid``, ``plot_vdf_rotation_comparison``
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from src.hermite_transform import DEFAULT_HERMITE_ORDER, vdf_to_hermite_spectra
from src.vdf_extract import VdfExtractor
from src.vdf_helpers import (
    create_xz_slice,
    get_b_field,
    get_bulk_velocity,
    get_vdf_plot_axes_parameters,
)
from src.vdf_rotation import get_rotated_vdf

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


def select_vdf_points(
    cellids,
    coords_re,
    x_range=None,
    y_range=None,
    z_range=None,
    points_re=None,
):
    """
    Select VDF-carrying cells by spatial box or nearest-point match.

    Parameters
    ----------
    cellids : numpy.ndarray
        VDF-carrying spatial cell IDs.
    coords_re : numpy.ndarray
        Cell center coordinates in Earth radii with shape ``(n_cells, 3)``.
    x_range, y_range, z_range : tuple of float, optional
        Inclusive ``(min, max)`` bounds in Earth radii. Cells outside any
        configured range are excluded. Ignored when ``points_re`` is given.
    points_re : iterable of array-like, optional
        Explicit coordinates; each is matched to its nearest VDF cell
        instead of using the range filters.

    Returns
    -------
    selected_cellids : numpy.ndarray
        Selected spatial cell IDs.
    selected_coords_re : numpy.ndarray
        Selected cell center coordinates in Earth radii.
    """

    coords_re = np.asarray(coords_re, dtype=float)

    if points_re:
        indices = sorted(
            {
                int(np.argmin(np.linalg.norm(coords_re - np.asarray(point, dtype=float), axis=1)))
                for point in points_re
            }
        )
        indices = np.asarray(indices, dtype=int)
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
    """
    Plot an Analysator 2D colormap with VDF-cell markers overlaid.

    Parameters
    ----------
    file_location : str or pathlib.Path
        Path to the ``.vlsv`` file.
    colormap_config : dict
        Keyword arguments forwarded to ``analysator.plot.plot_colormap``,
        plus the optional marker-styling keys ``show_all_vdf_cells``,
        ``all_points_style``, and ``selected_points_style``.
    all_coords_re : numpy.ndarray, optional
        All VDF-cell coordinates to mark, shape ``(n_cells, 3)``.
    selected_coords_re : numpy.ndarray, optional
        Selected VDF-cell coordinates to mark distinctly, shape
        ``(n_selected, 3)``.
    output_path : str or pathlib.Path, optional
        If given, the figure is saved to this path.
    figsize : tuple of float, optional
        Matplotlib figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

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
    """
    Plot side-by-side VDF xz-slices and Hermite-spectra slices.

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open reader for the source VLSV file.
    cellids : array-like of int
        Spatial cell IDs to plot, one row per cell.
    coords_re : numpy.ndarray
        Cell center coordinates in Earth radii, shape ``(n_cells, 3)``.
    order : int, optional
        Number of Hermite modes per axis.
    pop : str, optional
        Particle population name used by Analysator.
    vdf_cmap : str, optional
        Colormap for the VDF xz-slice panels.
    hermite_cmap : str, optional
        Colormap for the Hermite-spectra panels.
    output_path : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

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
    """
    Plot VDF mid-slices before and after (B, v_perp, B x v_perp) rotation.

    One row per selected cell: original xz-slice (fixed vy) next to the
    rotated-frame slice at fixed v_perp (the B-perpendicular bulk-flow
    axis), so both panels show the same "cut through the middle" idea in
    their respective frames. Each row's title reports B, the bulk velocity,
    the angle between them, and the total-density change from rotation
    (should be close to zero -- resampling introduces a small numerical
    loss, see ``src.vdf_rotation.get_rotated_vdf``).

    Parameters
    ----------
    reader : analysator.vlsvfile.VlsvReader
        Open reader for the source VLSV file.
    cellids : array-like of int
        Spatial cell IDs to plot, one row per cell.
    coords_re : numpy.ndarray
        Cell center coordinates in Earth radii, shape ``(n_cells, 3)``.
    pop : str, optional
        Particle population name used by Analysator.
    vdf_cmap : str, optional
        Colormap for both panels.
    output_path : str or pathlib.Path, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

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
