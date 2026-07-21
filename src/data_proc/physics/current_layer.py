"""
Current-layer (current-sheet) detection: builds a dense B-field grid over
the domain, computes the current density J = curl(B) and its Jacobian
G = grad(B) by finite differences (assumes a spatially uniform, y-invariant
[2D xz] grid -- the same assumption physics.point_topology's
read_smoothed_flux_grid already makes for the flux function), and finds the
highest-|J| "core" cells. A VDF cell is labeled current_layer only if its
own cellid IS one of these core cells (a direct match, see
labeling.snapshot_labeling.find_current_layer_cellids), no margin/expansion
step.

This module used to also derive per-core-cell LMN boundary-normal
coordinates (Minimum Gradient/Directional-Derivative Analysis, MGA/MDD;
Alho et al. 2024, Ann. Geophys. 42, 145,
https://doi.org/10.5194/angeo-42-145-2024) for the margin-expansion step
that selection no longer uses. That LMN basis builder
(compute_local_lmn_basis) moved to physics.vdf_transform -- it's a general
per-point boundary-normal-frame construction, not specific to current-layer
detection, and is kept there as a future option for VDF coordinate
transforms (an (L, M, N)-aligned frame, alongside the existing
(B, v_perp, B x v_perp) frame vdf_transform.build_rotation_matrix builds).
"""

import numpy as np

from src.data_proc.physics.point_topology import MU0, compute_ion_inertial_length
from src.data_proc.vdf_tools import R_EARTH, create_region_mask_re


def read_b_field_grid(reader):
    """
    Dense (z_cells, x_cells) grid of B, read natively from the .vlsv file
    (no external file needed). Mirrors point_topology.read_smoothed_flux_grid's
    exact index convention (y=0 plane, spatially uniform grid assumed).

    Returns (x_array, z_array, Bx_zx, By_zx, Bz_zx), all meters/tesla.
    """

    x_cells = int(reader.get_spatial_mesh_size()[0])
    z_cells = int(reader.get_spatial_mesh_size()[2])
    xsize = reader.read_parameter("xcells_ini")
    xmax = reader.read_parameter("xmax")
    xmin = reader.read_parameter("xmin")
    zmin = reader.read_parameter("zmin")
    dx = float((xmax - xmin) / xsize)

    x_array = float(xmin) + np.arange(x_cells) * dx
    z_array = float(zmin) + np.arange(z_cells) * dx

    cellids_zx = np.arange(1, x_cells * z_cells + 1, dtype=np.int64).reshape(z_cells, x_cells)
    b_field = np.asarray(reader.read_variable("B", cellids_zx.ravel()), dtype=float)

    Bx_zx = b_field[:, 0].reshape(z_cells, x_cells)
    By_zx = b_field[:, 1].reshape(z_cells, x_cells)
    Bz_zx = b_field[:, 2].reshape(z_cells, x_cells)

    return x_array, z_array, Bx_zx, By_zx, Bz_zx


def compute_b_jacobian_grid(Bx_zx, By_zx, Bz_zx, dx):
    """
    In-plane (x, z) partial derivatives of B via central finite differences,
    one vectorized np.gradient call per component/axis. axis=1 is x (the
    fast-varying axis in the (z_cells, x_cells) convention), axis=0 is z.

    Returns a dict with keys dBx_dx, dBx_dz, dBy_dx, dBy_dz, dBz_dx, dBz_dz.
    """

    return {
        "dBx_dx": np.gradient(Bx_zx, dx, axis=1),
        "dBx_dz": np.gradient(Bx_zx, dx, axis=0),
        "dBy_dx": np.gradient(By_zx, dx, axis=1),
        "dBy_dz": np.gradient(By_zx, dx, axis=0),
        "dBz_dx": np.gradient(Bz_zx, dx, axis=1),
        "dBz_dz": np.gradient(Bz_zx, dx, axis=0),
    }


def compute_current_density_grid(jacobian_grid):
    """
    Current density J = curl(B) / MU0 on the dense grid, reduced for a
    y-invariant (2D xz-plane) domain (d/dy = 0 everywhere):
    Jx = -dBy/dz, Jy = dBx/dz - dBz/dx, Jz = dBy/dx. The in-plane Jy is the
    classic reconnection/current-sheet current in this geometry.

    Returns (Jx_zx, Jy_zx, Jz_zx, Jmag_zx).
    """

    Jx_zx = -jacobian_grid["dBy_dz"] / MU0
    Jy_zx = (jacobian_grid["dBx_dz"] - jacobian_grid["dBz_dx"]) / MU0
    Jz_zx = jacobian_grid["dBy_dx"] / MU0
    Jmag_zx = np.sqrt(Jx_zx**2 + Jy_zx**2 + Jz_zx**2)

    return Jx_zx, Jy_zx, Jz_zx, Jmag_zx


def find_current_layer_core_records(reader, points_config, regions_re=None):
    """
    Grid points forming the current layer's "core": |J| >= core_fraction *
    peak(|J|), each with its ion inertial length d_i (from that point's own
    density).

    The peak is found *independently within each named box* in
    points_config["current_layer_selection"]["search_regions_re"] (e.g.
    a "dayside" and a "tail" box), not once globally -- a single global
    peak would let the strongest current structure in the domain (usually
    the dayside magnetopause) swallow the threshold, silently excluding a
    real but weaker one (e.g. the tail current sheet). Cores from every
    sub-region are unioned (a grid point matching more than one sub-region
    is only kept once). If search_regions_re isn't configured, the whole
    domain (or regions_re, if given) is searched as a single region.

    Grid points within min_r_re of Earth are excluded from every
    sub-region's search, before the peak is even computed -- field-aligned
    currents near the inner simulation boundary (a genuinely different
    physical structure from a cross-field current sheet) can otherwise
    compete with or even set a sub-region's peak once core_fraction is
    lowered, especially near the poles where a "dayside"/"tail" x-split
    alone doesn't separate them out. In practice these boundary-adjacent
    points are also recognizable by a suspiciously exact, uniform density
    (a fixed boundary-condition fill value) rather than physically-varying
    plasma, but min_r_re is the direct, general fix.

    Config: points_config["current_layer_selection"] (density_variable,
    core_fraction, search_regions_re, min_r_re).

    regions_re : dict of {name: region_re}, optional
        Outer restriction applied on top of every search sub-region, same
        union-of-boxes convention as everywhere else (see
        vdf_tools.create_region_mask_re). If omitted, sub-regions aren't
        further restricted.

    Returns a list of dicts: coord_m, coord_re, cellid, j_magnitude, rho,
    di_m.
    """

    current_layer_config = (points_config or {}).get("current_layer_selection", {})
    density_variable = current_layer_config.get("density_variable", "rho")
    core_fraction = float(current_layer_config.get("core_fraction", 0.8))
    search_regions_re = current_layer_config.get("search_regions_re") or {"_default": None}
    min_r_re = float(current_layer_config.get("min_r_re", 0.0))

    x_array, z_array, Bx_zx, By_zx, Bz_zx = read_b_field_grid(reader)
    dx = float(x_array[1] - x_array[0])
    jacobian_grid = compute_b_jacobian_grid(Bx_zx, By_zx, Bz_zx, dx)
    _Jx_zx, _Jy_zx, _Jz_zx, Jmag_zx = compute_current_density_grid(jacobian_grid)

    x_cells = len(x_array)
    z_cells = len(z_array)
    cellids_flat = np.arange(1, x_cells * z_cells + 1, dtype=np.int64)

    z_grid, x_grid = np.meshgrid(z_array, x_array, indexing="ij")
    coords_re_flat = np.stack(
        [x_grid.ravel() / R_EARTH, np.zeros(x_grid.size), z_grid.ravel() / R_EARTH],
        axis=1,
    )

    if regions_re:
        outer_mask_flat = np.zeros(coords_re_flat.shape[0], dtype=bool)
        for region_re in regions_re.values():
            outer_mask_flat |= create_region_mask_re(coords_re_flat, region_re)
    else:
        outer_mask_flat = np.ones(coords_re_flat.shape[0], dtype=bool)

    if min_r_re > 0:
        r_re_flat = np.sqrt(coords_re_flat[:, 0] ** 2 + coords_re_flat[:, 2] ** 2)
        outer_mask_flat &= r_re_flat >= min_r_re

    j_mag_flat = Jmag_zx.ravel()

    core_records = []
    seen_indices = set()

    for region_re in search_regions_re.values():
        if region_re is None:
            sub_region_mask = np.ones(coords_re_flat.shape[0], dtype=bool)
        else:
            sub_region_mask = create_region_mask_re(coords_re_flat, region_re)

        candidate_indices = np.flatnonzero(sub_region_mask & outer_mask_flat)
        if candidate_indices.size == 0:
            continue

        # Peak |J| must be searched only over real plasma (density > 0),
        # not the whole dense grid -- the inner boundary/vacuum region has
        # a strong, numerically large dipole-field curl that has nothing
        # to do with a plasma current sheet, and would otherwise dominate
        # the search.
        candidate_densities = np.asarray(
            reader.read_variable(density_variable, cellids_flat[candidate_indices]), dtype=float,
        )
        plasma_mask = candidate_densities > 0
        if not np.any(plasma_mask):
            continue

        plasma_indices = candidate_indices[plasma_mask]
        plasma_densities = candidate_densities[plasma_mask]
        candidate_j_mag = j_mag_flat[plasma_indices]
        if not np.any(np.isfinite(candidate_j_mag)):
            continue

        peak_j = float(np.nanmax(candidate_j_mag))
        if peak_j <= 0:
            continue

        core_mask_local = candidate_j_mag >= core_fraction * peak_j
        core_indices = plasma_indices[core_mask_local]
        core_densities = plasma_densities[core_mask_local]

        for index, number_density in zip(core_indices, core_densities):
            index = int(index)
            if index in seen_indices:
                continue
            seen_indices.add(index)

            x_re = float(coords_re_flat[index, 0])
            z_re = float(coords_re_flat[index, 2])

            core_records.append(
                {
                    "coord_m": [x_re * R_EARTH, 0.0, z_re * R_EARTH],
                    "coord_re": [x_re, 0.0, z_re],
                    "cellid": int(cellids_flat[index]),
                    "j_magnitude": float(j_mag_flat[index]),
                    "rho": float(number_density),
                    "di_m": float(compute_ion_inertial_length(number_density)),
                }
            )

    return core_records
