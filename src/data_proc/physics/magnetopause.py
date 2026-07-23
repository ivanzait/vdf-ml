"""
Solar wind / magnetosheath / inner-magnetosphere classification via
subsolar-anchored Shue et al. (1998)-shaped surfaces: r(theta) = r_mp * (2 / (1
+ cos(theta)))**alpha, theta measured from the Sun-Earth line (+x). Both the
magnetopause and the bow shock use this same functional shape (a common
simplification -- real bow shocks are closer to a hyperboloid, but this
avoids a second angular fit), anchored at their own subsolar standoff
distances (r_mp, r_bs) and sharing the canonical alpha (this run has no solar
wind monitor to fit alpha from, per SHUE_ALPHA_DEFAULT).

The flared surfaces only decide the solar-wind/magnetosheath split (that
boundary genuinely flares with angle); "inner_magnetosphere"/"lobes" are
then simple spheres around Earth (R < r_mp / R >= lobe_r_min_re), not the
flared shape -- see classify_magnetosphere_regions for the full priority
order, including the "undefined" catch-all for the r_mp-to-lobe_r_min_re gap.
"""

import numpy as np

from src.data_proc.vdf_tools import R_EARTH

SHUE_ALPHA_DEFAULT = 0.58


def find_subsolar_point(reader, x_scan_min_re=5.0, x_scan_max_re=30.0, n_scan_points=300, density_variable="rho"):
    """
    Locate the subsolar magnetopause and bow shock crossings along +x (z=y=0):
    scan density from x_scan_max_re (solar wind) inward to x_scan_min_re
    (magnetosphere). The magnetopause is the steepest drop in log(density)
    (sheath -> magnetosphere); the bow shock is the steepest rise (solar wind
    -> sheath), sunward of the magnetopause.

    Returns {"magnetopause": {x_re, cellid, rho_sheath, rho_sphere},
    "bow_shock": {x_re, cellid, rho_solar_wind, rho_sheath} or None if no
    rise was found sunward of the magnetopause (e.g. x_scan_max_re too
    small to reach undisturbed solar wind)}.
    """

    x_values_re = np.linspace(x_scan_max_re, x_scan_min_re, n_scan_points)
    cellids = np.asarray(
        [reader.get_cellid([x_re * R_EARTH, 0.0, 0.0]) for x_re in x_values_re],
        dtype=np.int64,
    )
    rho_values = np.asarray(reader.read_variable(density_variable, cellids), dtype=float)

    log_rho = np.log(np.clip(rho_values, 1e-30, None))
    d_log_rho = np.diff(log_rho)

    magnetopause_index = int(np.argmin(d_log_rho))
    magnetopause = {
        "x_re": float((x_values_re[magnetopause_index] + x_values_re[magnetopause_index + 1]) / 2.0),
        "cellid": int(cellids[magnetopause_index]),
        "rho_sheath": float(rho_values[magnetopause_index]),
        "rho_sphere": float(rho_values[magnetopause_index + 1]),
    }

    bow_shock_candidates = d_log_rho[:magnetopause_index]
    bow_shock = None
    if len(bow_shock_candidates) > 0:
        bow_shock_index = int(np.argmax(bow_shock_candidates))
        bow_shock = {
            "x_re": float((x_values_re[bow_shock_index] + x_values_re[bow_shock_index + 1]) / 2.0),
            "cellid": int(cellids[bow_shock_index]),
            "rho_solar_wind": float(rho_values[bow_shock_index]),
            "rho_sheath": float(rho_values[bow_shock_index + 1]),
        }

    return {"magnetopause": magnetopause, "bow_shock": bow_shock}


def fit_shue_model(reader, x_scan_min_re=5.0, x_scan_max_re=30.0, n_scan_points=300, density_variable="rho", alpha=SHUE_ALPHA_DEFAULT):
    """
    Fit subsolar-anchored Shue-shaped surfaces for the magnetopause and bow
    shock: r_mp/r_bs from the detected density crossings, both using the
    canonical alpha. r_bs_re is None if no bow shock crossing was found.
    """

    subsolar = find_subsolar_point(
        reader=reader,
        x_scan_min_re=x_scan_min_re,
        x_scan_max_re=x_scan_max_re,
        n_scan_points=n_scan_points,
        density_variable=density_variable,
    )
    return {
        "r_mp_re": subsolar["magnetopause"]["x_re"],
        "r_bs_re": subsolar["bow_shock"]["x_re"] if subsolar["bow_shock"] is not None else None,
        "alpha": float(alpha),
        "subsolar": subsolar,
    }


def shue_boundary_r_re(x_re, z_re, r_mp_re, alpha=SHUE_ALPHA_DEFAULT):
    """Shue et al. (1998) magnetopause radius toward (x_re, z_re): r_mp * (2 / (1 + cos(theta)))**alpha."""

    x_re = np.asarray(x_re, dtype=float)
    z_re = np.asarray(z_re, dtype=float)
    r_re = np.sqrt(x_re**2 + z_re**2)
    cos_theta = np.divide(x_re, r_re, out=np.zeros_like(r_re), where=r_re > 0)

    return r_mp_re * (2.0 / (1.0 + cos_theta)) ** alpha


def classify_magnetosphere_regions(vdf_coords_re, densities, r_mp_re, r_bs_re=None, alpha=SHUE_ALPHA_DEFAULT, lobe_r_min_re=None):
    """
    Label each VDF cell by a fixed priority order (current_layer/x_o_points
    take priority over all of these -- applied later, outside this
    function, by combine_ground_truth_labels; this function only assigns
    the base region labels current_layer/x_o_points can override):
    - "no_density_data": density <= 0 (e.g. inside the simulation's inner
      boundary/vacuum region near Earth) -- can't be geometrically wrong,
      but the density itself is invalid, so it's kept as its own label
      rather than folded into a spatial category that would misrepresent
      it as real plasma. Checked first, before any of the rules below.
    - "solar_wind": beyond the (flared) bow shock surface, if r_bs_re is
      given. Without a bow shock model, this split can't be made and
      everything sunward of the magnetopause is "magnetosheath" instead.
    - "magnetosheath": sunward of the magnetopause (Shue surface) but not
      solar_wind -- everything between the bow shock and the magnetopause.
    - "inner_magnetosphere": earthward of the magnetopause AND R < r_mp (R =
      distance from Earth center, r_mp = the Shue standoff distance) -- a
      simple spherical cutoff near Earth, not the flared Shue shape.
    - "lobes": earthward of the magnetopause AND R >= lobe_r_min_re
      (defaults to r_mp if not given) -- tail lobes/plasma sheet.
    - "undefined": earthward of the magnetopause but neither
      inner_magnetosphere nor lobes -- i.e. r_mp <= R < lobe_r_min_re, when
      lobe_r_min_re > r_mp. r_mp is a *dayside-only* standoff distance;
      lobe_r_min_re is deliberately set larger (e.g. 10.0) so
      near-Earth nightside plasma doesn't get mislabeled "lobes" (see
      SCHEMA.md) -- but that means the band between them genuinely isn't
      confidently either category, so it's left unlabeled rather than
      arbitrarily assigned. This is also the catch-all for any other cell
      that doesn't satisfy any rule above.

    No margin/buffer zone around the magnetopause surface any more (the
    old "boundary_layer" label) -- current_layer's physically-detected
    core+margin (see physics/current_layer.py) already identifies the real
    magnetopause current layer with much better precision, so a crude
    geometric buffer here would be redundant at best and wrong at worst.
    """

    x_re = vdf_coords_re[:, 0]
    z_re = vdf_coords_re[:, 2]
    r_re = np.sqrt(x_re**2 + z_re**2)
    r_shue_re = shue_boundary_r_re(x_re=x_re, z_re=z_re, r_mp_re=r_mp_re, alpha=alpha)
    inner_lobe_boundary_re = r_mp_re if lobe_r_min_re is None else float(lobe_r_min_re)

    densities = np.asarray(densities, dtype=float)
    positive_density = densities > 0

    labels = np.full(len(x_re), "undefined", dtype=object)
    labels[~positive_density] = "no_density_data"

    sunward_mask = positive_density & (r_re > r_shue_re)
    earthward_mask = positive_density & (r_re <= r_shue_re)

    if r_bs_re is not None:
        r_bow_shock_re = shue_boundary_r_re(x_re=x_re, z_re=z_re, r_mp_re=r_bs_re, alpha=alpha)
        solar_wind_mask = sunward_mask & (r_re > r_bow_shock_re)
    else:
        solar_wind_mask = np.zeros(len(x_re), dtype=bool)
    magnetosheath_mask = sunward_mask & ~solar_wind_mask

    inner_mask = earthward_mask & (r_re < r_mp_re)
    lobe_mask = earthward_mask & (r_re >= inner_lobe_boundary_re)

    labels[magnetosheath_mask] = "magnetosheath"
    labels[solar_wind_mask] = "solar_wind"
    labels[inner_mask] = "inner_magnetosphere"
    labels[lobe_mask] = "lobes"

    return labels
