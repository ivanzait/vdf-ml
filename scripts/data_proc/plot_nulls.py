#python scripts/data_proc/plot_nulls.py
#
# Edit the PARAMETERS block below, then run. No YAML config file involved --
# plotting logic lives in src/data_proc/plot_tools.py:plot_snapshot_topology.
#
# Colormap of one snapshot with detected X/O points, their physical search
# boxes/contours, and each X point's ion inertial length annotated. Sanity
# check before trusting an X/O-point ground truth built elsewhere: if the
# X search boxes never contain a VDF cell (small black dots), either widen
# half_width_di_normal or note this snapshot's VDF output is too sparse for
# the physical criterion to match anything.
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import analysator as pt
except ImportError:
    # Falls back to a local analysator checkout when it isn't installed as
    # a package (e.g. a plain git clone on a laptop instead of a cluster
    # module/PYTHONPATH setup). Override with the ANALYSATOR_PATH env var.
    analysator_path = os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    )
    sys.path.append(analysator_path)
    import analysator as pt

from src.data_proc.plot_tools import plot_snapshot_topology

# ----------------------------- PARAMETERS -----------------------------

FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.vlsv"
FLUX_FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.bin"
RUN_ID = "smoke_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "nulls" / RUN_ID

# Which detected point kind(s) to match/draw. Detection still runs for both
# (find_point_records doesn't take a kind filter), this just controls what
# ends up on the plot.
SHOW_X_POINTS = True
SHOW_O_POINTS = True

# Visualization zoom [xmin, xmax, zmin, zmax] in Re, independent of the
# physical search regions below. None -> auto from REGIONS_RE (with margin).
PLOT_BOXRE = [-20.0, 12.0, -8.0, 8.0]

# Only points inside at least one of these boxes are kept.
REGIONS_RE = {
    "nightside": {"x_between": [-30.0, -8.0], "z_abs_max": 6.0},
    "dayside": {"x_between": [5.0, 15.0], "z_abs_max": 10.0},
}

# Physical X/O-point detector settings, matching configs/create_dataset.yaml
# except for half_width_di_normal/outflow_aspect_ratio (see
# get_vdf_cellids_in_b_perp_di_box).
POINTS_CONFIG = {
    "point_region_names": list(REGIONS_RE),
    "regions_re": REGIONS_RE,
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
        "radius_gyroradii": 2.0,
        "y_half_width_re": 0.0,
        "manual_re": {"x_half_width_re": 0.4, "y_half_width_re": 0.0, "z_half_width_re": 0.3},
    },
}

COLORMAP_CONFIG = {"var": "rho"}

# ------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "nulls.png"

    plot_snapshot_topology(
        file_location=FILE_LOCATION,
        flux_file_location=FLUX_FILE_LOCATION,
        points_config=POINTS_CONFIG,
        output_path=output_path,
        colormap_config=COLORMAP_CONFIG,
        show_x_points=SHOW_X_POINTS,
        show_o_points=SHOW_O_POINTS,
        plot_boxre=PLOT_BOXRE,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
