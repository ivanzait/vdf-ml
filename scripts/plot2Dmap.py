#python scripts/plot2Dmap.py
#
# Edit the PARAMETERS block below, then run. No YAML config file involved --
# reusable plotting logic lives in src/plot_tools.py.
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

from src.plot_tools import (
    plot_colormap_with_vdf_markers,
    plot_vdf_and_hermite_grid,
    plot_vdf_rotation_comparison,
    select_vdf_points,
)
from src.vdf_helpers import get_vdf_cells_with_coords_re

# ----------------------------- PARAMETERS -----------------------------

FILE_LOCATION = "/Users/ivanzait/Downloads/bulk.0003408.vlsv"
RUN_ID = "smoke_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "plot2Dmap" / RUN_ID

# Point selection: cells with a VDF inside these Earth-radii bounds are
# marked distinctly and, if DETAIL_PLOT_ENABLED, get their own VDF/Hermite
# panel. Set any of these to None to skip that axis's filter.
X_RANGE = (-15.0, -10.0)
Y_RANGE = None
Z_RANGE = (-1.0, 1.0)

COLORMAP_CONFIG = {
    "var": "E",
    "operator": "z",
    "boxre": [-20, 12, -12, 12],
    # "cbtitle": r"$E_z$ [mV/m]",
    # "vmin": -20000,
    # "vmax": 20000,
    # "colormap": "RdYlGn",
    "show_all_vdf_cells": True,
    "all_points_style": {"marker": "o", "s": 6, "color": "black", "alpha": 0.6},
    "selected_points_style": {
        "marker": "*",
        "s": 300,
        "edgecolor": "black",
        "facecolor": "none",
        "linewidth": 1.5,
    },
}

HERMITE_ORDER = 22
POP = "avgs"
DETAIL_PLOT_ENABLED = True
ROTATION_COMPARISON_ENABLED = True
VDF_CMAP = "viridis"
HERMITE_CMAP = "RdBu_r"

# ------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = pt.vlsvfile.VlsvReader(str(FILE_LOCATION))
    cellids, coords_re = get_vdf_cells_with_coords_re(reader=reader)

    selected_cellids, selected_coords_re = select_vdf_points(
        cellids=cellids,
        coords_re=coords_re,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
    )
    print(f"Total VDF cells in file: {len(cellids)}")
    print(f"Selected {len(selected_cellids)} VDF point(s):")
    for cid, coord in zip(selected_cellids, selected_coords_re):
        print(f"  cid={int(cid)} coord_re={coord.tolist()}")

    colormap_output_path = OUTPUT_DIR / "colormap.png"
    plot_colormap_with_vdf_markers(
        file_location=FILE_LOCATION,
        colormap_config=COLORMAP_CONFIG,
        all_coords_re=coords_re,
        selected_coords_re=selected_coords_re,
        output_path=colormap_output_path,
    )
    print(f"Saved colormap plot to: {colormap_output_path}")

    if DETAIL_PLOT_ENABLED and len(selected_cellids) > 0:
        detail_output_path = OUTPUT_DIR / "vdf_and_hermite.png"
        plot_vdf_and_hermite_grid(
            reader=reader,
            cellids=selected_cellids,
            coords_re=selected_coords_re,
            order=HERMITE_ORDER,
            pop=POP,
            vdf_cmap=VDF_CMAP,
            hermite_cmap=HERMITE_CMAP,
            output_path=detail_output_path,
        )
        print(f"Saved VDF/Hermite detail plot to: {detail_output_path}")

    if ROTATION_COMPARISON_ENABLED and len(selected_cellids) > 0:
        rotation_output_path = OUTPUT_DIR / "rotation_comparison.png"
        plot_vdf_rotation_comparison(
            reader=reader,
            cellids=selected_cellids,
            coords_re=selected_coords_re,
            pop=POP,
            vdf_cmap=VDF_CMAP,
            output_path=rotation_output_path,
        )
        print(f"Saved rotation comparison plot to: {rotation_output_path}")


if __name__ == "__main__":
    main()
