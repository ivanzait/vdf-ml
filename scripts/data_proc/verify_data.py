#python scripts/data_proc/verify_data.py
#
# Edit src/data_proc/pipeline_config.py to change the snapshot/search-box/
# detector settings -- shared with extract_data.py so they can never
# disagree about what "this run" extracted and labeled.
#
# Visualization stages 2/3: one representative VDF per label (chosen
# uniformly at random from that label's cells), from the same ground truth
# extract_data.py assigns. Two plots:
#   2. where each representative VDF sits spatially, on a colormap
#   3. that VDF's three velocity-space cuts (vx-vy, vx-vz, vy-vz), sliced
#      through its own peak
#
# See also plot_nulls.py (X/O detector sanity check) and plot_vdf_hermite.py
# (manual VDF/Hermite/rotation drill-down) for other verification angles.
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

from src.data_proc import pipeline_config as config
from src.data_proc.labeling.snapshot_labeling import (
    compute_snapshot_ground_truth,
    pick_cluster_representative_cellids,
)
from src.data_proc.plot_tools import plot_cluster_vdf_examples, plot_cluster_vdf_positions

OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "verify_data" / config.RUN_ID


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = pt.vlsvfile.VlsvReader(config.FILE_LOCATION)

    print("Computing snapshot ground truth (active point substances + Shue-model regions)...")
    cellids, coords_re, labels, shue_fit, point_substance_records = (
        compute_snapshot_ground_truth(
            reader=reader,
            flux_file_location=config.FLUX_FILE_LOCATION,
            points_config=config.POINTS_CONFIG,
            regions_re=config.REGIONS_RE,
        )
    )
    print(f"{len(cellids)} VDF cells across {len(set(labels.tolist()))} labels")

    representative_cellids = pick_cluster_representative_cellids(
        vdf_cellids=cellids,
        labels=labels,
        random_state=config.RANDOM_STATE,
    )
    print(f"Representative cells: {representative_cellids}")

    # "On the fly": whichever point substances actually ran this snapshot
    # (points_config["active_point_substances"]) is what point_substance_records
    # holds keys for -- no hardcoded assumption about current_layer being
    # active. If it is, show its detected core on the positions plot.
    current_layer_records = point_substance_records.get("current_layer")

    positions_output_path = OUTPUT_DIR / "vdf_positions.png"
    plot_cluster_vdf_positions(
        file_location=config.FILE_LOCATION,
        reader=reader,
        representative_cellids=representative_cellids,
        output_path=positions_output_path,
        boxre=config.PLOT_BOXRE,
        current_layer_records=current_layer_records,
    )
    print(f"Saved VDF positions plot to: {positions_output_path}")

    examples_output_path = OUTPUT_DIR / "vdf_examples.png"
    plot_cluster_vdf_examples(
        reader=reader,
        representative_cellids=representative_cellids,
        output_path=examples_output_path,
        pop=config.POP,
        vdflim=config.VDFLIM,
    )
    print(f"Saved VDF examples plot to: {examples_output_path}")


if __name__ == "__main__":
    main()
