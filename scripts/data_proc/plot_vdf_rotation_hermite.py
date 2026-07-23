#python scripts/data_proc/plot_vdf_rotation_hermite.py
#
# Edit src/data_proc/pipeline_config.py to change the snapshot/search-box/
# detector settings -- shared with extract_data.py/verify_data.py so this
# never disagrees with them about what "this run" labeled. SUBSTANCE_LABEL
# below picks which label's representative VDF gets visualized.
#
# One three-panel figure: raw VDF -- rotated into local (B, v_perp,
# B x v_perp) frame -- log-space Hermite spectra, for one representative
# cell of SUBSTANCE_LABEL. Computed live for that single cell (no saved
# X_rotated.npy/X_hermite.npy needed) -- see plot_tools.plot_vdf_rotation_hermite.
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
from src.data_proc.labeling.snapshot_labeling import compute_snapshot_ground_truth
from src.data_proc.plot_tools import plot_vdf_rotation_hermite

SUBSTANCE_LABEL = "magnetosheath"

OUTPUT_DIR = PROJECT_ROOT / "data" / "plots" / "vdf_rotation_hermite" / config.RUN_ID


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = pt.vlsvfile.VlsvReader(config.FILE_LOCATION)

    print("Computing snapshot ground truth (active point substances + Shue-model regions)...")
    cellids, coords_re, labels, _shue_fit, _point_substance_records = (
        compute_snapshot_ground_truth(
            reader=reader,
            flux_file_location=config.FLUX_FILE_LOCATION,
            points_config=config.POINTS_CONFIG,
            regions_re=config.REGIONS_RE,
        )
    )

    output_path = OUTPUT_DIR / f"vdf_rotation_hermite_{SUBSTANCE_LABEL}.png"
    plot_vdf_rotation_hermite(
        reader=reader,
        cellids=cellids,
        coords_re=coords_re,
        labels=labels,
        label=SUBSTANCE_LABEL,
        order=config.HERMITE_ORDER,
        random_state=config.RANDOM_STATE,
        output_path=output_path,
    )
    print(f"Saved rotation + Hermite summary for {SUBSTANCE_LABEL!r} to: {output_path}")


if __name__ == "__main__":
    main()
