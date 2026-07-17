#python scripts/plot_dataset_labels.py
#
# Edit the PARAMETERS block below, then run. This does NOT reimplement
# anything -- it just calls the existing label-scatter plotting machinery:
#   src/plot_dataset.py: create_colormap_plot_jobs, plot_labeled_colormap, run_plot_jobs
#   src/plot_helpers.py: scatter_label_points (per-class marker/color styles),
#                             scatter_all_vdf_cells, draw_*_search_areas/boxes
# One PNG is produced per distinct timestep found in the dataset's
# metadata.csv, with every sampled point scattered and colored/marker-styled
# by its class_name (see SOURCE_POINT_STYLES in plot_helpers.py).
import os
import sys
from pathlib import Path

os.environ["PTNOLATEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

try:
    import analysator  # noqa: F401  (only checking it resolves before src imports)
except ImportError:
    # Falls back to a local analysator checkout when it isn't installed as
    # a package. Override with the ANALYSATOR_PATH env var.
    analysator_path = os.environ.get(
        "ANALYSATOR_PATH",
        "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator",
    )
    sys.path.append(analysator_path)

from src.data_proc.dataset_creation import load_dataset
from src.data_proc.plot_dataset import (
    create_colormap_plot_jobs,
    plot_labeled_colormap,
    run_plot_jobs,
)

# ----------------------------- PARAMETERS -----------------------------

DATASET_DIR = PROJECT_ROOT / "data" / "local_smoke_test" / "train" / "timesteps_3408_1"
OUTPUT_DIR = DATASET_DIR / "plots"

COLORMAP_CONFIG = {
    "boxre": [-30, 15, -13, 13],
    "vmin": -1500000.0,
    "vmax": 1500000.0,
    # Optional extras, taken from the dataset-creation config's `points`
    # block -- fill these in to also draw X/O-point search boxes/areas
    # (see add_dataset_sampling_plot_config in src/plot_dataset.py, which
    # does this automatically from a dataset-creation YAML path if you'd
    # rather point at one instead of copying values here):
    # "box_classes": ["x_point", "o_point"],
    # "x_selection": {...},              # points.x_selection from create_dataset config
    # "file_template_flux": "/Users/ivanzait/Downloads/bulk.000{timestep:04d}.bin",
    # "o_core_fraction": 0.7,            # points.o_selection.core_fraction
}

N_JOBS = 1

# ------------------------------------------------------------------------


def main():
    _, _, metadata = load_dataset(DATASET_DIR, mmap=True)
    print(f"Loaded metadata: {len(metadata)} samples, "
          f"{metadata['timestep'].nunique()} distinct timestep(s)")
    print("Class counts:")
    print(metadata["class_name"].value_counts().to_string())

    colormap_jobs = create_colormap_plot_jobs(
        metadata=metadata,
        output_dir=OUTPUT_DIR,
        colormap_config=COLORMAP_CONFIG,
    )
    run_plot_jobs(
        plot_function=plot_labeled_colormap,
        plot_jobs=colormap_jobs,
        n_jobs=N_JOBS,
    )
    print(f"Saved {len(colormap_jobs)} labeled colormap plot(s) to: {OUTPUT_DIR / 'colormaps'}")


if __name__ == "__main__":
    main()
