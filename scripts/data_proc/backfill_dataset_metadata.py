# Example: python scripts/backfill_dataset_metadata.py --config CONFIG --dataset-dir DATASET

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data_proc.config import load_config
from src.data_proc.dataset_metadata import backfill_dataset_spatial_metadata


def main(config_path, dataset_dir, n_jobs=4):
    """
    Recompute topology and backfill dataset spatial metadata.

    Parameters
    ----------
    config_path : str
        Path to the dataset creation YAML config.
    dataset_dir : str
        Dataset directory containing ``metadata.csv``.
    n_jobs : int, optional
        Number of parallel timestep processes.
    """

    config = load_config(config_path)
    metadata_path = backfill_dataset_spatial_metadata(
        dataset_dir=dataset_dir,
        config=config,
        n_jobs=n_jobs,
    )
    print(f"Updated: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerun topology detection and backfill hybrid X/O-point geometry."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Dataset creation YAML config used for bulk/flux topology.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory whose metadata.csv is updated in place.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel timestep workers (default: 4; use 1 for serial).",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        n_jobs=args.n_jobs,
    )
