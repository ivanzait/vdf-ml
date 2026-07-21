"""Generic, label-scheme-agnostic I/O for saved VDF datasets (X.npy/y.npy/metadata.csv)."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_proc.batches import iter_array_batches


def save_metadata(outdir, metadata):
    """
    Save dataset metadata.

    Parameters
    ----------
    outdir : str
        Directory where metadata file is saved.
    metadata : list of dict
        Metadata rows, one row per sample.
    """

    outdir = Path(outdir)

    pd.DataFrame(metadata).to_csv(
        outdir / "metadata.csv",
        index=False
    )


def load_dataset(dataset_dir, mmap=True):
    """
    Load saved VDF dataset.

    Parameters
    ----------
    dataset_dir : str
        Directory containing dataset arrays and metadata.
    mmap : bool, optional
        Whether to load the dataset as a memory-mapped array.

    Returns
    -------
    X : numpy.ndarray
        VDF samples.
    y : numpy.ndarray
        Integer labels.
    metadata : pandas.DataFrame
        Metadata table for the samples.
    """

    dataset_dir = Path(dataset_dir)

    X_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    metadata_path = dataset_dir / "metadata.csv"

    if mmap:
        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
    else:
        X = np.load(X_path)
        y = np.load(y_path)

    metadata = pd.read_csv(metadata_path)

    return X, y, metadata


def load_labeled_vdfs(dataset_dir, mmap=True, x_filename="X.npy"):
    """
    Load a snapshot dataset saved by labeling.snapshot_labeling.save_labeled_vdfs:
    X.npy + metadata.csv, no y.npy (label lives in metadata's ``label`` column).

    Parameters
    ----------
    dataset_dir : str
        Directory containing X.npy and metadata.csv.
    mmap : bool, optional
        Whether to load X as a memory-mapped array.
    x_filename : str, optional
        Which VDF array to load, e.g. "X_rotated.npy" instead of the
        default "X.npy" -- see save_labeled_vdfs' X_rotated parameter.
        Raises FileNotFoundError with a clear message if it doesn't exist
        (e.g. extract_data.py's BUILD_ROTATED_DATASET was off for this run).

    Returns
    -------
    X : numpy.ndarray
        VDF samples.
    metadata : pandas.DataFrame
        Metadata table for the samples, including the ``label`` column.
    """

    dataset_dir = Path(dataset_dir)
    x_path = dataset_dir / x_filename
    if not x_path.exists():
        raise FileNotFoundError(
            f"{x_path} not found -- re-run extract_data.py with the setting that produces it "
            f"(e.g. pipeline_config.BUILD_ROTATED_DATASET = True for X_rotated.npy)."
        )

    if mmap:
        X = np.load(x_path, mmap_mode="r")
    else:
        X = np.load(x_path)

    metadata = pd.read_csv(dataset_dir / "metadata.csv")

    return X, metadata


def print_vdf_statistics(X, batch_size=64):
    """
    Print statistics for VDF samples.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    batch_size : int
        Size of batches to process.
    """
    global_min = np.inf
    global_max = -np.inf
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    for _, batch in iter_array_batches(X, batch_size=batch_size):
        batch = np.asarray(batch, dtype=np.float64)

        global_min = min(global_min, batch.min())
        global_max = max(global_max, batch.max())

        total_sum += batch.sum()
        total_sum_sq += (batch ** 2).sum()
        total_count += batch.size

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    variance = max(variance, 0.0)
    std = np.sqrt(variance)

    print("VDF statistic:")
    print(f"min: {global_min}")
    print(f"max: {global_max}")
    print(f"mean: {mean}")
    print(f"std: {std}")
