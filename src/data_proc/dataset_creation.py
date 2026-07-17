import mmap
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_proc.batches import iter_array_batches
from src.data_proc.dataset_extraction import (
    extract_first_sample_from_specs,
    find_first_nonempty_timestep,
    flush_and_release_memmaps,
    get_worker_count,
    plan_dataset_sample_specs,
    write_remaining_timesteps,
)
from src.data_proc.dataset_sampling import (
    create_timestep_sample_specs_for_timestep,
    iter_chunks,
    iter_timestep_sample_specs,
    print_memory_usage,
    write_timestep_samples,
)



def create_dataset(
    config,
    start_timestep,
    n_timesteps,
    dataset_kind,
):
    """
    Create and save a labeled VDF dataset.

    Parameters
    ----------
    config : dict
        Dataset creation config.
    start_timestep : int
        First timestep to include.
    n_timesteps : int
        Number of consecutive timesteps to process.
    dataset_kind : {"train", "test"}
        Output dataset split name.
    """

    total_start = time.perf_counter()
    timesteps = list(range(start_timestep, start_timestep + n_timesteps))

    output_dirs = config["output_dirs"]
    output_dir = output_dirs[dataset_kind]

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset kind: {dataset_kind}")
    print(f"Output directory: {outdir}")

    creation_config = config.get("creation", {})
    default_n_jobs = int(creation_config.get("n_jobs", 1))
    planning_n_jobs = int(
        creation_config.get("planning_n_jobs", default_n_jobs)
    )
    extraction_n_jobs = int(
        creation_config.get("extraction_n_jobs", 1)
    )

    planning_worker_count = get_worker_count(
        planning_n_jobs,
        "creation.planning_n_jobs",
    )
    extraction_worker_count = get_worker_count(
        extraction_n_jobs,
        "creation.extraction_n_jobs",
    )

    print(
        f"Planning jobs: {planning_n_jobs} "
        f"({planning_worker_count} workers)"
    )
    print(
        f"Extraction jobs: {extraction_n_jobs} "
        f"({extraction_worker_count} workers)"
    )

    sample_specs_by_timestep, planning_elapsed = plan_dataset_sample_specs(
        config=config,
        timesteps=timesteps,
        planning_n_jobs=planning_n_jobs,
    )
    sample_counts_by_timestep = {
        int(timestep): len(sample_specs)
        for timestep, sample_specs in sample_specs_by_timestep.items()
    }
    n_samples = sum(sample_counts_by_timestep.values())

    print(f"Samples: {n_samples}")
    print(f"Timing planning: {planning_elapsed:.2f} s")
    print_memory_usage("after planning")

    if n_samples == 0:
        raise ValueError("No samples were found for the requested timesteps")

    extraction_start = time.perf_counter()
    metadata = []
    sample_index = 0

    first_timestep_index, first_timestep = find_first_nonempty_timestep(
        sample_counts_by_timestep=sample_counts_by_timestep,
        timesteps=timesteps,
    )
    first_sample_specs = sample_specs_by_timestep[first_timestep]
    first_sample, first_sample_iter = extract_first_sample_from_specs(
        first_sample_specs
    )

    X, y = create_memmap_dataset(
        outdir=outdir,
        n_samples=n_samples,
        sample_shape=first_sample["vdf"].shape,
        dtype=np.float32,
    )
    print_memory_usage("after memmap creation")

    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=[first_sample],
        sample_index=sample_index,
    )
    sample_index = write_timestep_samples(
        X=X,
        y=y,
        metadata=metadata,
        timestep_samples=first_sample_iter,
        sample_index=sample_index,
    )
    sample_specs_by_timestep.pop(first_timestep, None)
    flush_and_release_memmaps(X, y)
    print_memory_usage("after first timestep memmap release")

    remaining_timesteps = timesteps[first_timestep_index + 1:]
    sample_index = write_remaining_timesteps(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=remaining_timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )

    if sample_index != n_samples:
        raise RuntimeError(
            f"Expected to write {n_samples} samples, wrote {sample_index}"
        )

    extraction_elapsed = time.perf_counter() - extraction_start
    save_start = time.perf_counter()

    print_memory_usage("before flush")
    flush_and_release_memmaps(X, y)
    print_memory_usage("after flush and memmap release")

    save_metadata(
        outdir=outdir,
        metadata=metadata,
    )

    save_elapsed = time.perf_counter() - save_start
    total_elapsed = time.perf_counter() - total_start

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Samples written: {sample_index}")
    print(f"Timing extraction/write: {extraction_elapsed:.2f} s")
    print(f"Timing save/flush: {save_elapsed:.2f} s")
    print(f"Timing total: {total_elapsed:.2f} s")

    print(f"Saved X: {outdir / 'X.npy'}")
    print(f"Saved y: {outdir / 'y.npy'}")
    print(f"Saved metadata: {outdir / 'metadata.csv'}")


def create_memmap_dataset(outdir, n_samples, sample_shape, dtype=np.float32):
    """
    Create memory-mapped dataset file.

    Parameters
    ----------
    outdir : str
        Directory where dataset file is saved.
    n_samples : int
        Number of samples in the dataset.
    sample_shape : tuple of int
        Shape of one VDF sample.
    dtype : data-type, optional
        Desired data type for the array.

    Returns
    -------
    X : numpy.memmap
        Memory mapped array for the VDF.
    y : numpy.memmap
        Memory mapped array for the labels.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.lib.format.open_memmap(
        outdir / "X.npy",
        mode="w+",
        dtype=dtype,
        shape=(int(n_samples), *sample_shape)
    )

    y = np.lib.format.open_memmap(
        outdir / "y.npy",
        mode="w+",
        dtype=np.int64,
        shape=(int(n_samples),),
    )

    return X, y


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


def print_dataset_info(X, y, metadata):
    """
    Print info about a loaded VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples.
    y : numpy.ndarray
        Integer labels.
    metadata : pandas.DataFrame
        Metadata table for the samples.
    """
    print("Dataset information:")
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y shape: {y.shape}")
    print(f"y dtype: {y.dtype}")
    print(f"metadata shape: {metadata.shape}")
    print("\n")

    print("Metadata:")
    print(metadata)
    print("\n")

    if "class_name" in metadata.columns:
        print("Class counts:")
        print(metadata["class_name"].value_counts())

    print("\n")


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
