import os
import time
import mmap
import tempfile
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

from src.data_proc.dataset_sampling import (
    create_timestep_sample_specs_for_timestep,
    iter_chunks,
    iter_timestep_sample_specs,
    print_memory_usage,
    write_timestep_samples,
)

def release_memmap_pages(array):

    mmap_object = getattr(array, "_mmap", None)
    madvise = getattr(mmap_object, "madvise", None)
    dontneed = getattr(mmap, "MADV_DONTNEED", None)

    if madvise is None or dontneed is None:
        return False

    try:
        madvise(dontneed)
    except (OSError, ValueError):
        return False

    return True

def flush_and_release_memmaps(*arrays):

    for array in arrays:
        array.flush()

    for array in arrays:
        release_memmap_pages(array)

def get_worker_count(n_jobs, config_name):

    if n_jobs == 0:
        raise ValueError(f"{config_name} must be non-zero")

    if n_jobs < 0:
        return os.cpu_count() or 1

    return max(1, n_jobs)

def plan_dataset_sample_specs(config, timesteps, planning_n_jobs):

    print_memory_usage("before planning")
    planning_start = time.perf_counter()

    if planning_n_jobs == 1:
        sample_spec_results = [
            create_timestep_sample_specs_for_timestep(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        ]
    else:
        sample_spec_results = Parallel(n_jobs=planning_n_jobs)(
            delayed(create_timestep_sample_specs_for_timestep)(
                config=config,
                timestep=timestep,
            )
            for timestep in timesteps
        )

    return dict(sample_spec_results), time.perf_counter() - planning_start

def find_first_nonempty_timestep(sample_counts_by_timestep, timesteps):

    for timestep_index, timestep in enumerate(timesteps):
        if sample_counts_by_timestep[int(timestep)] > 0:
            return timestep_index, int(timestep)

    raise ValueError("No samples were found for the requested timesteps")

def extract_first_sample_from_specs(sample_specs):

    print_memory_usage("before first extraction")
    sample_iter = iter_timestep_sample_specs(sample_specs)

    try:
        first_sample = next(sample_iter)
    except StopIteration as error:
        raise ValueError("No samples were found for the first timestep") from error

    print_memory_usage("after first extraction")

    return first_sample, sample_iter

def write_remaining_timesteps(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):

    if extraction_n_jobs == 1:
        return write_timesteps_serial(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=X,
            y=y,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
        )

    return write_timesteps_parallel(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        y=y,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )

def write_timesteps_serial(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
):

    for timestep in timesteps:
        sample_specs = sample_specs_by_timestep.pop(int(timestep))
        sample_index = write_timestep_samples(
            X=X,
            y=y,
            metadata=metadata,
            timestep_samples=iter_timestep_sample_specs(sample_specs),
            sample_index=sample_index,
        )
        flush_and_release_memmaps(X, y)
        print_memory_usage(f"after timestep {int(timestep)} memmap release")

    return sample_index

def write_timesteps_parallel(
    sample_specs_by_timestep,
    X,
    y,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):

    with tempfile.TemporaryDirectory(
        prefix="extraction_",
        dir=Path(X.filename).parent,
    ) as temp_dir:
        temp_dir = Path(temp_dir)

        for timestep_chunk in iter_chunks(timesteps, extraction_worker_count):
            chunk_start = int(timestep_chunk[0])
            chunk_end = int(timestep_chunk[-1])
            print_memory_usage(
                f"before extraction chunk {chunk_start}-{chunk_end}"
            )
            chunk_specs_by_timestep = {
                int(timestep): sample_specs_by_timestep.pop(int(timestep))
                for timestep in timestep_chunk
            }
            chunk_results = extract_timestep_chunk_parallel(
                sample_specs_by_timestep=chunk_specs_by_timestep,
                timestep_chunk=timestep_chunk,
                extraction_n_jobs=extraction_n_jobs,
                temp_dir=temp_dir,
                sample_shape=X.shape[1:],
                dtype=X.dtype,
            )
            print_memory_usage(
                f"after extraction chunk {chunk_start}-{chunk_end}"
            )

            for extracted_timestep in chunk_results:
                sample_index = write_extracted_timestep(
                    X=X,
                    y=y,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    sample_index=sample_index,
                )

            print_memory_usage(f"after writing chunk {chunk_start}-{chunk_end}")
            flush_and_release_memmaps(X, y)
            print_memory_usage(
                f"after memmap release chunk {chunk_start}-{chunk_end}"
            )

    return sample_index

def extract_timestep_samples_to_temp(sample_specs, temp_dir, sample_shape, dtype):

    if not sample_specs:
        return {
            "n_samples": 0,
            "metadata": [],
            "X_path": None,
            "y_path": None,
        }

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir)
    X_path = temp_dir / f"timestep_{timestep}_X.npy"
    y_path = temp_dir / f"timestep_{timestep}_y.npy"

    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *sample_shape),
    )
    y_temp = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(len(sample_specs),),
    )

    metadata = []
    sample_index = write_timestep_samples(
        X=X_temp,
        y=y_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(sample_specs),
        sample_index=0,
    )

    if sample_index != len(sample_specs):
        raise RuntimeError(
            f"Expected to extract {len(sample_specs)} samples for timestep "
            f"{timestep}, extracted {sample_index}"
        )

    flush_and_release_memmaps(X_temp, y_temp)

    return {
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
        "y_path": str(y_path),
    }

def write_extracted_timestep(X, y, metadata, extracted_timestep, sample_index):

    n_samples = int(extracted_timestep["n_samples"])
    if n_samples == 0:
        return sample_index

    timestep_metadata = extracted_timestep["metadata"]
    if len(timestep_metadata) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} metadata rows, got {len(timestep_metadata)}"
        )

    X_temp = np.load(extracted_timestep["X_path"], mmap_mode="r")
    y_temp = np.load(extracted_timestep["y_path"], mmap_mode="r")
    write_end = sample_index + n_samples

    X[sample_index:write_end] = X_temp[:n_samples]
    y[sample_index:write_end] = y_temp[:n_samples]

    for metadata_row in timestep_metadata:
        output_row = dict(metadata_row)
        output_row["sample_index"] = sample_index + int(
            metadata_row["sample_index"]
        )
        metadata.append(output_row)

    release_memmap_pages(X_temp)
    release_memmap_pages(y_temp)

    return write_end

def extract_timestep_chunk_parallel(
    sample_specs_by_timestep,
    timestep_chunk,
    extraction_n_jobs,
    temp_dir,
    sample_shape,
    dtype,
):

    try:
        parallel = Parallel(n_jobs=extraction_n_jobs, return_as="generator")
    except TypeError:
        parallel = Parallel(n_jobs=extraction_n_jobs)

    return parallel(
        delayed(extract_timestep_samples_to_temp)(
            sample_specs=sample_specs_by_timestep[int(timestep)],
            temp_dir=temp_dir,
            sample_shape=sample_shape,
            dtype=dtype,
        )
        for timestep in timestep_chunk
    )

