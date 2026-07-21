from pathlib import Path
import time

from joblib import Parallel, delayed
import numpy as np

from src.data_proc.batches import iter_index_batches
from src.data_proc.config import create_path


def resolve_cache_config(config, dataset_dir, dataset_id, model_id):
    """
    Resolve VDF log-slice cache settings.

    Parameters
    ----------
    config : dict
        Cache configuration.
    dataset_dir : str or pathlib.Path
        Dataset directory.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.

    Returns
    -------
    dict
        Validated cache settings.
    """

    enabled = bool(config.get("enabled", False))
    if config.get("dir") is None:
        cache_dir = Path(dataset_dir) / "cache"
    else:
        cache_dir = create_path(
            path_template=config["dir"],
            dataset_id=dataset_id,
            model_id=model_id,
        )

    filename = config.get("filename", "xz_log_cache.npy")
    metadata_filename = config.get(
        "metadata_filename",
        "xz_log_cache_metadata.npz",
    )
    batch_size = int(config.get("batch_size", 64))
    n_jobs = int(config.get("n_jobs", 1))

    if batch_size <= 0:
        raise ValueError("cache.batch_size must be positive")
    if n_jobs == 0:
        raise ValueError("cache.n_jobs must be non-zero")

    return {
        "enabled": enabled,
        "cache_dir": Path(cache_dir),
        "cache_path": Path(cache_dir) / filename,
        "metadata_path": Path(cache_dir) / metadata_filename,
        "rebuild": bool(config.get("rebuild", False)),
        "batch_size": batch_size,
        "n_jobs": n_jobs,
    }


def create_or_load_log_slice_cache(X, input_config, cache_config):
    """
    Create or load cached log-scaled xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_config : dict
        Cache settings.

    Returns
    -------
    X_log : numpy.ndarray or None
        Cached log slices, or ``None`` when caching is disabled.
    cache_metadata : dict
        Cache metadata.
    """

    if not cache_config["enabled"]:
        return None, {"enabled": False}

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]

    if (
        not cache_config["rebuild"]
        and is_log_slice_cache_valid(
            X=X,
            input_config=input_config,
            cache_path=cache_path,
            metadata_path=metadata_path,
        )
    ):
        X_log = np.load(cache_path, mmap_mode="r")
        return X_log, load_log_slice_cache_metadata(metadata_path)

    cache_metadata = create_log_slice_cache(
        X=X,
        input_config=input_config,
        cache_config=cache_config,
    )
    X_log = np.load(cache_path, mmap_mode="r")

    return X_log, cache_metadata


def create_log_slice_cache(X, input_config, cache_config):
    """
    Create a memory-mapped cache of log-scaled xz VDF slices.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_config : dict
        Cache settings.

    Returns
    -------
    dict
        Cache metadata.
    """

    cache_path = cache_config["cache_path"]
    metadata_path = cache_config["metadata_path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_shape = (int(X.shape[0]), 1, *infer_plot_xz_slice_shape(X))
    X_log = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=cache_shape,
    )

    start = time.perf_counter()
    sample_indices = np.arange(int(X.shape[0]), dtype=int)
    batches = list(
        iter_index_batches(
            sample_indices,
            batch_size=cache_config["batch_size"],
        )
    )

    print(f"Creating xz log-slice cache: {cache_path}")
    print(f"Cache shape: {cache_shape}")
    print(f"Cache jobs: {cache_config['n_jobs']}")

    if cache_config["n_jobs"] == 1:
        for batch_indices in batches:
            write_log_slice_cache_batch(
                X=X,
                X_log=X_log,
                batch_indices=batch_indices,
                input_config=input_config,
            )
    else:
        Parallel(
            n_jobs=cache_config["n_jobs"],
            prefer="threads",
            require="sharedmem",
        )(
            delayed(write_log_slice_cache_batch)(
                X=X,
                X_log=X_log,
                batch_indices=batch_indices,
                input_config=input_config,
            )
            for batch_indices in batches
        )

    X_log.flush()
    elapsed = time.perf_counter() - start
    metadata = {
        "enabled": True,
        "cache_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "raw_vdf_shape": tuple(int(value) for value in X.shape),
        "cache_shape": tuple(int(value) for value in cache_shape),
        "log_eps": float(input_config["log_eps"]),
        "clip_negative_to_zero": bool(input_config["clip_negative_to_zero"]),
        "slice": input_config["slice"],
        "orientation": input_config["orientation"],
        "elapsed_seconds": float(elapsed),
    }
    save_log_slice_cache_metadata(metadata_path, metadata)

    print(f"Created xz log-slice cache in {elapsed:.2f} s")

    return metadata


def write_log_slice_cache_batch(X, X_log, batch_indices, input_config):
    """
    Write one batch of log-scaled xz slices into the cache.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    X_log : numpy.ndarray
        Log-slice cache.
    batch_indices : array-like of int
        Sample indices to write.
    input_config : dict
        VDF preprocessing settings.
    """

    for sample_index in batch_indices:
        X_log[int(sample_index), 0] = create_log_plot_xz_slice_from_dataset(
            X=X,
            sample_index=int(sample_index),
            log_eps=input_config["log_eps"],
            clip_negative_to_zero=input_config["clip_negative_to_zero"],
        )


def is_log_slice_cache_valid(X, input_config, cache_path, metadata_path):
    """
    Return whether an existing log-slice cache matches current settings.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    input_config : dict
        VDF preprocessing settings.
    cache_path : str or pathlib.Path
        Cache array path.
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    bool
        Whether the cache can be reused.
    """

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    if not cache_path.exists() or not metadata_path.exists():
        return False

    try:
        X_log = np.load(cache_path, mmap_mode="r")
        metadata = load_log_slice_cache_metadata(metadata_path)
    except Exception:
        return False

    expected_shape = (int(X.shape[0]), 1, *infer_plot_xz_slice_shape(X))

    return (
        tuple(X_log.shape) == expected_shape
        and tuple(metadata.get("raw_vdf_shape", ())) == tuple(X.shape)
        and tuple(metadata.get("cache_shape", ())) == expected_shape
        and float(metadata.get("log_eps", np.nan)) == float(input_config["log_eps"])
        and bool(metadata.get("clip_negative_to_zero")) == bool(
            input_config["clip_negative_to_zero"]
        )
        and metadata.get("slice") == input_config["slice"]
        and metadata.get("orientation") == input_config["orientation"]
    )


def save_log_slice_cache_metadata(metadata_path, metadata):
    """
    Save log-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.
    metadata : dict
        Cache metadata.
    """

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        metadata_path,
        enabled=np.asarray(metadata["enabled"]),
        cache_path=np.asarray(metadata["cache_path"]),
        metadata_path=np.asarray(metadata["metadata_path"]),
        raw_vdf_shape=np.asarray(metadata["raw_vdf_shape"], dtype=int),
        cache_shape=np.asarray(metadata["cache_shape"], dtype=int),
        log_eps=np.asarray(metadata["log_eps"]),
        clip_negative_to_zero=np.asarray(metadata["clip_negative_to_zero"]),
        slice=np.asarray(metadata["slice"]),
        orientation=np.asarray(metadata["orientation"]),
        elapsed_seconds=np.asarray(metadata["elapsed_seconds"]),
    )


def load_log_slice_cache_metadata(metadata_path):
    """
    Load log-slice cache metadata.

    Parameters
    ----------
    metadata_path : str or pathlib.Path
        Cache metadata path.

    Returns
    -------
    dict
        Cache metadata.
    """

    with np.load(metadata_path, allow_pickle=False) as metadata:
        return {
            "enabled": bool(metadata["enabled"].item()),
            "cache_path": str(metadata["cache_path"].item()),
            "metadata_path": str(metadata["metadata_path"].item()),
            "raw_vdf_shape": tuple(
                int(value) for value in metadata["raw_vdf_shape"]
            ),
            "cache_shape": tuple(int(value) for value in metadata["cache_shape"]),
            "log_eps": float(metadata["log_eps"].item()),
            "clip_negative_to_zero": bool(
                metadata["clip_negative_to_zero"].item()
            ),
            "slice": str(metadata["slice"].item()),
            "orientation": str(metadata["orientation"].item()),
            "elapsed_seconds": float(metadata["elapsed_seconds"].item()),
        }


def infer_plot_xz_slice_shape(X):
    """
    Infer plot-oriented xz-slice shape from a VDF dataset.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.

    Returns
    -------
    tuple of int
        Shape of one plot-oriented xz slice.
    """

    if len(X.shape) != 4:
        raise ValueError("X must have shape (n_samples, vx, vy, vz)")

    return int(X.shape[3]), int(X.shape[1])


def extract_plot_xz_slice_from_dataset(X, sample_index):
    """
    Extract the plot-oriented middle xz slice from a saved VDF sample.

    This matches ``src.data_proc.plot_tools.extract_plot_xz_slice`` without
    reading the full 3D VDF into memory.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.

    Returns
    -------
    numpy.ndarray
        Plot-oriented xz slice with shape ``(vz, vx)``.
    """

    mid_y = X.shape[2] // 2
    return np.asarray(X[int(sample_index), :, mid_y, :].T, dtype=np.float32)


def create_log_plot_xz_slice_from_dataset(
    X,
    sample_index,
    log_eps=1e-30,
    clip_negative_to_zero=True,
):
    """
    Create a log-scaled plot-oriented xz slice from a saved sample.

    Parameters
    ----------
    X : numpy.ndarray
        VDF samples with shape ``(n_samples, vx, vy, vz)``.
    sample_index : int
        Dataset sample index.
    log_eps : float, optional
        Small positive value added before log scaling.
    clip_negative_to_zero : bool, optional
        Whether to clip tiny negative values to zero before log scaling.

    Returns
    -------
    numpy.ndarray
        Log-scaled xz slice.
    """

    return create_log_slice(
        physical_slice=extract_plot_xz_slice_from_dataset(
            X=X,
            sample_index=sample_index,
        ),
        log_eps=log_eps,
        clip_negative_to_zero=clip_negative_to_zero,
    )


def create_log_slice(physical_slice, log_eps=1e-30, clip_negative_to_zero=True):
    """
    Convert a physical VDF slice to log10 scale.

    Parameters
    ----------
    physical_slice : numpy.ndarray
        Physical VDF slice.
    log_eps : float, optional
        Small positive value added before log scaling.
    clip_negative_to_zero : bool, optional
        Whether to clip negative values to zero before log scaling.

    Returns
    -------
    numpy.ndarray
        Log-scaled slice.
    """

    physical_slice = np.asarray(physical_slice, dtype=np.float32)
    if clip_negative_to_zero:
        physical_slice = np.maximum(physical_slice, 0.0)

    return np.log10(physical_slice + float(log_eps)).astype(
        np.float32,
        copy=False,
    )


def normalize_log_slice(slice_log, mean, std):
    """
    Normalize one log-scaled VDF slice.

    Parameters
    ----------
    slice_log : numpy.ndarray
        Log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.

    Returns
    -------
    numpy.ndarray
        Normalized slice.
    """

    return ((slice_log - float(mean)) / float(std)).astype(
        np.float32,
        copy=False,
    )


def denormalize_log_slice(normalized_slice, mean, std):
    """
    Convert a normalized log slice back to log10 scale.

    Parameters
    ----------
    normalized_slice : numpy.ndarray
        Normalized log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.

    Returns
    -------
    numpy.ndarray
        Log10-scaled VDF slice.
    """

    return np.asarray(normalized_slice, dtype=np.float32) * float(std) + float(mean)


def log_slice_to_physical(slice_log, log_eps=1e-30):
    """
    Convert a log10 VDF slice back to physical values.

    Parameters
    ----------
    slice_log : numpy.ndarray
        Log10-scaled VDF slice.
    log_eps : float, optional
        Small positive offset used during log scaling.

    Returns
    -------
    numpy.ndarray
        Non-negative physical VDF slice.
    """

    slice_log = np.asarray(slice_log, dtype=np.float64)
    slice_log = np.clip(slice_log, -300.0, 30.0)
    physical_slice = np.power(10.0, slice_log) - float(log_eps)
    physical_slice = np.maximum(physical_slice, 0.0)

    return np.asarray(physical_slice, dtype=np.float32)


def normalized_log_slice_to_physical(
    normalized_slice,
    mean,
    std,
    log_eps=1e-30,
):
    """
    Convert a normalized log slice back to physical values.

    Parameters
    ----------
    normalized_slice : numpy.ndarray
        Normalized log-scaled VDF slice.
    mean : float
        Training-set mean.
    std : float
        Training-set standard deviation.
    log_eps : float, optional
        Small positive offset used during log scaling.

    Returns
    -------
    numpy.ndarray
        Non-negative physical VDF slice.
    """

    return log_slice_to_physical(
        slice_log=denormalize_log_slice(
            normalized_slice=normalized_slice,
            mean=mean,
            std=std,
        ),
        log_eps=log_eps,
    )
