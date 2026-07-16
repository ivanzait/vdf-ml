import math

import numpy as np

DEFAULT_HERMITE_ORDER = 22


def hermite_polynomials(v_axis, order):
    """
    Evaluate Gaussian-weighted physicists' Hermite polynomials.

    Parameters
    ----------
    v_axis : numpy.ndarray
        Normalized velocity coordinates, shape ``(n,)``.
    order : int
        Number of Hermite orders to evaluate (``0`` to ``order - 1``).

    Returns
    -------
    numpy.ndarray
        Hermite values with shape ``(order, n)``.
    """

    v_axis = np.asarray(v_axis)
    hp = np.zeros((order, len(v_axis)))
    hp[0, :] = np.exp(-0.5 * v_axis**2)
    if order > 1:
        hp[1, :] = 2 * v_axis * np.exp(-0.5 * v_axis**2)
    for n in range(2, order):
        hp[n, :] = 2 * v_axis * hp[n - 1, :] - 2 * (n - 1) * hp[n - 2, :]
    return hp


def normalized_hermite_basis(shape, v_limits, order, vth, u, axis):
    """
    Build the normalized Hermite basis along one velocity axis.

    Parameters
    ----------
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    order : int
        Number of Hermite orders to evaluate.
    vth : float
        Thermal velocity used to normalize the velocity axis.
    u : array-like of float
        Bulk drift velocity ``[ux, uy, uz]``.
    axis : int
        Axis index (``0`` for vx, ``1`` for vy, ``2`` for vz).

    Returns
    -------
    numpy.ndarray
        Normalized Hermite basis with shape ``(order, shape[axis])``.
    """

    v_axis = np.linspace(v_limits[axis], v_limits[axis + 3], shape[axis])
    v_axis = (v_axis - u[axis]) / vth
    hermite_vals = hermite_polynomials(v_axis, order)
    for n in range(order):
        norm_const = math.sqrt((2**n) * math.factorial(n) * math.sqrt(math.pi) * vth)
        hermite_vals[n, :] /= norm_const
    return hermite_vals


def compute_drift_velocity(vdf, shape, v_limits):
    """
    Compute the bulk drift velocity of a 3D VDF.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with axis order ``[vx, vy, vz]`` and shape ``shape``.
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.

    Returns
    -------
    numpy.ndarray
        Drift velocity ``[ux, uy, uz]``.
    """

    vx = np.linspace(v_limits[0], v_limits[3], shape[0])
    vy = np.linspace(v_limits[1], v_limits[4], shape[1])
    vz = np.linspace(v_limits[2], v_limits[5], shape[2])

    dv1 = (v_limits[3] - v_limits[0]) / shape[0]
    dv2 = (v_limits[4] - v_limits[1]) / shape[1]
    dv3 = (v_limits[5] - v_limits[2]) / shape[2]
    cell_volume = dv1 * dv2 * dv3

    n = float(vdf.sum()) * cell_volume
    if n == 0:
        return np.zeros(3)

    u = np.array(
        [
            np.sum(vdf * vx[:, None, None]),
            np.sum(vdf * vy[None, :, None]),
            np.sum(vdf * vz[None, None, :]),
        ]
    ) * cell_volume

    return u / n


def compute_thermal_velocity(vdf, shape, v_limits, u):
    """
    Compute the isotropic thermal velocity of a 3D VDF.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with axis order ``[vx, vy, vz]`` and shape ``shape``.
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    u : array-like of float
        Drift velocity ``[ux, uy, uz]``.

    Returns
    -------
    float
        Thermal velocity, or ``0.0`` if the VDF has zero density.
    """

    vx = np.linspace(v_limits[0], v_limits[3], shape[0])
    vy = np.linspace(v_limits[1], v_limits[4], shape[1])
    vz = np.linspace(v_limits[2], v_limits[5], shape[2])

    dv = (v_limits[3] - v_limits[0]) / shape[0]
    dv3 = dv**3

    n = float(vdf.sum()) * dv3
    if n == 0:
        return 0.0

    Pxx = np.sum(vdf * (vx[:, None, None] - u[0]) ** 2) * dv3
    Pyy = np.sum(vdf * (vy[None, :, None] - u[1]) ** 2) * dv3
    Pzz = np.sum(vdf * (vz[None, None, :] - u[2]) ** 2) * dv3

    return math.sqrt((Pxx + Pyy + Pzz) / (3 * n))


def compute_hermite_spectra(vdf, shape, v_limits, order, vth, u):
    """
    Project a 3D VDF onto a separable Hermite basis.

    Vectorized, mathematically equivalent to summing
    ``vdf[i, j, k] * Hx[nx, i] * Hy[ny, j] * Hz[nz, k] * dv**3`` over the
    velocity grid for every Hermite mode ``(nx, ny, nz)``.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with axis order ``[vx, vy, vz]`` and shape ``shape``.
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    order : int
        Number of Hermite modes per axis.
    vth : float
        Thermal velocity used to normalize the velocity axes.
    u : array-like of float
        Drift velocity ``[ux, uy, uz]``.

    Returns
    -------
    numpy.ndarray
        Hermite spectra with shape ``(order, order, order)``.
    """

    dv = (v_limits[3] - v_limits[0]) / shape[0]

    hermite_x = normalized_hermite_basis(shape, v_limits, order, vth, u, 0)
    hermite_y = normalized_hermite_basis(shape, v_limits, order, vth, u, 1)
    hermite_z = normalized_hermite_basis(shape, v_limits, order, vth, u, 2)

    spectra = np.einsum(
        "ijk,ni,mj,lk->nml",
        vdf,
        hermite_x,
        hermite_y,
        hermite_z,
        optimize=True,
    )

    return spectra * dv**3


def vdf_to_hermite_spectra(vdf, shape, v_limits, order=DEFAULT_HERMITE_ORDER):
    """
    Convert a dense 3D VDF into its Hermite spectra.

    Parameters
    ----------
    vdf : numpy.ndarray
        Dense VDF array with axis order ``[vx, vy, vz]`` and shape ``shape``.
    shape : tuple of int
        VDF grid shape ``(nx, ny, nz)``.
    v_limits : array-like of float
        Velocity mesh extent ``[vxmin, vymin, vzmin, vxmax, vymax, vzmax]``.
    order : int, optional
        Number of Hermite modes per axis.

    Returns
    -------
    numpy.ndarray
        Hermite spectra with shape ``(order, order, order)``.
    """

    u = compute_drift_velocity(vdf, shape, v_limits)
    vth = compute_thermal_velocity(vdf, shape, v_limits, u)

    if vth == 0:
        return np.zeros((order, order, order))

    return compute_hermite_spectra(vdf, shape, v_limits, order, vth, u)
