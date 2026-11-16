"""Interface shared by the Python example problems."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

# points: shape=(n, d), axis 0 -> point, axis 1 -> coordinate.
POINTS_NDIM = 2


class IProblem(ABC):
    """Interface of every Python example problem.

    Every problem has the same interface as its C++ twin in
    ``gfo.cpp_problems``, so the two can be swapped in benchmarks.

    Attributes
    ----------
    name : str
        Problem name, the same as the class name of the C++ twin.
    f_opt : float
        Value of the known minimum.
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    __slots__ = ()

    name: ClassVar[str]
    f_opt: ClassVar[float]
    lo: float
    hi: float

    @abstractmethod
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``.

        Parameters
        ----------
        points : np.ndarray
            Points to evaluate, shape (n, d), dtype float64. Axis 0 is the
            point, axis 1 the coordinate. ``points`` is not modified.

        Returns
        -------
        np.ndarray
            The value of every point, shape (n,), dtype float64.

        Raises
        ------
        ValueError
            If ``points`` is not two dimensional.
        """

    @abstractmethod
    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum.

        Parameters
        ----------
        d : int
            Number of dimensions.

        Returns
        -------
        np.ndarray
            Location of the minimum, shape (d,).
        """


def validate_points(points: np.ndarray) -> np.ndarray:
    """Return ``points`` as a float64 array of shape (n, d).

    No copy is made when ``points`` is already a float64 array.

    Parameters
    ----------
    points : np.ndarray
        Points, shape (n, d).

    Returns
    -------
    np.ndarray
        The same points, shape (n, d), dtype float64.

    Raises
    ------
    ValueError
        If ``points`` is not two dimensional.
    """
    array: np.ndarray = np.asarray(points, dtype=float)
    if array.ndim != POINTS_NDIM:
        raise ValueError(
            f"Expected points with shape (n, d), got shape {array.shape}. "
            "Use points[np.newaxis, :] for a single point."
        )
    return array
