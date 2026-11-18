"""Axis parallel hyper-ellipsoid example problem. Convex, ill conditioned.

f(x) = sum_k (k + 1) (x_k - shift)^2      (weights 1 to d)
Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Ellipsoid(IProblem):
    """Axis parallel hyper-ellipsoid problem.

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Ellipsoid"
    f_opt: ClassVar[float] = 0.0
    shift: float = 2.0
    lo: float = -5.12
    hi: float = 5.12

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        shifted: np.ndarray = batch - self.shift
        # weights: shape=(d,), weight k + 1 for coordinate k.
        weights: np.ndarray = np.arange(1, batch.shape[1] + 1, dtype=float)
        return np.sum(weights * shifted * shifted, axis=1)

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift)
