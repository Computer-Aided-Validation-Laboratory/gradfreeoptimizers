"""Rosenbrock example problem. Non-convex, narrow curved valley.

f(x) = sum_{k=0}^{d-2} b (z_{k+1} - z_k^2)^2 + (a - z_k)^2,   z = x - shift
Domain: [-5, 10]^d. Minimum f = 0 at x_k = shift + a.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Rosenbrock(IProblem):
    """Rosenbrock problem. ``d = 1`` has no terms and gives 0.

    Attributes
    ----------
    shift : float
        Shift applied to every coordinate.
    a : float
        Valley floor location (the minimum is at z_k = a).
    b : float
        Valley steepness.
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Rosenbrock"
    f_opt: ClassVar[float] = 0.0
    shift: float = 2.0
    a: float = 1.0
    b: float = 100.0
    lo: float = -5.0
    hi: float = 10.0

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        shifted: np.ndarray = batch - self.shift
        # current, following: shape=(n, d - 1), coordinates k and k + 1.
        current: np.ndarray = shifted[:, :-1]
        following: np.ndarray = shifted[:, 1:]
        valley: np.ndarray = following - current * current
        floor: np.ndarray = self.a - current
        return np.sum(self.b * valley * valley + floor * floor, axis=1)

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift + self.a)
