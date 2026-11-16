"""Sphere example problem. Convex, unimodal, separable.

f(x) = sum_k (x_k - shift)^2
Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Sphere(IProblem):
    """Sphere problem.

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension. Shifted off the centre
        because swarms are biased towards it.
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Sphere"
    f_opt: ClassVar[float] = 0.0
    shift: float = 2.0
    lo: float = -5.12
    hi: float = 5.12

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        shifted: np.ndarray = batch - self.shift
        return np.sum(shifted * shifted, axis=1)

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift)
