"""Griewank example problem. Non-convex, couples dimensions through a product.

f(x) = sum_k z_k^2 / scale - prod_k cos(z_k / sqrt(k + 1)) + 1,   z = x - shift
Domain: [-600, 600]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Griewank(IProblem):
    """Griewank problem.

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    scale : float
        Divisor of the quadratic term (4000 in the standard form).
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Griewank"
    f_opt: ClassVar[float] = 0.0
    shift: float = 100.0
    scale: float = 4000.0
    lo: float = -600.0
    hi: float = 600.0

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        shifted: np.ndarray = batch - self.shift
        # root_index: shape=(d,), sqrt(k + 1) for coordinate k.
        root_index: np.ndarray = np.sqrt(
            np.arange(1, batch.shape[1] + 1, dtype=float)
        )
        quadratic: np.ndarray = np.sum(shifted * shifted / self.scale, axis=1)
        product: np.ndarray = np.prod(np.cos(shifted / root_index), axis=1)
        return quadratic - product + 1.0

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift)
