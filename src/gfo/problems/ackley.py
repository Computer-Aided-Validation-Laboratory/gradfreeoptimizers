"""Ackley example problem. Non-convex, shallow local minima around one basin.

f(x) = -a exp(-b sqrt(sum z_k^2 / d)) - exp(sum cos(c z_k) / d) + a + e,
z = x - shift
Domain: [-32.768, 32.768]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Ackley(IProblem):
    """Ackley problem.

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    a : float
        Depth of the outer envelope.
    b : float
        Width of the outer envelope.
    c : float
        Frequency of the ripples.
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Ackley"
    f_opt: ClassVar[float] = 0.0
    shift: float = 2.0
    a: float = 20.0
    b: float = 0.2
    c: float = 2.0 * np.pi
    lo: float = -32.768
    hi: float = 32.768

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        dims = float(batch.shape[1])
        shifted: np.ndarray = batch - self.shift
        mean_square: np.ndarray = np.sum(shifted * shifted, axis=1) / dims
        mean_cos: np.ndarray = np.sum(np.cos(self.c * shifted), axis=1) / dims
        envelope: np.ndarray = -self.a * np.exp(-self.b * np.sqrt(mean_square))
        return envelope - np.exp(mean_cos) + self.a + np.e

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift)
