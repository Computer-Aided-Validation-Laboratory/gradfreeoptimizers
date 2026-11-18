"""Rastrigin example problem. Non-convex, multimodal grid of local minima.

f(x) = A d + sum_k (z_k^2 - A cos(2 pi z_k)),   z = x - shift
Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems.interface import IProblem, validate_points


@dataclass(slots=True)
class Rastrigin(IProblem):
    """Rastrigin problem.

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    amplitude : float
        Height of the local minima ripples (A).
    lo, hi : float
        Default bounds, the same in every dimension.
    """

    name: ClassVar[str] = "Rastrigin"
    f_opt: ClassVar[float] = 0.0
    shift: float = 2.0
    amplitude: float = 10.0
    lo: float = -5.12
    hi: float = 5.12

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate every row of ``points``, shape (n, d). Returns shape (n,)."""
        batch: np.ndarray = validate_points(points)
        shifted: np.ndarray = batch - self.shift
        ripples: np.ndarray = self.amplitude * np.cos(2.0 * np.pi * shifted)
        offset: float = self.amplitude * batch.shape[1]
        return offset + np.sum(shifted * shifted - ripples, axis=1)

    def calc_x_opt(self, d: int) -> np.ndarray:
        """Calculate the location of the known minimum, shape (d,)."""
        return np.full(d, self.shift)
