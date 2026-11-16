"""Rosenbrock example problem. Non-convex, narrow curved valley.

f(x) = sum_{k=0}^{d-2} b (z_{k+1} - z_k^2)^2 + (a - z_k)^2,   z = x - shift
Domain: [-5, 10]^d. Minimum f = 0 at x_k = shift + a.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch


@dataclass(slots=True)
class Rosenbrock:
    """Rosenbrock problem (vectorized). ``d = 1`` has no terms and gives 0.

    Attributes
    ----------
    shift : float
        Shift applied to every coordinate.
    a : float
        Valley floor location (the minimum is at z_k = a).
    b : float
        Valley steepness.
    lo, hi : float
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Rosenbrock"
    shift: float = 2.0
    a: float = 1.0
    b: float = 100.0
    lo: float = -5.0
    hi: float = 10.0

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        shifted = batch - self.shift
        current = shifted[:, :-1]
        following = shifted[:, 1:]
        valley = following - current * current
        floor = self.a - current
        values = np.sum(self.b * valley * valley + floor * floor, axis=1)
        return finish_batch(values, single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, self.shift + self.a)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
