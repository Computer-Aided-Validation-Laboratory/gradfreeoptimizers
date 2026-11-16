"""Griewank example problem. Non-convex, couples dimensions through a product.

f(x) = sum_k z_k^2 / scale - prod_k cos(z_k / sqrt(k + 1)) + 1,   z = x - shift
Domain: [-600, 600]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch


@dataclass(slots=True)
class Griewank:
    """Griewank problem (vectorized).

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    scale : float
        Divisor of the quadratic term (4000 in the standard form).
    lo, hi : float
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Griewank"
    shift: float = 100.0
    scale: float = 4000.0
    lo: float = -600.0
    hi: float = 600.0

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        shifted = batch - self.shift
        root_index = np.sqrt(np.arange(1, batch.shape[1] + 1, dtype=float))
        quadratic = np.sum(shifted * shifted / self.scale, axis=1)
        product = np.prod(np.cos(shifted / root_index), axis=1)
        return finish_batch(quadratic - product + 1.0, single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, self.shift)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
