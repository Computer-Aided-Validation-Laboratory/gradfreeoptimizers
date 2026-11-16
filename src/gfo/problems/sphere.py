"""Sphere example problem. Convex, unimodal, separable.

f(x) = sum_k (x_k - shift)^2
Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch


@dataclass(slots=True)
class Sphere:
    """Sphere problem (vectorized).

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension. Shifted off the centre
        because swarms are biased towards it.
    lo, hi : float
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Sphere"
    shift: float = 2.0
    lo: float = -5.12
    hi: float = 5.12

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        shifted = batch - self.shift
        return finish_batch(np.sum(shifted * shifted, axis=1), single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, self.shift)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
