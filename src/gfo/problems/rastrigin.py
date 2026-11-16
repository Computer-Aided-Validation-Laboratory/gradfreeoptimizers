"""Rastrigin example problem. Non-convex, multimodal grid of local minima.

f(x) = A d + sum_k (z_k^2 - A cos(2 pi z_k)),   z = x - shift
Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch


@dataclass(slots=True)
class Rastrigin:
    """Rastrigin problem (vectorized).

    Attributes
    ----------
    shift : float
        Location of the minimum in every dimension.
    amplitude : float
        Height of the local minima ripples (A).
    lo, hi : float
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Rastrigin"
    shift: float = 2.0
    amplitude: float = 10.0
    lo: float = -5.12
    hi: float = 5.12

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        shifted = batch - self.shift
        ripples = self.amplitude * np.cos(2.0 * np.pi * shifted)
        values = self.amplitude * batch.shape[1] + np.sum(shifted * shifted - ripples, axis=1)
        return finish_batch(values, single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, self.shift)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
