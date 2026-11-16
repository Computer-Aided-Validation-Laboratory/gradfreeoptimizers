"""Ackley example problem. Non-convex, shallow local minima around one narrow basin.

f(x) = -a exp(-b sqrt(sum z_k^2 / d)) - exp(sum cos(c z_k) / d) + a + e,   z = x - shift
Domain: [-32.768, 32.768]^d. Minimum f = 0 at x_k = shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch


@dataclass(slots=True)
class Ackley:
    """Ackley problem (vectorized).

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
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Ackley"
    shift: float = 2.0
    a: float = 20.0
    b: float = 0.2
    c: float = 2.0 * np.pi
    lo: float = -32.768
    hi: float = 32.768

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        dims = float(batch.shape[1])
        shifted = batch - self.shift
        mean_square = np.sum(shifted * shifted, axis=1) / dims
        mean_cos = np.sum(np.cos(self.c * shifted), axis=1) / dims
        envelope = -self.a * np.exp(-self.b * np.sqrt(mean_square))
        values = envelope - np.exp(mean_cos) + self.a + np.e
        return finish_batch(values, single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, self.shift)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
