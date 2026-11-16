"""Schwefel example problem. Non-convex and deceptive.

f(x) = 418.9829 d - sum_k z_k sin(sqrt(|z_k|)),   z = x - shift
Not shifted by default, its minimum is already far from the centre.
Domain: [-500, 500]^d. Minimum f = 0 (to about 1e-11 per dimension) at
x_k = 420.9687462275036 + shift.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gfo.problems._batch import as_batch, finish_batch

# Location of the minimum of z sin(sqrt|z|) per dimension, and the matching
# constant that makes the minimum 0 (same values as the C++ version).
SCHWEFEL_X_OPT = 420.9687462275036
SCHWEFEL_CONSTANT = 418.982887272433799807913601398


@dataclass(slots=True)
class Schwefel:
    """Schwefel problem (vectorized).

    Attributes
    ----------
    shift : float
        Shift applied to every coordinate (0 = the standard function).
    lo, hi : float
        Default bounds (every dimension).
    """

    name: ClassVar[str] = "Schwefel"
    shift: float = 0.0
    lo: float = -500.0
    hi: float = 500.0

    def __call__(self, x) -> np.ndarray | float:
        """Evaluate one point ``(d,)`` (float) or a batch ``(n, d)`` (n values)."""
        batch, single = as_batch(x)
        shifted = batch - self.shift
        wave = np.sum(shifted * np.sin(np.sqrt(np.abs(shifted))), axis=1)
        return finish_batch(SCHWEFEL_CONSTANT * batch.shape[1] - wave, single)

    def x_opt(self, d: int) -> np.ndarray:
        """Location of the known minimum in ``d`` dimensions."""
        return np.full(d, SCHWEFEL_X_OPT + self.shift)

    def f_opt(self) -> float:
        """Value of the known minimum."""
        return 0.0
