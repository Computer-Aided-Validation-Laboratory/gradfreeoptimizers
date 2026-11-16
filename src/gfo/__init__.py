"""gfo: C++ powered gradient free optimizers with a Python interface.

Quick start::

    import gfo

    # Python objective (vectorized: X has shape (n, d), return n values).
    result = gfo.pso(gfo.problems.Rastrigin(), d=10, iterations=500)

    # Same problem in C++: the whole run stays in C++, particles in parallel.
    result = gfo.pso(gfo.cpp_problems.Rastrigin(), d=10, iterations=500)

    print(result.best_val, result.best_pos)

Contents:

- ``gfo.pso``: the particle swarm optimizer.
- ``gfo.problems``: Python example problems (numpy, vectorized).
- ``gfo.cpp_problems``: the same problems in C++ (needs the built extension).
- ``gfo.post``: plots and animations of saved optimizer runs (imported on
  first use, it pulls in matplotlib and pandas).
- ``gfo.PSOSettings``, ``gfo.PSOResult``, ``gfo.PSODiagnostics``: the C++ types.

The C++ extension is built with ``python scripts/build.py``.
"""

import importlib

from gfo import problems
from gfo._extension import EXTENSION_AVAILABLE, get_extension
from gfo.optimizer import pso

__version__ = "0.1.0"

# Names served by the compiled extension, resolved on first use so that
# ``import gfo`` works before the extension is built.
_EXTENSION_NAMES = {
    "cpp_problems": "problems",
    "PSOSettings": "PSOSettings",
    "PSOResult": "PSOResult",
    "PSODiagnostics": "PSODiagnostics",
}
_LAZY_SUBMODULES = ("post",)


def __getattr__(name: str):
    if name in _EXTENSION_NAMES:
        return getattr(get_extension(), _EXTENSION_NAMES[name])
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"gfo.{name}")
    raise AttributeError(f"module 'gfo' has no attribute {name!r}")


__all__ = [
    "EXTENSION_AVAILABLE", "PSODiagnostics", "PSOResult", "PSOSettings",
    "cpp_problems", "get_extension", "post", "problems", "pso",
]
