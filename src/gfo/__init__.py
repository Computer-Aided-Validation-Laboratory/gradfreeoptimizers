"""gfo: C++ powered gradient free optimizers with a Python interface.

Quick start::

    import gfo

    # Python objective: called once per iteration with the whole swarm,
    # points of shape (n, d), and returns the n values.
    problem = gfo.problems.Rastrigin()
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=10)

    # The same problem in C++: the whole run stays in C++, particles in
    # parallel.
    cpp_problem = gfo.cpp_problems.Rastrigin()
    result = gfo.pso(cpp_problem, cpp_problem.lo, cpp_problem.hi, d=10)

    print(result.best_val, result.best_pos)

Contents:

- ``gfo.pso``: the particle swarm optimizer.
- ``gfo.problems``: Python example problems (numpy, vectorized).
- ``gfo.cpp_problems``: the same problems in C++.
- ``gfo.post``: plots and animations of saved optimizer runs. Import it with
  ``from gfo import post`` (it pulls in matplotlib).
- ``gfo.PSOSettings``, ``gfo.PSOResult``, ``gfo.PSODiagnostics``: the C++
  types.
- ``gfo.OPENMP``, ``gfo.MAX_THREADS``: how the extension was built.

The C++ extension is built with ``python scripts/build.py``.
"""

try:
    from gfo import _pso
except ImportError as exc:
    raise ImportError(
        "The C++ extension gfo._pso is not built, or was built for another "
        "Python. Build it with: python scripts/build.py"
    ) from exc

from gfo import problems
from gfo.optimizer import CppProblem, pso

__version__ = "0.1.0"

cpp_problems = _pso.problems
PSOSettings = _pso.PSOSettings
PSOResult = _pso.PSOResult
PSODiagnostics = _pso.PSODiagnostics
OPENMP: bool = _pso.openmp
MAX_THREADS: int = _pso.max_threads

__all__ = [
    "MAX_THREADS",
    "OPENMP",
    "CppProblem",
    "PSODiagnostics",
    "PSOResult",
    "PSOSettings",
    "cpp_problems",
    "problems",
    "pso",
]
