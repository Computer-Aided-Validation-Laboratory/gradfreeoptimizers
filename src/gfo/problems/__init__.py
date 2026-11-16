"""Python example problems (vectorized numpy versions of the C++ ones).

Every problem is a dataclass with default valued parameters, so you can use
it as is or change what you like::

    import gfo

    problem = gfo.problems.Rastrigin(shift=1.0)  # minimum moved to x_k = 1
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=10)

Each problem implements ``IProblem``, the same interface as its C++ twin in
``gfo.cpp_problems``:

- ``evaluate(points)``: values of a batch of points, shape (n, d) -> (n,).
- ``lo``, ``hi``: default bounds, the same in every dimension.
- ``calc_x_opt(d)``, ``f_opt``: location and value of the known minimum.
- ``name``: the problem name.

========== =========================== =================== ======================
Problem    Type                        Default domain      Minimum
========== =========================== =================== ======================
Sphere     convex, unimodal            [-5.12, 5.12]^d     x_k = shift (2)
Ellipsoid  convex, ill conditioned     [-5.12, 5.12]^d     x_k = shift (2)
Rosenbrock non-convex, curved valley   [-5, 10]^d          x_k = shift + a (3)
Rastrigin  non-convex, multimodal grid [-5.12, 5.12]^d     x_k = shift (2)
Ackley     non-convex, narrow basin    [-32.768, 32.768]^d x_k = shift (2)
Griewank   non-convex, coupled         [-600, 600]^d       x_k = shift (100)
Schwefel   non-convex, deceptive       [-500, 500]^d       x_k = 420.9687 + shift
========== =========================== =================== ======================

Every minimum has f = 0.
"""

from gfo.problems.ackley import Ackley
from gfo.problems.ellipsoid import Ellipsoid
from gfo.problems.griewank import Griewank
from gfo.problems.interface import IProblem
from gfo.problems.rastrigin import Rastrigin
from gfo.problems.rosenbrock import Rosenbrock
from gfo.problems.schwefel import Schwefel
from gfo.problems.sphere import Sphere

# Every example problem by name, in the order the benchmarks run them.
EXAMPLE_PROBLEMS: dict[str, type[IProblem]] = {
    "Sphere": Sphere,
    "Ellipsoid": Ellipsoid,
    "Rosenbrock": Rosenbrock,
    "Rastrigin": Rastrigin,
    "Ackley": Ackley,
    "Griewank": Griewank,
    "Schwefel": Schwefel,
}

__all__ = [
    "EXAMPLE_PROBLEMS",
    "Ackley",
    "Ellipsoid",
    "Griewank",
    "IProblem",
    "Rastrigin",
    "Rosenbrock",
    "Schwefel",
    "Sphere",
]
