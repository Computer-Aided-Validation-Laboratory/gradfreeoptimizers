"""Validation against an independent implementation: pymoo's PSO.

Same problems, budget and coefficients (inertia, accelerations, velocity
clamp), pymoo with ``adaptive=False``, ``pertube_best=False`` and uniform
random initial positions like gfo. The two still differ in details
(initial velocities, boundary handling, see scripts/benchmark_pymoo.py), so
they are compared statistically over seeds, never run by run:

- Problems that are solvable at this budget: gfo's success rate must not be
  significantly lower than pymoo's (one-sided Fisher exact test). Success
  rates, not medians, because the errors are bimodal (solved or stuck in a
  local minimum) and medians jump between the modes.
- Hard problems nobody solves: gfo's median error must be within a factor of
  pymoo's.

Needs pymoo (``pip install -e ".[benchmark]"``), skipped otherwise.
"""

import math

import numpy as np
import pytest

import gfo

pytest.importorskip("pymoo")
from pymoo.algorithms.soo.nonconvex.pso import PSO  # noqa: E402
from pymoo.core.problem import Problem  # noqa: E402
from pymoo.operators.sampling.rnd import FloatRandomSampling  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.validation]

ITERATIONS = 200
# More seeds where runs are cheap (2D): success rates need them. Small
# samples mislead: on seeds 0-11 gfo solved Griewank 2D 3 times and pymoo 9,
# on seeds 0-39 both solved it 19 times.
SEEDS_BY_DIMENSION = {2: range(30), 10: range(12)}
# Significance level of the one-sided test "gfo solves less often".
SIGNIFICANCE = 0.01
# Hard problems: gfo's median error / pymoo's at most this. Measured 0.6-1.1.
MEDIAN_RATIO = 2.0


def calc_fewer_successes_p_value(successes: int, reference: int, runs: int) -> float:
    """One-sided Fisher exact test that ``successes`` is below ``reference``.

    Both counts are out of ``runs``. Returns the probability, if both share
    one success rate, of a split at least as lopsided as the one observed.
    """
    total: int = successes + reference
    tables: int = math.comb(2 * runs, runs)
    as_extreme: int = sum(
        math.comb(total, kk) * math.comb(2 * runs - total, runs - kk)
        for kk in range(max(0, total - runs), successes + 1)
    )
    return as_extreme / tables


class PymooBatchProblem(Problem):
    """pymoo problem evaluating a gfo example problem (pymoo needs a subclass)."""

    def __init__(self, problem: gfo.problems.IProblem, d: int) -> None:
        super().__init__(
            n_var=d, n_obj=1, xl=np.full(d, problem.lo), xu=np.full(d, problem.hi)
        )
        self.problem = problem

    def _evaluate(
        self, x: np.ndarray, out: dict[str, np.ndarray], *args: object, **kwargs: object
    ) -> None:
        out["F"] = self.problem.evaluate(x)


def calc_both_errors(name: str, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Final errors of gfo and pymoo for every seed used in ``d`` dimensions."""
    problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    n_particles = 20 + 2 * d
    gfo_errors, pymoo_errors = [], []
    for seed in SEEDS_BY_DIMENSION[d]:
        result = gfo.pso(
            problem.evaluate,
            problem.lo,
            problem.hi,
            d=d,
            n_particles=n_particles,
            iterations=ITERATIONS,
            seed=seed,
        )
        gfo_errors.append(result.best_val - problem.f_opt)
        algorithm = PSO(
            pop_size=n_particles,
            sampling=FloatRandomSampling(),
            w=gfo.PSOSettings().inertia,
            c1=gfo.PSOSettings().acceleration1,
            c2=gfo.PSOSettings().acceleration2,
            max_velocity_rate=gfo.PSOSettings().max_velocity_rate,
            adaptive=False,
            pertube_best=False,
        )
        reference = minimize(
            PymooBatchProblem(problem, d),
            algorithm,
            termination=("n_gen", ITERATIONS),
            seed=seed,
            verbose=False,
        )
        pymoo_errors.append(float(reference.F[0]) - problem.f_opt)
    return np.array(gfo_errors), np.array(pymoo_errors)


# (problem, d, tolerance on f - f_opt that counts as solved).
SOLVABLE_CASES = [
    ("Sphere", 2, 1e-12),
    ("Sphere", 10, 1e-8),
    ("Ellipsoid", 2, 1e-12),
    ("Ellipsoid", 10, 1e-7),
    ("Rosenbrock", 2, 1e-6),
    ("Rastrigin", 2, 1e-6),
    ("Ackley", 2, 1e-6),
    ("Ackley", 10, 1e-3),
    ("Griewank", 2, 1e-6),
    ("Schwefel", 2, 1e-3),
]


@pytest.mark.parametrize(
    ("name", "d", "tolerance"),
    SOLVABLE_CASES,
    ids=[f"{case[0]}-d{case[1]}" for case in SOLVABLE_CASES],
)
def test_solves_as_often_as_pymoo(name: str, d: int, tolerance: float) -> None:
    gfo_errors, pymoo_errors = calc_both_errors(name, d)
    gfo_solved = int(np.count_nonzero(gfo_errors < tolerance))
    pymoo_solved = int(np.count_nonzero(pymoo_errors < tolerance))
    p_value = calc_fewer_successes_p_value(gfo_solved, pymoo_solved, gfo_errors.size)

    assert p_value > SIGNIFICANCE, (
        f"gfo solved {gfo_solved}, pymoo {pymoo_solved} of {gfo_errors.size} "
        f"(p = {p_value:.2g})"
    )


@pytest.mark.parametrize(
    ("successes", "reference", "runs", "expected"),
    # By hand: sums of C(12, k) C(12, 12 - k) / C(24, 12).
    [(3, 9, 12, 0.019563), (6, 6, 12, 0.657865), (0, 12, 12, 3.698e-7), (5, 0, 5, 1.0)],
)
def test_fisher_p_value(
    successes: int, reference: int, runs: int, expected: float
) -> None:
    p_value = calc_fewer_successes_p_value(successes, reference, runs)

    assert p_value == pytest.approx(expected, rel=2e-3)


HARD_CASES = [("Rosenbrock", 10), ("Rastrigin", 10), ("Griewank", 10), ("Schwefel", 10)]


@pytest.mark.parametrize(
    ("name", "d"), HARD_CASES, ids=[f"{name}-d{d}" for name, d in HARD_CASES]
)
def test_hard_problems_as_good_as_pymoo(name: str, d: int) -> None:
    gfo_errors, pymoo_errors = calc_both_errors(name, d)

    assert np.median(gfo_errors) <= MEDIAN_RATIO * np.median(pymoo_errors)
