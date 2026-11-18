"""Validation: does the optimizer actually solve problems, and as expected?

The verification tests show the engine computes what it should. These show
the result is a working optimizer:

- It reliably solves the problems PSO is known to solve, and converges at
  the expected (linear) rate.
- It beats random search with the same budget by a wide margin.
- It solves textbook problems written independently of gfo.problems, with
  literature minima.
- It scales the right way with the swarm size, contracts onto the optimum,
  handles optima on the boundary, badly scaled dimensions, noise and
  plateaus.

Thresholds come from measured runs with a margin (see each constant), so a
change of random number generator should not trip them while a real loss of
performance does. The pymoo comparison is in test_pso_vs_pymoo.py.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

import gfo
from pso_testing import SwarmRecorder

pytestmark = pytest.mark.validation

SEEDS = range(10)
ITERATIONS = 500


def calc_swarm_size(d: int) -> int:
    """Swarm size used by the benchmarks: 20 + 2 d (pymoo's rule of thumb)."""
    return 20 + 2 * d


def calc_errors(name: str, d: int, iterations: int = ITERATIONS) -> np.ndarray:
    """Final error of the C++ problem ``name`` for every seed in SEEDS."""
    problem = getattr(gfo.cpp_problems, name)()
    errors = [
        gfo.pso(
            problem,
            problem.lo,
            problem.hi,
            d=d,
            n_particles=calc_swarm_size(d),
            iterations=iterations,
            seed=seed,
        ).best_val
        - problem.f_opt
        for seed in SEEDS
    ]
    return np.array(errors)


# (problem, d, tolerance on f - f_opt, runs out of 10 that must be within it).
# Measured on 20 seeds: every case below solved 19 or 20 times out of 20,
# Griewank 2D (a local minimum at 7.4e-3 next to the global one) 12 of 20.
SOLVED_CASES = [
    ("Sphere", 2, 1e-12, 10),
    ("Sphere", 10, 1e-12, 10),
    ("Ellipsoid", 2, 1e-12, 10),
    ("Ellipsoid", 10, 1e-12, 10),
    ("Rosenbrock", 2, 1e-8, 10),
    ("Rastrigin", 2, 1e-8, 8),
    ("Ackley", 2, 1e-8, 10),
    ("Ackley", 10, 1e-8, 10),
    ("Griewank", 2, 1e-8, 3),
]


@pytest.mark.parametrize(
    ("name", "d", "tolerance", "min_solved"),
    SOLVED_CASES,
    ids=[f"{case[0]}-d{case[1]}" for case in SOLVED_CASES],
)
def test_reliably_solves_the_problems_pso_should_solve(
    name: str, d: int, tolerance: float, min_solved: int
) -> None:
    errors = calc_errors(name, d)

    assert np.all(errors >= -1e-12), "below the known minimum"
    assert np.count_nonzero(errors < tolerance) >= min_solved


# Hard problems no plain PSO solves at this budget: the median error must
# stay below these, about twice the measured median (Rosenbrock 2.5,
# Rastrigin 9.0, Griewank 0.065, Schwefel 1450 for d = 10).
HARD_CASES = [
    ("Rosenbrock", 10, 6.0),
    ("Rastrigin", 10, 20.0),
    ("Griewank", 10, 0.15),
    ("Schwefel", 10, 2500.0),
]


@pytest.mark.parametrize(
    ("name", "d", "max_median"),
    HARD_CASES,
    ids=[f"{case[0]}-d{case[1]}" for case in HARD_CASES],
)
def test_hard_problems_stay_within_the_expected_error(
    name: str, d: int, max_median: float
) -> None:
    assert np.median(calc_errors(name, d)) < max_median


def calc_random_search_errors(name: str, d: int) -> np.ndarray:
    """Random search with the PSO budget: error for every seed in SEEDS."""
    problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    n_points = calc_swarm_size(d) * ITERATIONS
    errors = []
    for seed in SEEDS:
        points = np.random.default_rng(seed).uniform(
            problem.lo, problem.hi, size=(n_points, d)
        )
        errors.append(problem.evaluate(points).min() - problem.f_opt)
    return np.array(errors)


# Median PSO error / median random search error must be below this. Measured
# worst case 0.13 (Rastrigin 10D), most cases are 1e-3 or far below.
RANDOM_SEARCH_RATIO = 0.25
# Schwefel's minimum sits in a corner far from the next best basin, which
# traps PSO. In 2D, random search with 22000 points beats PSO and pymoo's PSO
# alike (median 118 vs 2.5), in 10D PSO is only slightly better (ratio 0.83).
RANDOM_SEARCH_CASES = [
    (name, d)
    for name in gfo.problems.EXAMPLE_PROBLEMS
    for d in (2, 10)
    if name != "Schwefel"
]


@pytest.mark.parametrize(
    ("name", "d"),
    RANDOM_SEARCH_CASES,
    ids=[f"{name}-d{d}" for name, d in RANDOM_SEARCH_CASES],
)
def test_beats_random_search_with_the_same_budget(name: str, d: int) -> None:
    pso_median = np.median(calc_errors(name, d))
    random_median = np.median(calc_random_search_errors(name, d))

    assert pso_median <= RANDOM_SEARCH_RATIO * random_median


def test_schwefel_beats_random_search_in_ten_dimensions() -> None:
    assert np.median(calc_errors("Schwefel", 10)) < np.median(
        calc_random_search_errors("Schwefel", 10)
    )


def test_converges_linearly_on_a_sphere() -> None:
    # Measured: about 7 decades per 100 iterations, 4 is the floor here.
    problem = gfo.cpp_problems.Sphere()
    for seed in SEEDS:
        loss = gfo.pso(
            problem, problem.lo, problem.hi, d=5, iterations=300, seed=seed
        ).loss_history
        decades = np.log10(loss[[99, 199, 299]])

        assert np.all(np.diff(decades) < -4.0)
        assert loss[-1] < 1e-15


@pytest.mark.parametrize(("name", "d"), [("Sphere", 50), ("Ellipsoid", 30)])
def test_solves_high_dimensional_convex_problems(name: str, d: int) -> None:
    problem = getattr(gfo.cpp_problems, name)()
    for seed in range(3):
        result = gfo.pso(
            problem,
            problem.lo,
            problem.hi,
            d=d,
            n_particles=calc_swarm_size(d),
            iterations=1000,
            seed=seed,
        )

        assert result.best_val < 1e-10
        np.testing.assert_allclose(result.best_pos, problem.calc_x_opt(d), atol=1e-5)


def test_more_particles_find_better_minima() -> None:
    # Measured medians for 5, 20 and 80 particles: 6.0, 2.5 and 0.50.
    problem = gfo.cpp_problems.Rastrigin()
    medians = []
    for n_particles in (5, 20, 80):
        errors = [
            gfo.pso(
                problem,
                problem.lo,
                problem.hi,
                d=5,
                n_particles=n_particles,
                iterations=300,
                seed=seed,
            ).best_val
            for seed in range(20)
        ]
        medians.append(np.median(errors))

    assert medians[0] > medians[1] > medians[2]


def test_swarm_contracts_onto_the_optimum() -> None:
    # Measured: the spread shrinks from 0.28 to 1.6e-7 of the domain width.
    centre = 0.3
    recorder = SwarmRecorder(lambda points: np.sum((points - centre) ** 2, axis=1))
    result = gfo.pso(recorder.evaluate, 0.0, 1.0, d=5, iterations=200, seed=0)
    points = recorder.stack_points()
    spread = points.std(axis=1).mean(axis=1)

    assert spread[0] > 0.2
    assert spread[-1] < 1e-4 * spread[0]
    np.testing.assert_allclose(points[-1].mean(axis=0), centre, atol=1e-4)
    np.testing.assert_allclose(result.best_pos, centre, atol=1e-6)


@dataclass(slots=True)
class TextbookProblem:
    """A 2D test function from the optimization literature.

    Written here independently of gfo.problems, with literature minima.

    Attributes
    ----------
    evaluate : callable
        Points of shape (n, d) -> (n,) values.
    lo, hi : list of float
        Search domain, per dimension.
    f_opt : float
        Global minimum value.
    x_opts : list of tuple of float
        Every global minimizer.
    """

    evaluate: Callable[[np.ndarray], np.ndarray]
    lo: list[float]
    hi: list[float]
    f_opt: float
    x_opts: list[tuple[float, ...]]


def evaluate_booth(x: np.ndarray) -> np.ndarray:
    return (x[:, 0] + 2 * x[:, 1] - 7) ** 2 + (2 * x[:, 0] + x[:, 1] - 5) ** 2


def evaluate_matyas(x: np.ndarray) -> np.ndarray:
    return 0.26 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 0.48 * x[:, 0] * x[:, 1]


def evaluate_himmelblau(x: np.ndarray) -> np.ndarray:
    return (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2


def evaluate_goldstein_price(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    first = 1 + (a + b + 1) ** 2 * (
        19 - 14 * a + 3 * a**2 - 14 * b + 6 * a * b + 3 * b**2
    )
    second = 30 + (2 * a - 3 * b) ** 2 * (
        18 - 32 * a + 12 * a**2 + 48 * b - 36 * a * b + 27 * b**2
    )
    return first * second


def evaluate_beale(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return (
        (1.5 - a + a * b) ** 2
        + (2.25 - a + a * b**2) ** 2
        + (2.625 - a + a * b**3) ** 2
    )


def evaluate_mccormick(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return np.sin(a + b) + (a - b) ** 2 - 1.5 * a + 2.5 * b + 1


def evaluate_three_hump_camel(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return 2 * a**2 - 1.05 * a**4 + a**6 / 6 + a * b + b**2


def evaluate_levi13(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return (
        np.sin(3 * np.pi * a) ** 2
        + (a - 1) ** 2 * (1 + np.sin(3 * np.pi * b) ** 2)
        + (b - 1) ** 2 * (1 + np.sin(2 * np.pi * b) ** 2)
    )


def evaluate_easom(x: np.ndarray) -> np.ndarray:
    a, b = x[:, 0], x[:, 1]
    return -np.cos(a) * np.cos(b) * np.exp(-((a - np.pi) ** 2 + (b - np.pi) ** 2))


# McCormick's minimizer is x = 1/2 - pi/3, y = x - 1.
MCCORMICK_X = 0.5 - np.pi / 3.0
TEXTBOOK_PROBLEMS: dict[str, TextbookProblem] = {
    "booth": TextbookProblem(evaluate_booth, [-10, -10], [10, 10], 0.0, [(1, 3)]),
    "matyas": TextbookProblem(evaluate_matyas, [-10, -10], [10, 10], 0.0, [(0, 0)]),
    "himmelblau": TextbookProblem(
        evaluate_himmelblau,
        [-5, -5],
        [5, 5],
        0.0,
        [
            (3.0, 2.0),
            (-2.805118, 3.131312),
            (-3.779310, -3.283186),
            (3.584428, -1.848126),
        ],
    ),
    "goldstein_price": TextbookProblem(
        evaluate_goldstein_price, [-2, -2], [2, 2], 3.0, [(0, -1)]
    ),
    "beale": TextbookProblem(evaluate_beale, [-4.5, -4.5], [4.5, 4.5], 0.0, [(3, 0.5)]),
    "mccormick": TextbookProblem(
        evaluate_mccormick,
        [-1.5, -3],
        [4, 4],
        np.sin(2 * MCCORMICK_X - 1) + 1 - 1.5 * MCCORMICK_X + 2.5 * (MCCORMICK_X - 1)
        + 1,
        [(MCCORMICK_X, MCCORMICK_X - 1)],
    ),
    "three_hump_camel": TextbookProblem(
        evaluate_three_hump_camel, [-5, -5], [5, 5], 0.0, [(0, 0)]
    ),
    "levi13": TextbookProblem(evaluate_levi13, [-10, -10], [10, 10], 0.0, [(1, 1)]),
    "easom": TextbookProblem(
        evaluate_easom, [-100, -100], [100, 100], -1.0, [(np.pi, np.pi)]
    ),
}
# Himmelblau's minimizers are only given to 6 decimals in the literature.
TEXTBOOK_X_ATOL = 1e-5
TEXTBOOK_F_ATOL = 1e-9


@pytest.mark.parametrize("name", list(TEXTBOOK_PROBLEMS))
def test_solves_textbook_problems(name: str) -> None:
    problem = TEXTBOOK_PROBLEMS[name]
    for seed in range(5):
        result = gfo.pso(
            problem.evaluate,
            problem.lo,
            problem.hi,
            n_particles=40,
            iterations=300,
            seed=seed,
        )
        distances = [
            np.abs(result.best_pos - np.array(x_opt)).max() for x_opt in problem.x_opts
        ]

        assert result.best_val == pytest.approx(problem.f_opt, abs=TEXTBOOK_F_ATOL)
        assert min(distances) < TEXTBOOK_X_ATOL


def test_finds_every_minimum_of_himmelblau_across_seeds() -> None:
    problem = TEXTBOOK_PROBLEMS["himmelblau"]
    found = set()
    for seed in range(20):
        result = gfo.pso(
            problem.evaluate,
            problem.lo,
            problem.hi,
            n_particles=40,
            iterations=200,
            seed=seed,
        )
        distances = [
            np.abs(result.best_pos - np.array(x_opt)).max() for x_opt in problem.x_opts
        ]
        found.add(int(np.argmin(distances)))

    assert found == {0, 1, 2, 3}


def test_ends_on_a_stationary_minimum_of_styblinski_tang() -> None:
    # Separable with two minima per dimension, only one of them global: PSO
    # does not always find the global one (measured 2 of 10 in 5D), but every
    # coordinate must end on one of the two 1D minima.
    minima_1d = np.array([-2.903534027771178, 2.746802732850583])
    for seed in range(5):
        result = gfo.pso(
            lambda x: 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=1),
            -5.0,
            5.0,
            d=5,
            n_particles=40,
            iterations=300,
            seed=seed,
        )
        distance = np.abs(result.best_pos[:, np.newaxis] - minima_1d).min(axis=1)

        assert np.all(distance < 1e-6)


@pytest.mark.parametrize(
    ("target", "expected"),
    [([4.9, -4.9, 4.9], [4.9, -4.9, 4.9]), ([10.0, -10.0, 10.0], [5.0, -5.0, 5.0])],
    ids=["near_corner", "outside_the_box"],
)
def test_finds_optima_on_or_beyond_the_boundary(
    target: list[float], expected: list[float]
) -> None:
    goal = np.array(target)
    result = gfo.pso(
        lambda x: np.sum((x - goal) ** 2, axis=1), -5.0, 5.0, d=3, iterations=300
    )

    np.testing.assert_allclose(result.best_pos, expected, rtol=0.0, atol=1e-9)


def test_linear_objective_ends_exactly_on_the_lower_corner() -> None:
    lo = np.array([-1.0, 2.0, -300.0])
    hi = np.array([3.0, 2.5, 100.0])
    result = gfo.pso(lambda x: np.sum(x, axis=1), lo, hi, iterations=200)

    np.testing.assert_array_equal(result.best_pos, lo)


def test_badly_scaled_dimensions_are_solved_in_their_own_units() -> None:
    # Bounds spanning 9 orders of magnitude: the search runs in the unit box,
    # so every dimension is solved to the same relative accuracy.
    lo = np.array([0.0, -1e3, 5.0, -1e-6])
    hi = np.array([1e-3, 1e3, 6.0, 1e-6])
    width = hi - lo
    optimum = lo + np.array([0.31, 0.62, 0.17, 0.83]) * width
    result = gfo.pso(
        lambda x: np.sum(((x - optimum) / width) ** 2, axis=1),
        lo,
        hi,
        n_particles=30,
        iterations=300,
    )

    np.testing.assert_allclose((result.best_pos - optimum) / width, 0.0, atol=1e-8)


@pytest.mark.parametrize(
    ("noise", "max_error"), [(1e-3, 0.05), (1e-2, 0.1)], ids=["1e-3", "1e-2"]
)
def test_noisy_objective_still_lands_near_the_optimum(
    noise: float, max_error: float
) -> None:
    # Measured worst position error over 5 seeds: 0.013 and 0.044.
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        result = gfo.pso(
            lambda x: np.sum((x - 1.0) ** 2, axis=1) + rng.normal(0.0, noise, len(x)),
            -5.0,
            5.0,
            d=3,
            n_particles=30,
            iterations=300,
            seed=seed,
        )

        assert np.abs(result.best_pos - 1.0).max() < max_error


def test_finds_the_bottom_plateau_of_a_staircase() -> None:
    # Piecewise constant: no slope anywhere, the bottom step is |x_k - 0.3| <
    # 0.1 for every k.
    for seed in range(5):
        result = gfo.pso(
            lambda x: np.sum(np.floor(np.abs(x - 0.3) * 10.0), axis=1),
            -5.0,
            5.0,
            d=4,
            n_particles=30,
            iterations=200,
            seed=seed,
        )

        assert result.best_val == 0.0
        assert np.abs(result.best_pos - 0.3).max() < 0.1
