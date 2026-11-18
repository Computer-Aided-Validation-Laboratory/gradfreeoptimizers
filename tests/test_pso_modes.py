"""Verification by consistency: runs that must agree, bit for bit.

- The evaluation modes (C++ problem, vectorized and elementwise Python) visit
  the same points.
- Results depend only on the seed and the algorithm settings, never on the
  thread count, the console or file output, or the run length.
- Invariance (metamorphic) properties: rescaling the objective or the domain
  by a power of two, or any strictly increasing transform of the objective,
  must not change which points the swarm visits.
"""

from pathlib import Path

import numpy as np
import pytest

import gfo
from gfo import _pso
from pso_testing import SwarmRecorder, evaluate_unit_rastrigin, record_unit_run

SMALL = {"n_particles": 16, "iterations": 60, "seed": 3}
PROBLEM_NAMES = list(gfo.problems.EXAMPLE_PROBLEMS)


def run_both_modes(
    name: str, d: int, **kwargs: int
) -> tuple[gfo.PSOResult, gfo.PSOResult]:
    """Run the same C++ problem in C++ mode and through a Python batch call."""
    problem = getattr(gfo.cpp_problems, name)()
    cpp_result = gfo.pso(problem, problem.lo, problem.hi, d=d, **kwargs)
    py_result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=d, **kwargs)
    return cpp_result, py_result


def assert_same_run(first: gfo.PSOResult, second: gfo.PSOResult) -> None:
    """Two runs are identical: best value, best position and loss history."""
    assert first.best_val == second.best_val
    np.testing.assert_array_equal(first.best_pos, second.best_pos)
    np.testing.assert_array_equal(first.loss_history, second.loss_history)


@pytest.mark.parametrize("d", [1, 2, 5])
@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_cpp_and_batch_modes_are_identical(name: str, d: int) -> None:
    assert_same_run(*run_both_modes(name, d=d, **SMALL))


def test_elementwise_visits_the_same_points_as_vectorized() -> None:
    problem = gfo.problems.Griewank()
    vectorized = SwarmRecorder(problem.evaluate)
    single_points: list[np.ndarray] = []

    def evaluate_point(point: np.ndarray) -> float:
        single_points.append(point.copy())
        return float(problem.evaluate(point[np.newaxis, :])[0])

    first = gfo.pso(vectorized.evaluate, problem.lo, problem.hi, d=3, **SMALL)
    second = gfo.pso(
        evaluate_point, problem.lo, problem.hi, d=3, vectorized=False, **SMALL
    )

    assert_same_run(first, second)
    np.testing.assert_array_equal(
        np.stack(single_points), vectorized.stack_points().reshape(-1, 3)
    )


@pytest.mark.parametrize("mode", ["cpp", "batch"])
def test_results_do_not_depend_on_threads(mode: str) -> None:
    problem = gfo.cpp_problems.Ackley()
    objective = problem if mode == "cpp" else problem.evaluate
    results = [
        gfo.pso(objective, problem.lo, problem.hi, d=6, threads=threads, **SMALL)
        for threads in (1, 2, 3, 0)
    ]

    for result in results[1:]:
        assert_same_run(results[0], result)


def test_same_seed_reproduces_and_other_seeds_differ() -> None:
    problem = gfo.problems.Rastrigin()

    def run(seed: int) -> gfo.PSOResult:
        settings = {**SMALL, "seed": seed}
        return gfo.pso(problem.evaluate, problem.lo, problem.hi, d=4, **settings)

    assert_same_run(run(3), run(3))
    # Seeds that differ only in their upper 32 bits must give other runs too.
    best_values = {run(seed).best_val for seed in (0, 1, 2**32, 2**40 + 1, 2**63)}
    assert len(best_values) == 5


def test_a_longer_run_extends_a_shorter_one() -> None:
    short, short_result = record_unit_run(
        evaluate_unit_rastrigin, 3, n_particles=12, iterations=25, seed=8
    )
    long, long_result = record_unit_run(
        evaluate_unit_rastrigin, 3, n_particles=12, iterations=80, seed=8
    )

    np.testing.assert_array_equal(
        long_result.loss_history[:25], short_result.loss_history
    )
    np.testing.assert_array_equal(long.stack_points()[:25], short.stack_points())


@pytest.mark.parametrize(
    "neutral",
    [
        {"threads": 1},
        {"verbose": True, "verbose_interval": 7},
        {"diagnostics": True, "diagnostics_interval": 5},
        {"save_loss": True, "save_history": True},
        {"optimizer_dir": "elsewhere", "save_loss": True},
    ],
    ids=["threads", "verbose", "diagnostics", "files", "folder"],
)
def test_output_settings_do_not_change_the_run(
    neutral: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    problem = gfo.problems.Rosenbrock()
    baseline = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=2, **SMALL)
    changed = gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=2,
        run_dir=tmp_path,
        **SMALL,
        **neutral,
    )
    capsys.readouterr()

    assert_same_run(baseline, changed)


@pytest.mark.parametrize(
    "change",
    [
        {"inertia": 0.6},
        {"acceleration1": 1.2},
        {"acceleration2": 1.2},
        {"max_velocity_rate": 0.1},
        {"initial_velocity_rate": 0.05},
        {"seed": 4},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_algorithm_setting_reaches_the_engine(change: dict) -> None:
    # n_particles is checked through the evaluation count instead: every
    # particle has its own random stream, so an extra particle that never
    # leads leaves the loss history of the others unchanged.
    problem = gfo.problems.Rosenbrock()
    baseline = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=2, **SMALL)
    changed = gfo.pso(
        problem.evaluate, problem.lo, problem.hi, d=2, **{**SMALL, **change}
    )

    assert not np.array_equal(baseline.loss_history, changed.loss_history)


def test_objective_scaled_by_a_power_of_two_visits_the_same_points() -> None:
    plain, plain_result = record_unit_run(evaluate_unit_rastrigin, 3, **SMALL)
    scaled, scaled_result = record_unit_run(
        lambda points: 0.125 * evaluate_unit_rastrigin(points), 3, **SMALL
    )

    np.testing.assert_array_equal(plain.stack_points(), scaled.stack_points())
    assert scaled_result.best_val == 0.125 * plain_result.best_val


def test_increasing_transform_of_the_objective_visits_the_same_points() -> None:
    # PSO only compares values, so any strictly increasing transform of the
    # objective is invisible to it.
    plain, _ = record_unit_run(evaluate_unit_rastrigin, 3, **SMALL)
    transformed, _ = record_unit_run(
        lambda points: np.arctan(evaluate_unit_rastrigin(points) - 7.0), 3, **SMALL
    )

    np.testing.assert_array_equal(plain.stack_points(), transformed.stack_points())


def test_domain_scaled_by_a_power_of_two_scales_the_points() -> None:
    scale = 64.0
    unit, _ = record_unit_run(evaluate_unit_rastrigin, 3, **SMALL)
    wide = SwarmRecorder(lambda points: evaluate_unit_rastrigin(points / scale))
    gfo.pso(wide.evaluate, 0.0, scale, d=3, **SMALL)

    np.testing.assert_array_equal(wide.stack_points(), scale * unit.stack_points())


def test_objective_cannot_disturb_the_swarm_through_its_input() -> None:
    def evaluate_and_scribble(points: np.ndarray) -> np.ndarray:
        values = evaluate_unit_rastrigin(points)
        points[:] = np.nan
        return values

    plain = gfo.pso(evaluate_unit_rastrigin, 0.0, 1.0, d=3, **SMALL)
    scribbled = gfo.pso(evaluate_and_scribble, 0.0, 1.0, d=3, **SMALL)

    assert_same_run(plain, scribbled)


@pytest.mark.parametrize("name", ["Sphere", "Schwefel"])
def test_low_level_binding_matches_gfo_pso(name: str) -> None:
    problem = getattr(gfo.cpp_problems, name)()
    settings = _pso.PSOSettings()
    settings.particles = SMALL["n_particles"]
    settings.iterations = SMALL["iterations"]
    settings.seed = SMALL["seed"]
    lo = [problem.lo] * 3
    hi = [problem.hi] * 3

    friendly = gfo.pso(problem, problem.lo, problem.hi, d=3, **SMALL)
    assert_same_run(friendly, _pso.optimize_problem(problem, 3, lo, hi, settings))
    assert_same_run(
        friendly, _pso.optimize_python(problem.evaluate, 3, lo, hi, settings)
    )


def test_low_level_default_bounds_search_the_unit_box() -> None:
    settings = _pso.PSOSettings()
    settings.particles = 8
    settings.iterations = 20
    settings.default_bounds = True
    recorder = SwarmRecorder(evaluate_unit_rastrigin)
    # The bounds are ignored, even invalid ones.
    _pso.optimize_python(recorder.evaluate, 2, [], [], settings)
    points = recorder.stack_points()

    assert np.all((points >= 0.0) & (points <= 1.0))
