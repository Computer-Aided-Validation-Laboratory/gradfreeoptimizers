"""Behaviour of gfo.pso in both evaluation modes."""

import json
from pathlib import Path

import numpy as np
import pytest

import gfo
from gfo import optimizer

SMALL = {"n_particles": 16, "iterations": 60, "seed": 3}

# Results of the original specklemimic PSO, which the engine reproduces bit
# for bit. Exact comparison: valid for the same compiler and libm, a rebuild
# with other flags (e.g. --native) may change the last bits and the trajectory.
GOLD_CASES: list[dict] = json.loads(
    (Path(__file__).parent / "gold" / "pso_gold.json").read_text()
)["cases"]
GOLD_IDS: list[str] = [f"{case['problem']}-d{case['d']}" for case in GOLD_CASES]


def run_both_modes(
    name: str, d: int, **kwargs: int
) -> tuple[gfo.PSOResult, gfo.PSOResult]:
    """Run the same C++ problem in C++ mode and through a Python batch call."""
    problem = getattr(gfo.cpp_problems, name)()
    cpp_result = gfo.pso(problem, problem.lo, problem.hi, d=d, **kwargs)
    py_result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=d, **kwargs)
    return cpp_result, py_result


@pytest.mark.parametrize("case", GOLD_CASES, ids=GOLD_IDS)
def test_matches_gold_results(case: dict) -> None:
    settings = {
        "d": case["d"],
        "n_particles": case["n_particles"],
        "iterations": case["iterations"],
        "seed": case["seed"],
    }
    for result in run_both_modes(case["problem"], **settings):
        assert result.best_val == case["best_val"]
        np.testing.assert_array_equal(result.best_pos, case["best_pos"])


@pytest.mark.parametrize("name", list(gfo.problems.EXAMPLE_PROBLEMS))
def test_cpp_and_batch_modes_are_identical(name: str) -> None:
    cpp_result, py_result = run_both_modes(name, d=5, **SMALL)

    assert cpp_result.best_val == py_result.best_val
    np.testing.assert_array_equal(cpp_result.best_pos, py_result.best_pos)
    np.testing.assert_array_equal(cpp_result.loss_history, py_result.loss_history)


def test_python_defaults_match_the_cpp_defaults() -> None:
    settings = gfo.PSOSettings()

    assert settings.particles == optimizer.DEFAULT_PARTICLES
    assert settings.iterations == optimizer.DEFAULT_ITERATIONS
    assert settings.inertia == optimizer.DEFAULT_INERTIA
    assert settings.acceleration1 == optimizer.DEFAULT_ACCELERATION
    assert settings.acceleration2 == optimizer.DEFAULT_ACCELERATION
    assert settings.max_velocity_rate == optimizer.DEFAULT_MAX_VELOCITY_RATE
    assert settings.initial_velocity_rate == optimizer.DEFAULT_INITIAL_VELOCITY_RATE
    assert settings.seed == optimizer.DEFAULT_SEED
    assert settings.optimizer_dir == optimizer.DEFAULT_OPTIMIZER_DIR


def test_results_do_not_depend_on_threads() -> None:
    problem = gfo.cpp_problems.Ackley()
    best_values: set[float] = set()
    for threads in (1, 2, 0):
        result = gfo.pso(
            problem, problem.lo, problem.hi, d=6, threads=threads, **SMALL
        )
        best_values.add(result.best_val)

    assert len(best_values) == 1


def test_seed_reproducible_and_different_seeds_differ() -> None:
    problem = gfo.problems.Rastrigin()
    first = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=4, **SMALL)
    again = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=4, **SMALL)
    other = gfo.pso(
        problem.evaluate, problem.lo, problem.hi, d=4, **{**SMALL, "seed": 4}
    )

    assert first.best_val == again.best_val
    assert first.best_val != other.best_val


def test_elementwise_matches_vectorized() -> None:
    problem = gfo.problems.Griewank()

    def evaluate_point(point: np.ndarray) -> float:
        return float(problem.evaluate(point[np.newaxis, :])[0])

    vectorized = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=3, **SMALL)
    elementwise = gfo.pso(
        evaluate_point, problem.lo, problem.hi, d=3, vectorized=False, **SMALL
    )

    assert vectorized.best_val == elementwise.best_val


def test_budget_history_and_bounds() -> None:
    problem = gfo.problems.Rosenbrock()
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=4, **SMALL)

    assert result.evaluations == SMALL["n_particles"] * SMALL["iterations"]
    assert result.loss_history.shape == (SMALL["iterations"],)
    assert np.all(np.diff(result.loss_history) <= 0.0)
    assert result.loss_history[-1] == result.best_val
    assert np.all(result.best_pos >= problem.lo)
    assert np.all(result.best_pos <= problem.hi)
    best_value = problem.evaluate(result.best_pos[np.newaxis, :])[0]
    assert result.best_val == pytest.approx(best_value, rel=1e-12)


def test_converges_to_the_analytic_minimum_of_a_sphere() -> None:
    problem = gfo.problems.Sphere()
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=3, iterations=300)

    np.testing.assert_allclose(result.best_pos, problem.calc_x_opt(3), atol=1e-6)


def test_python_objective_sees_every_particle_once_per_iteration() -> None:
    shapes: list[tuple[int, ...]] = []

    def evaluate(points: np.ndarray) -> np.ndarray:
        shapes.append(points.shape)
        return np.sum(points * points, axis=1)

    gfo.pso(evaluate, -1.0, 1.0, d=3, **SMALL)

    assert shapes == [(SMALL["n_particles"], 3)] * SMALL["iterations"]


def test_per_dimension_bounds_and_inferred_dimension() -> None:
    lo = [0.0, 10.0, -5.0]
    hi = [1.0, 20.0, -4.0]
    result = gfo.pso(lambda points: np.sum(points, axis=1), lo, hi, **SMALL)

    assert result.best_pos.shape == (3,)
    assert np.all(result.best_pos >= lo)
    assert np.all(result.best_pos <= hi)


def test_zero_iterations() -> None:
    problem = gfo.problems.Sphere()
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=2, iterations=0)

    assert result.evaluations == 0
    assert result.loss_history.size == 0


@pytest.mark.parametrize(
    ("lo", "hi", "kwargs"),
    [
        (0.0, 1.0, {"d": 0}),
        (0.0, 1.0, {"d": 2, "n_particles": 0}),
        (0.0, 1.0, {"d": 2, "iterations": -1}),
        (0.0, 1.0, {"d": 2, "seed": -1}),
        (0.0, 1.0, {"d": 2, "max_velocity_rate": 0.0}),
        (1.0, 0.0, {"d": 2}),
        ([0.0, 0.0, 0.0], 1.0, {"d": 2}),
        (0.0, 1.0, {}),
    ],
)
def test_invalid_settings_raise(lo: object, hi: object, kwargs: dict) -> None:
    problem = gfo.problems.Sphere()
    with pytest.raises(ValueError):
        gfo.pso(problem.evaluate, lo, hi, **{**SMALL, **kwargs})


def test_elementwise_flag_with_a_cpp_problem_raises() -> None:
    problem = gfo.cpp_problems.Sphere()
    with pytest.raises(ValueError, match="vectorized=False"):
        gfo.pso(problem, problem.lo, problem.hi, d=2, vectorized=False)


def test_wrong_return_shape_raises_with_hint() -> None:
    with pytest.raises(ValueError, match="vectorized=False"):
        gfo.pso(lambda points: float(np.sum(points)), 0.0, 1.0, d=2, **SMALL)


def test_non_numeric_return_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(lambda points: "nope", 0.0, 1.0, d=2, **SMALL)


def test_objective_exception_propagates() -> None:
    def evaluate_broken(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        gfo.pso(evaluate_broken, 0.0, 1.0, d=2, **SMALL)


def test_not_callable_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(42, 0.0, 1.0, d=2)


def test_files_written(tmp_path: Path) -> None:
    problem = gfo.cpp_problems.Sphere()
    run_dir = tmp_path / "run"
    gfo.pso(
        problem,
        problem.lo,
        problem.hi,
        d=2,
        run_dir=run_dir,
        optimizer_dir="opt",
        save_loss=True,
        save_history=True,
        **SMALL,
    )
    folder = run_dir / "opt"
    loss_lines = (folder / "loss.csv").read_text().splitlines()
    history_lines = (folder / "history.csv").read_text().splitlines()
    domain_lines = (folder / "domain.csv").read_text().splitlines()

    assert loss_lines[0] == "iteration,f"
    assert len(loss_lines) == SMALL["iterations"] + 1
    assert history_lines[0] == "iteration,x,y,f"
    assert len(history_lines) == SMALL["iterations"] * SMALL["n_particles"] + 1
    assert domain_lines[1] == "-5.12,-5.12,5.12,5.12"


def test_history_only_for_two_dimensions(tmp_path: Path) -> None:
    problem = gfo.problems.Sphere()
    gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=3,
        run_dir=tmp_path,
        save_loss=True,
        save_history=True,
        **SMALL,
    )

    assert (tmp_path / "optimizer" / "loss.csv").is_file()
    assert not (tmp_path / "optimizer" / "history.csv").exists()


def test_verbose_and_diagnostics_reach_python_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    problem = gfo.cpp_problems.Sphere()
    result = gfo.pso(
        problem, problem.lo, problem.hi, d=2, verbose=True, diagnostics=True, **SMALL
    )
    output = capsys.readouterr().out

    assert "PARTICLE SWARM OPTIMIZATION" in output
    assert "PSO DIAGNOSTICS" in output
    assert result.diagnostics.thread_times
    assert result.diagnostics.objective_s > 0.0


def test_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    problem = gfo.problems.Sphere()
    gfo.pso(problem.evaluate, problem.lo, problem.hi, d=2, **SMALL)

    assert capsys.readouterr().out == ""
