"""Behaviour of gfo.pso in both evaluation modes."""

import numpy as np
import pytest

import gfo

pytestmark = pytest.mark.skipif(not gfo.EXTENSION_AVAILABLE, reason="C++ extension not built")

SMALL = {"n_particles": 16, "iterations": 60, "seed": 3}


def run_both(name: str, d: int, **kwargs):
    """Same problem through the C++ path and the Python batch path."""
    cpp_problem = getattr(gfo.cpp_problems, name)()
    cpp_result = gfo.pso(cpp_problem, d=d, **kwargs)
    py_result = gfo.pso(lambda X: cpp_problem(X), d=d, lo=cpp_problem.lo, hi=cpp_problem.hi, **kwargs)
    return cpp_result, py_result


@pytest.mark.parametrize("name", list(gfo.problems.EXAMPLE_PROBLEMS))
def test_cpp_and_batch_modes_are_identical(name: str) -> None:
    cpp_result, py_result = run_both(name, d=5, **SMALL)
    assert cpp_result.best_val == py_result.best_val
    np.testing.assert_array_equal(cpp_result.best_pos, py_result.best_pos)
    np.testing.assert_array_equal(cpp_result.loss_history, py_result.loss_history)


def test_results_do_not_depend_on_threads() -> None:
    problem = gfo.cpp_problems.Ackley()
    results = [gfo.pso(problem, d=6, threads=threads, **SMALL) for threads in (1, 2, 0)]
    assert len({result.best_val for result in results}) == 1


def test_seed_reproducible_and_different_seeds_differ() -> None:
    problem = gfo.problems.Rastrigin()
    first = gfo.pso(problem, d=4, **SMALL)
    again = gfo.pso(problem, d=4, **SMALL)
    other = gfo.pso(problem, d=4, **{**SMALL, "seed": 4})
    assert first.best_val == again.best_val
    assert first.best_val != other.best_val


def test_elementwise_matches_vectorized() -> None:
    problem = gfo.problems.Griewank()
    vectorized = gfo.pso(problem, d=3, **SMALL)
    elementwise = gfo.pso(problem, d=3, vectorized=False, **SMALL)
    assert vectorized.best_val == elementwise.best_val


def test_budget_history_and_bounds() -> None:
    problem = gfo.problems.Rosenbrock()
    result = gfo.pso(problem, d=4, **SMALL)
    assert result.evaluations == SMALL["n_particles"] * SMALL["iterations"]
    assert result.loss_history.shape == (SMALL["iterations"],)
    assert np.all(np.diff(result.loss_history) <= 0.0)
    assert result.loss_history[-1] == result.best_val
    assert np.all(result.best_pos >= problem.lo) and np.all(result.best_pos <= problem.hi)
    assert result.best_val == pytest.approx(problem(result.best_pos), rel=1e-12)


def test_python_objective_sees_every_particle_once_per_iteration() -> None:
    shapes = []

    def objective(batch: np.ndarray) -> np.ndarray:
        shapes.append(batch.shape)
        return np.sum(batch * batch, axis=1)

    gfo.pso(objective, d=3, lo=-1.0, hi=1.0, **SMALL)
    assert shapes == [(SMALL["n_particles"], 3)] * SMALL["iterations"]


def test_per_dimension_bounds_and_inferred_dimension() -> None:
    lo, hi = [0.0, 10.0, -5.0], [1.0, 20.0, -4.0]
    result = gfo.pso(lambda X: np.sum(X, axis=1), lo=lo, hi=hi, **SMALL)
    assert result.best_pos.shape == (3,)
    assert np.all(result.best_pos >= lo) and np.all(result.best_pos <= hi)


def test_default_bounds_is_unit_hypercube() -> None:
    result = gfo.pso(lambda X: np.sum((X - 0.25) ** 2, axis=1), d=2, default_bounds=True, **SMALL)
    assert np.all(result.best_pos >= 0.0) and np.all(result.best_pos <= 1.0)
    np.testing.assert_allclose(result.best_pos, 0.25, atol=1e-2)


def test_zero_iterations() -> None:
    result = gfo.pso(gfo.problems.Sphere(), d=2, iterations=0)
    assert result.evaluations == 0 and result.loss_history.size == 0


@pytest.mark.parametrize("kwargs, error", [
    ({"d": 0}, ValueError),
    ({"d": 2, "n_particles": 0}, ValueError),
    ({"d": 2, "iterations": -1}, ValueError),
    ({"d": 2, "max_velocity_rate": 0.0}, ValueError),
    ({"d": 2, "lo": 1.0, "hi": 0.0}, ValueError),
    ({"d": 2, "lo": [0.0, 0.0, 0.0], "hi": 1.0}, ValueError),
    ({"d": 2, "seed": -1}, ValueError),
])
def test_invalid_settings_raise(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        gfo.pso(gfo.problems.Sphere(), **{**SMALL, **kwargs})


def test_missing_bounds_raise() -> None:
    with pytest.raises(ValueError, match="lo and hi"):
        gfo.pso(lambda X: np.zeros(len(X)), d=2)


def test_wrong_return_shape_raises_with_hint() -> None:
    with pytest.raises(ValueError, match="vectorized=False"):
        gfo.pso(lambda X: float(np.sum(X)), d=2, lo=0.0, hi=1.0, **SMALL)


def test_non_numeric_return_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(lambda X: "nope", d=2, lo=0.0, hi=1.0, **SMALL)


def test_objective_exception_propagates() -> None:
    def broken(batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        gfo.pso(broken, d=2, lo=0.0, hi=1.0, **SMALL)


def test_not_callable_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(42, d=2, lo=0.0, hi=1.0)


def test_files_written(tmp_path) -> None:
    run_dir = tmp_path / "run"
    gfo.pso(gfo.cpp_problems.Sphere(), d=2, run_dir=run_dir, optimizer_dir="opt",
            save_loss=True, save_history=True, **SMALL)
    folder = run_dir / "opt"
    loss_lines = (folder / "loss.csv").read_text().splitlines()
    history_lines = (folder / "history.csv").read_text().splitlines()
    assert loss_lines[0] == "iteration,f" and len(loss_lines) == SMALL["iterations"] + 1
    assert history_lines[0] == "iteration,x,y,f"
    assert len(history_lines) == SMALL["iterations"] * SMALL["n_particles"] + 1
    assert (folder / "domain.csv").read_text().splitlines()[1] == "-5.12,-5.12,5.12,5.12"


def test_history_only_for_two_dimensions(tmp_path) -> None:
    gfo.pso(gfo.problems.Sphere(), d=3, run_dir=tmp_path, save_loss=True, save_history=True, **SMALL)
    assert (tmp_path / "optimizer" / "loss.csv").is_file()
    assert not (tmp_path / "optimizer" / "history.csv").exists()


def test_verbose_and_diagnostics_reach_python_stdout(capsys) -> None:
    result = gfo.pso(gfo.cpp_problems.Sphere(), d=2, verbose=True, diagnostics=True, **SMALL)
    output = capsys.readouterr().out
    assert "PARTICLE SWARM OPTIMIZATION" in output
    assert "PSO DIAGNOSTICS" in output
    assert result.diagnostics.thread_times
    assert result.diagnostics.objective_s > 0.0


def test_silent_by_default(capsys) -> None:
    gfo.pso(gfo.problems.Sphere(), d=2, **SMALL)
    assert capsys.readouterr().out == ""
