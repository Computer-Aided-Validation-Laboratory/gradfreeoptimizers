"""The gfo.pso interface: inputs, errors, results, files and console output."""

from pathlib import Path

import numpy as np
import pytest

import gfo
from gfo import optimizer
from pso_testing import SwarmRecorder

SMALL = {"n_particles": 16, "iterations": 60, "seed": 3}


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
    assert not settings.default_bounds


def test_result_fields() -> None:
    problem = gfo.problems.Rosenbrock()
    result = gfo.pso(problem.evaluate, problem.lo, problem.hi, d=4, **SMALL)

    assert result.evaluations == SMALL["n_particles"] * SMALL["iterations"]
    assert result.best_pos.shape == (4,)
    assert result.best_pos.dtype == np.float64
    assert result.loss_history.shape == (SMALL["iterations"],)
    assert np.all(np.diff(result.loss_history) <= 0.0)
    assert result.loss_history[-1] == result.best_val
    best_value = problem.evaluate(result.best_pos[np.newaxis, :])[0]
    assert result.best_val == pytest.approx(best_value, rel=1e-12)


def test_result_arrays_are_copies() -> None:
    result = gfo.pso(gfo.cpp_problems.Sphere(), -1.0, 1.0, d=2, **SMALL)
    result.best_pos[:] = 99.0
    result.loss_history[:] = 99.0

    assert not np.any(result.best_pos == 99.0)
    assert not np.any(result.loss_history == 99.0)


@pytest.mark.parametrize("mode", ["cpp", "batch"])
def test_diagnostics_describe_the_run(mode: str) -> None:
    problem = gfo.cpp_problems.Sphere()
    objective = problem if mode == "cpp" else problem.evaluate
    result = gfo.pso(objective, problem.lo, problem.hi, d=3, threads=2, **SMALL)
    stats = result.diagnostics

    assert stats.evaluations == result.evaluations
    assert stats.threads == 2
    assert stats.batch == (mode == "batch")
    assert not stats.thread_times
    assert stats.total_s > 0.0
    parts = stats.setup_s + stats.swarm_s + stats.evaluate_s + stats.best_s
    assert parts <= stats.total_s * (1.0 + 1e-9)
    assert set(stats.to_dict()) == {
        "total_s",
        "setup_s",
        "swarm_s",
        "evaluate_s",
        "objective_s",
        "update_s",
        "best_s",
        "output_s",
        "evaluations",
        "threads",
        "batch",
        "thread_times",
    }


def test_python_objective_sees_every_particle_once_per_iteration() -> None:
    recorder = SwarmRecorder(lambda points: np.sum(points * points, axis=1))
    gfo.pso(recorder.evaluate, -1.0, 1.0, d=3, **SMALL)

    assert len(recorder.batches) == SMALL["iterations"]
    assert all(batch.shape == (SMALL["n_particles"], 3) for batch in recorder.batches)
    assert all(batch.dtype == np.float64 for batch in recorder.batches)


@pytest.mark.parametrize(
    ("lo", "hi", "d"),
    [
        (-2.0, 3.0, 3),
        ([0.0, 10.0, -5.0], [1.0, 20.0, -4.0], None),
        (-1.0, [1.0, 2.0], None),
        ([0.0, 0.0], 1.0, None),
        (np.array([-1e-6, 1e6]), np.array([1e-6, 2e6]), 2),
    ],
    ids=["scalar", "per_dimension", "scalar_lo", "scalar_hi", "arrays"],
)
def test_every_evaluated_point_is_inside_the_bounds(
    lo: object, hi: object, d: int | None
) -> None:
    recorder = SwarmRecorder(lambda points: np.sum(points, axis=1))
    result = gfo.pso(recorder.evaluate, lo, hi, d=d, **SMALL)
    points = recorder.stack_points()
    dims = points.shape[2]
    lo_array = np.broadcast_to(np.asarray(lo, dtype=float), (dims,))
    hi_array = np.broadcast_to(np.asarray(hi, dtype=float), (dims,))

    assert np.all(points >= lo_array)
    assert np.all(points <= hi_array)
    assert result.best_pos.shape == (dims,)


def test_zero_iterations() -> None:
    calls: list[int] = []

    def evaluate(points: np.ndarray) -> np.ndarray:
        calls.append(len(points))
        return np.zeros(len(points))

    result = gfo.pso(evaluate, -1.0, 1.0, d=2, iterations=0)

    assert calls == []
    assert result.evaluations == 0
    assert result.loss_history.size == 0
    assert result.best_val == np.inf


def test_single_particle_and_single_dimension() -> None:
    result = gfo.pso(lambda points: points[:, 0] ** 2, -1.0, 1.0, d=1, n_particles=1)

    assert result.evaluations == optimizer.DEFAULT_ITERATIONS
    assert np.isfinite(result.best_val)


def test_whole_number_floats_are_accepted_as_integers() -> None:
    result = gfo.pso(
        gfo.cpp_problems.Sphere(), -1.0, 1.0, d=2.0, n_particles=4.0, iterations=3.0
    )

    assert result.evaluations == 12


@pytest.mark.parametrize(
    "returned",
    [
        lambda points: np.sum(points, axis=1)[:, np.newaxis],
        lambda points: np.sum(points, axis=1).tolist(),
        lambda points: np.arange(len(points)),
        lambda points: np.sum(points, axis=1).astype(np.float32),
    ],
    ids=["column", "list", "integers", "float32"],
)
def test_objective_may_return_any_n_numbers(returned: object) -> None:
    result = gfo.pso(returned, 0.0, 1.0, d=2, **SMALL)

    assert np.isfinite(result.best_val)


@pytest.mark.parametrize(
    ("lo", "hi", "kwargs"),
    [
        (0.0, 1.0, {"d": 0}),
        (0.0, 1.0, {"d": -1}),
        (0.0, 1.0, {"d": 2.5}),
        (0.0, 1.0, {"d": True}),
        (0.0, 1.0, {"d": 2, "n_particles": 0}),
        (0.0, 1.0, {"d": 2, "iterations": -1}),
        (0.0, 1.0, {"d": 2, "seed": -1}),
        (0.0, 1.0, {"d": 2, "threads": -1}),
        (0.0, 1.0, {"d": 2, "verbose_interval": 0}),
        (0.0, 1.0, {"d": 2, "diagnostics_interval": -1}),
        (0.0, 1.0, {"d": 2, "max_velocity_rate": 0.0}),
        (0.0, 1.0, {"d": 2, "max_velocity_rate": np.nan}),
        (0.0, 1.0, {"d": 2, "initial_velocity_rate": -0.1}),
        (0.0, 1.0, {"d": 2, "optimizer_dir": "", "save_loss": True}),
        (1.0, 0.0, {"d": 2}),
        (1.0, 1.0, {"d": 2}),
        ([0.0, 2.0], [1.0, 1.0], {}),
        (np.nan, 1.0, {"d": 2}),
        (0.0, np.inf, {"d": 2}),
        ([0.0, 0.0, 0.0], 1.0, {"d": 2}),
        (np.zeros((2, 2)), 1.0, {"d": 2}),
        ([0.0, 0.0], [1.0, 1.0, 1.0], {}),
        (0.0, 1.0, {}),
    ],
)
def test_invalid_settings_raise(lo: object, hi: object, kwargs: dict) -> None:
    problem = gfo.problems.Sphere()
    with pytest.raises(ValueError):
        gfo.pso(problem.evaluate, lo, hi, **{**SMALL, **kwargs})


def test_seed_beyond_64_bits_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(gfo.cpp_problems.Sphere(), 0.0, 1.0, d=2, seed=2**64)


def test_elementwise_flag_with_a_cpp_problem_raises() -> None:
    problem = gfo.cpp_problems.Sphere()
    with pytest.raises(ValueError, match="vectorized=False"):
        gfo.pso(problem, problem.lo, problem.hi, d=2, vectorized=False)


def test_wrong_return_shape_raises_with_hint() -> None:
    with pytest.raises(ValueError, match="vectorized=False"):
        gfo.pso(lambda points: float(np.sum(points)), 0.0, 1.0, d=2, **SMALL)


def test_too_many_values_raise() -> None:
    with pytest.raises(ValueError, match="must return"):
        gfo.pso(lambda points: np.zeros(len(points) + 1), 0.0, 1.0, d=2, **SMALL)


def test_non_numeric_return_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(lambda points: "nope", 0.0, 1.0, d=2, **SMALL)


@pytest.mark.parametrize("vectorized", [True, False])
def test_objective_exception_propagates(vectorized: bool) -> None:
    def evaluate_broken(points: np.ndarray) -> np.ndarray:
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        gfo.pso(evaluate_broken, 0.0, 1.0, d=2, vectorized=vectorized, **SMALL)


def test_not_callable_raises() -> None:
    with pytest.raises(TypeError):
        gfo.pso(42, 0.0, 1.0, d=2)


def test_files_written(tmp_path: Path) -> None:
    recorder = SwarmRecorder(gfo.problems.Sphere().evaluate)
    run_dir = tmp_path / "run"
    result = gfo.pso(
        recorder.evaluate,
        -5.12,
        5.12,
        d=2,
        run_dir=run_dir,
        optimizer_dir="opt",
        save_loss=True,
        save_history=True,
        **SMALL,
    )
    folder = run_dir / "opt"
    loss = np.loadtxt(folder / "loss.csv", delimiter=",", skiprows=1)
    history = np.loadtxt(folder / "history.csv", delimiter=",", skiprows=1)

    assert (folder / "loss.csv").read_text().splitlines()[0] == "iteration,f"
    assert (folder / "history.csv").read_text().splitlines()[0] == "iteration,x,y,f"
    assert (folder / "domain.csv").read_text().splitlines()[1] == (
        "-5.12,-5.12,5.12,5.12"
    )
    np.testing.assert_array_equal(loss[:, 0], np.arange(SMALL["iterations"]))
    np.testing.assert_allclose(loss[:, 1], result.loss_history, rtol=1e-11)
    # history.csv holds every point the objective was given, in order.
    iterations = np.repeat(np.arange(SMALL["iterations"]), SMALL["n_particles"])
    np.testing.assert_array_equal(history[:, 0], iterations)
    np.testing.assert_allclose(
        history[:, 1:3], recorder.stack_points().reshape(-1, 2), rtol=1e-11
    )
    np.testing.assert_allclose(
        history[:, 3], recorder.stack_values().ravel(), rtol=1e-11, atol=1e-300
    )


def test_files_are_overwritten_by_the_next_run(tmp_path: Path) -> None:
    problem = gfo.cpp_problems.Sphere()
    for iterations in (40, 10):
        gfo.pso(
            problem,
            problem.lo,
            problem.hi,
            d=2,
            n_particles=4,
            iterations=iterations,
            run_dir=tmp_path,
            save_loss=True,
            save_history=True,
        )
    folder = tmp_path / "optimizer"

    assert len((folder / "loss.csv").read_text().splitlines()) == 11
    assert len((folder / "history.csv").read_text().splitlines()) == 41


def test_no_files_unless_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    problem = gfo.cpp_problems.Sphere()
    gfo.pso(problem, problem.lo, problem.hi, d=2, **SMALL)
    gfo.pso(problem, problem.lo, problem.hi, d=2, run_dir=tmp_path / "run", **SMALL)

    assert list(tmp_path.iterdir()) == []


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
    assert not (tmp_path / "optimizer" / "domain.csv").exists()


@pytest.mark.parametrize("mode", ["cpp", "batch"])
def test_verbose_and_diagnostics_reach_python_stdout(
    mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    problem = gfo.cpp_problems.Sphere()
    objective = problem if mode == "cpp" else problem.evaluate
    result = gfo.pso(
        objective,
        problem.lo,
        problem.hi,
        d=2,
        verbose=True,
        diagnostics=True,
        **SMALL,
    )
    output = capsys.readouterr().out

    assert "PARTICLE SWARM OPTIMIZATION" in output
    assert "PSO DIAGNOSTICS" in output
    assert "PSO finished" in output
    assert result.diagnostics.thread_times == (mode == "cpp")
    assert result.diagnostics.objective_s > 0.0


def test_verbose_interval_sets_the_progress_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gfo.pso(
        gfo.cpp_problems.Sphere(),
        -1.0,
        1.0,
        d=2,
        n_particles=4,
        iterations=50,
        verbose=True,
        verbose_interval=10,
    )
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(" | Iteration = ")
    ]

    # Iteration 0 is the burn in, then every 10th and the last one.
    assert [int(line.split()[3]) for line in lines] == [10, 20, 30, 40, 49]


@pytest.mark.parametrize("mode", ["cpp", "batch", "elementwise"])
def test_silent_by_default(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    problem = gfo.cpp_problems.Sphere()
    objective = {
        "cpp": problem,
        "batch": problem.evaluate,
        "elementwise": lambda point: float(np.sum(point * point)),
    }[mode]
    gfo.pso(
        objective,
        problem.lo,
        problem.hi,
        d=2,
        vectorized=mode != "elementwise",
        **SMALL,
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""


def test_reprs_are_informative() -> None:
    result = gfo.pso(gfo.cpp_problems.Sphere(), -1.0, 1.0, d=2, **SMALL)

    assert repr(result).startswith("<PSOResult best_val=")
    assert "particles=20" in repr(gfo.PSOSettings())
    assert repr(gfo.cpp_problems.Rastrigin(shift=1.5)).startswith(
        "Rastrigin(shift=1.5"
    )
