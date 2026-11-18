"""Post-processing of saved optimizer runs."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import gfo
from gfo import post
from pso_testing import SwarmBests, SwarmRecorder, replay_bests

N_PARTICLES = 12
ITERATIONS = 30
# The CSV files are written with 12 significant digits.
CSV_RTOL = 1e-11


@pytest.fixture
def run_2d(tmp_path: Path) -> tuple[Path, gfo.PSOResult]:
    """A saved 2D Rastrigin run: its optimizer folder and its result."""
    problem = gfo.problems.Rastrigin()
    result = gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=2,
        n_particles=N_PARTICLES,
        iterations=ITERATIONS,
        run_dir=tmp_path / "run",
        save_loss=True,
        save_history=True,
    )
    return tmp_path / "run" / "optimizer", result


def test_load_history_matches_the_run(run_2d: tuple[Path, gfo.PSOResult]) -> None:
    optimizer_dir, result = run_2d
    history = post.load_history(optimizer_dir)

    assert history.x.shape == (ITERATIONS, N_PARTICLES)
    assert history.domain == (-5.12, 5.12, -5.12, 5.12)
    # The best of every particle seen so far is the optimizer's global best.
    np.testing.assert_allclose(
        history.calc_best_so_far(), result.loss_history, rtol=CSV_RTOL
    )


def test_load_history_returns_every_evaluated_point(tmp_path: Path) -> None:
    recorder = SwarmRecorder(gfo.problems.Ackley().evaluate)
    gfo.pso(
        recorder.evaluate,
        [-32.768, -10.0],
        [32.768, 5.0],
        n_particles=N_PARTICLES,
        iterations=ITERATIONS,
        run_dir=tmp_path,
        save_history=True,
    )
    history = post.load_history(tmp_path / "optimizer")
    points = recorder.stack_points()

    assert history.domain == (-32.768, 32.768, -10.0, 5.0)
    np.testing.assert_allclose(history.x, points[:, :, 0], rtol=CSV_RTOL)
    np.testing.assert_allclose(history.y, points[:, :, 1], rtol=CSV_RTOL)


def save_recorded_2d_run(
    tmp_path: Path, objective: Callable[[np.ndarray], np.ndarray]
) -> tuple[post.OptimizerHistory, SwarmBests]:
    """Save a 2D run on [-5, 5]^2: its loaded history and its replayed bests."""
    recorder = SwarmRecorder(objective)
    gfo.pso(
        recorder.evaluate,
        -5.0,
        5.0,
        d=2,
        n_particles=N_PARTICLES,
        iterations=ITERATIONS,
        seed=1,
        run_dir=tmp_path,
        save_history=True,
    )
    bests = replay_bests(recorder.stack_points(), recorder.stack_values())
    return post.load_history(tmp_path / "optimizer"), bests


def test_best_path_follows_the_optimizer(tmp_path: Path) -> None:
    history, bests = save_recorded_2d_run(tmp_path, gfo.problems.Ackley().evaluate)
    best_x, best_y = history.calc_best_path()

    np.testing.assert_allclose(best_x, bests.gbest_pos[:, 0], rtol=CSV_RTOL)
    np.testing.assert_allclose(best_y, bests.gbest_pos[:, 1], rtol=CSV_RTOL)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "calc_best_path moves to a later point on an exact tie (<=), the "
        "optimizer keeps the earliest best (<)"
    ),
)
def test_best_path_follows_the_optimizer_on_ties(tmp_path: Path) -> None:
    history, bests = save_recorded_2d_run(
        tmp_path, lambda points: np.sum(np.floor(np.abs(points - 0.3) * 4.0), axis=1)
    )
    best_x, best_y = history.calc_best_path()

    np.testing.assert_allclose(best_x, bests.gbest_pos[:, 0], rtol=CSV_RTOL)
    np.testing.assert_allclose(best_y, bests.gbest_pos[:, 1], rtol=CSV_RTOL)


def test_load_loss_matches_the_run(run_2d: tuple[Path, gfo.PSOResult]) -> None:
    optimizer_dir, result = run_2d
    loss = post.load_loss(optimizer_dir)

    np.testing.assert_array_equal(loss.iterations, np.arange(ITERATIONS))
    np.testing.assert_allclose(loss.best, result.loss_history, rtol=CSV_RTOL)


def test_static_figures_of_a_2d_run(
    run_2d: tuple[Path, gfo.PSOResult], tmp_path: Path
) -> None:
    optimizer_dir, _ = run_2d
    saved = post.save_optimizer_figures(
        optimizer_dir, tmp_path / "figures", animation_format=None
    )

    names = sorted(path.name for path in saved)
    assert names == [
        "contour_explored.png",
        "contour_final.png",
        "loss_curve.png",
        "particle_paths.png",
    ]
    assert all(path.is_file() for path in saved)


def test_animations_of_a_short_2d_run(tmp_path: Path) -> None:
    problem = gfo.problems.Sphere()
    gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=2,
        n_particles=6,
        iterations=4,
        run_dir=tmp_path,
        save_history=True,
    )
    saved = post.save_optimizer_figures(
        tmp_path / "optimizer",
        tmp_path / "figures",
        include_static=False,
        animation_format=post.EAnimationFormat.GIF,
    )

    assert [path.name for path in saved] == [
        "loss_curve.png",
        "swarm.gif",
        "landscape_3d.gif",
    ]
    assert all(path.stat().st_size > 0 for path in saved)


def test_high_dimensional_run_only_gets_the_loss_curve(tmp_path: Path) -> None:
    problem = gfo.problems.Sphere()
    gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=5,
        iterations=20,
        run_dir=tmp_path,
        save_loss=True,
    )
    saved = post.save_optimizer_figures(tmp_path / "optimizer", tmp_path / "figures")

    assert [path.name for path in saved] == ["loss_curve.png"]


def test_missing_optimizer_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        post.save_optimizer_figures(tmp_path / "missing", tmp_path / "figures")


def test_optimizer_folder_without_data_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="save_loss"):
        post.save_optimizer_figures(tmp_path, tmp_path / "figures")


def test_malformed_csv_raises(tmp_path: Path) -> None:
    (tmp_path / "loss.csv").write_text("iteration,wrong\n0,1.0\n")

    with pytest.raises(ValueError, match="'f'"):
        post.load_loss(tmp_path)
