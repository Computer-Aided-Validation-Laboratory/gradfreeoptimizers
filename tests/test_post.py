"""Post-processing of a saved optimizer run (static figures only, animations are slow)."""

import pytest

import gfo

pytestmark = pytest.mark.skipif(not gfo.EXTENSION_AVAILABLE, reason="C++ extension not built")


def test_post_processing_of_a_2d_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    gfo.pso(gfo.problems.Rastrigin(), d=2, n_particles=12, iterations=30,
            run_dir=run_dir, save_loss=True, save_history=True)

    output_dir = tmp_path / "figures"
    options = gfo.post.OptimizerOptions(plot_animations=False, output_dir=output_dir)
    saved = gfo.post.run_optimizer_post_processing(run_dir, options)

    names = sorted(path.name for path in saved)
    assert names == ["contour_explored.png", "contour_final.png", "loss_curve.png", "particle_paths.png"]
    assert all(path.is_file() for path in saved)

    history = gfo.post.OptimizerReader(run_dir).read_history()
    assert history.x.shape == (30, 12)
    assert history.domain == (-5.12, 5.12, -5.12, 5.12)


def test_post_processing_of_a_high_dimensional_run(tmp_path) -> None:
    gfo.pso(gfo.problems.Sphere(), d=5, iterations=20, run_dir=tmp_path, save_loss=True)
    options = gfo.post.OptimizerOptions(output_dir=tmp_path / "figures")
    saved = gfo.post.run_optimizer_post_processing(tmp_path, options)
    assert [path.name for path in saved] == ["loss_curve.png"]


def test_missing_run_folder_returns_nothing(tmp_path) -> None:
    assert gfo.post.run_optimizer_post_processing(tmp_path / "missing") == []
