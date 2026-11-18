"""The standalone C++ example program (src/main.cpp), built without Python.

Its first run (Rastrigin, d = 2, 24 particles, 200 iterations, default seed)
saves its files, so the pure C++ build can be compared with the Python
binding of the same engine, and its files read back by gfo.post.

Build it first with ``python scripts/build.py --target example``, otherwise
these tests are skipped.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gfo
from gfo import post

pytestmark = pytest.mark.slow

EXECUTABLE = (
    Path(__file__).resolve().parents[1]
    / "build"
    / ("gfo_example.exe" if sys.platform == "win32" else "gfo_example")
)
TIMEOUT_S = 120
# The first run in main.cpp.
EXAMPLE_RUN = {"d": 2, "n_particles": 24, "iterations": 200}


@pytest.fixture(scope="module")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """Run the example once: its working folder and its console output."""
    if not EXECUTABLE.is_file():
        pytest.skip(f"C++ example not built ({EXECUTABLE.name}), run scripts/build.py")
    run_dir = tmp_path_factory.mktemp("cpp_example")
    completed = subprocess.run(
        [str(EXECUTABLE)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return run_dir, completed.stdout


def test_example_runs_to_the_end(example_run: tuple[Path, str]) -> None:
    _, output = example_run

    assert "PARTICLE SWARM OPTIMIZATION" in output
    assert "PSO DIAGNOSTICS" in output
    assert "PROGRAM FINISHED." in output
    for name in gfo.problems.EXAMPLE_PROBLEMS:
        assert f" | {name}" in output


def test_pure_cpp_build_matches_the_python_binding(
    example_run: tuple[Path, str],
) -> None:
    run_dir, _ = example_run
    problem = gfo.cpp_problems.Rastrigin()
    result = gfo.pso(problem, problem.lo, problem.hi, **EXAMPLE_RUN)
    loss = post.load_loss(run_dir / "cpp_example_run" / "optimizer")

    # loss.csv has 12 significant digits.
    np.testing.assert_allclose(loss.best, result.loss_history, rtol=1e-11)


def test_post_reads_the_cpp_files(
    example_run: tuple[Path, str], tmp_path: Path
) -> None:
    run_dir, _ = example_run
    saved = post.save_optimizer_figures(
        run_dir / "cpp_example_run" / "optimizer",
        tmp_path / "figures",
        animation_format=None,
    )

    assert len(saved) == 4
    assert all(path.stat().st_size > 0 for path in saved)
