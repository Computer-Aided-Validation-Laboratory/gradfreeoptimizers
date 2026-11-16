"""How the gfo Python interface works, start to finish.

    python scripts/example_python.py            # everything except the animations
    python scripts/example_python.py --animate  # also render the GIFs (slow)

1. A Python (vectorized) objective: the swarm update runs in C++, Python
   evaluates the whole swarm once per iteration.
2. The same problem as a C++ example problem: the whole run stays in C++.
3. Your own objective, vectorized or one point at a time.
4. A 2D run saved to disk and turned into figures by gfo.post.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import gfo
from gfo import post

DIMENSION = 10
ITERATIONS = 1000
PARTICLES = 40
RUN_DIR = Path("python_example_run")
FIGURE_DIR = Path("python_example_run_post")


def evaluate_bowl_batch(points: np.ndarray) -> np.ndarray:
    """Quartic bowl centred on 0.5, points shape (n, d) -> values shape (n,)."""
    return np.sum((points - 0.5) ** 4, axis=1)


def evaluate_bowl_point(point: np.ndarray) -> float:
    """The same bowl for one point, shape (d,) -> float."""
    return float(np.sum((point - 0.5) ** 4))


def main() -> None:
    """Run every example."""
    parser = argparse.ArgumentParser(description="gfo Python interface example.")
    parser.add_argument(
        "--animate", action="store_true", help="Also render the animations (slow)."
    )
    args = parser.parse_args()
    # Figures are only saved: no GUI backend (works without a display).
    matplotlib.use("Agg")

    # 1 and 2. Same problem, same seed: Python objective against C++ objective.
    problem = gfo.problems.Rastrigin()
    python_result = gfo.pso(
        problem.evaluate,
        problem.lo,
        problem.hi,
        d=DIMENSION,
        iterations=ITERATIONS,
        n_particles=PARTICLES,
    )
    cpp_problem = gfo.cpp_problems.Rastrigin()
    cpp_result = gfo.pso(
        cpp_problem,
        cpp_problem.lo,
        cpp_problem.hi,
        d=DIMENSION,
        iterations=ITERATIONS,
        n_particles=PARTICLES,
    )
    print(
        f"Python objective: best = {python_result.best_val:.6g} "
        f"in {python_result.diagnostics.total_s:.3f} s"
    )
    print(
        f"C++ objective   : best = {cpp_result.best_val:.6g} "
        f"in {cpp_result.diagnostics.total_s:.3f} s"
    )
    print(f"Best position (C++): {np.round(cpp_result.best_pos, 4)}\n")

    # 3. Your own functions: vectorized (fast) or one point at a time.
    fast = gfo.pso(evaluate_bowl_batch, -1.0, 1.0, d=3, iterations=300)
    slow = gfo.pso(
        evaluate_bowl_point, -1.0, 1.0, d=3, iterations=300, vectorized=False
    )
    print(
        f"Vectorized objective: best = {fast.best_val:.3g} "
        f"in {fast.diagnostics.total_s:.4f} s"
    )
    print(
        f"Per point objective : best = {slow.best_val:.3g} "
        f"in {slow.diagnostics.total_s:.4f} s\n"
    )

    # 4. A verbose 2D run with diagnostics, saved and post-processed.
    ackley = gfo.problems.Ackley()
    gfo.pso(
        ackley.evaluate,
        ackley.lo,
        ackley.hi,
        d=2,
        iterations=150,
        n_particles=24,
        verbose=True,
        verbose_interval=25,
        diagnostics=True,
        run_dir=RUN_DIR,
        save_loss=True,
        save_history=True,
    )
    animation_format: post.EAnimationFormat | None = None
    if args.animate:
        animation_format = post.EAnimationFormat.GIF
    saved: list[Path] = post.save_optimizer_figures(
        RUN_DIR / "optimizer", FIGURE_DIR, animation_format=animation_format
    )
    for path in saved:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
