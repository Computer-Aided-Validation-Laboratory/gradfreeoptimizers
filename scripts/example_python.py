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

import numpy as np

import gfo

DIMENSION = 10
ITERATIONS = 1000
RUN_DIR = Path("python_example_run")


def compare_modes() -> None:
    """Same problem, same seed: Python objective vs C++ objective."""
    python_result = gfo.pso(gfo.problems.Rastrigin(), d=DIMENSION, iterations=ITERATIONS,
                            n_particles=40)
    cpp_result = gfo.pso(gfo.cpp_problems.Rastrigin(), d=DIMENSION, iterations=ITERATIONS,
                         n_particles=40)
    print(f"Python objective: best = {python_result.best_val:.6g} "
          f"in {python_result.diagnostics.total_s:.3f} s")
    print(f"C++ objective   : best = {cpp_result.best_val:.6g} "
          f"in {cpp_result.diagnostics.total_s:.3f} s")
    print(f"Best position (C++): {np.round(cpp_result.best_pos, 4)}\n")


def custom_objectives() -> None:
    """Your own functions: vectorized (fast) or one point at a time."""
    def vectorized_bowl(batch: np.ndarray) -> np.ndarray:
        return np.sum((batch - 0.5) ** 4, axis=1)

    def point_bowl(point: np.ndarray) -> float:
        return float(np.sum((point - 0.5) ** 4))

    fast = gfo.pso(vectorized_bowl, d=3, lo=-1.0, hi=1.0, iterations=300)
    slow = gfo.pso(point_bowl, d=3, lo=-1.0, hi=1.0, iterations=300, vectorized=False)
    print(f"Vectorized objective : best = {fast.best_val:.3g} in {fast.diagnostics.total_s:.4f} s")
    print(f"Per point objective  : best = {slow.best_val:.3g} in {slow.diagnostics.total_s:.4f} s\n")


def saved_run(animate: bool) -> None:
    """A verbose 2D run with diagnostics, saved and post-processed."""
    problem = gfo.problems.Ackley()
    gfo.pso(problem, d=2, iterations=150, n_particles=24, verbose=True, verbose_interval=25,
            diagnostics=True, run_dir=RUN_DIR, save_loss=True, save_history=True)

    options = gfo.post.OptimizerOptions(plot_animations=animate)
    gfo.post.run_optimizer_post_processing(RUN_DIR, options)


def main() -> None:
    """Run every example."""
    parser = argparse.ArgumentParser(description="gfo Python interface example.")
    parser.add_argument("--animate", action="store_true", help="Also render the animations (slow).")
    args = parser.parse_args()

    compare_modes()
    custom_objectives()
    saved_run(args.animate)


if __name__ == "__main__":
    main()
