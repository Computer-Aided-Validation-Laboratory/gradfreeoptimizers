"""Make the optimizer figures and animations of one run folder.

Example
-------
    python scripts/plot_optimizer.py -r run1
    python scripts/plot_optimizer.py -r run1 --xlabel "Size (px)" --ylabel "Ratio"
    python scripts/plot_optimizer.py -r run1 --no-animations

Reads ``run1/optimizer/`` (written by ``gfo.pso(..., run_dir="run1",
save_loss=True, save_history=True)`` or the C++ optimizer) and saves the
figures to ``run1_post/optimizer/``. Runs with more than 2 parameters only get
the loss curve.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

from gfo import post
from gfo.optimizer import DEFAULT_OPTIMIZER_DIR

POST_SUFFIX = "_post"  # Default output: <run>_post/optimizer/ next to the run.
POST_SUBDIR = "optimizer"


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, save the figures and report what was saved."""
    parser = argparse.ArgumentParser(
        description="Plot and animate the optimizer data of a run folder."
    )
    parser.add_argument(
        "-r",
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder that contains the optimizer folder.",
    )
    parser.add_argument(
        "--optimizer-dir",
        default=DEFAULT_OPTIMIZER_DIR,
        help=f"Name of the optimizer folder (default: {DEFAULT_OPTIMIZER_DIR}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: <run-dir>_post/optimizer).",
    )
    parser.add_argument("--xlabel", default="x", help="Label of the first parameter.")
    parser.add_argument("--ylabel", default="y", help="Label of the second parameter.")
    parser.add_argument(
        "--no-static", action="store_true", help="Skip the static 2D figures."
    )
    parser.add_argument(
        "--no-animations",
        action="store_true",
        help="Skip the animations (they are the slow part).",
    )
    parser.add_argument(
        "--format",
        choices=[file_format.name.lower() for file_format in post.EAnimationFormat],
        default="gif",
        help="Animation format (mp4 needs ffmpeg, default: gif).",
    )
    args = parser.parse_args(argv)

    # Figures are only saved: no GUI backend (works without a display).
    matplotlib.use("Agg")
    run_dir: Path = args.run_dir.resolve()
    output_dir: Path = args.output
    if output_dir is None:
        output_dir = run_dir.parent / f"{run_dir.name}{POST_SUFFIX}" / POST_SUBDIR

    animation_format: post.EAnimationFormat | None = None
    if not args.no_animations:
        animation_format = post.EAnimationFormat[args.format.upper()]

    print(f"Optimizer post-processing of: {run_dir}")
    print(f"Saving to: {output_dir}")
    start: float = time.perf_counter()
    saved: list[Path] = post.save_optimizer_figures(
        run_dir / args.optimizer_dir,
        output_dir,
        labels=(args.xlabel, args.ylabel),
        include_static=not args.no_static,
        animation_format=animation_format,
    )
    for path in saved:
        print(f"Saved: {path}")
    print(f"Saved {len(saved)} file(s) in {time.perf_counter() - start:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
