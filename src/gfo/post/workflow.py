"""Make and save every optimizer figure of one run in a single call."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gfo.post.constants import HISTORY_FILENAME, LOSS_FILENAME
from gfo.post.loaders import OptimizerHistory, OptimizerLoss, load_history, load_loss
from gfo.post.plots import (
    DEFAULT_LABELS,
    AxisLabels,
    animate_landscape_3d,
    animate_swarm,
    plot_loss_contour,
    plot_loss_curve,
    plot_particle_paths,
)
from gfo.post.savers import EAnimationFormat, save_animation, save_figure


def save_and_close_figure(fig: Figure, path: Path) -> Path:
    """Save a figure, close it even if saving fails, and return its path."""
    try:
        save_figure(fig, path)
    finally:
        plt.close(fig)
    return path


def save_optimizer_figures(
    optimizer_dir: Path,
    output_dir: Path,
    *,
    labels: AxisLabels = DEFAULT_LABELS,
    include_static: bool = True,
    animation_format: EAnimationFormat | None = EAnimationFormat.GIF,
) -> list[Path]:
    """Make and save every optimizer figure of one run.

    Every run gets the loss curve. 2D runs saved with ``save_history=True``
    also get the trajectory and contour figures and the animations. The
    function is silent and closes every figure it makes.

    Parameters
    ----------
    optimizer_dir : Path
        Optimizer folder of the run, ``<run_dir>/<optimizer_dir>`` of
        ``gfo.pso``.
    output_dir : Path
        Folder the figures are saved in, created if needed.
    labels : tuple of str
        Axis labels of the two optimized parameters (2D runs).
    include_static : bool
        Save the trajectory and contour figures (2D runs).
    animation_format : EAnimationFormat or None
        Format of the animations (2D runs), None to skip them. The 3D
        landscape animation is slow (about a minute for 120 frames).

    Returns
    -------
    list of Path
        Every file saved, in the order it was saved.

    Raises
    ------
    FileNotFoundError
        If ``optimizer_dir`` does not exist or holds neither ``loss.csv`` nor
        ``history.csv``.
    ValueError
        If a file is malformed.
    RuntimeError
        If an MP4 is requested and ffmpeg is not installed.
    """
    source = Path(optimizer_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Optimizer folder not found: {source}")

    has_history: bool = (source / HISTORY_FILENAME).is_file()
    has_loss: bool = (source / LOSS_FILENAME).is_file()
    if not has_history and not has_loss:
        raise FileNotFoundError(
            f"No {LOSS_FILENAME} or {HISTORY_FILENAME} in {source}. Run gfo.pso "
            "with save_loss=True (any dimension) or save_history=True (2D)."
        )

    history: OptimizerHistory | None = load_history(source) if has_history else None
    if has_loss:
        loss: OptimizerLoss = load_loss(source)
    else:
        loss = OptimizerLoss(
            iterations=history.iterations, best=history.calc_best_so_far()
        )

    saved: list[Path] = []
    saved.append(
        save_and_close_figure(plot_loss_curve(loss), destination / "loss_curve.png")
    )
    if history is None:
        return saved

    if include_static:
        fig: Figure = plot_particle_paths(history, labels=labels)
        saved.append(save_and_close_figure(fig, destination / "particle_paths.png"))
        fig = plot_loss_contour(history, labels=labels)
        saved.append(save_and_close_figure(fig, destination / "contour_final.png"))
        fig = plot_loss_contour(history, labels=labels, cumulative=True)
        saved.append(save_and_close_figure(fig, destination / "contour_explored.png"))

    if animation_format is not None:
        suffix: str = animation_format.value
        for name, animate in (
            ("swarm", animate_swarm),
            ("landscape_3d", animate_landscape_3d),
        ):
            fig, anim = animate(history, labels=labels)
            path: Path = destination / f"{name}{suffix}"
            try:
                save_animation(anim, path)
            finally:
                plt.close(fig)
            saved.append(path)

    return saved
