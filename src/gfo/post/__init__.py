"""Post-processing of saved optimizer runs: loss curve, swarm figures, animations.

The optimizer saves its files when asked to (``gfo.pso(..., run_dir="run1",
save_loss=True, save_history=True)``). This package turns them into figures.
Every function is silent and raises on missing or malformed files::

    from pathlib import Path

    import matplotlib.pyplot as plt

    from gfo import post

    # Everything in one call.
    saved = post.save_optimizer_figures(Path("run1/optimizer"), Path("figures"))

    # Or step by step: load, plot, then save and/or show.
    history = post.load_history(Path("run1/optimizer"))
    fig = post.plot_particle_paths(history, labels=("Size (px)", "Ratio"))
    post.save_figure(fig, Path("figures/paths.png"))
    plt.show()

Runs with ``d != 2`` only save ``loss.csv``, so they only get the loss curve.
"""

from gfo.post.loaders import (
    OptimizerHistory,
    OptimizerLoss,
    load_history,
    load_loss,
)
from gfo.post.plots import (
    DEFAULT_LABELS,
    animate_landscape_3d,
    animate_swarm,
    plot_loss_contour,
    plot_loss_curve,
    plot_particle_paths,
)
from gfo.post.savers import EAnimationFormat, save_animation, save_figure
from gfo.post.workflow import save_optimizer_figures

__all__ = [
    "DEFAULT_LABELS",
    "EAnimationFormat",
    "OptimizerHistory",
    "OptimizerLoss",
    "animate_landscape_3d",
    "animate_swarm",
    "load_history",
    "load_loss",
    "plot_loss_contour",
    "plot_loss_curve",
    "plot_particle_paths",
    "save_animation",
    "save_figure",
    "save_optimizer_figures",
]
