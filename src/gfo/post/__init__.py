"""Post-processing of saved optimizer runs: loss curve, swarm figures, animations.

The optimizer saves its files when asked to (``gfo.pso(..., run_dir="run1",
save_loss=True, save_history=True)``). This package turns them into figures::

    import gfo

    gfo.post.run_optimizer_post_processing("run1")

    options = gfo.post.OptimizerOptions(xlabel="Speckle size (px)",
                                        ylabel="Ratio", plot_animations=False)
    gfo.post.run_optimizer_post_processing("run1", options)

Every reader returns ``None`` and logs a warning instead of raising, and every
plotting step is wrapped, so a missing file or one broken plot only skips that
figure. Runs with ``d != 2`` only get the loss curve.
"""

from gfo.post.loaders import OptimizerHistory, OptimizerLoss, OptimizerReader
from gfo.post.plots import (
    animate_landscape_3d, animate_swarm, finalize_animation, finalize_figure,
    plot_loss_contour, plot_loss_curve, plot_particle_paths,
)
from gfo.post.workflow import OptimizerOptions, run_optimizer_post_processing

__all__ = [
    "OptimizerHistory", "OptimizerLoss", "OptimizerOptions", "OptimizerReader",
    "animate_landscape_3d", "animate_swarm", "finalize_animation",
    "finalize_figure", "plot_loss_contour", "plot_loss_curve",
    "plot_particle_paths", "run_optimizer_post_processing",
]
