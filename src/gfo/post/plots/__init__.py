"""Optimizer plotting functions."""

from gfo.post.plots.common import apply_plot_style, finalize_animation, finalize_figure
from gfo.post.plots.optimizer import (
    animate_landscape_3d, animate_swarm, plot_loss_contour, plot_loss_curve,
    plot_particle_paths,
)

apply_plot_style()

__all__ = [
    "animate_landscape_3d", "animate_swarm", "apply_plot_style",
    "finalize_animation", "finalize_figure", "plot_loss_contour",
    "plot_loss_curve", "plot_particle_paths",
]
