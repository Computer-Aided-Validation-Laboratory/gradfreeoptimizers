"""Optimizer figures and animations.

Static: loss curve (any dimension), particle trajectories and loss contours
reconstructed from the particles (2D runs). Animated: the swarm next to a
growing loss curve, and a 3D landscape that fills in as particles reach new
positions (2D runs).

Every function only builds the figure. Save it with ``gfo.post.save_figure``
or ``gfo.post.save_animation``, show it with ``plt.show()``, and close it with
``plt.close(fig)`` when done.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation

from gfo.post.constants import ANIMATION_FPS, MAX_ANIMATION_FRAMES
from gfo.post.loaders import OptimizerHistory, OptimizerLoss

AxisLabels = tuple[str, str]
DEFAULT_LABELS: AxisLabels = ("x", "y")

LOSS_LABEL = "Loss $f$"
BEST_LOSS_LABEL = "Global best loss $f$"
LANDSCAPE_CMAP = "viridis"
ITERATION_CMAP = "plasma"

CONTOUR_LEVELS = 15
LOG_SCALE_MIN_RATIO = 10.0  # Log loss axis only if the loss spans this ratio.
LOG_PAD_FACTOR = 1.5  # Headroom on a log loss axis (multiplicative).
MIN_TRIANGULATION_POINTS = 3
# Distance (as a fraction of the domain) under which two particles are one point.
DUPLICATE_TOLERANCE = 1e-9
Z_MARGIN_FRACTION = 0.05  # Headroom on the 3D loss axis.
ZOOM_MARGIN_FRACTION = 0.1  # Margin around the particles on contour plots.
MIN_ZOOM_FRACTION = 0.01  # Zoom half-width (of the domain) if particles coincide.
LANDSCAPE_ELEVATION = 28.0  # 3D view angles in degrees.
LANDSCAPE_AZIMUTH = -58.0


def select_frames(n_iter: int) -> np.ndarray:
    """Select the iteration indices to animate, evenly skipped above the limit.

    Returns
    -------
    np.ndarray
        Sorted unique iteration indices, shape (<= MAX_ANIMATION_FRAMES,).
    """
    n_frames: int = min(n_iter, MAX_ANIMATION_FRAMES)
    frames: np.ndarray = np.linspace(0, n_iter - 1, n_frames)
    return np.unique(np.round(frames).astype(int))


def build_triangulation(
    x: np.ndarray, y: np.ndarray, domain: tuple[float, float, float, float]
) -> tuple[Triangulation, np.ndarray] | None:
    """Delaunay triangulate scattered particles.

    The triangulation is built in unit domain coordinates (the axes can have
    very different scales, e.g. size in px against a ratio) and then reused
    with the physical coordinates. Repeated positions are dropped.

    Parameters
    ----------
    x, y : np.ndarray
        Particle coordinates, shape (n_points,).
    domain : tuple of float
        Search bounds ``(x_lo, x_hi, y_lo, y_hi)``.

    Returns
    -------
    (Triangulation, np.ndarray) or None
        The triangulation and the indices of the points it uses, shape
        (n_used,), or None if there are too few distinct or non-collinear
        points.
    """
    x_lo, x_hi, y_lo, y_hi = domain
    unit_x: np.ndarray = (x - x_lo) / (x_hi - x_lo)
    unit_y: np.ndarray = (y - y_lo) / (y_hi - y_lo)

    keys: np.ndarray = np.round(np.column_stack((unit_x, unit_y)) / DUPLICATE_TOLERANCE)
    _, first_seen = np.unique(keys, axis=0, return_index=True)
    keep: np.ndarray = np.sort(first_seen)
    if keep.size < MIN_TRIANGULATION_POINTS:
        return None

    try:
        unit_tri = Triangulation(unit_x[keep], unit_y[keep])
    except (ValueError, RuntimeError):  # Collinear points.
        return None
    if unit_tri.triangles.shape[0] == 0:
        return None

    tri = Triangulation(x[keep], y[keep], triangles=unit_tri.triangles)
    return tri, keep


def calc_loss_range(history: OptimizerHistory) -> tuple[float, float]:
    """Calculate the loss range of a history, never empty (colour scales need it)."""
    f_min = float(history.f.min())
    f_max = float(history.f.max())
    if f_max <= f_min:
        f_max = f_min + 1.0
    return f_min, f_max


def check_log_scale(values: np.ndarray) -> bool:
    """Check whether positive ``values`` span enough decades for a log axis."""
    if values.min() <= 0:
        return False
    return float(values.max() / values.min()) >= LOG_SCALE_MIN_RATIO


def style_domain_axes(ax: Axes, history: OptimizerHistory, labels: AxisLabels) -> None:
    """Set the limits of ``ax`` to the search domain and label both axes."""
    x_lo, x_hi, y_lo, y_hi = history.domain
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


def calc_zoom_limits(
    values: np.ndarray, domain_lo: float, domain_hi: float
) -> tuple[float, float]:
    """Calculate the particle extent plus a margin, never outside the domain."""
    low = float(values.min())
    high = float(values.max())
    pad: float = (high - low) * ZOOM_MARGIN_FRACTION
    if pad == 0.0:
        pad = (domain_hi - domain_lo) * MIN_ZOOM_FRACTION
    return max(low - pad, domain_lo), min(high + pad, domain_hi)


def calc_padded_limits(values: np.ndarray, *, log_scale: bool) -> tuple[float, float]:
    """Calculate axis limits around ``values`` with some headroom."""
    low = float(values.min())
    high = float(values.max())
    if log_scale:
        return low / LOG_PAD_FACTOR, high * LOG_PAD_FACTOR
    pad: float = (high - low) * Z_MARGIN_FRACTION
    if pad == 0.0:
        pad = 1.0
    return low - pad, high + pad


def plot_loss_curve(loss: OptimizerLoss) -> Figure:
    """Plot the global best loss against iteration (any dimension).

    The loss axis is logarithmic when the loss is positive and spans at least
    ``LOG_SCALE_MIN_RATIO``.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(loss.iterations, loss.best, color="tab:blue", linewidth=1.8)
    if check_log_scale(loss.best):
        ax.set_yscale("log")
    ax.set_title("Optimizer Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(BEST_LOSS_LABEL)
    ax.grid(alpha=0.3)
    return fig


def plot_particle_paths(
    history: OptimizerHistory, *, labels: AxisLabels = DEFAULT_LABELS
) -> Figure:
    """Plot every particle position coloured by iteration, and the best path."""
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    # iteration_grid: shape=(n_iter, n_particles), the iteration of each point.
    iteration_grid: np.ndarray = np.broadcast_to(
        history.iterations[:, np.newaxis], history.x.shape
    )

    points = ax.scatter(
        history.x.ravel(),
        history.y.ravel(),
        c=iteration_grid.ravel(),
        cmap=ITERATION_CMAP,
        s=14,
        alpha=0.8,
    )
    best_x, best_y = history.calc_best_path()
    ax.plot(best_x, best_y, color="black", linewidth=1.2, label="Global best path")
    ax.scatter(
        history.x[0],
        history.y[0],
        marker="s",
        s=40,
        facecolors="none",
        edgecolors="black",
        label="Initial swarm",
    )
    ax.scatter(
        best_x[-1],
        best_y[-1],
        marker="*",
        s=260,
        color="gold",
        edgecolors="black",
        zorder=5,
        label="Final best",
    )

    style_domain_axes(ax, history, labels)
    ax.set_title("Swarm Trajectories")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(points, ax=ax, label="Iteration")
    return fig


def plot_loss_contour(
    history: OptimizerHistory,
    *,
    labels: AxisLabels = DEFAULT_LABELS,
    cumulative: bool = False,
) -> Figure:
    """Plot filled loss contours reconstructed from the particles alone.

    When there are too few distinct particle positions to triangulate, only
    the particles are drawn.

    Parameters
    ----------
    history : OptimizerHistory
        Particle history of a 2D run.
    labels : tuple of str
        Axis labels ``(x, y)``.
    cumulative : bool
        False uses only the final swarm. True uses every particle of every
        iteration, which shows everything the swarm explored.
    """
    if cumulative:
        x: np.ndarray = history.x.ravel()
        y: np.ndarray = history.y.ravel()
        f: np.ndarray = history.f.ravel()
        title = "Explored Loss Landscape (all iterations)"
    else:
        x = history.x[-1]
        y = history.y[-1]
        f = history.f[-1]
        title = "Loss Landscape at Final Iteration"

    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    built = build_triangulation(x, y, history.domain)
    if built is not None:
        tri, keep = built
        f_min = float(f[keep].min())
        f_max = float(f[keep].max())
        if f_max <= f_min:
            f_max = f_min + 1.0
        levels: np.ndarray = np.linspace(f_min, f_max, CONTOUR_LEVELS)
        filled = ax.tricontourf(tri, f[keep], levels=levels, cmap=LANDSCAPE_CMAP)
        ax.tricontour(
            tri, f[keep], levels=levels, colors="black", linewidths=0.4, alpha=0.5
        )
        fig.colorbar(filled, ax=ax, label=LOSS_LABEL)

    ax.scatter(
        x, y, s=22, color="white", edgecolors="black", linewidths=0.6, label="Particles"
    )
    best = int(np.argmin(f))
    ax.scatter(
        x[best],
        y[best],
        marker="*",
        s=260,
        color="gold",
        edgecolors="black",
        zorder=5,
        label="Best",
    )

    # Zoom to the particles, a converged swarm is a speck on the full domain.
    x_lo, x_hi, y_lo, y_hi = history.domain
    style_domain_axes(ax, history, labels)
    ax.set_xlim(*calc_zoom_limits(x, x_lo, x_hi))
    ax.set_ylim(*calc_zoom_limits(y, y_lo, y_hi))
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return fig


def animate_swarm(
    history: OptimizerHistory, *, labels: AxisLabels = DEFAULT_LABELS
) -> tuple[Figure, FuncAnimation]:
    """Animate the swarm over the search domain next to a growing loss curve.

    Returns
    -------
    (Figure, FuncAnimation)
        The figure and its animation. Keep a reference to the animation until
        it is saved or shown.
    """
    fig, (ax_swarm, ax_loss) = plt.subplots(
        1,
        2,
        figsize=(12, 5.5),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    f_min, f_max = calc_loss_range(history)
    norm = Normalize(vmin=f_min, vmax=f_max)
    best_x, best_y = history.calc_best_path()
    best_loss: np.ndarray = history.calc_best_so_far()

    # Swarm panel.
    explored = ax_swarm.scatter(
        [], [], s=6, color="0.75", zorder=1, label="Previously visited"
    )
    swarm = ax_swarm.scatter(
        history.x[0],
        history.y[0],
        c=history.f[0],
        cmap=LANDSCAPE_CMAP,
        norm=norm,
        s=50,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    (trail,) = ax_swarm.plot(
        [], [], color="red", linewidth=1.5, zorder=2, label="Global best path"
    )
    star = ax_swarm.scatter(
        [],
        [],
        marker="*",
        s=240,
        color="gold",
        edgecolors="black",
        zorder=4,
        label="Global best",
    )
    style_domain_axes(ax_swarm, history, labels)
    ax_swarm.set_title("Swarm")
    ax_swarm.legend(loc="upper right", fontsize=8)
    fig.colorbar(swarm, ax=ax_swarm, label=LOSS_LABEL)

    # Loss panel.
    (line,) = ax_loss.plot([], [], color="tab:blue", linewidth=1.8)
    (dot,) = ax_loss.plot([], [], "o", color="tab:red")
    ax_loss.set_xlim(history.iterations[0], history.iterations[-1])
    is_log: bool = check_log_scale(best_loss)
    if is_log:
        ax_loss.set_yscale("log")
    ax_loss.set_ylim(*calc_padded_limits(best_loss, log_scale=is_log))
    ax_loss.set_title("Convergence")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel(BEST_LOSS_LABEL)
    ax_loss.grid(alpha=0.3)

    def draw_frame(frame: int) -> tuple[Artist, ...]:
        swarm.set_offsets(np.column_stack((history.x[frame], history.y[frame])))
        swarm.set_array(history.f[frame])
        # visited: shape=(frame * n_particles, 2), every earlier position.
        visited: np.ndarray = np.column_stack(
            (history.x[:frame].ravel(), history.y[:frame].ravel())
        )
        explored.set_offsets(visited)
        trail.set_data(best_x[: frame + 1], best_y[: frame + 1])
        star.set_offsets([[best_x[frame], best_y[frame]]])
        line.set_data(history.iterations[: frame + 1], best_loss[: frame + 1])
        dot.set_data([history.iterations[frame]], [best_loss[frame]])
        fig.suptitle(
            f"Iteration {int(history.iterations[frame])}  |  "
            f"best $f$ = {best_loss[frame]:.4g}"
        )
        return swarm, explored, trail, star, line, dot

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=select_frames(history.f.shape[0]),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    return fig, anim


def animate_landscape_3d(
    history: OptimizerHistory, *, labels: AxisLabels = DEFAULT_LABELS
) -> tuple[Figure, FuncAnimation]:
    """Animate a 3D loss landscape that fills in as the swarm explores.

    The axes start empty. Each frame triangulates every particle position seen
    so far into a surface, so the landscape grows out from the initial swarm
    and settles as the swarm converges. Slow to render, every frame is
    triangulated from scratch.

    Returns
    -------
    (Figure, FuncAnimation)
        The figure and its animation. Keep a reference to the animation until
        it is saved or shown.
    """
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    f_min, f_max = calc_loss_range(history)
    norm = Normalize(vmin=f_min, vmax=f_max)
    z_low, z_high = calc_padded_limits(np.array([f_min, f_max]), log_scale=False)
    x_lo, x_hi, y_lo, y_hi = history.domain
    best_loss: np.ndarray = history.calc_best_so_far()

    scalar_map = ScalarMappable(norm=norm, cmap=LANDSCAPE_CMAP)
    fig.colorbar(scalar_map, ax=ax, shrink=0.6, aspect=14, pad=0.08, label=LOSS_LABEL)

    def draw_frame(frame: int) -> None:
        # Redraw from scratch, the surface changes shape every frame.
        ax.cla()
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_low, z_high)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(LOSS_LABEL)
        ax.view_init(elev=LANDSCAPE_ELEVATION, azim=LANDSCAPE_AZIMUTH)

        # seen_*: shape=((frame + 1) * n_particles,), every position so far.
        seen_x: np.ndarray = history.x[: frame + 1].ravel()
        seen_y: np.ndarray = history.y[: frame + 1].ravel()
        seen_f: np.ndarray = history.f[: frame + 1].ravel()
        built = build_triangulation(seen_x, seen_y, history.domain)
        if built is not None:
            tri, keep = built
            ax.plot_trisurf(
                tri,
                seen_f[keep],
                cmap=LANDSCAPE_CMAP,
                norm=norm,
                linewidth=0,
                antialiased=False,
                alpha=0.9,
            )

        ax.scatter(
            history.x[frame],
            history.y[frame],
            history.f[frame],
            c="red",
            s=18,
            depthshade=False,
        )
        ax.set_title(
            f"Iteration {int(history.iterations[frame])}  |  "
            f"best $f$ = {best_loss[frame]:.4g}"
        )

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=select_frames(history.f.shape[0]),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    return fig, anim
