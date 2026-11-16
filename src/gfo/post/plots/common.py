"""Shared plotting helpers: styling and saving figures/animations."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from gfo.post.console import log_error, log_info, log_warning
from gfo.post.constants import ANIMATION_DPI, ANIMATION_FPS, DEFAULT_DPI


def apply_plot_style() -> None:
    """Set global matplotlib defaults used by every figure."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "savefig.dpi": DEFAULT_DPI,
    })


def finalize_figure(fig: Figure, filename: str, output_dir: Path,
                    save: bool, show: bool) -> Optional[Path]:
    """Save and/or keep a figure open for ``plt.show()``.

    Parameters
    ----------
    fig : Figure
        Finished figure.
    filename : str
        File name inside ``output_dir``.
    output_dir : Path
        Destination folder, created if needed.
    save, show : bool
        Save to disk and/or leave open for interactive display.
        If ``show`` is False the figure is closed to free memory.

    Returns
    -------
    Path or None
        Saved path, or None if nothing was saved.
    """
    saved_path = None
    if save:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_path = output_dir / filename
            fig.savefig(saved_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            log_info(f"Saved: {saved_path}")
        except OSError as exc:
            log_error(f"Could not save {filename}: {exc}")
            saved_path = None
    if not show:
        plt.close(fig)
    return saved_path


def _pick_animation_writer(suffix: str):
    # GIF needs only Pillow. MP4 needs ffmpeg, which may not be installed.
    if suffix == ".mp4":
        if animation.writers.is_available("ffmpeg"):
            return animation.FFMpegWriter(fps=ANIMATION_FPS)
        log_warning("ffmpeg not found, saving a GIF instead of an MP4.")
    return animation.PillowWriter(fps=ANIMATION_FPS)


def finalize_animation(fig: Figure, anim: FuncAnimation, filename: str,
                       output_dir: Path, save: bool, show: bool) -> Optional[Path]:
    """Save and/or keep an animation open for ``plt.show()``.

    Parameters
    ----------
    fig : Figure
        Figure the animation draws on.
    anim : FuncAnimation
        Finished animation.
    filename : str
        File name inside ``output_dir``. ``.gif`` or ``.mp4`` (MP4 falls back
        to GIF if ffmpeg is missing).
    output_dir : Path
        Destination folder, created if needed.
    save, show : bool
        Save to disk and/or leave open for interactive display.
        If ``show`` is False the figure is closed to free memory.

    Returns
    -------
    Path or None
        Saved path, or None if nothing was saved.
    """
    saved_path = None
    if save:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            requested = output_dir / filename
            writer = _pick_animation_writer(requested.suffix)
            saved_path = requested
            if isinstance(writer, animation.PillowWriter):
                saved_path = requested.with_suffix(".gif")
            log_info(f"Rendering {saved_path.name}...")
            anim.save(saved_path, writer=writer, dpi=ANIMATION_DPI)
            log_info(f"Saved: {saved_path}")
        except (OSError, RuntimeError) as exc:
            log_error(f"Could not save {filename}: {exc}")
            saved_path = None
    if not show:
        plt.close(fig)
    return saved_path
