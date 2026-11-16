"""Save figures and animations to disk."""

from enum import Enum
from pathlib import Path

from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from gfo.post.constants import ANIMATION_DPI, ANIMATION_FPS, FIGURE_DPI


class EAnimationFormat(Enum):
    """File format of a saved animation (the value is the file suffix)."""

    GIF = ".gif"  # Needs only Pillow (a matplotlib dependency).
    MP4 = ".mp4"  # Needs ffmpeg on the PATH.


def save_figure(fig: Figure, path: Path) -> None:
    """Save a figure, creating the parent folder if needed.

    The figure is left open: close it with ``plt.close(fig)`` when done.

    Parameters
    ----------
    fig : Figure
        Finished figure.
    path : Path
        Destination file. The suffix picks the format (``.png``, ``.pdf``...).
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")


def save_animation(anim: FuncAnimation, path: Path) -> None:
    """Render and save an animation, creating the parent folder if needed.

    Parameters
    ----------
    anim : FuncAnimation
        Finished animation.
    path : Path
        Destination file, ending in ``.gif`` or ``.mp4``.

    Raises
    ------
    ValueError
        If the suffix is not one of ``EAnimationFormat``.
    RuntimeError
        If an MP4 is requested and ffmpeg is not installed.
    """
    destination = Path(path)
    valid_suffixes: list[str] = [file_format.value for file_format in EAnimationFormat]
    if destination.suffix not in valid_suffixes:
        raise ValueError(
            f"Animations are saved as one of {valid_suffixes}, got {destination.name}"
        )

    file_format = EAnimationFormat(destination.suffix)
    if file_format is EAnimationFormat.MP4:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Saving an MP4 needs ffmpeg, save a .gif instead")
        writer: animation.AbstractMovieWriter = animation.FFMpegWriter(
            fps=ANIMATION_FPS
        )
    else:
        writer = animation.PillowWriter(fps=ANIMATION_FPS)

    destination.parent.mkdir(parents=True, exist_ok=True)
    anim.save(destination, writer=writer, dpi=ANIMATION_DPI)
