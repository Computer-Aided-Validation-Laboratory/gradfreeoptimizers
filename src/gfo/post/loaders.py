"""Safe CSV readers for the files written by the C++ optimizer.

Every reader returns ``None`` (and logs a warning) instead of raising, so a
missing or broken file only skips the plots that need it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from gfo.post.console import log_error, log_warning
from gfo.post.constants import (
    COL_ITERATION, COL_VALUE, COL_X, COL_Y, DOMAIN_COLUMNS, OPT_DOMAIN_FILENAME,
    OPT_HISTORY_FILENAME, OPT_LOSS_FILENAME, OPT_POST_SUBDIR, OPT_SUBDIR,
    POST_SUFFIX,
)


@dataclass
class OptimizerHistory:
    """Every particle position and loss of a 2D optimizer run.

    Attributes
    ----------
    iterations : np.ndarray
        Iteration numbers, shape (n_iter,).
    x, y, f : np.ndarray
        Particle coordinates and loss, shape (n_iter, n_particles).
    domain : tuple of float
        Search bounds ``(x_lo, x_hi, y_lo, y_hi)``.
    """

    iterations: np.ndarray
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    domain: Tuple[float, float, float, float]

    def best_so_far(self) -> np.ndarray:
        """Global best loss up to and including each iteration, shape (n_iter,)."""
        return np.minimum.accumulate(self.f.min(axis=1))

    def best_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """Position of the global best after each iteration, shape (n_iter,) each."""
        flat_index = np.argmin(self.f, axis=1)
        rows = np.arange(self.f.shape[0])
        iter_x = self.x[rows, flat_index]
        iter_y = self.y[rows, flat_index]
        iter_best = self.f[rows, flat_index]

        # Hold the previous best position until an iteration improves on it.
        improved = iter_best <= np.minimum.accumulate(iter_best)
        last_improved = np.maximum.accumulate(np.where(improved, rows, 0))
        return iter_x[last_improved], iter_y[last_improved]


@dataclass
class OptimizerLoss:
    """Global best loss at every iteration (works for any dimension)."""

    iterations: np.ndarray
    best: np.ndarray


def read_csv_table(path: Path, required_cols: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """Read a CSV and check it has data and the required columns.

    Returns
    -------
    pandas.DataFrame or None
        None if the file is missing, unreadable, empty or lacks columns.
    """
    path = Path(path)
    if not path.is_file():
        log_warning(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log_warning(f"Could not read {path}: {exc}")
        return None

    df.columns = df.columns.str.strip()  # C++ writer can leave whitespace in headers
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        log_warning(f"{path.name} is missing column(s) {missing}, found {list(df.columns)}")
        return None
    if df.empty:
        log_warning(f"{path.name} contains no data")
        return None
    return df


def load_optimizer_history(history_path: Path, domain_path: Path) -> Optional[OptimizerHistory]:
    """Load ``history.csv`` (and ``domain.csv`` if present) of a 2D run.

    The domain falls back to the extent of the particles if the file is missing.

    Returns
    -------
    OptimizerHistory or None
        None on any read/shape problem (a warning is logged).
    """
    df = read_csv_table(history_path, (COL_ITERATION, COL_X, COL_Y, COL_VALUE))
    if df is None:
        return None

    cols = [COL_ITERATION, COL_X, COL_Y, COL_VALUE]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols)
    if df.empty:
        log_warning(f"{Path(history_path).name} has no valid rows")
        return None

    # The backend writes the same number of particles every iteration.
    df = df.sort_values(COL_ITERATION, kind="stable")
    counts = df.groupby(COL_ITERATION).size()
    if counts.nunique() != 1:
        log_warning(f"{Path(history_path).name} has a different particle count per iteration")
        return None
    n_particles = int(counts.iloc[0])
    n_iter = int(counts.size)

    def shape(col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=float).reshape(n_iter, n_particles)

    x, y, f = shape(COL_X), shape(COL_Y), shape(COL_VALUE)
    domain = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))

    domain_df = read_csv_table(domain_path, DOMAIN_COLUMNS)
    if domain_df is not None:
        row = domain_df.iloc[0]
        domain = (float(row["lo_x"]), float(row["hi_x"]), float(row["lo_y"]), float(row["hi_y"]))

    return OptimizerHistory(iterations=counts.index.to_numpy(dtype=float),
                            x=x, y=y, f=f, domain=domain)


def load_optimizer_loss(path: Path) -> Optional[OptimizerLoss]:
    """Load ``loss.csv`` (``iteration, f``) into an :class:`OptimizerLoss`."""
    df = read_csv_table(path, (COL_ITERATION, COL_VALUE))
    if df is None:
        return None

    df = df[[COL_ITERATION, COL_VALUE]].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        log_warning(f"{Path(path).name} has no valid rows")
        return None
    return OptimizerLoss(iterations=df[COL_ITERATION].to_numpy(dtype=float),
                         best=df[COL_VALUE].to_numpy(dtype=float))


class OptimizerReader:
    """Reads everything the optimizer saved in ``<run>/<optimizer_dir>/``.

    Parameters
    ----------
    run_dir : str or Path
        Run folder (the one holding the optimizer folder), i.e. the
        ``run_dir`` given to ``gfo.pso``.
    optimizer_dir : str
        Name of the optimizer folder inside ``run_dir`` (``optimizer_dir`` of
        ``gfo.pso``, default ``"optimizer"``).
    output_dir : str or Path, optional
        Where figures go. Default: ``<run>_post/optimizer`` next to the run
        folder.

    Attributes
    ----------
    is_valid : bool
        False if the run folder does not exist.
    has_optimizer_dir : bool
        False if the run folder has no optimizer subfolder.
    output_dir : Path
        Destination of the figures.
    """

    def __init__(self, run_dir, optimizer_dir: str = OPT_SUBDIR, output_dir=None) -> None:
        self.run_dir = Path(run_dir)
        self.optimizer_dir = self.run_dir / optimizer_dir
        self.is_valid = self.run_dir.is_dir()
        self.has_optimizer_dir = self.optimizer_dir.is_dir()
        resolved = self.run_dir.resolve()
        self.name = resolved.name
        default_output = resolved.parent / f"{resolved.name}{POST_SUFFIX}" / OPT_POST_SUBDIR
        self.output_dir = Path(output_dir) if output_dir is not None else default_output
        if not self.is_valid:
            log_error(f"Run folder not found: {self.run_dir}")

    def read_history(self) -> Optional[OptimizerHistory]:
        """Load the particle history. None if absent (runs with d != 2 do not save it)."""
        history_path = self.optimizer_dir / OPT_HISTORY_FILENAME
        if not history_path.is_file():
            return None
        return load_optimizer_history(history_path, self.optimizer_dir / OPT_DOMAIN_FILENAME)

    def read_loss(self) -> Optional[OptimizerLoss]:
        """Load the global best loss history. None if absent."""
        loss_path = self.optimizer_dir / OPT_LOSS_FILENAME
        if not loss_path.is_file():
            return None
        return load_optimizer_loss(loss_path)
