"""Readers for the CSV files written by the C++ optimizer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gfo.post.constants import (
    DOMAIN_COLUMNS,
    DOMAIN_FILENAME,
    HISTORY_COLUMNS,
    HISTORY_FILENAME,
    LOSS_COLUMNS,
    LOSS_FILENAME,
)


@dataclass(slots=True)
class OptimizerHistory:
    """Every particle position and loss of a 2D optimizer run.

    Attributes
    ----------
    iterations : np.ndarray
        Iteration numbers, shape (n_iter,).
    x, y, f : np.ndarray
        Particle coordinates and loss, shape (n_iter, n_particles). Axis 0 is
        the iteration, axis 1 the particle.
    domain : tuple of float
        Search bounds ``(x_lo, x_hi, y_lo, y_hi)``.
    """

    iterations: np.ndarray
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    domain: tuple[float, float, float, float]

    def calc_best_so_far(self) -> np.ndarray:
        """Calculate the global best loss after each iteration, shape (n_iter,)."""
        return np.minimum.accumulate(self.f.min(axis=1))

    def calc_best_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the global best position after each iteration.

        Returns
        -------
        (np.ndarray, np.ndarray)
            x and y of the global best, shape (n_iter,) each.
        """

        #Type hinting for clarity, not for performance. The arrays are small enough that the overhead of np.arange and np.argmin is negligible.
        rows: np.ndarray = np.arange(self.f.shape[0])
        best_particle: np.ndarray = np.argmin(self.f, axis=1)
        iter_x: np.ndarray = self.x[rows, best_particle]
        iter_y: np.ndarray = self.y[rows, best_particle]
        iter_best: np.ndarray = self.f[rows, best_particle]

        # Hold the previous best position until an iteration improves on it.
        improved: np.ndarray = iter_best <= np.minimum.accumulate(iter_best)
        last_improved: np.ndarray = np.maximum.accumulate(np.where(improved, rows, 0))
        return iter_x[last_improved], iter_y[last_improved]


@dataclass(slots=True)
class OptimizerLoss:
    """Global best loss after every iteration (any dimension).

    Attributes
    ----------
    iterations : np.ndarray
        Iteration numbers, shape (n_iter,).
    best : np.ndarray
        Global best loss after each iteration, shape (n_iter,).
    """

    iterations: np.ndarray
    best: np.ndarray


def read_csv_columns(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    """Read the named columns of a numeric CSV file with a header row.

    Inputs
    ------
    path : Path
        CSV file.
    columns : tuple of str
        Columns to read, in the order they are returned.

    Returns
    -------
    np.ndarray
        Shape (n_rows, len(columns)), dtype float64. Column j is
        ``columns[j]``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a column is missing, there are no data rows, or a value is not a
        number.
    """
    lines: list[str] = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} has no data rows")

    header: list[str] = [name.strip() for name in lines[0].split(",")]
    indices: list[int] = []
    for column in columns:
        if column not in header:
            raise ValueError(f"{path} has no column {column!r}, found {header}")
        indices.append(header.index(column))

    table: np.ndarray = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    return table[:, indices]


def load_history(optimizer_dir: Path) -> OptimizerHistory:
    """Load the particle history of a 2D run.

    Inputs
    ------
    optimizer_dir : Path
        Optimizer folder of the run, ``<run_dir>/<optimizer_dir>``, holding
        ``history.csv`` and ``domain.csv`` (written by
        ``gfo.pso(..., save_history=True)`` for 2D runs only).

    Returns
    -------
    OptimizerHistory
        Every particle of every iteration, sorted by iteration.

    Raises
    ------
    FileNotFoundError
        If ``history.csv`` or ``domain.csv`` does not exist.
    ValueError
        If a file is malformed, or the particle count changes between
        iterations.
    """
    # table: shape=(n_iter * n_particles, 4), columns HISTORY_COLUMNS.
    table: np.ndarray = read_csv_columns(
        Path(optimizer_dir) / HISTORY_FILENAME, HISTORY_COLUMNS
    )
    order: np.ndarray = np.argsort(table[:, 0], kind="stable")
    table = table[order]

    iterations, counts = np.unique(table[:, 0], return_counts=True)
    if np.any(counts != counts[0]):
        raise ValueError(
            f"{HISTORY_FILENAME} has a different particle count per iteration"
        )
    shape: tuple[int, int] = (iterations.size, int(counts[0]))

    domain_row: np.ndarray = read_csv_columns(
        Path(optimizer_dir) / DOMAIN_FILENAME, DOMAIN_COLUMNS
    )[0]
    x_lo, x_hi, y_lo, y_hi = (float(value) for value in domain_row)

    return OptimizerHistory(
        iterations=iterations,
        x=table[:, 1].reshape(shape),
        y=table[:, 2].reshape(shape),
        f=table[:, 3].reshape(shape),
        domain=(x_lo, x_hi, y_lo, y_hi),
    )


def load_loss(optimizer_dir: Path) -> OptimizerLoss:
    """Load the global best loss history of a run (any dimension).

    Inputs
    ------
    optimizer_dir : Path
        Optimizer folder of the run, ``<run_dir>/<optimizer_dir>``, holding
        ``loss.csv`` (written by ``gfo.pso(..., save_loss=True)``).

    Returns
    -------
    OptimizerLoss
        Global best loss after every iteration.

    Raises
    ------
    FileNotFoundError
        If ``loss.csv`` does not exist.
    ValueError
        If the file is malformed.
    """
    # table: shape=(n_iter, 2), columns LOSS_COLUMNS.
    table: np.ndarray = read_csv_columns(
        Path(optimizer_dir) / LOSS_FILENAME, LOSS_COLUMNS
    )
    return OptimizerLoss(iterations=table[:, 0], best=table[:, 1])
