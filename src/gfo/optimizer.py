"""Python interface to the C++ particle swarm optimizer.

``gfo.pso`` picks the evaluation mode from what it is given:

- A C++ example problem (``gfo.cpp_problems.Rastrigin()`` etc.): the whole run,
  objective included, stays in C++ and every particle is evaluated in
  parallel (OpenMP), with the GIL released.
- Any Python callable: it is called once per iteration with the whole swarm,
  an ``(n, d)`` array, and must return ``n`` values. The swarm update in
  between runs in C++ (in parallel). Pass ``vectorized=False`` for a function
  that takes a single point, it is then called once per particle.

For the same seed and settings both modes visit exactly the same points.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from gfo._extension import get_extension

Bounds = float | Sequence[float] | np.ndarray | None


def _resolve_dimension(d: int | None, lo: np.ndarray | None, hi: np.ndarray | None) -> int:
    # An explicit d wins, otherwise use the length of an array bound.
    if d is not None:
        if isinstance(d, bool) or int(d) != d or d < 1:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        return int(d)

    sizes = {bound.size for bound in (lo, hi) if bound is not None and bound.size > 1}
    if len(sizes) == 1:
        return sizes.pop()
    if len(sizes) > 1:
        raise ValueError(f"lo and hi have different lengths {sorted(sizes)}")
    raise ValueError("Pass the dimension d (it cannot be inferred from scalar bounds)")


def _to_bound_array(bound: Bounds, label: str) -> np.ndarray | None:
    if bound is None:
        return None
    array = np.atleast_1d(np.asarray(bound, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"{label} must be a number or a 1D sequence, got shape {array.shape}")
    return array


def _broadcast_bound(bound: np.ndarray, d: int, label: str) -> list[float]:
    if bound.size not in (1, d):
        raise ValueError(f"{label} has {bound.size} entries but d = {d}")
    return np.broadcast_to(bound, (d,)).tolist()


def resolve_bounds(objective: object, d: int | None, lo: Bounds, hi: Bounds,
                   default_bounds: bool) -> tuple[int, list[float], list[float]]:
    """Work out the dimension and per-dimension bounds of a run.

    Bounds not passed are taken from the objective's ``lo``/``hi`` attributes
    (every example problem has them). Scalars apply to every dimension.

    Returns
    -------
    (int, list of float, list of float)
        ``d``, ``lo`` and ``hi`` (each of length ``d``).
    """
    if lo is None:
        lo = getattr(objective, "lo", None)
    if hi is None:
        hi = getattr(objective, "hi", None)
    lo_array = _to_bound_array(lo, "lo")
    hi_array = _to_bound_array(hi, "hi")
    dims = _resolve_dimension(d, lo_array, hi_array)

    if default_bounds:
        return dims, [0.0] * dims, [1.0] * dims
    if lo_array is None or hi_array is None:
        raise ValueError("Pass lo and hi (or use a problem with default bounds, "
                         "or default_bounds=True for the unit hypercube)")
    return dims, _broadcast_bound(lo_array, dims, "lo"), _broadcast_bound(hi_array, dims, "hi")


def make_batch_objective(point_objective: Callable) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a single point objective ``f(x) -> float`` as a batch objective.

    The wrapped function calls ``f`` once per row, which is how elementwise
    problems are evaluated in pymoo.
    """
    def evaluate_rows(batch: np.ndarray) -> np.ndarray:
        values = np.empty(batch.shape[0])
        for i, row in enumerate(batch):
            values[i] = point_objective(row)
        return values

    return evaluate_rows


def _check_int(value: int, label: str, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def pso(objective: Callable,
        d: int | None = None,
        lo: Bounds = None,
        hi: Bounds = None,
        *,
        n_particles: int = 20,
        iterations: int = 100,
        inertia: float = 0.729,
        acceleration1: float = 1.49445,
        acceleration2: float = 1.49445,
        max_velocity_rate: float = 0.15,
        initial_velocity_rate: float = 0.02,
        seed: int = 42,
        default_bounds: bool = False,
        vectorized: bool = True,
        threads: int = 0,
        verbose: bool = False,
        verbose_interval: int = 1,
        diagnostics: bool = False,
        diagnostics_interval: int = 0,
        run_dir: str | Path | None = None,
        optimizer_dir: str = "optimizer",
        save_loss: bool = False,
        save_history: bool = False):
    """Minimize ``objective`` with the C++ particle swarm optimizer.

    Parameters
    ----------
    objective : callable or C++ problem
        A C++ example problem (``gfo.cpp_problems.*``, runs fully in C++), or
        a Python callable. By default the callable is vectorized: it gets an
        ``(n, d)`` array (one row per particle, physical coordinates) and
        returns ``n`` values. With ``vectorized=False`` it gets one point
        ``(d,)`` and returns a float.
    d : int, optional
        Dimension. Inferred from ``lo``/``hi`` if they are arrays.
    lo, hi : float or sequence of float, optional
        Bounds, a scalar applies to every dimension. Default: the objective's
        ``lo``/``hi`` attributes (every example problem has them).
    n_particles : int
        Swarm size.
    iterations : int
        Iterations. Every iteration evaluates every particle once, so the
        budget is ``n_particles * iterations`` evaluations.
    inertia, acceleration1, acceleration2 : float
        Inertia, cognitive and social coefficients (defaults: Clerc-Kennedy
        constriction values).
    max_velocity_rate : float
        Velocity clamp as a fraction of the domain width (pymoo's
        ``max_velocity_rate``).
    initial_velocity_rate : float
        Initial velocities are uniform in ``[-rate, rate]`` (fraction of the
        domain width). 0 starts the swarm at rest.
    seed : int
        Seed. Results do not depend on the thread count.
    default_bounds : bool
        Ignore ``lo``/``hi`` and search the unit hypercube.
    vectorized : bool
        Whether a Python objective takes the whole swarm (True) or one point.
    threads : int
        OpenMP threads for the particle loop, 0 = all cores.
    verbose : bool
        Print a start/end summary and a line every ``verbose_interval``
        iterations.
    verbose_interval : int
        Iterations between verbose lines.
    diagnostics : bool
        Time every section of the loop and print a breakdown (objective,
        particle update, bookkeeping, output) every ``diagnostics_interval``
        iterations and at the end.
    diagnostics_interval : int
        Iterations between diagnostics lines, 0 = about 10 lines per run.
    run_dir : str or Path, optional
        Folder the optimizer folder is created in (default: current folder).
    optimizer_dir : str
        Name of the optimizer folder inside ``run_dir``.
    save_loss : bool
        Save the global best of every iteration to ``loss.csv``.
    save_history : bool
        2D only: save every particle of every iteration to ``history.csv``
        and the bounds to ``domain.csv`` (used by ``gfo.post``).

    Returns
    -------
    gfo.PSOResult
        ``best_pos`` (physical coordinates), ``best_val``, ``loss_history``
        (global best after every iteration), ``evaluations`` and
        ``diagnostics`` (timings in seconds).
    """
    extension = get_extension()
    dims, lo_list, hi_list = resolve_bounds(objective, d, lo, hi, default_bounds)

    settings = extension.PSOSettings()
    settings.particles = _check_int(n_particles, "n_particles", 1)
    settings.iterations = _check_int(iterations, "iterations", 0)
    settings.seed = _check_int(seed, "seed", 0)
    settings.threads = _check_int(threads, "threads", 0)
    settings.verbose_interval = _check_int(verbose_interval, "verbose_interval", 1)
    settings.diagnostics_interval = _check_int(diagnostics_interval, "diagnostics_interval", 0)
    settings.inertia = float(inertia)
    settings.acceleration1 = float(acceleration1)
    settings.acceleration2 = float(acceleration2)
    settings.max_velocity_rate = float(max_velocity_rate)
    settings.initial_velocity_rate = float(initial_velocity_rate)
    settings.default_bounds = bool(default_bounds)
    settings.verbose = bool(verbose)
    settings.diagnostics = bool(diagnostics)
    settings.run_dir = "" if run_dir is None else os.fspath(run_dir)
    settings.optimizer_dir = str(optimizer_dir)
    settings.save_loss = bool(save_loss)
    settings.save_history = bool(save_history)

    if isinstance(objective, extension.CPP_PROBLEMS):
        return extension.optimize_problem(objective, dims, lo_list, hi_list, settings)

    if not callable(objective):
        raise TypeError(f"objective must be callable or a C++ problem, got {type(objective).__name__}")
    batch_objective = objective if vectorized else make_batch_objective(objective)
    return extension.optimize_python(batch_objective, dims, lo_list, hi_list, settings)
