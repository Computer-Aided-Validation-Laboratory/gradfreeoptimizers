"""Python interface to the C++ particle swarm optimizer.

``gfo.pso`` picks the evaluation mode from the objective it is given:

- A C++ example problem (``gfo.cpp_problems.Rastrigin()`` etc.): the whole
  run, objective included, stays in C++ and every particle is evaluated in
  parallel (OpenMP), with the GIL released.
- A Python callable: it is called once per iteration with the whole swarm,
  points of shape (n, d), and returns the n values. The swarm update in
  between runs in C++ (in parallel). With ``vectorized=False`` it takes a
  single point of shape (d,) and is called once per particle.

For the same seed and settings both modes visit exactly the same points.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from gfo import _pso

# Defaults of the original C++ optimizer (PSOSettings in src/pso/pso.hpp).
# Inertia and accelerations are the Clerc-Kennedy constriction values.
DEFAULT_PARTICLES = 20
DEFAULT_ITERATIONS = 100
DEFAULT_INERTIA = 0.729
DEFAULT_ACCELERATION = 1.49445
# Velocity clamp, as a fraction of the domain width (pymoo's meaning).
DEFAULT_MAX_VELOCITY_RATE = 0.15
# Initial velocities are uniform in +/- this fraction of the domain width.
DEFAULT_INITIAL_VELOCITY_RATE = 0.02
DEFAULT_SEED = 42
DEFAULT_OPTIMIZER_DIR = "optimizer"

# Every C++ example problem class. A C++ problem runs fully in C++.
CppProblem = (
    _pso.problems.Sphere
    | _pso.problems.Ellipsoid
    | _pso.problems.Rosenbrock
    | _pso.problems.Rastrigin
    | _pso.problems.Ackley
    | _pso.problems.Griewank
    | _pso.problems.Schwefel
)

# A scalar bound applies to every dimension.
Bounds = float | Sequence[float] | np.ndarray
BatchObjective = Callable[[np.ndarray], np.ndarray]
PointObjective = Callable[[np.ndarray], float]


def validate_integer(value: int, label: str, minimum: int) -> int:
    """Return ``value`` as an int, raising if it is not an integer >= minimum.

    Raises
    ------
    ValueError
        If ``value`` is a bool, not a whole number, or below ``minimum``.
    """
    is_bool: bool = isinstance(value, bool)
    if is_bool or int(value) != value or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def pso(
    objective: BatchObjective | PointObjective | CppProblem,
    lo: Bounds,
    hi: Bounds,
    *,
    d: int | None = None,
    n_particles: int = DEFAULT_PARTICLES,
    iterations: int = DEFAULT_ITERATIONS,
    inertia: float = DEFAULT_INERTIA,
    acceleration1: float = DEFAULT_ACCELERATION,
    acceleration2: float = DEFAULT_ACCELERATION,
    max_velocity_rate: float = DEFAULT_MAX_VELOCITY_RATE,
    initial_velocity_rate: float = DEFAULT_INITIAL_VELOCITY_RATE,
    seed: int = DEFAULT_SEED,
    vectorized: bool = True,
    threads: int = 0,
    verbose: bool = False,
    verbose_interval: int = 1,
    diagnostics: bool = False,
    diagnostics_interval: int = 0,
    run_dir: str | Path | None = None,
    optimizer_dir: str = DEFAULT_OPTIMIZER_DIR,
    save_loss: bool = False,
    save_history: bool = False,
) -> _pso.PSOResult:
    """Minimize ``objective`` with the C++ particle swarm optimizer.

    The search is unitless: bounds and objective must use one consistent
    system of units.

    Parameters
    ----------
    objective : callable or C++ problem
        A C++ example problem (``gfo.cpp_problems.*``, runs fully in C++), or
        a Python callable. By default the callable is vectorized: it gets
        points of shape (n, d), one row per particle in physical coordinates,
        and returns the n values, shape (n,). With ``vectorized=False`` it
        gets one point of shape (d,) and returns a float. The points passed
        to it are a fresh array every call.
    lo, hi : float or sequence of float
        Lower and upper bounds. A scalar applies to every dimension, a
        sequence has one entry per dimension. Every problem in
        ``gfo.problems`` and ``gfo.cpp_problems`` has default bounds in its
        ``lo`` and ``hi`` attributes.
    d : int, optional
        Number of dimensions. Needed when both bounds are scalars, otherwise
        inferred from the bounds.
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
    vectorized : bool
        Whether a Python objective takes the whole swarm (True) or one point
        (False). Must be True for a C++ problem.
    threads : int
        OpenMP threads for the particle loop, 0 = all cores.
    verbose : bool
        Print a start and end summary and a line every ``verbose_interval``
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
        and the bounds to ``domain.csv`` (read by ``gfo.post``).

    Returns
    -------
    gfo.PSOResult
        ``best_pos`` (shape (d,), physical coordinates), ``best_val``,
        ``loss_history`` (shape (iterations,), global best after every
        iteration), ``evaluations`` and ``diagnostics`` (timings in seconds).

    Raises
    ------
    TypeError
        If ``objective`` is neither callable nor a C++ problem, or a Python
        objective returns something that is not numbers.
    ValueError
        If a setting is invalid, the bounds do not match ``d``, ``lo >= hi``
        in some dimension, or a vectorized objective returns the wrong number
        of values.
    """
    # Check everything cheap before the run starts.
    is_cpp_problem: bool = isinstance(objective, CppProblem)
    if not is_cpp_problem and not callable(objective):
        raise TypeError(
            "objective must be callable or a gfo.cpp_problems problem, "
            f"got {type(objective).__name__}"
        )
    if is_cpp_problem and not vectorized:
        raise ValueError("vectorized=False only applies to Python objectives")

    # lo_array, hi_array: shape=(1,) for a scalar bound, (d,) otherwise.
    lo_array: np.ndarray = np.atleast_1d(np.asarray(lo, dtype=float))
    hi_array: np.ndarray = np.atleast_1d(np.asarray(hi, dtype=float))
    if d is None:
        is_lo_scalar: bool = lo_array.size == 1
        is_hi_scalar: bool = hi_array.size == 1
        if is_lo_scalar and is_hi_scalar:
            raise ValueError("Pass d, it cannot be inferred from scalar bounds")
        dims: int = hi_array.size if is_lo_scalar else lo_array.size
    else:
        dims = validate_integer(d, "d", minimum=1)

    for label, bound in (("lo", lo_array), ("hi", hi_array)):
        is_valid_size: bool = bound.size in (1, dims)
        if bound.ndim != 1 or not is_valid_size:
            raise ValueError(
                f"{label} must be a number or a sequence of d = {dims} numbers, "
                f"got shape {bound.shape}"
            )
    lo_list: list[float] = np.broadcast_to(lo_array, (dims,)).tolist()
    hi_list: list[float] = np.broadcast_to(hi_array, (dims,)).tolist()

    settings = _pso.PSOSettings()
    settings.particles = validate_integer(n_particles, "n_particles", minimum=1)
    settings.iterations = validate_integer(iterations, "iterations", minimum=0)
    settings.seed = validate_integer(seed, "seed", minimum=0)
    settings.threads = validate_integer(threads, "threads", minimum=0)
    settings.verbose_interval = validate_integer(
        verbose_interval, "verbose_interval", minimum=1
    )
    settings.diagnostics_interval = validate_integer(
        diagnostics_interval, "diagnostics_interval", minimum=0
    )
    settings.inertia = float(inertia)
    settings.acceleration1 = float(acceleration1)
    settings.acceleration2 = float(acceleration2)
    settings.max_velocity_rate = float(max_velocity_rate)
    settings.initial_velocity_rate = float(initial_velocity_rate)
    settings.verbose = bool(verbose)
    settings.diagnostics = bool(diagnostics)
    settings.run_dir = "" if run_dir is None else str(Path(run_dir))
    settings.optimizer_dir = str(optimizer_dir)
    settings.save_loss = bool(save_loss)
    settings.save_history = bool(save_history)

    if is_cpp_problem:
        return _pso.optimize_problem(objective, dims, lo_list, hi_list, settings)
    if vectorized:
        return _pso.optimize_python(objective, dims, lo_list, hi_list, settings)

    def evaluate_rows(points: np.ndarray) -> np.ndarray:
        # One Python call per particle, like pymoo's ElementwiseProblem.
        # points: shape=(n, d), values: shape=(n,).
        values: np.ndarray = np.empty(points.shape[0])
        for ii in range(points.shape[0]):
            values[ii] = objective(points[ii])
        return values

    return _pso.optimize_python(evaluate_rows, dims, lo_list, hi_list, settings)
