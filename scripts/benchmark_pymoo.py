"""Benchmark the gfo particle swarm optimizer against pymoo's PSO.

Three backends run the same example problems with the same budget:

- ``pymoo``   pymoo's PSO (pure Python), evaluating the Python example problem.
- ``gfo-py``  gfo's C++ PSO, evaluating the SAME Python example problem.
- ``gfo-cpp`` gfo's C++ PSO, evaluating the C++ twin of the problem (the whole
              run stays in C++, particles evaluated in parallel).

``--mode`` picks how the Python objective is called:

- ``vectorized`` (default): once per iteration with the whole swarm, (n, d).
  pymoo ``Problem`` against gfo's batch objective.
- ``elementwise``: once per particle, (d,). pymoo ``ElementwiseProblem``
  against gfo with ``vectorized=False``. Both loop over the particles in
  Python, so the objective cost is the same for both.

What is kept equal so the comparison is fair:

- Budget: ``n = 20 + 2 d`` particles and ``iterations`` iterations, i.e.
  ``n * iterations`` evaluations for every backend. The evaluation count each
  backend reports, and the points the Python objective really evaluated, are
  checked after every run and mismatches are flagged.
- PSO settings: inertia, both accelerations and the velocity clamp are the
  same. pymoo runs with ``adaptive=False`` and ``pertube_best=False`` and with
  uniform random initial positions like gfo (``--pymoo-lhs`` for pymoo's
  default Latin hypercube sampling).
- Objective: pymoo and gfo-py evaluate the very same Python object through the
  same counting wrapper. gfo-cpp's C++ problem is checked against it on random
  points before the benchmark.
- Timing: backends are interleaved run by run and their order rotates every
  run, every backend gets a warm-up run first, garbage is collected before
  every timed run, and every run of a test uses the same seed for all
  backends.

Remaining differences (inherent to the two implementations): initial
velocities (gfo: uniform in +/- initial_velocity_rate, pymoo: uniform in
[0, V_max)), the first evaluation (gfo moves the swarm once before it,
pymoo evaluates the initial positions), and boundary handling (gfo: clamp and
reflect the velocity, pymoo: resample, then random repair). pymoo is single
threaded, gfo uses ``--threads`` threads (0 = all cores, the default; use
``--threads 1`` for a single core comparison).

Usage
-----
    python scripts/benchmark_pymoo.py --quick
    python scripts/benchmark_pymoo.py
    python scripts/benchmark_pymoo.py --mode elementwise --dims 2 10 --runs 10
    python scripts/benchmark_pymoo.py --only Rastrigin Ackley --threads 1

Results (per run CSV, summary CSV, configuration and figures) are saved to
``benchmark_results/<timestamp>_<mode>/`` unless ``--no-save`` is given.
"""

import argparse
import csv
import gc
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gfo
from gfo.optimizer import (
    DEFAULT_ACCELERATION,
    DEFAULT_INERTIA,
    DEFAULT_MAX_VELOCITY_RATE,
)
from gfo.problems import IProblem

try:
    import pymoo
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.core.problem import ElementwiseProblem, Problem
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
except ImportError as exc:
    raise ImportError(
        'pymoo is not installed. Run: pip install -e ".[benchmark]"'
    ) from exc

# ---------------------------------------------------------------------------
# DEFAULTS (all can be changed from the command line).
ITERATIONS = 2000
NRUNS = 20
TOL = 1e-4  # A run is a success if (best value - known minimum) < TOL.
DIMS = (2, 10, 30)
INERTIA = DEFAULT_INERTIA  # Clerc-Kennedy constriction values.
ACCELERATION1 = DEFAULT_ACCELERATION
ACCELERATION2 = DEFAULT_ACCELERATION
# Fraction of the domain width, the same meaning in both libraries.
MAX_VELOCITY_RATE = DEFAULT_MAX_VELOCITY_RATE
SEED_BASE = 1000  # Run r uses seed SEED_BASE + r for every backend.
BASE_PARTICLES = 20  # n = BASE_PARTICLES + PARTICLES_PER_DIM * d.
PARTICLES_PER_DIM = 2

QUICK_ITERATIONS = 200
QUICK_RUNS = 3
QUICK_DIMS = (2, 10)

WARMUP_ITERATIONS = 20
WARMUP_DIMENSION = 2
CONSISTENCY_POINTS = 256  # Random points used to compare Python and C++ problems.
CONSISTENCY_DIMS = (1, 2, 10, 30)
CONSISTENCY_RTOL = 1e-12  # Allowed relative difference between the two versions.
# Allowed |f(x*) - f_opt|. Schwefel's constant is only exact to ~1e-11 per dim.
OPTIMUM_ATOL = 1e-8
MIN_SECONDS = 1e-12  # Guards divisions by a zero time.
# Floor of the median error on the log accuracy plot, relative to the tolerance,
# so that runs that hit the minimum exactly still show.
ERROR_FLOOR_FRACTION = 1e-6
BAR_GROUP_WIDTH = 0.8  # Fraction of the space between problems used by bars.

LINE_WIDTH = 150
DEFAULT_OUTPUT_ROOT = Path("benchmark_results")
PLOT_DPI = 200


class EBackend(Enum):
    """Optimizer backends (the value is the name used in the output)."""

    PYMOO = "pymoo"
    GFO_PY = "gfo-py"
    GFO_CPP = "gfo-cpp"


class EObjectiveMode(Enum):
    """How the Python objective is called."""

    VECTORIZED = "vectorized"  # Once per iteration with the whole swarm.
    ELEMENTWISE = "elementwise"  # Once per particle.


REFERENCE_BACKEND = EBackend.PYMOO


# ---------------------------------------------------------------------------
# CONFIGURATION AND RECORDS.
@dataclass(slots=True)
class BenchmarkConfig:
    """Everything that defines one benchmark."""

    iterations: int = ITERATIONS
    runs: int = NRUNS
    tol: float = TOL
    dims: tuple[int, ...] = DIMS
    problems: tuple[str, ...] = tuple(gfo.problems.EXAMPLE_PROBLEMS)
    backends: tuple[EBackend, ...] = tuple(EBackend)
    mode: EObjectiveMode = EObjectiveMode.VECTORIZED
    threads: int = 0
    inertia: float = INERTIA
    acceleration1: float = ACCELERATION1
    acceleration2: float = ACCELERATION2
    max_velocity_rate: float = MAX_VELOCITY_RATE
    seed_base: int = SEED_BASE
    pymoo_lhs: bool = False
    time_objective: bool = True


@dataclass(slots=True)
class RunRecord:
    """One run of one backend on one problem."""

    problem: str
    d: int
    n: int
    run: int
    seed: int
    backend: EBackend
    best_val: float
    error: float
    success: bool
    seconds: float
    # NaN when not measured (gfo-cpp, or --no-objective-timing).
    objective_seconds: float
    evaluations: int  # As reported by the backend.
    objective_points: int  # Points the Python objective evaluated (-1: gfo-cpp).
    expected_evaluations: int
    budget_ok: bool


@dataclass(slots=True)
class BackendSummary:
    """All runs of one backend on one (problem, d) test."""

    problem: str
    d: int
    n: int
    backend: EBackend
    runs: int
    successes: int
    mean_error: float
    median_error: float
    best_error: float
    worst_error: float
    total_seconds: float
    mean_seconds: float
    objective_seconds: float
    overhead_seconds: float
    budget_ok: bool
    # Reference total time / this backend's total time.
    speedup_total: float = math.nan
    # The same for the time outside the objective (Python objectives only).
    speedup_overhead: float = math.nan


@dataclass(slots=True)
class BenchmarkCase:
    """One (problem, d) combination with both versions of the problem."""

    name: str
    d: int
    n: int
    py_problem: IProblem
    cpp_problem: gfo.CppProblem
    lo: float
    hi: float
    f_opt: float
    optimum_check: float = math.nan
    records: list[RunRecord] = field(default_factory=list)


@dataclass(slots=True)
class BackendRun:
    """What one backend run returns."""

    best_val: float
    evaluations: int  # As reported by the backend.
    objective_points: int  # Points the Python objective evaluated (-1: gfo-cpp).
    seconds: float
    objective_seconds: float  # NaN when not measured.


# ---------------------------------------------------------------------------
# OBJECTIVE WRAPPER.
class CountedObjective:
    """Wraps a Python problem to count the points it evaluates and time it.

    The same wrapper is used for pymoo and gfo-py, so any overhead it adds is
    the same for both.
    """

    __slots__ = ("problem", "time_calls", "points", "seconds")

    def __init__(self, problem: IProblem, *, time_calls: bool) -> None:
        self.problem = problem
        self.time_calls = time_calls
        self.points = 0
        self.seconds = 0.0

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate points of shape (n, d), returns shape (n,)."""
        self.points += points.shape[0]
        if not self.time_calls:
            return self.problem.evaluate(points)
        start: float = time.perf_counter()
        values: np.ndarray = self.problem.evaluate(points)
        self.seconds += time.perf_counter() - start
        return values

    def evaluate_point(self, point: np.ndarray) -> float:
        """Evaluate one point of shape (d,)."""
        return float(self.evaluate_batch(point[np.newaxis, :])[0])

    def calc_measured_seconds(self) -> float:
        """Calculate the objective time, NaN if timing is off."""
        if self.time_calls:
            return self.seconds
        return math.nan


# ---------------------------------------------------------------------------
# PYMOO PROBLEMS. pymoo needs subclasses of its problem classes, the only
# inheritance here (required by pymoo's API).
class PymooVectorizedProblem(Problem):
    """pymoo problem that evaluates the whole population in one call."""

    def __init__(self, counted: CountedObjective, d: int, lo: float, hi: float) -> None:
        super().__init__(n_var=d, n_obj=1, xl=np.full(d, lo), xu=np.full(d, hi))
        self.counted = counted

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, np.ndarray],
        *args: object,
        **kwargs: object,
    ) -> None:
        out["F"] = self.counted.evaluate_batch(x)


class PymooElementwiseProblem(ElementwiseProblem):
    """pymoo problem that evaluates one point per call."""

    def __init__(self, counted: CountedObjective, d: int, lo: float, hi: float) -> None:
        super().__init__(n_var=d, n_obj=1, xl=np.full(d, lo), xu=np.full(d, hi))
        self.counted = counted

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, float],
        *args: object,
        **kwargs: object,
    ) -> None:
        out["F"] = self.counted.evaluate_point(x)


# ---------------------------------------------------------------------------
# BACKENDS.
def calc_swarm_size(d: int) -> int:
    """Calculate the swarm size for dimension ``d`` (grows with the dimension)."""
    return BASE_PARTICLES + PARTICLES_PER_DIM * d


def make_case(name: str, d: int) -> BenchmarkCase:
    """Build both versions of a problem for dimension ``d``."""
    py_problem: IProblem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    cpp_problem: gfo.CppProblem = getattr(gfo.cpp_problems, name)()
    optimum_value = float(py_problem.evaluate(py_problem.calc_x_opt(d)[np.newaxis])[0])
    return BenchmarkCase(
        name=name,
        d=d,
        n=calc_swarm_size(d),
        py_problem=py_problem,
        cpp_problem=cpp_problem,
        lo=py_problem.lo,
        hi=py_problem.hi,
        f_opt=py_problem.f_opt,
        optimum_check=optimum_value - py_problem.f_opt,
    )


def run_backend(
    config: BenchmarkConfig,
    case: BenchmarkCase,
    backend: EBackend,
    iterations: int,
    seed: int,
) -> BackendRun:
    """Run one backend once and time it."""
    is_vectorized: bool = config.mode is EObjectiveMode.VECTORIZED
    counted = CountedObjective(case.py_problem, time_calls=config.time_objective)

    gc.collect()
    start: float = time.perf_counter()
    if backend is EBackend.PYMOO:
        if is_vectorized:
            problem = PymooVectorizedProblem(counted, case.d, case.lo, case.hi)
        else:
            problem = PymooElementwiseProblem(counted, case.d, case.lo, case.hi)
        sampling = LHS() if config.pymoo_lhs else FloatRandomSampling()
        algorithm = PSO(
            pop_size=case.n,
            sampling=sampling,
            w=config.inertia,
            c1=config.acceleration1,
            c2=config.acceleration2,
            max_velocity_rate=config.max_velocity_rate,
            adaptive=False,
            pertube_best=False,
        )
        result = minimize(
            problem,
            algorithm,
            termination=("n_gen", iterations),
            seed=seed,
            verbose=False,
        )
        best = float(np.atleast_1d(result.F)[0])
        evaluations = int(result.algorithm.evaluator.n_eval)
    else:
        if backend is EBackend.GFO_CPP:
            # C++ problems are evaluated in C++, one particle per thread task.
            objective = case.cpp_problem
            uses_vectorized = True
        elif is_vectorized:
            objective = counted.evaluate_batch
            uses_vectorized = True
        else:
            objective = counted.evaluate_point
            uses_vectorized = False
        gfo_result = gfo.pso(
            objective,
            case.lo,
            case.hi,
            d=case.d,
            n_particles=case.n,
            iterations=iterations,
            inertia=config.inertia,
            acceleration1=config.acceleration1,
            acceleration2=config.acceleration2,
            max_velocity_rate=config.max_velocity_rate,
            seed=seed,
            vectorized=uses_vectorized,
            threads=config.threads,
        )
        best = float(gfo_result.best_val)
        evaluations = int(gfo_result.evaluations)
    seconds: float = time.perf_counter() - start

    if backend is EBackend.GFO_CPP:
        return BackendRun(best, evaluations, -1, seconds, math.nan)
    return BackendRun(
        best, evaluations, counted.points, seconds, counted.calc_measured_seconds()
    )


# ---------------------------------------------------------------------------
# CHECKS.
def check_problem_consistency(config: BenchmarkConfig) -> bool:
    """Compare every Python problem with its C++ twin on random points.

    Returns
    -------
    bool
        True if every pair agrees, has the same bounds and both hit their
        known minimum.
    """
    rng = np.random.default_rng(config.seed_base)
    all_ok = True
    print("Checking that the Python and C++ example problems agree...")
    for name in config.problems:
        py_problem: IProblem = gfo.problems.EXAMPLE_PROBLEMS[name]()
        cpp_problem: gfo.CppProblem = getattr(gfo.cpp_problems, name)()
        worst = 0.0
        optimum_ok = True
        for d in CONSISTENCY_DIMS:
            # points: shape=(CONSISTENCY_POINTS, d).
            points: np.ndarray = rng.uniform(
                py_problem.lo, py_problem.hi, size=(CONSISTENCY_POINTS, d)
            )
            py_values: np.ndarray = py_problem.evaluate(points)
            cpp_values: np.ndarray = cpp_problem.evaluate(points)
            scale: np.ndarray = np.maximum(1.0, np.abs(cpp_values))
            difference = float(np.max(np.abs(py_values - cpp_values) / scale))
            worst = max(worst, difference)

            for problem in (py_problem, cpp_problem):
                optimum: np.ndarray = problem.calc_x_opt(d)[np.newaxis, :]
                optimum_error = abs(problem.evaluate(optimum)[0] - problem.f_opt)
                optimum_ok = optimum_ok and optimum_error < OPTIMUM_ATOL

        py_bounds = (py_problem.lo, py_problem.hi)
        cpp_bounds = (cpp_problem.lo, cpp_problem.hi)
        same_bounds: bool = py_bounds == cpp_bounds
        ok: bool = worst <= CONSISTENCY_RTOL and optimum_ok and same_bounds
        all_ok = all_ok and ok

        optimum_text = "ok" if optimum_ok else "FAILED"
        bounds_text = "same" if same_bounds else "DIFFERENT"
        status_text = "ok" if ok else "MISMATCH"
        print(
            f"  {name:<11} max relative difference = {worst:.1e} | optimum check "
            f"{optimum_text} | bounds {bounds_text} | {status_text}"
        )
    return all_ok


# ---------------------------------------------------------------------------
# BENCHMARK.
def run_case(config: BenchmarkConfig, case: BenchmarkCase, *, verbose: bool) -> None:
    """Run every backend on one test case, interleaved, into ``case.records``."""
    expected: int = case.n * config.iterations
    print(
        f"\n--- {case.name} (d = {case.d}, n = {case.n}, "
        f"budget = {expected} evaluations) ---"
    )
    for run in range(config.runs):
        seed: int = config.seed_base + run
        # Rotate the backend order so no backend always runs first.
        shift: int = run % len(config.backends)
        order = config.backends[shift:] + config.backends[:shift]
        for backend in order:
            result: BackendRun = run_backend(
                config, case, backend, config.iterations, seed
            )
            error: float = result.best_val - case.f_opt
            # The backend's own count and (for Python objectives) the points the
            # objective really evaluated must both equal the budget.
            points_ok: bool = result.objective_points in (-1, expected)
            budget_ok: bool = result.evaluations == expected and points_ok
            record = RunRecord(
                problem=case.name,
                d=case.d,
                n=case.n,
                run=run,
                seed=seed,
                backend=backend,
                best_val=result.best_val,
                error=error,
                success=bool(error < config.tol),
                seconds=result.seconds,
                objective_seconds=result.objective_seconds,
                evaluations=result.evaluations,
                objective_points=result.objective_points,
                expected_evaluations=expected,
                budget_ok=budget_ok,
            )
            case.records.append(record)
            if not verbose and budget_ok:
                continue

            objective_text = ""
            if not math.isnan(record.objective_seconds):
                objective_text = f" (objective {record.objective_seconds:7.3f}s)"
            budget_text = ""
            if not budget_ok:
                budget_text = (
                    f"  !! evaluations reported {record.evaluations}, objective "
                    f"saw {record.objective_points}, expected {expected}"
                )
            print(
                f"  [run {run + 1:2d}/{config.runs}] {backend.value:<8}"
                f"| best = {record.best_val:11.4e} | error = {error:10.2e} "
                f"| time = {record.seconds:8.3f}s{objective_text}{budget_text}"
            )


def summarize_case(
    config: BenchmarkConfig, case: BenchmarkCase
) -> list[BackendSummary]:
    """Summarize every backend on one test case, with speedups against pymoo."""
    summaries: dict[EBackend, BackendSummary] = {}
    for backend in config.backends:
        records: list[RunRecord] = []
        for record in case.records:
            if record.backend is backend:
                records.append(record)
        # errors, seconds, objective: shape=(runs,).
        errors = np.array([record.error for record in records])
        seconds = np.array([record.seconds for record in records])
        objective = np.array([record.objective_seconds for record in records])
        objective_total = float(objective.sum())  # NaN if not measured.
        total_seconds = float(seconds.sum())
        summaries[backend] = BackendSummary(
            problem=case.name,
            d=case.d,
            n=case.n,
            backend=backend,
            runs=len(records),
            successes=int(np.sum(errors < config.tol)),
            mean_error=float(errors.mean()),
            median_error=float(np.median(errors)),
            best_error=float(errors.min()),
            worst_error=float(errors.max()),
            total_seconds=total_seconds,
            mean_seconds=float(seconds.mean()),
            objective_seconds=objective_total,
            overhead_seconds=total_seconds - objective_total,
            budget_ok=all(record.budget_ok for record in records),
        )

    reference: BackendSummary | None = summaries.get(REFERENCE_BACKEND)
    if reference is not None:
        for summary in summaries.values():
            total = max(summary.total_seconds, MIN_SECONDS)
            overhead = max(summary.overhead_seconds, MIN_SECONDS)
            summary.speedup_total = reference.total_seconds / total
            summary.speedup_overhead = reference.overhead_seconds / overhead
    return list(summaries.values())


# ---------------------------------------------------------------------------
# OUTPUT.
def write_csv(path: Path, rows: list[RunRecord] | list[BackendSummary]) -> None:
    """Write dataclass records to CSV, enums as their values."""
    dicts: list[dict[str, object]] = []
    for row in rows:
        values: dict[str, object] = asdict(row)
        values["backend"] = row.backend.value
        dicts.append(values)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dicts[0]))
        writer.writeheader()
        writer.writerows(dicts)


def plot_results(
    config: BenchmarkConfig, summaries: list[BackendSummary]
) -> list[tuple[plt.Figure, str]]:
    """Plot speedup and accuracy figures, one row per dimension.

    Returns
    -------
    list of (Figure, str)
        Every figure with its file name.
    """
    contenders: list[EBackend] = []
    for backend in config.backends:
        if backend is not REFERENCE_BACKEND:
            contenders.append(backend)
    has_reference: bool = REFERENCE_BACKEND in config.backends
    lookup: dict[tuple[str, int, EBackend], BackendSummary] = {}
    for summary in summaries:
        lookup[(summary.problem, summary.d, summary.backend)] = summary
    names: list[str] = list(config.problems)
    positions: np.ndarray = np.arange(len(names))
    width: float = BAR_GROUP_WIDTH / max(len(config.backends), 1)
    n_rows: int = len(config.dims)
    figure_size = (11, 4.2 * n_rows)

    figures: list[tuple[plt.Figure, str]] = []
    if has_reference and contenders:
        fig, axes = plt.subplots(
            n_rows, 1, figsize=figure_size, squeeze=False, layout="constrained"
        )
        for row, d in enumerate(config.dims):
            ax = axes[row, 0]
            for kk, backend in enumerate(contenders):
                speedups = [lookup[(name, d, backend)].speedup_total for name in names]
                offset: float = (kk - (len(contenders) - 1) / 2) * width
                bars = ax.bar(positions + offset, speedups, width, label=backend.value)
                ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Parity")
            ax.set_yscale("log")
            ax.set_xticks(positions, names, rotation=20)
            ax.set_ylabel("Speedup vs pymoo (total time)")
            ax.set_title(
                f"d = {d}  |  {config.mode.value} objective  |  "
                f"{config.iterations} iterations x {config.runs} runs"
            )
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            ax.legend(fontsize=8)
        figures.append((fig, "speedup.png"))

    error_floor: float = config.tol * ERROR_FLOOR_FRACTION
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figure_size, squeeze=False, layout="constrained"
    )
    for row, d in enumerate(config.dims):
        ax = axes[row, 0]
        for kk, backend in enumerate(config.backends):
            errors: list[float] = []
            for name in names:
                errors.append(max(lookup[(name, d, backend)].median_error, error_floor))
            offset = (kk - (len(config.backends) - 1) / 2) * width
            ax.bar(positions + offset, errors, width, label=backend.value)
        ax.axhline(
            config.tol,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Success tol ({config.tol:g})",
        )
        ax.set_yscale("log")
        ax.set_xticks(positions, names, rotation=20)
        ax.set_ylabel("Median error (best - f*)")
        ax.set_title(f"Accuracy, d = {d}")
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)
    figures.append((fig, "accuracy.png"))
    return figures


# ---------------------------------------------------------------------------
# COMMAND LINE.
def main(argv: list[str] | None = None) -> int:
    """Run the benchmark. Returns 1 if any check failed or any budget mismatched."""
    problem_names: list[str] = list(gfo.problems.EXAMPLE_PROBLEMS)
    backend_names: list[str] = [backend.value for backend in EBackend]
    parser = argparse.ArgumentParser(
        description="Benchmark gfo's PSO against pymoo's PSO."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke test: {QUICK_ITERATIONS} iterations, {QUICK_RUNS} runs, "
        f"d = {QUICK_DIMS}.",
    )
    parser.add_argument(
        "--iterations", type=int, help=f"Iterations per run (default {ITERATIONS})."
    )
    parser.add_argument("--runs", type=int, help=f"Runs per test (default {NRUNS}).")
    parser.add_argument(
        "--dims", type=int, nargs="+", help=f"Dimensions (default {DIMS})."
    )
    parser.add_argument(
        "--only", nargs="+", choices=problem_names, help="Only these problems."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=backend_names,
        default=backend_names,
        help="Backends to run (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in EObjectiveMode],
        default=EObjectiveMode.VECTORIZED.value,
        help="How the Python objective is called (default: vectorized).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="gfo OpenMP threads, 0 = all cores (default). pymoo uses one.",
    )
    parser.add_argument(
        "--tol", type=float, default=TOL, help=f"Success tolerance (default {TOL})."
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE,
        help=f"Seed of run 0 (default {SEED_BASE}).",
    )
    parser.add_argument(
        "--pymoo-lhs",
        action="store_true",
        help="Use pymoo's default LHS initial sampling instead of uniform random.",
    )
    parser.add_argument(
        "--no-objective-timing",
        action="store_true",
        help="Do not time the Python objective (removes the timer overhead).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Results folder (default: {DEFAULT_OUTPUT_ROOT}/<timestamp>_<mode>).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results or figures."
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not make figures.")
    parser.add_argument(
        "--show", action="store_true", help="Show the figures interactively."
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="No per run lines.")
    args = parser.parse_args(argv)

    # Configuration: defaults, then --quick, then explicit values.
    backends: list[EBackend] = []
    for name in args.backends:
        if EBackend(name) not in backends:
            backends.append(EBackend(name))
    config = BenchmarkConfig(
        mode=EObjectiveMode(args.mode),
        threads=args.threads,
        tol=args.tol,
        seed_base=args.seed_base,
        pymoo_lhs=args.pymoo_lhs,
        time_objective=not args.no_objective_timing,
        backends=tuple(backends),
    )
    if args.quick:
        config.iterations = QUICK_ITERATIONS
        config.runs = QUICK_RUNS
        config.dims = QUICK_DIMS
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.runs is not None:
        config.runs = args.runs
    if args.dims:
        config.dims = tuple(args.dims)
    if args.only:
        config.problems = tuple(args.only)

    is_invalid: bool = (
        config.iterations < 1
        or config.runs < 1
        or config.threads < 0
        or min(config.dims) < 1
    )
    if is_invalid:
        parser.error("iterations, runs and dims must be >= 1 and threads >= 0")

    environment: dict[str, object] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pymoo": pymoo.__version__,
        "gfo": gfo.__version__,
        "gfo_openmp": gfo.OPENMP,
        "gfo_max_threads": gfo.MAX_THREADS,
        "gfo_threads_used": config.threads or gfo.MAX_THREADS,
    }

    backend_list: str = " vs ".join(backend.value for backend in config.backends)
    sampling_text: str = "LHS" if config.pymoo_lhs else "uniform random"
    openmp_text: str = "on" if gfo.OPENMP else "OFF"
    print("-" * LINE_WIDTH)
    print(
        f"PSO BENCHMARK: {backend_list} ({config.mode.value} objective, "
        "interleaved runs)"
    )
    print("-" * LINE_WIDTH)
    print(
        f"Iterations = {config.iterations} | Runs per test = {config.runs} "
        f"| Success tolerance = {config.tol} | Dimensions = {list(config.dims)} "
        f"| Particles = {BASE_PARTICLES} + {PARTICLES_PER_DIM} d"
    )
    print(
        f"Inertia = {config.inertia} | Acceleration 1 = {config.acceleration1} "
        f"| Acceleration 2 = {config.acceleration2} | Max velocity rate = "
        f"{config.max_velocity_rate} | pymoo sampling = {sampling_text}, "
        "adaptive off, pertube_best off"
    )
    print(
        f"pymoo {pymoo.__version__} (1 thread) | gfo {gfo.__version__} "
        f"({environment['gfo_threads_used']} of {gfo.MAX_THREADS} threads, "
        f"OpenMP {openmp_text}) | Python {environment['python']} "
        f"| numpy {np.__version__}"
    )
    print("-" * LINE_WIDTH)

    consistent: bool = check_problem_consistency(config)
    if not consistent:
        print("WARNING: the Python and C++ problems disagree, gfo-cpp results are")
        print("NOT comparable.")

    # Warm up every backend on a tiny problem so imports and caches are not timed.
    print("Warming up every backend...")
    warmup_case: BenchmarkCase = make_case(config.problems[0], WARMUP_DIMENSION)
    for backend in config.backends:
        run_backend(config, warmup_case, backend, WARMUP_ITERATIONS, config.seed_base)

    start: float = time.perf_counter()
    records: list[RunRecord] = []
    summaries: list[BackendSummary] = []
    for d in config.dims:
        for name in config.problems:
            case: BenchmarkCase = make_case(name, d)
            run_case(config, case, verbose=not args.quiet)
            case_summaries: list[BackendSummary] = summarize_case(config, case)
            records.extend(case.records)
            summaries.extend(case_summaries)

            for summary in case_summaries:
                is_contender: bool = summary.backend is not REFERENCE_BACKEND
                speedup_text = ""
                if is_contender and not math.isnan(summary.speedup_total):
                    speedup_text = f" | speedup x{summary.speedup_total:6.1f}"
                    if not math.isnan(summary.speedup_overhead):
                        speedup_text += (
                            f" (overhead x{summary.speedup_overhead:6.1f})"
                        )
                objective_text = ""
                if not math.isnan(summary.objective_seconds):
                    objective_text = f" (objective {summary.objective_seconds:7.3f}s)"
                budget_text = "" if summary.budget_ok else "  !! BUDGET MISMATCH"
                print(
                    f"{case.name:<11}| d = {case.d:2d} | n = {case.n:3d} "
                    f"| f(x*) check = {case.optimum_check:8.1e} "
                    f"| {summary.backend.value:<8}"
                    f"| success = {summary.successes:2d}/{config.runs} "
                    f"| median err = {summary.median_error:9.2e} "
                    f"| worst err = {summary.worst_error:9.2e} "
                    f"| time = {summary.total_seconds:8.3f}s"
                    f"{objective_text}{speedup_text}{budget_text}"
                )

    # Overall speedups per dimension: geometric mean over the problems.
    if REFERENCE_BACKEND in config.backends:
        print("-" * LINE_WIDTH)
        print("OVERALL SPEEDUP VS PYMOO (geometric mean over problems, total time)")
        for d in config.dims:
            parts: list[str] = []
            for backend in config.backends:
                if backend is REFERENCE_BACKEND:
                    continue
                speedups: list[float] = []
                for summary in summaries:
                    if summary.d == d and summary.backend is backend:
                        speedups.append(summary.speedup_total)
                geometric_mean: float = math.exp(np.mean(np.log(speedups)))
                parts.append(f"{backend.value}: x{geometric_mean:.1f}")
            print(f"  d = {d:2d} | " + " | ".join(parts))

    budgets_ok: bool = all(record.budget_ok for record in records)
    budget_text = "all equal" if budgets_ok else "MISMATCH (see !! lines)"
    check_text = "passed" if consistent else "FAILED"
    print("-" * LINE_WIDTH)
    print("BENCHMARK FINISHED.")
    print(
        f"Total time: {time.perf_counter() - start:.2f}s | Evaluation budgets: "
        f"{budget_text} | Problem check: {check_text}"
    )

    output_dir: Path | None = None
    if not args.no_save:
        stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_ROOT / f"{stamp}_{config.mode.value}"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "runs.csv", records)
        write_csv(output_dir / "summary.csv", summaries)
        config_values: dict[str, object] = asdict(config)
        config_values["backends"] = [backend.value for backend in config.backends]
        config_values["mode"] = config.mode.value
        metadata = {"config": config_values, "environment": environment}
        (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
        print(f"Saved results to: {output_dir.resolve()}")

    wants_plots: bool = output_dir is not None or args.show
    if not args.no_plots and wants_plots:
        if not args.show:
            # Figures are only saved: no GUI backend (works without a display).
            plt.switch_backend("Agg")
        figures = plot_results(config, summaries)
        if output_dir is not None:
            for fig, filename in figures:
                fig.savefig(output_dir / filename, dpi=PLOT_DPI)
                print(f"Saved figure: {(output_dir / filename).resolve()}")
        if args.show:
            plt.show()
        plt.close("all")

    return 0 if budgets_ok and consistent else 1


if __name__ == "__main__":
    sys.exit(main())
