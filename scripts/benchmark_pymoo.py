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
  backend reports is checked after every run and mismatches are flagged.
- PSO settings: inertia, both accelerations and the velocity clamp are the
  same. pymoo runs with ``adaptive=False`` and ``pertube_best=False`` and with
  uniform random initial positions like gfo (``--pymoo-lhs`` for pymoo's
  default Latin hypercube sampling).
- Objective: pymoo and gfo-py evaluate the very same Python object. gfo-cpp's
  C++ problem is checked against it on random points before the benchmark.
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
``benchmark_results/<timestamp>/`` unless ``--no-save`` is given.
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
from pathlib import Path
from typing import Callable

import numpy as np

import gfo

try:
    import pymoo
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.core.problem import ElementwiseProblem, Problem
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    PYMOO_IMPORT_ERROR = None
except ImportError as exc:
    PYMOO_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# DEFAULTS (all can be changed from the command line).
ITERATIONS = 2000
NRUNS = 20
TOL = 1e-4                  # A run is a success if (best value - known minimum) < TOL.
DIMS = (2, 10, 30)
INERTIA = 0.729             # Clerc-Kennedy constriction values.
ACCELERATION1 = 1.49445
ACCELERATION2 = 1.49445
MAX_VELOCITY_RATE = 0.15    # Fraction of the domain width, same meaning in both libraries.
SEED_BASE = 1000            # Run r uses seed SEED_BASE + r for every backend.
BASE_PARTICLES = 20         # n = BASE_PARTICLES + PARTICLES_PER_DIM * d.
PARTICLES_PER_DIM = 2

QUICK_ITERATIONS = 200
QUICK_RUNS = 3
QUICK_DIMS = (2, 10)

WARMUP_ITERATIONS = 20
WARMUP_DIMENSION = 2
CONSISTENCY_POINTS = 256    # Random points used to compare the Python and C++ problems.
CONSISTENCY_DIMS = (1, 2, 10, 30)
CONSISTENCY_RTOL = 1e-12    # Allowed relative difference between the two versions.
OPTIMUM_ATOL = 1e-8         # Allowed |f(x*) - f_opt| (Schwefel's constant is only exact to ~1e-11 per dim).
MIN_SECONDS = 1e-12         # Guards divisions by a zero time.

BACKENDS = ("pymoo", "gfo-py", "gfo-cpp")
REFERENCE_BACKEND = "pymoo"
MODES = ("vectorized", "elementwise")
LINE_WIDTH = 150
DEFAULT_OUTPUT_ROOT = Path("benchmark_results")
PLOT_DPI = 200


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
    backends: tuple[str, ...] = BACKENDS
    mode: str = "vectorized"
    threads: int = 0
    inertia: float = INERTIA
    acceleration1: float = ACCELERATION1
    acceleration2: float = ACCELERATION2
    max_velocity_rate: float = MAX_VELOCITY_RATE
    seed_base: int = SEED_BASE
    pymoo_lhs: bool = False
    time_objective: bool = True

    def particles(self, d: int) -> int:
        """Swarm size for dimension ``d`` (grows with the dimension)."""
        return BASE_PARTICLES + PARTICLES_PER_DIM * d


@dataclass(slots=True)
class RunRecord:
    """One run of one backend on one problem."""

    problem: str
    d: int
    n: int
    run: int
    seed: int
    backend: str
    best_val: float
    error: float
    success: bool
    seconds: float
    objective_seconds: float    # NaN when not measured (gfo-cpp, or --no-objective-timing).
    evaluations: int            # As reported by the backend.
    objective_points: int       # Points the Python objective evaluated (-1 for gfo-cpp).
    expected_evaluations: int
    budget_ok: bool


@dataclass(slots=True)
class BackendSummary:
    """All runs of one backend on one (problem, d) test."""

    problem: str
    d: int
    n: int
    backend: str
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
    speedup_total: float = math.nan       # Reference total time / this backend's total time.
    speedup_overhead: float = math.nan    # Same, for the time outside the objective (Python objectives only).


@dataclass(slots=True)
class BenchmarkCase:
    """One (problem, d) combination with both versions of the problem."""

    name: str
    d: int
    n: int
    py_problem: object
    cpp_problem: object
    lo: float
    hi: float
    f_opt: float
    optimum_check: float = math.nan
    records: list[RunRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OBJECTIVE WRAPPER.
class CountedObjective:
    """Wraps a Python objective to count the points it evaluates and time it.

    The same wrapper is used for pymoo and gfo-py, so any overhead it adds is
    the same for both.
    """

    __slots__ = ("function", "time_calls", "points", "seconds")

    def __init__(self, function: Callable, time_calls: bool) -> None:
        self.function = function
        self.time_calls = time_calls
        self.points = 0
        self.seconds = 0.0

    def __call__(self, x):
        self.points += 1 if np.ndim(x) == 1 else len(x)
        if not self.time_calls:
            return self.function(x)
        start = time.perf_counter()
        values = self.function(x)
        self.seconds += time.perf_counter() - start
        return values

    def measured_seconds(self) -> float:
        """Objective time, NaN if timing is off."""
        return self.seconds if self.time_calls else math.nan


# ---------------------------------------------------------------------------
# BACKENDS. Each returns (best value, evaluations reported by the backend).
# pymoo needs subclasses of its problem classes, the only inheritance here.
class PymooVectorizedProblem(Problem):
    """pymoo problem that evaluates the whole population in one call."""

    def __init__(self, objective: Callable, d: int, lo: float, hi: float) -> None:
        super().__init__(n_var=d, n_obj=1, xl=np.full(d, lo), xu=np.full(d, hi))
        self.objective = objective

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective(x)


class PymooElementwiseProblem(ElementwiseProblem):
    """pymoo problem that evaluates one point per call."""

    def __init__(self, objective: Callable, d: int, lo: float, hi: float) -> None:
        super().__init__(n_var=d, n_obj=1, xl=np.full(d, lo), xu=np.full(d, hi))
        self.objective = objective

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective(x)


def run_pymoo(config: BenchmarkConfig, case: BenchmarkCase, objective: Callable,
              iterations: int, seed: int) -> tuple[float, int]:
    """One pymoo PSO run."""
    problem_class = PymooVectorizedProblem if config.mode == "vectorized" else PymooElementwiseProblem
    problem = problem_class(objective, case.d, case.lo, case.hi)
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
    result = minimize(problem, algorithm, termination=("n_gen", iterations),
                      seed=seed, verbose=False)
    best = float(np.atleast_1d(result.F)[0])
    return best, int(result.algorithm.evaluator.n_eval)


def run_gfo(config: BenchmarkConfig, case: BenchmarkCase, objective, iterations: int,
            seed: int) -> tuple[float, int]:
    """One gfo PSO run (Python callable or C++ problem)."""
    result = gfo.pso(
        objective,
        d=case.d,
        lo=case.lo,
        hi=case.hi,
        n_particles=case.n,
        iterations=iterations,
        inertia=config.inertia,
        acceleration1=config.acceleration1,
        acceleration2=config.acceleration2,
        max_velocity_rate=config.max_velocity_rate,
        seed=seed,
        vectorized=(config.mode == "vectorized"),
        threads=config.threads,
    )
    return float(result.best_val), int(result.evaluations)


def run_backend(config: BenchmarkConfig, case: BenchmarkCase, backend: str,
                iterations: int, seed: int) -> tuple[float, int, int, float, float]:
    """Run one backend once.

    Returns
    -------
    (float, int, int, float, float)
        Best value, evaluations reported by the backend, points the Python
        objective actually evaluated (-1 for gfo-cpp, which has no Python
        objective), wall time and objective time (NaN if not measured).
    """
    if backend == "gfo-cpp":
        objective = case.cpp_problem
        counted = None
    else:
        counted = CountedObjective(case.py_problem, config.time_objective)
        objective = counted

    gc.collect()
    start = time.perf_counter()
    if backend == "pymoo":
        best, evaluations = run_pymoo(config, case, objective, iterations, seed)
    else:
        best, evaluations = run_gfo(config, case, objective, iterations, seed)
    seconds = time.perf_counter() - start

    objective_seconds = math.nan if counted is None else counted.measured_seconds()
    points = -1 if counted is None else counted.points
    return best, evaluations, points, seconds, objective_seconds


# ---------------------------------------------------------------------------
# CHECKS.
def check_problem_consistency(config: BenchmarkConfig) -> bool:
    """Compare every Python problem with its C++ twin on random points.

    Returns
    -------
    bool
        True if every pair agrees and both hit their known minimum.
    """
    rng = np.random.default_rng(config.seed_base)
    all_ok = True
    print("Checking that the Python and C++ example problems agree...")
    for name in config.problems:
        py_problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
        cpp_problem = getattr(gfo.cpp_problems, name)()
        worst = 0.0
        optimum_ok = True
        for d in CONSISTENCY_DIMS:
            points = rng.uniform(py_problem.lo, py_problem.hi, size=(CONSISTENCY_POINTS, d))
            py_values = py_problem(points)
            cpp_values = cpp_problem(points)
            scale = np.maximum(1.0, np.abs(cpp_values))
            worst = max(worst, float(np.max(np.abs(py_values - cpp_values) / scale)))
            optimum_ok &= abs(py_problem(py_problem.x_opt(d)) - py_problem.f_opt()) < OPTIMUM_ATOL
            optimum_ok &= abs(cpp_problem(cpp_problem.x_opt(d)) - cpp_problem.f_opt()) < OPTIMUM_ATOL

        same_setup = (py_problem.lo, py_problem.hi) == (cpp_problem.lo, cpp_problem.hi)
        ok = worst <= CONSISTENCY_RTOL and optimum_ok and same_setup
        all_ok &= ok
        status = "ok" if ok else "MISMATCH"
        print(f"  {name:<11} max relative difference = {worst:.1e} | optimum check "
              f"{'ok' if optimum_ok else 'FAILED'} | bounds {'same' if same_setup else 'DIFFERENT'} | {status}")
    return all_ok


def warm_up(config: BenchmarkConfig) -> None:
    """Run every backend once on a tiny problem so imports and caches are not timed."""
    name = config.problems[0]
    case = make_case(config, name, WARMUP_DIMENSION)
    for backend in config.backends:
        run_backend(config, case, backend, WARMUP_ITERATIONS, config.seed_base)


# ---------------------------------------------------------------------------
# BENCHMARK.
def make_case(config: BenchmarkConfig, name: str, d: int) -> BenchmarkCase:
    """Build both versions of a problem for dimension ``d``."""
    py_problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    cpp_problem = getattr(gfo.cpp_problems, name)()
    case = BenchmarkCase(name=name, d=d, n=config.particles(d), py_problem=py_problem,
                    cpp_problem=cpp_problem, lo=py_problem.lo, hi=py_problem.hi,
                    f_opt=py_problem.f_opt())
    case.optimum_check = float(py_problem(py_problem.x_opt(d)) - case.f_opt)
    return case


def rotated(backends: tuple[str, ...], run: int) -> list[str]:
    """Backend order for one run, rotated so no backend always runs first."""
    shift = run % len(backends)
    return list(backends[shift:] + backends[:shift])


def run_case(config: BenchmarkConfig, case: BenchmarkCase, verbose: bool) -> None:
    """All runs of all backends on one test case, interleaved."""
    expected = case.n * config.iterations
    print(f"\n--- {case.name} (d = {case.d}, n = {case.n}, budget = {expected} evaluations) ---")
    for run in range(config.runs):
        seed = config.seed_base + run
        for backend in rotated(config.backends, run):
            best, evaluations, points, seconds, objective_seconds = run_backend(
                config, case, backend, config.iterations, seed)
            error = best - case.f_opt
            # The backend's own count and (for Python objectives) the points the
            # objective really evaluated must both equal the budget.
            points_ok = points in (-1, expected)
            record = RunRecord(
                problem=case.name, d=case.d, n=case.n, run=run, seed=seed,
                backend=backend, best_val=best, error=error,
                success=bool(error < config.tol), seconds=seconds,
                objective_seconds=objective_seconds, evaluations=evaluations,
                objective_points=points, expected_evaluations=expected,
                budget_ok=(evaluations == expected and points_ok))
            case.records.append(record)
            if verbose or not record.budget_ok:
                print_run(record, config.runs)


def print_run(record: RunRecord, runs: int) -> None:
    """One console line per run."""
    objective = "" if math.isnan(record.objective_seconds) else f" (objective {record.objective_seconds:7.3f}s)"
    budget = "" if record.budget_ok else (f"  !! evaluations reported {record.evaluations}, "
                                          f"objective saw {record.objective_points}, "
                                          f"expected {record.expected_evaluations}")
    print(f"  [run {record.run + 1:2d}/{runs}] {record.backend:<8}| best = {record.best_val:11.4e} "
          f"| error = {record.error:10.2e} | time = {record.seconds:8.3f}s{objective}{budget}")


def summarize_case(config: BenchmarkConfig, case: BenchmarkCase) -> list[BackendSummary]:
    """Summary of every backend on one test case, with speedups against pymoo."""
    summaries = {}
    for backend in config.backends:
        records = [record for record in case.records if record.backend == backend]
        errors = np.array([record.error for record in records])
        seconds = np.array([record.seconds for record in records])
        objective = np.array([record.objective_seconds for record in records])
        objective_total = float(objective.sum())  # NaN if not measured
        summaries[backend] = BackendSummary(
            problem=case.name, d=case.d, n=case.n, backend=backend, runs=len(records),
            successes=int(np.sum(errors < config.tol)),
            mean_error=float(errors.mean()), median_error=float(np.median(errors)),
            best_error=float(errors.min()), worst_error=float(errors.max()),
            total_seconds=float(seconds.sum()), mean_seconds=float(seconds.mean()),
            objective_seconds=objective_total,
            overhead_seconds=float(seconds.sum()) - objective_total,
            budget_ok=all(record.budget_ok for record in records))

    reference = summaries.get(REFERENCE_BACKEND)
    if reference is not None:
        for summary in summaries.values():
            summary.speedup_total = reference.total_seconds / max(summary.total_seconds, MIN_SECONDS)
            summary.speedup_overhead = (reference.overhead_seconds
                                        / max(summary.overhead_seconds, MIN_SECONDS))
    return list(summaries.values())


def print_case_summary(case: BenchmarkCase, summaries: list[BackendSummary], runs: int) -> None:
    """Summary lines of one test case."""
    for summary in summaries:
        speedup = ""
        if summary.backend != REFERENCE_BACKEND and not math.isnan(summary.speedup_total):
            speedup = f" | speedup x{summary.speedup_total:6.1f}"
            if not math.isnan(summary.speedup_overhead):
                speedup += f" (overhead x{summary.speedup_overhead:6.1f})"
        objective = ("" if math.isnan(summary.objective_seconds)
                     else f" (objective {summary.objective_seconds:7.3f}s)")
        budget = "" if summary.budget_ok else "  !! BUDGET MISMATCH"
        print(f"{case.name:<11}| d = {case.d:2d} | n = {case.n:3d} | f(x*) check = {case.optimum_check:8.1e} "
              f"| {summary.backend:<8}| success = {summary.successes:2d}/{runs} "
              f"| median err = {summary.median_error:9.2e} | worst err = {summary.worst_error:9.2e} "
              f"| time = {summary.total_seconds:8.3f}s{objective}{speedup}{budget}")


# ---------------------------------------------------------------------------
# OUTPUT.
def collect_environment(config: BenchmarkConfig) -> dict:
    """Versions and machine details saved next to the results."""
    extension = gfo.get_extension()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pymoo": pymoo.__version__,
        "gfo": gfo.__version__,
        "gfo_openmp": bool(extension.openmp),
        "gfo_max_threads": int(extension.max_threads),
        "gfo_threads_used": int(config.threads or extension.max_threads),
    }


def write_csv(path: Path, rows: list) -> None:
    """Write a list of dataclass records to CSV."""
    if not rows:
        return
    dicts = [asdict(row) for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dicts[0]))
        writer.writeheader()
        writer.writerows(dicts)


def save_results(output_dir: Path, config: BenchmarkConfig, environment: dict,
                 records: list[RunRecord], summaries: list[BackendSummary]) -> None:
    """Save the per run records, the summaries and the configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runs.csv", records)
    write_csv(output_dir / "summary.csv", summaries)
    metadata = {"config": asdict(config), "environment": environment}
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved results to: {output_dir.resolve()}")


def plot_results(output_dir: Path | None, config: BenchmarkConfig,
                 summaries: list[BackendSummary], show: bool) -> None:
    """Speedup and accuracy figures, one row per dimension."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contenders = [backend for backend in config.backends if backend != REFERENCE_BACKEND]
    has_reference = REFERENCE_BACKEND in config.backends
    lookup = {(s.problem, s.d, s.backend): s for s in summaries}
    names = list(config.problems)
    positions = np.arange(len(names))
    width = 0.8 / max(len(config.backends), 1)

    figures = []
    if has_reference and contenders:
        fig, axes = plt.subplots(len(config.dims), 1, figsize=(11, 4.2 * len(config.dims)),
                                 squeeze=False, layout="constrained")
        for row, d in enumerate(config.dims):
            ax = axes[row, 0]
            for k, backend in enumerate(contenders):
                speedups = [lookup[(name, d, backend)].speedup_total for name in names]
                offset = (k - (len(contenders) - 1) / 2) * width
                bars = ax.bar(positions + offset, speedups, width, label=backend)
                ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Parity (1x)")
            ax.set_yscale("log")
            ax.set_xticks(positions, names, rotation=20)
            ax.set_ylabel("Speedup vs pymoo (total time)")
            ax.set_title(f"d = {d}  |  {config.mode} objective  |  {config.iterations} iterations "
                         f"x {config.runs} runs")
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            ax.legend(fontsize=8)
        figures.append((fig, "speedup.png"))

    fig, axes = plt.subplots(len(config.dims), 1, figsize=(11, 4.2 * len(config.dims)),
                             squeeze=False, layout="constrained")
    for row, d in enumerate(config.dims):
        ax = axes[row, 0]
        for k, backend in enumerate(config.backends):
            # Median error, floored so exact zeros still show on the log axis.
            errors = [max(lookup[(name, d, backend)].median_error, config.tol * 1e-6) for name in names]
            offset = (k - (len(config.backends) - 1) / 2) * width
            ax.bar(positions + offset, errors, width, label=backend)
        ax.axhline(config.tol, color="red", linestyle="--", linewidth=1, label=f"Success tol ({config.tol:g})")
        ax.set_yscale("log")
        ax.set_xticks(positions, names, rotation=20)
        ax.set_ylabel("Median error (best - f*)")
        ax.set_title(f"Accuracy, d = {d}")
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)
    figures.append((fig, "accuracy.png"))

    for fig, filename in figures:
        if output_dir is not None:
            fig.savefig(output_dir / filename, dpi=PLOT_DPI)
            print(f"Saved figure: {(output_dir / filename).resolve()}")
    if show:
        plt.show()
    plt.close("all")


def print_overall(config: BenchmarkConfig, summaries: list[BackendSummary]) -> None:
    """Overall speedups per dimension (geometric mean over the problems)."""
    if REFERENCE_BACKEND not in config.backends:
        return
    contenders = [backend for backend in config.backends if backend != REFERENCE_BACKEND]
    print("-" * LINE_WIDTH)
    print("OVERALL SPEEDUP VS PYMOO (geometric mean over problems, total time)")
    for d in config.dims:
        parts = []
        for backend in contenders:
            speedups = [s.speedup_total for s in summaries if s.d == d and s.backend == backend]
            parts.append(f"{backend}: x{math.exp(np.mean(np.log(speedups))):.1f}")
        print(f"  d = {d:2d} | " + " | ".join(parts))


# ---------------------------------------------------------------------------
# COMMAND LINE.
def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark gfo's PSO against pymoo's PSO.")
    parser.add_argument("--quick", action="store_true",
                        help=f"Smoke test: {QUICK_ITERATIONS} iterations, {QUICK_RUNS} runs, d = {QUICK_DIMS}.")
    parser.add_argument("--iterations", type=int, help=f"Iterations per run (default {ITERATIONS}).")
    parser.add_argument("--runs", type=int, help=f"Runs per test (default {NRUNS}).")
    parser.add_argument("--dims", type=int, nargs="+", help=f"Dimensions (default {DIMS}).")
    parser.add_argument("--only", nargs="+", choices=list(gfo.problems.EXAMPLE_PROBLEMS),
                        help="Only these problems.")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS),
                        help="Backends to run (default: all).")
    parser.add_argument("--mode", choices=MODES, default="vectorized",
                        help="How the Python objective is called (default: vectorized).")
    parser.add_argument("--threads", type=int, default=0,
                        help="gfo OpenMP threads, 0 = all cores (default). pymoo is single threaded.")
    parser.add_argument("--tol", type=float, default=TOL, help=f"Success tolerance (default {TOL}).")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE, help=f"Seed of run 0 (default {SEED_BASE}).")
    parser.add_argument("--pymoo-lhs", action="store_true",
                        help="Use pymoo's default LHS initial sampling instead of uniform random.")
    parser.add_argument("--no-objective-timing", action="store_true",
                        help="Do not time the Python objective (removes the timer overhead).")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Results folder (default: {DEFAULT_OUTPUT_ROOT}/<timestamp>).")
    parser.add_argument("--no-save", action="store_true", help="Do not save results or figures.")
    parser.add_argument("--no-plots", action="store_true", help="Do not make figures.")
    parser.add_argument("--show", action="store_true", help="Show the figures interactively.")
    parser.add_argument("-q", "--quiet", action="store_true", help="No per run lines.")
    return parser


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Turn the command line into a configuration."""
    config = BenchmarkConfig(mode=args.mode, threads=args.threads, tol=args.tol,
                             seed_base=args.seed_base, pymoo_lhs=args.pymoo_lhs,
                             time_objective=not args.no_objective_timing,
                             backends=tuple(dict.fromkeys(args.backends)))
    if args.quick:
        config.iterations, config.runs, config.dims = QUICK_ITERATIONS, QUICK_RUNS, QUICK_DIMS
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.runs is not None:
        config.runs = args.runs
    if args.dims:
        config.dims = tuple(args.dims)
    if args.only:
        config.problems = tuple(args.only)

    invalid = (config.iterations < 1 or config.runs < 1 or config.threads < 0
               or min(config.dims) < 1)
    if invalid:
        sys.exit("ERROR: iterations, runs and dims must be >= 1 and threads >= 0.")
    return config


def main(argv=None) -> int:
    """Run the benchmark. Returns 1 if any check failed or any budget mismatched."""
    args = build_parser().parse_args(argv)
    if PYMOO_IMPORT_ERROR is not None:
        sys.exit(f"ERROR: pymoo is not installed ({PYMOO_IMPORT_ERROR}). "
                 "Run: pip install -e \".[benchmark]\"")
    if not gfo.EXTENSION_AVAILABLE:
        sys.exit("ERROR: the gfo C++ extension is not built. Run: python scripts/build.py")

    config = build_config(args)
    environment = collect_environment(config)

    print("-" * LINE_WIDTH)
    print("PSO BENCHMARK: " + " vs ".join(config.backends) + f" ({config.mode} objective, interleaved runs)")
    print("-" * LINE_WIDTH)
    print(f"Iterations = {config.iterations} | Runs per test = {config.runs} | Success tolerance = {config.tol} "
          f"| Dimensions = {list(config.dims)} | Particles = {BASE_PARTICLES} + {PARTICLES_PER_DIM} d")
    print(f"Inertia = {config.inertia} | Acceleration 1 = {config.acceleration1} | Acceleration 2 = "
          f"{config.acceleration2} | Max velocity rate = {config.max_velocity_rate} | pymoo sampling = "
          f"{'LHS' if config.pymoo_lhs else 'uniform random'}, adaptive off, pertube_best off")
    print(f"pymoo {environment['pymoo']} (1 thread) | gfo {environment['gfo']} "
          f"({environment['gfo_threads_used']} of {environment['gfo_max_threads']} threads, "
          f"OpenMP {'on' if environment['gfo_openmp'] else 'OFF'}) | Python {environment['python']} "
          f"| numpy {environment['numpy']}")
    print("-" * LINE_WIDTH)

    consistent = check_problem_consistency(config)
    if not consistent:
        print("WARNING: the Python and C++ problems disagree, gfo-cpp results are NOT comparable.")
    print("Warming up every backend...")
    warm_up(config)

    start = time.perf_counter()
    records: list[RunRecord] = []
    summaries: list[BackendSummary] = []
    for d in config.dims:
        for name in config.problems:
            case = make_case(config, name, d)
            run_case(config, case, verbose=not args.quiet)
            case_summaries = summarize_case(config, case)
            print_case_summary(case, case_summaries, config.runs)
            records.extend(case.records)
            summaries.extend(case_summaries)

    budgets_ok = all(record.budget_ok for record in records)
    print_overall(config, summaries)
    print("-" * LINE_WIDTH)
    print("BENCHMARK FINISHED.")
    print(f"Total time: {time.perf_counter() - start:.2f}s | Evaluation budgets: "
          f"{'all equal' if budgets_ok else 'MISMATCH (see !! lines)'} | Problem check: "
          f"{'passed' if consistent else 'FAILED'}")

    output_dir = None
    if not args.no_save:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output or DEFAULT_OUTPUT_ROOT / f"{stamp}_{config.mode}"
        save_results(output_dir, config, environment, records, summaries)
    if not args.no_plots and (output_dir is not None or args.show):
        plot_results(output_dir, config, summaries, args.show)
    return 0 if budgets_ok and consistent else 1


if __name__ == "__main__":
    sys.exit(main())
