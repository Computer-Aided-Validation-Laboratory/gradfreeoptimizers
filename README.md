# gradfreeoptimizers
A C++ powered collection of gradient free optimizers with a Python interface (`import gfo`).

v0.1 has one optimizer, a global-best particle swarm (PSO), with two ways to evaluate the objective:

| Objective | What runs where | Use it for |
|---|---|---|
| A C++ example problem, `gfo.cpp_problems.Rastrigin()` | The whole run stays in C++. Every particle is evaluated in parallel (OpenMP) with the GIL released. | Speed. Objectives written in C++. |
| Any Python callable | The swarm update runs in C++ (in parallel). Python evaluates the whole swarm once per iteration: `f(X)` with `X.shape == (n, d)` returns `n` values. | Objectives that only exist in Python. Vectorized numpy code. |

For the same seed and settings both modes visit exactly the same points, and results do not depend on the number of threads.

## Install

Needs Python >= 3.10 and a C++23 compiler with OpenMP (on Windows: MSYS2 ucrt64 g++).

```
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"            # numpy, pandas, matplotlib + pybind11, pymoo, pytest
python scripts/build.py            # builds src/gfo/_pso<suffix>, then checks the import
```

`pip install -e .` alone installs only the runtime dependencies. The extras are `build` (pybind11), `benchmark` (pymoo) and `dev` (pytest). `all` installs all three.

`scripts/build.py` options:
- `--example` also builds the C++ example program (`build/gfo_example`). `--example-only` builds only that.
- `--cxx PATH` picks the compiler (default: `$CXX`, then MSYS2 g++ on Windows, then `g++`).
- `--native` (`-march=native`), `--no-openmp`, `--debug`.

On Windows the GCC runtime, OpenMP and winpthread are linked statically, so the module needs no MSYS2 DLLs at run time.

## Quick start (Python)

```python
import gfo

# Python objective: an example problem (vectorized numpy), bounds come from the problem.
result = gfo.pso(gfo.problems.Rastrigin(), d=10, iterations=1000, n_particles=40)

# Same problem in C++: the whole run stays in C++.
result = gfo.pso(gfo.cpp_problems.Rastrigin(), d=10, iterations=1000, n_particles=40)

print(result.best_val, result.best_pos)    # best value, position (physical coordinates)
result.loss_history                        # global best after every iteration (numpy)
result.evaluations                         # n_particles * iterations
result.diagnostics.total_s                 # timings, see "Console output" below

# Your own objective. Vectorized: X has shape (n, d), return n values.
result = gfo.pso(lambda X: ((X - 0.5) ** 2).sum(axis=1), d=3, lo=-1.0, hi=1.0)

# One point at a time (called once per particle, like a pymoo ElementwiseProblem).
result = gfo.pso(lambda x: float(((x - 0.5) ** 2).sum()), d=3, lo=-1.0, hi=1.0, vectorized=False)
```

`scripts/example_python.py` runs all of this, plus a saved run that is post-processed.

### `gfo.pso` settings

Every setting has a default. The defaults reproduce the original specklemimic optimizer (v0.2) exactly.

| Argument | Default | Meaning |
|---|---|---|
| `d` | from `lo`/`hi` | Dimension. |
| `lo`, `hi` | the objective's `lo`/`hi` | Bounds, a scalar applies to every dimension. |
| `n_particles` | 20 | Swarm size. |
| `iterations` | 100 | Every iteration evaluates every particle once. |
| `inertia`, `acceleration1`, `acceleration2` | 0.729, 1.49445, 1.49445 | Clerc-Kennedy constriction values. |
| `max_velocity_rate` | 0.15 | Velocity clamp as a fraction of the domain width (pymoo's `max_velocity_rate`). |
| `initial_velocity_rate` | 0.02 | Initial velocities are uniform in `[-rate, rate]`. 0 starts the swarm at rest. |
| `seed` | 42 | Seed. |
| `default_bounds` | False | Ignore `lo`/`hi` and search the unit hypercube. |
| `vectorized` | True | Whether a Python objective takes the whole swarm or one point. |
| `threads` | 0 | OpenMP threads, 0 = all cores. |
| `verbose`, `verbose_interval` | False, 1 | Start/end summary and a line every `verbose_interval` iterations. |
| `diagnostics`, `diagnostics_interval` | False, 0 | Section timings (see below). 0 = about 10 lines per run. |
| `run_dir`, `optimizer_dir` | current folder, `"optimizer"` | Files go to `<run_dir>/<optimizer_dir>/`. |
| `save_loss` | False | Write `loss.csv` (`iteration, f`). |
| `save_history` | False | 2D only: write `history.csv` (`iteration, x, y, f`) and `domain.csv`. |

## Example problems

| Problem | Type | Default domain | Minimum |
|---|---|---|---|
| Sphere | convex, unimodal | [-5.12, 5.12]^d | f = 0 at x_k = shift (2) |
| Ellipsoid | convex, ill conditioned | [-5.12, 5.12]^d | f = 0 at x_k = shift (2) |
| Rosenbrock | non-convex, curved valley | [-5, 10]^d | f = 0 at x_k = shift + a (3) |
| Rastrigin | non-convex, multimodal grid | [-5.12, 5.12]^d | f = 0 at x_k = shift (2) |
| Ackley | non-convex, one narrow basin | [-32.768, 32.768]^d | f = 0 at x_k = shift (2) |
| Griewank | non-convex, coupled | [-600, 600]^d | f = 0 at x_k = shift (100) |
| Schwefel | non-convex, deceptive | [-500, 500]^d | f = 0 at x_k = 420.9687 + shift (0) |

Every problem exists three times with the same parameters and defaults:

- **C++**, `src/exampleProblems/<name>.hpp`:
  - a free function with default valued parameters, e.g. `problems::rastrigin(x, d, shift = 2.0, amplitude = 10.0)`;
  - a struct holding the parameters, default domain and known optimum, e.g. `problems::Rastrigin{.shift = 1.0}`. It is callable as `double(const double*, size_t)`, so it goes straight into `run_pso`.
- **Python**, `gfo.problems.Rastrigin(shift=1.0)`: a dataclass, vectorized with numpy.
- **C++ from Python**, `gfo.cpp_problems.Rastrigin(shift=1.0)`: the C++ struct, bound with pybind11.

All three have the same interface: `problem(x)` for one point `(d,)` or a batch `(n, d)`, plus `lo`, `hi`, `x_opt(d)`, `f_opt()` and `name`. The Python and C++ versions agree to about 1e-16 (checked by the tests and by the benchmark before it runs).

## C++ use

```cpp
#include "pso/pso.hpp"
#include "exampleProblems/exampleProblems.hpp"

problems::Rastrigin problem;                    // all defaults
std::vector<double> lo(d, problem.lo), hi(d, problem.hi);

PSOSettings settings;                           // every field has a default
settings.iterations = 500;
settings.verbose = true;

PSOResult result = run_pso(problem, d, lo, hi, settings);           // parallel C++ objective
PSOResult result = run_pso_batch(batch_objective, d, lo, hi, settings);  // one call per iteration
```

- Any `double f(const double* x, size_t d)` works with `run_pso`: a function, a lambda or a functor. It is called from several threads at once.
- `run_pso_batch` takes `double f(const double* X, size_t n, size_t d, double* out)`.
- The original call, `spsoptimzer(f, it, n, d, lo, hi, default_bounds, ...)`, still compiles and gives the same results.
- `src/main.cpp` is a small, commented example. Build it with `python scripts/build.py --example-only`.

## Console output

- `verbose=True` prints:
  - a start banner with every setting;
  - the original per-iteration line (` | Iteration = i | Global Best Value = ... | Time = ... ms |`);
  - an end summary with the best value, position, evaluations and time.
- `diagnostics=True` times every section of the loop. Every `diagnostics_interval` iterations it prints the time in the particle loop, the share spent in the objective versus the particle update, the time per evaluation and the current best.
- At the end of a diagnostics run there is a breakdown table:
  - setup, particle update, objective calls (split into time inside the objective and data transfer, for Python objectives), best bookkeeping, output and unaccounted time;
  - for C++ objectives, the thread time in the objective versus the update, and the parallel efficiency;
  - the mean time per evaluation, iteration time and evaluations per second.

The wall-clock timings are always measured, and returned in `result.diagnostics` (`total_s`, `swarm_s`, `evaluate_s`, `objective_s`, ...). Per-call thread timings need `diagnostics=True`. From Python the C++ output goes to `sys.stdout`, so it shows up in notebooks too. With neither flag set, a run prints nothing.

## Post-processing

Save a run (`run_dir=..., save_loss=True`, plus `save_history=True` for 2D), then:

```
python scripts/plot_optimizer.py -r run1 --xlabel "x" --ylabel "y" [--static-only]
```

or from Python, `gfo.post.run_optimizer_post_processing("run1", gfo.post.OptimizerOptions(...))`.

Figures go to `run1_post/optimizer/`:
- loss curve;
- particle trajectories;
- loss contours from the final swarm and from all iterations;
- swarm and 3D landscape animations (GIF, or MP4 with ffmpeg).

Runs with `d != 2` only get the loss curve. This is the optimizer part of the specklemimic post-processing, unchanged apart from the configurable folder names.

## Benchmark against pymoo

```
python scripts/benchmark_pymoo.py --quick                       # smoke test (seconds)
python scripts/benchmark_pymoo.py                               # 7 problems x d = 2, 10, 30 x 20 runs x 2000 iterations
python scripts/benchmark_pymoo.py --mode elementwise --threads 1
```

There are three backends on the same problems and budget (`n = 20 + 2d` particles, `n * iterations` evaluations):
- `pymoo`: pymoo's PSO on the Python problem.
- `gfo-py`: gfo on the same Python problem object.
- `gfo-cpp`: gfo on the C++ twin.

`--mode vectorized` (default) compares pymoo `Problem` with gfo's batch objective. `--mode elementwise` compares pymoo `ElementwiseProblem` with gfo using `vectorized=False`.

How the comparison is kept fair:
- **Same settings.** Inertia, accelerations and velocity clamp match. pymoo runs with `adaptive=False`, `pertube_best=False` and uniform random initial positions (`--pymoo-lhs` switches its default LHS back on).
- **Same problems.** The Python and C++ versions are checked against each other first.
- **Same budget.** Every run's evaluation count is checked; a mismatch is flagged and gives exit code 1.
- **Timing.** Runs are interleaved and the backend order rotates, every backend gets a warm-up run, and garbage is collected before every timed run.

Results go to `benchmark_results/<timestamp>_<mode>/`:
- `runs.csv` (every run);
- `summary.csv` (per problem, dimension and backend, with speedups);
- `config.json` (settings and versions);
- `speedup.png` and `accuracy.png`.

Differences that remain because the two libraries differ:
- initial velocities;
- gfo moves the swarm once before its first evaluation;
- boundary handling: gfo clamps and reflects the velocity, pymoo resamples and then repairs.

pymoo runs on one core. gfo uses `--threads` (all cores by default).

## Performance notes

- **Per-particle RNG.** Every (iteration, particle) pair builds its own `std::seed_seq` and `std::mt19937_64`, so results do not depend on the thread count. That costs about 6 us per particle per iteration, which is most of the optimizer's own time when the objective is cheap (a Rastrigin term is about 0.01 us). `diagnostics=True` shows it as "update" time. It is kept as in the original. A lighter generator is the obvious speed-up for a later version.
- **Thread count.** For small swarms (tens of particles), 4 threads is often faster than all 16, because waking every thread costs more than the work. Try `threads=` for your problem size.

## Tests

```
pytest
```

The tests check:
- the Python and C++ problems match;
- C++ mode, Python batch mode and every thread count give identical results;
- the evaluation budget, bounds and loss history are right;
- invalid input raises errors;
- the output files are written correctly;
- the post-processing works.

## Layout

```
src/
  pso/                 pso.hpp (optimizer), pso.cpp, pso_bindings.cpp (pybind11 -> gfo._pso)
  exampleProblems/     one header per C++ example problem, exampleProblems.hpp includes them all
  main.cpp             C++ example program
  gfo/                 Python package
    optimizer.py       gfo.pso
    problems/          Python example problems
    post/              post-processing (loaders, plots, workflow)
scripts/               build.py, benchmark_pymoo.py, example_python.py, plot_optimizer.py
tests/                 pytest suite
```
