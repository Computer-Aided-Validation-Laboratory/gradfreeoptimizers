"""Generate tests/gold/pso_regression_gold.json from the current engine.

The regression gold pins every point evaluated by a few runs that between
them use every algorithm setting, bit for bit (SHA-256 of the trajectory),
plus their loss histories. tests/gold/pso_gold.json, the ground truth from
the original specklemimic optimizer, is separate and never regenerated.

Regenerate only after a deliberate change of the optimizer's results (a new
random number generator, a changed update rule) and say so in the commit:

    python tests/gold/generate_regression_gold.py

Like all gold results it is exact for one compiler and libm: a rebuild with
other flags (e.g. --native) may change the last bits.
"""

import json
import sys
from pathlib import Path

import numpy as np

import gfo

TESTS_DIR = Path(__file__).resolve().parents[1]
GOLD_PATH = TESTS_DIR / "gold" / "pso_regression_gold.json"

# Every case: a problem, its mode, bounds, d and the gfo.pso settings. Python
# mode cases are recorded point by point, C++ mode cases by their results.
CASES: list[dict] = [
    {
        "label": "defaults",
        "problem": "Rastrigin",
        "mode": "python",
        "lo": -5.12,
        "hi": 5.12,
        "d": 2,
        "settings": {},
    },
    {
        "label": "coefficients",
        "problem": "Ackley",
        "mode": "python",
        "lo": -32.768,
        "hi": 32.768,
        "d": 5,
        "settings": {
            "n_particles": 24,
            "iterations": 80,
            "seed": 11,
            "inertia": 0.5,
            "acceleration1": 2.0,
            "acceleration2": 1.0,
        },
    },
    {
        "label": "velocity_rates",
        "problem": "Rosenbrock",
        "mode": "python",
        "lo": -5.0,
        "hi": 10.0,
        "d": 3,
        "settings": {
            "n_particles": 15,
            "iterations": 80,
            "seed": 5,
            "max_velocity_rate": 0.05,
            "initial_velocity_rate": 0.0,
        },
    },
    {
        "label": "per_dimension_bounds",
        "problem": "Griewank",
        "mode": "python",
        "lo": [-600.0, -300.0, -10.0, 0.0],
        "hi": [600.0, 300.0, 10.0, 50.0],
        "d": 4,
        "settings": {"n_particles": 18, "iterations": 70, "seed": 2},
    },
    {
        "label": "one_dimension_wide_seed",
        "problem": "Sphere",
        "mode": "python",
        "lo": -5.12,
        "hi": 5.12,
        "d": 1,
        "settings": {"n_particles": 7, "iterations": 50, "seed": 2**40 + 3},
    },
    {
        "label": "cpp_high_dimension",
        "problem": "Schwefel",
        "mode": "cpp",
        "lo": -500.0,
        "hi": 500.0,
        "d": 30,
        "settings": {"n_particles": 50, "iterations": 150, "seed": 9},
    },
]


def main() -> None:
    sys.path.insert(0, str(TESTS_DIR))
    from pso_testing import SwarmRecorder, calc_trajectory_hash

    results: list[dict] = []
    for case in CASES:
        entry: dict = dict(case)
        if case["mode"] == "cpp":
            problem = getattr(gfo.cpp_problems, case["problem"])()
            result = gfo.pso(
                problem, case["lo"], case["hi"], d=case["d"], **case["settings"]
            )
        else:
            recorder = SwarmRecorder(
                gfo.problems.EXAMPLE_PROBLEMS[case["problem"]]().evaluate
            )
            result = gfo.pso(
                recorder.evaluate,
                case["lo"],
                case["hi"],
                d=case["d"],
                **case["settings"],
            )
            entry["trajectory_sha256"] = calc_trajectory_hash(
                recorder.stack_points(), recorder.stack_values()
            )
        entry["best_val"] = result.best_val
        entry["best_pos"] = result.best_pos.tolist()
        entry["loss_history"] = result.loss_history.tolist()
        results.append(entry)

    payload = {
        "note": (
            "Regression gold of the current engine, written by "
            "generate_regression_gold.py. Regenerate only after a deliberate "
            "change of the optimizer's results."
        ),
        "numpy": np.__version__,
        "cases": results,
    }
    GOLD_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"Wrote {len(results)} cases to {GOLD_PATH}")


if __name__ == "__main__":
    main()
