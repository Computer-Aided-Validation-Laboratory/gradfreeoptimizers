"""Gold regression tests: exact results that must never change by accident.

- pso_gold.json: results of the original specklemimic PSO (the ground
  truth), which the engine reproduces bit for bit. Never regenerated.
- pso_regression_gold.json: every evaluated point (SHA-256) and the loss
  history of runs that use every algorithm setting, from the current engine.
  Regenerate with tests/gold/generate_regression_gold.py after a deliberate
  change only.

Exact comparison: valid for the same compiler and libm, a rebuild with other
flags (e.g. --native) may change the last bits and the trajectory.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import gfo
from pso_testing import SwarmRecorder, calc_trajectory_hash

GOLD_DIR = Path(__file__).parent / "gold"
TRUTH_CASES: list[dict] = json.loads((GOLD_DIR / "pso_gold.json").read_text())[
    "cases"
]
REGRESSION_CASES: list[dict] = json.loads(
    (GOLD_DIR / "pso_regression_gold.json").read_text()
)["cases"]


@pytest.mark.parametrize(
    "case", TRUTH_CASES, ids=[f"{c['problem']}-d{c['d']}" for c in TRUTH_CASES]
)
@pytest.mark.parametrize("mode", ["cpp", "batch"])
def test_matches_the_ground_truth(case: dict, mode: str) -> None:
    problem = getattr(gfo.cpp_problems, case["problem"])()
    objective = problem if mode == "cpp" else problem.evaluate
    result = gfo.pso(
        objective,
        problem.lo,
        problem.hi,
        d=case["d"],
        n_particles=case["n_particles"],
        iterations=case["iterations"],
        seed=case["seed"],
    )

    assert result.best_val == case["best_val"]
    np.testing.assert_array_equal(result.best_pos, case["best_pos"])


@pytest.mark.parametrize(
    "case", REGRESSION_CASES, ids=[case["label"] for case in REGRESSION_CASES]
)
def test_matches_the_regression_gold(case: dict) -> None:
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
            recorder.evaluate, case["lo"], case["hi"], d=case["d"], **case["settings"]
        )
        trajectory_hash = calc_trajectory_hash(
            recorder.stack_points(), recorder.stack_values()
        )
        assert trajectory_hash == case["trajectory_sha256"]

    assert result.best_val == case["best_val"]
    np.testing.assert_array_equal(result.best_pos, case["best_pos"])
    np.testing.assert_array_equal(result.loss_history, case["loss_history"])
