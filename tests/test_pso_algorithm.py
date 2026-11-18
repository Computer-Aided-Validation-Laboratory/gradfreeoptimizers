"""Verification of the PSO algorithm itself, independent of its random numbers.

Every run here is replayed from the points the objective saw (see
pso_testing.py). The replay rebuilds positions, velocities, personal and
global bests, then checks every step against the update equations

    v <- clamp(w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), -vmax, vmax)
    x <- reflect(x + v)                  (unit box, velocity flips at a wall)

for some r1, r2 in [0, 1). No random number is reproduced, so these tests
survive a change of random number generator.
"""

from collections.abc import Callable

import numpy as np
import pytest

import gfo
from pso_testing import (
    UNIT_HI,
    UNIT_LO,
    SwarmRecorder,
    calc_ks_uniform,
    calc_swarm_steps,
    evaluate_unit_rastrigin,
    find_velocity_violations,
    record_unit_run,
    replay_bests,
)

D = 4
RUN = {"n_particles": 30, "iterations": 60, "seed": 5}
# Below this many checked steps a replay test proves too little.
MIN_CHECKED_STEPS = 3000
# Kolmogorov-Smirnov critical value at a 0.1 % significance level: c / sqrt(n).
KS_CRITICAL_FACTOR = 1.95

COEFFICIENTS: dict[str, dict[str, float]] = {
    "defaults": {
        "inertia": 0.729,
        "acceleration1": 1.49445,
        "acceleration2": 1.49445,
        "max_velocity_rate": 0.15,
    },
    "inertia_only": {
        "inertia": 0.9,
        "acceleration1": 0.0,
        "acceleration2": 0.0,
        "max_velocity_rate": 0.15,
    },
    "asymmetric": {
        "inertia": 0.4,
        "acceleration1": 0.5,
        "acceleration2": 2.0,
        "max_velocity_rate": 0.15,
    },
    "clamp_heavy": {
        "inertia": 1.2,
        "acceleration1": 2.5,
        "acceleration2": 2.5,
        "max_velocity_rate": 0.02,
    },
}


def sum_squares_around(centre: float) -> Callable[[np.ndarray], np.ndarray]:
    """Sphere on the unit box with its minimum at x_k = centre."""

    def evaluate(points: np.ndarray) -> np.ndarray:
        return np.sum((points - centre) ** 2, axis=1)

    return evaluate


@pytest.mark.parametrize("initial_velocity_rate", [0.0, 0.02])
@pytest.mark.parametrize("label", list(COEFFICIENTS))
def test_every_velocity_update_follows_the_pso_equations(
    label: str, initial_velocity_rate: float
) -> None:
    coefficients = COEFFICIENTS[label]
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        D,
        initial_velocity_rate=initial_velocity_rate,
        **RUN,
        **coefficients,
    )
    steps = calc_swarm_steps(
        recorder.stack_points(),
        recorder.stack_values(),
        starts_at_rest=initial_velocity_rate == 0.0,
    )
    violations = find_velocity_violations(steps, **coefficients)

    assert steps.velocity.size > MIN_CHECKED_STEPS
    assert not np.any(violations), f"{violations.sum()} steps break the update"


def evaluate_unit_staircase(points: np.ndarray) -> np.ndarray:
    """Piecewise constant landscape on the unit box: exact ties everywhere."""
    return np.sum(np.floor(np.abs(points - 0.3) * 4.0), axis=1)


def test_velocity_updates_follow_the_equations_on_plateaus() -> None:
    # Ties decide whether a best moves: only a strict improvement may move it,
    # so a replay with strict bests must explain every step.
    coefficients = COEFFICIENTS["defaults"]
    recorder, _ = record_unit_run(
        evaluate_unit_staircase, D, initial_velocity_rate=0.02, **RUN, **coefficients
    )
    values = recorder.stack_values()
    steps = calc_swarm_steps(recorder.stack_points(), values, starts_at_rest=False)
    violations = find_velocity_violations(steps, **coefficients)

    assert np.count_nonzero(values[1:] == values[:-1]) > 1000
    assert steps.velocity.size > MIN_CHECKED_STEPS
    assert not np.any(violations), f"{violations.sum()} steps break the update"


def test_the_two_random_numbers_are_drawn_independently() -> None:
    # With r1 == r2 every step would be r (c1 a + c2 b), between 0 and
    # c1 a + c2 b. Independent draws land outside that range whenever the two
    # pulls point in opposite directions and the weaker one gets the larger r.
    coefficients = COEFFICIENTS["defaults"]
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        D,
        initial_velocity_rate=0.02,
        **{**RUN, "iterations": 120},
        **{**coefficients, "max_velocity_rate": 1.0},
    )
    steps = calc_swarm_steps(
        recorder.stack_points(), recorder.stack_values(), starts_at_rest=False
    )
    cognitive = coefficients["acceleration1"] * steps.cognitive
    social = coefficients["acceleration2"] * steps.social
    opposed = (cognitive * social < 0.0) & (np.abs(cognitive + social) > 1e-6)
    pull = (cognitive + social)[opposed]
    attraction = (
        steps.velocity - coefficients["inertia"] * steps.previous_velocity
    )[opposed]
    outside = (attraction < np.minimum(pull, 0.0) - 1e-9) | (
        attraction > np.maximum(pull, 0.0) + 1e-9
    )

    assert opposed.sum() > 500
    assert outside.mean() > 0.1


@pytest.mark.parametrize(
    ("label", "wrong"),
    [
        ("defaults", {"inertia": 0.779}),
        ("inertia_only", {"inertia": 0.95}),
        ("asymmetric", {"acceleration1": 2.0, "acceleration2": 0.5}),
        ("asymmetric", {"acceleration1": 0.4, "acceleration2": 1.6}),
        ("clamp_heavy", {"max_velocity_rate": 0.03}),
        ("clamp_heavy", {"max_velocity_rate": 0.01}),
    ],
)
def test_the_replay_detects_wrong_coefficients(
    label: str, wrong: dict[str, float]
) -> None:
    # Negative control: the replay check must fail for coefficients the run
    # did not use, otherwise passing it proves nothing.
    coefficients = COEFFICIENTS[label]
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin, D, initial_velocity_rate=0.02, **RUN, **coefficients
    )
    steps = calc_swarm_steps(
        recorder.stack_points(), recorder.stack_values(), starts_at_rest=False
    )

    assert np.any(find_velocity_violations(steps, **{**coefficients, **wrong}))


def test_velocity_clamp_is_reached_exactly() -> None:
    coefficients = COEFFICIENTS["clamp_heavy"]
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin, D, initial_velocity_rate=0.02, **RUN, **coefficients
    )
    steps = calc_swarm_steps(
        recorder.stack_points(), recorder.stack_values(), starts_at_rest=False
    )
    speed = np.abs(steps.velocity)
    vmax = coefficients["max_velocity_rate"]

    assert speed.max() == pytest.approx(vmax, abs=1e-15)
    assert np.count_nonzero(np.isclose(speed, vmax, rtol=0.0, atol=1e-15)) > 1000


@pytest.mark.parametrize(
    ("acceleration1", "acceleration2"), [(0.3, 0.0), (0.0, 0.3)], ids=["r1", "r2"]
)
def test_random_coefficients_are_uniform_on_the_unit_interval(
    acceleration1: float, acceleration2: float
) -> None:
    # With one attraction switched off, every free step gives the other
    # random number exactly: r = (v - w v_prev) / (c (best - x)).
    inertia = 0.5
    recorder, _ = record_unit_run(
        sum_squares_around(0.5),
        5,
        n_particles=40,
        iterations=100,
        seed=1,
        inertia=inertia,
        acceleration1=acceleration1,
        acceleration2=acceleration2,
        max_velocity_rate=1.0,
        initial_velocity_rate=0.05,
    )
    steps = calc_swarm_steps(
        recorder.stack_points(), recorder.stack_values(), starts_at_rest=False
    )
    pull = steps.cognitive if acceleration1 > 0.0 else steps.social
    coefficient = acceleration1 + acceleration2
    usable = np.abs(pull) > 1e-3
    samples = (steps.velocity[usable] - inertia * steps.previous_velocity[usable]) / (
        coefficient * pull[usable]
    )

    assert samples.size > 500
    assert samples.min() >= -1e-9
    assert samples.max() < 1.0 + 1e-9
    assert calc_ks_uniform(samples) < KS_CRITICAL_FACTOR / np.sqrt(samples.size)


def test_inertia_alone_scales_the_velocity_every_iteration() -> None:
    inertia = 0.9
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        D,
        inertia=inertia,
        acceleration1=0.0,
        acceleration2=0.0,
        initial_velocity_rate=0.1,
        **RUN,
    )
    steps = calc_swarm_steps(
        recorder.stack_points(), recorder.stack_values(), starts_at_rest=False
    )

    np.testing.assert_allclose(
        steps.velocity, inertia * steps.previous_velocity, rtol=0.0, atol=1e-12
    )


def test_swarm_freezes_without_inertia_and_attraction() -> None:
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        D,
        inertia=0.0,
        acceleration1=0.0,
        acceleration2=0.0,
        **RUN,
    )
    points = recorder.stack_points()

    np.testing.assert_array_equal(points, np.broadcast_to(points[0], points.shape))


def test_particles_bounce_off_the_walls() -> None:
    # Inertia 1 and no attraction: every particle flies straight and must
    # bounce like a billiard ball, x -> reflect(x + v), v -> -v at a wall.
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        3,
        n_particles=10,
        iterations=60,
        seed=2,
        inertia=1.0,
        acceleration1=0.0,
        acceleration2=0.0,
        max_velocity_rate=0.5,
        initial_velocity_rate=0.3,
    )
    points = recorder.stack_points()
    is_interior = (points > UNIT_LO) & (points < UNIT_HI)
    n_iter, n_particles, dims = points.shape

    simulated = np.full(points.shape, np.nan)
    for jj in range(n_particles):
        for kk in range(dims):
            track = is_interior[:, jj, kk]
            # The first two consecutive interior positions give the velocity.
            start = int(np.argmax(track[:-1] & track[1:]))
            x = points[start + 1, jj, kk]
            v = x - points[start, jj, kk]
            simulated[: start + 2, jj, kk] = points[: start + 2, jj, kk]
            for ii in range(start + 2, n_iter):
                x += v
                if x < UNIT_LO:
                    x, v = UNIT_LO, -v
                elif x > UNIT_HI:
                    x, v = UNIT_HI, -v
                simulated[ii, jj, kk] = x

    assert np.count_nonzero(~is_interior) > 100
    np.testing.assert_allclose(points, simulated, rtol=0.0, atol=1e-10)


def test_first_iteration_skips_the_velocity_update() -> None:
    # Iteration 0 is a burn in: positions only move by the initial velocity,
    # so they cannot depend on the inertia or the accelerations.
    first_points = []
    for coefficients in COEFFICIENTS.values():
        settings = {**coefficients, "max_velocity_rate": 0.15}
        recorder, _ = record_unit_run(
            evaluate_unit_rastrigin, D, initial_velocity_rate=0.02, **RUN, **settings
        )
        first_points.append(recorder.batches[0])

    for points in first_points[1:]:
        np.testing.assert_array_equal(points, first_points[0])


@pytest.mark.parametrize(
    ("initial_velocity_rate", "max_velocity_rate", "largest_move"),
    [(0.02, 0.15, 0.02), (0.05, 0.01, 0.01)],
    ids=["initial_rate", "clamped"],
)
def test_initial_velocity_only_nudges_the_initial_positions(
    initial_velocity_rate: float, max_velocity_rate: float, largest_move: float
) -> None:
    at_rest, _ = record_unit_run(
        evaluate_unit_rastrigin, D, initial_velocity_rate=0.0, **RUN
    )
    nudged, _ = record_unit_run(
        evaluate_unit_rastrigin,
        D,
        initial_velocity_rate=initial_velocity_rate,
        max_velocity_rate=max_velocity_rate,
        **RUN,
    )
    move = np.abs(nudged.batches[0] - at_rest.batches[0])

    assert move.max() <= largest_move + 1e-15
    assert move.max() > 0.5 * largest_move


def test_initial_positions_are_uniform_in_the_box() -> None:
    recorder, _ = record_unit_run(
        evaluate_unit_rastrigin,
        5,
        n_particles=400,
        iterations=1,
        seed=9,
        initial_velocity_rate=0.0,
    )
    samples = recorder.batches[0].ravel()

    assert calc_ks_uniform(samples) < KS_CRITICAL_FACTOR / np.sqrt(samples.size)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(UNIT_LO, UNIT_HI), ([-3.0, 0.5, 100.0, -1e-3], [5.0, 0.75, 300.0, 1e-3])],
    ids=["unit_box", "per_dimension_bounds"],
)
def test_result_is_the_running_best_of_every_evaluation(
    lo: float | list[float], hi: float | list[float]
) -> None:
    lo_array = np.broadcast_to(np.asarray(lo, dtype=float), (D,))
    hi_array = np.broadcast_to(np.asarray(hi, dtype=float), (D,))

    def evaluate_normalized(points: np.ndarray) -> np.ndarray:
        return evaluate_unit_rastrigin((points - lo_array) / (hi_array - lo_array))

    recorder = SwarmRecorder(evaluate_normalized)
    result = gfo.pso(recorder.evaluate, lo, hi, d=D, **RUN)
    points = recorder.stack_points()
    values = recorder.stack_values()
    bests = replay_bests(points, values)
    flat_best = int(np.argmin(values.ravel()))

    np.testing.assert_array_equal(result.loss_history, bests.gbest_val)
    assert result.best_val == values.ravel()[flat_best]
    # Bit for bit: the result maps the best internal position to physical
    # coordinates exactly like the point the objective was given.
    np.testing.assert_array_equal(result.best_pos, points.reshape(-1, D)[flat_best])


def test_ties_keep_the_earliest_best() -> None:
    recorder, result = record_unit_run(lambda points: np.zeros(len(points)), D, **RUN)

    np.testing.assert_array_equal(result.best_pos, recorder.batches[0][0])
    assert result.best_val == 0.0


def test_bounds_only_rescale_the_unit_box_search() -> None:
    lo = np.array([-3.0, 0.5, 100.0, -1e-3])
    hi = np.array([5.0, 0.75, 300.0, 1e-3])
    unit, _ = record_unit_run(evaluate_unit_rastrigin, D, **RUN)
    physical = SwarmRecorder(
        lambda points: evaluate_unit_rastrigin((points - lo) / (hi - lo))
    )
    gfo.pso(physical.evaluate, lo, hi, **RUN)

    np.testing.assert_allclose(
        (physical.stack_points() - lo) / (hi - lo),
        unit.stack_points(),
        rtol=0.0,
        atol=1e-12,
    )


def test_nan_values_never_become_the_best() -> None:
    def evaluate_with_holes(points: np.ndarray) -> np.ndarray:
        values = evaluate_unit_rastrigin(points)
        return np.where(points[:, 0] > 0.5, np.nan, values)

    recorder, result = record_unit_run(evaluate_with_holes, D, **RUN)
    values = recorder.stack_values()
    bests = replay_bests(recorder.stack_points(), values)

    assert np.isfinite(result.best_val)
    assert result.best_val == np.nanmin(values)
    np.testing.assert_array_equal(result.loss_history, bests.gbest_val)


def test_all_nan_objective_reports_no_best() -> None:
    # Current behaviour, pinned so a change is deliberate: nothing is ever
    # accepted as a best, so the result is +inf at the lower bounds.
    result = gfo.pso(
        lambda points: np.full(len(points), np.nan), -2.0, 3.0, d=D, **RUN
    )

    assert result.best_val == np.inf
    assert np.all(result.loss_history == np.inf)
    np.testing.assert_array_equal(result.best_pos, np.full(D, -2.0))
