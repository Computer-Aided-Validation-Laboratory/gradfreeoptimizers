"""Shared tools for the PSO tests: a recording objective and swarm replays.

The replays rebuild what the optimizer must have done from the points it
evaluated, without its random numbers. They check the algorithm itself, so
they keep working when the random number generator changes (unlike the gold
tests).

Runs that are replayed use the unit box [0, 1]^d as bounds: the physical
points the objective gets are then exactly the optimizer's internal
positions, ``lo + u * (hi - lo) == u``.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import gfo

UNIT_LO = 0.0
UNIT_HI = 1.0
# Velocities rebuilt from consecutive positions in [0, 1] are off by a few ulp.
VELOCITY_ATOL = 1e-12
# A velocity this close (relative) to the clamp counts as clamped.
CLAMP_RTOL = 1e-9
# Multimodal test landscape on the unit box: Rastrigin with its minimum at
# x_k = 0.2 and a few ripples per dimension, so the swarm keeps moving.
UNIT_RASTRIGIN = gfo.problems.Rastrigin(shift=2.0, amplitude=10.0)
UNIT_RASTRIGIN_SCALE = 10.0


@dataclass(slots=True)
class SwarmRecorder:
    """Vectorized Python objective that records every batch it evaluates.

    Pass ``recorder.evaluate`` to ``gfo.pso``.

    Attributes
    ----------
    objective : callable
        The objective being recorded, points of shape (n, d) -> (n,) values.
    batches : list of np.ndarray
        Copy of the points of every call, each shape (n, d).
    batch_values : list of np.ndarray
        Copy of the values returned by every call, each shape (n,).
    """

    objective: Callable[[np.ndarray], np.ndarray]
    batches: list[np.ndarray] = field(default_factory=list)
    batch_values: list[np.ndarray] = field(default_factory=list)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate and record a batch, shape (n, d) -> (n,)."""
        values: np.ndarray = np.asarray(self.objective(points), dtype=float)
        self.batches.append(points.copy())
        self.batch_values.append(values.copy())
        return values

    def stack_points(self) -> np.ndarray:
        """Every recorded point, shape (iterations, n, d)."""
        return np.stack(self.batches)

    def stack_values(self) -> np.ndarray:
        """Every recorded value, shape (iterations, n)."""
        return np.stack(self.batch_values)


@dataclass(slots=True)
class SwarmBests:
    """Personal and global bests after every iteration of a replayed run.

    Attributes
    ----------
    pbest_pos : np.ndarray
        Personal best positions, shape (iterations, n, d).
    gbest_pos : np.ndarray
        Global best position, shape (iterations, d).
    gbest_val : np.ndarray
        Global best value, shape (iterations,).
    """

    pbest_pos: np.ndarray
    gbest_pos: np.ndarray
    gbest_val: np.ndarray


@dataclass(slots=True)
class SwarmSteps:
    """Every velocity update of a replayed run that can be checked.

    A step (iteration i >= 1, particle, dimension) is checkable when the
    velocity before it and after it are both known, i.e. the particle was
    not reflected at a wall in either iteration. All arrays have shape (m,).

    Attributes
    ----------
    velocity : np.ndarray
        Velocity after the update (clamped).
    previous_velocity : np.ndarray
        Velocity before the update.
    cognitive : np.ndarray
        Personal best minus position before the update.
    social : np.ndarray
        Global best minus position before the update.
    """

    velocity: np.ndarray
    previous_velocity: np.ndarray
    cognitive: np.ndarray
    social: np.ndarray


def evaluate_unit_rastrigin(points: np.ndarray) -> np.ndarray:
    """Multimodal landscape on the unit box, shape (n, d) -> (n,)."""
    return UNIT_RASTRIGIN.evaluate(UNIT_RASTRIGIN_SCALE * points)


def record_unit_run(
    objective: Callable[[np.ndarray], np.ndarray], d: int, **settings: float
) -> tuple[SwarmRecorder, gfo.PSOResult]:
    """Run ``gfo.pso`` on the unit box and record every evaluated batch."""
    recorder = SwarmRecorder(objective)
    result = gfo.pso(recorder.evaluate, UNIT_LO, UNIT_HI, d=d, **settings)
    return recorder, result


def calc_trajectory_hash(points: np.ndarray, values: np.ndarray) -> str:
    """SHA-256 of every evaluated point and value (little endian float64)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def replay_bests(points: np.ndarray, values: np.ndarray) -> SwarmBests:
    """Rebuild the personal and global bests the way the optimizer keeps them.

    A best only moves on a strict improvement (so NaN never becomes a best),
    the global best scans the particles in order, and both start at the
    origin with value +inf.

    Inputs
    ------
    points : np.ndarray
        Evaluated points, shape (iterations, n, d).
    values : np.ndarray
        Their values, shape (iterations, n).

    Returns
    -------
    SwarmBests
        The bests after every iteration.
    """
    n_iter, n_particles, dims = points.shape
    pbest_pos: np.ndarray = np.zeros((n_particles, dims))
    pbest_val: np.ndarray = np.full(n_particles, np.inf)
    gbest_pos: np.ndarray = np.zeros(dims)
    gbest_val: float = np.inf

    pbest_history: np.ndarray = np.empty((n_iter, n_particles, dims))
    gbest_history: np.ndarray = np.empty((n_iter, dims))
    gbest_val_history: np.ndarray = np.empty(n_iter)
    for ii in range(n_iter):
        improved: np.ndarray = values[ii] < pbest_val
        pbest_val = np.where(improved, values[ii], pbest_val)
        pbest_pos = np.where(improved[:, np.newaxis], points[ii], pbest_pos)
        for jj in range(n_particles):
            if pbest_val[jj] < gbest_val:
                gbest_val = float(pbest_val[jj])
                gbest_pos = pbest_pos[jj].copy()
        pbest_history[ii] = pbest_pos
        gbest_history[ii] = gbest_pos
        gbest_val_history[ii] = gbest_val
    return SwarmBests(pbest_history, gbest_history, gbest_val_history)


def calc_swarm_steps(
    points: np.ndarray, values: np.ndarray, *, starts_at_rest: bool
) -> SwarmSteps:
    """Collect every checkable velocity update of a run on the unit box.

    Inputs
    ------
    points : np.ndarray
        Evaluated points, shape (iterations, n, d), unit box coordinates.
    values : np.ndarray
        Their values, shape (iterations, n).
    starts_at_rest : bool
        True if the run used ``initial_velocity_rate=0``: the velocity after
        iteration 0 is then known to be 0, otherwise it is unknown.

    Returns
    -------
    SwarmSteps
        The checkable steps, flattened.
    """
    bests: SwarmBests = replay_bests(points, values)
    is_interior: np.ndarray = (points > UNIT_LO) & (points < UNIT_HI)
    # Velocity after each iteration, NaN where a wall reflection hides it.
    velocity: np.ndarray = np.full(points.shape, np.nan)
    if starts_at_rest:
        velocity[0] = 0.0
    velocity[1:] = np.where(is_interior[1:], points[1:] - points[:-1], np.nan)

    position_before: np.ndarray = points[:-1]
    cognitive: np.ndarray = bests.pbest_pos[:-1] - position_before
    social: np.ndarray = bests.gbest_pos[:-1, np.newaxis, :] - position_before
    after: np.ndarray = velocity[1:]
    before: np.ndarray = velocity[:-1]
    checkable: np.ndarray = np.isfinite(after) & np.isfinite(before)
    return SwarmSteps(
        velocity=after[checkable],
        previous_velocity=before[checkable],
        cognitive=cognitive[checkable],
        social=social[checkable],
    )


def find_velocity_violations(
    steps: SwarmSteps,
    *,
    inertia: float,
    acceleration1: float,
    acceleration2: float,
    max_velocity_rate: float,
) -> np.ndarray:
    """Flag the steps no pair of random numbers r1, r2 in [0, 1) explains.

    The update is ``v = clamp(w v_prev + c1 r1 cognitive + c2 r2 social,
    -vmax, vmax)``. An unclamped step must lie in the interval spanned by r1
    and r2, a clamped one must have an unclamped value that can reach the
    clamp.

    Returns
    -------
    np.ndarray
        True for every violating step, shape (m,).
    """
    inertia_term: np.ndarray = inertia * steps.previous_velocity
    cognitive_term: np.ndarray = acceleration1 * steps.cognitive
    social_term: np.ndarray = acceleration2 * steps.social
    lowest: np.ndarray = (
        inertia_term + np.minimum(cognitive_term, 0.0) + np.minimum(social_term, 0.0)
    )
    highest: np.ndarray = (
        inertia_term + np.maximum(cognitive_term, 0.0) + np.maximum(social_term, 0.0)
    )

    speed: np.ndarray = np.abs(steps.velocity)
    is_clamped: np.ndarray = speed >= max_velocity_rate * (1.0 - CLAMP_RTOL)
    is_too_fast: np.ndarray = speed > max_velocity_rate + VELOCITY_ATOL
    is_free_ok: np.ndarray = (steps.velocity >= lowest - VELOCITY_ATOL) & (
        steps.velocity <= highest + VELOCITY_ATOL
    )
    is_clamp_ok: np.ndarray = np.where(
        steps.velocity > 0.0,
        highest >= max_velocity_rate - VELOCITY_ATOL,
        lowest <= -max_velocity_rate + VELOCITY_ATOL,
    )
    return is_too_fast | np.where(is_clamped, ~is_clamp_ok, ~is_free_ok)


def calc_ks_uniform(samples: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between samples and U(0, 1)."""
    ordered: np.ndarray = np.sort(samples)
    count: int = ordered.size
    above: np.ndarray = np.arange(1, count + 1) / count - ordered
    below: np.ndarray = ordered - np.arange(count) / count
    return float(max(above.max(), below.max()))
