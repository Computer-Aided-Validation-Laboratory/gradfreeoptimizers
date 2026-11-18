"""The example problems: correct formulas, and Python matching C++.

Python matching C++ alone cannot catch a mistake made in both, so the
formulas are also checked against values computed by hand and against plain
loop implementations of the textbook definitions written here.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import gfo
from gfo.problems import IProblem

PROBLEM_NAMES = list(gfo.problems.EXAMPLE_PROBLEMS)
DIMS = (1, 2, 10, 30)
N_POINTS = 128
# Both versions use the same operation order, only libm differences remain.
RTOL = 1e-12
# Schwefel's constant is only exact to about 1e-11 per dimension.
OPTIMUM_ATOL = 1e-8
BOUND_FIELDS = ("lo", "hi")
# Literature value of Schwefel's constant (418.9829 d), written out here
# rather than imported so a typo in gfo's copy would show.
SCHWEFEL_CONSTANT = 418.9828872724338


def make_problem_pair(name: str, **params: float) -> tuple[IProblem, gfo.CppProblem]:
    """Build the Python and the C++ version of a problem."""
    return (
        gfo.problems.EXAMPLE_PROBLEMS[name](**params),
        getattr(gfo.cpp_problems, name)(**params),
    )


def make_random_points(problem: IProblem, d: int, seed: int) -> np.ndarray:
    """N_POINTS uniform random points in the problem's default domain."""
    return np.random.default_rng(seed).uniform(problem.lo, problem.hi, (N_POINTS, d))


def evaluate_textbook(name: str, x: np.ndarray, params: dict[str, float]) -> float:
    """Textbook definition of every example problem, one point, plain loops."""
    z = [xk - params.get("shift", 0.0) for xk in x]
    d = len(z)
    if name == "Sphere":
        return sum(zk * zk for zk in z)
    if name == "Ellipsoid":
        return sum((k + 1) * zk * zk for k, zk in enumerate(z))
    if name == "Rosenbrock":
        a, b = params["a"], params["b"]
        return sum(
            b * (z[k + 1] - z[k] ** 2) ** 2 + (a - z[k]) ** 2 for k in range(d - 1)
        )
    if name == "Rastrigin":
        amp = params["amplitude"]
        return amp * d + sum(zk * zk - amp * math.cos(2 * math.pi * zk) for zk in z)
    if name == "Ackley":
        a, b, c = params["a"], params["b"], params["c"]
        mean_square = sum(zk * zk for zk in z) / d
        mean_cos = sum(math.cos(c * zk) for zk in z) / d
        envelope = -a * math.exp(-b * math.sqrt(mean_square))
        return envelope - math.exp(mean_cos) + a + math.e
    if name == "Griewank":
        quadratic = sum(zk * zk for zk in z) / params["scale"]
        product = math.prod(math.cos(zk / math.sqrt(k + 1)) for k, zk in enumerate(z))
        return 1.0 + quadratic - product
    if name == "Schwefel":
        wave = sum(zk * math.sin(math.sqrt(abs(zk))) for zk in z)
        return SCHWEFEL_CONSTANT * d - wave
    raise ValueError(f"No textbook definition for {name}")


# Values worked out by hand for the standard (unshifted) functions.
HAND_VALUES = [
    ("Sphere", {"shift": 0.0}, [1.0, 2.0, 3.0], 14.0),
    ("Ellipsoid", {"shift": 0.0}, [1.0, 1.0, 1.0], 6.0),
    ("Rosenbrock", {"shift": 0.0}, [0.0, 0.0], 1.0),
    ("Rosenbrock", {"shift": 0.0}, [-1.0, 1.0], 4.0),
    ("Rosenbrock", {"shift": 0.0}, [1.0, 1.0, 1.0], 0.0),
    ("Rastrigin", {"shift": 0.0}, [1.0, 1.0], 2.0),
    ("Rastrigin", {"shift": 0.0}, [0.5], 20.25),
    ("Ackley", {"shift": 0.0}, [0.0, 0.0], 0.0),
    # -20 exp(-0.2) - exp(cos 2 pi) + 20 + e = 20 (1 - exp(-0.2)).
    ("Ackley", {"shift": 0.0}, [1.0, 1.0], 20.0 * (1.0 - math.exp(-0.2))),
    ("Griewank", {"shift": 0.0}, [0.0, 0.0, 0.0], 0.0),
    # cos(pi / 1) cos(pi sqrt 2 / sqrt 2) = 1, leaves (pi^2 + 2 pi^2) / 4000.
    (
        "Griewank",
        {"shift": 0.0},
        [math.pi, math.pi * math.sqrt(2.0)],
        3.0 * math.pi**2 / 4000.0,
    ),
    ("Schwefel", {"shift": 0.0}, [0.0, 0.0], 2.0 * SCHWEFEL_CONSTANT),
]


@pytest.mark.parametrize(
    ("name", "params", "point", "expected"),
    HAND_VALUES,
    ids=[f"{case[0]}-{case[2]}" for case in HAND_VALUES],
)
def test_values_worked_out_by_hand(
    name: str, params: dict[str, float], point: list[float], expected: float
) -> None:
    for problem in make_problem_pair(name, **params):
        value = problem.evaluate(np.array([point]))[0]

        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("d", (1, 3, 8))
def test_matches_the_textbook_definition(name: str, d: int) -> None:
    py_problem, cpp_problem = make_problem_pair(name)
    params = {field: getattr(cpp_problem, field) for field in cpp_problem.fields}
    points = make_random_points(py_problem, d, seed=d)
    expected = [evaluate_textbook(name, point, params) for point in points]

    np.testing.assert_allclose(
        py_problem.evaluate(points), expected, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_python_matches_cpp(name: str, d: int) -> None:
    py_problem, cpp_problem = make_problem_pair(name)
    points = make_random_points(py_problem, d, seed=d)

    np.testing.assert_allclose(
        py_problem.evaluate(points), cpp_problem.evaluate(points), rtol=RTOL, atol=RTOL
    )


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_python_and_cpp_have_the_same_parameters(name: str) -> None:
    py_problem, cpp_problem = make_problem_pair(name)

    assert py_problem.name == cpp_problem.name == name
    assert py_problem.f_opt == cpp_problem.f_opt
    for field_name in cpp_problem.fields:
        py_value = getattr(py_problem, field_name)
        assert py_value == pytest.approx(getattr(cpp_problem, field_name))


def calc_changed_params(name: str) -> dict[str, float]:
    """Every non-bound parameter of a problem moved away from its default."""
    _, cpp_problem = make_problem_pair(name)
    return {
        field: 1.5 * getattr(cpp_problem, field) + 0.25
        for field in cpp_problem.fields
        if field not in BOUND_FIELDS
    }


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_every_parameter_reaches_both_versions(name: str) -> None:
    for field, value in calc_changed_params(name).items():
        default_py, _ = make_problem_pair(name)
        py_problem, cpp_problem = make_problem_pair(name, **{field: value})
        points = make_random_points(py_problem, 4, seed=1)
        params = {key: getattr(cpp_problem, key) for key in cpp_problem.fields}
        expected = [evaluate_textbook(name, point, params) for point in points]

        assert not np.allclose(py_problem.evaluate(points), default_py.evaluate(points))
        np.testing.assert_allclose(
            cpp_problem.evaluate(points), py_problem.evaluate(points), rtol=RTOL
        )
        np.testing.assert_allclose(py_problem.evaluate(points), expected, rtol=1e-10)


def test_cpp_parameters_are_writable() -> None:
    cpp_problem = gfo.cpp_problems.Rastrigin()
    cpp_problem.shift = 0.5
    cpp_problem.amplitude = 3.0

    assert cpp_problem.evaluate(np.full((1, 2), 0.5))[0] == pytest.approx(0.0)
    # z = 0.5: 0.25 - 3 cos(pi) + 3.
    assert cpp_problem.evaluate(np.full((1, 1), 1.0))[0] == pytest.approx(6.25)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_known_minimum(name: str, d: int) -> None:
    for problem in make_problem_pair(name):
        optimum = problem.calc_x_opt(d)

        assert optimum.shape == (d,)
        assert np.all((optimum >= problem.lo) & (optimum <= problem.hi))
        assert problem.evaluate(optimum[np.newaxis, :])[0] == pytest.approx(
            problem.f_opt, abs=OPTIMUM_ATOL
        )


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_known_minimum_is_the_lowest_value(name: str) -> None:
    for problem in make_problem_pair(name):
        points = np.random.default_rng(0).uniform(problem.lo, problem.hi, (20000, 5))

        assert np.all(problem.evaluate(points) >= problem.f_opt - OPTIMUM_ATOL)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_known_minimum_is_a_strict_local_minimum(name: str) -> None:
    py_problem, _ = make_problem_pair(name)
    d = 4
    rng = np.random.default_rng(1)
    directions = rng.normal(size=(500, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    width = py_problem.hi - py_problem.lo
    nearby = py_problem.calc_x_opt(d) + 1e-4 * width * directions

    assert np.all(py_problem.evaluate(nearby) > py_problem.f_opt + OPTIMUM_ATOL)


SYMMETRIC_UNDER_PERMUTATION = ("Sphere", "Rastrigin", "Ackley", "Schwefel")
SYMMETRIC_UNDER_MIRROR = ("Sphere", "Ellipsoid", "Rastrigin", "Ackley", "Griewank")


@pytest.mark.parametrize("name", SYMMETRIC_UNDER_PERMUTATION)
def test_coordinates_are_interchangeable(name: str) -> None:
    for problem in make_problem_pair(name):
        points = make_random_points(problem, 6, seed=2)
        shuffled = points[:, np.random.default_rng(3).permutation(6)]

        np.testing.assert_allclose(
            problem.evaluate(shuffled), problem.evaluate(points), rtol=1e-12
        )


@pytest.mark.parametrize("name", SYMMETRIC_UNDER_MIRROR)
def test_mirror_image_through_the_minimum_has_the_same_value(name: str) -> None:
    for problem in make_problem_pair(name):
        centre = problem.calc_x_opt(3)
        offsets = np.random.default_rng(4).uniform(-1.0, 1.0, (N_POINTS, 3))

        np.testing.assert_allclose(
            problem.evaluate(centre + offsets),
            problem.evaluate(centre - offsets),
            rtol=1e-12,
        )


INPUT_LAYOUTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "integers": lambda points: np.round(points).astype(np.int64),
    "float32": lambda points: points.astype(np.float32),
    "fortran_order": np.asfortranarray,
    "strided": lambda points: np.repeat(points, 2, axis=1)[:, ::2],
    "list": lambda points: points.tolist(),
}


@pytest.mark.parametrize("layout", list(INPUT_LAYOUTS))
@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_any_array_layout_gives_the_same_values(name: str, layout: str) -> None:
    for problem in make_problem_pair(name):
        points = make_random_points(problem, 3, seed=5)
        converted = INPUT_LAYOUTS[layout](points)
        expected = problem.evaluate(np.array(converted, dtype=float))
        values = problem.evaluate(converted)

        assert values.shape == (N_POINTS,)
        assert values.dtype == np.float64
        np.testing.assert_allclose(values, expected, rtol=1e-15)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_evaluate_does_not_modify_its_input(name: str) -> None:
    for problem in make_problem_pair(name):
        points = make_random_points(problem, 3, seed=6)
        before = points.copy()
        problem.evaluate(points)

        np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_empty_batch_gives_no_values(name: str) -> None:
    for problem in make_problem_pair(name):
        assert problem.evaluate(np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("shape", [(3,), (2, 3, 4)], ids=["1d", "3d"])
def test_points_must_be_two_dimensional(name: str, shape: tuple[int, ...]) -> None:
    for problem in make_problem_pair(name):
        with pytest.raises(ValueError, match="shape"):
            problem.evaluate(np.zeros(shape))
