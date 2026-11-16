"""The Python example problems must match their C++ twins exactly."""

import numpy as np
import pytest

import gfo

pytestmark = pytest.mark.skipif(not gfo.EXTENSION_AVAILABLE, reason="C++ extension not built")

NAMES = list(gfo.problems.EXAMPLE_PROBLEMS)
DIMS = (1, 2, 10, 30)
RTOL = 1e-12
N_POINTS = 128


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_python_matches_cpp(name: str, d: int) -> None:
    py_problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    cpp_problem = getattr(gfo.cpp_problems, name)()
    points = np.random.default_rng(d).uniform(py_problem.lo, py_problem.hi, size=(N_POINTS, d))
    np.testing.assert_allclose(py_problem(points), cpp_problem(points), rtol=RTOL, atol=RTOL)


@pytest.mark.parametrize("name", NAMES)
def test_same_defaults(name: str) -> None:
    py_problem = gfo.problems.EXAMPLE_PROBLEMS[name]()
    cpp_problem = getattr(gfo.cpp_problems, name)()
    assert py_problem.name == cpp_problem.name == name
    assert (py_problem.lo, py_problem.hi) == (cpp_problem.lo, cpp_problem.hi)
    for field_name in cpp_problem.fields:
        assert getattr(py_problem, field_name) == pytest.approx(getattr(cpp_problem, field_name))


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_known_minimum(name: str, d: int) -> None:
    for problem in (gfo.problems.EXAMPLE_PROBLEMS[name](), getattr(gfo.cpp_problems, name)()):
        assert problem(problem.x_opt(d)) == pytest.approx(problem.f_opt(), abs=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_single_point_and_batch_agree(name: str) -> None:
    for problem in (gfo.problems.EXAMPLE_PROBLEMS[name](), getattr(gfo.cpp_problems, name)()):
        points = np.random.default_rng(1).uniform(problem.lo, problem.hi, size=(5, 3))
        batch = problem(points)
        assert isinstance(problem(points[0]), float)
        np.testing.assert_array_equal(batch, [problem(row) for row in points])


def test_parameters_change_the_problem() -> None:
    py_problem = gfo.problems.Rastrigin(shift=-1.0, amplitude=5.0)
    cpp_problem = gfo.cpp_problems.Rastrigin(shift=-1.0, amplitude=5.0)
    assert py_problem(np.full(4, -1.0)) == pytest.approx(0.0)
    assert cpp_problem(np.full(4, -1.0)) == pytest.approx(0.0)
    point = np.array([0.3, -2.0, 1.7])
    assert py_problem(point) == pytest.approx(cpp_problem(point), rel=RTOL)

    cpp_problem.shift = 0.5
    assert cpp_problem(np.full(2, 0.5)) == pytest.approx(0.0)
    assert "shift=0.5" in repr(cpp_problem)
