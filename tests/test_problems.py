"""The Python example problems must match their C++ twins."""

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


def make_problem_pair(name: str) -> tuple[IProblem, gfo.CppProblem]:
    """Build the Python and the C++ version of a problem with default values."""
    return gfo.problems.EXAMPLE_PROBLEMS[name](), getattr(gfo.cpp_problems, name)()


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_python_matches_cpp(name: str, d: int) -> None:
    py_problem, cpp_problem = make_problem_pair(name)
    points = np.random.default_rng(d).uniform(
        py_problem.lo, py_problem.hi, size=(N_POINTS, d)
    )

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


@pytest.mark.parametrize("name", PROBLEM_NAMES)
@pytest.mark.parametrize("d", DIMS)
def test_known_minimum(name: str, d: int) -> None:
    for problem in make_problem_pair(name):
        optimum = problem.calc_x_opt(d)[np.newaxis, :]

        assert problem.evaluate(optimum)[0] == pytest.approx(
            problem.f_opt, abs=OPTIMUM_ATOL
        )


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_single_point_without_batch_axis_raises(name: str) -> None:
    for problem in make_problem_pair(name):
        with pytest.raises(ValueError, match="shape"):
            problem.evaluate(np.zeros(3))


def test_parameters_move_the_minimum() -> None:
    py_problem = gfo.problems.Rastrigin(shift=-1.0, amplitude=5.0)
    cpp_problem = gfo.cpp_problems.Rastrigin(shift=-1.0, amplitude=5.0)
    minimum = np.full((1, 4), -1.0)

    assert py_problem.evaluate(minimum)[0] == pytest.approx(0.0)
    assert cpp_problem.evaluate(minimum)[0] == pytest.approx(0.0)

    cpp_problem.shift = 0.5
    assert cpp_problem.evaluate(np.full((1, 2), 0.5))[0] == pytest.approx(0.0)
