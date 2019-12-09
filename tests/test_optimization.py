"""Tests for the optimizations library
"""

import pytest
import numpy as np
from autodiff import operations, optimizations
from autodiff.structures import Number, Array

def rosenbrock(x0, a=1, b=100):
    return (a - x0[0]) ** 2 + b * (x0[1] - x0[0] ** 2) ** 2

def bowl(x, xstar=np.array((1, 1))):
    return (x - xstar) @ (x - xstar)

def quadratic(x, xstar=1):
    return (x - xstar) ** 2

def gradient_rosenbrock(x0):
    x = x0[0]
    y = x0[1]
    drdx = -2 * (1 - x) - 400 * x * (-x ** 2 + y)
    drdy = 200 *(-x**2 + y)
    return drdx, drdy

def gradient_quadratic(x, xstar=1):
    return 2 * (x - xstar)

def test_bfgs():

    initial_guess = Array([Number(2),Number(1)])
    xstar, _, _ = optimizations.bfgs(rosenbrock, initial_guess)
    assert xstar[0].val == pytest.approx(1)
    assert xstar[1].val == pytest.approx(1)

def test_bfgs_correct_start():

    initial_guess = Array([Number(1),Number(1)])
    xstar, _, _ = optimizations.bfgs(rosenbrock, initial_guess)
    assert xstar[0].val == pytest.approx(1)
    assert xstar[1].val == pytest.approx(1)

def test_bfgs_symbolic():
    initial_guess = [2, 1]
    xstar, _, _ = optimizations.bfgs_symbolic(
        rosenbrock,
        gradient_rosenbrock,
        initial_guess
    )

    assert xstar[0] == pytest.approx(1)
    assert xstar[1] == pytest.approx(1)

def test_bfgs_symbolic_correct_start():
    initial_guess = [1, 1]
    xstar, _, _ = optimizations.bfgs_symbolic(
        rosenbrock,
        gradient_rosenbrock,
        initial_guess
    )

    assert xstar[0] == pytest.approx(1)
    assert xstar[1] == pytest.approx(1)

def test_bfgs_scalar():
    initial_guess = Number(2)
    xstar, _, _ = optimizations.bfgs(quadratic, initial_guess)
    assert xstar.val == pytest.approx(1)

def test_bfgs_scalar_correct_start():
    initial_guess = Number(1)
    xstar, _, _ = optimizations.bfgs(quadratic, initial_guess)
    assert xstar.val == pytest.approx(1)

def test_bfgs_verbose():

    initial_guess = Array([Number(2),Number(1)])
    xstar, _, _ = optimizations.bfgs(rosenbrock, initial_guess, verbose=True)
    assert xstar[0].val == pytest.approx(1)
    assert xstar[1].val == pytest.approx(1)

def test_bfgs_scalar_verbose():
    initial_guess = Number(2)
    xstar, _, _ = optimizations.bfgs(quadratic, initial_guess, verbose=True)
    assert xstar.val == pytest.approx(1)

def test_bfgs_symbolic_verbose():
    initial_guess = [2, 1]
    xstar, _, _ = optimizations.bfgs_symbolic(
        rosenbrock,
        gradient_rosenbrock,
        initial_guess,
        verbose=True
    )

    assert xstar[0] == pytest.approx(1)
    assert xstar[1] == pytest.approx(1)

def test_bfgs_symbolic_scalar():
    initial_guess = 2
    xstar, _, _ = optimizations.bfgs_symbolic(
        quadratic,
        gradient_quadratic,
        initial_guess
    )
    assert xstar == pytest.approx(1)

def test_bfgs_symbolic_scalar_correct_start():
    initial_guess = 1
    xstar, _, _ = optimizations.bfgs_symbolic(
        quadratic,
        gradient_quadratic,
        initial_guess
    )
    assert xstar == pytest.approx(1)

def test_steepest_descent():

    initial_guess = Array([Number(1.1),Number(1.1)])
    _, xstar, _, _ = optimizations.steepest_descent(bowl, initial_guess, iterations=400)
    # print(xstar)
    assert xstar[0].val == pytest.approx(1, abs=1e-3)
    assert xstar[1].val == pytest.approx(1, abs=1e-3)

def test_steepest_descent_armijo():
    """Should trigger the line search
    """
    initial_guess = Array([Number(2),Number(1)])
    _, xstar, _, _ = optimizations.steepest_descent(rosenbrock, initial_guess, iterations=50)

def test_steepest_descent_verbose_correct_start():

    initial_guess = Array([Number(1),Number(1)])
    _, xstar, _, _ = optimizations.steepest_descent(
        bowl,
        initial_guess,
        iterations=400,
        verbose=True
    )
    # print(xstar)
    assert xstar[0].val == pytest.approx(1, abs=1e-3)
    assert xstar[1].val == pytest.approx(1, abs=1e-3)

def test_steepest_descent_scalar():

    initial_guess = Number(1.1)
    xstar, _, _ = optimizations.steepest_descent(quadratic, initial_guess, iterations=400)
    print(xstar)
    assert xstar.val == pytest.approx(1, abs=1e-3)
