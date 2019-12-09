"""Tests for the root-finding module
"""

import sys
import pytest
sys.path.append('..')

import pytest
import numpy as np
from autodiff import operations, root_finding
from autodiff.structures import Number, Array

# def func_array(x):
#     return x[0] ** 2 * (x[1] + 2)

def func_bowl(x):
    return (x[0] - 1) ** 2 + (x[1] - 1) ** 2

def func_scalar(x):
    return 2 * (x - 1) ** 2

def test_newtons_method_vector():
    initial_guess = Array((Number(-0.1), Number(-1)))
    xstar, _ = root_finding.newtons_method(func_bowl, initial_guess, tolerance=1e-6)
    assert xstar[0].val == pytest.approx(1, abs=1e-3)
    assert xstar[1].val == pytest.approx(1, abs=1e-3)

def test_newtons_method_scalar():
    initial_guess = Number(2)
    xstar, _ = root_finding.newtons_method(func_scalar, initial_guess)
    assert xstar.val == pytest.approx(1, abs=1e-3)

def test_newtons_method_vector_show_fxn():
    initial_guess = Array((Number(-0.1), Number(-1)))
    xstar, _, _ = root_finding.newtons_method(func_bowl, initial_guess, tolerance=1e-6, show_fxn=True)
    assert xstar[0].val == pytest.approx(1, abs=1e-3)
    assert xstar[1].val == pytest.approx(1, abs=1e-3)

def test_newtons_method_scalar_show_fxn():
    initial_guess = Number(2)
    xstar, _, _ = root_finding.newtons_method(func_scalar, initial_guess, show_fxn=True)
    assert xstar.val == pytest.approx(1, abs=1e-3)

def test_newtons_method_vector_verbose():
    initial_guess = Array((Number(-0.1), Number(-1)))
    xstar, _ = root_finding.newtons_method(func_bowl, initial_guess, tolerance=1e-6, verbose=True)
    assert xstar[0].val == pytest.approx(1, abs=1e-3)
    assert xstar[1].val == pytest.approx(1, abs=1e-3)

def test_newtons_method_scalar_verbose():
    initial_guess = Number(2)
    xstar, _ = root_finding.newtons_method(func_scalar, initial_guess, verbose=True)
    assert xstar.val == pytest.approx(1, abs=1e-3)