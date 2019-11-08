import sys
import pytest
sys.path.append('..')

from autodiff import operations
from autodiff.structures import Number
import numpy as np

def test_returns_number():
    a = Number(np.pi / 2)
    sina = operations.sin(a)
    assert isinstance(sina, Number)

def test_repr():
    a = Number(1.0)
    assert repr(a) == 'Number(val=1.0)'

def test_sin():
    a = Number(np.pi / 2)
    sina = operations.sin(a)
    assert sina.val == pytest.approx(1)

def test_sin_deriv():
    a = Number(np.pi / 2)
    sina = operations.sin(a)
    assert sina.deriv[a] == pytest.approx(0)