"""Tests for the Array class
"""

import sys
import pytest
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
from autodiff.structures import Array
import numpy as np

num2 = Number(2)
num3 = Number(3)


def test_len():
    q = Array((num2, num3))
    assert len(q) == 2

def test_repr():
    q = Array((num2, num3))
    assert repr(q) == 'Array([Number(val=2) Number(val=3)])'

def test_indexing():
    q = Array((num2, num3))
    assert q[0] == num2

def test_iter():
    q = Array((num2, num3))
    q_ = []
    for el in q:
        q_.append(el)

    assert q_ == [num2, num3]

def test_setitem():
    q = Array((num2, num3))
    q[0] = num2
    assert q[0] == num2

def test_add():
    q = Array((num2, num3))
    w = q + q
    assert w[0].val == 4
    assert w[1].val == 6

def test_sub_array():
    q = Array((num2, num3))
    w = q - q
    assert w[0].val == 0
    assert w[1].val == 0

def test_sub_scalar():
    q = Array((num2, num3))
    w = 10 - q
    assert w[0].val == 8
    assert w[1].val == 7

def test_mul_array():
    q = Array((num2, num3))
    w = q * q
    assert w[0].val == 4
    assert w[1].val == 9

def test_mul_scalar():
    q = Array((num2, num3))
    w = 2 * q
    assert w[0].val == 4
    assert w[1].val == 6

def test_matmul_dot():
    q = Array((num2, num3))
    w = q @ q
    assert w.val == 13

def test_div_array():
    q = Array((num2, num3))
    w = q / q
    assert w[0].val == pytest.approx(1)
    assert w[1].val == pytest.approx(1)

def test_array_func():
    q = Array((
        Number(0),
        Number(1)
    ))

    w = operations.exp(q)
    assert w[0].val == pytest.approx(1)
    assert w[1].val == pytest.approx(np.exp(1))

if __name__ == '__main__':
    # print(repr(Array((num2, num3))))
    q = Array((num2, num3))
    print((q @ q)[()].val)