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

def test_str():
    q = Array((num2, num3))
    assert str(q) == '[Number(val=2) Number(val=3)]'

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

def test_add_scalar():
    q = Array((num2, num3))
    w = q + 1
    assert w[0].val == 3
    assert w[1].val == 4

def test_radd_scalar():
    q = Array((num2, num3))
    w = 2 + q
    assert w[0].val == 4
    assert w[1].val == 5

def test_sub_array():
    q = Array((num2, num3))
    w = q - q
    assert w[0].val == 0
    assert w[1].val == 0

def test_rsub_scalar():
    q = Array((num2, num3))
    w = 10 - q
    assert w[0].val == 8
    assert w[1].val == 7

def test_sub_scalar():
    q = Array((num2, num3))
    w = q - 1
    assert w[0].val == 1
    assert w[1].val == 2

def test_mul_array():
    q = Array((num2, num3))
    w = q * q
    assert w[0].val == 4
    assert w[1].val == 9

def test_mul_scalar():
    q = Array((num2, num3))
    w = q * 2
    assert w[0].val == 4
    assert w[1].val == 6

def test_rmul_scalar():
    q = Array((num2, num3))
    w = 2 * q
    assert w[0].val == 4
    assert w[1].val == 6

def test_matmul_dot():
    q = Array((num2, num3))
    w = q @ q
    assert w.val == 13

def test_matmul_np_array():
    q = Array((num2, num3))
    v = np.array((2, 3))
    assert (q @ v).val == 13

def test_rmatmul_np_array():
    q = Array((num2, num3))
    # v = np.array((2, 3))
    v = [2, 3]
    assert (v @ q).val == 13

def test_matmul_matrix():
    q = Array((num2, num3))
    w = np.eye(2)
    v = q @ w

def test_div_array():
    q = Array((num2, num3))
    w = q / q
    assert w[0].val == pytest.approx(1)
    assert w[1].val == pytest.approx(1)

def test_div_scalar():
    q = Array((num2, num3))
    w = q / 2
    assert w[0].val == pytest.approx(1)
    assert w[1].val == pytest.approx(3 / 2)

def test_rdiv():
    q = Array((num2, num3))
    w = 12 / q
    assert w[0].val == pytest.approx(6)
    assert w[1].val == pytest.approx(4)

def test_array_func():
    q = Array((
        Number(0),
        Number(1)
    ))

    w = operations.exp(q)
    assert w[0].val == pytest.approx(1)
    assert w[1].val == pytest.approx(np.exp(1))

def test_array_only_number():
    q = Array((num2, num2))
    with pytest.raises(ValueError):
        q[0] = 1

def test_pow():
    q = Array((num2, num3))
    w = q ** 2
    assert w[0].val == 4
    assert w[1].val == 9

def test_rpow():
    q = Array((num2, num3))
    v = [2, 2]
    w = v ** q
    assert w[0].val == 4
    assert w[1].val == 8

def test_neg():
    q = Array((num2, num3))
    w = -q
    assert w[0].val == -2
    assert w[1].val == -3

def test_jacobian_self():
    q = Array((num2, num3))
    jac = q.jacobian(q._data)
    assert jac[0, 0] == 1
    assert jac[0, 1] == 0
    assert jac[1, 0] == 0
    assert jac[1, 1] == 1

def test_jacobian_scalar():
    q = Array((num2, num3))

def test_array_all_numbers():
    with pytest.raises(ValueError):
        Array((num3, num2, 1))

if __name__ == '__main__':
    # print(repr(Array((num2, num3))))
    # q = Array((num2, num3))
    # q @ q
    # print((q @ q)[()].val)
    # test_matmul_dot()
    test_jacobian_self()