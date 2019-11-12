import sys
import pytest
sys.path.append('..')

import autodiff.Number as number
from autodiff.Number import Number
import numpy as np

# Useful numbers for tests
a = Number(np.pi / 2)
num_log_1 = Number(np.log(1))
num_e = Number(np.exp(1))
sina = number.sin(a)
cosa = number.cos(a)

# Some integers
# Test values for overloaded operations
num2 = Number(2)
num3 = Number(3)
num4 = Number(4)

def test_returns_number():
    a = Number(np.pi / 2)
    sina = number.sin(a)
    assert isinstance(sina, Number)

def test_repr():
    a = Number(1.0)
    assert repr(a) == 'Number(val=1.0)'

def test_self_deriv():
    a = Number(2)
    assert a.deriv[a] == 1

def test_provided_deriv():
    a = Number(3, 4)
    assert a.deriv[a] == 4



def test_sin():
    assert sina.val == pytest.approx(1)

def test_sin_deriv():
    assert sina.deriv[a] == pytest.approx(0)

def test_cos():    
    assert cosa.val == pytest.approx(0)

def cos_deriv():
    assert cosa.deriv[a] == pytest.approx(-1)

def test_add():
    assert (num2 + num3).val == 5

def test_add_first_deriv():
    assert (num2 + num3).deriv[num2] == 1

def test_add_second_deriv():
    assert (num2 + num3).deriv[num3] == 1

def test_mixed_add():
    assert (num2 + 3).val == 5

def test_mixed_radd():
    assert (3 + num2).val == 5

def test_mixed_add_returns_number():
    assert isinstance(num3 + 2, Number)

def test_mixed_radd_returns_number():
    assert isinstance(3 + num2, Number)

def test_mixed_add_deriv():
    assert (num2 + num3).deriv[num2] == 1

def test_mixed_add_deriv_number_only():
    """Test that adding a Number() to an int() only stores a partial derivative for the Number()
    """
    with pytest.raises(KeyError):
        (num2 + 3).deriv[3]

def test_sub():
    assert (num3 - num2).val == 1

def test_sub_deriv():
    assert (num3 - num2).deriv[num3] == 1

def test_sub_deriv():
    assert (num3 - num2).deriv[num2] == -1

def test_mixed_sub():
    assert (num3 - 2).val == 1

def test_mixed_rsub():
    assert (3 - num2).val == 1

def test_mixed_subtract_deriv_number_only():
    with pytest.raises(KeyError):
        (num3 - 2).deriv[2]

def test_mul():
    assert (num3 * num2).val == 6

def test_mul_returns_number():
    assert isinstance(num3 * num2, Number)

def test_mixed_mul():
    assert (num2 * 3).val == 6

def test_mixed_rmul():
    assert (3 * num2).val == 6

def test_mixed_rmul_deriv_number_only():
    with pytest.raises(KeyError):
        (3 * num2).deriv[3]

def test_mul_deriv_first():
    assert (num2 * num3).deriv[num2] == num3

def test_mul_deriv_second():
    assert (num2 * num3).deriv[num3] == num2

def test_mixed_mul_deriv():
    assert (3 * num2).deriv[num2] == 3

def test_div():
    assert (num4 / num2).val == 2

def test_div_deriv_first():
    assert (num4 / num2).deriv[num4] == pytest.approx(1 / 2)

def test_div_deriv_second():
    assert (num4 / num2).deriv[num2] == -1

def test_mixed_div():
    assert (num4 / 2).val == 2

def test_mixed_div_deriv_num_only():
    with pytest.raises(KeyError):
        (num4 / 2).deriv[2]

def test_mixed_rdiv():
    assert (4 / num2).val == 2

def test_mixed_rdiv_deriv():
    assert (4 / num2).deriv[num2] == -1

def test_pow():
    assert (num4 ** num2).val == 16
    
def test_pow_deriv_first():
    assert (num4 ** num2).deriv[num2] == 8

def test_pow_deriv_second():
    assert (num4 ** num2).deriv[num2] == pytest.approx(4 ** 2 * np.log(4))

def test_mixed_pow():
    assert (num4 ** 2).val == 16

def test_mixed_rpow():
    assert (4 ** num2).val == 16

def test_mixed_rpow_deriv_number_only():
    with pytest.raises(KeyError):
        (4 ** num2).deriv[4]

def test_exp():
    assert number.exp(num_log_1).val == 1

def test_exp_deriv():
    assert number.exp(num_log_1).deriv[num_log_1] == 1

def test_log():
    assert number.log(num_e).val == 1

def test_log_deriv():
    assert number.log(num_e).deriv[num_e] == np.exp(-1)

def test_negate():
    a = Number(1)
    assert (-a).val == -1

def test_negate_deriv():
    a = Number(1)
    assert (-a).deriv[a] == -1

def test_negate_multiple_partials():
    derivs = {
        num2: 4,
        num3: 5,
        num4: 6
    }

    # A number with partial derivatives w.r.t. 2, 3, and 4
    a = Number(2, derivs)
    nega = -a

    assert -a.deriv[num2] == -4
    assert -a.deriv[num3] == -5
    assert -a.deriv[num4] == -6


def test_duplicate_value():
    out = num2 + num3
    new_3 = Number(3)
    with pytest.raises(KeyError):
        out.deriv[new_3]

if __name__ == '__main__':
    print(num2 ** num4)