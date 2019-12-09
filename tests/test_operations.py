import sys
import pytest
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
import numpy as np

# Useful numbers for tests
a = Number(np.pi / 2)
num_log_1 = Number(np.log(1))
num_e = Number(np.exp(1))
sina = operations.sin(a)
cosa = operations.cos(a)

# Some integers
# Test values for overloaded operations
num1 = Number(1)
num2 = Number(2)
num3 = Number(3)
num4 = Number(4)

def test_returns_number():
    a = Number(np.pi / 2)
    sina = operations.sin(a)
    assert isinstance(sina, Number)

def test_repr():
    a = Number(1.0)
    assert repr(a) == 'Number(val=1.0)'

def test_self_deriv():
    a = Number(2)
    assert a.jacobian(a) == 1

def test_provided_deriv():
    a = Number(3, 4)
    assert a.jacobian(a) == 4

def test_sin():
    assert sina.val == pytest.approx(1)

def test_sin_method():
    assert a.sin().val == pytest.approx(1)

def test_sin_deriv():
    assert sina.jacobian(a) == pytest.approx(0)

def test_cos():    
    assert cosa.val == pytest.approx(0)

def test_cos_method():
    assert a.cos().val == pytest.approx(0)

def test_cos_deriv():
    assert cosa.jacobian(a) == pytest.approx(-1)

def test_tan():
    a = Number(np.pi / 4)
    tana = operations.tan(a)
    assert tana.val == pytest.approx(1)

def test_tan_method():
    a = Number(np.pi / 4)
    assert a.tan().val == pytest.approx(1)

def test_tan_deriv():
    a = Number(np.pi / 4)
    tana = operations.tan(a)
    assert tana.jacobian(a) == pytest.approx(2)

def test_add():
    assert (num2 + num3).val == 5

def test_add_shared_partial():
    """Test the case where both numbers in an __add__ share a partial derivative
    """
    t = Number(10)
    a = Number(2, deriv={t: 2})
    b = Number(3, deriv={t: 4})
    aplusb = a + b
    assert aplusb.jacobian(t) == 6
    assert aplusb.val == 5
    assert aplusb.jacobian(a) == 1
    assert aplusb.jacobian(b) == 1

def test_add_same_number():
    assert (num2 + num2).jacobian(num2) == 2

def test_add_first_deriv():
    assert (num2 + num3).jacobian(num2) == 1

def test_add_second_deriv():
    assert (num2 + num3).jacobian(num3) == 1

def test_mixed_add():
    assert (num2 + 3).val == 5

def test_mixed_radd():
    assert (3 + num2).val == 5

def test_mixed_add_returns_number():
    assert isinstance(num3 + 2, Number)

def test_mixed_radd_returns_number():
    assert isinstance(3 + num2, Number)

def test_mixed_add_deriv():
    assert (num2 + num3).jacobian(num2) == 1

def test_mixed_add_deriv_number_only():
    """Test that adding a Number() to an int() only stores a partial derivative for the Number()
    """
    with pytest.raises(KeyError):
        (num2 + 3)._deriv[3]

def test_sub():
    assert (num3 - num2).val == 1

def test_sub_deriv_first():
    assert (num3 - num2).jacobian(num3) == 1

def test_sub_deriv_second():
    assert (num3 - num2).jacobian(num2) == -1

def test_mixed_sub():
    assert (num3 - 2).val == 1

def test_mixed_rsub():
    assert (3 - num2).val == 1

def test_sub_shared_partial():
    """Test the case where both numbers in an __sub__ share a partial derivative
    """
    t = Number(10)
    a = Number(2, deriv={t: 2})
    b = Number(3, deriv={t: 4})
    aminusb = a - b
    assert aminusb.jacobian(t) == -2
    assert aminusb.val == -1
    assert aminusb.jacobian(a) == 1
    assert aminusb.jacobian(b) == -1

def test_sub_same_number():
    assert (num2 - num2).jacobian(num2) == 0

def test_mixed_subtract_deriv_number_only():
    with pytest.raises(KeyError):
        (num3 - 2)._deriv[2]

def test_mul():
    assert (num3 * num2).val == 6

def test_mul_returns_number():
    assert isinstance(num3 * num2, Number)

def test_mul_shared_partial():
    """Test the case where both numbers in an __mul__ share a partial derivative
    """
    t = Number(10)
    a = Number(2, deriv={t: 2})
    b = Number(3, deriv={t: 4})

    atimesb = a * b
    assert atimesb.jacobian(t) == 14
    assert atimesb.val == 6
    assert atimesb.jacobian(a) == 3
    assert atimesb.jacobian(b) == 2

def test_mul_same_number():
    """Test a case where we multiply
    """
    result = num2 * num2
    assert result.val == 4
    assert result.jacobian(num2) == 4

def test_mixed_mul():
    assert (num2 * 3).val == 6

def test_mixed_rmul():
    assert (3 * num2).val == 6

def test_mixed_rmul_deriv_number_only():
    with pytest.raises(KeyError):
        (3 * num2)._deriv[3]

def test_mul_deriv_first():
    assert (num2 * num3).jacobian(num2) == 3

def test_mul_deriv_second():
    assert (num2 * num3).jacobian(num3) == 2

def test_mixed_mul_deriv():
    assert (3 * num2).jacobian(num2) == 3

def test_div():
    assert (num4 / num2).val == 2

def test_div_same_number():
    assert (num2 / num2).jacobian(num2) == 0

def test_div_deriv_first():
    assert (num4 / num2).jacobian(num4) == pytest.approx(1 / 2)

def test_div_deriv_second():
    assert (num4 / num2).jacobian(num2) == -1

def test_div_shared_partial():
    """Test the case where both numbers in an __div__ share a partial derivative
    """
    t = Number(10)
    a = Number(4, deriv={t: 2})
    b = Number(2, deriv={t: 4})

    atimesb = a / b
    assert atimesb.jacobian(t) == -3
    assert atimesb.val == 2
    assert atimesb.jacobian(a) == pytest.approx(1 / 2)
    assert atimesb.jacobian(b) == -1

def test_mixed_div():
    assert (num4 / 2).val == 2

def test_mixed_div_deriv_number_only():
    with pytest.raises(KeyError):
        (num4 / 2)._deriv[2]

def test_mixed_rdiv():
    assert (4 / num2).val == 2

def test_mixed_rdiv_deriv():
    assert (4 / num2).jacobian(num2) == -1

def test_pow():
    assert (num4 ** num2).val == 16

def test_pow_shared_partial():
    """Test the case where both numbers in an __div__ share a partial derivative
    """
    t = Number(10)
    a = Number(4, deriv={t: 2})
    b = Number(2, deriv={t: 4})

    atotheb = a ** b
    assert atotheb.jacobian(t) == pytest.approx(16 + 64 * np.log(4))
    assert atotheb.val == 16
    assert atotheb.jacobian(a) == pytest.approx(8)
    assert atotheb.jacobian(b) == pytest.approx(16 * np.log(4))

def test_pow_same_number():
    result = num2 ** num2
    assert result.jacobian(num2) == pytest.approx(4 * np.log(2) + 4)

def test_pow_deriv_first():
    assert (num4 ** num2).jacobian(num4) == 8

def test_pow_deriv_second():
    assert (num4 ** num2).jacobian(num2) == pytest.approx(4 ** 2 * np.log(4))

def test_mixed_pow():
    assert (num4 ** 2).val == 16

def test_mixed_rpow():
    assert (4 ** num2).val == 16

def test_mixed_rpow_deriv_number_only():
    with pytest.raises(KeyError):
        (4 ** num2)._deriv[4]

def test_exp():
    assert operations.exp(num_log_1).val == 1

def test_exp_method():
    assert num_log_1.exp().val == 1

def test_exp_deriv():
    assert operations.exp(num_log_1).jacobian(num_log_1) == 1

def test_log():
    assert operations.log(num_e).val == 1

def test_log_deriv():
    assert operations.log(num_e).jacobian(num_e) == np.exp(-1)

def test_negate():
    a = Number(1)
    assert (-a).val == -1

def test_negate_deriv():
    a = Number(1)
    assert (-a).jacobian(a) == -1

def test_negate_multiple_partials():
    derivs = {
        num2: 4,
        num3: 5,
        num4: 6
    }

    # A number with partial derivatives w.r.t. 2, 3, and 4
    a = Number(2, derivs)
    nega = -a

    assert -a.jacobian(num2) == -4
    assert -a.jacobian(num3) == -5
    assert -a.jacobian(num4) == -6


def test_duplicate_value():
    out = num2 + num3
    new_3 = Number(3)
    with pytest.raises(KeyError):
        out._deriv[new_3]

def test_function_composition():
    step1 = operations.sin(a)
    step2 = operations.cos(step1)
    assert step2.jacobian(a) == pytest.approx(0)
    assert step2.val == pytest.approx(np.cos(1))

def test_sin_with_constant():
    result = operations.sin(2 * a)
    assert result.jacobian(a) == pytest.approx(-2)
    assert result.val == pytest.approx(0)

def test_longer_composition():
    num4 = Number(4)
    step1 = num2 + num3
    step2 = num3 + num4
    step3 = step1 * step2

    assert step3.jacobian(num2) == 7
    assert step3.jacobian(num3) == 12
    assert step3.jacobian(num4) == 5

def test_jacobian():
    assert sina.jacobian(a) == pytest.approx(0)

def test_jacobian_no_partial():
    """Test for if the user tries to create a jacobian for a partial derivative they haven't used
    """
    with pytest.raises(ValueError):
        sina.jacobian(Number(100))

def test_jacobian_requires_order():
    with pytest.raises(TypeError):
        sina.jacobian()

def test_jacobian_multi_input():
    result = num3 * num4
    # assert result.jacobian((num3, num4)) == [4, 3]
    jacobian = result.jacobian((num3, num4))
    assert jacobian[0] == 4
    assert jacobian[1] == 3

def test_logistic():
    assert operations.logistic(num_log_1).val == pytest.approx(1 / 2)
    assert num_log_1.logistic().val == pytest.approx(1 / 2)

def test_logistic_deriv():
    operations.logistic(num_log_1).jacobian(num_log_1) == pytest.approx(1 / 4)

def test_asin():
    a = Number(1 / np.sqrt(2))
    assert operations.asin(a).val == pytest.approx(np.pi / 4)

def test_asin_deriv():
    a = Number(1 / np.sqrt(2))
    assert operations.asin(a).jacobian(a) == pytest.approx(np.sqrt(2))

def test_acos():
    a = Number(1 / np.sqrt(2))
    assert operations.acos(a).val == pytest.approx(np.pi / 4)

def test_acos_deriv():
    a = Number(1 / np.sqrt(2))
    assert operations.acos(a).jacobian(a) == pytest.approx(-np.sqrt(2))

def test_atan():
    assert operations.atan(num1).val == pytest.approx(np.pi / 4)

def test_atan_deriv():
    assert operations.atan(num1).jacobian(num1) == pytest.approx(1 / 2)

def test_cosh():
    assert operations.cosh(num_log_1).val == 1

def test_cosh_deriv():
    assert operations.cosh(num_log_1).jacobian(num_log_1) == 0

def test_sinh():
    assert operations.sinh(num_log_1).val == 0

def test_sinh_deri():
    assert operations.sinh(num_log_1).jacobian(num_log_1) == 1

def test_tanh():
    assert operations.tanh(num2).val == np.tanh(2)

def test_tanh_deriv():
    assert operations.tanh(num2).jacobian(num2) == pytest.approx(-np.tanh(2) ** 2 + 1)

def test_sqrt():
    assert operations.sqrt(num4).val == 2

def test_sqrt_deriv():
    assert operations.sqrt(num4).jacobian(num4) == pytest.approx(1 / 4)

def test_equatility():
    a = Number(1)
    b = Number(1, deriv={a: 4})
    c = Number(1, deriv={a: 4})
    assert b == c

#if __name__ == '__main__':
#    t = Number(10)
#    a = Number(4, deriv={t: 2})
#    b = Number(2, deriv={t: 4})
#    
#    atotheb = a ** b
#    a = Number(np.pi / 2)
#    b = Number(3 * np.pi / 2)
#    
#    
#    step1 = operations.sin(a)
#    step2 = operations.sin(b)
#    step3 = operations.sin(a + b)
#
#
#    print(step3.jacobian(a))
#if __name__ == '__main__':
#    print((num2 * num2).val)