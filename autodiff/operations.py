"""Collection of tools to make new elementary operations
"""

from functools import wraps
from autodiff.structures import Number
import numpy as np


def elementary(deriv_func):
    """Decorator to create an elementary operation

    This takes as an argument a function that calculates the derivative of the function
    the user is calculating. When the decorated function is called, `@elementary` also calls
    `deriv_func` and stores both the value and derivative in a new `Number()` object

    Example:
        >>> import numpy as np
        >>> def sin_deriv(x):
        ...     return {x: np.cos(x.val) * x.deriv[x]}
        >>> @elementary(sin_deriv)
        ... def sin(x):
        ...     return np.sin(x.val)
        >>> a = Number(np.pi / 2)
        >>> sina = sin(a)
        >>> sina.val
        1.0
        >>> sina.deriv[a]
        0.0

    
    Args:
        deriv_func (function): Function specifying the derivative of this function. Must return a dictionary where each key-value pair is the partial derivative of the decorated function
    
    Returns:
        function: Decorated function
    """

    def inner(func):
        @wraps(func)
        def inner_func(*args):
            
            value = func(*args)
            deriv = deriv_func(*args)
            return Number(value, deriv)

        return inner_func
    return inner

def add_deriv(x,y):
    """Derivative of additions, one of x and y has to be a Number object
    
    Args:
        x: a Number
        y: a Number object or an int/float to be added
    
    Returns:
        The derivative of the sum of x and y
    """
    try:
        d={}
        for key in x.deriv.keys():
            if key in y.deriv.keys():
                d[key] = x.deriv[key] + y.deriv[key]
            else:
                d[key] = x.deriv[key]
        for key in y.deriv.keys():
            if not key in x.deriv.keys():
                d[key] = y.deriv[key]
    except AttributeError:
        d = x.deriv
    return d

@elementary(add_deriv)
def add(x,y):
    """add two numbers together, one of x and y has to be a Number object
    
    Args:
        x: a Number object or an int/float
        y: a Number object or an int/float to be added
    
    Returns:
        value of the sum
    """
    try:
        s = x.val + y.val
    except:
        s = x.val + y
    return s

def subtract_deriv(x,y):
    """Derivative of subtractions, one of x and y has to be a Number object
    
    Args:
        x: a Number
        y: a Number object or an int/float to be subtracted
    
    Returns:
        The derivative of the difference of x and y
    """
    try:
        d={}
        for key in x.deriv.keys():
            if key in y.deriv.keys():
                d[key] = x.deriv[key] - y.deriv[key]
            else:
                d[key] = x.deriv[key]
        for key in y.deriv.keys():
            if not key in x.deriv.keys():
                d[key] = -y.deriv[key]
    except AttributeError:
        d = x.deriv
    return d

@elementary(subtract_deriv)
def subtract(x,y):
    """Subtract one number from another, one of x and y has to be a Number object
    
    Args:
        x: a Number object
        y: a Number object or an int/float to be subtracted
    
    Returns:
        value of the difference
    """
    try:
        s = x.val - y.val
    except:
        s = x.val - y
    return s

def mul_deriv(x,y):
    """Derivative of multiplication, one of x and y has to be a Number object
    
    Args:
        x: a Number
        y: a Number object or an int/float to be multiplied
    
    Returns:
        The derivative of the product of x and y
    """
    if (x==y):
        return pow_deriv(x,2)
    try:
        d={}
        for key in x.deriv.keys():
            if key in y.deriv.keys():
                #product rule
                d[key] = x.deriv[key] * y.val + y.deriv[key] * x.val
            else:
                d[key] = x.deriv[key] * y.val
        for key in y.deriv.keys():
            if not key in x.deriv.keys():
                d[key] = y.deriv[key] * x.val
    except AttributeError:
        d = {}
        for key in x.deriv.keys():
            d[key] = x.deriv[key]*y
    return d

@elementary(mul_deriv)
def mul(x,y):
    """Subtract one number from another, one of x and y has to be a Number object
    
    Args:
        x: a Number object
        y: a Number object or an int/float to be subtracted
    
    Returns:
        value of the difference
    """
    try:
        s = x.val * y.val
    except AttributeError:
        s = x.val * y
    return s

def div_deriv(x,y):
    """Derivative of division, one of x and y has to be a Number object
    
    Args:
        x: a Number
        y: a Number object or an int/float to be divided
    
    Returns:
        The derivative of the quotient of x and y
    """
    try:
        d={}
        for key in x.deriv.keys():
            if key in y.deriv.keys():
                #quotient rule
                d[key] = (-y.deriv[key] * x.val + x.deriv[key] * y.val)/(y.val**2)
            else:
                d[key] = x.deriv[key] / y.val
        for key in y.deriv.keys():
            if not key in x.deriv.keys():
                # d[key] = x.val / y.deriv[key]
                d[key] = -x.val / (y.val ** 2) * y.deriv[key]
    except AttributeError:
        d = {}
        for key in x.deriv.keys():
            d[key] = x.deriv[key] / y
    return d

@elementary(div_deriv)
def div(x,y):
    """Subtract one number from another, one of x and y has to be a Number object
    
    Args:
        x: a Number object
        y: a Number object or an int/float to be subtracted
    
    Returns:
        value of the difference
    """
    try:
        s = x.val / y.val
    except AttributeError:
        s = x.val / y
    return s


def pow_deriv(x, a):
    """Derivative of power of a Number
    
    Args:
        x: a Number
        a: a Number
    
    Returns:
        The derivative of the power
    """
    d = {}

    # All the derivates w.r.t x
    try:
        for key in x.deriv.keys():
            if key in a.deriv.keys():
                # Using the chain rule for powers
                d[key] = (((a.val * x.deriv[key]) / x.val) + (a.deriv[key] * np.log(x.val))) * (x.val ** a.val)
            else:
                d[key] = a.val * x.val ** (a.val - 1) * x.deriv[key]
    except AttributeError:
        try:
            for key in x.deriv.keys():
                d[key] = a * x.val ** (a - 1) * x.deriv[key]
        except AttributeError:
            # x isn't a Number(). Just go through
            pass

    # All the derivatives w.r.t a
    try:
        for key in a.deriv.keys():
            if key not in x.deriv.keys():
                d[key] = x.val ** a.val * np.log(x.val) * a.deriv[key]
    except AttributeError:
        try:
            for key in a.deriv.keys():
                d[key] = x ** a.val * np.log(x) * a.deriv[key]
        except AttributeError:
            # a isn't a Number(). Just go through
            pass

    return d

@elementary(pow_deriv)
def power(x,y):
    """Subtract one number from another, one of x and y has to be a Number object
    
    Args:
        x: a Number object
        y: a Number object or an int/float to be subtracted
    
    Returns:
        value of the difference
    """
    # return x.val**y
    try:
        return x.val ** y.val
    except AttributeError:
        try:
            return x ** y.val
        except AttributeError:
            return x.val ** y
            # except AttributeError:
            #     return x ** y

def sin_deriv(x):
    """Derivative of sin(x)
    
    Args:
        x (structures.Number()): Number to take the sin of. Must have a ``deriv`` attribute
    
    Returns:
        dict: dictionary of partial derivatives
    """
    d={}
    for key in x.deriv.keys():
        d[key] = np.cos(x.val) * x.deriv[key]
    d[x] = np.cos(x.val) * x.deriv[x]
    return d

@elementary(sin_deriv)
def sin(x):
    """Take the sin(x)
    
    Args:
        x (Number): Number to take the sin of
    
    Returns:
        float: sin(x.val)
    """
    return np.sin(x.val)

def cos_deriv(x):
    """Derivative of cos(x)
    
    Args:
        x (structures.Number()): Number to take the cos of. Must have a ``deriv`` attribute
    
    Returns:
        dict: dictionary of partial derivatives
    """
    d={}
    for key in x.deriv.keys():
        d[key] = -np.sin(x.val) * x.deriv[key]
    d[x] = -np.sin(x.val) * x.deriv[x]
    return d

@elementary(cos_deriv)
def cos(x):
    """Take the cos(x)
    
    Args:
        x (Number): Number to take the cos of
    
    Returns:
        float: cos(x.val)
    """
    return np.cos(x.val)

def tan_deriv(x):
    """Derivative of tan(x)

    Args: 
        x (structures.Numbers()): Number to take the tan of. Must have a ``deriv`` attribute
    
    Returns:
        dict: dictionary of partial derivatives
    """
    d={}
    for key in x.deriv.keys():
        d[key] = (np.tan(x.val)**2 + 1) * x.deriv[key]
    d[x] = (np.tan(x.val)**2 + 1) * x.deriv[x]
    return d

@elementary(tan_deriv)
def tan(x):
    """Take the tan(x)

    Args:
        x (Number): Number to take the tan of

    Returns:
        float: tan(x.val)
    """
    return np.tan(x.val)

def exp_deriv(x):
    """Derivative of exp(x)
    
    Args:
        x (structures.Number()): Number to take the exp of. Must have a ``deriv`` attribute
    
    Returns:
        dict: dictionary of partial derivatives
    """
    d={}
    for key in x.deriv.keys():
        d[key] = np.exp(x.val) * x.deriv[key]
    d[x] = np.exp(x.val) * x.deriv[x]
    return d

@elementary(exp_deriv)
def exp(x):
    """Take the exp(x)
    
    Args:
        x (Number): Number to take the exp of
    
    Returns:
        float: exp(x.val)
    """
    return np.exp(x.val)

def log_deriv(x, y=np.exp(1)):
    """Derivative of log(x) at base y
    
    Args:
        x (structures.Number()): Number to take the log of. Must have a ``deriv`` attribute
        y (a Number object or an int/float): Base of the logarithm.
    
    Returns:
        dict: dictionary of partial derivatives
    """
    d = {}
    # Use the chain rule to find partials w.r.t everything x depends on
    for key in x.deriv.keys():
        # d[key] = 1 / (x.val * np.log(y.deriv[key]))
        d[key] = 1 / (x.val) * x.deriv[key]

    try:
        d[x] = 1 / (x.val * np.log(y.val))
    except AttributeError:
        d[x] = 1 / (x.val * np.log(y))

    return d

@elementary(log_deriv)
def log(x, y=np.exp(1)):
    """Take the log(x) at base y
    
    Args:
        x (Number): Number to take the log of
        y (a Number object or an int/float): Base of the logarithm.
    
    Returns:
        float: log(x.val)
    """
    try:
        s = np.log(x.val) / np.log(y.val)
    except AttributeError:
        s = np.log(x.val) / np.log(y)
    return s

def negate_deriv(x):
    """Derivative of a negation
    
    Args:
        x (Number): Number to negate
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    return {key: -deriv for key, deriv in x.deriv.items()}

@elementary(negate_deriv)
def negate(x):
    """Negate
    
    Args:
        x (Number): Negate a number
    
    Returns:
        Number: -x
    """
    return - x.val

def asin_deriv(x):
    """Arcsin derivative
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}

    for key in x.deriv.keys():
        d[key] = 1 / np.sqrt(-x.val ** 2 + 1) * x.deriv[key]

    # return 1 / np.sqrt(-x ** 2 + 1)
    return d

@elementary(asin_deriv)
def asin(x):
    """Arcsin
    
    Args:
        x (Number): Value
    
    Returns:
        Number: asin(x)
    """
    return np.arcsin(x.val)

def acos_deriv(x):
    """Derivative of arccos
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}
    for key in x.deriv.keys():
        d[key] = - 1 / np.sqrt(-x ** 2 + 1) * x.deriv[key]

    return d

@elementary(acos_deriv)
def acos(x):
    """Arccos
    
    Args:
        x (Number): Value
    
    Returns:
        Number: acos(x)
    """
    return np.arccos(x.val)

def atan_deriv(x):
    """Derivative of atan
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}
    for key in x.deriv.keys():
        d[key] = 1 / (x ** 2 + 1) * x.deriv[key]
    return d

@elementary(atan_deriv)
def atan(x):
    """Arctan
    
    Args:
        x (Number): Value
    
    Returns:
        Number: atan(x)
    """
    return np.arctan(x.val)

def cosh_deriv(x):
    """Hyperbolic cosine
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}
    for key in x.deriv.keys():
        d[key] = np.sinh(x.val) * x.deriv[key]
    return d

@elementary(cosh_deriv)
def cosh(x):
    """Hyperbolic cosine
    
    Args:
        x (Number): Value
    
    Returns:
        Number: cosh(x)
    """
    return np.cosh(x.val)

def sinh_deriv(x):
    """Hyperbolic sin derivative
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}
    for key in x.deriv.keys():
        d[key] = np.cosh(x.val) * x.deriv[key]
    return d

@elementary(sinh_deriv)
def sinh(x):
    """Hyperbolic sine
    
    Args:
        x (Number): Value
    
    Returns:
        Number: sinh(x)
    """
    return np.sinh(x.val)

def tanh_deriv(x):
    """Hyperbolic tan derivative
    
    Args:
        x (Number): Value
    
    Returns:
        dict: Partial derivatives w.r.t. everything x had a partial w.r.t.
    """
    d = {}
    for key in x.deriv.keys():
        d[key] = -np.tanh(x.val) ** 2 + 1
    return d

@elementary(tanh_deriv)
def tanh(x):
    """Hyperbolic tan
    
    Args:
        x (Number): Value
    
    Returns:
        Number: tanh(x)
    """
    return np.tanh(x.val)