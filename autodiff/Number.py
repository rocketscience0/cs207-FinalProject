"""Collection of tools to make new elementary operations
"""

from functools import wraps
#from structures import Number
import numpy as np

class Number():

    def __init__(self, val, deriv=None):

        self.val = val
        if deriv is None:
            self.deriv = {
                self: 1
            }
        elif isinstance(deriv, dict):
            self.deriv = deriv
            #keep also a copy of the derivative w.r.t. itself
            self.deriv[self] = 1
        else:
            self.deriv = {
                    self: deriv
                    }

    def __repr__(self):
        return f'Number(val={self.val})'
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return subtract(self, other)
    
    def __rsub__(self, other):
        return subtract(self, other)
    
    def __mul__(self, other):
         return mul(self, other)
    
    def __rmul__(self, other):
        return mul(self, other)
    
    def __div__(self, other):
        return div(self, other)
    
    def __rdiv__(self, other):
        return div(self, other)
    
    def __pow__(self, other):
        return power(self, other)
    
    def sin(self):
        return sin(self)
    
    def cos(self):
        return cos(self)
    
    def tan(self):
        return sin(self)/cos(self)

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
    except:
        d = x.deriv
        d[y] = 1
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
                d[key] = y.deriv[key]
    except:
        d = x.deriv
        d[y] = 1
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
    except:
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
    except:
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
                d[key] = (y.deriv[key] * x.val - x.deriv[key] * y.val)/(y.val**2)
            else:
                d[key] = x.deriv[key] / y.val
        for key in y.deriv.keys():
            if not key in x.deriv.keys():
                d[key] = x.val / y.deriv[key]
    except:
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
    except:
        s = x.val / y
    return s


def pow_deriv(x,a):
    """Derivative of power of a Number
    
    Args:
        x: a Number
        a: an integer/float of the degree
    
    Returns:
        The derivative of the power
    """
    d={}
    for key in x.deriv.keys():
        d[key] = a * x.val**(a-1) * x.deriv[key]
    d[x] = a * x.val**(a-1) * x.deriv[x]
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
    return x.val**y

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
        float: Sin(x.val)
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