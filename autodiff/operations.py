"""Collection of tools to make new elementary operations
"""

from .structures import Number
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
        print('Inside wrap()')
        def inner_func(*args):
            value = func(*args)
            deriv = deriv_func(*args)
            return Number(value, deriv)

        return inner_func
    return inner

def sin_deriv(x):
    """Derivative of sin(x)
    
    Args:
        x (structures.Number()): Number to take the sin of. Must have a ``deriv`` attribute
    
    Returns:
        dict: dictionary of partial derivatives (in this case with just one key)
    """
    return {x: np.cos(x.val) * x.deriv[x]}

@elementary(sin_deriv)
def sin(x):
    """Take the sin(x)
    
    Args:
        x (Number): Number to take the sin of
    
    Returns:
        float: Sin(x.val)
    """
    return np.sin(x.val)


if __name__ == '__main__':
    a = Number(np.pi / 2)

    sina = sin(a)
    print(sina)
    print(sina.deriv)