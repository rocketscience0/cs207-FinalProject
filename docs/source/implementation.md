# Implementation

```eval_rst
.. note:: A full API documentation is avalable here: :mod:`autodiff`
```

## External dependencies
The `autodiff` package proper requires only `numpy`. Running tests requires `pytest` and `codecov`, while generating this documentation requires `sphinx` (version 1.7.9).

## Core data structures and classes

Currently, the `autodiff` package has one core data structure, the `Number`. A `Number` is a scalar that stores a value and a derivative. Future versions will include the `array`, which subclasses the `numpy.ndarray` to support functions with vector inputs.

### Important attributes of the `Number` class
The `Number` class has only two attributes, a value (`val`) and a `dict` of partial derivatives (`deriv`). The user can can define a new type of number easily:

```python
class NewInt(Number):
    def __init__(self, a, b):
        super(self).__init__(a, b)
        self.val = int(a)
        self.deriv = b
```


## Methods and name attributes
The `Number` class overloads the following common elementary operations:

- `+`
- `-`
- `*`
- `/`
- `**`
- `@`

```eval_rst
We have also included the following elementary operations, all of which use their `numpy` counterparts and live in the :mod:`autodiff.operations` module.
```

- `autodiff.operations.sin()`
- `autodiff.operations.cos()`
- `autodiff.operations.tan()`
- `autodiff.operations.asin()`
- `autodiff.operations.acos()`
- `autodiff.operations.atan()`
- `autodiff.operations.log()`
- `autodiff.operations.exp()`
- `autodiff.operations.sqrt()`

Defining custom elementary functions is straightforward, using the `elementary` decorator (this is the same method we use internally). The decorator takes one input, a function with the same arguments as the elementary operation, but calculates the derivative of the operation rather than the value. We call this derivative function internally.

```python
def my_pow_deriv(a, b):
    """ Returns the derivative of my_pow at a and b
    """
    return b * a ** (b - 1)

@elementary(my_pow_deriv)
def my_pow(a, b):
    return pow(a, b)
```

```python
def sin_deriv(a):
    """ Returns the derivative of the sin() elemental operation"""
    try:
        return a.deriv * np.cos(a.value)
    except AttributeError
        return np.cos(a)
    
@elementary(sin_deriv)
def sin(a):
    try ...
    return np.sin(a)
```

The `Number()` class overloads `__add__` and `__radd__`, along with other elementary operations as follows. The `autodiff.array` class overloads vector operations similarly.
```python
# From autodiff.operations
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


# In autodiff.structures
class Number():
    ...

    def __add__(self, other):
        '''
        Overloads the add method to add a number object to another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)
    
    def __radd__(self, other):
        '''
        Overloads the right add method to add a number object to another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)

```

## To include in future versions

At this time, `autodiff` only supports scalar functions with scalar outputs. Soon, we will also support vector functions with vector outputs. An `autodiff.array` will subclass `numpy.array`, but will hold `Number` objects. Therefore, matrix operations will be available as they are in `numpy`, including:

- Matrix multiplication (`@`, `dot`)
- Element-wise operations (`+`, `-`, `*`, `/`, `**`)

There will be a few differences when defining function with vector outputs. Rather than each value of the `deriv` dict being a scalar, a vector `deriv` value will instead be an `array`---interpreted as a column of the Jacobian. Once again, it will be necessary for the user to specify in which order he or she will like their Jacobian. Internally, we will treat the user's specified order as a set of seed vectors to calculate each column of the Jacobian.