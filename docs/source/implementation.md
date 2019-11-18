# Implementation

## External dependencies
The `autodiff` package proper requires only `numpy`. Running tests requires `pytest` and `codecov`, while generating this documentation requires `sphinx` (version 1.7.9).

## Core data structures and classes

Currently, the `autodiff` package has one core data structure, the `Number`. A `Number` is a scalar that stores a value and a derivative. Future versions will include the `array`, which subclasses the `numpy.ndarray` to support functions with vector inputs. A full api documentation is available [here](./api-doc/autodiff).

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

We have also included the following elementary operations, all of which use their `numpy` counterparts.

- `autodiff.sin()`
- `autodiff.cos()`
- `autodiff.tan()`
- `autodiff.asin()`
- `autodiff.acos()`
- `autodiff.atan()`
- `autodiff.log()`
- `autodiff.exp()`
- `autodiff.sqrt()`

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

The `Number()` class overloads `__mul__` and `__rmul__`, along with other elementary operations as follows. The `autodiff.array` class overloads vector operations similarly.
```python
x = Number(2)
y = Number(3)

class Number():
    ...

    def _mult_deriv(self, other):
        try:
            self.deriv[self] * other.value

        except ...

    @elementary(self._mult_deriv)
    def __mul__(self, other):
        
        try:
            out = Number(self.value, other.value)
            out.deriv = _mult_deriv(self, other)

        except ...

```

## To include in future versions

At this time, `autodiff` only supports scalar functions with scalar outputs. Soon, we will also support vector functions with vector outputs. An `autodiff.array` will subclass `numpy.array`, but will hold `Number` objects. Therefore, matrix operations will be available as they are in `numpy`, including:

- Matrix multiplication (`@`, `dot`)
- Element-wise operations (`+`, `-`, `*`, `/`, `**`)