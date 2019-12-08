# Implementation

```eval_rst
.. note:: A full API documentation is avalable here: :mod:`autodiff`
```

## External dependencies
The `autodiff` package proper requires only `numpy`. Running tests requires `pytest` and `codecov`, while generating this documentation requires `sphinx` (version 1.7.9).

## Core data structures and classes

Currently, the `autodiff` package has two core data structure, `Number` and `Array`. A `Number` is a scalar that stores a value and a derivative. `Array` subclasses the `numpy.ndarray` to support functions with vector inputs. It holds a 1-d array of `Number` objects.

### Important attributes of the `Number` class
The `Number` class has only two attributes, a value (`val`) and a `dict` of partial derivatives (`deriv`). The user can can define a new type of number easily:

```python
class NewInt(Number):
    def __init__(self, a, b):
        super(self).__init__(a, b)
        self.val = int(a)
        self.deriv = b
```

The `Array` class has one attribute, which is a `lst` that holds internally a list of `Number` objects
```python
def __init__(self, iterable):
        self._lst = []
        flat_iterable = np.array(iterable).flatten()
        for elt in flat_iterable:
            # Check if element is number type
            try:
                val = elt.val
                self._lst.append(elt)
            except:
                self._lst.append(Number(elt))
        self._lst = np.array(self._lst).reshape(np.shape(iterable))
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
The ```Array``` class overloads the following operations:

- `+`
- `-`
- `*`
- `/`
- `**`

These will either support operations between two `Array` objects, or one `Array` object and one `Number` object.

Note here that `+` will perform a concatenation of two `Array` objects, to use the element-wise addition, user could call `.add(other)` method.
Similar methods are:

- `.add(other)`
- `.subtract(other)`
- `.multiply(other)`
- `.divide(other)`
- `.power(other)`
- `.dot(other)`

Except `.add(other)`, these all perform the same functionalities as overloaded functions specified above. `.dot(other)` performs a dot product of two arrays, as is used in `numpy`.

Moreover, `Array` will support the following operations, which will perform element-wise operations on each element when called:

- `.sum()`
- `.sin()`
- `.cos()`
- `.tan()`
- `.exp()`

Different than the `Number` class, these operations doesn't live in the `operations.py` module. They are directly overloaded or implemented inside the `Array` class. For example, `+` is overloaded as:
```python
def __add__(self, other):
        if isinstance(other, Number):
            out = self._lst+[other]
            return Array(out)
        if isinstance(other, Array):
            out = np.append(self._lst,other._lst)
        else:
            try: 
                is_iterable = iter(other)
            except:
                raise TypeError("can only concatenate iterable s(not {})".format(type(other)))
        return Array(out)
```

To access the derivatives, `Array` implements a jacobian method, which will return another `Array` object in 2-d, holding each row as an element of the original array, each column as the element of `order` to take partial derivatives with respect to.
```python
def jacobian(self, order):
        '''
        Returns the jacobian matrix by the order specified.
        
        Args:
            order: the order to return the jacobian matrix in. Has to be not null
        
        Returns:
            a list of partial derivatives specified by the order.
        '''

        def _partial(deriv, key):
            try:
                return deriv[key]
            except KeyError:
                raise ValueError(
                    f'No derivative with respect to {repr(order)}'
                )
        j = []
        for element in self._lst:
            jacobian = []
            try:
                for key in order:
                    jacobian.append(_partial(element.deriv, key))
            except TypeError:
                # The user specified a scalar order
                jacobian.append(_partial(element.deriv, order))
            j.append(jacobian)
        j = Array(j)
        return j
```