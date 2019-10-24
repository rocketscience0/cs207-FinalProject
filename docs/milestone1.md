# Milestone 1 Document

## Introduction

- Finite differencing equation:
$$
f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
$$

- Automatic differentiation (AD) can compute derivatives to machine precision without using finite differencing
    - Finite differencing can become unstable depending on step size and the particular function we're trying to differentiate.
    - The accuracy of finite differencing also depends on choice of step size $h$.
- While symbolic math packages (such as `sympy`) can do the same, symbolic math becomes complex with arbitrary functions, and requires that every function have an analytical representation.
    - **_NOTE_**: Include example of function you can't differentiate symbolically

- *Why is this important?*
    - Derivatives are one of the most common steps in science and engineering.
        - Are necessary in many optimization algorithms, which are extremely useful in machine learning and engineering.
        - Simulations


## Background

- Chain rule
- Graph structure
    - Elementary functions
- Example calculation



## How to use `autodiff`

High-level interaction with `autodiff` is simple:

Using common elementary functions
```python
import autodiff as ad
from autodiff.elementary_functions import elementary
import numpy as np

x = ad.Number(2)

# A sample function
def testfunc(x):
    # return x**2
    return NewElementary(a)

# A function to return a derivative
deriv = ad.diff(testfunc)
deriv_at_x = deriv(x)
```

It is also possible to define vector functions
```python
x = ad.array([1, 2, 3, 4])

def testfunc(x):
    return x.T @ x
```

- Additional ideas:
    - Finding optima?
    - Generating computational graph?

## Software Organization


### Directory structure

```
.
├── README.md
├── setup.py
└── docs
    └── milestone1.md
└── demos
    └── simple_demo.py
    └── ...
    └── complex_demo.py
└── autodiff
    └── __init__.py
└── tests
    └── test_autodiff.py
```

#### Modules

- `autodiff`
    - Main package
    - **_NOTE_**: Expand more
- `test_autodiff`
    - Run tests for this package

- The directory `demos` contains a series of demos and examples for using our package, ranging from simple to complex.

#### Testing
- All tests live in `tests/test_autodiff.py`
- We will use both `TravisCI` and `CodeCov` to distribute reports.

#### Installation and packaging
**Subject to change in final package**

1. Clone from github
```bash
git clone https://github.com/rocketscience0/cs207-FinalProject.git
```

1. Install
```bash
python setup.py
```
**_NOTE_**: Add content about `DistUtils` and why. Do we want to use `PyPI`?


## Implementation

You can also define custom elementary operations using the `elementary` decorator.

#### Core data structures
The core data structure is a `Number`, which stores a value and a derivative:

```python
x = autodiff.Number(3)
y = x**2
```

```python
>>> print(y.value)
9
>>> print(y.deriv[y])
6
```


Defining a new type of number is easy:

```python
class NewInt(Number):
    def __init__(self, a, b):
        super(self).__init__(a, b)
        self.value = int(a)
        self.deriv = b
```

The `autodiff` package also works for functions with multiple inputs:
```python
x = autodiff.Number(2)
y = autodiff.Number(3)

def f(x, y, a=3):
    return a * x * y

q = f(x, y, a=3)
```
```python
>>> q.deriv[x]
9
>>> q.deriv[y]
6
>>> q.deriv
{x: 9, y: 6}
```
Because `a` does not have a `deriv`, there is no `q.grad[a]`.

It is also possible to return the gradient as an array if the user specifies an order of inputs:
```python
>>> q.deriv.asarray((x, y))
array([9, 6])
```

If a function returns an array:
```python
x = autodiff.Number(np.pi / 2)
y = autodiff.Number(3 * np.pi / 2)

def f(x, y):
    return autodiff.array((
        y * autodiff.sin(x),
        x * autodiff.sin(y)
    ))
q = f(x,y)
>>>q.deriv[x]
autodiff.array([0, 1])
>>>q.deriv[y]
autodiff.array([1, 0])
```

Again, it is possible to get the entire jacobian if the user specifies an order:
```python
>>>q.jacobian((x, y))
autodiff.array([[0, 1],
                [1, 0]])
```



#### Methods and name attributes
The number class overloads the following common elementary operations:

- `+`
- `-`
- `*`
- `/`
- `**`
- `@`

We have also included the following elementary operations

- `autodiff.sin()`
- `autodiff.cos()`
- `autodiff.tan()`
- `autodiff.asin()`
- `autodiff.acos()`
- `autodiff.atan()`
- `autodiff.log()`
- `autodiff.exp()`

Defining custom elementary functions is straightforward, using the `elementary` decorator (this is the same method we use internally).
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
        return a.deriv * np.cos(a)
    except AttributError
        return np.cos(a)
    
@elementary(sin_deriv)
def sin(a):
    return np.sin(a)
```

#### Classes

`autodiff.array` inherits from `numpy.array`, but also stores the jacobian.
`autodiff.Number` is the base class for a numeric type.

Many elementary operations rely on their `numpy` counterparts, but also include their derivative.