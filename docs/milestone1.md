# Milestone 1 Document

## Introduction

The software deploys the Forward Mode of Automatic Differentiation (Forward AD) to evaluate the derivatives of functions. Specifically, the software efficiently evaluates at machine precision the Jacobian matrix (a $m\times n$ matrix) of any function $f: R^n \rightarrow R^m$.

**Importance of the Jacobian Matrix**

Evaluating the Jacobian matrix is significant. The Jacobian matrix provides information of (partial) derivatives of the function, which can be used in many different ways. For example, it can be used to linearly approximate the function about the differentiation point, and it can be used to find the function’s extrema and roots. Moreover, Jacobian matrix is critical in a lot of optimization problems across different scientific fields.  

One example of the utilization of Jacobian matrix is the famous Newton’s Method for root finding. The core of the method is that it utilizes evaluation of the Jacobian matrix in each step of its iterations to linearly approximate the given function.

**Importance of High Accuracy and Efficiency**

It is important to evaluate the Jacobian matrix with high accuracy, as small errors could accumulate in higher dimensions and over iterations. It is also important to evaluate the Jacobian matrix efficiently, considering the time cost of the calculation as the complexity and dimension of the function go up.

**Problems with Classical Methods**

Both Symbolic Differentiation and Finite Difference Method face issues of increasing errors and increasing time cost in evaluating the derivatives as dimensions and complexities of the function go up.

**Advantages of the Software (Forward AD)**

Our software, by using Forward AD, can evaluate the Jacobian Matrix of any function $f: R^n \rightarrow R^m$ up to machine precision efficiently. Every function, regardless of its complexity, executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication and division) and elementary functions (sin, cos, log, etc). Using this fact, AD applies chain rule repeatedly on these operations, and can therefore compute derivatives of any order up to machine precision of the given function. As the order increases, the complexity of AD calculation is not worse than the original function, therefore achieving efficiency. 

- This software aims to numerically evaluate the deivative of any function with high precision utilizing automatic differentiation.

- Finite differencing equation:
$$
f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}
$$

- Automatic differentiation (AD) can compute derivatives to machine precision without using numerical differentiation or symbolic differentiation.

    - Numerical differentiation, i.e., differentiation with the method of finite difference, can become unstable depending on step size and the particular function we're trying to differentiate. The accuracy of finite differencing also depends on choice of step size $h$.

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
    return x**2

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
>>> print(y.deriv[x])
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
>>> x.value
2
>>> x.deriv[x]
1
```

```python
>>> q.deriv[x]
9
>>> q.deriv[y]
6
>>> q.deriv
{x: 9, y: 6}
```
Because `a` does not have a `deriv`, there is no `q.deriv[a]`.

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

```python
x = autodiff.array((1, 2, 3))

def f(x):
    return x.T @ x

q = f(x)

>>>q.deriv[x]
autodiff.array([2, 4, 6])

>>>q.value
13
```

```python
x = autodiff.array((1, 2, 3))

def f(x):
    return 2 * x

q = f(x)

>>>q.deriv[x]
autodiff.array((2, 2, 2))

>>>q.value
autodiff.array((2, 4, 6))

>>>q.jacobian(x)
?
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
- `autodiff.sqrt()`

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
    except AttributeError
        return np.cos(a)
    
@elementary(sin_deriv)
def sin(a):
    return np.sin(a)
```

When any elementary operation takes in two `Number()` objects (objects that have `deriv` attributes), that elementary operation will return a `Number()` with a partial derivative with respect to every key of both `Number()` objects:
```python
>>>x = autodiff.array((1, 2))
>>>y = autodiff.array((3, 4))
>>>x.deriv
{
    x[0]: 1,
    x[1]: 1,
}

>>>y.deriv
{
    y[0]: 1,
    y[1]: 1,
}

>>>q = x.T @ y
>>>q.deriv
{
    x[0]: 3,
    x[1]: 4,
    y[0]: 1,
    y[1]: 2,
}
>>>q.jacobian((*x, *y))
autodiff.array([3, 4, 1, 2])
>>>q.jacobian((*x, *y)).shape
(4,)
```

```python
>>>z = autodiff.array((5, 6))
w = q @ z
>>>w.deriv
{
    x[0]: array([15, 18]),
    x[1]: array([20, 24]),
    y[0]: array([5, 6]),
    y[1]: array([10, 12]),
    z[0]: array([11, 0]),
    z[1]: array([0, 11]),
}
>>>w.jacobian((*x, *y, *z)).shape
(2, 6)
```

`Number()` overloads `__mul__` and `__rmul__`:
```python
x = Number(2)
y = Number(3)

class Number():
    ...

    def _mult_deriv(self, other):
        try:
            self.deriv[self] * other.value

        except ...

    def __mul__(self, other):
        
        try:
            out = Number(self.value, other.value)
            out.deriv = _mult_deriv(self, other)

        except ...

```


#### Classes

`autodiff.array` inherits from `numpy.array`, but also stores the jacobian.
`autodiff.Number` is the base class for a numeric type.

Many elementary operations rely on their `numpy` counterparts, but also include their derivative.