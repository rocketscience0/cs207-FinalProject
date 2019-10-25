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

x = ad.Int(2)

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

Defining custom elementary functions
```python
def my_pow_deriv(a, b):
    """ Returns the derivative of my_pow at a and b
    """
    return b * a ** (b - 1)

@elementary(my_pow_deriv)
def my_pow(a, b):
    return pow(a, b)
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
    - Implements the forward mode of automatic differentiation
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
NEW VERSION:
1. Ensure setuptools, pip are up to date
```bash
python -m pip install --upgrade pip setuptools
```

2. Install package from github
```bash
pip install git+https://github.com/rocketscience0/cs207-FinalProject.git
```

3. 
```bash
python setup.py
```


Our workflow is based off of this (guide)[https://packaging.python.org/tutorials/installing-packages/].

`setup.py` will specify required pieces of metadata, such as the version and dependencies. We will use `setuptools` as a distribution build tool. Why `setuptools` as opposed to `distutils`? As noted by the (Python Packaging User Guide)[https://packaging.python.org/guides/tool-recommendations/], `setuptools` is outside the standard library, allowing for consistency across different Python versions. 

In class, we discussed also using the tool `pipenv`, a further abstraction that creates a virtual environment under the hood. Because we do not plan on having a complicated set of dependencies at the moment (nor implementations that rely on specific versions), it is unlikely that we will need a virtul environment. If this becomes an issue, we can manually create a virtual environment or switch over to `pipenv` as needed.

Later on, we may publish a final version of our package (currently a Github repo) as an open-source Python package on PyPI. Using pip will allow users to easily install via `pip install autodiff`.


## Implementation

Our custom data types `autodiff.Int()` and `autodiff.Float()` work will most common elementary functions:

- `+`
- `-`
- `*`
- `/`
- `**`
- `@`
- `np.sin`
- `np.cos`

You can also define custom elementary operations using the `elementar` decorator.

#### Core data structures
The core data structure is a `Number`, which stores a value and a derivative:

```python
x = autodiff.Number(3)
y = x**2

>>> print(y.value)
9
>>> print(y.deriv)
6
```


Defining a new type of number is easy

```python
class NewInt(Number):
    def __init__(self, a, b):
        super(self).__init(a, b)
        self.value = int(a)
        self.deriv = b

```

#### Classes

#### 