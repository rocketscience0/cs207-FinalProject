# Proposed extension

As an extension of the `autodiff` package, we would like to propose making two additional submodules that make use of automatic differentiation: an `optimization` package that contains an easy-to-use API for finding the optima of scalar- and vector-valued functions, and a `rootfinding` package that can find roots of scalar- and vector-valued functions.

These will be two new new submodules under the `autodiff` package, so the new directory structure will be:
```
.
├── autodiff
│   ├── __init__.py
│   ├── operations.py
│   ├── optimization.py
│   ├── rootfinding.py
│   └── structures.py
...
```

For `autodiff.optimization`, we will allow the user to use two different optimization algorithms: `L-BFGS` and gradient descent. These will be an extension of the optimization example in our current `docs` folder.

In `autodiff.rootfinding`, we will have convenient API that allows the user to find the roots of complex functions using Newton's method, an extension of the `root_finding` example in our `docs` folder.

## Proposed use cases

### Optimization

```python
>>> from autodiff import optimize
>>> def rosenbrock(x, y, a=1, b=100):
...     """A test function to optimize"""
...     return (a - x) ** 2 + b * (y - x ** 2) ** 2

>>> # Find correct answer for optima and root finding
>>> test_func = lambda p: rosenbrock(p[0], p[1])
>>> x_0 = autodiff.array([20, 20])
>>> opt = optimization.Optimizer(func=test_func, x_0=x_0, algorithm='L-BFGS')

>>> # Run the optimization
>>> opt.run()
autodiff.array([1., 1.])

>>> # Analyze performance
>>> opt.time
Wall clock time: 1 s
CPU time (user): 0.5 s
CPU time (sys): 0.4 s
```

### Root finding
```python
>>> from autodiff import rootfinding
>>> def test_func(x,):
...     return 0.5 * x ** 2 + 2 * x + 1

>>> rf = rootfinding.RootFinder(test_func, x_0=0)
>>> rf.run()
>>> rf.root()
... -3.41421356237309
```

## Comparison to finite differencing
We will also include finite-differencing versions of the optimization and root finding algorithms in `optimization` and `rootfinding` as instructional tools to demonstrate the differences between automatic differentiation and finite differencing.