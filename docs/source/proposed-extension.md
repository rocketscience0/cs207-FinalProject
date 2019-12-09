# Proposed extension

As an extension of the `autodiff` package, we would like to propose making two additional submodules that make use of automatic differentiation: an `optimization` package that contains an easy-to-use API for finding the optima of scalar- and vector-valued functions, and a `rootfinding` package that can find roots of scalar- and vector-valued functions.

These are two new submodules under the `autodiff` package, so the new directory structure will be:
```
.
├── autodiff
│   ├── __init__.py
│   ├── operations.py
│   ├── optimization.py
│   ├── root_finding.py
│   └── structures.py
...
```

For `autodiff.optimization`, we will allow the user to use two different optimization algorithms: `bfgs`, `bfgs_symbolic` and `steepest_descent`. These is also an extension of the optimization example in our current `docs` folder.

In `autodiff.rootfinding`, we will have convenient API that allows the user to find the roots of complex functions using Newton's method, an extension of the `root_finding` example in our `docs` folder.

## Proposed use cases

### Optimization

```python
>>> from autodiff.structures import Number
>>> from autodiff.structures import Array
>>> from autodiff.optimizations import bfgs
>>> import timeit
>>> initial_guess = Array([Number(2),Number(1)])

>>> def rosenbrock(x0):
...     return (1-x0[0])**2+100*(x0[1]-x0[0]**2)**2

>>> initial_time2 = timeit.timeit()

>>> results = bfgs(rosenbrock,initial_guess)
>>> print("Xstar:",results[0])
Xstar: [Number(val=1.0000000000025382) Number(val=1.0000000000050797)]
>>> print("Minimum:",results[1])
Minimum: Number(val=6.4435273497518935e-24)

>>> time_for_optimization = initial_time2-final_time2
>>> print("Time for symbolic bfgs to perform optimization",time_for_optimization,'total time taken is',time_for_optimization)
Time for symbolic bfgs to perform optimization 0.0005920969999999581 total time taken is 0.0005920969999999581
      
```

### Root finding
There are two methods in Root_finding, `newtons_method` and `secant_method`.

```python
>>> from autodiff import rootfinding
>>> def test_func(x,):
...     return 0.5 * x ** 2 + 2 * x + 1

>>> rf = rootfinding.newtons_method(test_func, x_0=Number(0))
>>> rf[0] #root
... -0.585786437626905
>>> rf[1] #jacobians
[2.0, 1.5, 1.4166666666666665, 1.4142156862745097, 1.4142135623746899]
```

## Comparison to traditional methods
We will also include bfgs_symbolic method of the optimization in `optimization` as instructional tools to demonstrate the differences between automatic differentiation and traditional method of user calculating derivatives by themselves. Not counting the time to do derivatives, we see that AD still holds a time efficiency advantage when we time the two methods.

```python
>>> from sympy import * 
>>> import sympy

>>> initial_time = timeit.timeit()

>>> x, y = symbols('x y') 

>>> rb = (1-x)**2+100*(y-x**2)**2

>>> # Use sympy.diff() method 
>>> par1 = diff(rb, x) 
>>> par2 = diff(rb,y)

>>> def gradientRosenbrock(x0):
>>>     x=x0[0]
>>>     y=x0[1]
>>>     drdx = -2*(1 - x) - 400*x*(-x**2 + y)
>>>     drdy = 200 *(-x**2 + y)
>>>     return drdx,drdy

>>> final_time=timeit.timeit()
>>> time_for_sympy = initial_time-final_time

>>> initial_time1 = timeit.timeit()
>>> def gradientRosenbrock(x0):
...     x=x0[0]
...     y=x0[1]
...     drdx = -2*(1 - x) - 400*x*(-x**2 + y)
...     drdy = 200 *(-x**2 + y)
...     return drdx,drdy
>>> results = bfgs_symbolic(rosenbrock,gradientRosenbrock,[2,1])
>>> time_for_optimization_symbolic = initial_time1-final_time1
>>> print("Time for symbolic bfgs to perform optimization",time_for_optimization_symbolic,'total time taken is',time_for_optimization_symbolic+time_for_sympy)
Time for symbolic bfgs to perform optimization 0.005963177000012365 total time taken is 0.007416142000003845
```
Note that above with bfgs, we achieved a total time of 0.00059, which is an advantage compared to the traditional `bfgs_symbolic`.