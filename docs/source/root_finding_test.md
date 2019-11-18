## Root-Finding Algorithm 

Here, we demonstrate a use case of a root-finding algorithm that requires calculation of the Jacobian.

The test function is `$y = 5x^2+10x-8$`. 

```python
>>> import autodiff.operations as operations
>>> from autodiff.structures import Number
```

```python
def func(x):
    return 5 * x ** 2 + 10 * x - 8
```

Newton's method is implemented to find the root of the test function. The user should be able to find the root and access the Jacobian of each iteration. 

How does this implementation work?
1. The function and its derivative are evaluated at `$x_0$`, then `$x_1$` is calculated as `$x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}$`. The Jacobian is stored in the `jacobians` list.

2. The evaluation of the function at `$x_1$` is compared with the threshhold, in this case `$10^{-7}$`. The absolute value of the function's value at `$x_1$` is larger than the threshold, so `$x_0$`'s value is updated, i.e., `$x_0 = x_1$`.

3. The function and its derivative is again evaluated at the new `$x_0$`. Derivative is stored in the `jacobians` list. This process is repeated until the threshold is met.

4. The `Newton()` method returns the root and the `jacobians` list. User may access each step's derivative from this list.

```python
def newtons_method(func, initial_guess):
    
    # Store a list of jacobians from each iteration
    jacobians = []
    x0 = initial_guess
	fxn = func(initial_guess)
    fpxn = fxn.jacobian(initial_guess)
    x1 = x0 - fxn/fpxn
	jacobians.append(fpxn)
    
    while abs(fxn.val) > 1e-7:
        x0 = x1
		fxn = func(x0)
		fpxn = fxn.jacobian(x0)
		jacobians.append(fpxn)
        x1 = x0- fxn / fpxn
        
    return x1, jacobians
```

We can instantiate a `Number(5)` as the initial guess ($x_0$) of the root. 

```python
>>> x0 = Number(5)
```

The `Newton()` method takes the test function and the initial guess. 
```python
>>> xstar, jacobians = newtons_method(func, x0)
>>> print(xstar, jacobians[-1])
Number(val=0.6124515496597099) 16.124515496597116
```


