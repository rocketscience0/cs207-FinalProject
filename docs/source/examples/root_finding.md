# Root finding with Newton's method

The test function is `$y = 5x^2+10x-8$`. Newton's method is implemented to find the root of the test function. The user should be able to find the root and access the Jacobian of each iteration.

First, we instantiate a `Number(5)` as the initial guess (`$x_0$`) of the root. The `Newton()` method takes the test function and the initial guess. 

Second, the function and its derivative are evaluated at `$x_0$`, then `$x_1$` is calculated as `$x_1 = x_0-\frac{f(x_0)}.
{f'(x_0)}$`. The jacobian is stored in the `jacobians` list

Third, the evaluation of the function at $x_1$ is compared with the threshhold, in this case `$10^{-7}$`. The absolute value of the function's value at `$x_1$` is larger than the threshold, so `$x_0$`'s value is updated, i.e., `$x_0 = x_1$`.

Fourth, the function and its derivative is again evaluated at the new `$x_0$`. Derivative is stored in the `jacobians` list. This process is repeated until the threshold is met.

Fifth, the `Newton()` method returns the root and the `jacobians` list. User may access each step's derivative from this list.


```python
import sys
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
import numpy as np
from copy import deepcopy

def func(x):
    return 5 * x ** 2 + 10 * x - 8

def newtons_method(func, initial_guess):
    
    #stores a list of jacobians from each iteration
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
    

x0 = Number(5)
xstar, jacobians = newtons_method(func, x0)

print(xstar, jacobians[-1])
```

    Number(val=0.6124515496597099) 16.124515496597116



```python

```


```python

```


```python

```


```python

```
