# BFGS optimization

The test function is `$y = 5x^2+10x-8$`. BFGS's method is implemented to find the minimum of the test function. The user should be able to find the x and y of the minmium as well as access the jacobian of eaach optimization step.

First, we instantiate a `Number(5)` as the initial guess (`$x_0$`) of the root to the minimum. The `bfgs()` method takes the test function and the initial guess. 

Second, the function and its derivative are evaluated at `$x_0$`. BFGS requires a speculated Hessian, and the initial guess is usually an identity matrix, or in the scalar case, `1`. The initial guess of hessian is stored in `$b_0$` Then an intermediate `$s_0$` is determined through solving `$b_0s_0=-\nabla func(x_0)$`

Third, `$x_1$`'s value is set to be `$x_0+s_0$`

Fourth, another intermediate `$y_0$`'s value is set to be `$\nabla(x_1)-\nabla(x_0)$`

Fifth, `$b_1$` is updated and its value is equal to`$b_1=b_0+\Delta b_0$`, where $\Delta b_0$ is equivalent to `$\frac{y_0}{s_0}-b0$`

Sixth, `$b_0$` is set to be 


```python
import sys
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
import numpy as np


def func(x):
    return 5 * x ** 2 + 10 * x - 8

def bfgs(func, initial_guess):
    
    #bfgs for scalar functions
    
    x0 = initial_guess
    
    #initial guess of hessian
    b0 = 1
    
    fxn0 = func(x0)

    fpxn0 = fxn0.jacobian(x0)
    
    jacobians = []
    
    jacobians.append(fpxn0)
    
    while(np.abs(fpxn0)>1*10**-7):
        fxn0 = func(x0)

        fpxn0 = fxn0.jacobian(x0)
        print(fxn0)

        s0 = -fpxn0/b0

        x1=x0+s0 
        
        fxn1 = func(x1)
        fpxn1 = fxn1.jacobian(x1)
        
        
        y0 = fpxn1-fpxn0
        
        if y0 == 0:
            break
            
        #delta_b = y0**2/(y0*s0)-b0*s0**2*b0/(s0*b0*s0)
        delta_b = y0/s0-b0
        b1 = b0 + delta_b
        
        x0 = x1
        
        b0 = b1
        
        jacobians.append(fpxn1)

        
        
    return x0,func(x0),jacobians
    

xstar,minimum,jacobians = bfgs(func,x0)
jacobians
```

    Number(val=-12.8)
    Number(val=3.200000000000001)
    Number(val=-13.0)





    [-2.0, 18.0, 0.0]
