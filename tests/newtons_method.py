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
    
    
    while fxn.val > 1e-7:
        x0 = x1

        fxn = func(x0)

        fpxn = fxn.jacobian(x0)

        jacobians.append(fpxn)
        
        x1 = x0- fxn / fpxn
        
    return x1, jacobians
    
if __name__ == '__main__':
    x0 = Number(5)
    xstar, jacobians = newtons_method(func, x0)

    print(xstar, jacobians[-1])