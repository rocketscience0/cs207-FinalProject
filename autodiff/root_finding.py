from autodiff import operations
from autodiff.structures import Number
import numpy as np
from copy import deepcopy

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

def secant_method(func, initial_guess,iterations = 100):
    
    #The secant method uses finite difference version of Newton's method
    # Aims to compare the efficiency with Newton's method
    if len(initial_guess) != 2 | len(initial_guess)==None:
        raise ValueError("Please enter two initial guesses")

    
    #initialize the guesses
    xn_2 = initial_guess[0].val
    xn_1 = initial_guess[1].val
    
    fxn_2 = func(xn_2)
    fxn_1 = func(xn_1)
    
    xn = (xn_2*fxn_1-xn_1*fxn_2)/(fxn_1-fxn_2)
    fxn = func(xn)
    
    for i in range(iterations):
        if fxn !=0:
            xn_2=xn_1
            xn_1 = xn
            fxn_2 = func(xn_2)
            fxn_1 = func(xn_1)

            xn = (xn_2*fxn_1-xn_1*fxn_2)/(fxn_1-fxn_2)
            fxn = func(xn)
    return xn