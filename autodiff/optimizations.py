import sys
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
import numpy as np


def bfgs(func, initial_guess,iterations =100):
    """Use BFGS method to find the local minimum/maxinum of the function
    Args:
        func: the function that the user wants to optimize
        initial_guess: a Number object for the initial guess
        iterations: number of maximum iterations

    Returns:
        x0: the x value of the local extremum
        func(x0): the value of the local extremum
        jacobians: the jacobians of each optimization step
        """    
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
  

def steepest_descent(func,initial_guess,iterations = 100,step_size=0.01):

    """Use steepest_descent method to find the local minimum/maxinum of the function
    Args:
        func: the function that the user wants to optimize 
        initial_guess: A number object for the initial guess
        iterations: number of maximum iterations
        step_size: the size of each step

    Returns:
        x0: the x value of the local extremum
        func(x0): the value of the local extremum
        jacobians: the jacobians of each optimization step
        """    
    #bfgs for scalar functions

    x0=initial_guess
    jacobians = []
    s = -func(initial_guess).jacobian(x0)
    jacobians.append(s)
    for i in range(iterations):
        if np.abs(s)>1*10**-7:
            x0 = x0 + step_size*s
            s = -func(x0).jacobian(x0)
            jacobians.append(s)
    
    
    return x0,func(x0),jacobians

        
        
    
