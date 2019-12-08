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
    print('checkpoint1',type(initial_guess))
    print('checkpoint2',isinstance(initial_guess,Array))
    if isinstance(initial_guess,Number): 
    #bfgs for scalar functions
    
        x0 = initial_guess
        
        #initial guess of hessian
        b0 = 1
        
        fxn0 = func(x0)

        fpxn0 = fxn0.jacobian(x0)
        
        jacobians = []
        
        jacobians.append(fpxn0)
        

        for i in range(iterations):
            # need a stopping criterion
            fxn0 = func(x0)

            fpxn0 = fxn0.jacobian(x0)

            s0 = -fpxn0/b0

            x1=x0+s0 
                
            fxn1 = func(x1)
            fpxn1 = fxn1.jacobian(x1)
                
                
            y0 = fpxn1-fpxn0
                
            if y0 == 0:
                break
                    
            delta_b = y0/s0-b0
            b1 = b0 + delta_b
                
            x0 = x1
                
            b0 = b1
                
            jacobians.append(fpxn1)

        return x0,func(x0),jacobians

        print('checkpoint1',type(initial_guess))

    if isinstance(initial_guess,Array):
        
        print('checkpoint2','asdasdasdasdasd')

        x0 = initial_guess

        #initial guess of hessian
        h0 = np.identity(len(x0))

        fxn0 = func(x0)

        fpxn0 = fxn0.jacobian(x0)

        jacobians = []

        jacobians.append(fpxn0)

        s0 = -np.dot(h0,fpxn0) #np.array multiply with scalar would be fine

        x1 = x0 + s0

        fxn1 = func(x1)

        fpxn1 = fxn1.jacobian(x1)

        y0 = np.array(fpxn1 - fpxn0)

        rho0 = 1/(y0.T*s0)
        intermediate_1 = np.identity(len(x0))-np.dot(np.dot(s0,rho0),y0.T)
        intermediate_2 = np.dot(np.dot(intermediate_1,h0),intermediate_1)
        intermediate_3 = np.dot(np.dot(rho0,s0),s0.T)

        delta_H = intermediate_2 + intermediate_3


        for i in range(iterations):
            #need a stopping criterion
            jacobians.append(fpxn1)

            h0 = h0 + delta_H
            x0 = x1 
            fxn0 = func(x0)
            fpxn0 = fxn0.jacobian(x0)

            s0 = -np.dot(h0,fpxn0) #np.array multiply with scalar would be fine
            x1 = x0 + s0
            fxn1 = func(x1)
            fpxn1 = fxn1.jacobian(x1)
            y0 = np.array(fpxn1 - fpxn0)

            rho = 1/(y0.T*s0)
            intermediate_1 = np.identity(len(x0))-np.dot(np.dot(s0,rho0),y0.T)
            intermediate_2 = np.dot(np.dot(intermediate_1,h0),intermediate_1)
            intermediate_3 = np.dot(np.dot(rho0,s0),s0.T)
            delta_H = intermediate_2 + intermediate_3

        return x0,func(x0),jacobians
  

def gradient_descent(func,initial_guess,iterations = 100,step_size=0.01):

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
    #gradient descent for scalar functions
    if isinstance(initial_guess,Number):
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

    elif isinstance(initial_guess,Array):

        # e.g. R2 --> R1
        x0=initial_guess
        jacobians = []
        s = -np.array(func(initial_guess).jacobian(x0))#negative sign cannot be used on a list, so turn into a np array
        jacobians.append(s)

        for i in range(iterations):

            #if np.abs(s.sum())>10**-7:
            for j in range(len(s)):
                x0 = np.array(x0).reshape(-1,1) + np.array(step_size*s[j]).reshape(-1,1)

                x0 = Array(x0.reshape(len(x0)))
            print(i,x0)
            s = -np.array(func(x0).jacobian(x0))

            jacobians.append(s)
        return x0,func(x0),jacobians

        
        
    
