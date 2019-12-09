import sys
sys.path.append('..')

import autodiff.operations as operations
from autodiff.structures import Number
from autodiff.structures import Array
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

    if isinstance(initial_guess,Array):
        
        jacobians = []

        for i in range(iterations):

            if i == 0:
                
                x0 = initial_guess
                #initial guess of hessian
                H = np.identity(len(x0))
            else:
                x0 = x1
                H = deltaH
                
            fxn0 = func(x0)
            fpxn0 = fxn0.jacobian(x0)     
            jacobians.append(fpxn0)
            if np.abs(fpxn0.all())<10**-7:
                #optimization condition is met
                break
                
            s = -np.dot(H,fpxn0) #np.array multiply with scalar would be fine
            x1 = x0 + s
            fxn1 = func(x1)
            fpxn1 = fxn1.jacobian(x1)
            y = np.array(fpxn1 - fpxn0)
            rho0 = 1/(np.dot(y.T,s))
            rhokykT = rho0*y.T
            skrhokykT = np.dot(s.reshape([len(x0),1]),rhokykT.reshape([1,len(x0)]))

            ykskT = np.dot(np.reshape(y,[len(x0),1]),s.reshape([len(x0),1]).T)

            rhokykskT = rho0*ykskT

            skskT = np.dot(s.reshape([len(x0),1]),s.reshape([len(x0),1]).T)

            rhokskskT = rho0*skskT

            #define delta H
            deltaH = np.dot((np.identity(2)-skrhokykT),np.dot(H,(np.identity(2)-rhokykskT)))+rhokskskT


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
        jacobians = []
        
        s = np.zeros(len(initial_guess))

        for i in range(iterations):
            if i == 0:
                x0 = initial_guess
            else:
                alpha = step_size
                #perform line search using Armijo condition
                while func(x0+alpha*s).val>func(x0).val+alpha*0.0001*np.dot(np.transpose(-1*s),s):
                    alpha = alpha/2
                
                x0 = x0+alpha*s
            
            
            for j in range(len(s)):
                s[j] = func(x0).jacobian(x0[j])*-1
            
            print(i,x0,func(x0))
            if np.abs(s).all()<10**-7:
                break
            jacobians.append(s)
        return i,x0,func(x0),jacobians
