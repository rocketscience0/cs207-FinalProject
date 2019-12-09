from autodiff import operations
from autodiff.structures import Number
from autodiff.structures import Array
import numpy as np
from copy import deepcopy

def newtons_method(func, initial_guess, iterations=100,tolerance = 10**-7,verbose = False, show_fxn=False):
    
    """Use Newton's method to find the root of the function
    Args:
        func: the function that the user wants to find root for
        initial_guess: a number object for the initial guess
        iterations: number of maximum iterations
        show_fxn: if true, return function value at the final xstar

    Returns:
        xn: the x value of the root
        jacobians: the jacobian at each step of the root_finding
        fxn: func(xn). If root_finding is successful, this value should be 0
    """    
    if isinstance(initial_guess,Number):
        #scalar case
        jacobians = []

        x0 = initial_guess

        fxn = func(initial_guess)

        fpxn = fxn.jacobian(initial_guess)

        x1 = x0 - fxn/fpxn

        jacobians.append(fpxn)

        for i in range(iterations):
            if abs(fxn.val) > 1e-7:
                x0 = x1

                fxn = func(x0)

                fpxn = fxn.jacobian(x0)

                jacobians.append(fpxn)

                x1 = x0- fxn / fpxn

        if show_fxn:
            return x1, jacobians,fxn
        else:
            return x1, jacobians
    elif isinstance(initial_guess,Array):
        jacobians = []
        
        for i in range(iterations):
            
            if i == 0:
                x0= initial_guess
            else:
                x0 = x1

            fxn = func(x0)
            fpxn = []
            if verbose:
                print(i,x0,fxn)
            if abs(fxn.val) < tolerance:
                break
                
            for i in range(len(x0)):
                fpxn.append(fxn.jacobian(x0[i]))

            jacobians.append(fpxn)
            #print(fpxn)
            x1 = []
            for j in range(len(x0)):
                x1.append(x0[j]-fxn/fpxn[j])
                
        if show_fxn:
            return x1, jacobians,fxn
        else:
            return x1, jacobians
        
def secant_method(func, initial_guess,iterations = 100,show_fxn=False):
    """Use secant method to find the root of the function
    Args:
        func: the function that the user wants to find root for
        initial_guess: a list of length=2 that containts two initial guesses close to the root
        iterations: number of maximum iterations
        show_fxn: if true, return function value at the final xstar

    Returns:
        xn: the x value of the root
        fxn: func(xn). If root_finding is successful, this value should be 0
    """

    #The secant method uses finite difference version of Newton's method
    # Aims to compare the efficiency with Newton's method

    if isinstance(initial_guess[0],Number):
        #for scalar case
        if len(initial_guess) != 2 | len(initial_guess) == None:
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
        if show_fxn:
            return xn,fxn
        else:
            return xn
    
    if isinstance(initial_guess[0],Array)|isinstance(initial_guess[0],np.ndarray):
        #For vector case
        if len(initial_guess) != 2 | len(initial_guess)==None:
            raise ValueError("Please enter two initial guesses")
    
        xn_2 = []
        xn_1 = []        
        for i in range(len(initial_guess[0])):
            xn_2.append(initial_guess[0][i].val)
            xn_1.append(initial_guess[1][i].val)
        
        xn_2 = np.array(xn_2)
        xn_1 = np.array(xn_1)
        print(type(xn_2))
        print(xn_1)
        fxn_2 = func(xn_2)
        fxn_1 = func(xn_1)
        print(type(fxn_2))
        
        xn = (xn_2*fxn_1-xn_1*fxn_2)/(fxn_1-fxn_2)
        print('xn',xn)
        fxn = func(xn)
        print('fxn',fxn)
        for i in range(iterations):
            if fxn !=0:
                xn_2=xn_1
                xn_1 = xn
                fxn_2 = func(xn_2)
                fxn_1 = func(xn_1)

                xn = (xn_2*fxn_1-xn_1*fxn_2)/(fxn_1-fxn_2)
                fxn = func(xn)
                print('xn',xn)
                print(fxn_2)

        if show_fxn:
            return xn,fxn
        else:
            return xn