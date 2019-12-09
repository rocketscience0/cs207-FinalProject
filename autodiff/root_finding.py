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
        if isinstance(func(initial_guess),tuple):
            for i in range(iterations):

                if i == 0:
                    x0= initial_guess
                else:
                    x0 = x1

                fxn = func(x0)
                fpxn = []
                if verbose:
                    print(i,x0,fxn)

                vector = []
                for n in range(len(fxn)):
                    vector.append(fxn[n].val)


                if np.linalg.norm(vector)< tolerance:
                    break

                for k in range(len(fxn)):
                    fpxn_row = []
                    for j in range(len(x0)):
                        fpxn_row.append(fxn[k].jacobian(x0[j]))
                    fpxn.append(fpxn_row)
                jacobians.append(fpxn)
                x1 = x0 - np.dot(np.linalg.inv(fpxn),fxn)

            if show_fxn:
                return x1, jacobians,fxn
            else:
                return x1, jacobians
            
        else:
            for i in range(iterations):

                if i == 0:
                    x0= initial_guess
                else:
                    x0 = x1

                fxn = func(x0)
                fpxn = []
                if verbose:
                    print(i,x0,fxn)

                if abs(fxn.val)< tolerance:
                    break

                for j in range(len(x0)):
                    fpxn.append(fxn.jacobian(x0[j]))
                jacobians.append(fpxn)
                print(fpxn)
                x1 = x0 - np.dot(np.reciprocal(fpxn),fxn)

            if show_fxn:
                return x1, jacobians,fxn
            else:
                return x1, jacobians
