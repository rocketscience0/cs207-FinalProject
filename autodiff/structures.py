"""Data structures for autodiff
"""
from autodiff import operations

class Number():
    '''
    Number class is the core data structure for 'autodiff'. It instantiates a Number 
    object by specifying a value and a derivative. Derivative with respect to itself 
    is automatically instantiated to 1.
    
    Args:
        val, value of the Number
        deriv, a dictionary of partial derivatives. It is automatically instantiated to
            {self: 1} unless otherwise specified.
    
    Returns:
        Number, an object to perform automatic differentiation on.
        
    Example:
        >>> import autodiff
        >>> x = autodiff.Number(3)
        >>> x.value
        3
        >>> x.deriv[x]
        1
        >>> a = autodiff.Number(3,2)
        >>> a.value
        3
        >>> a.deriv[a]
        2
    '''
    
    def __init__(self, val, deriv=None):

        self.val = val
        if deriv is None:
            self.deriv = {
                self: 1
            }
        elif isinstance(deriv, dict):
            self.deriv = deriv
            #keep also a copy of the derivative w.r.t. itself
            self.deriv[self] = 1
        else:
            self.deriv = {
                    self: deriv
                    }

    def __repr__(self):
        '''
        Overloads the print method to give a string representation of Number object.
        
        Returns:
            a string specifiying the value of the Number object.
        '''
        return f'Number(val={self.val})'
    
    def __add__(self, other):
        '''
        Overloads the add method to add a number object to another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)
    
    def __radd__(self, other):
        '''
        Overloads the right add method to add a number object to another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)
    
    def __sub__(self, other):
        '''
        Overloads the subtract method to subtract from a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be subtracted
        
        Returns:
            another Number object, which is the difference
        '''
        return operations.subtract(self, other)
    
    def __rsub__(self, other):
        '''
        Overloads the right subtract method to subtract from a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be subtracted
        
        Returns:
            another Number object, which is the difference
        '''
        return -operations.subtract(self, other)

    def __mul__(self, other):
        '''
        Overloads the multiply method to multiply a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be multiplied
        
        Returns:
            another Number object, which is the product
        '''
        return operations.mul(self, other)
    
    def __rmul__(self, other):
        '''
        Overloads the right multiply method to multiply a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be multiplied
        
        Returns:
            another Number object, which is the product
        '''
        return operations.mul(self, other)
    
    def __truediv__(self, other):
        '''
        Overloads the division method to divide a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be divided
        
        Returns:
            another Number object, which is the quotient
        '''
        return operations.div(self, other)

    def __rtruediv__(self, other):
        '''
        Overloads the right division method to divide a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be divided
        
        Returns:
            another Number object, which is the quotient
        '''
        return operations.div(self, other) ** -1
    
    def __pow__(self, other):
        '''
        Overloads the power method to power a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be the power
        
        Returns:
            another Number object, which is the original number ** other
        '''
        return operations.power(self, other)

    def __rpow__(self, other):
        '''
        Overloads the right power method to power a number object by another Number object or an integer/float.
        
        Args:
            other, a Number object or an integer or float to be the power
        
        Returns:
            another Number object, which is the original number ** other
        '''
        return operations.power(other, self)

    def __neg__(self):
        '''
        Overloads the negation method to negate a number object
        
        Returns:
            another Number object, which is the negation of the original number
        '''
        return operations.negate(self)
    
    def sin(self):
        '''
        Calculates the sin of the Number object.
        
        Returns:
            another Number object, which is sin of the original one.
        '''
        return operations.sin(self)
    
    def cos(self):
        '''
        Calculates the cosine of the Number object.
        
        Returns:
            another Number object, which is cosine of the original one.
        '''
        return operations.cos(self)
    
    def tan(self):
        '''
        Calculates the tangent of the Number object.
        
        Returns:
            another Number object, which is tangent of the original one.
        '''
        return operations.tan(self)

    def exp(self):
        '''
        Calculates the exponential of Number object.
        
        Returns:
            another Number object, which is the exponential of the original one.
        '''
        return operations.exp(self)

    def jacobian(self, order):
        '''
        Returns the jacobian matrix by the order specified.
        
        Args:
            order, the order to return the jacobian matrix in. Has to be not null
        
        Returns:
            a list of partial derivatives specified by the order.
        '''

        def _partial(deriv, key):
            try:
                return deriv[key]
            except KeyError:
                raise ValueError(
                    f'No derivative with respect to {repr(order)}'
                )

        jacobian = []
        try:
            for key in order:
                jacobian.append(_partial(self.deriv, key))
        except TypeError:
            # The user specified a scalar order
            jacobian = _partial(self.deriv, order)

        return jacobian