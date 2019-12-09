"""Data structures for autodiff
"""
from autodiff import operations
import numpy as np

class Number():
    '''
    Number class is the core data structure for 'autodiff'. It instantiates a Number 
    object by specifying a value and a derivative. Derivative with respect to itself 
    is automatically instantiated to 1.
    
    Args:
        val: value of the Number
        deriv: a dictionary of partial derivatives. It is automatically instantiated to
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
            self._deriv = {
                self: 1
            }
        elif isinstance(deriv, dict):
            self._deriv = deriv
            #keep also a copy of the derivative w.r.t. itself
            self._deriv[self] = 1
        else:
            self._deriv = {
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
            other: a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)
    
    def __radd__(self, other):
        '''
        Overloads the right add method to add a number object to another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be added.
        
        Returns:
            another Number object, which is the sum.
        '''
        return operations.add(self, other)
    
    def __sub__(self, other):
        '''
        Overloads the subtract method to subtract from a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be subtracted
        
        Returns:
            another Number object, which is the difference
        '''
        return operations.subtract(self, other)
    
    def __rsub__(self, other):
        '''
        Overloads the right subtract method to subtract from a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be subtracted
        
        Returns:
            another Number object, which is the difference
        '''
        return -operations.subtract(self, other)

    def __mul__(self, other):
        '''
        Overloads the multiply method to multiply a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be multiplied
        
        Returns:
            another Number object, which is the product
        '''
        return operations.mul(self, other)
    
    def __rmul__(self, other):
        '''
        Overloads the right multiply method to multiply a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be multiplied
        
        Returns:
            another Number object, which is the product
        '''
        return operations.mul(self, other)
    
    def __truediv__(self, other):
        '''
        Overloads the division method to divide a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be divided
        
        Returns:
            another Number object, which is the quotient
        '''
        return operations.div(self, other)

    def __rtruediv__(self, other):
        '''
        Overloads the right division method to divide a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be divided
        
        Returns:
            another Number object, which is the quotient
        '''
        return operations.div(self, other) ** -1
    
    def __pow__(self, other):
        '''
        Overloads the power method to power a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be the power
        
        Returns:
            another Number object, which is the original number ** other
        '''
        return operations.power(self, other)

    def __rpow__(self, other):
        '''
        Overloads the right power method to power a number object by another Number object or an integer/float.
        
        Args:
            other: a Number object or an integer or float to be the power
        
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

    def logistic(self):
        
        '''
        Calculates the logistic of Number object.
        
        Returns:
            another Number object, which is the logistic of the original one.
        '''

        return operations.logistic(self)

    def jacobian(self, order):
        '''
        Returns the jacobian matrix by the order specified.
        
        Args:
            order: the order to return the jacobian matrix in. Has to be not null
        
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
        
        #if order is a single Number object
        if isinstance(order, Number):
            return _partial(self._deriv, order)
        
        jacobian = []
        for key in order:
            jacobian.append(_partial(self._deriv, key))
        return jacobian

    def __hash__(self):
        return id(self)
  
    def __eq__(self, other):
        '''
        Overloads the Comparison Operator to check whether two autodiff.Number 
        objects are equal to each other
   
        Args:
            other: the other autodiff.Number object to be compared with
   
        Returns:
            True if two Autodiff.Number objects are equal, False otherwise.
        '''
        try:
            if self.val == other.val:
                deriv_self = self._deriv.copy()
                deriv_other = other._deriv.copy()
                deriv_self.pop(self)
                deriv_other.pop(other)
                if deriv_self==deriv_other:
                    return True
            return False
        except AttributeError:
            #if other is not even a autodiff.Number
            return False
   
    def __ne__(self, other):
        '''
        Overloads the Comparison Operator to check whether two autodiff.Number 
        objects are not equal to each other
   
        Args:
            other: the other autodiff.Number object to be compared with
   
        Returns:
            True if two Autodiff.Number objects are not equal, False otherwise.
        '''
        return not self.__eq__(other)
   
#    def __gt__(self, other):
#        '''
#        Overloads the Comparison Operator to check whether this autodiff.Number 
#        object is greater than the other one
       
#        Args:
#            other: the other autodiff.Number object to be compared with
       
#        Returns:
#            True if this autodiff.Number has a greater value, False otherwise.
           
#        Raises:
#            exception when the other is not an autodiff.Number object
#        '''
#        try:
#            if (self.val > other.val):
#                return True
#            return False
#        except Exception:
#            raise Exception('cannot compare autodiff.Number object with a non-Number object')
   
#    def __lt__(self, other):
#        '''
#        Overloads the Comparison Operator to check whether this autodiff.Number 
#        object is less than the other one
       
#        Args:
#            other: the other autodiff.Number object to be compared with
       
#        Returns:
#            True if this autodiff.Number has a smaller value, False otherwise.
           
#        Raises:
#            exception when the other is not an autodiff.Number object
#        '''
#        try:
#            if (self.val < other.val):
#                return True
#            return False
#        except Exception:
#            raise Exception('cannot compare autodiff.Number object with a non-Number object')



class Array():

    def __init__(self, iterable):
        self._data = np.array(iterable, dtype=np.object)
        if not all(map(lambda i: isinstance(i, Number), self._data)):
            raise ValueError('all elements of iterable must be of Number() type')

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f'Array({self._data})'

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        if isinstance(val, Number):
            self._data[idx] = val
        else:
            raise ValueError('invalid literal for Number(): {}'.format(val))

    def __add__(self, other):
        try:
            # If they're both Array2 objects
            return Array(self._data.__add__(other._data))
        except AttributeError:
            return Array(self._data.__add__(other))

    def __radd__(self, other):
        return self._data.__radd__(other)

    def __sub__(self, other):
        try:
            # If they're both Array2 objects
            return Array(self._data.__sub__(other._data))
        except AttributeError:
            return Array(self._data.__sub__(other))

    def __rsub__(self, other):
        return self._data.__rsub__(other)

    def __mul__(self, other):
        try:
            # If they're both Array2 objects
            return Array(self._data.__mul__(other._data))
        except AttributeError:
            return Array(self._data.__mul__(other))

    def __rmul__(self, other):
        return self._data.__rmul__(other)

    def __truediv__(self, other):
        try:
            # If they're both Array2 objects
            return Array(self._data.__truediv__(other._data))
        except AttributeError:
            return Array(self._data.__truediv__(other))

    def __rtruediv__(self, other):
        return self._data.__rtruediv__(other)

    def __matmul__(self, other):
        try:
            # If they're both Array objects
            out = self._data.__matmul__(other._data)
        except AttributeError:
            out = self._data.__matmul__(other)
        return(out)

    def __rmatmul__(self, other):
        return self._data.__rmatmul__(other)

    def __pow__(self, other):
        try:
            # If they're both Array2 objects
            return Array(self._data.__pow__(other._data))
        except AttributeError:
            return Array(self._data.__pow__(other))

    def __rpow__(self, other):
        return self._data.__rpow__(other)

    def __neg__(self):
        return Array(self._data.__neg__())
    
    def sin(self):
        return operations.sin(self)
    
    def cos(self):
        return operations.cos(self)
    
    def tan(self):
        return operations.tan(self)
    
    def exp(self):
        return operations.exp(self)
    
    def logistic(self):
        return operations.logistic(self)

    def jacobian(self, order):
        def _partial(deriv, key):
            try:
                return deriv[key]
            except KeyError:
                # If there's no partial, it's zero
                return 0

        jacobian = []
        for element in self._data:
            # Check if order is iterable
            try:
                jacobian_row = []
                for key in order:
                    jacobian_row.append(_partial(element._deriv, key))
                jacobian.append(jacobian_row)
            except TypeError:
                jacobian.append(_partial(element._deriv, order))
        
        return np.array(jacobian)

if __name__ == '__main__':
    q = Array((Number(1), Number(2)))
    print(q.jacobian(q._data))  # eye(2)
    print(q.jacobian(q._data[0]))  # [1, 0].T