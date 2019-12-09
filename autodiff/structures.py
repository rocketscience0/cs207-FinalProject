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
        >>> x = autodiff.structures.Number(3)
        >>> x.value
        3
        >>> x.jacobian(x)
        1
        >>> a = autodiff.structures.Number(3,2)
        >>> a.value
        3
        >>> a.jacobian(x)
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
    
    def asin(self):
        '''
        Calculates the arcsin of the Number object.
        
        Returns:
            another Number object, which is arcsin of the original one.
        '''
        return operations.asin(self)
    
    def sinh(self):
        '''
        Calculates the sinh of the Number object.
        
        Returns:
            another Number object, which is sinh of the original one.
        '''
        return operations.sinh(self)
    
    def cos(self):
        '''
        Calculates the cosine of the Number object.
        
        Returns:
            another Number object, which is cosine of the original one.
        '''
        return operations.cos(self)
    
    def acos(self):
        '''
        Calculates the arccosine of the Number object.
        
        Returns:
            another Number object, which is arccosine of the original one.
        '''
        return operations.acos(self)
    
    def cosh(self):
        '''
        Calculates the cosine-h of the Number object.
        
        Returns:
            another Number object, which is cosine-h of the original one.
        '''
        return operations.cosh(self)
    
    def tan(self):
        '''
        Calculates the tangent of the Number object.
        
        Returns:
            another Number object, which is tangent of the original one.
        '''
        return operations.tan(self)
    
    def atan(self):
        '''
        Calculates the arc-tangent of the Number object.
        
        Returns:
            another Number object, which is arc-tangent of the original one.
        '''
        return operations.atan(self)
    
    def tanh(self):
        '''
        Calculates the tangent-h of the Number object.
        
        Returns:
            another Number object, which is tangent-h of the original one.
        '''
        return operations.tanh(self)
    
    def sqrt(self):
        '''
        calculates the square root of the Number object.
        
        Returns:
            another number object, which is the square root of the original one.
        '''
        return operations.sqrt(self)
    

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
    
    def log(self, base = np.exp(1)):
        '''
        Calculates the log of Number object.
        
        Args:
            base: the base to take log with
        
        Returns:
            another Number object, which is the log of the original one.
        '''

        return operations.log(self, base)

    def jacobian(self, order):
        '''
        Returns the jacobian matrix by the order specified.
        
        Args:
            order: the order to return the jacobian matrix in. Has to be not null
        
        Returns:
            a np.ndarray of partial derivatives specified by the order.
            When order is a single element, it returns a scaler
        '''
        
        def _partial(deriv, key):
            try:
                return deriv[key]
            except KeyError:
                # If there's no partial, it's zero
                return 0
        
        #if order is a single Number object
        if isinstance(order, Number):
            return _partial(self._deriv, order)
        
        jacobian = []
        for key in order:
            jacobian.append(_partial(self._deriv, key))
        return np.array(jacobian)

    def __hash__(self):
        return id(self)
  
    def __eq__(self, other):
        '''
        Overloads the Comparison Operator to check whether two Number 
        objects are equal to each other
   
        Args:
            other: the other Number object to be compared with
   
        Returns:
            True if two Number objects are equal, False otherwise.
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
        Overloads the Comparison Operator to check whether two Number 
        objects are not equal to each other
   
        Args:
            other: the other Number object to be compared with
   
        Returns:
            True if two Number objects are not equal, False otherwise.
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
    '''
    Array class is another core data structure for 'autodiff'. It instantiates an Array 
    object as a list of Number objects. It holds the list internally as an np.ndarray
    
    Args:
        iterable: a iterable objects of Numbers to be held in Array. If lists of 
        ints/floats are passed, it will convert them to Number objects first 
    
    Returns:
        Array, an object to perform automatic differentiation on.
        
    Example:
        >>> import autodiff
        >>> x = autodiff.structures.Number(3)
        >>> a = autodiff.structures.Number(3,2)
        >>> arr = autodiff.structures.Array([x, a])
        Array([Number(val=3) Number(val=3)])
        >>> arr.jacobian(a)
        array([0, 2])
    '''

    def __init__(self, iterable):
        self._data = np.array(iterable, dtype=np.object)
        for i, d in enumerate(self._data):
            if not isinstance(d, Number):
                self._data[i] = Number(d)

    def __str__(self):
        '''
        Overloads the string method to give a string representation of Array object.
        
        Returns:
            a string specifiying the value of each Number object in Array.
        '''
        return str(self._data)

    def __repr__(self):
        '''
        Overloads the print method to give a string representation of Array object.
        
        Returns:
            a string specifiying the value of each Number object in Array.
        '''
        return f'Array({self._data})'

    def __len__(self):
        '''
        Overloads the len() method to give the length of Array.
        
        Returns:
            an integer representing number of elements in Array.
        '''
        return len(self._data)

    def __getitem__(self, idx):
        '''
        Overloads [idx] to get an item of the Array at a specific postion.
        
        Args:
            idx, the index to take element at.
        
        Returns:
            a Number object, which is at position 'idx' of the Array.
        '''
        return self._data[idx]

    def __setitem__(self, idx, val):
        '''
        Sets an item of the Array at a specific postion with the value specified.
        
        Args:
            idx, the index to set element at
            val, the Number object to put in Array
        '''
        if isinstance(val, Number):
            self._data[idx] = val
        else:
            raise ValueError('invalid literal for Number(): {}'.format(val))

    def __add__(self, other):
        '''
        Overloads addition to add two arrays, or a Number/int/float to an array.
        
        Args:
            other, another Array of same length to perform element-wise addition on,
            or a Number object to be added, or an integer/float to be added.
        
        Returns:
            an Array object, which is the sum.
        '''
        try:
            # If they're both Array2 objects
            return Array(self._data.__add__(other._data))
        except AttributeError:
            return Array(self._data.__add__(other))

    def __radd__(self, other):
        '''
        Overloads right addition to add two arrays, or a Number/int/float to an array.
        
        Args:
            other, another Array of same length to perform element-wise addition on,
            or a Number object to be added, or an integer/float to be added.
        
        Returns:
            an Array object, which is the sum.
        '''
        return self._data.__radd__(other)

    def __sub__(self, other):
        '''
        Overloads subtraction to subtract two arrays, or a Number/int/float from an array.
        
        Args:
            other, another Array of same length to perform element-wise subtraction on,
            or a Number object to be subtracted, or an integer/float to be subtracted.
        
        Returns:
            an Array object, which is the difference.
        '''
        try:
            # If they're both Array2 objects
            return Array(self._data.__sub__(other._data))
        except AttributeError:
            return Array(self._data.__sub__(other))

    def __rsub__(self, other):
        '''
        Overloads right subtraction to subtract two arrays, or a Number/int/float 
        from an array.
        
        Args:
            other, another Array of same length to perform element-wise subtraction on,
            or a Number object to be subtracted, or an integer/float to be subtracted.
        
        Returns:
            an Array object, which is the difference.
        '''
        return self._data.__rsub__(other)

    def __mul__(self, other):
        '''
        Overloads multiplication to multiply two arrays, or a Number/int/float to an array.
        
        Args:
            other, another Array of same length to perform element-wise multiplication on,
            or a Number object to be multiplied, or an integer/float to be multiplied.
        
        Returns:
            an Array object, which is the product.
        '''
        try:
            # If they're both Array2 objects
            return Array(self._data.__mul__(other._data))
        except AttributeError:
            return Array(self._data.__mul__(other))

    def __rmul__(self, other):
        '''
        Overloads right multiplication to multiply two arrays, or a Number/int/float 
        to an array.
        
        Args:
            other, another Array of same length to perform element-wise multiplication on,
            or a Number object to be multiplied, or an integer/float to be multiplied.
        
        Returns:
            an Array object, which is the product.
        '''
        return self._data.__rmul__(other)

    def __truediv__(self, other):
        '''
        Overloads division to divide two arrays, or a Number/int/float from an array.
        
        Args:
            other, another Array of same length to perform element-wise division on,
            or a Number object to divide by, or an integer/float to divide by.
        
        Returns:
            an Array object, which is the quotient.
        '''
        try:
            # If they're both Array2 objects
            return Array(self._data.__truediv__(other._data))
        except AttributeError:
            return Array(self._data.__truediv__(other))

    def __rtruediv__(self, other):
        '''
        Overloads right division to divide two arrays, or a Number/int/float from an array.
        
        Args:
            other, another Array of same length to perform element-wise division on,
            or a Number object to divide by, or an integer/float to divide by.
        
        Returns:
            an Array object, which is the quotient.
        '''
        return self._data.__rtruediv__(other)

    def __matmul__(self, other):
        '''
        Overloads matrix multiplication to multiply two matrices, 
        or a Number/int/float to a matrix.
        
        Args:
            other, another matrix to be multiplied,
            or a Number object to be multiplied, or an integer/float to be multiplied.
        
        Returns:
            an Array object, which is the product.
        '''
        try:
            # If they're both Array objects
            out = self._data.__matmul__(other._data)
        except AttributeError:
            out = self._data.__matmul__(other)
        return(out)

    def __rmatmul__(self, other):
        '''
        Overloads right matrix multiplication to multiply two matrices, 
        or a Number/int/float to a matrix.
        
        Args:
            other, another matrix to be multiplied,
            or a Number object to be multiplied, or an integer/float to be multiplied.
        
        Returns:
            an Array object, which is the product.
        '''
        return self._data.__rmatmul__(other)

    def __pow__(self, other):
        '''
        Overloads power to take the power of an Array object.
        
        Args:
            other, another Array of same dimension to perform element-wise power by,
            or a Number object to be take the power of, or an integer/float to take
            the power of.
        
        Returns:
            an Array object, which is the power.
        '''
        try:
            # If they're both Array2 objects
            return Array(self._data.__pow__(other._data))
        except AttributeError:
            return Array(self._data.__pow__(other))

    def __rpow__(self, other):
        '''
        Overloads right power to take the power of an Array object.
        
        Args:
            other, another Array of same dimension to perform element-wise power by,
            or a Number object to be take the power of, or an integer/float to take
            the power of.
        
        Returns:
            an Array object, which is the power.
        '''
        return self._data.__rpow__(other)

    def __neg__(self):
        '''
        Overloads negation on Array objects.
        
        Returns:
            another Array object, which is the negated original Array.
        '''
        return Array(self._data.__neg__())
    
    def sin(self):
        '''
        Calculates the sin of the Array object.
        
        Returns:
            another Array object, which is element-wise sin of the original one.
        '''
        return operations.sin(self)
    
    def asin(self):
        '''
        Calculates the arcsin of the Array object.
        
        Returns:
            another Array object, which is element-wise arcsin of the original one.
        '''
        return operations.asin(self)
    
    def sinh(self):
        '''
        Calculates the sinh of the Array object.
        
        Returns:
            another Array object, which is element-wise sinh of the original one.
        '''
        return operations.sinh(self)
    
    def cos(self):
        '''
        Calculates the cosine of the Array object.
        
        Returns:
            another Array object, which is element-wise cosine of the original one.
        '''
        return operations.cos(self)
    
    def acos(self):
        '''
        Calculates the arccosine of the Array object.
        
        Returns:
            another Array object, which is element-wise arccosine of the original one.
        '''
        return operations.acos(self)
    
    def cosh(self):
        '''
        Calculates the cosine-h of the Array object.
        
        Returns:
            another Array object, which is element-wise cosine-h of the original one.
        '''
        return operations.cosh(self)
    
    def tan(self):
        '''
        Calculates the tangent of the Array object.
        
        Returns:
            another Array object, which is element-wise tangent of the original one.
        '''
        return operations.tan(self)
    
    def atan(self):
        '''
        Calculates the arc-tangent of the Array object.
        
        Returns:
            another Array object, which is element-wise arc-tangent of the original one.
        '''
        return operations.atan(self)
    
    def tanh(self):
        '''
        Calculates the tangent-h of the Array object.
        
        Returns:
            another Array object, which is element-wise tangent-h of the original one.
        '''
        return operations.tanh(self)
    
    def sqrt(self):
        '''
        calculates the square root of the Array object.
        
        Returns:
            another Array object, which is the element-wise square root of the original one.
        '''
        return operations.sqrt(self)
    

    def exp(self):
        '''
        Calculates the exponential of Array object.
        
        Returns:
            another Array object, which is the element-wise exponential of the original one.
        '''
        return operations.exp(self)

    def logistic(self):
        
        '''
        Calculates the logistic of Array object.
        
        Returns:
            another Array object, which is the element-wise logistic of the original one.
        '''

        return operations.logistic(self)
    
    def log(self, base = np.exp(1)):
        '''
        Calculates the log of Array object.
        
        Args:
            base: the base to take log with
        
        Returns:
            another Array object, which is the element-wise log of the original one.
        '''

        return operations.log(self, base)
    
    def dot(self, other):
        '''
        Calculates the dot product of two Array objects.
        
        Args:
            other, another Array object of same length to take the dot product of
        
        Returns:
            a Number object, which is dot product of the two Arrays.
        '''
        return np.dot(self, other)

    def jacobian(self, order):
        '''
        Returns the jacobian matrix by the order specified.
        
        Args:
            order: the order to return the jacobian matrix in. Has to be not null
        
        Returns:
            a np.ndarray of partial derivatives specified by the order. Each row is
            an element in the original array, each column is the order specified.
            When order is a single element, it returns a flat array.
        '''
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
    
    def __eq__(self, other):
        '''
        Overloads the Comparison Operator to check whether two Array 
        objects are equal to each other
   
        Args:
            other: the other Array object to be compared with
   
        Returns:
            True if two Array objects are equal, False otherwise.
        '''
        try:
            initial = True
            if len(self)==len(other):
                for i in range(len(self)):
                    if not (self[i]==other[i]):
                        initial = False
            return initial
        except AttributeError:
            #if other is not even a autodiff.Number
            return False
   
    def __ne__(self, other):
        '''
        Overloads the Comparison Operator to check whether two Array 
        objects are not equal to each other
   
        Args:
            other: the other Array object to be compared with
   
        Returns:
            True if two Array objects are not equal, False otherwise.
        '''
        return not self.__eq__(other)

