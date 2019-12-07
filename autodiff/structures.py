"""Data structures for autodiff
"""
from autodiff import operations

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

        jacobian = []
        try:
            for key in order:
                jacobian.append(_partial(self.deriv, key))
        except TypeError:
            # The user specified a scalar order
            jacobian = _partial(self.deriv, order)

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
                deriv_self = self.deriv.copy()
                deriv_other = other.deriv.copy()
                deriv_self.pop(self)
                deriv_other.pop(other)
                if deriv_self==deriv_other:
                    return True
            return False
        except Exception:
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
        if self==other:
            return False
        return True
    
    def __gt__(self, other):
        '''
        Overloads the Comparison Operator to check whether this autodiff.Number 
        object is greater than the other one
        
        Args:
            other: the other autodiff.Number object to be compared with
        
        Returns:
            True if this autodiff.Number has a greater value, False otherwise.
            
        Raises:
            exception when the other is not an autodiff.Number object
        '''
        try:
            if (self.val > other.val):
                return True
            return False
        except Exception:
            raise Exception('cannot compare autodiff.Number object with a non-Number object')
        
    def __lt__(self, other):
        '''
        Overloads the Comparison Operator to check whether this autodiff.Number 
        object is less than the other one
        
        Args:
            other: the other autodiff.Number object to be compared with
        
        Returns:
            True if this autodiff.Number has a smaller value, False otherwise.
            
        Raises:
            exception when the other is not an autodiff.Number object
        '''
        try:
            if (self.val < other.val):
                return True
            return False
        except Exception:
            raise Exception('cannot compare autodiff.Number object with a non-Number object')


class ArrayIterator:
    def __init__(self, numbers):
        self.numbers = numbers
        self.index = 0

    def __next__(self):
        try:
            number = self.numbers[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1 
        return number

    def __iter__(self):
        return self


class Array:

    def __init__(self, iterable):
        self._lst = []
        flat_iterable = np.array(iterable).flatten()
        for elt in flat_iterable:
            # Check if element is number type
            if not isinstance(elt, Number):
                number_elt = Number(elt)
            else:
                number_elt = elt
            self._lst.append(number_elt)
        self._lst = np.array(self._lst).reshape(np.shape(iterable))

    def __len__(self):
        return len(self._lst)

    def __repr__(self):
        return f'Array({self._lst})'

    def __getitem__(self, idx):
        return self._lst[idx]

    def __setitem__(self, idx, val):
        self.__setattr__(idx, val)

    def __delitem__(self, idx):
        self.__delattr__(idx)

    def __iter__(self):
        return ArrayIterator(self._lst)

    # Element-wise addition of 2 vectors or 1 vector, 1 scalar
    def add(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.add(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.add(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.add(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Concatenate
    def __add__(self, other):
        if isinstance(other, Array):
            out = np.append(self._lst,other._lst)
        else:
            try: 
                is_iterable = iter(other)
            except:
                raise TypeError("can only concatenate iterable s(not {})".format(type(other)))
        return Array(out)
    
    def __radd__(self, other):
        if isinstance(other, Array):
            out = np.append(self._lst,other._lst)
        else:
            try: 
                is_iterable = iter(other)
            except:
                raise TypeError("can only concatenate iterable s(not {})".format(type(other)))
        return Array(out)

    # Element-wise subtraction of 2 vectors or 1 vector, 1 scalar
    def subtract(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.subtract(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Element-wise subtraction of 2 vectors or 1 vector, 1 scalar
    def __sub__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.subtract(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))
    
    def __rsub__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.subtract(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.subtract(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Scalar multiplication or element-wise multiplication
    def __mul__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.mul(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    def __rmul__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.mul(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Element-wise multiplication, Numpy-specific
    def multiply(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                raise TypeError("{} object is not subscriptable".format(type(other)))
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.mul(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Can only separate 2D arrays
    # TODO: NOT WORKING RN
    # def __matmul__(self, other):
    #     out = None
    #     shape = None
    #     if isinstance(other, Array):
    #         if np.shape(self._lst)[1] == np.shape(other._lst)[0]:
    #             shape = (np.shape(self._lst)[0], np.shape(other._lst)[1])
    #             out = np.full(shape,Number(0))
    #             for row_idx in range(shape[0]):
    #                 for col_idx in range(shape[1]):
    #                     new = Number(0)
    #                     for prod_idx in range(np.shape(self._lst)[1]):
    #                         prod = operations.mul(self._lst[row_idx, prod_idx], other._lst[prod_idx, col_idx])
    #                         new = operations.add(new, prod)
    #                     out[row_idx][col_idx] = new
    #         else:
    #             raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
    #     else:
    #         try:
    #             is_iterable = iter(other)
    #         except:
    #             raise TypeError("{} object is not subscriptable".format(type(other)))
    #         if np.shape(self._lst)[1] == np.shape(other)[0]:
    #             shape = (np.shape(self._lst)[0], np.shape(other)[1])
    #             out = np.array(np.shape(shape))
    #             for row_idx in range(shape[0]):
    #                 for col_idx in range(shape[1]):
    #                     new = Number(0)
    #                     for prod_idx in range(np.shape(self._lst)[1]):
    #                         prod = operations.mul(self._lst[row_idx, prod_idx], other[prod_idx, col_idx])
    #                         new = operations.add(new, prod)
    #                     out[row_idx, col_idx] = new
    #         else:
    #             raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
    #     return Array(out)  

    # def __rmatmul__(self, other):     

    # Element-wise division, Numpy-specific
    def divide(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                raise TypeError("{} object is not subscriptable".format(type(other)))
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Scalar or element-wise division
    def __truediv__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.div(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    def __rtruediv__(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.div(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.div(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Scalar or element-wise power, like Numpy
    def power(self, other):
        out = []
        if isinstance(other, Array):
            if np.shape(self._lst) == np.shape(other._lst):
                for self_elt,other_elt in zip(self._lst.flatten(), other._lst.flatten()):
                    out.append(operations.power(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other._lst)))
        else:
            try:
                is_iterable = iter(other)
            except:
                for elt in self._lst:
                    out.append(operations.power(elt,other))
                return Array(out)
            if np.shape(self._lst) == np.shape(other):
                other_flattened = np.array(other).flatten()
                for self_elt,other_elt in zip(self._lst.flatten(), other_flattened):
                    out.append(operations.power(self_elt,other_elt))
            else:
                raise ValueError("operands could not be broadcast together with shapes {} and {}".format(np.shape(self._lst),np.shape(other)))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    def __pow__(self, other):
        raise TypeError("unsupported operand type(s) for ** or pow(): {} and {}".format(type(self), type(other)))

    def __rpow__(self, other):
        raise TypeError("unsupported operand type(s) for ** or pow(): {} and {}".format(type(self), type(other)))

    def __neg__(self):
        out = []
        flat_iterable = self._lst.flatten()
        for elt in flat_iterable:
            out.append(operations.negate(elt))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    def sin(self):
        out = []
        flat_iterable = self._lst.flatten()
        for elt in flat_iterable:
            out.append(operations.sin(elt))
        return Array(np.array(out).reshape(np.shape(self._lst)))
    
    def cos(self):
        out = []
        flat_iterable = self._lst.flatten()
        for elt in flat_iterable:
            out.append(operations.cos(elt))
        return Array(np.array(out).reshape(np.shape(self._lst)))
    
    def tan(self):
        out = []
        flat_iterable = self._lst.flatten()
        for elt in flat_iterable:
            out.append(operations.tan(elt))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    def exp(self):
        out = []
        flat_iterable = self._lst.flatten()
        for elt in flat_iterable:
            out.append(operations.exp(elt))
        return Array(np.array(out).reshape(np.shape(self._lst)))

    # Summing across all elements in a vector, regardless of axis
    def sum(self):
        out = np.sum(self._lst).val
        return Number(out)

    # Dot product
    def dot(self, other):
        product = self.multiply(other)
        out = product.sum()
        return out
