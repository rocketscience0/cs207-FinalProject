"""Data structures for autodiff
"""
import operations

class Number():

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
        return f'Number(val={self.val})'
    
    def __add__(self, other):
        return operations.add(self, other)
    
    def __radd__(self, other):
        return operations.add(self, other)
    
    def __sub__(self, other):
        return operations.subtract(self, other)
    
    def __rsub__(self, other):
        return -operations.subtract(self, other)

    def __mul__(self, other):
         return operations.mul(self, other)
    
    def __rmul__(self, other):
        return operations.mul(self, other)
    
    def __truediv__(self, other):
        return operations.div(self, other)

    def __rtruediv__(self, other):
        return operations.div(self, other) ** -1
    
    def __rdiv__(self, other):
        return operations.div(self, other)
    
    def __pow__(self, other):
        return operations.power(self, other)

    def __neg__(self):
        return operations.negate(self)
    
    def sin(self):
        return operations.sin(self)
    
    def cos(self):
        return operations.cos(self)
    
    def tan(self):
        return operations.tan(self)

    def exp(self):
        return operations.exp(self)

    def log(self):
        return operations.log(self)

    def jacobian(self, order):
        return list(self.deriv.keys())[1].val