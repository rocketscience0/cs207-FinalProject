"""Data structures for autodiff
"""

class Number():

    def __init__(self, val, deriv=None):

        self.val = val
        if deriv is None:
            self.deriv = {
                self: 1
            }
        else:
            self.deriv = deriv

    def __repr__(self):
        return f'Number(val={self.val})'