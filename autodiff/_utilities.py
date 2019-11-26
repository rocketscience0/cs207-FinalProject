"""Utilities for writing code
"""

import sympy as sp
x = sp.symbols('x')
def f_and_deriv(f, wrt=x):
    sp.printing.pprint('Function:')
    sp.printing.pprint(f)
    print('\n', sp.printing.pycode(f))
    sp.printing.pprint('\nDeriv:')
    sp.printing.pprint(f.diff(wrt))
    print('\n', sp.printing.pycode(f.diff(wrt)))

if __name__ == '__main__':
    f_and_deriv(sp.asin(x))