## How to use `autodiff`

The core data structure is a `Number`, which stores both a value and a dictionary of derivatives. After instantiation, a number's derivative with respect to itself will be automatically set to `1` unless otherwise specified:

```python
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
···

Using elementary operations will update derivatives according to the chain rule:
```python
>>> import autodiff
>>> x = autodiff.Number(3)
>>> y = x**2
>>> y.value
9
>>> y.deriv[x]
6
>>> y.deriv[y]
1
```

Note that the `deriv` attribute is a dict storing partial derivatives with respect to each `Number` object involved in preceding elementary operations. 

When any elementary operation takes in two `Number()` objects, that elementary operation will return a `Number()` with a partial derivative with respect to every key of both `Number()` objects:

```python
>>> x = autodiff.Number(2)
>>> y = autodiff.Number(3)
>>> def f(x, y, a=3):
>>>     return a * x * y
>>> q = f(x, y, a=3)
>>> q.deriv[x]
9
>>> q.deriv[y]
6
>>> q.deriv
{Number(value=2): 9, Number(value=3): 6}
```

Similarly, `autodiff` can work with vector functions of scalars. In these cases, each value in `deriv` is an array with the same shape as the output vector:

```python
x = autodiff.Number(np.pi / 2)
y = autodiff.Number(3 * np.pi / 2)

def f(x, y):
    return autodiff.array((
        y * autodiff.sin(x),
        x * autodiff.sin(y)
    ))
q = f(x,y)
```
```python
>>> q.deriv[x]
autodiff.array([0, 1])
>>> q.deriv[y]
autodiff.array([1, 0])
```

The `autodiff` package also works for scalar functions of vectors and vector functions of vectors, which behave the same.

Of course, most users will like to work with Jacobians and gradients rather than a `dict` of partial derivatives. Doing so is simple through the `jacobian` method. When an expression returns a scalar, `jacobian` will return that expression's gradient. When an expression returns a vector, `jacobian` will return that expression's Jacobian as a two-dimensional array.

```python
>>> x = autodiff.array((1, 2))
>>> y = autodiff.array((3, 4))
>>> q.jacobian((*x, *y))
autodiff.array([3, 4, 1, 2])
>>> q.jacobian((*x, *y)).shape
(4,)
```

Or with a vector function:
```python
>>> x = autodiff.Number(np.pi / 2)
>>> y = autodiff.Number(3 * np.pi / 2)

>>> def f(x, y):
        return autodiff.array((
            y * autodiff.sin(x),
            x * autodiff.sin(y)
        ))

>>> q = f(x,y)
>>> q.jacobian((x, y))
autodiff.array([[0, 1],
                [1, 0]])
>>> q.jacobian((x, y)).shape
(2, 2)
```
Note that `autodiff.Number.jacobian()` does require the user to specify an order of input `Number` objects to ensure consistency within the user's own code. Otherwise, `autodiff` would have to infer which element belongs to which function input. As the user strings together multiple elementary operations, it is likely that `autodiff`'s understanding would differ from the user's. An example of the suggested usage is:

```python
x = autodiff.Number(2)
y = autodiff.Number(3)
z = autodiff.Number(4)

order = (x, y, z)

# The gradient of f1 and f2 do not have an inherent order.
# If we displayed Numbers in the order they were used, the implied order would be
# (grad_x, grad_z, grad_y)---likely not what the user desires.
f1 = x**z
f2 = f1 * x * y
```
```python
>>> f2.deriv[x]
240
>>> f2.deriv[y]
32
>>> f2.deriv[z]
66.542
>>> f2.jacobian((order))
autodiff.array([240, 32, 66.542])
```

Right now, 'Number' supports basic operations, including addition, subtraction, multiplication, division, power, sin, cos, tan, exponential, log, and negation. 