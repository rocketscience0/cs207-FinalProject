## How to use `autodiff`

The core data structure is a `Number`, which stores both a value and a dictionary of derivatives. After instantiation, a number's derivative with respect to itself will be automatically set to `1` unless otherwise specified:

```python
>>> import autodiff
>>> x = autodiff.structures.Number(3)
>>> x.value
3
>>> x.deriv[x]
1
>>> a = autodiff.structures.Number(3,2)
>>> a.value
3
>>> a.deriv[a]
2
```
Also, note that `import autodiff` doesn't import `numpy` automatically. Therefore, if the user wants to use the `numpy` package, user will need to manually import it.

Using elementary operations will update derivatives according to the chain rule:
```python
>>> import autodiff
>>> x = autodiff.structures.Number(3)
>>> y = x**2
>>> y.value
9
>>> y.deriv[x]
6
>>> y.deriv[y]
1
```


Right now, 'Number' overloads basic operations, including addition, subtraction, multiplication, division and power.

To call other basic operations such as sin, cos, tan, exponential, log, and negation, simply call the following on Number:

```python
>>> x = autodiff.structures.Number(np.pi/6)
>>> y = x.sin()
>>> y.value
1/2
```

Note that the `deriv` attribute is a dict storing partial derivatives with respect to each `Number` object involved in preceding elementary operations. 

When any elementary operation takes in two `Number()` objects, that elementary operation will return a `Number()` with a partial derivative with respect to every key of both `Number()` objects:

```python
>>> x = autodiff.structures.Number(2)
>>> y = autodiff.structures.Number(3)
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
Of course, most users will like to work with Jacobians and gradients rather than a `dict` of partial derivatives. Doing so is simple through the `jacobian` method. When an expression returns a scalar, `jacobian` will return that expression's gradient. When an expression returns a vector, `jacobian` will return that expression's Jacobian as an `Array` object.

```python
>>> q.jacobian([x,y])
Array([Number(val=9) Number(val=6)])
```

`autodiff` also has an `Array` class that holds an array of `Number` object. 
To initialize an `Array`, user has to pass it an iterable object, such as an 1-d list or an 1-d np.array.

```python
>>> import autodiff
>>> import numpy as np
>>> x = autodiff.structures.Number(np.pi / 2)
>>> y = autodiff.structures.Number(3 * np.pi / 2)

>>> def f(x, y):
>>>     return autodiff.structures.Array((
>>>         y * x.sin(),
>>>         x * y.sin()
>>>     ))
>>> q = f(x,y)
>>> q
Array([Number(val=4.71238898038469) Number(val=-1.5707963267948966)])
```
`Array` objects doesn't hold `deriv` attributes as in the case with `Number`. To see each value's derivative with respect to some values, user can call `.jacobian(order)` method on Array, where `order` is a list of elements specified by the user, or a single element. It returns another Array object where each row is every element in array in original order, each column is the order specified as inputs.
Note that `autodiff.Number.jacobian()` does require the user to specify an order of input `Number` objects to ensure consistency within the user's own code. Otherwise, `autodiff` would have to infer which element belongs to which function input. As the user strings together multiple elementary operations, it is likely that `autodiff`'s understanding would differ from the user's. An example of the suggested usage is:
```python
>>> q.jacobian(x)
Array([[Number(val=2.8855060405826847e-16)]
 [Number(val=-1.0)]])
>>> q.jacobian([x,y])
Array([[Number(val=2.8855060405826847e-16) Number(val=1.0)]
 [Number(val=-1.0) Number(val=-2.8855060405826847e-16)]])
```
Alternatively, user can call `.deriv(element)` on specific elements in the `Array`.
```python
>>>q[0].deriv[y]
1.0
```

The `autodiff` package also works for scalar functions of vectors and vector functions of vectors, which behave the same.

autodiff.Array also overloads basic operations: addition, multiplication, division, and subtraction, and power. 
When multiplication, division, and power are called, they perform element-wise operations on Arrays.

```python
>>> x = autodiff.structures.Array((1, 2))
>>> y = autodiff.structures.Array((3, 4))
>>> q = x*y
>>>q
Array([Number(val=3) Number(val=8)])
```
If we're multiplying an `Array` by a `Number`, it multiply the same `Number` passed on every element of the array.
```python
>>> x = autodiff.structures.Array((1, 2))
>>> z = autodiff.structures.Number(2)
>>> q = x*z
>>> q
Array([Number(val=2) Number(val=4)])
```
When add is called, it performs a concatenation of two Arrays.
```python
>>> x = autodiff.structures.Array((1, 2))
>>> y = autodiff.structures.Array((3, 4))
>>> q = x+y
>>>q
Array([Number(val=1) Number(val=2) Number(val=3) Number(val=4)])
```
However, `subtract` and `-` will perform the same operation, which is element-wise subtraction.
```python
>>> x = autodiff.structures.Array((1, 2))
>>> y = autodiff.structures.Array((3, 4))
>>> q = x-y
>>>q
Array([Number(val=-2) Number(val=-2)])
```
`Array` also supports operations such as sin, cos, tan, and exp. When these operations are called, it is performed on each element of the Array object.
```python
>>> x = autodiff.structures.Array((1, 2))
>>> q = x.sin()
>>> q
Array([Number(val=0.8414709848078965) Number(val=0.9092974268256817)])
```
Also, it supports sum and dot methods as it is used in numpy.
```python
>>> x = autodiff.structures.Array((1, 2))
>>> q = x.sum()
Number(val=3)
```