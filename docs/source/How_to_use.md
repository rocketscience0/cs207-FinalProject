## How to use `autodiff`

The core data structure is a `Number`, which stores both a value and a dictionary of derivatives, which is hidden from the user. After instantiation, a number's derivative with respect to itself will be automatically set to `1` unless otherwise specified.

To access the partial derivatives with respect to x, user could call `.jacobian(x)` method on the current Number.

To access the value of number, user could call `.val` on the current Number.

Number is initialized with `val`, and `deriv`. If you want to specify the derivative of the number w.r.t. itself, you could pass deriv as an integer or a float. If you have a list of derivatives to give to this `Number`, you could pass it as a dictionary

```python
>>> from autodiff.structures import Number
>>> import numpy as np
>>> x = Number(3)
>>> x.val
3
>>> x.jacobian(x)
1
>>> a = Number(3,2)
>>> a.val
3
>>> a.jacobian(a)
2
>>> b = Number(3, {x: 2, a:4})
>>> b.jacobian(a)
4
```

Also, note that `import autodiff` doesn't import `numpy` automatically. Therefore, if the user wants to use the `numpy` package, user will need to manually import it.

Using elementary operations will update derivatives according to the chain rule:

```python
>>> x = Number(3)
>>> y = x**2
>>> y.val
9
>>> y.jacobian(x)
6
>>> y.jacobian(y)
1
```


Right now, 'Number' overloads basic operations, including addition, subtraction, multiplication, division and power.

To call other basic operations such as sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, exponential, log, sqrt, and negation, simply call the following on Number:

```python
>>> x = Number(np.pi/6)
>>> y = x.sin()
>>> y.val
1/2
```

`exp` method assumes that the base is `e`. When it is not `e`, user could directly use the power method, such as `2**x`.

Similarly, log method assumes that the base is `e` when no parameter `base` is passed. When user wants to specify a base, he/she could do the following:

```python
>>> a = Number(3)
>>> a.log()
Number(val=1.0986122886681098)
>>> a.log(10)
Number(val=0.47712125471966244)
```

Note that the `_deriv` attribute is a dict storing partial derivatives with respect to each `Number` object involved in preceding elementary operations. `_deriv` is hidden from the user. User could only call `.jacobian()` to access the partial derivatives

When any elementary operation takes in two `Number()` objects, that elementary operation will return a `Number()` with a partial derivative with respect to every key of both `Number()` objects:

```python
>>> x = Number(2)
>>> y = Number(3)
>>> def f(x, y, a=3):
>>>     return a * x * y
>>> q = f(x, y, a=3)
>>> q.jacobian(x)
9
>>> q.jacobian(y)
6
```
User will also be able to call the jacobian with a list of elements to take partial derivatives with respect to. When an expression returns a scalar, `jacobian` will return that expression's gradient. When an expression returns a vector, `jacobian` will return that expression's Jacobian as a `np.ndarray` of integers/floats.

```python
>>> q.jacobian([x,y])
array([Number(val=9) Number(val=6)])
```

`autodiff` also has an `Array` class that holds an array of `Number` object. 
To initialize an `Array`, user has to pass it an iterable object, such as an 1-d list or an 1-d np.array.

When the user passes a list of `Number` objects, it will be stored as it is. When a list of integers/floats is passed, every element will be converted to a `Number` object and then convert to an `Array`.

```python
>>> from autodiff.structures import Number
>>> from autodiff.structures import Array
>>> import numpy as np
>>> x = Number(np.pi / 2)
>>> y = Number(3 * np.pi / 2)

>>> def f(x, y):
>>>     return Array((
>>>         y * x.sin(),
>>>         x * y.sin()
>>>     ))
>>> q = f(x,y)
>>> q
Array([Number(val=4.71238898038469) Number(val=-1.5707963267948966)])
```
`Array` objects doesn't hold `_deriv` attributes  internally as in the case with `Number`. To see each value's derivative with respect to other values, user can call `.jacobian(order)` method on Array, where `order` is a list of elements specified by the user, or a single element. When order is a list, it returns a `np.ndarray` array object where each row is every element in array in original order, each column is the order specified as inputs. When order is a single element, it returns a scaler.

Note that `autodiff.Number.jacobian()` does require the user to specify an order of input `Number` objects to ensure consistency within the user's own code. Otherwise, `autodiff` would have to infer which element belongs to which function input. As the user strings together multiple elementary operations, it is likely that `autodiff`'s understanding would differ from the user's. An example of the suggested usage is:

```python
>>> q.jacobian(x)
array([ 2.88550604e-16, -1.00000000e+00])
>>> q.jacobian([x,y])
array([[ 2.88550604e-16,  1.00000000e+00],
       [-1.00000000e+00, -2.88550604e-16]])
```
Alternatively, user can call `.jacobian(element)` on specific elements in the `Array`.
```python
>>>q[0].jacobian(y)
1.0
```

The `autodiff` package also works for scalar functions of vectors and vector functions of vectors, which behave the same.

`Array` also overloads basic operations: addition, multiplication, division, and subtraction, and power. They perform element-wise operations on Arrays, which mimics what  `np.array` does.

```python
>>> x = Array((1, 2))
>>> y = Array((3, 4))
>>> q = x*y
>>> q
Array([Number(val=3) Number(val=8)])
```

If we're multiplying an `Array` by a `Number`, it multiplies the same `Number` passed on every element of the array.

```python
>>> x = Array((1, 2))
>>> z = Number(2)
>>> q = x*z
>>> q
Array([Number(val=2) Number(val=4)])
```

`Array` also supports operations such as sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, logistic, log and exp. When these operations are called, it is performed on each element of the Array object. `exp` assumes that the base is `e`. When the base is not `e`, user could directly use the `pow` method (**).
```python
>>> x = Array((1, 2))
>>> q = x.sin()
>>> q
Array([Number(val=0.8414709848078965) Number(val=0.9092974268256817)])
```
Also, it supports dot product methods as it is used in numpy.
```python
>>> x = Array((1, 2))
>>> y = Array((3, 4))
>>> x.dot(y)
Number(val=11)
```

We also implemented three optimization methods in our optimization module, `steepest descent`, `AD BFGS` and `symbolic BFGS`. The former two use automatic differentiation to obtain gradient information and the last one requires user input of the expression of the gradient.

User can first import his/her desired optimization method. Then, call it on a provided function and an initial guess (which could be an Array or a Number, depending on the function). The results will return three elements: `xstar`, which is the point that achieved optimization (a Number or list of Numbers); `minimum`, which is the optimized quantity (a Number); and `jacobian at each step`, which gives a list of jacobians for user's reference.
Below is a simple example using bfgs:

```python
>>> from autodiff.structures import Number
>>> from autodiff.structures import Array
>>> from autodiff.optimizations import bfgs
>>> initial_guess = Array([Number(2),Number(1)])

>>> def rosenbrock(x0):
>>>     return (1-x0[0])**2+100*(x0[1]-x0[0]**2)**2

>>> results = bfgs(rosenbrock,initial_guess)
>>> results[0]
[Number(val=1.0000000000025382) Number(val=1.0000000000050797)]
>>> results[1]
Number(val=6.4435273497518935e-24)
>>> results[2]
[array([2402, -600]), array([-5.52902304e+12, -1.15187980e+09]), array([-474645.77945484,  127109.93018289]), array([ 1.62663315e+09, -2.70845802e+07]), array([-8619.39185109,  2161.73842208]), array([-144.79886656,   36.76686628]), array([1.99114433e+00, 3.55840767e-04]), array([1.99094760e+00, 4.05114252e-04]), array([1.96305014, 0.00739234]), array([1.9349555 , 0.01442883]), array([1.87896168, 0.02845251]), array([1.79486979, 0.04951253]), array([1.65477644, 0.08459553]), array([1.43057943, 0.14073564]), array([1.06628965, 0.23194665]), array([0.47793023, 0.37924961]), array([-0.47390953,  0.61758141]), array([-2.01036637,  1.00259974]), array([-4.48450951,  1.62438092]), array([-8.45367483,  2.63103416]), array([-14.76319038,   4.28099834]), array([-4.62373441,  1.75866303]), array([141.6035648 , -30.46772268]), array([ 7.1036742 , -1.70213995]), array([10.42215839, -2.72209527]), array([13.5437883 , -3.72650641]), array([16.21168154, -4.6779326 ]), array([16.2320604 , -4.88405766]), array([10.21216085, -3.12699453]), array([ 5.13098972, -1.52792632]), array([14.78025278, -5.5124494 ]), array([-1.52082729,  0.8129434 ]), array([ 3.23558391, -1.1679951 ]), array([ 7.90182702, -3.37442389]), array([ 2.23734678, -0.8351361 ]), array([ 0.98042779, -0.31955033]), array([ 2.87258631, -1.30884655]), array([ 0.71918344, -0.28586232]), array([ 0.36444279, -0.14819236]), array([ 0.46808941, -0.22547022]), array([-0.01787606,  0.01191596]), array([ 0.01227375, -0.00612718]), array([ 0.00044299, -0.00020912]), array([ 6.03745724e-08, -1.87348359e-08]), array([3.74411613e-12, 6.66133815e-13])]
```
Symbolic bfgs will take in three inputs: a function, derivative of the function (calculated by the user), and the initial guess. It is slower and implemented as a traditional method of optimization for user to compare to.

```python
>>> def gradientRosenbrock(x0):
>>>     x=x0[0]
>>>     y=x0[1]
>>>     drdx = -2*(1 - x) - 400*x*(-x**2 + y)
>>>     drdy = 200 *(-x**2 + y)
>>>     return drdx,drdy
>>> results = bfgs_symbolic(rosenbrock,gradientRosenbrock,[2,1])
>>> results[0]
[1. 1.]
>>> results[1]
6.4435273497518935e-24
```