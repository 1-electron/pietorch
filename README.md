# pietorch
a python implementation of pytorch. in a nutshell, it is autograd + graphs.

## example
```python
from Tensor import Tensor
from Op import Pow

X = Tensor(val=10, name="X")
Z = Pow(X, 2)
Z.val  # 100
Z.backward()  # compute gradients
X.grad  # gradient is 2x = 20
```