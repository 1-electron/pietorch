# pietorch
a python implementation of pytorch. in a nutshell, it is autograd plus graphs.

## examples
```python
from Tensor import Tensor
from Op import Pow

X = Tensor(val=10, name="X")
Z = Pow(X, 2)
Z.val  # 100
Z.backward()  # compute gradients
X.grad  # gradient is 2x = 20
```
a more realistic example.
```python
from nn import Tensor, Add, Multiply, Pow, Relu, Module, Loss

class Net(Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.X = Tensor(val=-2, name="X")
        self.Y = Tensor(val=5, name="Y")
        self.Z = Tensor(val=-4, name="Z")
        
    def forward(self, x):
        self.Q = Add(self.X, self.Y)
        self.F = Multiply(self.Q, self.Z)
        return self.F

model = Net()
```