# pietorch
an implementation of pytorch from scratch. in a nutshell, it is a computational graph plus autograd.

## examples
```python
from PieTorch import Tensor, Pow

X = Tensor(val=10, name="X")
Z = Pow(X, 2)
print(Z.val)  # 100
Z.backward()  # compute gradients
print(X.grad)  # gradient is 2x = 20
```
a more realistic example.
```python
from nn import Tensor, Add, Multiply, Pow, Relu, Module, Loss, Optimizer

class Net(Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.Y = Tensor(val=5, name="Y")
        self.Z = Tensor(val=-4, name="Z")
        
    def forward(self, x):
        q = Add(x, self.Y)
        y = Relu(q)
        f = Multiply(y, self.Z)
        return f

model = Net()
output = model(-2)
output.val  # prints -12.0

# get data
data = Tensor(val=-2, name="target")
target = Tensor(val=5, name="target")

# define loss
criterion = Loss()

# define optimizer
optimizer = Optimizer(model.parameters(), learning_rate=1)

for i in range(10):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # compute gradients
    optimizer.step()  # backpropagate
```