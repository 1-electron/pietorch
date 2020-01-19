import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Multiply(object):
    def __new__(self, x, y):
        """
        x and y are Tensor objects.
        """

        op = _Multiplier(x, y)
        name = "Multiply"
        return Tensor(val=op.evaluate(), parents=[x, y], 
                      op=op, terminal=False, name=name)
        
class _Multiplier(object):
    def __init__(self, x, y):
        self.x = float(x.val)
        self.y = float(y.val)
        
    def evaluate(self):
        return self.x * self.y

    def f(self, a, b):
        return a * b

    def compute_parents_grads(self):
        # partial derivatives https://github.com/HIPS/autograd/issues/437
        self.df_dx = elementwise_grad(self.f, 0)
        self.df_dy = elementwise_grad(self.f, 1)
        return [self.df_dx(self.x, self.y), self.df_dy(self.x, self.y)]