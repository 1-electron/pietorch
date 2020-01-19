import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Pow(object):
    def __new__(self, x, y):
        op = _Power(x, y)
        name = "Power"
        return Tensor(val=op.evaluate(), parents=[x], 
                      op=op, terminal=False, name=name)
        
class _Power(object):
    def __init__(self, x, y):
        self.x = float(x.val)
        self.y = float(y)  # unlike other operations, y is not a tensor
        
    def evaluate(self):
        return self.x ** self.y

    def f(self, x, y):
        return x ** y

    def compute_parents_grads(self):
        # return a list of gradients
        self.df_dx = elementwise_grad(self.f, 0)
        return [self.df_dx(self.x, self.y)]


