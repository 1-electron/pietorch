import numpy as np
from autograd import grad, elementwise_grad
from .Tensor import Tensor


class Matmul(object):
    def __new__(self, x, y):

        if isinstance(x, Tensor) is False:
            x = Tensor(val=x, name="input")
        if isinstance(y, Tensor) is False:
            y = Tensor(val=y, name="input")

        op = _Matmultiplier(x, y)
        name = "Matmul"
        return Tensor(val=op.evaluate(), parents=[x, y], op=op, terminal=False, 
            name=name)
        
class _Matmultiplier(object):
    def __init__(self, x, y):
        self.x = x.val
        self.y = y.val
        
    def evaluate(self):
        return np.dot(self.x, self.y)

    def f(self, x, y):
        return np.dot(x, y)

    def compute_parents_grads(self):
        # return a list of gradients
        self.df_dx = elementwise_grad(self.f, 0)
        self.df_dy = elementwise_grad(self.f, 1)
        return [self.df_dx(self.x, self.y), self.df_dy(self.x, self.y)]