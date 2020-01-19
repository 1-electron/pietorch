import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Relu(object):
    def __new__(self, x):

        if type(x) is not Tensor:
            x = Tensor(val=x)

        op = _Relu(x)
        name = "Relu"
        return Tensor(val=op.evaluate(), parents=[x], op=op, terminal=False, 
            name=name)
        
class _Relu(object):
    def __init__(self, x):
        self.x = float(x.val)
        
    def evaluate(self):
        return max(0, self.x)

    def compute_parents_grads(self):
        if self.x > 0:
            self.df_dx = 1
        else:
            self.df_dx = 0
        return [self.df_dx]