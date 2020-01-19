import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Loss(object):
    def __new__(self):

        op = _Loss()
        name = "Loss"
        return Tensor(op=op, terminal=False, name=name)
        
class _Loss(object):
    def __init__(self):
        pass
        
    def _compute_loss(self, pred, y):
        return pred.val - y.val  # remember, everything is a tensor