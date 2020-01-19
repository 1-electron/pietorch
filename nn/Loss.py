import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Loss(object):
    def __init__(self):
        self.t = Tensor(name="Loss")

    def __call__(self, output, y):
        """
        everything is a tensor: inputs and outputs.
        """

        # compute scalar loss
        loss_ = output.val - y.val

        # everything is a tensor so wrap it with a tensor
        self.t.val = loss_
        
        # connect loss tensor with the output tensor
        # guard conditions necessary to prevent duplicate appends
        if output not in self.t.parents:
            self.t.parents.append(output)

        if self.t not in output.children:
            output.children.append(self.t)
        
        return self.t