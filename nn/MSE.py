import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class MSE(object):
    def __init__(self):
        self.op = _MSE()
        self.t = Tensor(name="MSE", op=self.op, terminal=False)  # always set non weights to terminal=False
        self.instantiated = False

    def __call__(self, output, y):
        """
        everything is a tensor: inputs and outputs.

        output is a model's output (in the form of a tensor).

        y is the target.

        returns a tensor.
        """

        # everything is a tensor
        if isinstance(y, Tensor) is False:
            y = Tensor(val=y, name="input")

        # compute scalar loss
        loss_ = (output.val - y.val) ** 2

        # everything is a tensor so wrap it with a tensor
        self.t.val = loss_

        # put output tensor in op so we compute dL/dx wrt x, where x=output
        self.op.x = output.val
        self.op.y = y.val  # we dont really need this since we dont care about dL/dy, ie loss wrt label
        
        # connect loss tensor with the output tensor
        # guard conditions necessary to prevent duplicate appends
        
        # scenario 1: we've called loss once already, so no need to update parents
        if self.instantiated:
            pass
        
        # scenario 2: first time calling loss, so we need to update its parents and children
        else:
            if output not in self.t.parents:
                self.t.parents.append(output)

            if self.t not in output.children:
                output.children.append(self.t)
            
            self.instantiated = True  # by setting to True, loss object will not be able to update parents or children
        return self.t


class _MSE(object):
    def __init__(self):
        pass
        
    def evaluate(self):
        pass

    def f(self, output_val, y_val):
        return (output_val - y_val) ** 2

    def compute_parents_grads(self):
        """
        x = output
        y = label, we dont need to compute gradients wrt label
        """
        # called by the tensor created by Loss
        self.df_dx = elementwise_grad(self.f, 0)  # return dL/dx, or loss' gradient wrt its parent
        # self.df_dy = elementwise_grad(self.f, 1)  # no need to compute gradient wrt label
        return [self.df_dx(self.x, self.y)]