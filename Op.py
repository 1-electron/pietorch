import autograd.numpy as np 
from autograd import grad, elementwise_grad
from Tensor import Tensor

"""
Add is an operation. when Add is called, an Adder object is 
created and used to instantiate a Tensor object, which is 
then returned to caller.
"""
class Add(object):
    def __new__(self, x, y):  # https://stackoverflow.com/questions/53485171/how-to-return-objects-straight-after-instantiating-classes-in-python
        op = _Adder(x, y)
        name = "Add"
        return Tensor(val=op.evaluate(), parents=[x, y], 
                      op=op, terminal=False, name=name)
        
"""
the ADDER object will contain methods/attributes for the ADD
operation. certain ADDER attributes and methods will be used
to instantiate a TENSOR object. it is this TENSOR object that
is subsequently returned.

we used the adder object to organize node state such as 
gradient. it is not necessary but it makes the code more 
readable.
"""
class _Adder(object):
    def __init__(self, x, y):
        self.x = float(x.val)
        self.y = float(y.val)
        
    def evaluate(self):
        return self.x + self.y

    def f(self, x, y):
        return x + y

    def compute_parents_grads(self):
        # return a list of gradients
        self.df_dx = elementwise_grad(self.f, 0)
        self.df_dy = elementwise_grad(self.f, 1)
        return [self.df_dx(self.x, self.y), self.df_dy(self.x, self.y)]


class MULTIPLY(object):
    def __new__(self, x, y):
        """
        x and y are Tensor objects.
        """

        op = _MULTIPLIER(x, y)
        name = "Multiply"
        return Tensor(val=op.evaluate(), parents=[x, y], 
                      op=op, terminal=False, name=name)
        
class _MULTIPLIER(object):
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