import autograd.numpy as np 
from autograd import grad, elementwise_grad
from .Tensor import Tensor

class Pow(object):
    """Return a new tensor object as part of an exponent operation.
    """
    def __new__(self, base, exponent):
        """

        __new__ is the only method that is called before __init__ is called. we
        use __new__ to return a properly specified Tensor object.
        """

        # type checking... cast to Tensor; everything is a Tensor
        if type(base) is not Tensor:
            base = Tensor(val=base)

        # return a properly instantiated Tensor object
        op = _Power(base, exponent)  # instead of a literal, val is an op
        name = "Power"
        return Tensor(val=op.evaluate(), parents=[base], op=op, terminal=False, 
            name=name)
        
        
class _Power(object):
    """
    returns a _Power object, which knows its base, exponents, and gradient.,
    """
    def __init__(self, x, y):
        self.x = float(x.val)
        self.y = float(y)  # y is not a tensor since it's an exponent
        
    def evaluate(self):
        return self.x ** self.y

    def f(self, x, y):
        return x ** y

    def compute_parents_grads(self):
        # return a list of computed gradients
        self.df_dx = elementwise_grad(self.f, 0)
        return [self.df_dx(self.x, self.y)]


