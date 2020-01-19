class Module(object):
    def __init__(self):
        pass
        
    def forward(self, x):
        """to be implemented by child"""
        raise NotImplementedError
    
    def __call__(self, x):
        # call Module object as if it's a function
        return self.forward(x)
    
    def parameters(self):
        """
        return list of weights (ie terminal Tensors). to be used by Optimizer.
        """
        
        # first, fetch the complete list of tensors
        attrs = [getattr(self, T) for T in dir(self) if type(getattr(self, T)) is Tensor]
        
        # then, filter for terminal tensors
        ls_weights = list(filter(lambda x: x.terminal is True, attrs))
        return ls_weights