class Module(object):
    def __init__(self):
        pass
        
    def forward(self, x):
        """to be implemented by child"""
        raise NotImplementedError
    
    def __call__(self, x):
        # call Module object as if it's a function
        return self.forward(x)