class Optimizer(object):
    
    def __init__(self):
        
        # find all Tensors which will be updated
        # https://stackoverflow.com/questions/633127/viewing-all-defined-variables
        self.ls_tensors = [v for k, v in globals().copy().items() if type(v) is Tensor]
        
    def step(self):
        for T in self.ls_tensors:
            
            # if T is a terminal node, that means it is a learnable weight 
            if T.terminal:
                T.update_val_by_accumulated_gradient()
            else:
                pass