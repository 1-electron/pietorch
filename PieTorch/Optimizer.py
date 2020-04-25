class Optimizer(object):
    """an optimizer exhaustively updates any leaf tensors by their respective
    accumulated gradients.

    parameters
    ----------
    observed_params : list of tensors
        an optimizer object will inspect each tensor. if the tensor is a 
        terminal tensor, then the optimizer will update the tensor's value by
        its corresponding gradient. if the tensor is not a terminal tensor - ie
        it has parents (eg Add tensor) - then it is ignored my the optimizer.
    """
    
    def __init__(self, observed_params, learning_rate=0.001):
        self.observed_params = observed_params
        self.learning_rate = learning_rate
        
    def step(self):
        for T in self.observed_params:
            
            # guard condition: Tensor should be a terminal node
            if T.terminal:
                T.update_val_by_accumulated_gradient(self.learning_rate)
            else:
                print("node is not terminal")

    def zero_grad(self):
        """
        zero out all accumulated gradients for leaf tensors. we do not need to
        zero out accumulated gradients for non-leaf tensors because those 
        tensors are never updated by the optimizer.
        """
        for T in self.observed_params:
            if T.terminal:
                T.accumulated_grad = 0
            else:
                print("node is not terminal")  # no need to zero out