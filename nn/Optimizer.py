class Optimizer(object):
    
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
        zero out all accumulated gradients for leaves.
        """
        for T in self.observed_params:
            if T.terminal:
                T.accumulated_grad = 0
            else:
                print("node is not terminal")