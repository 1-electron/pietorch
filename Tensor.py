"""
everything in PieTorch is a Tensor object.

a Tensor object may contain an op. it contains an op when the  Tensor is created 
from an operation, for example c = a + b. in this case, A and B are both Tensors, 
and the addition operation creates C, a Tensor. C contains the Adder op, which 
provides contextual information that is unique to C, eg C's parents.

a Tensor object knows its children. if A + B = C, then C is A's child. this 
bookkeeping is important because backprogation starts at a leaf node, and the 
actual gradient for an update (dLoss/dX) must be accumulated by traversing 
through its children.
"""

class Tensor(object):
    def __init__(self, val=0, parents=[], forward=None, 
                 name=None, op=None, terminal=True):
        
        self.val = float(val)
        self.parents = parents  # eg C = A + B, then A and B are C's parents
        self.children = []  # eg C = A + B, then C is A's and B's child 
        self.forward = forward  # forward function
        self.grad = 0  # the value of its own gradient wrt its child
        self.name = name
        self.op = op  # op contains context eg parents
        self.terminal = terminal

        self.color = "White"  # used for dfs
        self.stack = []
        
        # when Tensor is instantiated from an operation, it will have parents
        if len(self.parents) > 0:
            self._update_parent()
        
    def _update_parent(self):
        """
        if a Tensor is instantiated as result of an operation, update 
        its parents.
        """
        for parent in self.parents:
            parent.children.append(self)
        
    def backward(self):
        """
        backward method accomplishes two tasks.
        
        first, it recursively updates each Tensor with its respective 
        gradient.
        
        then, it prints the trace from a leaf to the root Tensor. (the
        root Tensor is the caller, eg Loss.backward().)
        """
        
        # step 1: recursively compute gradients for each node; gradients not accumulated
        self._compute_grad()
        
        # step 2: construct depth first search tree
        self.color = "Gray"
        self.stack.append(self)
        self._dfs()
        
    def _dfs(self):
        while len(self.stack) > 0:
            
            # inspect the most recent Tensor
            curr_T = self.stack[-1]  
            
            # get a list of unvisited parents
            ls_unvisited_parents = [p for p in curr_T.parents if p.color is "White"]
            
            # scenario 1: if current Tensor doesnt have anymore unvisited children, we can remove it
            # a Tensor would not have any unvisited children if (a) it was a leaf or (b) an operation that has explored all leaves
            if len(ls_unvisited_parents) == 0:
                curr_T.color = "Black"
                self.stack.pop()
            
            # scenario 2: Tensor has unvisited children
            # only Tensors that are operations will have unvisited children
            if len(ls_unvisited_parents) > 0:
                
                next_T = ls_unvisited_parents[0]
                
                # scenario 2a: one of its unvisited children is a leaf
                if next_T.terminal:
                    
                    # color Tensor black so we dont double count it
                    next_T.color = "Black"
                    self.stack.append(next_T)
                    
                    # we've found a leaf so lets backpropagate
                    for s in reversed(self.stack):
                        print(s.name)
                    print("-"*20)
                    
                    # remove leaf from stack
                    self.stack.pop()
                
                # scenario 2b: T has an unvisited child that is not a leaf, so we add to stack and keep moving
                else:
                    next_T.color = "Gray"
                    self.stack.append(next_T)
        
    def _compute_grad(self):
        
        # base case: leaf Tensor, will have a gradient relative to its child
        if self.terminal:
            pass
        else:
            # compute gradients for its parents
            # eg if C = A + B, then return dC/dA and dC/dB when at Tensor C
            ls_gradients = self.op.compute_parents_grads()
            
            # then,, update its parent with the corresponding gradient
            for i in range(len(self.parents)):
                self.parents[i].grad = ls_gradients[i]
                
            # recursively call _compute_grad() on its parent
            for T in self.parents:
                T._compute_grad()
                
    def _update(self):
        """called by optimizer."""
        self.val = self.val - self.grad