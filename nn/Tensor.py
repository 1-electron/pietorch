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

a Tensor's gradient is wrt its child. if F = Q + Z, then tensor Q's gradient is
dF/dQ. ie to increase F, update Q by df/dQ.
"""

class Tensor(object):
    def __init__(self, val=0, parents=[], forward=None, 
                 name=None, op=None, terminal=True):
        
        self.val = float(val)
        self.parents = parents  # eg C = A + B, then A and B are C's parents
        self.children = []  # eg C = A + B, then C is A's and B's child 
        self.forward = forward  # forward function
        self.grad = None  # the value of its own gradient wrt its child
        self.accumulated_grad = None  # accumulated gradient wrt root
        self.name = name
        self.op = op  # op contains context
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
        
        first, it recursively updates Tensors with its respective gradient.
        
        then, it prints the trace from a leaf to the root Tensor. (the
        root Tensor is the caller, eg Loss.backward().)
        """
        
        # step 1: recursively compute gradients; gradients not accumulated
        self._compute_grad()
        
        # step 2: construct depth first search tree
        self.color = "Gray"
        self.stack.append(self)
        self._dfs()
        

    def _dfs(self):
        """
        warning: confusing terminology. in a typical graph, we search "downwards"
        towards the children. in backpropagation, we do the opposite. we start
        a node and then search "upwards" towards parents.
        """

        # stack is emptied once all leaves have been visited
        while len(self.stack) > 0:
            
            # inspect the most recent Tensor in the stack
            curr_T = self.stack[-1]  
            
            # for that Tensor, find its unvisited parents
            ls_unvisited_parents = [p for p in curr_T.parents if p.color is "White"]
            
            # scenario 1: if current Tensor doesnt have anymore unvisited parents, 
            # we can remove it
            
            # a Tensor would not have any unvisited parents if it was (a) a 
            # terminal tensor or (b) an op tensor whose parents have been fully
            # explored
            if len(ls_unvisited_parents) == 0:
                curr_T.color = "Black"
                self.stack.pop()
            
            # scenario 2: Tensor has unvisited parents, which means it is an 
            # op tensor
            if len(ls_unvisited_parents) > 0:

                # look at one of its parents
                next_T = ls_unvisited_parents[0]
                
                # scenario 2a: the parent is terminal
                if next_T.terminal:
                    
                    # color it black so we dont double count it later on
                    next_T.color = "Black"

                    # add it to the stack for printing
                    self.stack.append(next_T)
                    
                    # we've found a terminal so we can accumulate gradients
                    self._accumulate_grads(next_T)
                     
                    # remove tensor from stack
                    self.stack.pop()
                
                # scenario 2b: the parent is not a leaf...
                else:
                    # ... so we add to stack and keep moving
                    next_T.color = "Gray"
                    self.stack.append(next_T)
    
    def _accumulate_grads(self, tensor):
        """
        once dfs reaches a terminal Tensor, we can accumulate gradients via 
        chain rule.
        """

        # guard condition: make sure tensor is a terminal
        if tensor.terminal is False:
            raise Exception('not a terminal tensor')

        tensor.accumulated_grad = 1

        for t in reversed(self.stack):
            print(t.name, t.grad)
            tensor.accumulated_grad *= t.grad

        print("accumulated gradient for ", tensor.name, "is ", tensor.accumulated_grad)
        print("-"*20)
    

    def _compute_grad(self):
        """
        let C = A + B. calling _compute_grad() on Tensor C returns dC/dA and
        dC/dB. in the subsequent step, dC/dA is placed on C and dC/dB is 
        placed on B.
        """

        # guard condition: if we're at a node that doenst have children, then
        # set its grad to 1 otherwise chain rule will zero out
        if len(self.children) == 0:
            self.grad = 1
        
        # base case: leaf Tensor
        if self.terminal:
            pass
        else:
            # eg if C = A + B, then return dC/dA and dC/dB when at Tensor C
            ls_gradients = self.op.compute_parents_grads()
            
            # then, place dC/dA on A and dC/dB on B
            for i in range(len(self.parents)):
                self.parents[i].grad = ls_gradients[i]
                
            # recursively call _compute_grad() on its parent
            for T in self.parents:
                T._compute_grad()
                
    def update_val_by_accumulated_gradient(self):
        """
        called by optimizer.
        """
        self.val = self.val + self.accumulated_grad