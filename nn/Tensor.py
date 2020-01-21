

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

a Tensor's gradient is wrt its child. if F = Q + Z, then Q.grad is
dF/dQ. ie to increase F, update Q by df/dQ.
"""

class Tensor(object):
    def __init__(self, terminal=True, val=0, parents=[], forward=None, 
                 name=None, op=None):
        
        self.val = float(val)
        self.parents = parents  # eg C = A + B, then A and B are C's parents
        self.children = []  # eg C = A + B, then C is A's and B's child 
        self.forward = forward
        self.grad = None  # # eg C = A + B, then A.grad is dC/dA
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
        
        first, it recursively computes gradient for each tensor in its ancestry.

        then, it prints the trace from a leaf to the root Tensor. (the
        root Tensor is the caller, eg Loss.backward().)
        """
        
        # step 1: populate every tensor with its gradient
        self._compute_grad(self)
        self.grad = 1  # dF/dF = 1
        
        # step 2: accumulate gradients using dfs
        self.color = "Gray"
        self.stack.append(self)
        self._dfs()

        # step 3: reset every tensor by marking it "white"
        self.stack.append(self)
        self._mark_tensors_white()


    def _compute_grad(self, T):
        """
        _compute_grad is a wrapper around each Tensor's op's compute_parents_grads()
        method.

        let T = A + B. calling T._compute_grad() returns dC/dA and dC/dB. 
        in the subsequent step, dC/dA is placed on A and dC/dB is placed on B.
        """

        # base case: leaf Tensor
        if T.terminal:
            pass
        
        else:
            # eg if C = A + B, then Tensor C._compute_grad returns [dC/dA, dC/dB]
            ls_gradients = T.op.compute_parents_grads()  # gradients of parents wrt curr tensor
            
            # then, place dC/dA on A and dC/dB on B
            print(">>>>", T.name, T.parents)
            for i in range(len(T.parents)):
                T.parents[i].grad = ls_gradients[i]
                
            # recursively call _compute_grad() on its parent
            for parent in T.parents:
                self._compute_grad(parent)

    def _mark_tensors_white(self):
        """
        after _dfs is called, tensors will be marked "gray" or "black". we want to
        reset them by marking them "white".
        """
        while len(self.stack) > 0:
            curr_T = self.stack[-1]
            ls_unvisited_parents = [p for p in curr_T.parents if p.color is not "White"]

            if len(ls_unvisited_parents) == 0:
                curr_T.color = "White"
                self.stack.pop()

            if len(ls_unvisited_parents) > 0:

                # look at one of its parents
                next_T = ls_unvisited_parents[0]
                
                # scenario 2a: the parent is terminal
                if next_T.terminal:
                    
                    # mark terminal "white", no need to add to the stack since we wont do anything else with this tensor
                    next_T.color = "White"
                
                # scenario 2b: the parent is not a leaf...
                else:
                    # ... so we add to stack and keep moving
                    next_T.color = "White"
                    self.stack.append(next_T)


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
    

                
    def update_val_by_accumulated_gradient(self):
        """
        called by optimizer.
        """
        self.val = self.val + self.accumulated_grad