from nn import Tensor, Add, Multiply, Pow, Relu, Module, Loss

class Net(Module):
    def __init__(self):
        super(Net, self).__init__() 
        self.X = Tensor(val=-2, name="X")
        self.Y = Tensor(val=5, name="Y")
        self.Z = Tensor(val=-4, name="Z")
        
    def forward(self, x):
        self.Q = Add(self.X, self.Y)
        self.F = Multiply(self.Q, self.Z)
        return self.F

def run():
    model = Net()
    output = model(3)
    output.parents
    output.val
    criterion = Loss()
    loss = criterion(output, output)
    loss.backward()

if __name__ == "__main__":
    run()