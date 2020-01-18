import unittest
from Tensor import Tensor
from Op import Add, MULTIPLY

class Test_PieTorch(unittest.TestCase):

    def setUp(self):
        self.X = Tensor(val=-2, name="X")
        self.Y = Tensor(val=5, name="Y")
        self.Q = Add(self.X, self.Y)
        self.Z = Tensor(val=-4, name="Z")
        self.F = MULTIPLY(self.Q, self.Z)

    def test_forward_pass(self):
        self.assertEqual(self.F.val, -12.0)

if __name__ == "__main__":
    unittest.main()