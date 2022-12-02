from lib.tensor import Tensor
from lib.micrograd import Value

from abc import abstractmethod

class Module:

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    @abstractmethod    
    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor.rand((in_features, out_features), dtype = float, requires_grad = True)
        self.b = Tensor.rand((1, out_features), dtype = float, requires_grad = True)
        
    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [ v for row in self.W.data for v in row] + [v for row in self.b.data for v in row]


class ReLU(Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        return x.relu()

class MSE(Module):
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return ((x - y)**2).mean()
    

