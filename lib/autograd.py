import math
from tensor import Tensor

class NodeFunction:
    def __init__(self, v1, v2, out):
        self.v1 = v1
        self.v2 = v2
        self.out = out
    
    @staticmethod
    def backward(self):
        pass

    def next_functions(self):
        if not isinstance(self.v1, Tensor):
            grad_fn1 = None
        if not isinstance(self.v2, Tensor):
            grad_fn2 = None
        return grad_fn1, grad_fn2

class AccumulateGrad(NodeFunction):
    def __init__(self, v):
        self.v = v
        self.out = v
    
    def backward(self):
        self.v.grad += self.out.grad
    
    def next_functions(self):
        return []

class AddBackward(NodeFunction):
    def __init__(self, v1, v2, out):
        super().__init__(v1, v2, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad
        self.v2.grad+=self.out.grad
    
class MulBackward(NodeFunction):
    def __init__(self, v1, v2, out):
        super().__init__(v1, v2, out)
    
    def backward(self):
        self.v1.grad+=self.v2.data*self.out.grad
        self.v2.grad+=self.v1.data*self.out.grad

class MatMulBackward(NodeFunction):
    def __init__(self, v1, v2, out):
        super().__init__(v1, v2, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad @ self.v2.data.T
        self.v2.grad+=self.v1.data.T @ self.out.grad

class SinBackward(NodeFunction):
    def __init__(self, v1, out):
        super().__init__(v1, None, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad * self.out.cos()

class CosBackward(NodeFunction):
    def __init__(self, v1, out):
        super().__init__(v1, None, out)
    
    def backward(self):
        self.v1.grad+=-self.out.grad * self.out.sin()

class PowBackward(NodeFunction):
    def __init__(self, v1, v2, out):
        super().__init__(v1, v2, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad * self.v2.data * self.v1.data**(self.v2.data-1)

class LogBackward(NodeFunction):
    def __init__(self, v1, out):
        super().__init__(v1, None, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad * self.out.data**-1

class ExpBackward(NodeFunction):
    def __init__(self, v1, out):
        super().__init__(v1, None, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad * self.out.data

class ReLUBackward(NodeFunction):
    def __init__(self, v1, out):
        super().__init__(v1, None, out)
    
    def backward(self):
        self.v1.grad+=self.out.grad * Tensor([[1 if x>0 else 0 for x in Xr] for Xr in self.v1.data], requires_grad=True)
