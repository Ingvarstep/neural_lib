from lib.structs import Tensor
# from lib.nn import Linear

class Optimizer:
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
        
class SGD(Optimizer):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for param in self.params:
            param.data -= self.lr*param.grad

class Adam(Optimizer):
    def __init__(self, params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = [0. for _ in range(len(self.params))]
        self.v = [0. for _ in range(len(self.params))]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*param.grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*param.grad**2
            
            m_hat = self.m[i] / (1-self.beta1**self.t)
            v_hat = self.v[i] / (1-self.beta2**self.t)
            
            param.data -= self.lr*m_hat / (v_hat**0.5 + self.eps)