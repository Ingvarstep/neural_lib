import random
import math

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

class Value:
    def __init__(self, data, dtype, label = '', children = (), requires_grad = False):
        self.data = data
        if type(self.data) is not dtype:
            self.data = dtype(self.data)
        self.requires_grad = requires_grad
        self.grad = 0
        self.dtype = dtype
        self.label = label
        self.children = set(children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data = {self.data}, dtype = {self.dtype}, label = {self.label})"
    
    def backward(self):
        visited = set()
        topo = []
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    dfs(child)
                topo.append(node)
        dfs(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
        
    def relu(self):
        out = Value(max(0, self.data), self.dtype, children=(self,))

        def _backward():
            self.grad += out.grad * (self.data>0)
        out._backward = _backward

        return out

    def zero_grad(self):
        self.grad = 0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, self.dtype, None)
        
        if other.dtype!=self.dtype:
            raise TypeError(f"Expected {self.dtype}, got {other.dtype}")
        
        out = Value(self.data+other.data, self.dtype, children=(self, other))

        def _backward():
            self.grad+= out.grad
            other.grad+= out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, self.dtype, None)
        
        if other.dtype!=self.dtype:
            raise TypeError(f"Expected {self.dtype}, got {other.dtype}")
        
        out = Value(self.data+other.data, self.dtype, children=(self, other))

        def _backward():
            self.grad+= out.grad*other.data
            other.grad+= out.grad*self.data
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Expected int or float, got {type(other)}")
        
        out = Value(self.data**other, self.dtype, children=(self,))

        def _backward():
            self.grad += out.grad*other*self.data**(other-1)
        out._backward = _backward

        return out

    def __neg__(self):
        return -1*self
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1


class Tensor:
    def __init__(self, data, dtype, view = None, requires_grad = False, grad = True):
        self.data = data
        #check if amount of columns is the same
        for row in data:
            if len(row)!=len(data[0]):
                raise ValueError("Amount of columns is not the same")


        #replace all values with Value objects
        for i in range(len(data)):
            for j in range(len(data[0])):
                if not isinstance(data[i][j], Value):
                    data[i][j] = Value(data[i][j], dtype, requires_grad = requires_grad)

        self.dtype = dtype
        self.requires_grad = requires_grad
        self.view = view
        if grad:
            self.grad = self.zero_grad() # Tensor([[0. for _ in self.data[0]] for _ in self.data], dtype=self.dtype, grad = False)
        else:
            self.grad = None
        self._backward = lambda: None
        self.grad_fn = None
    
    def requires_grad_(self, requires_grad = True):
        self.requires_grad = requires_grad
        for row in self.data:
            for val in row:
                val.requires_grad = requires_grad
        return self

    def __repr__(self):
        return f"Tensor(data = {self.data}, dtype = {self.dtype})"

    def __getitem__(self, index):
        if index > len(self.data):
            raise IndexError("Index out of range")
        out = Tensor([self.data[index]], self.dtype)
        return out

    def __matmul__(self, mat):
        if len(self.data[0])!=len(mat.data):
            raise ValueError("Amount of columns in first matrix is not equal to amount of rows in second matrix")
        out = Tensor([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*mat.data)] for X_row in self.data], self.dtype)

        self.grad_fn = MatMulBackward(self, mat, out)

        return out
    
    def __add__(self, other):
        if len(self.data)!=len(other.data) or len(self.data[0])!=len(other.data[0]):
            if len(other.data)==1 and len(other.data[0])==len(self.data[0]):
                out =  Tensor([[self.data[i][j]+other.data[0][j] for j in range(len(self.data[0]))] for i in range(len(self.data))], self.dtype)
                return out 
            elif len(self.data)==1 and len(self.data[0])==len(other.data[0]):
                out =  Tensor([[other.data[i][j]+self.data[0][j] for j in range(len(other.data[0]))] for i in range(len(other.data))], self.dtype)
                return out
            raise ValueError("Amount of rows or columns is not the same")
        out = Tensor([[a+b for a,b in zip(X_row,Y_row)] for X_row,Y_row in zip(self.data, other.data)], self.dtype)

        self.grad_fn = AddBackward(self, other, out)

        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor([[a*other for a in X_row] for X_row in self.data], self.dtype)
            # return out
        elif len(self.data)!=len(other.data) or len(self.data[0])!=len(other.data[0]):
            raise ValueError("Amount of rows or columns is not the same")
        else:
            out = Tensor([[a*b for a,b in zip(X_row,Y_row)] for X_row,Y_row in zip(self.data, other.data)], self.dtype)
            
        self.grad_fn = MulBackward(self, other, out)

        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Expected int or float, got {type(other)}")
        out = Tensor([[a**other for a in X_row] for X_row in self.data], self.dtype)

        self.grad_fn = PowBackward(self, other, out)

        return out
    
    def __neg__(self):
        return -1*self
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def transpose(self):
        out = Tensor([[self.data[j][i] for j in range(len(self.data))] for i in range(len(self.data[0]))], self.dtype)
        if self.grad is not None:
            out.grad = self.grad.transpose()
        return out

    def micrograd_backward(self):
        if self.requires_grad:
            for row in self.data:
                for value in row:
                    value.backward()
        else:
            raise ValueError("This Tensor does not require grad")
    def backward(self):
        topo = []
        seen = set()
        def dfs(self.grad_fn):
            if self not in seen:
                seen.add(self)
                for i in self.grad_fn.next_functions:
                    dfs(i)
                topo.append(self)
        dfs(self.grad_fn)
        self.grad +=1
        for i in reversed(topo):
            i.backward()
        
    def micrograd_zero_grad(self):
        if self.requires_grad:
            for row in self.data:
                for value in row:
                    value.grad = 0
        else:
            raise ValueError("This Tensor does not require grad")

    def zero_grad(self):
        self.grad = Tensor([[0. for _ in self.data[0]] for _ in self.data], dtype=self.dtype, grad = False)

    @classmethod
    def rand(self, shape, dtype = float, requires_grad = False):
        data = [[random.random() for i in range(shape[1])] for j in range(shape[0])]
        return Tensor(data, dtype, requires_grad = requires_grad)

    def parameters(self):
        if self.requires_grad:
            return [value for row in self.data for value in row]
        else:
            raise ValueError("This Tensor does not require grad")
    
    def relu(self):
        out = Tensor([[a.relu() for a in X_row] for X_row in self.data], self.dtype)
        self.grad_fn = ReLUBackward(self, out)
        return out
    
    def mean(self):
        val = Value(0, self.dtype)
        count = 0
        for row in self.data:
            for value in row:
                val+=value
                count+=1
        out = val/count
        return out

    def cos(self):
        out = Tensor([[math.cos(a) for a in X_row] for X_row in self.data], self.dtype)
        return out
    
    def sin(self):
        out = Tensor([[math.sin(a) for a in X_row] for X_row in self.data], self.dtype)
        return out 

    def tanh(self):
        out = Tensor([[math.tanh(a) for a in X_row] for X_row in self.data], self.dtype)
        return out




