import random

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
    def __init__(self, data, dtype, view = None, requires_grad = False):
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
        self._backward = lambda: None
    
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
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor([[a*other for a in X_row] for X_row in self.data], self.dtype)
            return out

        if len(self.data)!=len(other.data) or len(self.data[0])!=len(other.data[0]):
            raise ValueError("Amount of rows or columns is not the same")
        out = Tensor([[a*b for a,b in zip(X_row,Y_row)] for X_row,Y_row in zip(self.data, other.data)], self.dtype)
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Expected int or float, got {type(other)}")
        out = Tensor([[a**other for a in X_row] for X_row in self.data], self.dtype)
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
    
    def backward(self):
        if self.requires_grad:
            for row in self.data:
                for value in row:
                    value.backward()
        else:
            raise ValueError("This Tensor does not require grad")
    
    def zero_grad(self):
        if self.requires_grad:
            for row in self.data:
                for value in row:
                    value.grad = 0
        else:
            raise ValueError("This Tensor does not require grad")
    
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




