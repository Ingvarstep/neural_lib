
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

