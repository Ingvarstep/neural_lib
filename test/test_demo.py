from lib.tensor import Tensor
from lib.nn import Linear, MSE
from lib.optim import SGD


def test_autograd():
    data = Tensor.rand((10, 100), dtype = float)

    targets = Tensor.rand((10, 1), dtype = float)

    model = Linear(100, 1)

    criterion = MSE()

    optimizer = SGD(model.parameters(), lr = 0.01)

    for _ in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(loss)

def test_micrograd():
    data = Tensor.rand((10, 100), dtype = float)

    targets = Tensor.rand((10, 1), dtype = float)

    model = Linear(100, 1)

    criterion = MSE()

    optimizer = SGD(model.micro_parameters(), lr = 0.01)

    for _ in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(loss)
