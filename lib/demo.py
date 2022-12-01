from lib.structs import Value, Tensor
from lib.nn import Linear, ReLU, MSE
from lib.optim import SGD, Adam
from tqdm import tqdm


def test_linear():
    data = Tensor.rand((10, 100), dtype = float)

    targets = Tensor.rand((10, 1), dtype = float)

    model = Linear(100, 1)

    criterion = MSE()

    optimizer = SGD(model.parameters(), lr = 0.01)

    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(loss)