from lib.structs import Tensor
from lib.nn import Linear, MSE
from lib.optim import Adam
from tqdm import tqdm

data = Tensor.rand((10, 100), dtype = float)

targets = Tensor.rand((10, 1), dtype = float)

model = Linear(100, 1)

criterion = MSE()

optimizer = Adam(model.parameters(), lr = 0.01)

for epoch in tqdm(range(10)):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
