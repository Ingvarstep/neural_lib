# neural_lib

Basic neural networks library written in Python for educational purposes 

###Structure

The code is divided into the following modules and, with some approximation, copies PyTorch APIs:
	* structs.py - where are realized Value and Tensor objects, the last one store data in the form of a matrix, with x rows and y columns; the single value of this matrix is a Value object. 
	* nn.py - collects different architectures of neural networks (currently, only one Linear layer) with loss functions;
	* optim.py - contains different optimization algorithms as single objects;


### Example

You can check an example of library usage in the demo.py file.
  
```python
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

```

### Tests

To run tests execute following command:

```bash
python -m pytest
```
    
