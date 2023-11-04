# Important notations
# shape->(rows,columns) = shape(2,4)  => 2-rows and 4 column values
#

import torch
import numpy as np

# Tensor Initialization
data = ([3,5],[4,6])
x_data = torch.tensor(data)


data1 = [[7,9],[8,6],[6,9]]
tensor_data = torch.tensor(data1)

numpyData = np.array([1,2,3,4,5])
tensor1 = torch.from_numpy(numpyData)  # same with torch.tensor(numpyData)

x_ones = torch.ones_like(tensor_data)
print(x_ones)
x_rands = torch.rand_like(tensor_data, dtype= torch.float)
print(x_rands)

# Tensor Attributes
print(x_data.shape) # describing the shape
print(x_data.dtype) # describing the data type
print(x_data.device) # where it is stored