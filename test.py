import numpy as np
import torch
import os

a = torch.zeros(5,5)
b = torch.zeros(5,5)

a[1,1] = 1
a[3,3] = 0.1
b[1,1] = 1
b[3,3] = 1

loss = torch.nn.BCELoss()

c = loss(a,b)
print(c)