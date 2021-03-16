import numpy as np
import torch


a = torch.rand((2,2))

b = a.max(-1)[1]

print(a)
print(b)