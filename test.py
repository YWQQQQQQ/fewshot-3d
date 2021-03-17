import numpy as np
import torch
import os

a = torch.eye(5).bool()
b = torch.zeros(5,5)

b[a] = 5
print(b)