import numpy as np
import torch
from torch import nn
import os


m = nn.Dropout(p=0.5)
input = torch.ones(20, 5, 5)
output = m(input)
print(output)