import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable

a=torch.zeros(3,2,3)
b,c = torch.split(a,1,1)
print(b.shape)
print(c.shape)