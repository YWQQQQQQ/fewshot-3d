import numpy as np
import torch
import os

b = torch.ones(5,4)
c = torch.sum(b, -1)

print(b/c)
#print(c)
#print(b[:,0,:].shape)