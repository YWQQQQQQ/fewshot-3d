import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable
from torch.nn import functional as F

xs = np.random.randint(low=0,high=10,size=(2,2))
ss = np.eye(2)==1
ss[1,0] = True
L = []
for x,s in zip(xs,ss):
    L.append(x[s])

print(xs)
print(ss)
print(L)

