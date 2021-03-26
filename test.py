import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable

x=Variable(torch.tensor(0.1), requires_grad=True)
y=x*x+2
print(y)

y.backward()
print(x.grad, '\n', x.grad_fn)

