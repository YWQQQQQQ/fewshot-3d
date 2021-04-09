import numpy as np
import torch
from torch import nn
import os
from torch.autograd import Variable
from torch.nn import functional as F

def sim_cal(node_feats):
    x_i = node_feats
    x_j = node_feats.transpose(1,2)
    x_ij = torch.bmm(x_i, x_j)
    x_norm = torch.norm(node_feats, p=2, dim=-1).unsqueeze(-1)
    x_norm = torch.bmm(x_norm,x_norm.transpose(1,2))
    x_sim = torch.div(x_ij, x_norm)
    return x_sim

nn.Tanh()
