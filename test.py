import numpy as np
import torch
import random
torch.random.manual_seed(1)
x=torch.rand((2,3,5))



def knn(x, k):
    # x: num_tasks*(num_supports+num_query), num_features, num_points
    # k: num_neighborhood

    xx_n = torch.bmm(x.transpose(2, 1), x)  # (xx`+ yy`+ zz`)
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)  # (x^2 + y^2 + z^2)
    x_n_square = x_square.transpose(2,1)  # (x`^2 + y`^2 + z`^2)
    inverse_distance = -(x_square - 2*xx_n + x_n_square)  # -( (x-x`)^2 + (y-y`)^2 + (z-z`)^2 )

    dis, idx = inverse_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx

