import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def knn(x, k):
    # x: num_tasks*(num_supports+num_query), num_features, num_points
    # k: num_neighborhood

    xx_n = torch.bmm(x.transpose(2, 1), x)  # (xx`+ yy`+ zz`)
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)  # (x^2 + y^2 + z^2)
    x_n_square = x_square.transpose(1,2)  # (x`^2 + y`^2 + z`^2)
    inverse_distance = -(x_square - 2*xx_n + x_n_square)  # -( (x-x`)^2 + (y-y`)^2 + (z-z`)^2 )

    dis, idx = inverse_distance.topk(k=k+1, dim=-1)  # (batch_size, num_points, k+1)
    return idx[:,:,1:]


class LDGCNN(nn.Module):
    def __init__(self, args):
        super(LDGCNN, self).__init__()

        self.k = args.k
        self.device = args.device
        self.num_emb_feats = args.num_emb_feats
        self.num_outs = args.num_ways

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(self.num_emb_feats)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((3+64)*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d((3+64+64)*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d((3+64+64+64)*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d((3+64+64+64+128), self.num_emb_feats, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.mlp1 = nn.Sequential(nn.Linear(self.num_emb_feats, 256),
                                  self.bn6,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.5))

        self.mlp2 = nn.Sequential(nn.Linear(256, 64),
                                  self.bn7,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.5))
        self.mlp3 = nn.Sequential(nn.Linear(64, self.num_outs),
                                  nn.Dropout(0.5))

    def forward(self, x):

        x1 = self.get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = torch.cat([x, x1], dim=1)
        x2 = self.get_graph_feature(x2, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = torch.cat([x, x1, x2], dim=1)
        x3 = self.get_graph_feature(x3, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = torch.cat([x, x1, x2, x3], dim=1)
        x4 = self.get_graph_feature(x4, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x5 = torch.cat((x,x1, x2, x3, x4), dim=1)

        x5 = self.conv5(x5)
        #x5_1 = F.adaptive_max_pool1d(x5, 1).view(num_samples, -1)
        #x5_2 = F.adaptive_avg_pool1d(x5, 1).view(num_samples, -1)
        #x5 = torch.cat((x5_1, x5_2), 1)
        x5 = x5.max(dim=-1, keepdim=False)[0]

        x6 = self.mlp1(x5)
        x7 = self.mlp2(x6)
        outs = self.mlp3(x7)
        outs = F.normalize(outs, p=1, dim=-1)
        return outs

    def get_graph_feature(self, x, k=20, idx=None):
        num_samples, num_features, num_points = x.size()  # num_tasks*(num_supports+num_query), num_features, num_points

        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)

        #device = torch.device(self.device)
        # device = torch.device('cpu')

        idx_base = torch.arange(0, num_samples, device=self.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(num_samples * num_points, num_features)[idx, :]
        feature = feature.view(num_samples, num_points, k, num_features)
        x = x.view(num_samples, num_points, 1, num_features).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous() # num_samples, num_features, num_poinbts, num_neighbors

        return feature


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(128, args.num_emb_feats, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.num_emb_feats),
                                   nn.ReLU())

        self.mlp1 = nn.Sequential(nn.Linear(args.num_emb_feats, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(0.5))

        self.mlp2 = nn.Sequential(nn.Linear(256, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Dropout(0.5))

        self.mlp3 = nn.Sequential(nn.Linear(64, args.num_ways),
                                  nn.Dropout(0.5))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default='20')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_emb_feats', type=int, default='1024')
    parser.add_argument('--num_ways', type=int, default='5')

    args = parser.parse_args()

    x = torch.zeros((20,3,512)).to(args.device)

    model = PointNet(args).to(args.device)

    out = model(x)
    print(out.shape)
