import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def sim_cal(node_feats):
    x_i = node_feats
    x_j = torch.transpose(node_feats, 1, 2)
    x_ij = torch.bmm(x_i, x_j)
    x_norm = torch.norm(node_feats, p=2, dim=-1).unsqueeze(-1)
    x_norm = torch.bmm(x_norm,x_norm.transpose(1,2))
    x_sim = torch.div(x_ij, x_norm)
    return x_sim


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, num_node_feats, device, dropout=0):
        super(EdgeUpdateNetwork, self).__init__()
        self.num_node_feats = num_node_feats
        self.device = device
        self.dropout = dropout
        # layers
        self.conv1 = nn.Conv2d(in_channels=2*num_node_feats,
                         out_channels=num_node_feats,
                         kernel_size=1,
                         bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(num_features=num_node_feats).to(device)
        self.relu1 = nn.ReLU().to(device)

        self.conv2 = nn.Conv2d(in_channels=2*num_node_feats,
                         out_channels=num_node_feats,
                         kernel_size=1,
                         bias=False).to(device)
        self.bn2 = nn.BatchNorm2d(num_features=num_node_feats).to(device)
        self.relu2 = nn.ReLU().to(device)


    def forward(self, node_feats):
        # node_feats: num_tasks*num_queries, num_supports+1, num_features
        # compute abs(x_i, x_j)
        num_tasks, num_samples, num_feats = node_feats.size()


        x_i = node_feats.unsqueeze(2).repeat(1,1,num_samples,1)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.cat((x_i, x_j), -1)
        x_ij = torch.transpose(x_ij, 1,3)

        sim = torch.transpose(self.relu1(self.bn1(self.conv1(x_ij))),1,3).unsqueeze(1)
        dsim = torch.transpose(self.relu2(self.bn2(self.conv2(x_ij))),1,3).unsqueeze(1)

        diag_mask = 1.0 - torch.eye(num_samples).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(num_tasks, 2, 1, 1, num_feats).to(self.device)
        edge_feats = F.normalize(torch.cat([sim, dsim], 1)*diag_mask, p=1, dim=-1)
        return edge_feats


class NodeUpdateNetwork(nn.Module):
    def __init__(self, num_node_feats, device, dropout=0):
        super(NodeUpdateNetwork, self).__init__()
        self.num_node_feats = num_node_feats
        self.device = device
        self.dropout = dropout

        # layers
        self.conv = nn.Conv1d(in_channels=num_node_feats,
                         out_channels=num_node_feats,
                         kernel_size=1,
                         bias=False).to(device)
        self.bn = nn.BatchNorm1d(num_features=num_node_feats).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, node_feats, edge_feats):
        # node_feats: num_batch(num_tasks*num_qry), num_samples(num_sp+1), node_feats(num_emb_feat)
        num_batches, num_samples, num_feats = node_feats.size()

        # Mask the node to itself connection (self-loop)
        # diag_mask: num_batch, 2, num_samples, num_samples

        sim = edge_feats[:,0]
        dsim = edge_feats[:,1]

        node_feats_r = node_feats.unsqueeze(1).repeat(1,num_samples,1,1)
        sim_feats = torch.mean(sim*node_feats_r, 2)
        dsim_feats = torch.mean(dsim*node_feats_r, 2)

        node_feats = torch.transpose(node_feats + sim_feats - dsim_feats,1,2)

        node_feats = self.relu(self.bn(self.conv(node_feats)))
        node_feats = node_feats.transpose(1, 2)
        return node_feats


class GraphNetwork(nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()
        self.num_node_feats = args.num_node_feats
        self.num_layers = args.num_graph_layers
        self.dropout = args.dropout
        self.device = args.device

        for l in range(self.num_layers):
            node2edge_net = EdgeUpdateNetwork(  # num_in_feats=self.num_emb_feats+self.num_node_feats,
                num_node_feats=self.num_node_feats,
                device=self.device,
                dropout=self.dropout)

            edge2node_net = NodeUpdateNetwork(
                num_node_feats=self.num_node_feats,
                device=self.device,
                dropout=self.dropout)
            self.add_module('node2edge_net{}'.format(l + 1), node2edge_net)
            self.add_module('edge2node_net{}'.format(l + 1), edge2node_net)

    def forward(self, node_feats, edge_feats):
        # for each layer
        edge_feat_list = []

        # ori_node_feats = node_feats
        edge_feat_list.append(sim_cal(node_feats))

        for l in range(self.num_layers):
            # (2) node to edge
            edge_feats = self._modules['node2edge_net{}'.format(l + 1)](node_feats)

            # (1) edge to node
            node_feats = self._modules['edge2node_net{}'.format(l + 1)](node_feats, edge_feats)

            edge_feat_list.append(sim_cal(node_feats))

        return edge_feat_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--num_node_feats', type=int, default='128')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--dropout', type=float, default='0')

    args = parser.parse_args()

    nodes = torch.ones((4, 6, 128)).to(args.device)
    edges = torch.ones((4, 2, 6, 6)).to(args.device)
    gnn = GraphNetwork(args)
    result = gnn(nodes, edges)
    print(result[0].shape)
