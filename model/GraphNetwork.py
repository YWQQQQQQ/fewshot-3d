import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class EdgeUpdateNetwork(nn.Module):
    def __init__(self, num_in_feats, device, ratio=[0.5], dropout=0):
        super(EdgeUpdateNetwork, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_feats_list = [int(num_in_feats * r) for r in ratio]
        self.device = device
        self.dropout = dropout
        self.num_layers = len(self.num_feats_list)
        # layers
        for l in range(self.num_layers):
            conv = nn.Conv2d(in_channels=self.num_feats_list[l - 1] if l > 0 else self.num_in_feats,
                             out_channels=self.num_feats_list[l],
                             kernel_size=1,
                             bias=False)
            bn = nn.BatchNorm2d(num_features=self.num_feats_list[l])
            l_relu = nn.LeakyReLU()

            self.add_module('conv{}'.format(l+1), conv)
            self.add_module('bn{}'.format(l+1), bn)
            self.add_module('l_relu{}'.format(l+1), l_relu)

            if self.dropout > 0:
                drop = nn.Dropout2d(p=self.dropout)
                self.add_module('drop{}'.format(l+1), drop)

        else:
            conv_out = nn.Conv2d(in_channels=self.num_feats_list[-1],
                                 out_channels=1,
                                 kernel_size=1,
                                 bias=False)
            self.add_module('conv_out', conv_out)

    def forward(self, node_feats, edge_feats):
        # node_feats: num_tasks*num_queries, num_supports+1, num_features
        # compute abs(x_i, x_j)
        num_tasks, num_samples, num_feats = node_feats.size()
        x_i = node_feats.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        for l in range(self.num_layers):
            x_ij = self._modules['conv{}'.format(l+1)](x_ij)
            x_ij = self._modules['bn{}'.format(l+1)](x_ij)
            x_ij = self._modules['l_relu{}'.format(l+1)](x_ij)
            if self.dropout > 0:
                x_ij = self._modules['drop{}'.format(l+1)](x_ij)
        else:
            x_ij = self._modules['conv_out'](x_ij)

        dsim_val = torch.sigmoid(x_ij)
        sim_val = 1.0 - dsim_val

        #diag_mask = 1.0 - torch.eye(num_samples).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(self.device)
        #edge_feats = edge_feats * diag_mask
        #merge_sum = torch.sum(edge_feats, -1, True)
        # set diagonal as zero and normalize
        edge_feats = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feats, p=1, dim=-1) #* merge_sum
        force_edge_feat = torch.eye(num_samples).unsqueeze(0).repeat(num_tasks, 1, 1).bool()
        edge_feats[:,0,:,:][force_edge_feat] = 1
        edge_feats[:,1,:,:][force_edge_feat] = 0

        edge_feats = edge_feats + 1e-6  # Prevent division by zero
        edge_feats = edge_feats / torch.sum(edge_feats, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feats


class NodeUpdateNetwork(nn.Module):
    def __init__(self, num_in_feats, num_node_feats, device, ratio=[2, 1], feat_p=0, edge_p=0, dropout=0):
        super(NodeUpdateNetwork, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_node_feats = num_node_feats
        self.num_feats_list = [int(num_node_feats * r) for r in ratio]
        self.device = device
        self.edge_drop = edge_p
        self.feat_drop = feat_p
        self.dropout = dropout
        self.num_layers = len(self.num_feats_list)

        self.move_step = Variable(torch.tensor(0.3), requires_grad=True).to(self.device)
        # layers
        if self.edge_drop > 0:
            drop = nn.Dropout(p=self.edge_drop)
            self.add_module('edge_drop0', drop)
        for l in range(self.num_layers):
            conv = nn.Conv1d(in_channels=self.num_feats_list[l - 1] if l > 0 else (self.num_in_feats),
                             out_channels=self.num_feats_list[l],
                             kernel_size=1,
                             bias=False)
            bn = nn.BatchNorm1d(num_features=self.num_feats_list[l])
            l_relu = nn.LeakyReLU()

            self.add_module('conv{}'.format(l+1), conv)
            self.add_module('bn{}'.format(l+1), bn)
            self.add_module('l_relu{}'.format(l+1), l_relu)

            if self.dropout > 0:
                drop = nn.Dropout1d(p=self.dropout)
                self.add_module('drop{}'.format(l+1), drop)

        if self.feat_drop > 0:
            drop = nn.Dropout(p=self.feat_drop)
            self.add_module('feat_drop0', drop)

    def forward(self, node_feats, edge_feats):
        # node_feats: num_batch(num_tasks*num_qry), num_samples(num_sp+1), node_feats(num_emb_feat)
        num_batches, num_samples, num_feats = node_feats.size()

        # Mask the node to itself connection (self-loop)
        # diag_mask: num_batch, 2, num_samples, num_samples
        diag_mask = 1.0 - torch.eye(num_samples).unsqueeze(0).repeat(num_batches, 1, 1).to(self.device)
        if self.edge_drop > 0:
            diag_mask = self._modules['edge_drop0'](diag_mask)
        diag_mask = diag_mask.unsqueeze(1).repeat(1, 2, 1, 1)
        # set diagonal as zero and normalize
        edge_feats = edge_feats * diag_mask

        # compute attention and aggregate
        aggr_feats = torch.bmm(torch.cat(torch.split(edge_feats, 1, 1), 2).squeeze(1), node_feats)
        sim_feats, dsim_feats = torch.split(aggr_feats, num_samples, 1)
        sim_feats = sim_feats.transpose(1,2)
        dsim_feats = dsim_feats.transpose(1,2)

        node_feats = node_feats.transpose(1,2)

        #node_feats = torch.cat([node_feats, torch.cat(aggr_feats.split(num_samples, 1), -1)], -1).transpose(1, 2)
        #node_feats = node_feats + self.move_step*(aggr_feats[:, :num_samples, :] - aggr_feats[:, num_samples:, :])
        #node_feats = node_feats.transpose(1,2)
        # non-linear transform
        for l in range(self.num_layers):
            node_feats = self._modules['conv{}'.format(l + 1)](node_feats)
            node_feats = self._modules['bn{}'.format(l + 1)](node_feats)
            node_feats = self._modules['l_relu{}'.format(l + 1)](node_feats)
            if self.dropout > 0:
                node_feats = self._modules['drop{}'.format(l+1)](node_feats)

            sim_feats = self._modules['conv{}'.format(l+1)](sim_feats)
            sim_feats = self._modules['bn{}'.format(l+1)](sim_feats)
            sim_feats = self._modules['l_relu{}'.format(l+1)](sim_feats)
            if self.dropout > 0:
                sim_feats = self._modules['drop{}'.format(l+1)](sim_feats)

            dsim_feats = self._modules['conv{}'.format(l + 1)](dsim_feats)
            dsim_feats = self._modules['bn{}'.format(l + 1)](dsim_feats)
            dsim_feats = self._modules['l_relu{}'.format(l + 1)](dsim_feats)
            if self.dropout > 0:
                dsim_feats = self._modules['drop{}'.format(l+1)](dsim_feats)

        if self.feat_drop > 0:
            sim_feats = self._modules['feat_drop0'](sim_feats)
        if self.feat_drop > 0:
            dsim_feats = self._modules['feat_drop0'](dsim_feats)

        node_feats = node_feats + self.move_step * (sim_feats - dsim_feats)
        node_feats = node_feats.transpose(1,2)
        return node_feats


class GraphNetwork(nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()
        self.num_emb_feats = args.num_emb_feats
        self.num_node_feats = args.num_node_feats
        self.num_layers = args.num_graph_layers
        self.edge_p = args.edge_p
        self.feat_p = args.feat_p
        self.device = args.device

        node2edge_net = EdgeUpdateNetwork(num_in_feats=self.num_emb_feats,
                                          device=self.device)

        self.add_module('node2edge_net{}'.format(0), node2edge_net)

        for l in range(self.num_layers):
            edge2node_net = NodeUpdateNetwork(#num_in_feats=self.num_emb_feats+self.num_node_feats if l>0 else self.num_emb_feats,
                                              num_in_feats=self.num_node_feats if l > 0 else self.num_emb_feats,
                                              num_node_feats=self.num_node_feats,
                                              device=self.device,
                                              #ratio=[2,1] if l>0 else [0.5,1],
                                              feat_p=self.feat_p,
                                              edge_p=self.edge_p,
                                              dropout=0)

            node2edge_net = EdgeUpdateNetwork(#num_in_feats=self.num_emb_feats+self.num_node_feats,
                                              num_in_feats=self.num_node_feats,
                                              device=self.device,
                                              dropout=0.3)

            self.add_module('edge2node_net{}'.format(l+1), edge2node_net)
            self.add_module('node2edge_net{}'.format(l+1), node2edge_net)

    def forward(self, node_feats, edge_feats):
        # for each layer
        edge_feat_list = []

        #ori_node_feats = node_feats
        edge_feats = self._modules['node2edge_net{}'.format(0)](node_feats, edge_feats)
        edge_feat_list.append(edge_feats)

        for l in range(self.num_layers):
            # (1) edge to node
            node_feats = self._modules['edge2node_net{}'.format(l+1)](node_feats, edge_feats)
            #node_feats = torch.cat([ori_node_feats, node_feats], -1)

            # (2) node to edge
            edge_feats = self._modules['node2edge_net{}'.format(l+1)](node_feats, edge_feats)


            edge_feat_list.append(edge_feats)

        return edge_feat_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_emb_feats', type=int, default='64')

    parser.add_argument('--num_node_feats', type=int, default='128')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--feat_p', type=float, default='0')
    parser.add_argument('--edge_p', type=float, default='0')

    args = parser.parse_args()

    nodes = torch.ones((4,6,64))
    edges = torch.ones((4,2,6,6))
    gnn = GraphNetwork(args)
    result = gnn(nodes, edges)
    print(result[0].shape)
