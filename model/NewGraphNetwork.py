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
    x_norm = torch.bmm(x_norm,x_norm.transpose(1,2))+1e-6
    x_sim = x_ij / x_norm
    x_sim = x_sim.unsqueeze(1)
    x_dsim = 1.0-x_sim
    predict = torch.cat((x_sim, x_dsim), 1)
    #diag_mask = (1-torch.eye(x_i.shape[1])).unsqueeze(0).unsqueeze(0).repeat(x_i.shape[0],2,1,1).cuda()
    #predict = predict*diag_mask
    #force_edge_feat = torch.cat((torch.eye(x_i.shape[1]).unsqueeze(0),
    #                             torch.zeros(x_i.shape[1], x_i.shape[1]).unsqueeze(0)), 0).unsqueeze(0).repeat(x_i.shape[0],1,1,1).cuda()
    #predict = predict + force_edge_feat

    return torch.clamp(predict,0,1)


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, num_node_feats, device, ratio=[1], p=0):
        super(EdgeUpdateNetwork, self).__init__()
        self.num_node_feats = num_node_feats
        self.device = device
        self.num_feats_list = [int(num_node_feats * r) for r in ratio]
        self.p = p
        self.num_layers = len(self.num_feats_list)

        # layers
        for l in range(self.num_layers):
            conv = nn.Conv3d(in_channels=3,
                             out_channels=1,
                             kernel_size=[1,1,1],
                             bias=True)
            #bn = nn.BatchNorm3d(num_features=1)
            tanh = nn.Tanh()
            #l_relu = nn.LeakyReLU()
            if self.p > 0:
                drop = nn.Dropout3d(p=self.p)

            self.add_module('conv{}'.format(l+1), conv)
            #self.add_module('bn{}'.format(l+1), bn)
            self.add_module('tanh{}'.format(l+1), tanh)
            #self.add_module('l_relu{}'.format(l+1), l_relu)
            if self.p > 0:
                self.add_module('drop{}'.format(l + 1), drop)
        '''
        else:
            conv = nn.Conv2d(in_channels=self.num_node_feats,
                                 out_channels=self.num_node_feats,
                                 kernel_size=1,
                                 bias=False)
            bn = nn.BatchNorm2d(num_features=self.num_node_feats)
            tanh = nn.Tanh()
            self.add_module('conv', conv)
            self.add_module('bn', bn)
            self.add_module('tanh',tanh)
        '''
        '''
        bn = nn.BatchNorm2d(num_features=self.num_feats_list[l])
        l_relu = nn.LeakyReLU()
        if self.p > 0:
            drop = nn.Dropout2d(p=self.p)

        
        '''
    def forward(self, node_feats, edge_feats):
        # node_feats: num_tasks*num_queries, num_supports+1, num_features
        # compute abs(x_i, x_j)
        num_tasks, num_samples, num_feats = node_feats.size()
        x_i = node_feats.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = (x_i - x_j).unsqueeze(1)
        x_i = x_i.repeat(1,1,num_samples,1).unsqueeze(1)
        #x_j = x_j.repeat(1,num_samples,1,1)
        edge_feats = edge_feats.unsqueeze(1)
        edge_feats = torch.cat((x_i, x_ij, edge_feats), 1)
        edge_feats = torch.transpose(edge_feats, 2, 4)
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        for l in range(self.num_layers):
            edge_feats = self._modules['conv{}'.format(l+1)](edge_feats)
            #edge_feats = self._modules['bn{}'.format(l+1)](edge_feats)
            edge_feats = self._modules['tanh{}'.format(l+1)](edge_feats)
            #edge_feats = self._modules['l_relu{}'.format(l+1)](edge_feats)
            if self.p > 0:
                edge_feats = self._modules['drop{}'.format(l+1)](edge_feats)
        '''
        else:
            edge_feats = edge_feats.squeeze()

            edge_feats = self._modules['conv'](edge_feats)
            edge_feats = self._modules['bn'](edge_feats)
            edge_feats = self._modules['tanh'](edge_feats)
        '''
        '''
        new_edge_feats = torch.zeros(num_tasks, 1, num_feats, num_samples, num_samples).to(self.device)
        for i in range(num_samples):
            for j in range(num_samples):
                    new_edge_feats[:,:,:,i,j] = self.conv(ipt_edge_feats[:,:,:,i,j])
        new_edge_feats = new_edge_feats.squeeze(1)
        '''
        edge_feats = edge_feats.squeeze(1)
        diag_mask = 1.0 - torch.eye(num_samples).unsqueeze(0).unsqueeze(0).repeat(num_tasks, num_feats, 1, 1).to(self.device)
        edge_feats = edge_feats * diag_mask
        # set diagonal as zero and normalize
        edge_feats = edge_feats.transpose(1,3)

        return edge_feats


class NodeUpdateNetwork(nn.Module):
    def __init__(self, num_node_feats, device, ratio=[1], p=0):
        super(NodeUpdateNetwork, self).__init__()
        self.num_node_feats = num_node_feats
        self.num_feats_list = [int(num_node_feats * r) for r in ratio]
        self.device = device
        self.p = p
        self.num_layers = len(self.num_feats_list)
        # layers
        conv = nn.Conv2d(in_channels=2,
                         out_channels=1,
                         kernel_size=1,
                         bias=False)
        #bn = nn.BatchNorm2d(num_features=1)
        l_relu = nn.LeakyReLU()
        self.add_module('conv', conv)
        #self.add_module('bn{}'.format(l+1), bn)
        self.add_module('l_relu', l_relu)
        for l in range(self.num_layers):
            conv = nn.Conv1d(in_channels=self.num_feats_list[l-1] if l > 0 else self.num_node_feats,
                             out_channels=self.num_feats_list[l],
                             kernel_size=1,
                             bias=False)
            bn = nn.BatchNorm1d(num_features=self.num_feats_list[l])
            l_relu = nn.LeakyReLU()
            self.add_module('conv{}'.format(l+1), conv)
            self.add_module('bn{}'.format(l+1), bn)
            self.add_module('l_relu{}'.format(l+1), l_relu)
        
        '''
        for l in range(self.num_layers):
            conv = nn.Conv1d(in_channels=self.num_feats_list[l - 1] if l > 0 else 2*self.num_node_feats,
                             out_channels=self.num_feats_list[l],
                             kernel_size=1,
                             bias=False)
            bn = nn.BatchNorm1d(num_features=self.num_feats_list[l])
            l_relu = nn.LeakyReLU()

            self.add_module('conv{}'.format(l+1), conv)
            self.add_module('bn{}'.format(l+1), bn)
            self.add_module('l_relu{}'.format(l+1), l_relu)
            if self.p > 0:
                drop = nn.p(p=self.p)
                self.add_module('drop{}'.format(l + 1), drop)
        '''
    def forward(self, node_feats, edge_feats):
        # node_feats: num_batch(num_tasks*num_qry), num_samples(num_sp+1), node_feats(num_emb_feat)
        num_batches, num_samples, num_feats = node_feats.size()

        # Mask the node to itself connection (self-loop)
        # diag_mask: num_batch, 2, num_samples, num_samples
        #aggr_feats = torch.sum(edge_feats,2)
        aggr_feats = torch.sum(edge_feats,2).unsqueeze(1)
        node_feats = node_feats.unsqueeze(1)


        #node_feats = torch.cat([node_feats, torch.cat(aggr_feats.split(num_samples, 1), -1)], -1).transpose(1, 2)
        #node_feats = torch.cat([node_feats, aggr_feats], -1).transpose(1, 2)
        node_feats = torch.cat([node_feats, aggr_feats], 1)
        
        # non-linear transform
        node_feats = self._modules['conv'](node_feats)
        node_feats = self._modules['l_relu'](node_feats)
        node_feats = node_feats.squeeze().transpose(1,2)
        #print(node_feats.shape)
        for l in range(self.num_layers):
            node_feats = self._modules['conv{}'.format(l+1)](node_feats)
            node_feats = self._modules['bn{}'.format(l+1)](node_feats)
            node_feats = self._modules['l_relu{}'.format(l+1)](node_feats)
           
        node_feats = node_feats.transpose(1, 2)
        #node_feats = node_feats.squeeze()
        return node_feats


class GraphNetwork(nn.Module):
    def __init__(self, args):
        super(GraphNetwork, self).__init__()
        self.num_emb_feats = args.num_emb_feats
        self.num_node_feats = args.num_node_feats
        self.num_layers = args.num_graph_layers
        self.p = args.p
        self.device = args.device

        for l in range(self.num_layers):
            node2edge_net = EdgeUpdateNetwork(num_node_feats=self.num_node_feats,
                                              device=self.device,
                                              p=self.p)

            edge2node_net = NodeUpdateNetwork(num_node_feats=self.num_node_feats,
                                              device=self.device,
                                              p=self.p)

            self.add_module('node2edge_net{}'.format(l+1), node2edge_net)
            self.add_module('edge2node_net{}'.format(l + 1), edge2node_net)

    def forward(self, node_feats, edge_feats):
        # for each layer
        edge_feat_list = []
        #residual = node_feats
        edge_feat_list.append(sim_cal(node_feats))

        for l in range(self.num_layers):
            # (2) node to edge
            edge_feats = self._modules['node2edge_net{}'.format(l + 1)](node_feats, edge_feats)
            # (1) edge to node
            #residual = node_feats
            node_feats = self._modules['edge2node_net{}'.format(l+1)](node_feats, edge_feats)
            edge_feat_list.append(sim_cal(node_feats))
            #node_feats = (node_feats+residual)/2

        return edge_feat_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_emb_feats', type=int, default='128')

    parser.add_argument('--num_node_feats', type=int, default='128')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--p', type=float, default='0')
    parser.add_argument('--edge_p', type=float, default='0')

    args = parser.parse_args()

    nodes = torch.rand((4, 6, 128)).to(args.device)
    edges = torch.rand((4, 6, 6, 128)).to(args.device)
    gnn = GraphNetwork(args).to(args.device)
    result = gnn(nodes, edges)
    print(result[0].shape)
