import torch
from torch import optim
from torch import nn
from data import DataLoder
from model.EmbeddingNetwork import EmbeddingNetwork
from model.GraphNetwork import GraphNetwork

import argparse
from utils import *
import datetime
import os


class Model:
    def __init__(self, args, partition='train'):

        # fundamental setting
        self.root = args.root
        self.device = args.device
        self.partition = partition
        self.train_iters = args.train_iters
        self.test_iters = args.test_iters
        self.val_interval = args.val_interval

        # fewshot task setting
        self.num_layers = args.num_graph_layers
        self.num_tasks = args.num_tasks
        self.num_points = args.num_points
        self.num_emb_feats = args.num_emb_feats
        self.num_ways = args.num_ways
        self.num_supports = self.num_ways * args.num_shots
        self.num_all_queries = args.num_ways * args.num_queries
        self.num_samples = self.num_supports + self.num_all_queries
        self.support_edge_mask = torch.zeros(args.num_tasks, self.num_samples, self.num_samples).to(self.device)
        self.support_edge_mask[:, :self.num_supports, :self.num_supports] = 1
        self.query_edge_mask = 1 - self.support_edge_mask
        self.evaluation_mask = torch.ones(args.num_tasks, self.num_samples, self.num_samples).to(self.device)
        self.evaluation_mask[:, self.num_supports:, self.num_supports:] = 0

        # create log file
        if self.partition == 'train':
            if not os.path.exists(os.path.join(self.root, args.expr)):
                os.mkdir(args.expr)

            self.expr_folder = os.path.join(args.expr, str(datetime.datetime.now())[5:19].replace(':', '-'))
            if not os.path.exists(self.expr_folder):
                os.mkdir(self.expr_folder)

            self.logger = get_logger(self.expr_folder, 'train.log')

        # build dataloader
        if self.partition == 'train':
            self.train_dataloader = DataLoder.ModelNet40Loader(args, partition='train')

        self.test_dataloader = DataLoder.ModelNet40Loader(args, partition='test')

        # build model
        self.embeddingNet = EmbeddingNetwork(args)
        self.graphNet = GraphNetwork(args)

        # build optimizer
        module_params = list(self.embeddingNet.parameters()) + list(self.graphNet.parameters())
        self.optimizer = optim.Adam(params=module_params, lr=args.lr, weight_decay=args.weight_decay)

        # define losses
        self.edge_loss = nn.BCELoss(reduction='none')
        self.node_loss = nn.CrossEntropyLoss(reduction='none')

        # define metrics
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0

        # load pretrained model & optimizer
        self.last_iter = 0
        if args.ckpt is not None:
            model_ckpt = os.path.join(args.expr, args.ckpt, 'model.pt')
            if os.path.exists(model_ckpt):
                state = torch.load(model_ckpt)

                self.embeddingNet.load_state_dict(state['emb'])
                self.graphNet.load_state_dict(state['gnn'])
                self.optimizer.load_state_dict(state['optim'])
                self.last_iter = state['iter']

        # build scheduler
        lambda_lr = lambda iter: 0.5 ** (iter // args.dec_lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)
        self.lr_scheduler.last_epoch = self.last_iter

    def train(self):
        for iter in range(self.last_iter, self.train_iters):
            self.optimizer.zero_grad()
            sp_data, sp_label, _, qry_data, qry_label, _ = self.train_dataloader.get_task_batch()

            # set as single data
            full_data = torch.cat([sp_data, qry_data], 0)  # num_tasks*num_sp + num_tasks*num_qry, num_feat, num_points
            full_label = torch.cat([sp_label, qry_label], 1)
            full_edge = label2edge(full_label).to(self.device)

            # set init edge
            init_edge = full_edge.clone()  # num_tasks x 2 x num_samples x num_samples
            init_edge[:, :, self.num_supports:, :] = 0.5
            init_edge[:, :, :, self.num_supports:] = 0.5
            for i in range(self.num_all_queries):
                init_edge[:, 0, self.num_supports + i, self.num_supports + i] = 1.0
                init_edge[:, 1, self.num_supports + i, self.num_supports + i] = 0.0

            # set as train mode
            self.embeddingNet.train()
            self.graphNet.train()

            # (1) encode data
            full_data = self.embeddingNet(full_data)
            sp_data = full_data[:self.num_tasks * self.num_supports, :].view(self.num_tasks, self.num_supports, self.num_emb_feats)
            qry_data = full_data[self.num_tasks * self.num_supports:, :].view(self.num_tasks, self.num_all_queries, self.num_emb_feats)
            sp_data = sp_data.unsqueeze(1).repeat(1, self.num_all_queries, 1, 1)
            sp_data = sp_data.view(self.num_tasks * self.num_all_queries, self.num_supports, self.num_emb_feats)
            qry_data = qry_data.view(self.num_tasks * self.num_all_queries, 1, self.num_emb_feats)
            input_node_feat = torch.cat([sp_data, qry_data],
                                        1)  # (num_tasks x num_total_queries) x (num_support + 1) x featdim
            input_edge_feat = 0.5 * torch.ones(self.num_tasks, 2, self.num_supports + 1, self.num_supports + 1).to(
                self.device)  # num_tasks x 2 x (num_support + 1) x (num_support + 1)

            input_edge_feat[:, :, :self.num_supports, :self.num_supports] = init_edge[:, :, :self.num_supports, :self.num_supports]  # num_tasks x 2 x (num_support + 1) x (num_support + 1)
            input_edge_feat = input_edge_feat.unsqueeze(1).repeat(1, self.num_all_queries, 1, 1, 1)
            input_edge_feat = input_edge_feat.view(self.num_tasks * self.num_all_queries, 2, self.num_supports + 1,
                                                   self.num_supports + 1)  # (num_tasks x num_queries) x 2 x (num_support + 1) x (num_support + 1)

            # logit: (num_tasks x num_queries) x 2 x (num_support + 1) x (num_support + 1)
            logit_layers = self.graphNet(node_feats=input_node_feat, edge_feats=input_edge_feat)

            logit_layers = [logit_layer.view(self.num_tasks, self.num_all_queries, 2, self.num_supports + 1, self.num_supports + 1) for
                            logit_layer in logit_layers]

            # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
            full_logit_layers = []
            for l in range(self.num_layers):
                full_logit_layers.append(torch.zeros(self.num_tasks, 2, self.num_samples, self.num_samples).to(self.device))

            for l in range(args.num_layers):
                full_logit_layers[l][:, :, :self.num_supports, :self.num_supports] = logit_layers[l][:, :, :, :self.num_supports,
                                                                           :self.num_supports].mean(1)
                full_logit_layers[l][:, :, :self.num_supports, self.num_supports:] = logit_layers[l][:, :, :, :self.num_supports,
                                                                           -1].transpose(1, 2).transpose(2, 3)
                full_logit_layers[l][:, :, self.num_supports:, :self.num_supports] = logit_layers[l][:, :, :, -1,
                                                                           :self.num_supports].transpose(1, 2)

            # (4) compute loss
            full_edge_loss_layers = [self.edge_loss((1 - full_logit_layer[:, 0]), (1 - full_edge[:, 0])) for
                                     full_logit_layer in full_logit_layers]

            # weighted edge loss for balancing pos/neg
            pos_query_edge_loss_layers = [
                torch.sum(full_edge_loss_layer * self.query_edge_mask * full_edge[:, 0] * self.evaluation_mask) / torch.sum(
                    self.query_edge_mask * full_edge[:, 0] * self.evaluation_mask) for full_edge_loss_layer in
                full_edge_loss_layers]
            neg_query_edge_loss_layers = [
                torch.sum(full_edge_loss_layer * self.query_edge_mask * (1 - full_edge[:, 0]) * self.evaluation_mask) / torch.sum(
                    self.query_edge_mask * (1 - full_edge[:, 0]) * self.evaluation_mask) for full_edge_loss_layer in
                full_edge_loss_layers]
            query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                      (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                      zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

            # compute accuracy
            full_edge_accr_layers = [hit(full_logit_layer, 1 - full_edge[:, 0].long()) for full_logit_layer in
                                     full_logit_layers]
            query_edge_accr_layers = [torch.sum(full_edge_accr_layer * self.query_edge_mask * self.evaluation_mask) / torch.sum(
                self.query_edge_mask * self.evaluation_mask) for full_edge_accr_layer in full_edge_accr_layers]

            # compute  accuracy (num_tasks x num_quries x num_ways)
            query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, self.num_supports:, :self.num_supports],
                                                one_hot_encode(self.num_ways, sp_label.long())) for
                                      full_logit_layer in
                                      full_logit_layers]  # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
            query_node_accr_layers = [
                torch.eq(torch.max(query_node_pred_layer, -1)[1], qry_label.long()).float().mean() for
                query_node_pred_layer in query_node_pred_layers]

            total_loss_layers = query_edge_loss_layers

            # update model
            total_loss = []
            for l in range(args.num_layers - 1):
                total_loss += [total_loss_layers[l].view(-1) * 0.5]
            total_loss += [total_loss_layers[-1].view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))

            total_loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            # logging
            self.logger.info(' {0} th iteration, edge_loss: {1}, edge_accr: {2}, node_accr: {3}'
                             .format(iter,
                                     query_edge_loss_layers[-1],
                                     query_edge_accr_layers[-1],
                                     query_node_accr_layers[-1]
                                     ))

            # evaluation
            if iter % self.val_interval == 0:
                self.val_acc = self.evaluate()

                self.logger.info(' {0} th iteration, val_accr: {1}'.format(iter, self.val_acc))

                torch.save({'iter': iter,
                            'emb': self.embeddingNet.state_dict(),
                            'gnn': self.graphNet.state_dict(),
                            'optim': self.optimizer.state_dict()}, os.path.join(self.expr_folder, 'model.pt'))

    def evaluate(self):
        for iter in range(self.test_iters):
            sp_data, sp_label, _, qry_data, qry_label, _ = self.test_dataloader.get_task_batch()

            full_data = torch.cat([sp_data, qry_data], 0)
            full_label = torch.cat([sp_label, qry_label], 1)
            full_edge = label2edge(full_label)

            init_edge = full_edge.clone()
            init_edge[:, :, self.num_supports:, :] = 0.5
            init_edge[:, :, :, self.num_supports:] = 0.5
            for i in range(self.num_all_queries):
                init_edge[:, 0, self.num_supports + i, self.num_supports + i] = 1.0
                init_edge[:, 1, self.num_supports + i, self.num_supports + i] = 0.0


            # set as train mode
            self.embeddingNet.eval()
            self.graphNet.eval()

            full_data = self.embeddingNet(full_data)
            sp_data = full_data[:self.num_tasks * self.num_supports, :].view(self.num_tasks, self.num_supports,
                                                                             self.num_emb_feats)
            qry_data = full_data[self.num_tasks * self.num_supports:, :].view(self.num_tasks, self.num_all_queries,
                                                                              self.num_emb_feats)
            sp_data = sp_data.unsqueeze(1).repeat(1, self.num_all_queries, 1, 1)
            sp_data = sp_data.view(self.num_tasks * self.num_all_queries, self.num_supports, self.num_emb_feats)
            qry_data = qry_data.view(self.num_tasks * self.num_all_queries, 1, self.num_emb_feats)
            input_node_feat = torch.cat([sp_data, qry_data],
                                        1)  # (num_tasks x num_total_queries) x (num_support + 1) x featdim
            input_edge_feat = 0.5 * torch.ones(self.num_tasks, 2, self.num_supports + 1, self.num_supports + 1).to(
                self.device)  # num_tasks x 2 x (num_support + 1) x (num_support + 1)

            input_edge_feat[:, :, :self.num_supports, :self.num_supports] = init_edge[:, :, :self.num_supports,
                                                                            :self.num_supports]  # num_tasks x 2 x (num_support + 1) x (num_support + 1)
            input_edge_feat = input_edge_feat.unsqueeze(1).repeat(1, self.num_all_queries, 1, 1, 1)
            input_edge_feat = input_edge_feat.view(self.num_tasks * self.num_all_queries, 2, self.num_supports + 1,
                                                   self.num_supports + 1)  # (num_tasks x num_queries) x 2 x (num_support + 1) x (num_support + 1)

            logit = self.graphNet(node_feats=input_node_feat, edge_feats=input_edge_feat)[-1]
            logit = logit.view(self.num_tasks, self.num_all_queries, 2, self.num_supports + 1, self.num_supports + 1)
            logit = logit[:, :, 0, -1, :-1].squeeze()

            val_pred = torch.bmm(logit, one_hot_encode(self.num_ways, sp_label.long()))
            val_acc = torch.eq(torch.max(val_pred, -1)[1], qry_label.long()).float().mean()

            return val_acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')

    # Fundamental setting
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_ways', type=int, default='5')
    parser.add_argument('--num_shots', type=int, default='1')
    parser.add_argument('--num_tasks', type=int, default='5')
    parser.add_argument('--num_queries', type=int, default='1')
    parser.add_argument('--seed', type=float, default='0')
    parser.add_argument('--train_iters', type=int, default='2000')
    parser.add_argument('--test_iters', type=int, default='10')
    parser.add_argument('--val_interval', type=int, default='50')
    parser.add_argument('--expr', type=str, default='experiment/')
    parser.add_argument('--ckpt', type=str, default=None)

    # hyper-parameter setting
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--weight_decay', type=float, default='0.99')
    parser.add_argument('--dec_lr', type=float, default='30')


    # data loading setting
    parser.add_argument('--dataset_name', type=str, default='ModelNet40')
    parser.add_argument('--test_size', type=float, default='0.2')
    parser.add_argument('--num_points', type=int, default='1024')

    # data transform setting
    parser.add_argument('--shift_range', type=float, default='1')
    parser.add_argument('--angle_range', type=float, default='6.28')
    parser.add_argument('--max_scale', type=float, default='2')
    parser.add_argument('--min_scale', type=float, default='0.5')
    parser.add_argument('--sigma', type=float, default='0.01')
    parser.add_argument('--clip', type=float, default='0.02')

    # Embedding setting
    parser.add_argument('--k', type=int, default='20')
    parser.add_argument('--num_emb_feats', type=int, default='1024')

    # GraphNetwork section
    parser.add_argument('--num_node_feats', type=int, default='1024')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--edge_p', type=float, default='0')
    parser.add_argument('--feat_p', type=float, default='0')

    args = parser.parse_args()

    model = Model(args, partition='train')
    try:
        model.train()
    finally:
        model.logger.shutdown()
