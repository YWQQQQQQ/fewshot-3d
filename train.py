import torch
from torch import optim
from torch import nn
from data import DataLoader
from model import EmbeddingNetwork
from model import NewGraphNetwork, EGNN
from loss.CircleLoss import CircleLoss, get_sp_sn
import argparse
from utils import *
import datetime
import os
import logging
from torch.nn import functional as F

class Model:
    def __init__(self, args, partition='train'):
        self.i = 0
        self.j = 0
        # fundamental setting
        self.root = args.root
        self.device = args.device
        self.partition = partition
        self.train_iters = args.train_iters
        self.test_iters = args.test_iters
        self.train_interval = args.train_interval
        self.val_interval = args.val_interval
        self.emb_net = args.emb_net
        self.gnn_net = args.gnn_net

        # fewshot task setting
        self.num_layers = args.num_graph_layers if self.gnn_net == 'egnn' else args.num_graph_layers+1
        self.num_tasks = args.num_tasks
        self.num_points = args.num_points
        self.num_emb_feats = args.num_emb_feats
        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.num_supports = self.num_ways * self.num_shots
        self.num_queries = args.num_ways * 1
        self.num_samples = self.num_supports + 1
        self.sp_edge_mask = torch.zeros(self.num_tasks*self.num_queries, self.num_samples, self.num_samples).to(self.device)
        self.sp_edge_mask[:, :self.num_supports, :self.num_supports] = 1
        self.qry_edge_mask = 1 - self.sp_edge_mask
        self.evaluation_mask = 1 - torch.eye(self.num_samples).unsqueeze(0).repeat(self.num_tasks*self.num_queries, 1, 1).to(self.device)

        # create log file
        if self.partition == 'train':
            if not os.path.exists(os.path.join(self.root, args.expr)):
                os.mkdir(args.expr)

            self.expr_folder = os.path.join(args.expr, str(datetime.datetime.now())[5:19].replace(':', '-').replace(' ', '-'))
            if not os.path.exists(self.expr_folder):
                os.mkdir(self.expr_folder)

            self.logger = get_logger(self.expr_folder, 'train.log')

        # build dataloader
        if self.partition == 'train':
            self.train_dataloader = DataLoader.DatasetLoader(args, partition='train')

        self.test_dataloader = DataLoader.DatasetLoader(args, partition='test')

        # build model
        if self.emb_net == 'ldgcnn':
            self.embeddingNet = EmbeddingNetwork.LDGCNN(args).to(self.device)
        elif self.emb_net == 'pointnet':
            self.embeddingNet = EmbeddingNetwork.PointNet(args).to(self.device)
        if self.gnn_net == 'egnn':
            self.graphNet = EGNN.GraphNetwork(args).to(self.device)
        elif self.gnn_net == 'ours':
            self.graphNet = NewGraphNetwork.GraphNetwork(args).to(self.device)


        # build optimizer
        module_params = list(self.embeddingNet.parameters()) + list(self.graphNet.parameters())
        self.optimizer = optim.Adam(params=module_params, lr=args.lr, weight_decay=args.weight_decay)

        # define losses
        #self.edge_loss = nn.BCELoss(reduction='none')
        #self.node_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = args.loss
        if self.loss == 'circle':
            self.edge_loss = CircleLoss(m=0.25, gamma=32)
        elif self.loss == 'bce':
            self.edge_loss = nn.BCELoss(reduction='none')
        # define metrics
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0
        self.best_accs = [0 for _ in range(self.num_layers+1)]
        self.smooth_qry_loss = []
        self.smooth_sp_loss = []
        self.smooth_edge_acc = []
        self.smooth_node_acc = []
        self.smooth_avg_node_acc = []
        # load pretrained model & optimizer
        self.last_iter = 0
        if args.ckpt is not None:
            model_ckpt = os.path.join(args.expr, args.ckpt, 'model.pt')
            if os.path.exists(model_ckpt):
                print('exist!')
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
            # sp_data: num_tasks*num_sp, num_feat, num_pts
            # sp_label: num_tasks, num_sp
            # qry_data: num_tasks*num_qry, num_feat, num_pts
            # qry_label: num_tasks, num_qry
            sp_data, sp_label, _, qry_data, qry_label, _ = self.train_dataloader.get_task_batch()

            # concat to get emb_feat easier
            # full_data: num_tasks*num_sp + num_tasks*qry, num_feat, num_pts
            # full_label: num_tasks, num_samples
            # full_edge: num_tasks, 2, num_samples, num_samples
            full_data = torch.cat([sp_data, qry_data], 0)
            full_edge = label2edge(sp_label, qry_label).to(self.device)

            # set as train mode
            self.embeddingNet.train()
            self.graphNet.train()

            # full data: num_tasks*num_sp + num_tasks*qry, num_emb_feat
            # sp_data: num_tasks, num_sp, num_emb_feat
            # qry_data: num_tasks*num_qry, 1, num_emb_feat
            full_data = self.embeddingNet(full_data)
            sp_data = full_data[:self.num_tasks * self.num_supports, :].view(self.num_tasks, self.num_supports, self.num_emb_feats)
            qry_data = full_data[self.num_tasks * self.num_supports:, :].view(self.num_tasks*self.num_queries, 1, self.num_emb_feats)

            # sp_data: num_tasks*num_qry, num_sp, num_emb_feat
            sp_data = sp_data.unsqueeze(1).repeat(1, self.num_queries, 1, 1)
            sp_data = sp_data.view(self.num_tasks * self.num_queries, self.num_supports, self.num_emb_feats)

            # concat the sp and qry to a graph
            # input_node_feat: num_tasks*num_qry, num_sp+1, num_emb_feat
            # input_edge_feat: num_tasks*num_qry, 2, num_sp+1, num_sp+1
            input_node_feat = torch.cat([sp_data, qry_data], 1)

            # set the qry to others as 0.5 while keep qry to itself as 1
            #input_edge_feat = 0.5 * torch.ones(self.num_tasks*self.num_queries, self.num_samples, self.num_samples, self.num_emb_feats).to(self.device)

            input_edge_feat = full_edge.clone()

            # qry to others
            input_edge_feat[:, :, -1, :-1] = 0.5
            input_edge_feat[:, :, :-1, -1] = 0.5

            # qry to itself
            input_edge_feat[:, 0, -1, -1] = 1
            input_edge_feat[:, 1, -1, -1] = 0
            #input_edge_feat = input_edge_feat.unsqueeze(-1).repeat(1,1,1,1,self.num_emb_feats)
            if self.gnn_net == 'ours':
                input_edge_feat = input_edge_feat[:,0].unsqueeze(-1).repeat(1, 1, 1, self.num_emb_feats)

            # logit_layers: num_layers, num_tasks*num_qry, 2, num_sp+1, num_sp+1
            logit_layers = self.graphNet(node_feats=input_node_feat, edge_feats=input_edge_feat)
            #logit_layers = [torch.mean(logit_layer, -1) for logit_layer in logit_layers]
            # self.i -= 1
            # if self.i == -1:
            #     print(logit_layers[0][self.j,1,:,:])
            #     print()
            #     print(logit_layers[-1][self.j,1,:,:])
            #     print()
            #     print(qry_label.view(-1)[self.j])
            #     self.i = int(input())
            #     self.j = (self.j+1)%5

            # compute loss
            # full_edge_loss_layers: num_layers, num_tasks*num_qry, num_sp+1, num_sp+1
            
            if self.loss == 'circle':
                sp_edge_loss_layers = []
                qry_edge_loss_layers = []
                for logit_layer in logit_layers:
                    sp_sp, sp_sn, qry_sp, qry_sn = get_sp_sn(logit_layer[:, 0], full_edge, self.sp_edge_mask, self.qry_edge_mask)
                    sp_edge_loss_layers.append(self.edge_loss(sp_sp, sp_sn))
                    qry_edge_loss_layers.append(self.edge_loss(qry_sp, qry_sn))

                total_loss_layers = [sp_edge_loss_layer + qry_edge_loss_layer
                                     for sp_edge_loss_layer, qry_edge_loss_layer in
                                     zip(sp_edge_loss_layers, qry_edge_loss_layers)]
            #total_loss_layers = qry_edge_loss_layers
            elif self.loss == 'bce':
                # compute loss
                # full_edge_loss_layers: num_layers, num_tasks*num_qry, num_sp+1, num_sp+1
                full_edge_loss_layers = [self.edge_loss(logit_layer[:, 0], full_edge[:, 0]) for
                                         logit_layer in logit_layers]
            
                sp_edge_loss_layers = [full_edge_loss_layer*self.sp_edge_mask*self.evaluation_mask
                                        for full_edge_loss_layer in full_edge_loss_layers]
                qry_edge_loss_layers = [full_edge_loss_layer*self.qry_edge_mask*self.evaluation_mask
                                        for full_edge_loss_layer in full_edge_loss_layers]
            
              # weighted edge loss for balancing pos/neg
                num_pos_sp_edge = torch.sum(full_edge[:, 0]*self.sp_edge_mask*self.evaluation_mask)
                if num_pos_sp_edge > 0:
                    pos_sp_edge_loss_layers = [torch.sum(sp_edge_loss_layer*full_edge[:, 0]) / num_pos_sp_edge
                                              for sp_edge_loss_layer in sp_edge_loss_layers]
                else:
                    pos_sp_edge_loss_layers = [0 for _ in range(self.num_layers)]
                num_neg_sp_edge = torch.sum(full_edge[:, 1]*self.sp_edge_mask*self.evaluation_mask)
                neg_sp_edge_loss_layers = [torch.sum(sp_edge_loss_layer*full_edge[:, 1]) / num_neg_sp_edge
                                              for sp_edge_loss_layer in sp_edge_loss_layers]

                num_pos_qry_edge = torch.sum(full_edge[:, 0]*self.qry_edge_mask*self.evaluation_mask)
                pos_qry_edge_loss_layers = [torch.sum(qry_edge_loss_layer*full_edge[:, 0]) / num_pos_qry_edge
                                              for qry_edge_loss_layer in qry_edge_loss_layers]

                num_neg_qry_edge = torch.sum(full_edge[:, 1]*self.qry_edge_mask*self.evaluation_mask)
                neg_qry_edge_loss_layers = [torch.sum(qry_edge_loss_layer*full_edge[:, 1]) / num_neg_qry_edge
                                              for qry_edge_loss_layer in qry_edge_loss_layers]
                
                sp_edge_loss_layers = [pos_sp_edge_loss_layer + neg_sp_edge_loss_layer for 
                                    (pos_sp_edge_loss_layer, neg_sp_edge_loss_layer) in 
                                    zip(pos_sp_edge_loss_layers, neg_sp_edge_loss_layers)]

            # last layer is not important
                sp_edge_loss_layers[-1] = 0
    
                qry_edge_loss_layers = [pos_qry_edge_loss_layer + neg_qry_edge_loss_layer for
                                          (pos_qry_edge_loss_layer, neg_qry_edge_loss_layer) in
                                          zip(pos_qry_edge_loss_layers, neg_qry_edge_loss_layers)]
            
                total_loss_layers = [sp_edge_loss_layer + qry_edge_loss_layer
                            for sp_edge_loss_layer, qry_edge_loss_layer in zip(sp_edge_loss_layers, qry_edge_loss_layers) ]
            

            #total_loss_layers = qry_edge_loss_layers
            # compute accuracy
            # edge
            num_qry_edge = torch.sum(self.qry_edge_mask*self.evaluation_mask)
            all_edge_pred_layers = [hit(logit_layer, full_edge[:, 1].long()) for logit_layer in logit_layers]
            qry_edge_acc_layers = [torch.sum(all_edge_pred_layer*self.qry_edge_mask*self.evaluation_mask)
                                      / num_qry_edge for all_edge_pred_layer in all_edge_pred_layers]

            # node
            num_qry_node = self.num_tasks*self.num_queries
            sp_label_n = sp_label.unsqueeze(1).repeat(1, self.num_queries, 1).view(self.num_tasks*self.num_queries, self.num_supports)
            
            sum_logit_layer = sum(logit_layers)
            all_node_pred_layer = torch.bmm(sum_logit_layer[:,0,:,:-1], one_hot_encode(self.num_ways, sp_label_n).to(self.device)).max(-1)[1]
            qry_node_pred_layer = all_node_pred_layer[:, -1]
            qry_node_acc_layer = torch.sum(torch.eq(qry_node_pred_layer, qry_label.view(-1))).float() / num_qry_node
            
            all_node_pred_layers = [torch.bmm(logit_layer[:, 0, :, :-1],
                                              one_hot_encode(self.num_ways, sp_label_n).to(self.device)).max(-1)[1]
                                    for logit_layer in logit_layers]
            qry_node_pred_layers = [all_node_pred_layer[:, -1] for all_node_pred_layer in all_node_pred_layers]
            qry_node_acc_layers = [torch.sum(torch.eq(qry_node_pred_layer,
                                                      qry_label.view(-1))).float() / num_qry_node
                                                      for qry_node_pred_layer in qry_node_pred_layers]


            # update model, last layer has more weight
            total_loss = []
            #total_loss.append(total_loss_layers[0].view(-1))
            #total_loss.append(total_loss_layers[-1].view(-1))
            for i, total_loss_layer in enumerate(total_loss_layers):
                if i < len(total_loss_layers)-1:
                    total_loss += [total_loss_layer.view(-1) * 0.5]
                else:
                    total_loss += [total_loss_layer.view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))

            total_loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            # logging
            # self.smooth_qry_loss.append(list(zip(pos_qry_edge_loss_layers, neg_qry_edge_loss_layers)))
            # self.smooth_sp_loss.append(list(zip(pos_sp_edge_loss_layers, neg_sp_edge_loss_layers)))
            self.smooth_qry_loss.append(qry_edge_loss_layers)
            self.smooth_sp_loss.append(sp_edge_loss_layers)
            self.smooth_edge_acc.append(qry_edge_acc_layers)
            self.smooth_node_acc.append(qry_node_acc_layers)
            self.smooth_avg_node_acc.append(qry_node_acc_layer)

            if iter % self.train_interval == 0:
                self.smooth_qry_loss = torch.mean(torch.tensor(self.smooth_qry_loss), 0)
                self.smooth_sp_loss = torch.mean(torch.tensor(self.smooth_sp_loss), 0)
                self.smooth_edge_acc = torch.mean(torch.tensor(self.smooth_edge_acc), 0)
                self.smooth_node_acc = torch.mean(torch.tensor(self.smooth_node_acc), 0)
                self.smooth_avg_node_acc = torch.mean(torch.tensor(self.smooth_avg_node_acc))
                for i, (sp_loss, qry_loss, edge_acc, node_acc) in enumerate(
                        zip(self.smooth_sp_loss, self.smooth_qry_loss, self.smooth_edge_acc, self.smooth_node_acc)):
                    self.logger.info(' {0} th iteration, '
                                     'sp_loss_{5}: {1:.3f}, '
                                     'qry_loss_{5}: {2:.3f}, '
                                     'edge_accr_{5}: {3:.3f}, '
                                     'node_accr_{5}: {4:.3f}'
                                     .format(iter,
                                             sp_loss,
                                             qry_loss,
                                             edge_acc,
                                             node_acc,
                                             i
                                             ))
                '''
                for i, (sp_loss, qry_loss, edge_acc, node_acc) in enumerate(zip(self.smooth_sp_loss, self.smooth_qry_loss, self.smooth_edge_acc, self.smooth_node_acc)):
                    self.logger.info(' {0} th iteration, '
                                     'pos_sp_loss_{7}: {1:.3f}, '
                                     'neg_sp_loss_{7}: {2:.3f}, '
                                     'pos_qry_loss_{7}: {3:.3f}, '
                                     'neg_qry_loss_{7}: {4:.3f}, '
                                     'edge_accr_{7}: {5:.3f}, '
                                     'node_accr_{7}: {6:.3f}'
                                     .format(iter,
                                             sp_loss[0],
                                             sp_loss[1],
                                             qry_loss[0],
                                             qry_loss[1],
                                             edge_acc,
                                             node_acc,
                                             i
                                             ))
                '''

                self.logger.info(' {0} th iteration, avg_node_accr: {1:.3f}'
                                 .format(iter, self.smooth_avg_node_acc))
                self.smooth_sp_loss = []
                self.smooth_qry_loss = []
                self.smooth_edge_acc = []
                self.smooth_node_acc = []
                self.smooth_avg_node_acc = []
            # evaluation

            if iter % self.val_interval == 0 and iter != 0:
                val_accs = self.evaluate()
                for i, val_acc in enumerate(val_accs):
                    self.best_accs[i] = max(self.best_accs[i], val_acc)
                    self.logger.info(' {0} th iteration, val_acc_{3}: {1:.3f}, best_val_acc: {2:.3f}'
                                     .format(iter, val_acc, self.best_accs[i], i))

                torch.save({'iter': iter,
                            'emb': self.embeddingNet.state_dict(),
                            'gnn': self.graphNet.state_dict(),
                            'optim': self.optimizer.state_dict()}, os.path.join(self.expr_folder, 'model.pt'))

    def evaluate(self):
        # set as test mode
        self.embeddingNet.eval()
        self.graphNet.eval()
        qry_node_preds = torch.zeros(self.test_iters, self.num_layers+1, self.num_tasks*self.num_queries).to(self.device)
        qry_labels = torch.zeros(self.test_iters, self.num_tasks*self.num_queries).to(self.device)
        qry_node_accs = []
        with torch.no_grad():
            #for name, param in self.graphNet.named_parameters():
            #    print(name, '      ', param)

            for iter in range(self.test_iters):
                sp_data, sp_label, _, qry_data, qry_label, _ = self.test_dataloader.get_task_batch()

                full_data = torch.cat([sp_data, qry_data], 0)
                full_edge = label2edge(sp_label, qry_label).to(self.device)

                # full data: num_tasks*num_sp + num_tasks*qry, num_emb_feat
                # sp_data: num_tasks, num_sp, num_emb_feat
                # qry_data: num_tasks*num_qry, 1, num_emb_feat
                full_data = self.embeddingNet(full_data)
                sp_data = full_data[:self.num_tasks * self.num_supports, :].view(self.num_tasks, self.num_supports,
                                                                                 self.num_emb_feats)
                qry_data = full_data[self.num_tasks * self.num_supports:, :].view(self.num_tasks * self.num_queries, 1,
                                                                                  self.num_emb_feats)

                # sp_data: num_tasks*num_qry, num_sp, num_emb_feat
                sp_data = sp_data.unsqueeze(1).repeat(1, self.num_queries, 1, 1)
                sp_data = sp_data.view(self.num_tasks * self.num_queries, self.num_supports, self.num_emb_feats)

                # concat the sp and qry to a graph
                # input_node_feat: num_tasks*num_qry, num_sp+1, num_emb_feat
                # input_edge_feat: num_tasks*num_qry, 2, num_sp+1, num_sp+1
                input_node_feat = torch.cat([sp_data, qry_data], 1)

                # set the qry to others as 0.5 while keep qry to itself as 1
                #input_edge_feat = 0.5*torch.ones(self.num_tasks*self.num_queries, self.num_samples, self.num_samples, self.num_emb_feats).to(self.device)
                input_edge_feat = full_edge.clone()

                # qry to others
                input_edge_feat[:, :, -1, :-1] = 0.5
                input_edge_feat[:, :, :-1, -1] = 0.5

                # qry to itself
                input_edge_feat[:, 0, -1, -1] = 1
                input_edge_feat[:, 1, -1, -1] = 0
                #input_edge_feat = input_edge_feat.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_emb_feats)
                if self.gnn_net == 'ours':
                    input_edge_feat = input_edge_feat[:,0].unsqueeze(-1).repeat(1,1,1,self.num_emb_feats)

                # logit_layers: num_layers, num_tasks*num_qry, 2, num_sp+1, num_sp+1
                logit_layers = self.graphNet(node_feats=input_node_feat, edge_feats=input_edge_feat)
                #logit_layers = [F.normalize(torch.sum(logit_layer, -1), p=1, dim=-1) for logit_layer in logit_layers]

                # node
                sp_label_n = sp_label.unsqueeze(1).repeat(1, self.num_queries, 1).view(
                                self.num_tasks * self.num_queries, self.num_supports)
                
                
                # print(logit_layers[0][0,0,:,:])
                # print()
                # print(logit_layers[-1][0,0,:,:])
                # print()
                # print(qry_label.view(-1)[0])
                # input()
                sum_logit_layer = sum(logit_layers)
                all_node_pred_layer = torch.bmm(sum_logit_layer[:,0,:,:-1], 
                                                one_hot_encode(self.num_ways, sp_label_n).to(self.device)
                                                ).max(-1)[1]
                qry_node_pred_layer = all_node_pred_layer[:, -1] 

                all_node_pred_layers = [torch.bmm(logit_layer[:, 0, :, :-1],
                                                   one_hot_encode(self.num_ways, sp_label_n).to(self.device)).max(-1)[1]
                                                   for logit_layer in logit_layers]

                for i, all_node_pred_layer in enumerate(all_node_pred_layers):
                    qry_node_preds[iter, i, :] = all_node_pred_layer[:, -1]
                qry_node_preds[iter, -1, :] = qry_node_pred_layer

                qry_labels[iter, :] = qry_label.view(-1)

            num_qry_node = self.num_tasks * self.num_queries * self.test_iters
            for i in range(self.num_layers+1):
                qry_node_accs.append(torch.sum(torch.eq(qry_node_preds[:, i, :], qry_labels)).float() / num_qry_node)

        return qry_node_accs




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')

    # Fundamental setting
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_ways', type=int, default='5')
    parser.add_argument('--num_shots', type=int, default='5')
    parser.add_argument('--num_tasks', type=int, default='5')
    #parser.add_argument('--num_queries', type=int, default='1')
    #parser.add_argument('--seed', type=float, default='0')
    parser.add_argument('--train_iters', type=int, default='5001')
    parser.add_argument('--test_iters', type=int, default='500')
    parser.add_argument('--train_interval', type=int, default='100')
    parser.add_argument('--val_interval', type=int, default='1000')
    parser.add_argument('--expr', type=str, default='experiment/')
    #parser.add_argument('--ckpt', type=str, default='04-07-14-51-02')
    parser.add_argument('--ckpt', type=str, default=None)
    #parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mode', type=str, default='train')

    # hyper-parameter setting
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--weight_decay', type=float, default='1e-6')
    parser.add_argument('--dec_lr', type=float, default='1000')


    # data loading setting
    parser.add_argument('--dataset', type=str, default='ShapeNetCore')
    parser.add_argument('--test_size', type=float, default='0.5')
    parser.add_argument('--num_points', type=int, default='1024')

    # data transform setting
    parser.add_argument('--shift_range', type=float, default='0')
    parser.add_argument('--x_range', type=float, default='0')
    parser.add_argument('--y_range', type=float, default='0')
    parser.add_argument('--z_range', type=float, default='6.28')
    parser.add_argument('--max_scale', type=float, default='1')
    parser.add_argument('--min_scale', type=float, default='1')
    parser.add_argument('--sigma', type=float, default='0.02')
    parser.add_argument('--clip', type=float, default='0.04')

    # Embedding setting
    parser.add_argument('--k', type=int, default='20')
    parser.add_argument('--num_emb_feats', type=int, default='128')
    parser.add_argument('--emb_net', type=str, default='pointnet')

    # GraphNetwork section
    parser.add_argument('--gnn_net', type=str, default='ours')
    #parser.add_argument('--gnn_net', type=str, default='egnn')
    parser.add_argument('--num_node_feats', type=int, default='128')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--edge_p', type=float, default='0.3')
    parser.add_argument('--p', type=float, default='0')
    parser.add_argument('--dropout', type=float, default='0')
    
    # loss setting
    parser.add_argument('--loss', type=str, default='circle')
    args = parser.parse_args()

    model = Model(args, partition=args.mode)
    try:
        if args.mode == 'train':
            model.train()
        else:
            val_acc = model.evaluate()
            print(val_acc)
    finally:
        logging.shutdown()

