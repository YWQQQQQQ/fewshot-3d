import torch
from torch import optim
from torch import nn
from data import DataLoder
from model import EmbeddingNetwork, GraphNetwork
import argparse
from glob import glob
from utils import *
import logging
import datetime
import os

def train(args, expr_folder):
    ROOT = args.root
    device = args.device

    # set edge mask (to distinguish support and query edges)
    num_tasks = args.num_tasks
    num_points = args.num_points
    num_emb_feats = args.num_emb_feats
    num_ways = args.num_ways
    num_supports = num_ways * args.num_shots
    num_all_queries = args.num_ways * args.num_queries
    num_samples = num_supports + num_all_queries
    support_edge_mask = torch.zeros(args.num_tasks, num_samples, num_samples).to(args.device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask

    evaluation_mask = torch.ones(args.num_tasks, num_samples, num_samples).to(args.device)
    evaluation_mask[:, num_supports:, num_supports:] = 0
    if args.dataset_name == 'ModelNet40':
        train_dataloader = DataLoder.ModelNet40Loader(args, partition='train')
        test_dataloader = DataLoder.ModelNet40Loader(args, partition='test')
    else:
        raise NotImplemented

    if not os.path.exists(glob(ROOT, 'expr')):
        os.mkdir(glob(ROOT, 'expr'))

    # networks
    embeddingNet = EmbeddingNetwork.Embedding(args)
    graphNet = GraphNetwork.GraphNetwork(args)

    if args.expr is not None and os.path.exists(glob(ROOT, 'expr', args.expr)):
        emb_ckpt = glob(ROOT, 'expr', args.expr, 'emb.tar')
        emb_ckpt = torch.load(emb_ckpt)
        gnn_ckpt = glob(ROOT, 'expr', args.expr, 'gnn.tar')
        gnn_ckpt = torch.load(gnn_ckpt)
        embeddingNet.load_state_dict(emb_ckpt['model_state_dict'])
        graphNet.load_state_dict(gnn_ckpt['model_state_dict'])

    # optimizer
    module_params = list(embeddingNet.parameters())+list(graphNet.parameters())
    optimizer = optim.Adam(params=module_params, lr=args.lr, weight_decay=args.weigh_decay)
    lambda_lr = lambda lr: lr*(0.5**(iter // args.dec_lr))
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    # loss
    edge_loss = nn.BCELoss(reduction='none')
    node_loss = nn.CrossEntropyLoss(reduction='none')

    # metrics
    train_acc, val_acc, test_acc = 0


    for iter in range(args.train_iterations):
        optimizer.zero_grad()
        sp_data, sp_label, _, qry_data, qry_label, _ = train_dataloader.get_task_batch()

        # set as single data
        full_data = torch.cat([sp_data, qry_data], 0)  # num_tasks*num_sp + num_tasks*num_qry, num_feat, num_points
        full_label = torch.cat([sp_label, qry_label], 1)
        full_edge = label2edge(full_label).to(device)

        # set init edge
        init_edge = full_edge.clone()  # num_tasks x 2 x num_samples x num_samples
        init_edge[:, :, num_supports:, :] = 0.5
        init_edge[:, :, :, num_supports:] = 0.5
        for i in range(num_all_queries):
            init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
            init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

        # set as train mode
        embeddingNet.train()
        graphNet.train()

        # (1) encode data
        full_data = embeddingNet(full_data)
        sp_data = full_data[:num_tasks*num_supports, :].view(num_tasks, num_supports, num_emb_feats)
        qry_data = full_data[num_tasks*num_supports:, :].view(num_tasks, num_all_queries, num_emb_feats)
        sp_data = sp_data.unsqueeze(1).repeat(1, num_all_queries, 1, 1)
        sp_data = sp_data.view(num_tasks*num_all_queries, num_supports, num_emb_feats)
        qry_data = qry_data.view(num_tasks*num_all_queries, num_emb_feats)
        input_node_feat = torch.cat([sp_data, qry_data], 1)  # (num_tasks x num_total_queries) x (num_support + 1) x featdim
        input_edge_feat = 0.5 * torch.ones(num_tasks, 2, num_supports + 1, num_supports + 1).to(device)  # num_tasks x 2 x (num_support + 1) x (num_support + 1)

        input_edge_feat[:, :, :num_supports, :num_supports] = init_edge[:, :, :num_supports, :num_supports] # num_tasks x 2 x (num_support + 1) x (num_support + 1)
        input_edge_feat = input_edge_feat.unsqueeze(1).repeat(1, num_all_queries, 1, 1)
        input_edge_feat = input_edge_feat.view(num_tasks*num_all_queries, 2, num_supports + 1, num_supports + 1) #(num_tasks x num_queries) x 2 x (num_support + 1) x (num_support + 1)

        # logit: (num_tasks x num_queries) x 2 x (num_support + 1) x (num_support + 1)
        logit_layers = graphNet(node_feat=input_node_feat, edge_feat=input_edge_feat)

        logit_layers = [logit_layer.view(num_tasks, num_all_queries, 2, num_supports + 1, num_supports + 1) for logit_layer in logit_layers]

        # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
        full_logit_layers = []
        for l in range(args.num_layers):
            full_logit_layers.append(torch.zeros(args.num_tasks, 2, num_samples, num_samples).to(device))

        for l in range(args.num_layers):
            full_logit_layers[l][:, :, :num_supports, :num_supports] = logit_layers[l][:, :, :, :num_supports, :num_supports].mean(1)
            full_logit_layers[l][:, :, :num_supports, num_supports:] = logit_layers[l][:, :, :, :num_supports, -1].transpose(1, 2).transpose(2, 3)
            full_logit_layers[l][:, :, num_supports:, :num_supports] = logit_layers[l][:, :, :, -1, :num_supports].transpose(1, 2)

        # (4) compute loss
        full_edge_loss_layers = [edge_loss((1 - full_logit_layer[:, 0]), (1 - full_edge[:, 0])) for
                                 full_logit_layer in full_logit_layers]

        # weighted edge loss for balancing pos/neg
        pos_query_edge_loss_layers = [
            torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(
                query_edge_mask * full_edge[:, 0] * evaluation_mask) for full_edge_loss_layer in
            full_edge_loss_layers]
        neg_query_edge_loss_layers = [
            torch.sum(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / torch.sum(
                query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) for full_edge_loss_layer in
            full_edge_loss_layers]
        query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                  (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                  zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]

        # compute accuracy
        full_edge_accr_layers = [hit(full_logit_layer, 1 - full_edge[:, 0].long()) for full_logit_layer in
                                 full_logit_layers]
        query_edge_accr_layers = [torch.sum(full_edge_accr_layer * query_edge_mask * evaluation_mask) / torch.sum(
            query_edge_mask * evaluation_mask) for full_edge_accr_layer in full_edge_accr_layers]

        # compute  accuracy (num_tasks x num_quries x num_ways)
        query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports],
                                            one_hot_encode(num_ways, sp_label.long())) for
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

        optimizer.step()
        lr_scheduler.step()

        # logging
        logging.info('{0}th iteration, edge_loss:{1}, edge_accr:{2}, node_accr:{3}'.format(iter,
                                                                                           query_edge_loss_layers[-1],
                                                                                           query_edge_accr_layers[-1],
                                                                                           query_node_accr_layers[-1]
                                                                                           ))

        # evaluation
        if iter % args.val_interval == 0:
            val_acc = evaluate(partition='val')

            logging.info('{0}th iteration, val_accr:{1}'.format(iter,val_acc))

            save_checkpoint(embeddingNet.state_dict(), expr_folder, iter, 'embedding')
            save_checkpoint(graphNet.state_dict(), expr_folder, iter, 'graph')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')

    # Fundamental setting
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_ways', type=int, default='5')
    parser.add_argument('--num_shots', type=int, default='1')
    parser.add_argument('--num_tasks', type=int, default='5')
    parser.add_argument('--num_queries', type=int, default='1')
    parser.add_argument('--seed', type=float, default='0')
    parser.add_argument('--train_iterations', type=int, default='200')

    # data loading setting
    parser.add_argument('--dataset_name', type=str, default='ModelNet40')
    parser.add_argument('--test_size', type=float, default='0.2')
    parser.add_argument('--num_points', type=int, default='1024')

    # data transform setting
    parser.add_argument('--shift_range', type=float, default='2')
    parser.add_argument('--angle_range', type=float, default='6.28')
    parser.add_argument('--max_scale', type=float, default='2')
    parser.add_argument('--min_scale', type=float, default='0.5')
    parser.add_argument('--sigma', type=float, default='0.01')
    parser.add_argument('--clip', type=float, default='0.02')
    parser.add_argument('--test_size', type=float, default='0.2')

    # Embedding setting
    parser.add_argument('--k', type=int, default='20')
    parser.add_argument('--num_emb_feats', type=int, default='1024')

    # GraphNetwork section
    parser.add_argument('--num_node_feats', type=int, default='1024')
    parser.add_argument('--num_graph_layers', type=int, default='3')
    parser.add_argument('--edge_p', type=float, default='0')
    parser.add_argument('--feat_p', type=float, default='0')

    args = parser.parse_args()

    if not os.path.exists(args.expr):
        os.mkdir(args.expr)

    expr_folder = glob(args.expr, str(datetime.datetime.now())[5:19])
    if not os.path.exists(expr_folder):
        os.mkdir(expr_folder)

    logging.basicConfig(level=logging.INFO, filename=glob(expr_folder, 'train.log'), filemode='w')

    train(args, expr_folder)