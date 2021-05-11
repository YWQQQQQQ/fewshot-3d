import os
from tqdm import tqdm
import glob
import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import logging
import matplotlib.pyplot as plt


def download(root, dataset='ModelNet40'):
    BASE_DIR = os.path.join(root, 'dataset')
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    if dataset == 'ModelNet40':
        if not os.path.exists(os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = 'modelnet40_ply_hdf5_2048.zip'
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv -r %s %s' % (zipfile[:-4], BASE_DIR))
            os.system('rm %s' % (zipfile))
    elif dataset == 'ShapeNetCore':
        if not os.path.exists(os.path.join(BASE_DIR, 'shapenetcorev2_hdf5_2048')):
            www = 'https://drive.google.com/file/d/16aNARDkJz7jgGI_e9kpyHJ3_3KGQfANr/view?usp=sharing'
            zipfile = 'shapenetcorev2_hdf5_2048.zip'
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv -r %s %s' % (zipfile[:-4], BASE_DIR))
            os.system('rm %s' % (zipfile))






def label2edge(sp_label, qry_label):
    # get size
    num_tasks, num_supports = sp_label.size()
    _, num_queries = qry_label.size()

    full_edge = torch.zeros(num_queries, num_tasks, num_supports+1, num_supports+1)

    for q_idx in range(num_queries):

        label = torch.cat([sp_label, qry_label[:, q_idx].unsqueeze(-1)], 1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_supports+1)
        label_j = label_i.transpose(1, 2)

        full_edge[q_idx, :, :, :] = torch.eq(label_i, label_j).float()
        #print(full_edge.shape)
    full_edge = full_edge.transpose(0, 1).contiguous()
    full_edge = full_edge.view(num_tasks*num_queries, 1, num_supports+1, num_supports+1)
    full_edge = torch.cat([full_edge, 1-full_edge], 1)

    # full_edge: num_tasks*num_queries, 2, num_supports+1, num_supports+1
    return full_edge


def hit(logit, label):
    pred = logit.max(1)[1]
    hit = torch.eq(pred, label).float()
    return hit


def one_hot_encode(num_classes, class_idx):
    return torch.eye(num_classes)[class_idx]


def get_logger(expr='./experiment', filename='train.log'):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    log_name = os.path.join(expr, filename)
    fh = logging.FileHandler(log_name, mode='w', encoding='utf-8')
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y %b %d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def plot_pc(pc):
    if len(pc.size()) == 2:
        num_pc = 1
        pcs = [pc]
    else:
        num_pc = pc.size()[0]
        pcs = pc

    fig = plt.figure()
    for i, pc in enumerate(pcs):
        x = pc[0, :]
        y = pc[1, :]
        z = pc[2, :]

        ax = fig.add_subplot(1, num_pc, i + 1, projection='3d')
        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        #plt.title(shape_name[int(sp_abs_label[task][i].item())] + ' ' + str(sp_rel_label[task][i]))
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()


def plot_transform_pc(pc_ori, pc_trans, labels, args):
    num_tasks = args.num_tasks
    num_ways = args.num_ways
    num_ways = args.num_ways
    root = args.root

    abs_label, rel_label = labels
    shape_name = []
    if args.dataset == 'ModelNet40':
        with open(os.path.join(root, 'dataset/modelnet40_ply_hdf5_2048/shape_names.txt'), 'r') as f:
            for i in range(40):
                shape_name.append(f.readline())
    elif args.dataset == 'ShapeNetCore':
        with open(os.path.join(root, 'dataset/shapenetcorev2_hdf5_2048/shape_names.txt'), 'r') as f:
            for i in range(55):
                shape_name.append(f.readline())
    for t in range(num_tasks):
        fig = plt.figure()
        for i in range(num_ways):
            x = pc_ori[t*num_ways+i, 0, :]
            y = pc_ori[t*num_ways+i, 1, :]
            z = pc_ori[t*num_ways+i, 2, :]

            ax = fig.add_subplot(2, num_ways, i + 1, projection='3d')
            ax.scatter(x,  # x
                       y,  # y
                       z,  # z
                       cmap='Blues',
                       marker="o")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.title(shape_name[int(abs_label[t][i].item())] +' '+ str(rel_label[t][i]))
            plt.xlabel('x')
            plt.ylabel('y')

            x = pc_trans[t*num_ways+i, 0, :]
            y = pc_trans[t*num_ways+i, 1, :]
            z = pc_trans[t*num_ways+i, 2, :]

            ax = fig.add_subplot(2, num_ways, num_ways + i+1, projection='3d')
            ax.scatter(x,  # x
                       y,  # y
                       z,  # z
                       cmap='Blues',
                       marker="o")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.title(shape_name[int(abs_label[t][i].item())] +' '+ str(rel_label[t][i]))
            plt.xlabel('x')
            plt.ylabel('y')
        plt.show()

if __name__ == '__main__':
    None