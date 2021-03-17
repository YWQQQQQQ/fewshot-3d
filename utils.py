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


def download(root):
    BASE_DIR = root
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def dataset_to_fewshot_dataset(root, seed=0, test_size=0.2):
    download(root)

    BASE_DIR = root
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'modelnet_fewshot')
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(os.path.join(OUTPUT_DIR, 'train.h5')) or not os.path.exists(os.path.join(OUTPUT_DIR, 'train.h5')):

        num_classes = 40

        # Read dataset
        all_data = [[] for _ in range(num_classes)]

        for h5_name in tqdm(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_*.h5'))):

            f = h5py.File(h5_name)
            datas = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64').squeeze()
            f.close()

            for data, label in zip(datas, labels):
                all_data[label].append(data)

        # Split the datset set into train set (default 32 classes) and test set (default 8 classes)
        full_class_list = [i for i in range(40)]
        train_class_idx, test_class_idx = train_test_split(full_class_list, test_size=test_size, random_state=seed)

        train_data = {'data':None, 'label':None}
        test_data = {'data':None, 'label':None}

        train_data['data'] = np.concatenate([all_data[idx] for idx in train_class_idx], axis=0)
        train_data['label'] = np.concatenate([np.repeat(idx, len(all_data[idx])) for idx in train_class_idx], axis=0)

        test_data['data'] = np.concatenate([all_data[idx] for idx in test_class_idx], axis=0)
        test_data['label'] = np.concatenate([np.repeat(idx, len(all_data[idx])) for idx in test_class_idx], axis=0)

        # Create h5py file
        train_hf = h5py.File(os.path.join(OUTPUT_DIR, 'train.h5'), 'w')
        test_hf = h5py.File(os.path.join(OUTPUT_DIR, 'test.h5'), 'w')

        train_hf.create_dataset('data', data=train_data['data'])
        train_hf.create_dataset('label', data=train_data['label'])

        test_hf.create_dataset('data', data=test_data['data'])
        test_hf.create_dataset('label', data=test_data['label'])

        train_hf.close()
        test_hf.close()


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
    root = args.root

    abs_label, rel_label = labels
    shape_name = []
    with open(os.path.join(root, 'dataset/modelnet40_ply_hdf5_2048/shape_names.txt'), 'r') as f:
        for i in range(40):
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
    dataset_to_fewshot_dataset('./', 0)
