import os
from tqdm import tqdm
import glob
import h5py
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def download(root):
    BASE_DIR = root
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
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

def label2edge(label):
    # get size
    num_samples = label.size(1)

    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)

    # compute edge
    edge = torch.eq(label_i, label_j).float()

    # expand
    edge = edge.unsqueeze(1)
    edge = torch.cat([edge, 1 - edge], 1)
    return edge


def hit(logit, label):
    pred = logit.max(1)[1]
    hit = torch.eq(pred, label).float()
    return hit


def one_hot_encode(num_classes, class_idx):
    return torch.eye(num_classes)[class_idx]

if __name__ == '__main__':
    dataset_to_fewshot_dataset('./', 0)