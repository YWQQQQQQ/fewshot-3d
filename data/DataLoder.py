import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
import random
import torch
from data import Transforms3D


class ModelNet40Loader(Dataset):
    def __init__(self, args, partition='train'):
        super(ModelNet40Loader, self).__init__()
        # Dataset basic info

        self.partition = partition # train, val, test
        self.num_points = args.num_points # 1024, 2048
        self.root = args.root
        self.num_classes = 40
        self.test_size = args.test_size
        self.device = args.device
        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.num_supports = args.num_ways * args.num_shots
        self.num_queries = args.num_ways * 1
        self.num_tasks = args.num_tasks

        # Define transform methods
        if self.partition == 'train':
            self.transform = transforms.Compose([Transforms3D.List2Tensor(args),
                                                 Transforms3D.Translation(args),
                                                 Transforms3D.Scaling(args),
                                                 Transforms3D.Perturbation(args),
                                                 Transforms3D.Rotation(args)]
                                                )
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([Transforms3D.List2Tensor(args),
                                                 ])

        # load dataset
        self.data, self.classes = self.load_data()

    def load_data(self):
        # Check if dataset exists. If not, download
        dataset_to_fewshot_dataset(root=self.root, test_size=self.test_size)

        # dataset path
        BASE_DIR = self.root
        DATA_DIR = os.path.join(BASE_DIR, 'dataset')

        # Read dataset
        all_data = [[] for c in range(self.num_classes)]

        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet_fewshot', '%s.h5' % self.partition)):
            f = h5py.File(h5_name, 'r')
            datas = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64').squeeze()
            f.close()

            for data, label in zip(datas, labels):
                all_data[label].append(data[:self.num_points, :])
        idx_classes = []
        for i, data in enumerate(all_data):
            if len(data) > 0:
                idx_classes.append(i)
        return all_data, idx_classes

    def get_task_batch(self):
        # init task batch data
        sp_data, sp_rel_label, sp_abs_label, qry_data, qry_rel_label, qry_abs_label = [], [], [], [], [], []

        for _ in range(self.num_tasks):
            data = np.zeros(shape=[self.num_supports, self.num_points, 3],
                            dtype='float32')
            rel_label = np.zeros(shape=[self.num_supports],
                             dtype='float32')
            abs_label = np.zeros(shape=[self.num_supports],
                                 dtype='float32')
            sp_data.append(data)
            sp_rel_label.append(rel_label)
            sp_abs_label.append(abs_label)

        for _ in range(self.num_tasks):
            data = np.zeros(shape=[self.num_queries, self.num_points, 3],
                            dtype='float32')
            rel_label = np.zeros(shape=[self.num_queries],
                             dtype='float32')
            abs_label = np.zeros(shape=[self.num_queries],
                                 dtype='float32')
            qry_data.append(data)
            qry_rel_label.append(rel_label)
            qry_abs_label.append(abs_label)

        # for each task
        for t_idx in range(self.num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(self.classes, self.num_ways)

            # for each sampled class in task
            for c_idx in range(self.num_ways):
                # sample data for support and query (num_shots + 1)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], self.num_shots + 1)

                # load sample for support set
                for i_idx in range(self.num_shots):
                    # set data
                    sp_data[t_idx][i_idx + c_idx * self.num_shots] = class_data_list[i_idx]
                    sp_rel_label[t_idx][i_idx + c_idx * self.num_shots] = c_idx
                    sp_abs_label[t_idx][i_idx + c_idx * self.num_shots] = task_class_list[c_idx]

                # load sample for query set
                qry_data[t_idx][c_idx] = class_data_list[self.num_shots]
                qry_rel_label[t_idx][c_idx] = c_idx
                qry_abs_label[t_idx][c_idx] = task_class_list[c_idx]

        new_sp_data = self.transform(sp_data)
        new_qry_data = self.transform(qry_data)

        sp_rel_label = torch.tensor(sp_rel_label).long().to(self.device)
        qry_rel_label = torch.tensor(qry_rel_label).long().to(self.device)

        sp_abs_label = torch.tensor(sp_abs_label).long().to(self.device)
        qry_abs_label = torch.tensor(qry_abs_label).long().to(self.device)

        return [new_sp_data, sp_rel_label, sp_abs_label, new_qry_data, qry_rel_label, qry_abs_label]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../')
    parser.add_argument('--num_points', type=int, default='1024')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--shift_range', type=float, default='1')
    parser.add_argument('--angle_range', type=float, default='6.28')
    parser.add_argument('--max_scale', type=float, default='1,3')
    parser.add_argument('--min_scale', type=float, default='0.7')
    parser.add_argument('--sigma', type=float, default='0.01')
    parser.add_argument('--clip', type=float, default='0.02')
    parser.add_argument('--test_size', type=float, default='0.2')
    parser.add_argument('--num_ways', type=int, default='5')
    parser.add_argument('--num_shots', type=int, default='1')
    parser.add_argument('--num_tasks', type=int, default='5')
    #parser.add_argument('--num_queries', type=int, default='1')
    parser.add_argument('--seed', type=float, default='0')

    args = parser.parse_args()

    dataset = ModelNet40Loader(args)

    new_sp_data, sp_rel_label, sp_abs_label, new_qry_data, qry_rel_label, qry_abs_label = dataset.get_task_batch()

    plot_transform_pc(new_sp_data, new_qry_data, [sp_abs_label, sp_rel_label], args)



