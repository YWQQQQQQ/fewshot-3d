import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import utils
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

        self.num_ways = args.num_ways
        self.num_shots = args.num_shots
        self.num_queries = args.num_queries
        self.num_tasks = args.num_tasks
        self.seed = args.seed

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
        utils.dataset_to_fewshot_dataset(root=self.root, test_size=self.test_size)

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
        random.seed(self.seed)

        # init task batch data
        sp_data, sp_rel_label, sp_abs_label, qry_data, qry_rel_label, qry_abs_label = [], [], [], [], [], []

        num_supports = self.num_ways * self.num_shots
        for _ in range(self.num_tasks):
            data = np.zeros(shape=[num_supports, self.num_points, 3],
                            dtype='float32')
            rel_label = np.zeros(shape=[num_supports],
                             dtype='float32')
            abs_label = np.zeros(shape=[num_supports],
                                 dtype='float32')
            sp_data.append(data)
            sp_rel_label.append(rel_label)
            sp_abs_label.append(abs_label)

        num_total_queries = self.num_ways * self.num_queries
        for _ in range(self.num_tasks):
            data = np.zeros(shape=[num_total_queries, self.num_points, 3],
                            dtype='float32')
            rel_label = np.zeros(shape=[num_total_queries],
                             dtype='float32')
            abs_label = np.zeros(shape=[num_total_queries],
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
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], self.num_shots + self.num_queries)

                # load sample for support set
                for i_idx in range(self.num_shots):
                    # set data
                    sp_data[t_idx][i_idx + c_idx * self.num_shots] = class_data_list[i_idx]
                    sp_rel_label[t_idx][i_idx + c_idx * self.num_shots] = c_idx
                    sp_abs_label[t_idx][i_idx + c_idx * self.num_shots] = task_class_list[c_idx]

                # load sample for query set
                for i_idx in range(self.num_queries):
                    qry_data[t_idx][i_idx + c_idx * self.num_queries] = class_data_list[self.num_shots + i_idx]
                    qry_rel_label[t_idx][i_idx + c_idx * self.num_queries] = c_idx
                    qry_abs_label[t_idx][i_idx + c_idx * self.num_queries] = task_class_list[c_idx]

        new_sp_data = self.transform(sp_data)
        new_qry_data = self.transform(qry_data)

        sp_rel_label = torch.tensor(sp_rel_label)
        qry_rel_label = torch.tensor(qry_rel_label)

        sp_abs_label = torch.tensor(sp_abs_label)
        qry_abs_label = torch.tensor(qry_abs_label)

        return [new_sp_data, sp_rel_label, sp_abs_label, new_qry_data, qry_rel_label, qry_abs_label]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../')
    parser.add_argument('--num_points', type=int, default='1024')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--shift_range', type=float, default='2')
    parser.add_argument('--angle_range', type=float, default='6.28')
    parser.add_argument('--max_scale', type=float, default='2')
    parser.add_argument('--min_scale', type=float, default='0.5')
    parser.add_argument('--sigma', type=float, default='0.01')
    parser.add_argument('--clip', type=float, default='0.02')
    parser.add_argument('--test_size', type=float, default='0.2')
    parser.add_argument('--num_ways', type=int, default='5')
    parser.add_argument('--num_shots', type=int, default='1')
    parser.add_argument('--num_tasks', type=int, default='5')
    parser.add_argument('--num_queries', type=int, default='1')
    parser.add_argument('--seed', type=float, default='0')

    args = parser.parse_args()

    shape_name = []
    with open('../dataset/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as f:
        for i in range(40):
            shape_name.append(f.readline())

    dataset = ModelNet40Loader(args)


    new_sp_data, sp_data, sp_abs_label, new_qry_data, qry_data, qry_abs_label = dataset.get_task_batch()
    # print(new_sp_data.shape)
    # print(sp_data.shape)
    # print(sp_abs_label.shape)
    # print(new_qry_data.shape)
    # print(qry_data.shape)
    # print(qry_abs_label.shape)

    fig = plt.figure()
    for i in range(args.num_ways):
        data = sp_data[3][i]
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        ax = fig.add_subplot(2,5,i+1, projection='3d')
        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.title(shape_name[int(sp_abs_label[3][i].item())])
        plt.xlabel('x')
        plt.ylabel('y')

        data = new_sp_data[15+i]
        x = data[0, :]
        y = data[1, :]
        z = data[2, :]

        ax = fig.add_subplot(2,5,i+6, projection='3d')
        ax.scatter(x.cpu(),  # x
                   y.cpu(),  # y
                   z.cpu(),  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.title(shape_name[int(sp_abs_label[3][i].item())])
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()
    fig = plt.figure()

    for i in range(args.num_ways):
        data = qry_data[3][i]
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        ax = fig.add_subplot(2,5,i+1, projection='3d')
        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.title(shape_name[int(qry_abs_label[3][i].item())])
        plt.xlabel('x')
        plt.ylabel('y')

        data = new_qry_data[15+i]
        x = data[0, :]
        y = data[1, :]
        z = data[2, :]

        ax = fig.add_subplot(2,5,i+6, projection='3d')
        ax.scatter(x.cpu(),  # x
                   y.cpu(),  # y
                   z.cpu(),  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.title(shape_name[int(qry_abs_label[3][i].item())])
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()
        #print(torch.sum((qry_data[i,:,0]-qry_data[i,:,1])**2))
        #print(torch.sum((new_qry_data[i,:,0]-new_qry_data[i,:,1])**2))

