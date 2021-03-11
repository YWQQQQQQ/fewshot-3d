import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import utils
import argparse
from tqdm import tqdm
import random
import torch
from transforms import Transforms3D


class ModelNet40Loader(Dataset):
    def __init__(self, args, partition='train'):
        super(ModelNet40Loader, self).__init__()
        # Dataset basic info
        self.args = args
        self.partition = partition # train, val, test
        self.num_points = self.args.num_points # 1024, 2048
        self.root = self.args.dataset_root
        self.num_classes = 40

        # Define transform methods
        if self.partition == 'train':
            self.transform = transforms.Compose([Transforms3D.List2Array(),
                                                 Transforms3D.ToTensor(self.args),
                                                 Transforms3D.Translation(self.args),
                                                 Transforms3D.Scaling(self.args),
                                                 Transforms3D.Perturbation(self.args),
                                                 Transforms3D.Rotation(self.args)
                                                 ])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([Transforms3D.List2Array(),
                                                 Transforms3D.ToTensor(self.args)
                                                 ])

        # load dataset
        self.data = self.load_data()

    def load_data(self):
        # Check if dataset exists. If not, download
        utils.download()

        # dataset path
        BASE_DIR = self.root
        DATA_DIR = os.path.join(BASE_DIR, 'dataset')

        # Read dataset
        all_data = [[] for c in range(self.num_classes)]

        for h5_name in tqdm(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % self.partition))):
            f = h5py.File(h5_name)
            datas = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64').squeeze()
            f.close()

            for data, label in zip(datas, labels):
                all_data[label].append(data[:self.num_points, :])
        return all_data  # num_classes, num_object

    def get_task_batch(self,
                       num_tasks=4,
                       num_ways=5,
                       num_shots=1,
                       num_queries=1,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        # init task batch data
        sp_data, sp_rel_label, sp_abs_label, qry_data, qry_rel_label, qry_abs_label = [], [], [], [], [], []

        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks, self.num_points, 3],
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')

            sp_data.append(data)
            sp_rel_label.append(label)
            sp_abs_label.append(label)

        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks, self.num_points, 3],
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            qry_data.append(data)
            qry_rel_label.append(label)
            qry_abs_label.append(label)

        # get full class list in dataset
        full_class_list = [i for i in range(40)]

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    sp_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    sp_rel_label[i_idx + c_idx * num_shots][t_idx] = c_idx
                    sp_abs_label[i_idx + c_idx * num_shots][t_idx] = task_class_list[c_idx]

                # load sample for query set
                for i_idx in range(num_queries):
                    qry_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    qry_rel_label[i_idx + c_idx * num_queries][t_idx] = c_idx
                    qry_abs_label[i_idx + c_idx * num_shots][t_idx] = task_class_list[c_idx]

        new_sp_data = self.transform(sp_data)
        new_qry_data = self.transform(qry_data)

        sp_rel_label = torch.tensor(sp_rel_label).view(-1)
        qry_rel_label = torch.tensor(qry_rel_label).view(-1)

        sp_abs_label = torch.tensor(sp_abs_label).view(-1)
        qry_abs_label = torch.tensor(qry_abs_label).view(-1)

        return [sp_data, new_sp_data, sp_abs_label, qry_data, new_qry_data, qry_abs_label]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--num_points', type=int, default='1024')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--shift_range', type=float, default='0')
    parser.add_argument('--angle_range', type=float, default='0')
    parser.add_argument('--max_scale', type=float, default='1')
    parser.add_argument('--min_scale', type=float, default='1')
    parser.add_argument('--sigma', type=float, default='0.1')
    parser.add_argument('--clip', type=float, default='0.2')


    args = parser.parse_args()

    shape_name = []
    with open('../dataset/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as f:
        for i in range(40):
            shape_name.append(f.readline())

    dataset = ModelNet40Loader(args)
    sp_data, new_sp_data, sp_abs_label, qry_data, new_qry_data, qry_abs_label = dataset.get_task_batch()
    fig = plt.figure()
    for i in range(4):
        data = qry_data[0][i]
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        ax = fig.add_subplot(2,5,i+1, projection='3d')
        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   cmap='Blues',
                   marker="o")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.title(shape_name[int(sp_abs_label[i].item())])
        plt.xlabel('x')
        plt.ylabel('y')

        data = new_qry_data[i]
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

        plt.title(shape_name[int(sp_abs_label[i].item())])
        plt.xlabel('x')
        plt.ylabel('y')

    plt.show()

        #print(torch.sum((qry_data[i,:,0]-qry_data[i,:,1])**2))
        #print(torch.sum((new_qry_data[i,:,0]-new_qry_data[i,:,1])**2))

