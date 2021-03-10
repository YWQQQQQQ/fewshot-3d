import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from transforms.Transforms import Translation, Rotation
import utils
import argparse
from tqdm import tqdm
import random
import torch


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
            self.transform = transforms.Compose([Translation(),
                                                 Rotation(),
                                                 ToTensor()
                                                 ])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([ToTensor()
                                                 ])

        # load dataset
        self.data = self.load_data(partition)

    def load_data(self, partition):
        # Check if dataset exists. If not, download
        utils.download()

        # dataset path
        BASE_DIR = self.root
        DATA_DIR = os.path.join(BASE_DIR, 'dataset')

        # Read dataset
        all_data = [[] for c in range(self.num_classes)]

        for h5_name in tqdm(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))):
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
        support_data, support_label, query_data, query_label = [], [], [], []

        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks, self.num_points, 3],
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)

        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks, self.num_points, 3],
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

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
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

            support_data = self.transform(support_data)
            support_data = self.transform(support_data)

            # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
            support_data = torch.stack([data.float().to(self.args.device) for data in support_data], 1)
            support_label = torch.stack([torch.from_numpy(label).float().to(self.args.device) for label in support_label],
                                        1)
            query_data = torch.stack([data.float().to(self.args.device) for data in query_data], 1)
            query_label = torch.stack([torch.from_numpy(label).float().to(self.args.device) for label in query_label], 1)

            return [support_data, support_label, query_data, query_label]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification with Few-shot Learning')
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--num_points', type=int, default='1024')
    args = parser.parse_args()

    dataset = ModelNet40Loader(args)
    support_data, support_label, query_data, query_label = dataset.get_task_batch()
    print(support_data.shape)
    print(support_label.shape)
    print(query_data.shape)
    print(query_label.shape)

    #print(len(dataset.data))
    #for data in dataset.data:
    #    print(len(data))
