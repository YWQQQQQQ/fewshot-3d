import numpy as np
import torch


class List2Tensor:
    def __init__(self, args):
        self.device = args.device

    def __call__(self, pcs):
        pcs = torch.tensor(pcs).to(self.device)
        test = pcs[2,3,:,:]
        pc_size = pcs.size()  # num_tasks, num_samples, num_points, num_features
        pcs = pcs.view(pc_size[0]*pc_size[1], pc_size[2], pc_size[3])
        pcs = pcs.transpose(1,2)
        return pcs


class Translation:
    def __init__(self, args):
        self.shift_range = args.shift_range
        self.device = args.device

    def __call__(self, pcs):
        pc_size = pcs.size()  # batch size*(num_support+num_query), num_features, num_points

        # Translation matrix construction
        trans_matrix = torch.eye(pc_size[1] + 1, dtype=torch.float32).unsqueeze(0)\
            .repeat(pc_size[0],1,1).to(self.device)  # batch size*(num_support+num_query), num_features, num_features

        x_shift = (torch.rand(pc_size[0]) - 0.5) * self.shift_range
        y_shift = (torch.rand(pc_size[0]) - 0.5) * self.shift_range
        z_shift = (torch.rand(pc_size[0]) - 0.5) * self.shift_range

        trans_matrix[:, 0, 3] = x_shift
        trans_matrix[:, 1, 3] = y_shift
        trans_matrix[:, 2, 3] = z_shift

        # Add an extra one for input pcs
        extra_one = torch.ones((pc_size[0], 1, pc_size[2])).to(self.device)
        pcs = torch.cat((pcs, extra_one), 1)

        # Translation
        pcs = torch.bmm(trans_matrix, pcs)
        return pcs[:, :-1, :]


class Scaling:
    def __init__(self, args):
        self.max_scale = args.max_scale
        self.min_scale = args.min_scale
        self.device = args.device

    def __call__(self, pcs):
        pc_size = pcs.size()  # batch size*(num_support+num_query), num_features, num_points

        scale_vector = torch.rand(pc_size[0])*(self.max_scale-self.min_scale)+self.min_scale
        scale_vector = scale_vector.to(self.device)
        scale_vector = scale_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, pc_size[1], pc_size[2])

        pcs = scale_vector*pcs
        return pcs


class Perturbation:
    def __init__(self, args):
        self.sigma = args.sigma
        self.clip = args.clip
        self.device = args.device

    def __call__(self, pcs):
        pc_size = pcs.size()  # batch size*(num_support+num_query), num_features, num_points

        clip_max = self.clip
        clip_min = -1 * self.clip

        jitter = torch.randn(pc_size)*self.sigma
        jitter = jitter.to(self.device)
        jitter = (jitter > clip_min)*jitter  # clip at clip_min
        jitter = (jitter < clip_max)*jitter  # clip at clip_max

        pcs += jitter
        return pcs


class Rotation:
    def __init__(self, args):
        self.device = args.device
        self.x_range = args.x_range
        self.y_range = args.y_range
        self.z_range = args.z_range

    def __call__(self, pcs):
        pc_size = pcs.size()  # num_tasks*num_samples, num_points, num_features

        pcs_center = torch.mean(pcs, -1) 
        # Rotation matrix construction
        # x rotation
        x_rot_matrix = torch.eye(pc_size[1]).unsqueeze(0).repeat(pc_size[0],1,1).to(self.device)
        x_theta = torch.rand(pc_size[0]) * self.x_range

        x_rot_matrix[:, 1, 1] = torch.cos(x_theta)
        x_rot_matrix[:, 1, 2] = torch.sin(x_theta)
        x_rot_matrix[:, 2, 1] = -torch.sin(x_theta)
        x_rot_matrix[:, 2, 2] = torch.cos(x_theta)

        # y rotation
        y_rot_matrix = torch.eye(pc_size[1]).unsqueeze(0).repeat(pc_size[0],1,1).to(self.device)
        y_theta = torch.rand(pc_size[0]) * self.y_range

        y_rot_matrix[:, 0, 0] = torch.cos(y_theta)
        y_rot_matrix[:, 0, 2] = -torch.sin(y_theta)
        y_rot_matrix[:, 2, 0] = torch.sin(y_theta)
        y_rot_matrix[:, 2, 2] = torch.cos(y_theta)

        # z rotation
        z_rot_matrix = torch.eye(pc_size[1]).unsqueeze(0).repeat(pc_size[0],1,1).to(self.device)
        z_theta = torch.rand(pc_size[0]) * self.z_range

        z_rot_matrix[:, 0, 0] = torch.cos(z_theta)
        z_rot_matrix[:, 0, 1] = torch.sin(z_theta)
        z_rot_matrix[:, 1, 0] = -torch.sin(z_theta)
        z_rot_matrix[:, 1, 1] = torch.cos(z_theta)

        # Final rotation
        rot_matrix = torch.bmm(torch.bmm(x_rot_matrix, y_rot_matrix), z_rot_matrix)

        pcs = torch.bmm(rot_matrix, pcs)
        pcs_new_center = torch.mean(pcs, -1)
        pcs += (pcs_new_center-pcs_center).unsqueeze(-1).repeat(1,1,pc_size[-1])
        # print('theta:',z_theta)
        # print('matrix:',(z_rot_matrix))
        # print('pcs0:',pcs[0,:,0])
        # print('newpcs0:',new_pcs[0,:,0])
        # print('pcs1:',pcs[0,:,1])
        # print('newpcs1:',new_pcs[0,:,1])

        return pcs


if __name__ == '__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt

