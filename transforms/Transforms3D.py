import numpy as np
import torch


class List2Array:
    def __init__(self):
        None

    def __call__(self, pcs):
        num_samples = len(pcs)
        if len(pcs) > 0:
            num_tasks, num_points, num_features = pcs[0].shape

        array_pcs = np.zeros((num_samples*num_tasks, num_points, num_features),dtype='float32')
        for i, pc in enumerate(pcs):
            array_pcs[i*num_tasks:(i+1)*num_tasks, :, :] = pc
        array_pcs = array_pcs.transpose(0, 2, 1)
        return array_pcs


class ToTensor:
    def __init__(self, args):
        self.device = args.device

    def __call__(self, pcs):
        return torch.from_numpy(pcs).to(self.device)


class Translation:
    def __init__(self, args):
        self.shift_range = args.shift_range
        self.device = args.device

    def __call__(self, pcs):
        pc_size = pcs.size()  # batch size*(num_support+num_query), features, num_points

        # Translation matrix construction
        trans_matrix = torch.eye(pc_size[1] + 1, dtype=torch.float32).to(self.device)
        x_shift = (torch.rand(1) - 0.5) * self.shift_range
        y_shift = (torch.rand(1) - 0.5) * self.shift_range
        z_shift = (torch.rand(1) - 0.5) * self.shift_range

        trans_matrix[0, 3] = x_shift
        trans_matrix[1, 3] = y_shift
        trans_matrix[2, 3] = z_shift

        trans_matrix = trans_matrix.unsqueeze(0).repeat(pc_size[0], 1, 1)

        # Add an extra one for input pcs
        extra_one = torch.ones((pc_size[0], 1, pc_size[2])).to(self.device)
        pcs = torch.cat((pcs, extra_one), 1)

        # Translation
        pcs = torch.bmm(trans_matrix, pcs)
        return pcs[:, :-1, :]


class Rotation:
    def __init__(self, args):
        self.angle_range = args.angle_range
        self.device = args.device
        if self.angle_range < 0 or self.angle_range > 2*np.pi:
            raise NotImplementedError('Angle range should be between 0 and 2*pi')


    def __call__(self, pcs):
        pc_size = pcs.size()  # num_tasks*num_samples, num_points, num_features

        # Rotation matrix construction
        # x rotation
        x_rot_matrix = torch.eye(pc_size[1]).to(self.device)
        x_theta = torch.rand(1) * self.angle_range

        x_rot_matrix[1, 1] = torch.cos(x_theta)
        x_rot_matrix[1, 2] = torch.sin(x_theta)
        x_rot_matrix[2, 1] = -torch.sin(x_theta)
        x_rot_matrix[2, 2] = torch.cos(x_theta)

        # y rotation
        y_rot_matrix = torch.eye(pc_size[1]).to(self.device)
        y_theta = torch.rand(1) * self.angle_range

        y_rot_matrix[0, 0] = torch.cos(y_theta)
        y_rot_matrix[0, 2] = -torch.sin(y_theta)
        y_rot_matrix[2, 0] = torch.sin(y_theta)
        y_rot_matrix[2, 2] = torch.cos(y_theta)

        # z rotation
        z_rot_matrix = torch.eye(pc_size[1]).to(self.device)
        z_theta = torch.rand(1) * self.angle_range

        z_rot_matrix[0, 0] = torch.cos(z_theta)
        z_rot_matrix[0, 1] = torch.sin(z_theta)
        z_rot_matrix[1, 0] = -torch.sin(z_theta)
        z_rot_matrix[1, 1] = torch.cos(z_theta)

        # Final rotation
        rot_matrix = torch.mm(torch.mm(x_rot_matrix, y_rot_matrix), z_rot_matrix)
        rot_matrix = rot_matrix.unsqueeze(0).repeat(pc_size[0], 1, 1)

        new_pcs = torch.bmm(rot_matrix, pcs)

        # print('theta:',z_theta)
        # print('matrix:',(z_rot_matrix))
        # print('pcs0:',pcs[0,:,0])
        # print('newpcs0:',new_pcs[0,:,0])
        # print('pcs1:',pcs[0,:,1])
        # print('newpcs1:',new_pcs[0,:,1])

        return new_pcs, pcs


if __name__ == '__main__':
    from torchvision import transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        List2Array(),
        ToTensor(),
        Translation(),
        Rotation()])
    l = [np.zeros((4, 1024, 3))]*5
    l = transform(l)
    points = l[0]
    x = points[0,:]
    print(points)
    y = points[1,:]
    z = points[2,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,  # x
               y,  # y
               z,  # z
               c=z,  # height data for color
               cmap='Blues',
               marker="o")
    plt.show()
