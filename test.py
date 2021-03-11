import numpy as np


def h(pcs, angle_range=None):
    pc_size = pcs.shape  # batch size, (num_support+num_query), num_points, features

    # Rotation matrix construction
    # x rotation
    x_rot_matrix = np.eye(pc_size[3])
    if angle_range is None:
        x_theta = np.random.rand(1) * 2 * np.pi
    else:
        x_theta = np.random.rand(1) * 2 * angle_range

    x_rot_matrix[1, 1] = np.cos(x_theta)
    x_rot_matrix[1, 2] = -np.sin(x_theta)
    x_rot_matrix[2, 1] = np.sin(x_theta)
    x_rot_matrix[2, 2] = np.cos(x_theta)

    # y rotation
    y_rot_matrix = np.eye(pc_size[3])
    y_theta = np.random.rand(1) * 2 * angle_range

    y_rot_matrix[0, 0] = np.cos(y_theta)
    y_rot_matrix[0, 2] = np.sin(y_theta)
    y_rot_matrix[2, 0] = -np.sin(y_theta)
    y_rot_matrix[2, 2] = np.cos(y_theta)

    # z rotation
    z_rot_matrix = np.eye(pc_size[3])
    z_theta = np.random.rand(1) * 2 * angle_range

    z_rot_matrix[0, 0] = np.cos(z_theta)
    z_rot_matrix[0, 1] = -np.sin(z_theta)
    z_rot_matrix[1, 0] = np.sin(z_theta)
    z_rot_matrix[1, 1] = np.cos(z_theta)

    # Final rotation
    rot_matrix = np.dot(np.dot(x_rot_matrix, y_rot_matrix), z_rot_matrix)

    pcs = pcs.reshape(pc_size[0] * pc_size[1], pc_size[2], pc_size[3]).transpose(0, 2, 1)
    for i in range(pc_size[0] * pc_size[1]):
        pcs[i, :, :] = np.dot(rot_matrix, pcs[i, :, :])
    pcs = pcs.transpose(0, 2, 1).reshape(pc_size[0], pc_size[1], pc_size[2], pc_size[3])
    return pcs
import random
import torch
#pcs = np.ones((4,4,5,3))
print(torch.rand(1))