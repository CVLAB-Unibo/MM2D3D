import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def augment_and_scale_3d_torch(
    points,
    scale,
    full_scale,
    noisy_rot=0.0,
    flip_x=0.0,
    flip_y=0.0,
    rot_z=0.0,
    rot_y=0.0,
    transl=False,
):
    """
    3D point cloud augmentation and scaling from points (in meters) to voxels
    :param points: 3D points in meters
    :param scale: voxel scale in 1 / m, e.g. 20 corresponds to 5cm voxels
    :param full_scale: size of the receptive field of SparseConvNet
    :param noisy_rot: scale of random noise added to all elements of a rotation matrix
    :param flip_x: probability of flipping the x-axis (left-right in nuScenes LiDAR coordinate system)
    :param flip_y: probability of flipping the y-axis (left-right in Kitti LiDAR coordinate system)
    :param rot_z: angle in rad around the z-axis (up-axis)
    :param transl: True or False, random translation inside the receptive field of the SCN, defined by full_scale
    :return coords: the coordinates that are given as input to SparseConvNet
    """
    if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0 or rot_y > 0:
        rot_matrix = torch.eye(3, dtype=torch.float32)
        if noisy_rot > 0:
            # add noise to rotation matrix
            rot_matrix = R.from_rotvec(R.random().as_euler("xyz") * noisy_rot).as_matrix()
        if flip_x > 0:
            # flip x axis: multiply element at (0, 0) with 1 or -1
            rot_matrix[0][0] *= torch.randint(2, ()) * 2 - 1
        if flip_y > 0:
            # flip y axis: multiply element at (1, 1) with 1 or -1
            rot_matrix[1][1] *= torch.randint(2, ()) * 2 - 1
        if rot_z > 0:
            # rotate around z-axis (up-axis)
            theta = torch.rand(()) * rot_z
            z_rot_matrix = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta), torch.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            )
            rot_matrix = rot_matrix.matmul(z_rot_matrix)
        if rot_y > 0:
            # rotate around z-axis (up-axis)
            theta = torch.rand(()) * rot_y
            y_rot_matrix = torch.tensor(
                [
                    [torch.cos(theta), 0, torch.sin(theta)],
                    [0, 1, 0],
                    [-torch.sin(theta), 0, torch.cos(theta)],
                ],
                dtype=torch.float32,
            )
            rot_matrix = rot_matrix.matmul(y_rot_matrix)
        points = torch.matmul(
            points, torch.tensor(rot_matrix, device=points.device, dtype=points.dtype)
        )

    # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
    coords = points * scale
    # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
    coords -= coords.min(0)[0]

    if transl:
        # random translation inside receptive field of SCN
        offset = torch.clip(
            full_scale - coords.max(0)[0] - 0.001, min=0, max=None
        ) * torch.rand((3), device=coords.device)
        coords += offset

    return coords


def augment_and_scale_3d(
    points,
    scale,
    full_scale,
    noisy_rot=0.0,
    flip_x=0.0,
    flip_y=0.0,
    rot_z=0.0,
    rot_y=0.0,
    transl=False,
):
    """
    3D point cloud augmentation and scaling from points (in meters) to voxels
    :param points: 3D points in meters
    :param scale: voxel scale in 1 / m, e.g. 20 corresponds to 5cm voxels
    :param full_scale: size of the receptive field of SparseConvNet
    :param noisy_rot: scale of random noise added to all elements of a rotation matrix
    :param flip_x: probability of flipping the x-axis (left-right in nuScenes LiDAR coordinate system)
    :param flip_y: probability of flipping the y-axis (left-right in Kitti LiDAR coordinate system)
    :param rot_z: angle in rad around the z-axis (up-axis)
    :param transl: True or False, random translation inside the receptive field of the SCN, defined by full_scale
    :return coords: the coordinates that are given as input to SparseConvNet
    """
    rot_matrix = np.eye(3, dtype=np.float32)
    if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0 or rot_y > 0:
        if noisy_rot > 0:
            # add noise to rotation matrix
            rot_matrix += np.random.randn(3, 3) * noisy_rot

        if flip_x > 0:
            # flip x axis: multiply element at (0, 0) with 1 or -1
            rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        if flip_y > 0:
            # flip y axis: multiply element at (1, 1) with 1 or -1
            rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        if rot_z > 0:
            # rotate around z-axis (up-axis)
            theta = np.random.rand() * rot_z
            z_rot_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            rot_matrix = rot_matrix.dot(z_rot_matrix)
        if rot_y > 0:
            # rotate around z-axis (up-axis)
            theta = np.random.rand() * rot_y
            y_rot_matrix = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ],
                dtype=np.float32,
            )
            rot_matrix = rot_matrix.dot(y_rot_matrix)
        points = points.dot(rot_matrix)

    # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
    coords = points * scale
    # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
    min_value = coords.min(0)
    coords -= min_value
    offset = np.zeros(3)

    if transl:
        # random translation inside receptive field of SCN
        offset = np.clip(
            full_scale - coords.max(0) - 0.001, a_min=0, a_max=None
        ) * np.random.rand(3)
        coords += offset

    return coords, min_value, offset, rot_matrix
