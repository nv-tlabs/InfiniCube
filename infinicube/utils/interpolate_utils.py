# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def inter_two_poses_matrix(prev_pose, next_pose, t):
    """
    new pose = (1 - t) * prev_pose + t * next_pose.
    - linear interpolation for translation
    - slerp interpolation for rotation

    Args:
        prev_pose: np.ndarray, shape (4, 4), dtype=np.float32, previous pose
        next_pose: np.ndarray, shape (4, 4), dtype=np.float32, next pose
        t: float, interpolation factor

    Returns:
        np.ndarray, shape (4, 4), dtype=np.float32, interpolated pose

    Note:
        if input is list, also return list.
    """
    input_is_list = isinstance(prev_pose, list)
    prev_pose = np.array(prev_pose)
    next_pose = np.array(next_pose)

    prev_translation = prev_pose[:3, 3]
    next_translation = next_pose[:3, 3]
    translation = (1 - t) * prev_translation + t * next_translation

    prev_rotation = R.from_matrix(prev_pose[:3, :3])
    next_rotation = R.from_matrix(next_pose[:3, :3])

    times = [0, 1]
    rotations = R.from_quat([prev_rotation.as_quat(), next_rotation.as_quat()])
    rotation = Slerp(times, rotations)(t)

    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation.as_matrix()
    new_pose[:3, 3] = translation

    if input_is_list:
        return new_pose.tolist()
    else:
        return new_pose


def inter_two_poses(wxyz_a, position_a, wxyz_b, position_b, alpha):
    """
    interpolate two poses between two key poses.

    Args:
        wxyz_a (numpy.ndarray): the quaternion of the first pose, shape (4,).
        position_a (numpy.ndarray): the position of the first pose, shape (3,).
        wxyz_b (numpy.ndarray): the quaternion of the second pose, shape (4,).
        position_b (numpy.ndarray): the position of the second pose, shape (3,).
        alpha (float): the interpolation factor, range [0, 1].

    Returns:
        tuple: the interpolated quaternion and position.
            - interp_wxyz (numpy.ndarray): the interpolated quaternion, shape (4,).
            - interp_position (numpy.ndarray): the interpolated position, shape (3,).
    """
    key_rots = R.from_quat(np.stack([wxyz_a, wxyz_b], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp(1.0 - alpha)
    interp_wxyz = interp_rot.as_quat()
    interp_position = position_a * alpha + position_b * (1.0 - alpha)

    return interp_wxyz, interp_position


def inter_poses(key_wxyz, key_position, n_out_poses, sigma=1.0):
    """
    interpolate the quaternions and positions of the key poses, generate a series of smooth transition poses.

    Args:
        key_wxyz (numpy.ndarray): the quaternions of the key poses, shape (n_key_poses, 4).
        key_position (numpy.ndarray): the positions of the key poses, shape (n_key_poses, 3).
        n_out_poses (int): the number of output poses to generate.
        sigma (float, optional): the parameter to control the smoothness of the interpolation, default is 1.

    Returns:
        tuple: the interpolated quaternions and positions.
            - out_wxyz (numpy.ndarray): the interpolated quaternions, shape (n_out_poses, 4).
            - out_position (numpy.ndarray): the interpolated positions, shape (n_out_poses, 3).
    """
    n_key_poses = len(key_wxyz)
    out_wxyz = []
    out_position = []
    for i in range(n_out_poses):
        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-((np.abs(i / n_out_poses * n_key_poses - w) / sigma) ** 2))
        w = w + 1e-6
        w /= np.sum(w)
        cur_wxyz = key_wxyz[0]
        cur_position = key_position[0]
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_wxyz, cur_position = inter_two_poses(
                cur_wxyz,
                cur_position,
                key_wxyz[j + 1],
                key_position[j + 1],
                cur_w / (cur_w + w[j + 1]),
            )
            cur_w += w[j + 1]
        out_wxyz.append(cur_wxyz)
        out_position.append(cur_position)

    return np.stack(out_wxyz), np.stack(out_position)


def inter_two_poses_uniform(key_wxyz, key_position, n_out_poses):
    """
    interpolate two poses uniformly in L2 space of the position.
    i.e. the moving speed is constant.

    Args:
        key_wxyz (numpy.ndarray): key poses' quaternion, shape (2, 4).
        key_position (numpy.ndarray): key poses' position, shape (2, 3).
        n_out_poses (int): number of output poses.

    Returns:
        tuple: interpolated quaternion and position.
            - out_wxyz (numpy.ndarray): interpolated quaternion, shape (n_out_poses, 4).
            - out_position (numpy.ndarray): interpolated position, shape (n_out_poses, 3).
    """
    assert key_wxyz.shape == (2, 4), key_wxyz.shape
    assert key_position.shape == (2, 3), key_position.shape
    key_rots = R.from_quat(key_wxyz)
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp(np.linspace(0, 1, n_out_poses))
    interp_wxyz = interp_rot.as_quat()
    interp_position = np.linspace(key_position[0], key_position[1], n_out_poses)

    return interp_wxyz, interp_position


def interpolate_polyline_to_points(polyline, segment_interval=0.025):
    """
    polyline:
        numpy.ndarray, shape (N, 3) or list of points

    Returns:
        points: numpy array, shape (interpolate_num*N, 3)
    """
    polyline = np.array(polyline)
    # calculate the distance between adjacent points
    diffs = np.diff(polyline, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # calculate the number of points to interpolate for each segment
    n_points = np.maximum(np.ceil(distances / segment_interval).astype(int), 2)

    # calculate the total number of points (excluding the last point)
    total_points = np.sum(n_points - 1)

    # pre-allocate the result array
    result = np.zeros((total_points + 1, 3))

    # calculate the starting index for each segment
    start_indices = np.zeros(len(n_points), dtype=int)
    start_indices[1:] = np.cumsum(n_points[:-1] - 1)

    # interpolate each segment
    for i in range(len(n_points)):
        if i == 0:
            result[0] = polyline[0]

        # calculate the points to interpolate for the current segment
        t = np.linspace(0, 1, n_points[i])[:-1]
        segment_points = polyline[i][None, :] + t[:, None] * diffs[i]

        # add the interpolated points to the result array
        end_idx = start_indices[i] + len(segment_points)
        result[start_indices[i] : end_idx] = segment_points

    # add the last point
    result[-1] = polyline[-1]

    return result


def quaternion_mean(quaternions):
    quaternions = np.array(quaternions)

    for i in range(len(quaternions)):
        if quaternions[i, 0] < 0:  # if w is negative, flip
            quaternions[i] = -quaternions[i]

    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion = mean_quaternion / np.linalg.norm(mean_quaternion)  # normalize
    return mean_quaternion


def rotation_matrix_mean(rotation_matrices):
    rotations = [R.from_matrix(R_matrix) for R_matrix in rotation_matrices]

    quaternions = [rotation.as_quat() for rotation in rotations]

    mean_quaternion = quaternion_mean(quaternions)

    mean_rotation = R.from_quat(mean_quaternion)

    mean_rotation_matrix = mean_rotation.as_matrix()
    return mean_rotation_matrix


if __name__ == "__main__":
    key_wxyz = np.array([[0, 0, 0, 1], [0, 0, 0.707, 0.707]])
    key_position = np.array([[0, 0, 0], [1, 1, 1]])
    n_out_poses = 10
    out_wxyz, out_position = inter_two_poses_uniform(
        key_wxyz, key_position, n_out_poses
    )
    print(out_wxyz)
    print(out_position)
