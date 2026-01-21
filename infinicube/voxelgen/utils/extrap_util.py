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

from pathlib import Path

import fvdb
import torch
from loguru import logger
from termcolor import colored

from infinicube import get_sample
from infinicube.data_process.waymo_utils import keep_car_only_in_object_info
from infinicube.utils.bbox_utils import build_scene_bounding_boxes_from_object_info

# ============================================================================
# Map Data Loading
# ============================================================================


def get_wds_data(
    clip,
    key_frame_interval=76.8,
    webdataset_root="data/",
    ego_trajectory_key="pose.front.npy",
):
    """
    Load data from webdataset for a given clip

    Args:
        clip: str, clip name
        key_frame_interval: float or None
            if None, return all the frames
            if float, return key frames with the interval of key_frame_interval
        webdataset_root: str, path to webdataset root
        ego_trajectory_key: str, key of the ego trajectory

    Returns:
        wds_data: dict with following keys:
            'road_edge': torch.Tensor in original (waymo) coordinates
            'road_line': torch.Tensor in original (waymo) coordinates
            'dense_road_surface': torch.Tensor in original (waymo) coordinates
            'boxes_3d': torch.Tensor, shape (N, 8, 3), 3D bounding boxes in original (waymo) coordinates
            'ego_trajectory': torch.Tensor, shape (K, 4, 4), ego trajectory in world coordinates, opencv convention
    """
    webdataset_root = Path(webdataset_root).resolve()

    if key_frame_interval is not None:
        logger.info(f"Key frame interval: {key_frame_interval}")
    else:
        logger.info("Return all frames")

    # road edge
    road_edge = get_sample(f"{webdataset_root}/3d_road_edge_voxelsize_025/{clip}.tar")[
        "road_edge.npy"
    ]
    road_edge = torch.tensor(road_edge)

    # road line
    road_line = get_sample(f"{webdataset_root}/3d_road_line_voxelsize_025/{clip}.tar")[
        "road_line.npy"
    ]
    road_line = torch.tensor(road_line)

    # road surface
    road_surface = get_sample(
        f"{webdataset_root}/3d_road_surface_voxelsize_04/{clip}.tar"
    )["road_surface.npy"]
    road_surface = torch.tensor(road_surface)

    # bbox
    static_object_info = get_sample(f"{webdataset_root}/static_object_info/{clip}.tar")[
        "000000.static_object_info.json"
    ]
    static_object_info = keep_car_only_in_object_info(static_object_info)
    boxes_3d_corners = build_scene_bounding_boxes_from_object_info(static_object_info)

    # pose
    pose = get_sample(f"{webdataset_root}/pose/{clip}.tar")
    pose = {k: torch.tensor(v) for k, v in pose.items() if ego_trajectory_key in k}
    # sort according to the k, and concatenate the values together
    ego_trajectory = torch.stack([v for k, v in sorted(pose.items())], axis=0)

    if key_frame_interval is not None:
        key_frame_indices = get_key_frames_indices(ego_trajectory, key_frame_interval)
        ego_trajectory = ego_trajectory[key_frame_indices]

    wds_data = {
        "road_edge": road_edge,
        "road_line": road_line,
        "road_surface": road_surface,
        "boxes_3d": boxes_3d_corners,
        "ego_trajectory": ego_trajectory,
    }

    return wds_data


def get_key_frames_indices(ego_trajectory_world, key_frame_distance=25):
    """
    Select key frames based on distance traveled

    Args:
        ego_trajectory_world: torch.Tensor
            shape (K, 4, 4), ego trajectory in world coordinates, opencv convention
        key_frame_distance: float, measured in meters
            distance threshold for selecting key frames

    Returns:
        key_frames_indices: list
            list of key frames indices
    """
    indices = list(range(ego_trajectory_world.shape[0]))

    distance = torch.norm(
        ego_trajectory_world[1:, :3, 3] - ego_trajectory_world[:-1, :3, 3], dim=1
    )  # shape (K-1, )
    accumulated_distance = torch.cumsum(distance, dim=0)  # shape (K-1, )
    accumulated_distance = torch.cat(
        [torch.tensor([0.0]).to(accumulated_distance), accumulated_distance], axis=0
    )  # shape (K, )

    # select key frames, e.g. every 30 meters
    key_frames_indices = []
    for i in indices:
        if accumulated_distance[i] < 0:
            continue
        if accumulated_distance[i] >= 0:
            key_frames_indices.append(i)
            accumulated_distance -= key_frame_distance

    # add the last frame
    if indices[-1] not in key_frames_indices:
        key_frames_indices.append(indices[-1])

    return key_frames_indices


# ============================================================================
# Coordinate Transformation
# ============================================================================


def get_relative_transforms(poses):
    """
    Given camera poses, compute the relative transforms based on the first pose.
    Note that opencv / FLU convention have different output!! be careful

    Args:
        poses: torch.Tensor, shape [N, 4, 4]

    Returns:
        relative poses: torch.Tensor, shape [N, 4, 4]
    """
    # cam2cam0
    relative_poses = torch.matmul(
        torch.inverse(poses[0]), poses
    )  # world2cam0 * cam2world

    return relative_poses


def transform_points(points, transformation_matrix):
    """
    Transform points using transformation matrix.

    Args:
        points: torch.Tensor, shape [N, 3]
        transformation_matrix: torch.Tensor, shape [4, 4]

    Returns:
        transformed points: torch.Tensor, shape [N, 3]
    """
    points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    points = torch.matmul(transformation_matrix, points.t()).t()[:, :3]

    return points


def transform_poses(poses, transformation_matrix):
    """
    Transform poses using transformation matrix.

    Args:
        poses: torch.Tensor, shape [N, 4, 4]
            in coordinate A
        transformation_matrix: torch.Tensor, shape [4, 4]
            transforming points from coordinate A to coordinate B

    Returns:
        transformed poses: torch.Tensor, shape [N, 4, 4]
    """
    T_world_A = poses
    T_B_A = transformation_matrix
    T_A_B = torch.inverse(T_B_A)
    T_world_B = torch.matmul(T_world_A, T_A_B)

    return T_world_B


def transform_grid(grid, transformation_matrix):
    """
    Transform grid using transformation matrix.

    Args:
        grid: fvdb.GridBatch
        transformation_matrix: torch.Tensor, shape [4, 4]

    Returns:
        transformed grid: fvdb.GridBatch
    """
    assert grid.grid_count == 1, "Only support single grid transformation"
    xyz = grid.grid_to_world(grid.ijk.float()).jdata  # [N, 3]
    xyz = transform_points(xyz, transformation_matrix)
    new_grid = fvdb.gridbatch_from_points(
        xyz, voxel_sizes=grid.voxel_sizes, origins=grid.voxel_sizes / 2
    )

    return new_grid


def transform_grid_and_semantic(grid, semantic, transformation_matrix, subdivide=False):
    """
    Transform grid and semantic using transformation matrix.

    Args:
        grid: fvdb.GridBatch
        semantic: torch.Tensor, shape [N, ]
        transformation_matrix: torch.Tensor, shape [4, 4]
        subdivide: bool
            whether to subdivide the grid before transformation,
            it will fix the issue of holes after transformation

    Return:
        new_grid: fvdb.GridBatch
        new_semantic: torch.Tensor, shape [N', ]
    """
    from infinicube.voxelgen.utils.color_util import semantic_from_points

    assert grid.grid_count == 1, "Only support single grid transformation"

    xyz = grid.grid_to_world(grid.ijk.float()).jdata

    if subdivide:
        subdivide_grid = grid.subdivided_grid(2)  # inplace
        xyz_fine = subdivide_grid.grid_to_world(
            subdivide_grid.ijk.float()
        ).jdata  # [N, 3]
    else:
        xyz_fine = xyz

    xyz = transform_points(xyz.double(), transformation_matrix.double()).float()
    xyz_fine = transform_points(
        xyz_fine.double(), transformation_matrix.double()
    ).float()

    new_grid = fvdb.gridbatch_from_points(
        xyz_fine, voxel_sizes=grid.voxel_sizes, origins=grid.voxel_sizes / 2
    )
    new_grid_xyz = new_grid.grid_to_world(new_grid.ijk.float()).jdata  # [N, 3]
    new_grid_semantic = semantic_from_points(new_grid_xyz, xyz, semantic)

    return new_grid, new_grid_semantic


# ============================================================================
# Trajectory Generation
# ============================================================================


def generate_camera_poses_from_batch_trajectory(
    target_pose_num, pose_distance_interval, batch_trajectory
):
    """
    Given initial camera trajectory from batch data (e.g. K frames),
    we extrapolate the trajectory to N frames, while keeping the distance between poses constant.

    A very simple way is get the direction from batch_trajectory, and then generate new poses based on the direction.

    Args:
        target_pose_num: int
            N, target number of poses to generate
        pose_distance_interval: float
            distance between poses in the generated trajectory
        batch_trajectory: torch.Tensor
            some initial camera poses from batch data, shape [K, 4, 4], FLU convention

    Returns:
        camera poses: torch.Tensor
            shape [N, 4, 4], FLU convention
    """
    # get the direction from batch_trajectory
    K = batch_trajectory.shape[0]

    if target_pose_num <= K:
        print(
            colored(
                f"target_pose_num ({target_pose_num}) <= K ({K}), return the original trajectory",
                "green",
                attrs=["bold"],
            )
        )
        return batch_trajectory[:target_pose_num]

    direction = batch_trajectory[-1, :3, 0]  # front direction of the last pose
    direction = direction / torch.norm(direction)

    # generate new poses based on the direction
    new_poses = []

    print(
        colored(
            f"Generating remaining {target_pose_num - K} poses", "green", attrs=["bold"]
        )
    )
    for i in range(target_pose_num - K):
        new_pose = batch_trajectory[-1].clone()
        new_pose[:3, 3] = new_pose[:3, 3] + direction * pose_distance_interval * (i + 1)
        new_poses.append(new_pose)

    if len(new_poses) == 0:
        return batch_trajectory
    else:
        new_poses = torch.stack(new_poses, dim=0)
        return torch.cat([batch_trajectory, new_poses], dim=0)
