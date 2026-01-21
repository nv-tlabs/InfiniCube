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
import torch
from pycg.isometry import Isometry


def to_opengl(ray_d):
    """
    transform the ray direction vector into OpenGL convention (-z is FRONT, +x is RIGHT, +y is UP)
    FLU to RUB


            z                        y
            |  x (front)             |
            |/                       |
    y <-----o    ===========>>       o----> x
                                    /
                                 z /
                                (back)
    Args:
        ray_d : torch.tensor
            shape [*, 3]
    """
    return torch.cat([-ray_d[..., 1:2], ray_d[..., 2:3], -ray_d[..., 0:1]], dim=-1)


def from_opengl(ray_d):
    """
    transform the ray direction vector from OpenGL convention to our convention (+y is front, +x is right, +z is up)
    FLU to RFU


            z                        y
            |  x (front)             |
            |/                       |
    y <-----o    <<===========       o----> x
                                    /
                                 z /
                                (back)
    Args:
        ray_d : torch.tensor
            shape [*, 3]
    """
    return torch.cat([-ray_d[..., 2:3], -ray_d[..., 0:1], ray_d[..., 1:2]], dim=-1)


def pose_from_yaw_elevation(yaw: float, elevation: float, distance: float) -> Isometry:
    init_pose = Isometry(t=[0.0, 0.0, distance]) @ Isometry.from_axis_angle("+X", 180.0)
    return (
        Isometry.from_axis_angle("Y", 180 + yaw)
        @ Isometry.from_axis_angle("X", -elevation)
        @ init_pose
    )


def create_360_degree_pose(n_view=12, evalation=30, distance=1.2 * 1.28):
    pose_list = []
    for i in range(n_view):
        yaw = i * 360 / n_view
        pose_list.append(pose_from_yaw_elevation(yaw, evalation, distance).matrix)
    return torch.from_numpy(np.stack(pose_list)).float()


def create_plucker_coords_torch(pose_batch, intrinsic_batch):
    """
    Args:
        pose_batch: torch.Tensor
            shape [B, 4, 4]
        intrinsic_batch: torch.Tensor
            shape [B, 6]

    Returns:
        coords: torch.Tensor
            shape [B, H, W, 6]
    """
    B = pose_batch.shape[0]
    camera_origin = pose_batch[:, :3, 3]
    fx, fy, cx, cy, w, h = intrinsic_batch.unbind(1)
    w, h = int(w[0]), int(h[0])
    ii, jj = torch.meshgrid(
        torch.arange(w), torch.arange(h), indexing="xy"
    )  # attention, indexing is 'xy'
    ii, jj = ii.to(pose_batch), jj.to(pose_batch)
    uu, vv = (
        (ii - cx[:, None, None]) / fx[:, None, None],
        (jj - cy[:, None, None]) / fy[:, None, None],
    )
    local_xyz = torch.stack(
        [uu, vv, torch.ones_like(uu).to(uu)], dim=-1
    )  # (B, H, W, 3)

    local_xyz = torch.cat(
        [local_xyz, torch.ones((B, int(h), int(w), 1)).to(local_xyz)], axis=-1
    )
    pixel_xyz = torch.einsum("bij, bhwj->bhwi", pose_batch, local_xyz)[
        :, :, :, :3
    ]  # (B, H, W, 3) # ! fix error

    d = pixel_xyz - camera_origin[:, None, None, :]
    d = d / torch.norm(d, dim=-1, keepdim=True)

    # cross product between d and camera_origin
    p = torch.cross(d, camera_origin[:, None, None, :], dim=-1)
    coords = torch.cat([d, p], dim=-1)
    return coords


def camera_intrinsic_list_to_matrix(intrinsic_list, normalize_pixel=False):
    """
    Args:
        intrinsic_list: [..., 6]
            [fx, fy, cx, cy, w, h]
    """
    if isinstance(intrinsic_list, list):
        intrinsic_list = torch.stack(intrinsic_list)
    fx, fy, cx, cy, w, h = intrinsic_list.unbind(-1)
    intrinsic_matrix = torch.zeros(
        intrinsic_list.shape[:-1] + (3, 3), device=intrinsic_list.device
    )

    intrinsic_matrix[..., 0, 0] = fx
    intrinsic_matrix[..., 1, 1] = fy
    intrinsic_matrix[..., 0, 2] = cx
    intrinsic_matrix[..., 1, 2] = cy
    intrinsic_matrix[..., 2, 2] = 1

    if normalize_pixel:
        intrinsic_matrix[..., 0, :] /= w[..., None]
        intrinsic_matrix[..., 1, :] /= h[..., None]

    return intrinsic_matrix


def create_rays_from_intrinsic_torch(pose_matric, intrinsic):
    """
    Returns rays in the world coordinate system

    Args:
        pose_matric: (4, 4)
        intrinsic: (6, ), [fx, fy, cx, cy, w, h]
    Returns:
        camera_origin: (3, )
        d: (H, W, 3)
    """
    camera_origin = pose_matric[:3, 3]
    fx, fy, cx, cy, w, h = intrinsic
    ii, jj = torch.meshgrid(
        torch.arange(w), torch.arange(h), indexing="xy"
    )  # attention, indexing is 'xy'
    ii, jj = ii.to(pose_matric), jj.to(pose_matric)
    uu, vv = (ii - cx) / fx, (jj - cy) / fy
    local_xyz = torch.stack([uu, vv, torch.ones_like(uu).to(uu)], dim=-1)  # (H, W, 3)

    local_xyz = torch.cat(
        [local_xyz, torch.ones((int(h), int(w), 1)).to(local_xyz)], axis=-1
    )
    pixel_xyz = torch.einsum("ij, hwj->hwi", pose_matric, local_xyz)[
        :, :, :3
    ]  # (H, W, 3) # ! fix error

    d = pixel_xyz - camera_origin
    # normalize the direction
    d = d / torch.norm(d, dim=-1, keepdim=True)

    return camera_origin, d


def create_rays_from_intrinsic_torch_batch(pose_matric, intrinsic):
    """
    Args:
        pose_matric: (B, 4, 4)
        intrinsic: (B, 6), [fx, fy, cx, cy, w, h]
    Returns:
        camera_origin: (B, 3)
        d: (B, H, W, 3)
    """
    camera_origin = pose_matric[:, :3, 3]  # (B, 3)
    fx, fy, cx, cy, w, h = intrinsic.unbind(1)  # [B,]
    w, h = int(w[0]), int(h[0])
    # attention, indexing is 'xy'
    ii, jj = torch.meshgrid(
        torch.arange(w).to(intrinsic.device),
        torch.arange(h).to(intrinsic.device),
        indexing="xy",
    )

    ii = ii[None].repeat(pose_matric.shape[0], 1, 1)  # (B, H, W)
    jj = jj[None].repeat(pose_matric.shape[0], 1, 1)  # (B, H, W)

    uu, vv = (
        (ii - cx[:, None, None]) / fx[:, None, None],
        (jj - cy[:, None, None]) / fy[:, None, None],
    )
    local_xyz = torch.stack(
        [uu, vv, torch.ones_like(uu, device=uu.device)], dim=-1
    )  # (B, H, W, 3)
    local_xyz = torch.cat(
        [local_xyz, torch.ones((local_xyz.shape[0], int(h), int(w), 1)).to(local_xyz)],
        axis=-1,
    )
    pixel_xyz = torch.einsum("bij, bhwj->bhwi", pose_matric, local_xyz)[
        :, :, :, :3
    ]  # (B, H, W, 3) # ! fix error

    d = pixel_xyz - camera_origin[:, None, None, :]  # (B, H, W, 3)
    # normalize the direction
    d = d / torch.norm(d, dim=-1, keepdim=True)  # (B, H, W, 3)

    return camera_origin, d


def rel_render_feature_to_real(batch_features, free_space, grid):
    _rel_xyz = batch_features[:, :, :3]
    _scaling = batch_features[:, :, 3:6]
    _rots = batch_features[:, :, 6:10]
    _opacities = batch_features[:, :, 10:11]
    _color = batch_features[:, :, 11:14]

    rel_pos = get_rel_pos(_rel_xyz, free_space, grid)

    base_pos = grid.grid_to_world(grid.ijk.float() - 0.5).jdata.unsqueeze(1) + rel_pos
    base_pos = base_pos.view(-1, 3)

    scaling = (torch.exp(_scaling) * grid.voxel_sizes[0, 0]).view(-1, 3)

    rotation = torch.nn.functional.normalize(_rots.view(-1, 4), dim=1)
    opacity = torch.sigmoid(_opacities.view(-1, 1))

    color = _color.view(-1, 3)

    return base_pos, scaling, rotation, opacity, color


def get_rel_pos(_rel_xyz, free_space, grid):
    """
    Args:
        _rel_xyz: torch.tensor
            shape [..., 3]
        free_space: str
            something like 'soft'
        grid: GridBatch
            fvdb grid

    Returns:
        rel_pos: torch.tensor
            shape [..., 3], the position within a voxel (compared with the voxel center)
    """
    if free_space == "hard":
        rel_pos = torch.sigmoid(_rel_xyz) * grid.voxel_sizes[0]
    elif free_space == "soft":
        # free space [-1, 2]
        rel_pos = (torch.sigmoid(_rel_xyz) * 3 - 1) * grid.voxel_sizes[0]
    elif free_space == "soft-2":
        # free space [-2, 3]
        rel_pos = (torch.sigmoid(_rel_xyz) * 5 - 2) * grid.voxel_sizes[0]
    elif free_space == "soft-3":
        # free space [-3, 4]
        rel_pos = (torch.sigmoid(_rel_xyz) * 7 - 3) * grid.voxel_sizes[0]
    elif free_space == "soft-4":
        # free space [-4, 5]
        rel_pos = (torch.sigmoid(_rel_xyz) * 9 - 4) * grid.voxel_sizes[0]
    elif free_space == "soft-5":
        # free space [-5, 6]
        rel_pos = (torch.sigmoid(_rel_xyz) * 11 - 5) * grid.voxel_sizes[0]
    elif free_space == "free-1":
        rel_pos = _rel_xyz * grid.voxel_sizes[0]
    elif free_space == "free-2":
        rel_pos = _rel_xyz
    elif free_space == "center":
        rel_pos = (torch.zeros_like(_rel_xyz) + 0.5) * grid.voxel_sizes[0]
    # elif free_space == "center-gaussian":
    #     # we interpret the rel_pos as the mean of a gaussian distribution
    #     z = torch.randn_like(_rel_xyz) * grid.voxel_sizes[0]
    else:
        raise NotImplementedError

    return rel_pos
