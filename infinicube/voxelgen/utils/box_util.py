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

import torch


def get_points_in_cuboid_torch(points, cuboid):
    """
    Get points that are inside a cuboid defined by object_lwh and object_to_world transformation.

    Args:
        points: torch.Tensor, shape=(N, 3), points in the grid coordinate
        cuboid: dict, object information, with the following keys:
            - object_lwh: list, l, w, h
            - object_to_world: list, shape=(4, 4), the transformation matrix from object to world coordinate

    Returns:
        points_in_cuboid: torch.Tensor, shape=(N, 3), the points transformed to cuboid coordinate
        mask: torch.Tensor, shape=(N,), the mask of points inside the cuboid
    """
    box_l, box_w, box_h = cuboid["object_lwh"]

    object_to_world = torch.tensor(cuboid["object_to_world"]).to(points.device)
    grid_to_object = torch.inverse(object_to_world)

    # transform points to object coordinate
    points_homogeneous = torch.cat(
        [points, torch.ones(points.shape[0], 1).to(points.device)], dim=1
    )
    points_in_cuboid = torch.matmul(points_homogeneous, grid_to_object.T)
    points_in_cuboid = points_in_cuboid[:, :3]

    # filter points inside the cuboid
    mask = (
        (points_in_cuboid[:, 0] >= -box_l / 2)
        & (points_in_cuboid[:, 0] <= box_l / 2)
        & (points_in_cuboid[:, 1] >= -box_w / 2)
        & (points_in_cuboid[:, 1] <= box_w / 2)
        & (points_in_cuboid[:, 2] >= -box_h / 2)
        & (points_in_cuboid[:, 2] <= box_h / 2)
    )

    return points_in_cuboid, mask
