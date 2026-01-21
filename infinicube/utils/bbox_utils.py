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

from typing import Dict

import numpy as np
import trimesh


def to_cuboid_corners(object_lwh, object_to_another_coord):
    """

    bbox format:
        h
        ^  w
        | /
        |/
        o -------> l (heading)

       3 ---------------- 0
      /|                 /|
     / |                / |
    2 ---------------- 1  |
    |  |               |  |
    |  7 ------------- |- 4
    | /                | /
    6 ---------------- 5


    Args:
    - object_lwh: list, l, w, h
    - object_to_another_coord: np.ndarray, shape=(4, 4), transformation matrix from object to another coordinate system (e.g. world)

    Returns:
    - corners: np.ndarray, shape=(8, 3), the 8 corners of the object in the world coordinate
    """

    size = np.array(object_lwh)
    corners_obj = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
    )
    corners_obj = corners_obj * size
    corners_obj = corners_obj - size / 2
    # pad 1 for homogeneous coordinates
    corners_obj = np.concatenate([corners_obj, np.ones((8, 1))], axis=1)
    corners = np.einsum("ij,kj->ki", object_to_another_coord, corners_obj)[:, :3]

    return corners


def build_scene_bounding_boxes_from_object_info(
    all_object_dict: Dict,
    apply_object_to_world: bool = True,
    world_to_target_coord: np.ndarray = np.eye(4),
    aabb_half_range: np.ndarray = None,
):
    """
    Create a scene consisting all car object, rescale them, transform them and merge them

    Args:
        all_object_dict: dict, dict containing all object information, webdataset's all_object_info
        apply_object_to_world: bool, if True, apply object_to_world to the bounding box
        world_to_target_coord: np.array, [4, 4], transform matrix to transform bounding box from world to another coordinate system
        aabb_half_range: np.array, [3,], half range of the axis aligned bounding box to filter out objects

    Returns:
        all_cuboids: np.ndarray, shape=(N, 8, 3), the 8 corners of the object in the world coordinate
    """
    all_cuboids = []

    for gid, object_info in all_object_dict.items():
        object_to_world = np.array(object_info["object_to_world"])
        target_lwh = np.array(object_info["object_lwh"])
        is_car = object_info["object_type"] == "car"

        if is_car:
            if apply_object_to_world:
                object_to_target_coord = world_to_target_coord @ object_to_world
            else:
                object_to_target_coord = world_to_target_coord

            corners = to_cuboid_corners(target_lwh, object_to_target_coord)

            if aabb_half_range is not None:
                aabb_half_range_np = np.array(aabb_half_range)
                aabb_range = np.stack([-aabb_half_range_np, aabb_half_range_np])
                aabb = aabb_range.tolist()

                vertices_inside = trimesh.bounds.contains(aabb, corners.reshape(-1, 3))
                if np.all(vertices_inside):
                    all_cuboids.append(corners)
            else:
                all_cuboids.append(corners)

    if len(all_cuboids) == 0:
        return np.zeros((0, 8, 3))

    return np.stack(all_cuboids, axis=0)
