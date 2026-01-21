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
from typing import Dict

import numpy as np
import trimesh


def build_scene_mesh_from_all_object_info(
    all_object_dict: Dict,
    world_transform: np.ndarray = np.eye(4),
    aabb_half_range: np.ndarray = None,
    plyfile: str = Path(__file__).parent.parent / "assets" / "car.ply",
):
    """
    Create a scene consisting all car object, rescale them, transform them and merge them

    Args:
        all_object_dict: dict, dict containing all object information, webdataset's all_object_info
            {
                'object_gid1': {
                    'object_to_world': np.ndarray, [4, 4], object to world transform matrix
                    'object_lwh': np.ndarray, [3,], object length, width, height
                    'object_type': str, object type
                }, ...
            }
        world_transform: np.array, [4, 4], world transform matrix to transform meshes to another coordinate system
        aabb_half_range: np.array, [3,], half range of the axis aligned bounding box to filter out objects
        plyfile: str, path to the ply file

    Returns:
        scene_mesh_vertices: np.ndarray, [N, 3], scene mesh vertices
        scene_mesh_faces: np.ndarray, [M, 3], scene mesh faces
    """
    mesh = trimesh.load(plyfile)
    mesh_bounds = mesh.bounds
    mesh_lwh = mesh_bounds[1] - mesh_bounds[0]  # [3,]

    scene_meshes = []
    for gid, object_info in all_object_dict.items():
        object_to_world = np.array(object_info["object_to_world"])
        target_lwh = np.array(object_info["object_lwh"])
        is_car = object_info["object_type"] == "car"
        rescale = target_lwh / mesh_lwh

        if is_car:
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_scale(rescale)
            transformed_mesh.apply_transform(world_transform @ object_to_world)

            if aabb_half_range is not None:
                aabb_half_range_np = np.array(aabb_half_range)
                aabb_range = np.stack([-aabb_half_range_np, aabb_half_range_np])
                aabb = aabb_range.tolist()

                vertices_inside = trimesh.bounds.contains(
                    aabb, transformed_mesh.vertices
                )
                if np.all(vertices_inside):
                    scene_meshes.append(transformed_mesh)

            else:
                scene_meshes.append(transformed_mesh)

    scene_meshes = trimesh.util.concatenate(scene_meshes)
    scene_mesh_vertices = np.asarray(scene_meshes.vertices)
    scene_mesh_faces = np.asarray(scene_meshes.faces)

    return scene_mesh_vertices, scene_mesh_faces
