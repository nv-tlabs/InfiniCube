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
from typing import Dict, Union

import fvdb
import numpy as np
import scipy.sparse as sp
import torch
import torch_scatter
import trimesh
from fvdb import GridBatch, JaggedTensor

from infinicube.camera.pinhole import PinholeCamera
from infinicube.data_process.waymo_utils import keep_car_only_in_object_info
from infinicube.utils.semantic_utils import WAYMO_CATEGORY_NAMES


def cc_removal_func(fvdb_grid, min_connected_voxels, grid_batch_kwargs):
    # Cross indices on 3x3x3 voxels.
    cc_inds = [4, 10, 12, 13, 14, 16, 22]

    # [N_ijk, 7], if neighbor voxel exists, store their unique index, else -1
    nn_inds = fvdb_grid.neighbor_indexes(fvdb_grid.ijk, 1).jdata.view(-1, 27)[
        :, cc_inds
    ]
    nn_mask = nn_inds > -1  # valid mask
    col_ind = nn_inds[nn_mask]
    row_ptr = torch.cumsum(torch.sum(nn_mask, dim=1), 0)
    row_ptr = torch.cat([torch.zeros(1).long().to(row_ptr.device), row_ptr])

    # if considered dense matrix, it would be [N_ijk, N_ijk]
    sp_mat = sp.csr_matrix(
        (
            np.ones(col_ind.size(0)),
            col_ind.cpu().numpy().astype(int),
            row_ptr.cpu().numpy().astype(int),
        )
    )
    _, component = sp.csgraph.connected_components(
        sp_mat, directed=False
    )  # belong to which component
    _, count = np.unique(
        component, return_counts=True
    )  # and how many voxels in each component

    cc_mask = (count > min_connected_voxels)[component]
    cc_mask = torch.from_numpy(cc_mask).to(fvdb_grid.device)

    print("Before CC removal:", fvdb_grid.total_voxels)
    print("After CC removal:", cc_mask.sum().item())

    fvdb_grid.set_from_ijk(fvdb_grid.ijk.jdata[cc_mask], **grid_batch_kwargs)

    return fvdb_grid


def points_to_fvdb(
    points,
    points_to_world,
    attrs=None,
    voxel_sizes=[0.1, 0.1, 0.1],
    origins=[0.05, 0.05, 0.05],
    bound_min=None,
    bound_max=None,
    cc_removal=False,
    cc_removal_min=-1,
    extra_meshes=None,
):
    """
    change points to fvdb grid

    Args:
    - points: torch.Tensor, shape=(N, 3), in a coordinate where we will consider as grid coordinate. `X-front, Y-left, Z-up`.
    - points_coord_to_world: torch.Tensor, shape=(4, 4), If the points' coordinate is something like vehicle cooridnate,
        we will have a transformation matrix converting points to world coordinate. Known as grid-to-world because the grid originates from points.

        Why we need this? Answer: the camera pose is stored in world coordinate. We will calculate camera-to-grid matrix at training.

    - attrs: dict of torch.Tensor, shape=(N, ?), attributes of points if exists (for example `semantics`). ? can be any number
    - voxel_size: float, size of voxel
    - bound: list of float, bound of fvdb grid

    Returns:
    - fvdb_grid: fvdb.GridBatch, the fvdb grid
    - attrs_for_voxel: dict of torch.Tensor, the attributes of the voxel

    Example:
        fvdb_grid.ijk.r_shape = (N_voxel, 3)
        attrs_for_voxel['semantics'].shape = (N_voxel,)
    """
    reduce_for_attrs = {
        "semantics": "argmax-category",
        "instance": "argmax-category",
        "dynamic_object_flag": "max",
    }

    grid_batch_kwargs = {"voxel_sizes": voxel_sizes, "origins": origins}

    grid_to_world = points_to_world  # grid originates from points, so they are the same

    # if bound_min and bound_max are not provided, we use all points
    if bound_min is None and bound_max is None:
        cropped_xyz = points
        cropped_attrs = attrs
    else:
        if bound_min is None:
            bound_min = [-1e7, -1e7, -1e7]
        if bound_max is None:
            bound_max = [1e7, 1e7, 1e7]

        mask = (
            (points[:, 0] >= bound_min[0])
            & (points[:, 0] < bound_max[0])
            & (points[:, 1] >= bound_min[1])
            & (points[:, 1] < bound_max[1])
            & (points[:, 2] >= bound_min[2])
            & (points[:, 2] < bound_max[2])
        )

        cropped_xyz = points[mask]
        cropped_attrs = (
            {k: v[mask] for k, v in attrs.items()} if attrs is not None else None
        )

    # build grid from points
    fvdb_grid = GridBatch("cuda")
    fvdb_grid = fvdb.gridbatch_from_points(cropped_xyz, **grid_batch_kwargs)

    # connected-component removal! This will reduce the number of voxels
    if cc_removal:
        # this is very, very large for 0.1m voxels, do it carefully
        if cc_removal_min == -1:
            min_cc_components = min((int(1 / voxel_sizes[0] / voxel_sizes[0]), 30))
        else:
            min_cc_components = cc_removal_min

        fvdb_grid = cc_removal_func(fvdb_grid, min_cc_components, grid_batch_kwargs)

    # now some points fall out of the grid, we need to filter them out. The remaining are valid voxels
    # world_to_grid returns a float tensor coordinate respecting to origins and voxel_sizes, when given any input point cloud coordinate (can be outside the grid)
    pts_vox_idx = fvdb_grid.ijk_to_index(
        fvdb_grid.world_to_grid(cropped_xyz).jdata.round().long()
    ).jdata
    pts_valid_mask = pts_vox_idx >= 0  # -1 means out of voxels

    # filter out invalid points (but usually there is no invalid points, since we have already filtered them out in the mask above)
    valid_pts_vox_idx = pts_vox_idx[pts_valid_mask]
    valid_attrs = (
        {k: v[pts_valid_mask] for k, v in cropped_attrs.items()}
        if cropped_attrs is not None
        else None
    )

    grid_voxel_attrs = {}
    # encode attributes
    if valid_attrs is not None:
        for attr_name, attr_value in valid_attrs.items():
            reduce_strategy = reduce_for_attrs[attr_name]
            if reduce_strategy == "argmax-category":
                unique_categories = torch.unique(attr_value)
                category_counts = []
                for category in unique_categories:
                    # torch_scatter version
                    category_count = torch_scatter.scatter_sum(
                        (attr_value == category).float(),
                        valid_pts_vox_idx,
                        dim=0,
                        dim_size=fvdb_grid.total_voxels,
                    )

                    # tensor version (slower. need to verify)
                    # category_count = torch.zeros(fvdb_grid.total_voxels, device=fvdb_grid.device).scatter_add(0, valid_pts_vox_idx, (attr_value == category).float())

                    category_counts.append(category_count)

                voxel_categories = torch.stack(category_counts, dim=1).argmax(dim=1)
                voxel_categories = unique_categories[voxel_categories]

                grid_voxel_attrs[attr_name] = voxel_categories

            elif reduce_strategy == "max":
                # torch_scatter version
                max_values = torch_scatter.scatter_max(
                    attr_value,
                    valid_pts_vox_idx,
                    dim=0,
                    dim_size=fvdb_grid.total_voxels,
                )[0]

                # tensor version (slower. need to verify)
                # max_values = torch.zeros(fvdb_grid.total_voxels, device=fvdb_grid.device).scatter_max(0, valid_pts_vox_idx, attr_value)[0]

                grid_voxel_attrs[attr_name] = max_values

            else:
                raise NotImplementedError(
                    f"Reduce strategy for {attr_name} is not implemented."
                )

    grid_voxel_attrs["grid_to_world"] = grid_to_world

    return fvdb_grid, grid_voxel_attrs


def generate_object_points_canonical_from_cad_model(
    car_object_info: Dict,
    cad_model_location=Path(__file__).parent.parent / "assets" / "car.ply",
):
    """
    Generate canonical points data for dynamic cars with cad model.

    Returns:
        object_points_canonical_data: dict,
            {'000000.xxx_object_info.json': {
                'object_gid1': {
                    'object_xyz': numpy.ndarray shape (M_i, 3)
                    'object_semantic': int waymo semantic category index,
                    'object_corner': numpy.ndarray shape (8, 3) canonical cuboid corners
                },
                'object_gid2': {
                    'object_xyz': numpy.ndarray shape (M_i, 3)
                    'object_semantic': int waymo semantic category index,
                    'object_corner': numpy.ndarray shape (8, 3) canonical cuboid corners
                },
                ...
            }}
    """
    # gather all dynamic car object info from dynamic_car_object_info
    all_car_gid_to_lwh = {}
    all_car_gid_to_type = {}

    for frame_idx_key, frame_object_data in car_object_info.items():
        if frame_idx_key.endswith(".json"):
            for object_gid, object_data in frame_object_data.items():
                if object_gid not in all_car_gid_to_lwh:
                    all_car_gid_to_lwh[object_gid] = object_data["object_lwh"]
                    all_car_gid_to_type[object_gid] = object_data["object_type"]

    # create a new dynamic_object_points_canonical_data
    new_object_points_canonical_data = {}
    cad_model_mesh = trimesh.load(cad_model_location)
    mesh_bounds = cad_model_mesh.bounds
    mesh_lwh = mesh_bounds[1] - mesh_bounds[0]  # [3,]

    mesh_vertices_list = []
    mesh_faces_list = []

    for object_gid in all_car_gid_to_lwh.keys():
        target_lwh = np.array(all_car_gid_to_lwh[object_gid])
        rescale = target_lwh / mesh_lwh
        transformed_mesh = cad_model_mesh.copy()
        transformed_mesh.apply_scale(rescale)
        mesh_vertices_list.append(torch.from_numpy(transformed_mesh.vertices).cuda())
        mesh_faces_list.append(torch.from_numpy(transformed_mesh.faces).cuda())

    if len(mesh_vertices_list) == 0:
        return {}

    mesh_vertices_jagged = JaggedTensor(mesh_vertices_list)
    mesh_faces_jagged = JaggedTensor(mesh_faces_list)

    # voxelize the mesh and extract points
    waymo_car_semantic_index = WAYMO_CATEGORY_NAMES.index("CAR")

    batched_object_voxels = fvdb.gridbatch_from_mesh(
        mesh_vertices_jagged,
        mesh_faces_jagged,
        voxel_sizes=[0.1, 0.1, 0.1],
        origins=[0.05, 0.05, 0.05],
    )
    batched_object_points = batched_object_voxels.grid_to_world(
        batched_object_voxels.ijk.float()
    )

    for gid, object_points_jagged in zip(
        all_car_gid_to_lwh.keys(), batched_object_points
    ):
        object_points = object_points_jagged.jdata
        new_object_points_canonical_data[gid + "_xyz"] = object_points.cpu().numpy()
        new_object_points_canonical_data[gid + "_semantic"] = waymo_car_semantic_index

    return new_object_points_canonical_data


def get_instance_id_for_fvdb_scene_points(
    fvdb_scene_points_in_world,
    fvdb_scene_semantic,
    static_object_info,
    enlarge_lwh_factor=1.0,
):
    """
    Get instance id for fvdb scene points. We have the bounding box

    Args:
        fvdb_scene_points_in_world: torch.Tensor, shape (N_point, 3)
        fvdb_scene_semantic: torch.Tensor, shape (N_point, )
        static_object_info: dict,
            "{frame_idx:06d}.static_object_info.json" : {
                object_gid1: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                object_gid2: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                ...
            }
            here world is in the waymo data's origin. Used for getting instance id for fvdb_scene_grid_or_points

    Returns:
        instance_id: torch.Tensor, shape (N_point, )
    """
    static_object_info_one_frame = static_object_info["000000.static_object_info.json"]

    # Filter out car category mask
    is_car_mask = (
        (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("CAR"))
        | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("TRUCK"))
        | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("BUS"))
        | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("OTHER_VEHICLE"))
    )

    # Initialize instance_id to 0 (background)
    instance_id = torch.zeros(
        fvdb_scene_points_in_world.shape[0],
        dtype=torch.int32,
        device=fvdb_scene_points_in_world.device,
    )

    # Only process car category points
    car_points = fvdb_scene_points_in_world[is_car_mask]
    car_instance_id = torch.zeros(
        car_points.shape[0], dtype=torch.int32, device=car_points.device
    )

    # Iterate through each static object bounding box
    for object_gid, object_data in static_object_info_one_frame.items():
        # Get object information
        object_to_world = torch.tensor(
            object_data["object_to_world"],
            dtype=torch.float32,
            device=car_points.device,
        )
        object_lwh = torch.tensor(
            object_data["object_lwh"], dtype=torch.float32, device=car_points.device
        )  # [l, w, h]
        object_id_int = object_data["object_id_int"]

        # Compute world_to_object transformation matrix
        world_to_object = torch.inverse(object_to_world)

        # Transform car points from world coordinate to object local coordinate
        car_points_homogeneous = torch.cat(
            [car_points, torch.ones(car_points.shape[0], 1, device=car_points.device)],
            dim=1,
        )  # [N_car, 4]
        car_points_in_object = (world_to_object @ car_points_homogeneous.T).T[
            :, :3
        ]  # [N_car, 3]

        # Check if points are inside the bounding box
        # Bounding box is centered at origin with range [-l/2, l/2] x [-w/2, w/2] x [-h/2, h/2]
        half_lwh = object_lwh / 2.0 * enlarge_lwh_factor
        in_box_mask = (
            (car_points_in_object[:, 0].abs() <= half_lwh[0])
            & (car_points_in_object[:, 1].abs() <= half_lwh[1])
            & (car_points_in_object[:, 2].abs() <= half_lwh[2])
        )

        # Assign object_id_int to points inside the bounding box
        car_instance_id[in_box_mask] = object_id_int

    # Assign car instance_id back to the original instance_id tensor
    instance_id[is_car_mask] = car_instance_id

    return instance_id


def generate_infinicube_buffer_from_fvdb_grid(
    camera_model: PinholeCamera,
    camera_poses_in_world: torch.Tensor,
    fvdb_scene_grid_or_points: Union[GridBatch, torch.Tensor],
    fvdb_scene_semantic: torch.Tensor,
    fvdb_grid_to_world: torch.Tensor,
    static_object_info: Dict,
    dynamic_object_info: Dict = None,
    dynamic_object_points_canonical_data: Dict = None,
    cad_model_for_static_object: bool = False,
    cad_model_for_dynamic_objects: bool = False,
    cad_model_location=Path(__file__).parent.parent / "assets" / "car.ply",
    voxel_sizes=[0.2, 0.2, 0.2],
    enlarge_lwh_factor=1.2,
):
    """
    Args:
        camera_model: PinholeCamera

        camera_poses: torch.Tensor, (N, 4, 4) or (4, 4)

        fvdb_scene_grid_or_points: GridBatch or torch.Tensor
            if GridBatch, suppose it have grid.total_voxels = N_voxel
            if torch.Tensor, suppose it is still unvoxelized point cloud, shape (N_point, 3)

        fvdb_scene_semantic: torch.Tensor
            if GridBatch, fvdb_scene_semantic.shape = (N_voxel, )
            if torch.Tensor, fvdb_scene_semantic.shape = (N_point, )

        fvdb_grid_to_world: torch.Tensor,
            No matter fvdb_scene_grid_or_points is GridBatch or torch.Tensor, fvdb_grid_to_world takes effect.
            transform points in the grid to waymo data 's origin. (4, 4)

        static_object_info: dict,
            "{frame_idx:06d}.static_object_info.json" : {
                object_gid1: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                object_gid2: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                ...
            }
            here world is in the waymo data's origin. Used for getting instance id for fvdb_scene_grid_or_points

        dynamic_object_info: dict,
            "{frame_idx:06d}.dynamic_object_info.json" : {
                object_gid1: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                object_gid2: {'object_to_world': (4, 4), 'object_lwh': (l, w, h), 'object_is_moving': False, 'object_type': 'car', 'object_id_int': int},
                ...
            }
            here world is in the waymo data's origin.

        dynamic_object_points_canonical_data: dict,
            {'gid1_xyz': numpy.ndarray shape (M_i, 3)
             'gid1_semantic': int waymo semantic category index,
             'gid1_corner': numpy.ndarray shape (8, 3) canonical cuboid corners
            }

        cad_model_for_dynamic_objects: bool = False,
            use cad model to replace dynamic_object_points_canonical_data.

        cad_model_location: Path, location of cad model


    Returns:
        semantic_buffer: torch.Tensor, (N, H, W) or (H, W)
        instance_buffer: torch.Tensor, (N, H, W) or (H, W)
        depth_buffer: torch.Tensor, (N, H, W) or (H, W)
    """
    # store the output
    depth_buffer_list = []
    semantic_buffer_list = []
    instance_buffer_list = []

    single_pose_input = False
    if len(camera_poses_in_world.shape) == 2:
        camera_poses_in_world = camera_poses_in_world.unsqueeze(0)
        single_pose_input = True

    # prepare static / dynamic object points canonical data
    static_car_object_info = keep_car_only_in_object_info(static_object_info)
    dynamic_car_object_info = keep_car_only_in_object_info(dynamic_object_info)

    if cad_model_for_static_object:
        static_object_points_canonical_data = (
            generate_object_points_canonical_from_cad_model(
                static_car_object_info, cad_model_location
            )
        )
    else:
        static_object_points_canonical_data = {}

    if cad_model_for_dynamic_objects:
        dynamic_object_points_canonical_data = (
            generate_object_points_canonical_from_cad_model(
                dynamic_car_object_info, cad_model_location
            )
        )
    else:
        dynamic_object_points_canonical_data = dynamic_object_points_canonical_data

    adding_object_points_canonical_data = {
        **static_object_points_canonical_data,
        **dynamic_object_points_canonical_data,
    }

    # load static scene points and semantic
    if isinstance(fvdb_scene_grid_or_points, GridBatch):
        fvdb_scene_points = fvdb_scene_grid_or_points.grid_to_world(
            fvdb_scene_grid_or_points.ijk.float()
        ).jdata
    else:
        fvdb_scene_points = fvdb_scene_grid_or_points

    # if static object are cad model, we remove all car points from the scene
    if cad_model_for_static_object:
        is_car_mask = (
            (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("CAR"))
            | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("TRUCK"))
            | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("BUS"))
            | (fvdb_scene_semantic == WAYMO_CATEGORY_NAMES.index("OTHER_VEHICLE"))
        )
        fvdb_scene_points = fvdb_scene_points[~is_car_mask]
        fvdb_scene_semantic = fvdb_scene_semantic[~is_car_mask]

    fvdb_scene_points_in_world = camera_model.transform_points(
        fvdb_scene_points, fvdb_grid_to_world
    )
    fvdb_scene_instanceid = get_instance_id_for_fvdb_scene_points(
        fvdb_scene_points_in_world,
        fvdb_scene_semantic,
        static_object_info,
        enlarge_lwh_factor,
    )

    # assemble static / dynamic objects into the scene
    for frame_index, camera_pose in enumerate(camera_poses_in_world):
        # put static / dynamic objects into the scene
        dynamic_object_data_this_frame = dynamic_car_object_info[
            f"{frame_index:06d}.dynamic_object_info.json"
        ]
        static_object_data_this_frame = static_car_object_info[
            f"{frame_index:06d}.static_object_info.json"
        ]

        adding_object_data_this_frame = dynamic_object_data_this_frame
        if cad_model_for_static_object:
            adding_object_data_this_frame.update(static_object_data_this_frame)

        added_object_points_list = [np.zeros((0, 3))]
        added_object_semantic_list = [torch.zeros(0)]
        added_object_instanceid_list = [torch.zeros(0)]

        for object_gid, object_data in adding_object_data_this_frame.items():
            object_to_world = np.array(object_data["object_to_world"])
            object_points = adding_object_points_canonical_data[object_gid + "_xyz"]
            object_points_in_waymo = camera_model.transform_points(
                object_points, object_to_world
            )
            object_semantic = adding_object_points_canonical_data[
                object_gid + "_semantic"
            ]  # int
            object_instanceid = object_data["object_id_int"]  # int

            added_object_points_list.append(object_points_in_waymo)
            added_object_semantic_list.append(
                torch.ones(object_points_in_waymo.shape[0]) * object_semantic
            )
            added_object_instanceid_list.append(
                torch.ones(object_points_in_waymo.shape[0]) * object_instanceid
            )

        added_object_points = torch.from_numpy(
            np.concatenate(added_object_points_list, axis=0)
        ).to(fvdb_scene_points_in_world)
        added_object_semantic = torch.cat(added_object_semantic_list, dim=0).to(
            fvdb_scene_semantic
        )
        added_object_instanceid = torch.cat(added_object_instanceid_list, dim=0).to(
            fvdb_scene_instanceid
        )

        fvdb_scene_points_with_added_objects_in_waymo = torch.cat(
            [fvdb_scene_points_in_world, added_object_points], dim=0
        )
        fvdb_scene_semantic_with_added_objects_in_waymo = torch.cat(
            [fvdb_scene_semantic, added_object_semantic], dim=0
        )
        fvdb_scene_instanceid_with_added_objects_in_waymo = torch.cat(
            [fvdb_scene_instanceid, added_object_instanceid], dim=0
        )

        # build fvdb grid with dynamic objects
        fvdb_scene_grid_with_dynamic_objects, grid_voxel_attrs = points_to_fvdb(
            fvdb_scene_points_with_added_objects_in_waymo,
            fvdb_grid_to_world,
            attrs={
                "semantics": fvdb_scene_semantic_with_added_objects_in_waymo,
                "instance": fvdb_scene_instanceid_with_added_objects_in_waymo,
            },
            voxel_sizes=voxel_sizes,
            origins=[voxel_sizes[0] / 2, voxel_sizes[1] / 2, voxel_sizes[2] / 2],
        )

        # render the scene
        depth = camera_model.get_zdepth_map_from_voxel(
            camera_pose, fvdb_scene_grid_with_dynamic_objects
        )
        semantic = camera_model.get_semantic_map_from_voxel(
            camera_pose,
            fvdb_scene_grid_with_dynamic_objects,
            grid_voxel_attrs["semantics"].to(torch.int32),
        )
        # reuse the function
        instance = camera_model.get_semantic_map_from_voxel(
            camera_pose,
            fvdb_scene_grid_with_dynamic_objects,
            grid_voxel_attrs["instance"].to(torch.int32),
        )

        depth_buffer_list.append(depth)
        semantic_buffer_list.append(semantic)
        instance_buffer_list.append(instance)

    depth_buffer = torch.stack(depth_buffer_list, dim=0)
    semantic_buffer = torch.stack(semantic_buffer_list, dim=0)
    instance_buffer = torch.stack(instance_buffer_list, dim=0)

    if single_pose_input:
        depth_buffer = depth_buffer.squeeze(0)
        semantic_buffer = semantic_buffer.squeeze(0)
        instance_buffer = instance_buffer.squeeze(0)

    return depth_buffer, semantic_buffer, instance_buffer


if __name__ == "__main__":
    from infinicube import get_sample
    from infinicube.utils.depth_utils import vis_depth
    from infinicube.utils.fileio_utils import write_video_file
    from infinicube.utils.instance_utils import coloring_instance_map
    from infinicube.utils.semantic_utils import semantic_to_color

    camera_intrinsic = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/intrinsic/10107710434105775874_760_000_780_000.tar"
    )["intrinsic.front.npy"]
    dynamic_object_info = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/dynamic_object_info/10107710434105775874_760_000_780_000.tar"
    )
    static_object_info = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/static_object_info/10107710434105775874_760_000_780_000.tar"
    )
    dynamic_object_points_canonical_data = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/dynamic_object_points_canonical/10107710434105775874_760_000_780_000.tar"
    )["dynamic_object_points_canonical.npz"]
    static_fvdb_scene = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/pc_voxelsize_01/10107710434105775874_760_000_780_000.tar"
    )["pcd.vs01.pth"]
    camera_poses = get_sample(
        "/home/yiflu/holodeck/yiflu/InfiniCube-release/data/pose/10107710434105775874_760_000_780_000.tar"
    )
    camera_poses = torch.from_numpy(
        np.stack([camera_poses[f"{i:06d}.pose.front.npy"] for i in range(10)])
    ).cuda()
    camera_model = PinholeCamera.from_numpy(camera_intrinsic, device="cuda")

    depth_buffer, semantic_buffer, instance_buffer = (
        generate_infinicube_buffer_from_fvdb_grid(
            camera_model=camera_model,
            camera_poses_in_world=camera_poses,
            fvdb_scene_grid_or_points=static_fvdb_scene["points"],
            fvdb_scene_semantic=static_fvdb_scene["semantics"],
            fvdb_grid_to_world=static_fvdb_scene["pc_to_world"],
            static_object_info=static_object_info,
            dynamic_object_info=dynamic_object_info,
            dynamic_object_points_canonical_data=dynamic_object_points_canonical_data,
            cad_model_for_dynamic_objects=True,
            cad_model_location=Path(__file__).parent.parent / "assets" / "car.ply",
        )
    )

    # Visualize depth buffer
    print("Visualizing depth buffer...")
    depth_frames = []
    for i in range(depth_buffer.shape[0]):
        depth_vis = vis_depth(depth_buffer[i])  # Returns (H, W, 3) in range [0, 255]
        if torch.is_tensor(depth_vis):
            depth_vis = depth_vis.cpu().numpy()
        depth_frames.append(depth_vis)

    # Visualize semantic buffer
    print("Visualizing semantic buffer...")
    semantic_frames = []
    for i in range(semantic_buffer.shape[0]):
        semantic_vis = semantic_to_color(
            semantic_buffer[i].cpu().numpy().astype(np.int32)
        )  # Returns (H, W, 3) in range [0, 1]
        semantic_vis = (semantic_vis * 255).astype(np.uint8)  # Convert to [0, 255]
        semantic_frames.append(semantic_vis)

    # Visualize instance buffer
    print("Visualizing instance buffer...")
    instance_vis = coloring_instance_map(
        instance_buffer
    )  # Returns (N, H, W, 3) in range [0, 1]
    if torch.is_tensor(instance_vis):
        instance_vis = (
            (instance_vis * 255).cpu().numpy().astype(np.uint8)
        )  # Convert to [0, 255]
    else:
        instance_vis = (instance_vis * 255).astype(np.uint8)
    instance_frames = [instance_vis[i] for i in range(instance_vis.shape[0])]

    # Write videos
    print("Writing depth video...")
    write_video_file(depth_frames, "depth_visualization.mp4", fps=10)

    print("Writing semantic video...")
    write_video_file(semantic_frames, "semantic_visualization.mp4", fps=10)

    print("Writing instance video...")
    write_video_file(instance_frames, "instance_visualization.mp4", fps=10)

    print("All videos saved successfully!")
