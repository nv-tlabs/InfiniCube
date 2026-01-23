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

import imageio
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from infinicube.camera.pinhole import PinholeCamera
from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.data.base import list_collate
from infinicube.voxelgen.data.waymo_wds import WaymoWdsDataset
from infinicube.voxelgen.models.base_model import BaseModel
from infinicube.voxelgen.utils.voxel_util import (
    offscreen_voxel_list_to_mesh_renderer,
)
from infinicube.utils.semantic_utils import semantic_to_color
from infinicube.utils.depth_utils import vis_depth


def test_waymo_wds():
    dataset = WaymoWdsDataset(
        wds_root_url="data/",
        # wds_scene_list_file="12027892938363296829_4086_280_4106_280",
        wds_scene_list_file="4164064449185492261_400_000_420_000",
        attr_subfolders=[
            # "pc_with_map_without_car_voxelsize_01",
            "pc_voxelsize_01",
            "pose",
            "intrinsic",
            "image_480p_front",
            "skymask_480p_front",
            "voxel_depth_100_480p_front",
            "static_object_info",
            "dynamic_object_info",
            "dynamic_object_points_canonical",
            "3d_road_edge_voxelsize_025",
            "3d_road_line_voxelsize_025",
            "3d_road_surface_voxelsize_04",
        ],
        spec=[
            DS.INPUT_PC,
            DS.GT_SEMANTIC,
            DS.IMAGES_INPUT,
            DS.IMAGES,
            DS.IMAGES_INPUT_DEPTH,
            DS.IMAGES_DEPTH_VOXEL,
            DS.BOXES_3D,
            DS.MAPS_3D,
        ],
        replace_all_car_with_cad=False,
        grid_crop_bbox_min=[-10.24, -51.2, -12.8],
        grid_crop_bbox_max=[92.16, 51.2, 38.4],
        fvdb_grid_type="vs02",
        finest_voxel_size_goal="vs02",
        input_slect_ids=[0],
        sup_slect_ids=[0],
        input_frame_offsets=[0],
        sup_frame_offsets=[0, 1, 2],
        n_image_per_iter_sup=3,
        input_depth_type="voxel_depth_100",
        shuffle_buffer=128,
        random_seed=0,
        split="val",
        frame_start_num=30,
        frame_end_num=170,
        front_view_input_wh=(832, 480),
        grid_crop_augment=False,
        grid_crop_augment_range=[12.8, 12.8, 3.2],
        map_types=["road_edge", "road_line", "road_surface"],
    )

    for data in dataset:
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape, value.dtype)
            else:
                print(key, value)

        break

    return dataset

def test_waymo_wds_dataloader():
    dataset = test_waymo_wds()
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=1, collate_fn=list_collate
    )
    Path("visualization/dataset").mkdir(exist_ok=True)
    for idx, data in enumerate(dataloader):
        BaseModel.generate_fvdb_grid_on_the_fly(data)
        grid_semantic_pairs = [(data[DS.INPUT_PC], data[DS.GT_SEMANTIC][0])]
        rendered_image = offscreen_voxel_list_to_mesh_renderer(grid_semantic_pairs)
        imageio.imwrite(
            f"visualization/dataset/{idx}_voxel_render.png",
            rendered_image,
        )

        camera_model = PinholeCamera.from_tensor(data[DS.IMAGES_INPUT_INTRINSIC][0][0])
        voxel_depth = camera_model.get_zdepth_map_from_voxel(
            data[DS.IMAGES_INPUT_POSE][0][0],
            data[DS.INPUT_PC],
        )
        voxel_depth_rgb = vis_depth(voxel_depth.cpu().numpy())
        voxel_semantic = camera_model.get_semantic_map_from_voxel(
            data[DS.IMAGES_INPUT_POSE][0][0],
            data[DS.INPUT_PC],
            data[DS.GT_SEMANTIC][0],
        )
        voxel_semantic_rgb = (semantic_to_color(voxel_semantic) * 255).astype(np.uint8)

        image_rgb = (data[DS.IMAGES_INPUT][0][0].cpu().numpy() * 255).astype(np.uint8)

        # draw map on the image
        for map_type, map_points_list in data[DS.MAPS_3D].items():
            if map_type == 'road_edge':
                colors = np.array([255, 0, 0])
            elif map_type == 'road_line':
                colors = np.array([0, 255, 0])
            elif map_type == 'road_surface':
                colors = np.array([0, 0, 255])
            else:
                raise ValueError(f"Unsupported map type: {map_type}")

            draw_map = camera_model.draw_points(
                data[DS.IMAGES_INPUT_POSE][0][0],
                map_points_list[0],
                colors=colors.reshape(1, 3).repeat(map_points_list[0].shape[0], axis=0),
                radius=1,
            )
            imageio.imwrite(f"visualization/dataset/{idx}_{map_type}.png", draw_map)

        # draw bounding boxes on the image using draw hull depth
        draw_hull_depth = camera_model.draw_hull_depth(
            data[DS.IMAGES_INPUT_POSE][0][0].cpu().numpy(),
            data[DS.BOXES_3D][0].cpu().numpy(),
            colors=np.array([255, 0, 0]),
            depth_max=122.5,
        )
        imageio.imwrite(f"visualization/dataset/{idx}_bounding_boxes.png", draw_hull_depth)

        imageio.imwrite(f"visualization/dataset/{idx}_image.png", image_rgb)
        imageio.imwrite(f"visualization/dataset/{idx}_voxel_depth.png", voxel_depth_rgb)
        imageio.imwrite(f"visualization/dataset/{idx}_voxel_semantic.png", voxel_semantic_rgb)

        print(f"Visualization saved for batch {idx}")


def test_waymo_wds_bbox():
    from pytorch3d.ops.iou_box3d import _check_coplanar, _check_nonzero

    dataset = test_waymo_wds()
    for data in dataset:
        _check_coplanar(data[DS.BOXES_3D].float())
        _check_nonzero(data[DS.BOXES_3D].float())


if __name__ == "__main__":
    test_waymo_wds_bbox()
