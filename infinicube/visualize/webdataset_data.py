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

import os
from time import sleep

import click
import numpy as np
import viser

from infinicube import get_sample
from infinicube.data_process.waymo_utils import (
    keep_car_only_in_object_info,
    object_info_to_cuboid,
)
from infinicube.utils.semantic_utils import semantic_to_color
from infinicube.visualize.utils import create_bbox_line_segments


def add_bboxes_to_scene(
    server, bbox_data, world_to_pc, folder_name="bbox", color=(0, 255, 0)
):
    """
    Add bounding boxes to the viser scene.

    Args:
        server: viser.ViserServer, the viser server instance
        bbox_data: dict, bounding box data containing object info
        world_to_pc: np.ndarray, shape (4, 4), transformation from world to point cloud coordinates
        color: tuple, RGB color for the bounding boxes
    """
    for obj_id, obj_info in bbox_data.items():
        # Get corners in world coordinates using waymo_utils function
        corners_world = object_info_to_cuboid(obj_info)

        # Transform corners from world to point cloud coordinates
        corners_world_homo = np.concatenate([corners_world, np.ones((8, 1))], axis=1)
        corners_pc = (world_to_pc @ corners_world_homo.T).T[:, :3]

        # Create line segments for the bounding box
        line_segments = create_bbox_line_segments(corners_pc)

        server.scene.add_line_segments(
            name=f"{folder_name}/{obj_id}",
            points=line_segments,
            colors=color,
            line_width=2.0,
        )


def visualize_wds_pc(root, clip_id, vis_bbox):
    path = os.path.join(root, "pc_voxelsize_01", f"{clip_id}.tar")
    data = get_sample(path)["pcd.vs01.pth"]
    points = data["points"].cpu().numpy()
    semantics = data["semantics"].cpu().numpy()
    visualizaiton_color = semantic_to_color(semantics)
    pc_to_world = data["pc_to_world"].cpu().numpy()
    world_to_pc = np.linalg.inv(pc_to_world)

    server = viser.ViserServer()
    server.scene.add_point_cloud(
        name="ground_truth_pc", points=points, colors=visualizaiton_color
    )

    if vis_bbox:
        # Load and visualize static bounding boxes
        bbox_path = os.path.join(root, "static_object_info", f"{clip_id}.tar")
        bbox_data = keep_car_only_in_object_info(get_sample(bbox_path))[
            "000100.static_object_info.json"
        ]
        add_bboxes_to_scene(
            server, bbox_data, world_to_pc, folder_name="static_bbox", color=(0, 255, 0)
        )
        bbox_path = os.path.join(root, "dynamic_object_info", f"{clip_id}.tar")
        bbox_data = keep_car_only_in_object_info(get_sample(bbox_path))[
            "000100.dynamic_object_info.json"
        ]
        add_bboxes_to_scene(
            server,
            bbox_data,
            world_to_pc,
            folder_name="dynamic_bbox",
            color=(255, 0, 0),
        )

    while True:
        sleep(1)


@click.command()
@click.option("--root", "-r", default="data/", help="root path to the waymo webdataset")
@click.option(
    "--clip_id", "-c", default="10107710434105775874_760_000_780_000", help="clip id"
)
@click.option("--vis_bbox", "-b", is_flag=True, help="visualize static bbox")
def main(root, clip_id, vis_bbox):
    visualize_wds_pc(root, clip_id, vis_bbox)


if __name__ == "__main__":
    main()
