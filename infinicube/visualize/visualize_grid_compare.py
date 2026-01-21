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

"""
python infinicube/visualize/visualize_grid_compare.py \
    -p visualization/voxel_generation_single_chunk/diffusion_64x64x64_dense_vs02_map_cond/test_starting_at_50 \
    -p visualization/voxel_generation_single_chunk/diffusion_64x64x64_dense_vs02_map_cond/test_starting_at_50_map_grid

"""

import os
import time
from pathlib import Path

import click
import numpy as np
import viser
import viser.transforms as tf
from pycg import image, render, vis

from infinicube.utils.fileio_utils import read_fvdb_grid_and_semantic
from infinicube.utils.semantic_utils import semantic_to_color
from infinicube.utils.viser_gui_utils import set_kill_key_button
from infinicube.voxelgen.utils.voxel_util import single_semantic_voxel_to_mesh


def render_point_cloud(
    server, points, colors=None, point_size=0.1, name="/simple_pc", port=8080
):
    """
    points: [N, 3]
    colors: [3,] or [N, 3]
    """
    setattr(server, "alive", True)

    if colors is None:
        colors = (90, 200, 255)

    server.scene.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=point_size,
    )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1 / 60)

        if not server.alive:
            server.scene.reset()
            break


def render_multiple_polygon_mesh(
    server, vertices_list, face_list, color_list, name="/simple_mesh", port=8080
):
    """
    Render multiple polygon meshes without texture

    Args:
        vertices_list (List[ndarray])
            A list of numpy array of vertex positions. Each array should have shape (V, 3).
        faces_list (List[ndarray])
            A list of numpy array of face indices. Each array should have shape (F, 3).
        color_list (List[Tuple[int, int, int] | Tuple[float, float, float] | ndarray])
            A list of color of the mesh as an RGB tuple.
    """
    setattr(server, "alive", True)

    for i, vertices in enumerate(vertices_list):
        server.add_mesh_simple(
            name=name + f"_{i}",
            vertices=vertices,
            faces=face_list[i],
            color=color_list[i],
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0, 0, 0),
        )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            camera = client.camera
            # ic(id, camera.position, camera.wxyz)
        time.sleep(1 / 60)

        if not server.alive:
            server.scene.reset()
            break

    time.sleep(1)


def visualize_grid_compare(paths, port, type, size):
    assert len(paths) > 0, "No path is provided"
    center = True

    for path in paths:
        assert os.path.isdir(path), f"{path} is not a directory"
    server = viser.ViserServer(port=port)
    gui_kill_button = server.gui.add_button(
        "Kill", hint="Press this button to kill this sample"
    )
    set_kill_key_button(server, gui_kill_button)

    folder_p = Path(paths[0])
    gt_pt_files = [x for x in os.listdir(folder_p) if "_gt.pt" in x]
    gt_pt_files = [os.path.join(folder_p, x) for x in gt_pt_files]
    sample_num = len(gt_pt_files)
    print(f"Total {sample_num} samples")

    pred_pt_files_diff = []

    for i in range(sample_num):
        vox_ijk = []
        vox_semantic = []

        pred_pt_files_diff = []

        gt_file = (Path(paths[0]) / f"{i}_gt.pt").as_posix()
        assert gt_file.exists(), "the first path must contain the gt grid as *_gt.pt"
        for path in paths:
            folder_p = Path(path)
            pred_file = (folder_p / f"{i}.pt").as_posix()
            pred_pt_files_diff.append(pred_file)

        print_info = "\n".join(pred_pt_files_diff)
        print(f"Visualizing {i + 1}/{sample_num}, from \n{gt_file} and \n{print_info}")

        # add GT
        gt_voxel_dict = read_fvdb_grid_and_semantic(gt_file)
        vox_ijk.append(gt_voxel_dict["ijk"])
        vox_semantic.append(gt_voxel_dict["semantics"])

        interval = int(
            (np.max(gt_voxel_dict["ijk"][:, 1]) - np.min(gt_voxel_dict["ijk"][:, 1]))
            * 1.1
        )

        # add preds
        for idx, pred_file in enumerate(pred_pt_files_diff):
            pred_voxel_dict = read_fvdb_grid_and_semantic(pred_file)
            pred_voxel_dict["ijk"][:, 1] += (
                idx + 1
            ) * interval  # shift the voxel. pred is on the right (+Y)
            vox_ijk.append(pred_voxel_dict["ijk"])
            vox_semantic.append(pred_voxel_dict["semantics"])

        vox_ijk = np.concatenate(vox_ijk, axis=0)
        vox_semantic = np.concatenate(vox_semantic, axis=0)

        if center:
            vox_ijk_center = np.round(np.mean(vox_ijk, axis=0))
            vox_ijk = vox_ijk - vox_ijk_center
            vox_ijk = vox_ijk.astype(np.int32)

        if type == "voxel":
            semantic_labels = np.unique(vox_semantic)
            cube_v_list = []
            cube_f_list = []
            cube_color_list = []
            geometry_list = []

            for semantic in semantic_labels:
                mask = vox_semantic == semantic
                if sum(mask) == 0:
                    continue

                cube_v_i, cube_f_i = single_semantic_voxel_to_mesh(vox_ijk[mask])
                color_i = semantic_to_color(semantic)

                cube_v_list.append(cube_v_i)
                cube_f_list.append(cube_f_i)
                cube_color_list.append(color_i)

                geometry = vis.mesh(
                    cube_v_i,
                    cube_f_i,
                    np.array(color_i).reshape(1, 3).repeat(cube_v_i.shape[0], axis=0),
                )
                geometry_list.append(geometry)

            save_render = True
            if save_render:
                scene: render.Scene = vis.show_3d(
                    geometry_list,
                    show=False,
                    up_axis="+Z",
                    default_camera_kwargs={
                        "pitch_angle": 80.0,
                        "fill_percent": 0.8,
                        "fov": 80.0,
                        "plane_angle": 270,
                    },
                )
                img = scene.render_filament()
                image.write(img, f"visualization/grid_compare/{i}.png")

            render_multiple_polygon_mesh(
                server, cube_v_list, cube_f_list, cube_color_list
            )

        elif type == "pc":
            vox_ijk = vox_ijk * 0.1
            vox_semantic_color = semantic_to_color(vox_semantic, palette=palette)
            render_point_cloud(
                server, vox_ijk, vox_semantic_color, point_size=size, port=port
            )


@click.command()
@click.option("--paths", "-p", multiple=True, help="directories of .pt files")
@click.option("--port", "-o", default=8080, help="port number")
@click.option(
    "--type",
    "-t",
    default="voxel",
    help="voxel or pc. voxel can not be used for 1024 resolution grid.",
)
@click.option("--size", "-s", default=0.05, help="point size for point cloud")
def main(paths, port, type, size):
    visualize_grid_compare(paths, port, type, size)


if __name__ == "__main__":
    main()
