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

import time

import click
import numpy as np
import trimesh
import viser
from pycg import vis

from infinicube.utils.fileio_utils import read_fvdb_grid_and_semantic
from infinicube.utils.record_utils import record_plugin
from infinicube.utils.semantic_utils import semantic_to_color
from infinicube.utils.viser_gui_utils import set_kill_gui, set_recording_gui


def render_trimesh(meshes, basic_name="/simple_trimesh", port=8080):
    """
    Render a trimesh object, can be textured.

    Args:
        mesh (trimesh.Trimesh) | or list of (trimesh.Trimesh)
            one or multiple trimesh object.
    """
    server = viser.ViserServer(port=port)
    set_recording_gui(server)
    set_kill_gui(server)

    # make it to list if only one object
    if isinstance(meshes, trimesh.Trimesh) or isinstance(meshes, trimesh.Scene):
        meshes = [meshes]

    for i, mesh in enumerate(meshes):
        name = basic_name + f"_{i}"
        server.object_handlers["mesh"].append(
            server.scene.add_mesh_trimesh(name=name, mesh=mesh)
        )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            client.camera.fov = np.deg2rad(server.fovy_modifier.value)
            record_plugin(server, client)
        time.sleep(1 / 60)


def render_point_cloud(
    points, colors=None, point_size=0.1, name="/simple_pc", port=8080
):
    """
    points: [N, 3]
    colors: [3,] or [N, 3]
    """
    server = viser.ViserServer(port=port)
    set_recording_gui(server)
    set_kill_gui(server)

    server.object_handlers["pointcloud"].append(
        server.scene.add_point_cloud(
            name=name,
            points=points,
            colors=colors,
            point_size=point_size,
        )
    )

    while True:
        clients = server.get_clients()
        for id, client in clients.items():
            client.camera.fov = np.deg2rad(server.fovy_modifier.value)
            record_plugin(server, client)
        time.sleep(1 / 60)


@click.command()
@click.option(
    "--path", "-p", help="path to the .pkl / .pt / .tar file containing fvdb grid"
)
@click.option("--port", "-o", default=8080, help="port number")
@click.option(
    "--render_type", "-t", default="mesh", help="render as mesh or point cloud"
)
def main(path, port, render_type):
    voxel_data = read_fvdb_grid_and_semantic(path)
    colors_float32 = semantic_to_color(voxel_data["semantics"])
    colors_uint8 = (colors_float32 * 255).astype(np.uint8)

    if render_type == "mesh":
        # mesh object
        geom = vis.wireframe_bbox(
            voxel_data["voxel_corners"][:, 0, :],
            voxel_data["voxel_corners"][:, 1, :],
            color=colors_uint8,
            solid=True,
        )
        trimesh_object = trimesh.Trimesh(
            vertices=np.asarray(geom.vertices),
            faces=np.asarray(geom.triangles),
            vertex_colors=np.asarray(geom.vertex_colors),
        )
        render_trimesh(trimesh_object, port=port)

    elif render_type == "pc":
        # point cloud
        points = voxel_data["voxel_centers"]
        render_point_cloud(points, colors_uint8, port=port)


if __name__ == "__main__":
    main()
