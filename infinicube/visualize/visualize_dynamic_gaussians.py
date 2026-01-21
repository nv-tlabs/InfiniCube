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
Visualize dynamic gaussians scene

python infinicube/visualize/visualize_dynamic_gaussians.py \
    -p visualization/infinicube_inference/gaussian_scene_generation/trajectory_pose_sample_1frame/13679757109245957439_4167_170_4187_170/
"""

import pickle
import time
from copy import deepcopy
from pathlib import Path

import click
import numpy as np
import torch
import viser

from infinicube import get_sample
from infinicube.utils.gaussian_io_utils import load_gaussian
from infinicube.utils.gaussian_render_utils import (
    RGB2SH,
    client_rendering_and_set_background,
    standard_3dgs_rendering_func,
    transform2tensor,
)
from infinicube.utils.record_utils import record_plugin
from infinicube.utils.sky_utils import read_skybox
from infinicube.utils.viser_gui_utils import set_kill_gui, set_recording_gui
from infinicube.voxelgen.utils.extrap_util import transform_points


def warp_dynamic_3dgs_rendering_func(
    camera_to_world,
    height,
    width,
    vfov,
    hfov,
    static_gaussians,
    timestep_to_object_gaussians,
    timestep,
    scale_modifier=1,
    skybox_dict=None,
):
    merged_gaussians = compose_bg_and_object_gaussians(
        static_gaussians, timestep_to_object_gaussians[timestep]
    )
    return standard_3dgs_rendering_func(
        camera_to_world,
        height,
        width,
        vfov,
        hfov,
        merged_gaussians,
        scale_modifier,
        skybox_dict,
    )


def compose_bg_and_object_gaussians(
    static_gaussians_dict, merged_object_gaussians_dict
):
    """
    static_gaussians_dict: dict, standard gaussian representation used in GS_utils, including
        xyz, opacity, scaling, rotation, feature

    merged_object_gaussians_dict: including
        xyz, opacity, scaling, rotation, feature

    we just cat them together
    """
    merged_gaussians = deepcopy(static_gaussians_dict)
    merged_gaussians["xyz"] = torch.cat(
        [merged_gaussians["xyz"], merged_object_gaussians_dict["xyz"]], dim=0
    )
    merged_gaussians["opacity"] = torch.cat(
        [merged_gaussians["opacity"], merged_object_gaussians_dict["opacity"]], dim=0
    )
    merged_gaussians["scaling"] = torch.cat(
        [merged_gaussians["scaling"], merged_object_gaussians_dict["scaling"]], dim=0
    )
    merged_gaussians["rotation"] = torch.cat(
        [merged_gaussians["rotation"], merged_object_gaussians_dict["rotation"]], dim=0
    )
    merged_gaussians["features"] = torch.cat(
        [merged_gaussians["features"], merged_object_gaussians_dict["features"]], dim=0
    )
    return merged_gaussians


def add_feature_to_gaussian_dict(gs_dict):
    if "features" not in gs_dict:
        gs_dict["features"] = RGB2SH(gs_dict["rgbs"]).reshape(-1, 1, 3)  # (N, 1, 3)
        gs_dict["sh_degree"] = 0
    return gs_dict


def visualize_dynamic_scene(
    static_gaussians,
    object_gaussians,
    all_dynamic_object_info,
    skybox_dict,
    switch_time,
):
    """
    static_gaussians are standard gaussian representation used in GS_utils, including
        xyz, opacity, scaling, rotation, feature, sh_degree

    object_gaussians are a dict:
        key: gid
        value: dict with xyz, opacity, scaling, rotation, feature, sh_degree

    dynamic_object_info are list of dict, each dict contains:
        - gid -> object_to_world: 4x4 matrix

    skybox_dict is the skybox information
    """
    server = viser.ViserServer()
    server.configure_theme(dark_mode=True)
    server.world_axes.visible = True
    set_recording_gui(server)
    set_kill_gui(server)
    height = 1280
    width = 1920
    frame_num = len(
        [i for i in all_dynamic_object_info.keys() if ".dynamic_object_info.json" in i]
    )

    with server.gui.add_folder("Control", expand_by_default=True):
        gui_scale_modifier = server.gui.add_slider(
            "Scale", min=0, max=2.0, step=0.05, initial_value=1.0
        )
        gui_skybox_checkbox = server.gui.add_checkbox(
            "Skybox",
            initial_value=False,
        )
        gui_skymask_checkbox = server.gui.add_checkbox(
            "Skymask",
            initial_value=False,
        )
        gui_select_frame_button = server.gui.add_slider(
            "Select Frame",
            hint="Press this button to select a frame.",
            min=0,
            max=frame_num,
            step=1,
            initial_value=0,
        )

    timestep_to_object_gaussians = {}

    for i in range(frame_num):
        merged_object_gaussians = {
            "xyz": torch.zeros(0, *static_gaussians["xyz"].shape[1:]).cuda(),
            "opacity": torch.zeros(0, *static_gaussians["opacity"].shape[1:]).cuda(),
            "scaling": torch.zeros(0, *static_gaussians["scaling"].shape[1:]).cuda(),
            "rotation": torch.zeros(0, *static_gaussians["rotation"].shape[1:]).cuda(),
            "features": torch.zeros(0, *static_gaussians["features"].shape[1:]).cuda(),
            "sh_degree": static_gaussians["sh_degree"],
        }

        dynamic_cuboids = all_dynamic_object_info[f"{i:06d}.dynamic_object_info.json"]
        for gid, object_info in dynamic_cuboids.items():
            # if object gauassian number is too small, we skip it
            if object_gaussians[gid]["xyz"].shape[0] < 2000:
                print(
                    f"Object {gid} has too few gaussians {object_gaussians[gid]['xyz'].shape[0]} in frame {i}, skip it"
                )
                continue
            xyz_current = transform_points(
                object_gaussians[gid]["xyz"],
                torch.Tensor(object_info["object_to_world"]).cuda(),
            )
            merged_object_gaussians["xyz"] = torch.cat(
                [merged_object_gaussians["xyz"], xyz_current], dim=0
            )
            merged_object_gaussians["opacity"] = torch.cat(
                [merged_object_gaussians["opacity"], object_gaussians[gid]["opacity"]],
                dim=0,
            )
            merged_object_gaussians["scaling"] = torch.cat(
                [merged_object_gaussians["scaling"], object_gaussians[gid]["scaling"]],
                dim=0,
            )
            merged_object_gaussians["rotation"] = torch.cat(
                [
                    merged_object_gaussians["rotation"],
                    object_gaussians[gid]["rotation"],
                ],
                dim=0,
            )
            merged_object_gaussians["features"] = torch.cat(
                [
                    merged_object_gaussians["features"],
                    object_gaussians[gid]["features"],
                ],
                dim=0,
            )
            assert (
                merged_object_gaussians["sh_degree"]
                == object_gaussians[gid]["sh_degree"]
            )

        timestep_to_object_gaussians[i] = merged_object_gaussians

    cnt = 0
    while True:
        if gui_select_frame_button.value != 0:
            cnt = int(gui_select_frame_button.value - 1)

        start_time = time.time()
        while True:
            # break to next frame
            if time.time() - start_time > switch_time:
                break

            clients = server.get_clients()
            for id, client in clients.items():
                vfov = np.deg2rad(server.fovy_modifier.value)
                scale_modifier = gui_scale_modifier.value
                use_skybox = gui_skybox_checkbox.value
                apply_skymask = gui_skymask_checkbox.value
                skybox_dict["apply_skymask"] = apply_skymask

                warp_dynamic_rendering_kwargs = {
                    "height": height,
                    "width": width,
                    "static_gaussians": static_gaussians,
                    "timestep_to_object_gaussians": timestep_to_object_gaussians,
                    "timestep": cnt,
                    "vfov": vfov,
                    "scale_modifier": scale_modifier,
                    "skybox_dict": skybox_dict if use_skybox else None,
                }

                client_rendering_and_set_background(
                    client,
                    warp_dynamic_3dgs_rendering_func,
                    warp_dynamic_rendering_kwargs,
                )
                # we can always change the full_rendering_kwargs when offscreen rendering
                record_plugin(
                    server,
                    client,
                    warp_dynamic_3dgs_rendering_func,
                    warp_dynamic_rendering_kwargs,
                    0.3,
                )

            time.sleep(1 / 20)

        cnt += 1
        cnt = cnt % frame_num


@click.command()
@click.option(
    "--scene_folder",
    "-p",
    type=str,
    required=True,
    help="Path to the gaussian scene folder",
)
@click.option(
    "--switch_time",
    type=float,
    default=0.25,
    help="Time to switch between different timestamp",
)
def main(scene_folder, switch_time):
    exp_folder = Path(scene_folder)
    static_gaussians_path = exp_folder / "decoded_gs_static.pkl"
    static_gaussians = load_gaussian(static_gaussians_path.as_posix())
    static_gaussians = transform2tensor(static_gaussians)

    skybox_dict = read_skybox(static_gaussians_path.as_posix())

    object_gaussian_path = exp_folder / "decoded_gs_object.pkl"
    if object_gaussian_path.exists():
        with open(object_gaussian_path, "rb") as f:
            object_gaussians = pickle.load(f)
        object_gaussians = {
            k: add_feature_to_gaussian_dict(v) for k, v in object_gaussians.items()
        }
        object_gaussians = {k: transform2tensor(v) for k, v in object_gaussians.items()}

        dynamic_object_info_path = exp_folder / "dynamic_object_info.tar"
        dynamic_object_info_sample = get_sample(dynamic_object_info_path)
    else:
        object_gaussians = {}
        dynamic_object_info_sample = {"000000.dynamic_object_info.json": {}}

    visualize_dynamic_scene(
        static_gaussians,
        object_gaussians,
        dynamic_object_info_sample,
        skybox_dict,
        switch_time,
    )


if __name__ == "__main__":
    main()
