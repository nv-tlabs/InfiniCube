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

import numpy as np
import viser
import viser.transforms as tf
from termcolor import colored


def build_camera_to_world(camera):
    cam_wxyz = camera.wxyz
    cam_pos = camera.position

    R_c2w = tf.SO3(np.asarray(cam_wxyz)).as_matrix()  # camera to world
    T_c2w = cam_pos  # camera to world
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R_c2w
    pose_matrix[:3, 3] = T_c2w

    return pose_matrix


def set_kill_key_button(server, gui_kill_button):
    @gui_kill_button.on_click
    def _(event: viser.GuiEvent) -> None:
        print(f"{colored('Killing the sample.', 'red', attrs=['bold'])}")
        setattr(server, "alive", False)
        time.sleep(0.3)

        client = event.client
        assert client is not None

        client.add_notification(
            title="Finished",
            body="The recording is finished.",
            auto_close_seconds=5.0,
        )


def set_record_key_button(server, gui_record_keyframe_button):
    server.__setattr__("key_wxyzs", [])
    server.__setattr__("key_positions", [])

    @gui_record_keyframe_button.on_click
    def _(_) -> None:
        assert len(server.get_clients()) == 1, "Only support one client now."
        for id, client in server.get_clients().items():
            key_wxyz = client.camera.wxyz
            key_position = client.camera.position
            server.key_wxyzs.append(key_wxyz.tolist())
            server.key_positions.append(key_position.tolist())
            print(f"{len(server.key_wxyzs)} keyframes recorded!")
        time.sleep(0.3)


def set_recording_gui(server):
    gui_record_keyframe_button = server.gui.add_button(
        "Set Keyframe", hint="Press this button to mark as keyframe."
    )
    gui_record_checkbox = server.gui.add_checkbox("Recording", initial_value=False)
    gui_fovy_modifier = server.gui.add_slider(
        "Vertical FoV (degree)", min=25, max=120, step=0.5, initial_value=35
    )
    gui_render_dropdown = server.gui.add_dropdown(
        "Render Engine",
        [
            "ThreeJS (Viser)",
            "3DGS (gsplat, offscreen)",
            "Blender (PyCG, offscreen)",
            "Filament (PyCG, offscreen)",
        ],
        initial_value="ThreeJS (Viser)",
    )

    set_record_key_button(server, gui_record_keyframe_button)
    server.__setattr__("recording_status", gui_record_checkbox)
    server.__setattr__("fovy_modifier", gui_fovy_modifier)
    server.__setattr__("render_engine", gui_render_dropdown)
    server.__setattr__("object_handlers", {"pointcloud": [], "mesh": []})


def set_kill_gui(server):
    server.__setattr__("alive", True)
    gui_kill_button = server.gui.add_button(
        "Kill", hint="Press this button to kill this sample"
    )
    set_kill_key_button(server, gui_kill_button)
