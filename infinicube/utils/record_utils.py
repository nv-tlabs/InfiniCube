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

import tempfile
import time

import imageio.v3 as imageio
import numpy as np
import viser.transforms as tf
from pycg import animation, render, vis
from pycg.animation import InterpType
from pycg.isometry import Isometry
from termcolor import colored
from tqdm import tqdm

from infinicube.utils.fileio_utils import write_video_file
from infinicube.utils.interpolate_utils import inter_poses, inter_two_poses_uniform


def record_plugin(
    server,
    client,
    render_engine_func=None,
    render_engine_kwargs=None,
    time2sleep=1 / 60,
):
    """
    Note 2: we need to have a camera.on_update callback for gaussian-splatting related recording,
            becuase we implement it by setting the background image of the client.
            Setting this camera.on_update function directly to the real-time rending will cause async issues,
            but here this problem is alleivated by our mannual update, with render_engine_func and render_engine_kwargs.

    Once we have everything set up, we can use this function to record a video of the gaussian-splatting process.
        1. get key frames from input()
        2. interpolate camera poses using inter_poses()
        3. get client, change client.camera.wxyz and client.camera.position, this will take effect in the viewer.
        4. get render result from client.get_render()
        5. save video
    """
    # See Note 1
    if not server.recording_status.value:
        return

    print(
        f"{colored('HINT:', 'green')}",
        "If you want to exit recording mode, uncheck the `Recording` checkbox and enter exit when being asked for the output file.",
    )

    ### get interpolated poses ###
    interpolated_poses = None

    if len(server.key_wxyzs) == 0:
        interpolated_pose_file = input(
            f"Enter interpolated pose {colored('.npy', 'green')} file if you have, or just enter nothing: "
        )
        if interpolated_pose_file.strip("'").strip('"').endswith(".npy"):
            interpolated_poses = np.load(interpolated_pose_file.strip("'").strip('"'))
            print(
                f"{colored('Interpolated poses loaded from file.', 'cyan', attrs=['bold'])}"
            )

    if interpolated_poses is None:
        if len(server.key_wxyzs) == 0:
            key_wxyz = input("Enter key poses (wxyz): ")
            key_position = input("Enter key poses (position): ")
        else:
            print(
                f"{colored('Loading key poses from the GUI.', 'cyan', attrs=['bold'])}"
            )
            print(
                f"{colored('Key frame wxyz:', 'cyan', attrs=['bold'])}",
                server.key_wxyzs,
            )
            print(
                f"{colored('Key frame position:', 'cyan', attrs=['bold'])}",
                server.key_positions,
            )
            key_wxyz = str(server.key_wxyzs)
            key_position = str(server.key_positions)

        if not key_wxyz or not key_position:
            print(f"{colored('Key poses not specified.', 'red')}")
            return

        key_wxyz = np.array(eval(key_wxyz))
        key_position = np.array(eval(key_position))

        if len(key_wxyz) == 1:
            print("only 1 pose is provided, save image directly.")
            interpolated_wxyz, interpolated_position = key_wxyz, key_position
        elif len(key_wxyz) == 2:
            print("only 2 poses are provided, using uniform interpolation.")
            interpolate_frame_num = (
                input("Enter the number of interpolated frames (default 120): ")
                or "120"
            )
            interpolate_frame_num = int(interpolate_frame_num)
            interpolated_wxyz, interpolated_position = inter_two_poses_uniform(
                key_wxyz, key_position, interpolate_frame_num
            )
        else:
            interpolate_frame_num = (
                input("Enter the number of interpolated frames (default 120): ")
                or "120"
            )
            interpolate_frame_num = int(interpolate_frame_num)
            interpolated_wxyz, interpolated_position = inter_poses(
                key_wxyz, key_position, interpolate_frame_num
            )

        interpolated_poses = tf.SE3.from_rotation_and_translation(
            tf.SO3(interpolated_wxyz), interpolated_position
        ).as_matrix()

    output_file = input(
        "Enter the output video path (ends with .mp4 or .png). Enter exit to reset the keyframes: "
    )
    if output_file is None:
        return

    if output_file == "exit":
        server.__setattr__("key_wxyzs", [])
        server.__setattr__("key_positions", [])
        return

    render_height = input("Enter the render height (default 1280): ") or "1280"
    render_width = input("Enter the render width (default 1920): ") or "1920"
    render_vfov_deg = (
        input(
            f"Enter the render vfov in degree (default {np.rad2deg(client.camera.fov)}): "
        )
        or None
    )

    output_file = output_file.strip('"').strip("'")
    if not (output_file.endswith(".mp4") or output_file.endswith(".png")):
        print(f"{colored('Output file has a bad suffix, try again.', 'red')}")
        return

    render_height = eval(render_height)
    render_width = eval(render_width)

    if render_vfov_deg is None:
        render_vfov_deg = np.rad2deg(client.camera.fov)
    else:
        render_vfov_deg = eval(render_vfov_deg)

    print(
        f"{colored('Camera vfov in degree:', 'cyan', attrs=['bold'])}", render_vfov_deg
    )

    # calculate fx, fy, cx, cy according to the vfov, height, width, do not depend on aspect ratio (we specify fx=fy)
    camera_vfov_rad = np.deg2rad(render_vfov_deg)
    fy = render_height / 2 / np.tan(np.deg2rad(render_vfov_deg) / 2)
    fx = fy
    cy = render_height / 2
    cx = render_width / 2
    camera_hfov_rad = 2 * np.arctan(render_width / 2 / fx)

    np.save(
        output_file.replace(".mp4", ".npy").replace(".png", ".npy"), interpolated_poses
    )

    render_result = []
    if server.render_engine.value == "ThreeJS (Viser)":
        for i in tqdm(range(len(interpolated_poses))):
            client.camera.wxyz = tf.SO3.from_matrix(interpolated_poses[i][:3, :3]).wxyz
            client.camera.position = interpolated_poses[i][:3, 3]
            client.camera.fov = camera_vfov_rad

            render_result_cur_frame = client.camera.get_render(
                height=render_height, width=render_width, transport_format="png"
            )
            time.sleep(time2sleep)
            render_result_cur_frame = render_result_cur_frame / 255.0
            rgb = render_result_cur_frame[:, :, :3]  # 提取 RGB 通道
            alpha = render_result_cur_frame[:, :, 3:]  # 提取 Alpha 通道
            white_background = np.ones_like(rgb)  # 创建一个全白的背景
            render_result_cur_frame = rgb * alpha + white_background * (1 - alpha)
            render_result_cur_frame = (render_result_cur_frame * 255).astype(np.uint8)

            render_result.append(render_result_cur_frame)

    elif server.render_engine.value == "3DGS (gsplat, offscreen)":
        print(
            f"{colored('Improve this, no need to set background. and need to set vfov', 'red', attrs=['bold'])}"
        )
        for i in tqdm(range(len(interpolated_poses))):
            # support dynamic reconstruction in infinicube
            if "timestep" in render_engine_kwargs:
                render_engine_kwargs["timestep"] = i

            render_engine_kwargs["camera_to_world"] = interpolated_poses[i]
            render_engine_kwargs["height"] = render_height
            render_engine_kwargs["width"] = render_width
            render_engine_kwargs["vfov"] = camera_vfov_rad
            render_engine_kwargs["hfov"] = camera_hfov_rad

            render_result_cur_frame = render_engine_func(**render_engine_kwargs)
            time.sleep(time2sleep)

            render_result.append(render_result_cur_frame)

    # pycg render engine, offscreen
    else:
        scene = render.Scene(up_axis="+Z")
        scene.camera_intrinsic = render.CameraIntrinsic(
            render_width, render_height, fx, fy, cx, cy
        )

        cam_animator = animation.FreePoseAnimator(InterpType.BEZIER)
        for t in range(len(interpolated_poses)):
            cam_animator.set_keyframe(
                t, Isometry.from_matrix(interpolated_poses[t], ortho=True)
            )
        scene.animator.set_relative_camera(cam_animator)

        # convert to pycg object types
        # 1) handler point cloud -> pycg
        for pointcloud_handler in server.object_handlers["pointcloud"]:
            scene.add_object(
                vis.pointcloud(
                    pointcloud_handler.points, color=pointcloud_handler.colors
                ),
                pointcloud_handler.name,
            )

        # 2) handler mesh (GLB binary) -> pycg
        if "mesh" in server.object_handlers:
            try:
                import open3d as o3d
            except ImportError:
                import open3d_pycg as o3d

        for mesh_handler in server.object_handlers["mesh"]:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
                temp_file.write(
                    mesh_handler.glb_data
                )  # A binary payload containing the GLB data.
                temp_file_path = temp_file.name
            o3d_mesh = o3d.io.read_triangle_mesh(temp_file_path)
            vertices = np.asarray(o3d_mesh.vertices)
            triangles = np.asarray(o3d_mesh.triangles)
            pycg_mesh = vis.mesh(
                vertices, triangles, color=np.asarray(o3d_mesh.vertex_colors)
            )

            scene.add_object(pycg_mesh, mesh_handler.name)

        scene.point_size = 10
        want_shadow = input("Do you want shadow? (y/n, default y): ")
        if want_shadow == "n":
            pass
        else:
            scene.camera_pose = scene.animator.get_relative_camera().get_value(0)
            render.ThemeDiffuseShadow(sun_energy=4).apply_to(scene)
            scene.remove_object("auto_plane", non_exist_ok=True)

        if server.render_engine.value == "Blender (PyCG, offscreen)":
            for t, img in tqdm(scene.render_blender_animation()):
                render_result.append(img)

        elif server.render_engine.value == "Filament (PyCG, offscreen)":
            for t, img in tqdm(scene.render_filament_animation()):
                render_result.append(img)

    # save video or image
    if len(render_result) == 1:
        print(f"{colored('Only 1 frame, save as image.', 'yellow')}")
        imageio.imwrite(output_file.replace(".mp4", ".png"), render_result[0])
    else:
        write_video_file(render_result, output_file, use_jiahui_params=True)
