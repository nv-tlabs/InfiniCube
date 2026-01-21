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
Unified Guidance Buffer Generation
Support two generation modes:
1. trajectory: Follow the original dataset trajectory
2. blocks: Generate by blocks with custom camera trajectories (using viser GUI)

Example usage:
# Trajectory mode, sample every 1 frames from the original trajectory
python infinicube/inference/guidance_buffer_generation.py \
    --mode trajectory \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --make_dynamic --offset_unit frame --offset 1

# Trajectory mode, sample every 2 meters from the original trajectory
python infinicube/inference/guidance_buffer_generation.py \
    --mode trajectory \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --make_dynamic --offset 2

# Blocks mode, we don't add dynamic objects
python infinicube/inference/guidance_buffer_generation.py \
    --mode blocks \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/blocks 
"""

import shutil
import time
from pathlib import Path

import click
import numpy as np
import torch
import viser
import viser.transforms as tf
from loguru import logger
from termcolor import colored, cprint

from infinicube import get_sample
from infinicube.camera.base import flu_to_opencv, opencv_to_flu
from infinicube.camera.pinhole import PinholeCamera
from infinicube.data_process.waymo_utils import (
    distribute_object_info_across_all_frames,
    imageencoder_imageio_png,
    keep_car_only_in_object_info,
)
from infinicube.utils.buffer_utils import (
    generate_coordinate_buffer_from_memory_global_norm,
)
from infinicube.utils.depth_utils import vis_depth
from infinicube.utils.fileio_utils import write_video_file
from infinicube.utils.fvdb_utils import generate_infinicube_buffer_from_fvdb_grid
from infinicube.utils.interpolate_utils import inter_poses, inter_two_poses_uniform
from infinicube.utils.semantic_utils import (
    generate_rgb_semantic_buffer,
    semantic_to_color,
)
from infinicube.utils.viser_gui_utils import set_kill_key_button
from infinicube.utils.wds_utils import write_to_tar
from infinicube.voxelgen.utils.extrap_util import get_key_frames_indices

RESOLUTION_ANNO = {
    "480p": (480, 832),
    "720p": (720, 1280),
}

# ============================================================================
# Viser GUI Utility Functions for Blocks Mode
# ============================================================================


def set_record_key_button(server, gui_record_keyframe_button):
    """Set up keyframe recording button."""
    gui_record_keyframe_button.__setattr__("key_wxyzs", [])
    gui_record_keyframe_button.__setattr__("key_positions", [])

    @gui_record_keyframe_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert len(server.get_clients()) == 1, "Only support one client now."
        for id, client in server.get_clients().items():
            key_wxyz = client.camera.wxyz
            key_position = client.camera.position
            gui_record_keyframe_button.key_wxyzs.append(key_wxyz.tolist())
            gui_record_keyframe_button.key_positions.append(key_position.tolist())
            print(f"{len(gui_record_keyframe_button.key_wxyzs)} keyframes recorded!")

            if not hasattr(gui_record_keyframe_button, "first_frame_wxyz"):
                gui_record_keyframe_button.__setattr__("first_frame_wxyz", key_wxyz)
                gui_record_keyframe_button.__setattr__(
                    "first_frame_position", key_position
                )
                cprint("First frame recorded!", "red", attrs=["bold"])

        if hasattr(gui_record_keyframe_button, "notification_handle"):
            gui_record_keyframe_button.notification_handle.remove()

        gui_record_keyframe_button.__setattr__(
            "notification_handle",
            client.add_notification(
                title=f"{len(gui_record_keyframe_button.key_wxyzs)} keyframes are recorded",
                body=f"Their location are {gui_record_keyframe_button.key_positions}. At least 2 keyframes are required for interpolation the trajectory.",
                loading=False,
                with_close_button=True,
                auto_close=False,
                color="yellow",
            ),
        )

        time.sleep(0.3)


def set_save_current_pass_button(
    server, gui_save_current_pass_button, gui_record_keyframe_button
):
    """Set up pass saving button."""
    gui_save_current_pass_button.__setattr__("key_wxyzs_multi_pass", [])
    gui_save_current_pass_button.__setattr__("key_positions_multi_pass", [])

    @gui_save_current_pass_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert len(server.get_clients()) == 1, "Only support one client now."

        key_wxyzs = gui_record_keyframe_button.key_wxyzs
        key_positions = gui_record_keyframe_button.key_positions

        gui_save_current_pass_button.key_wxyzs_multi_pass.append(key_wxyzs)
        gui_save_current_pass_button.key_positions_multi_pass.append(key_positions)

        gui_record_keyframe_button.__setattr__("key_wxyzs", [])
        gui_record_keyframe_button.__setattr__("key_positions", [])

        cprint(
            f"Pass saved with {len(gui_save_current_pass_button.key_wxyzs_multi_pass[-1])} keyframes!",
            "green",
        )
        print("We will empty the keyframes for the next pass.")

        client = event.client
        assert client is not None
        client.add_notification(
            title="Pass saved",
            body=f"The pass with {len(gui_save_current_pass_button.key_wxyzs_multi_pass[-1])} keyframes is saved.",
            auto_close_seconds=5.0,
        )


def set_force_reset_pass_button(
    server, gui_force_reset_pass_button, gui_record_keyframe_button
):
    """Set up force reset button."""

    @gui_force_reset_pass_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert len(server.get_clients()) == 1, "Only support one client now."
        camera = server.get_clients()[0].camera
        camera.wxyz = gui_record_keyframe_button.first_frame_wxyz
        camera.position = gui_record_keyframe_button.first_frame_position

        gui_record_keyframe_button.key_wxyzs.append(
            gui_record_keyframe_button.first_frame_wxyz
        )
        gui_record_keyframe_button.key_positions.append(
            gui_record_keyframe_button.first_frame_position
        )

        cprint("Force to reset to the first keyframe!", "green")
        cprint(
            "automatically add the first keyframe to the keyframes.",
            "green",
            attrs=["bold"],
        )
        print(f"{len(gui_record_keyframe_button.key_wxyzs)} keyframes recorded!")

        client = event.client
        assert client is not None
        client.add_notification(
            title="Reset to first frame",
            body=f"The camera is reset to the first frame position, and the first frame is automatically added as a keyframe (total {len(gui_record_keyframe_button.key_wxyzs)} keyframes).",
            auto_close_seconds=4.0,
        )

        time.sleep(0.3)


def set_forward_button(server, gui_cropper_anchor_buttons):
    """Set up forward movement button."""

    @gui_cropper_anchor_buttons.on_click
    def _(event: viser.GuiEvent) -> None:
        assert len(server.get_clients()) == 1, "Only support one client now."
        for id, client in server.get_clients().items():
            camera_position = client.camera.position
            camera_wxyz = client.camera.wxyz
            camera_pose = tf.SE3.from_rotation_and_translation(
                tf.SO3(camera_wxyz), camera_position
            ).as_matrix()
            camera_front_direction = camera_pose[:3, 2]  # z axis direction

            new_camera_position = camera_position + 20 * camera_front_direction
            client.camera.position = new_camera_position

            client = event.client
            assert client is not None
            client.add_notification(
                title="Forward 20m",
                body="The camera is moved forward 20m.",
                auto_close_seconds=2.0,
            )


def render_point_cloud_with_viser(
    points, colors, point_size=0.1, name="/simple_pc", interpolate_frame_num=150
):
    """
    Visualize point cloud with viser and record camera trajectories.

    Args:
        points: np.ndarray, [N, 3]
        colors: np.ndarray, [3,] or [N, 3]
        point_size: float
        name: str
        interpolate_frame_num: int

    Returns:
        interpolated_poses_multi_pass: list of np.ndarray
    """
    server = viser.ViserServer()
    setattr(server, "alive", True)

    gui_record_keyframe_button = server.gui.add_button(
        "Record Keyframe", hint="Press this button to mark as keyframe."
    )
    gui_save_current_pass_button = server.gui.add_button(
        "Save Current Pass", hint="Press this button to save the current pass"
    )
    gui_force_reset_pass_button = server.gui.add_button(
        "Force Reset to first frame position",
        hint="Press this button to reset the camera to the first frame",
    )
    gui_forward_20m_button = server.gui.add_button(
        "Forward 20m", hint="Press this button to moving forward 20m"
    )
    gui_always_zero_heigth_checkbox = server.add_gui_checkbox(
        "Always Zero Height",
        hint="Press this button to always set the height to zero",
        initial_value=False,
    )
    gui_kill_button = server.gui.add_button(
        "Finish", hint="Press this button to finish the recording"
    )

    set_record_key_button(server, gui_record_keyframe_button)
    set_save_current_pass_button(
        server, gui_save_current_pass_button, gui_record_keyframe_button
    )
    set_force_reset_pass_button(
        server, gui_force_reset_pass_button, gui_record_keyframe_button
    )
    set_forward_button(server, gui_forward_20m_button)
    set_kill_key_button(server, gui_kill_button)

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
            camera.fov = 35 / 180 * np.pi
            if gui_always_zero_heigth_checkbox.value:
                camera.position = np.array([camera.position[0], camera.position[1], 0])

        if not server.alive:
            server.stop()
            break

        time.sleep(1)

    key_wxyzs_multi_pass = gui_save_current_pass_button.key_wxyzs_multi_pass
    key_positions_multi_pass = gui_save_current_pass_button.key_positions_multi_pass
    print(f"Total {len(key_wxyzs_multi_pass)} passes are recorded.")

    interpolated_poses_multi_pass = []
    for i, (key_wxyzs, key_positions) in enumerate(
        zip(key_wxyzs_multi_pass, key_positions_multi_pass)
    ):
        assert len(key_wxyzs) == len(key_positions), (
            "The number of keyframes should be the same"
        )
        assert len(key_wxyzs) > 1, "Less than 2 keyframe is recorded"

        # interpolate the keyframes, convert to 4x4 transformation matrix. Note that in viser, camera convention is already opencv
        key_positions = np.array(key_positions)
        key_wxyzs = np.array(key_wxyzs)

        # Use fixed interpolate frame number
        using_interpolate_frame_num = interpolate_frame_num

        print(
            f"Interpolating {len(key_positions)} keyframes with {colored(using_interpolate_frame_num, 'green', attrs=['bold'])} frames"
        )

        if len(key_positions) == 2:
            # repeat the first keyframe
            print(
                "Only two keyframes are recorded. We will use uniform interpolation in L2 space"
            )
            interpolated_wxyz, interpolated_position = inter_two_poses_uniform(
                key_wxyzs, key_positions, using_interpolate_frame_num
            )
            interpolated_poses = tf.SE3.from_rotation_and_translation(
                tf.SO3(interpolated_wxyz), interpolated_position
            ).as_matrix()

        else:
            print(
                "More than two keyframes are recorded. We will use general interpolation"
            )
            interpolated_wxyz, interpolated_position = inter_poses(
                key_wxyzs, key_positions, using_interpolate_frame_num, sigma=1
            )
            interpolated_poses = tf.SE3.from_rotation_and_translation(
                tf.SO3(interpolated_wxyz), interpolated_position
            ).as_matrix()

        interpolated_poses_multi_pass.append(interpolated_poses.astype(np.float32))

    return interpolated_poses_multi_pass


# ============================================================================
# Data Loading Functions
# ============================================================================


def setup_camera_model(clip, data_root, resolution):
    """
    Setup and rescale camera model based on target resolution.

    Args:
        clip: str
        data_root: str or Path
        resolution: str, e.g., '480p' or '720p'

    Returns:
        camera_model: PinholeCamera
    """
    target_hw = RESOLUTION_ANNO[resolution]
    intrinsic_tensor = get_intrinsic(clip, data_root)
    camera_model = PinholeCamera.from_numpy(intrinsic_tensor)
    camera_model.rescale(
        ratio_h=target_hw[0] / intrinsic_tensor[5],
        ratio_w=target_hw[1] / intrinsic_tensor[4],
    )
    return camera_model


def get_pose_and_transform(clip, data_root="data", ego_trajectory_key="pose.front.npy"):
    """
    Convert camera poses into a new coordinate, where the first-camera's FLU is the origin.

    Args:
        clip: str
        data_root: str
        ego_trajectory_key: str

    Returns:
        camera_pose_opencv_relative: torch.Tensor, (N, 4, 4)
        camera_pose_flu_0: torch.Tensor, (4, 4)
    """
    data_root = Path(data_root).resolve()
    pose_data = get_sample(data_root / f"pose/{clip}.tar")
    pose = {k: torch.tensor(v) for k, v in pose_data.items() if ego_trajectory_key in k}

    # sort and concatenate, opencv convention
    camera_trajectory = torch.stack([v for k, v in sorted(pose.items())], axis=0)

    # transform the ego trajectory, set the first frame as the origin
    camera_trajectory_flu = opencv_to_flu(camera_trajectory)
    camera_pose_flu_0 = camera_trajectory_flu[0]

    camera_pose_flu_relative = torch.einsum(
        "ij, bjk -> bik", camera_trajectory_flu[0].inverse(), camera_trajectory_flu
    )

    camera_pose_opencv_relative = flu_to_opencv(camera_pose_flu_relative)

    return camera_pose_opencv_relative, camera_pose_flu_0


def get_intrinsic(clip, data_root="data", intrinsic_key="intrinsic.front.npy"):
    """
    Load camera intrinsic.

    Args:
        clip: str
        data_root: str
        intrinsic_key: str

    Returns:
        intrinsic: np.ndarray, (6,), [fx fy cx cy w h]
    """
    data_root = Path(data_root).resolve()
    intrinsic_data = get_sample(data_root / f"intrinsic/{clip}.tar")
    intrinsic = intrinsic_data[intrinsic_key]

    return intrinsic


def load_voxel(voxel_root, clip, extrap_voxel_time=None):
    """
    Load voxel from disk.

    Args:
        voxel_root: str
        clip: str
        extrap_voxel_time: int or None

    Returns:
        fvdb_grid: GridBatch
        fvdb_semantic: torch.Tensor
        voxel_path: Path for selected voxel file
    """
    voxel_root_p = Path(voxel_root).resolve()

    if extrap_voxel_time is None:
        voxel_files = list((voxel_root_p / clip).glob("*.pt"))
        voxel_files = sorted(voxel_files, key=lambda x: int(x.stem))
        voxel_path = voxel_files[-1]
        extrap_voxel_time = int(voxel_path.stem)
    else:
        voxel_path = voxel_root_p / clip / f"{extrap_voxel_time}.pt"

    if not voxel_path.exists():
        raise FileNotFoundError(f"Voxel {voxel_path} does not exist")

    voxel = torch.load(voxel_path)
    fvdb_grid = voxel["points"].cuda()
    fvdb_semantic = voxel["semantics"].long().cuda()

    return fvdb_grid, fvdb_semantic, voxel_path


def get_object_info_key_indices(clip, data_root, num_required_frames):
    """
    Generate key pose indices for object info, handling cases where
    more frames are needed than available.

    Args:
        clip: str
        data_root: str or Path
        num_required_frames: int, number of frames needed

    Returns:
        key_pose_indices: list of int
    """
    data_root_p = Path(data_root).resolve()
    static_object_info_tar = data_root_p / "static_object_info" / f"{clip}.tar"
    static_temp = get_sample(static_object_info_tar)
    num_available_frames = len(
        [k for k in static_temp.keys() if ".static_object_info.json" in k]
    )

    if num_required_frames > num_available_frames:
        print(
            f"the required frames {num_required_frames} is more than the object info {num_available_frames}"
        )
        cprint(
            f"we will repeat the object info for the last {num_required_frames - num_available_frames} frames.",
            "red",
            attrs=["bold"],
        )
        key_pose_indices = list(range(num_available_frames)) + [
            num_available_frames - 1
        ] * (num_required_frames - num_available_frames)
    else:
        if num_required_frames < num_available_frames:
            cprint(
                f"we use the first {num_required_frames} frames from object info",
                "green",
                attrs=["bold"],
            )
        key_pose_indices = list(range(num_required_frames))

    return key_pose_indices


def load_object_info(
    clip, data_root, world_to_grid, key_pose_indices=None, make_dynamic=False
):
    """
    Load static and dynamic object info from separate data sources.

    Args:
        clip: str
        data_root: str or Path
        world_to_grid: np.ndarray, (4, 4),
            waymo world coordinate to current grid coordinate transformation matrix
            we use this to transform the object info to the grid coordinate
        key_pose_indices: list of int or None, if None, load all frames
        make_dynamic: bool, make dynamic object info dict contains empty for each frame if true

    Returns:
        static_object_info: dict
        dynamic_object_info: dict
    """
    data_root_p = Path(data_root).resolve()

    static_object_info_tar = data_root_p / "static_object_info" / f"{clip}.tar"
    dynamic_object_info_tar = data_root_p / "dynamic_object_info" / f"{clip}.tar"

    static_object_info_all = get_sample(static_object_info_tar)
    dynamic_object_info_all = get_sample(dynamic_object_info_tar)

    static_object_info_all = keep_car_only_in_object_info(static_object_info_all)
    dynamic_object_info_all = keep_car_only_in_object_info(dynamic_object_info_all)
    distribute_object_info_across_all_frames(dynamic_object_info_all)

    if not make_dynamic:
        empty_dynamic_object_info = {k: {} for k in dynamic_object_info_all.keys()}
        dynamic_object_info_all = empty_dynamic_object_info

    # transform static object and dynamic object to the grid coordinate
    for key, value in static_object_info_all.items():
        if ".static_object_info.json" in key:
            for object_gid, object_info in value.items():
                object_info["object_to_world"] = (
                    world_to_grid.astype(np.float64)
                    @ np.array(object_info["object_to_world"], dtype=np.float64)
                ).tolist()

    for key, value in dynamic_object_info_all.items():
        if ".dynamic_object_info.json" in key:
            for object_gid, object_info in value.items():
                object_info["object_to_world"] = (
                    world_to_grid.astype(np.float64)
                    @ np.array(object_info["object_to_world"], dtype=np.float64)
                ).tolist()

    # If key_pose_indices is provided, resample the object info
    if key_pose_indices is not None:
        static_object_info = {}
        dynamic_object_info = {}

        for idx, frame_idx in enumerate(key_pose_indices):
            old_static_key = f"{frame_idx:06d}.static_object_info.json"
            old_dynamic_key = f"{frame_idx:06d}.dynamic_object_info.json"
            new_static_key = f"{idx:06d}.static_object_info.json"
            new_dynamic_key = f"{idx:06d}.dynamic_object_info.json"

            if old_static_key in static_object_info_all:
                static_object_info[new_static_key] = static_object_info_all[
                    old_static_key
                ]
            if old_dynamic_key in dynamic_object_info_all:
                dynamic_object_info[new_dynamic_key] = dynamic_object_info_all[
                    old_dynamic_key
                ]

        return static_object_info, dynamic_object_info
    else:
        return static_object_info_all, dynamic_object_info_all


# ============================================================================
# Guidance Buffer Generation Core Function
# ============================================================================


def generate_guidance_buffer_and_save(
    clip,
    output_folder,
    resolution,
    camera_model,
    camera_poses,
    fvdb_grid,
    fvdb_semantic,
    static_object_info,
    dynamic_object_info,
    video_prompt,
    disable_video_generation,
    video_checkpoint_path,
    use_wan_1pt3b,
):
    """
    Generate and save guidance buffer using the new API.
    Save both tar files (for training) and video files (for visualization).
    Optionally generate video using WanVideoGenerator.

    Args:
        clip: str
        output_folder: Path
        resolution: str
        camera_model: PinholeCamera
        camera_poses: torch.Tensor
        fvdb_grid: GridBatch
        fvdb_semantic: torch.Tensor
        static_object_info: dict, contains each frame's static object info in grid coordinate
        dynamic_object_info: dict, contains each frame's dynamic object info in grid coordinate
        video_prompt: str or None, text prompt for video generation
        disable_video_generation: bool, whether to disable video generation
        video_checkpoint_path: str, path to video generation checkpoint
    """
    # Grid to world is identity since voxel is already in the first camera's FLU coordinate
    fvdb_grid_to_world = torch.eye(4, device="cuda")

    # Generate buffer using the new API
    depth_buffer, semantic_buffer, instance_buffer = (
        generate_infinicube_buffer_from_fvdb_grid(
            camera_model=camera_model,
            camera_poses_in_world=camera_poses,
            fvdb_scene_grid_or_points=fvdb_grid,
            fvdb_scene_semantic=fvdb_semantic,
            fvdb_grid_to_world=fvdb_grid_to_world,
            static_object_info=static_object_info,
            dynamic_object_info=dynamic_object_info,
            cad_model_for_dynamic_objects=True,
            cad_model_for_static_object=True,
            enlarge_lwh_factor=1.2,
        )
    )

    # Define output files
    depth_buffer_tar_file = output_folder / f"voxel_depth_100_{resolution}_front.tar"
    semantic_buffer_video_file = (
        output_folder / f"semantic_buffer_video_{resolution}_front.mp4"
    )
    instance_buffer_tar_file = output_folder / f"instance_buffer_{resolution}_front.tar"
    depth_vis_video_file = output_folder / f"depth_vis_video_{resolution}_front.mp4"
    coordinate_buffer_video_file = (
        output_folder / f"coordinate_buffer_video_{resolution}_front.mp4"
    )
    pose_tar_file = output_folder / "pose.tar"
    intrinsic_tar_file = output_folder / "intrinsic.tar"
    dynamic_object_info_tar_file = output_folder / "dynamic_object_info.tar"

    # Create tar samples
    depth_sample = {}
    instance_sample = {}
    pose_sample = {}

    # Prepare video frames
    depth_vis_frames = []

    # Process each frame
    for idx in range(depth_buffer.shape[0]):
        # Depth buffer: multiply by 100 and convert to uint16
        depth_np = depth_buffer[idx].cpu().numpy()
        depth_uint16 = (depth_np * 100).astype(np.uint16)
        depth_sample[f"{idx:06d}.voxel_depth_100.front.png"] = imageencoder_imageio_png(
            depth_uint16
        )

        instance_np = instance_buffer[idx].cpu().numpy().astype(np.uint16)
        instance_sample[f"{idx:06d}.instance_buffer.front.png"] = (
            imageencoder_imageio_png(instance_np)
        )

        # Depth visualization for video
        depth_vis = vis_depth(depth_buffer[idx])
        if torch.is_tensor(depth_vis):
            depth_vis = depth_vis.cpu().numpy()
        depth_vis_frames.append(depth_vis)

        # Pose: save camera pose for each frame
        pose_sample[f"{idx:06d}.pose.front.npy"] = camera_poses[idx].cpu().numpy()

    # Semantic buffer: convert to uint8
    semantic_rgb = semantic_to_color(
        semantic_buffer
    )  # (H, W, 3) in range [0, 1]
    semantic_rgb = (semantic_rgb * 255).astype(
        np.uint8
    )  # Convert to [0, 255]
    semantic_buffer_rgb_frames = generate_rgb_semantic_buffer(
        semantic_rgb, instance_buffer.cpu().numpy().astype(np.uint16)
    )

    # Generate coordinate buffer from depth, intrinsic, and poses
    # For forward motion, use larger final_scale_clip to accommodate larger scene range
    coordinate_buffer = generate_coordinate_buffer_from_memory_global_norm(
        depth_buffer=depth_buffer,
        camera_model=camera_model,
        camera_poses=camera_poses,
        percentile=0.05,
    )  # [N, H, W, 3], range [0, 1]

    # Convert to uint8 for video saving (scale from [0, 1] to [0, 255])
    coordinate_buffer_uint8 = (coordinate_buffer * 255).cpu().numpy().astype(np.uint8)
    coordinate_buffer_frames = [
        coordinate_buffer_uint8[i] for i in range(coordinate_buffer_uint8.shape[0])
    ]

    # Intrinsic: save the same intrinsic for each frame
    intrinsic_sample = {"intrinsic.front.npy": camera_model.intrinsics}

    # Write tar files
    write_to_tar(depth_sample, depth_buffer_tar_file, __key__=clip)
    write_to_tar(instance_sample, instance_buffer_tar_file, __key__=clip)
    write_to_tar(pose_sample, pose_tar_file, __key__=clip)
    write_to_tar(intrinsic_sample, intrinsic_tar_file, __key__=clip)
    write_to_tar(dynamic_object_info, dynamic_object_info_tar_file, __key__=clip)

    # Write video files
    write_video_file(semantic_buffer_rgb_frames, semantic_buffer_video_file, fps=10)
    write_video_file(depth_vis_frames, depth_vis_video_file, fps=10)
    write_video_file(coordinate_buffer_frames, coordinate_buffer_video_file, fps=10)

    logger.info(f"Saved guidance buffer to {output_folder}")
    logger.info(f"  - Depth tar: {depth_buffer_tar_file.name}")
    logger.info(f"  - Instance tar: {instance_buffer_tar_file.name}")
    logger.info(f"  - Pose tar: {pose_tar_file.name}")
    logger.info(f"  - Intrinsic tar: {intrinsic_tar_file.name}")
    logger.info(f"  - Depth vis video: {depth_vis_video_file.name}")
    logger.info(f"  - Semantic buffer video: {semantic_buffer_video_file.name}")
    logger.info(f"  - Coordinate buffer video: {coordinate_buffer_video_file.name}")

    # Generate video using WanVideoGenerator if not disabled
    if not disable_video_generation:
        try:
            from infinicube.videogen import WanVideoGenerator

            semantic_buffer_np = np.stack(semantic_buffer_rgb_frames, axis=0)[:93]
            coordinate_buffer_np = np.stack(coordinate_buffer_frames, axis=0)[:93]

            logger.info(
                f"  - Semantic buffer shape (first 93 frames): {semantic_buffer_np.shape}"
            )
            logger.info(
                f"  - Coordinate buffer shape (first 93 frames): {coordinate_buffer_np.shape}"
            )

            # Initialize generator (only once per process if possible)
            if not hasattr(generate_guidance_buffer_and_save, "_video_generator"):
                logger.info(
                    f"  - Initializing WanVideoGenerator with checkpoint: {video_checkpoint_path}"
                )
                generate_guidance_buffer_and_save._video_generator = WanVideoGenerator(
                    checkpoint_path=video_checkpoint_path,
                    device="cuda:0",
                    torch_dtype=torch.bfloat16,
                    buffer_channels=16,
                    enable_vram_management=True,
                    use_wan_1pt3b=use_wan_1pt3b,
                )

            generator = generate_guidance_buffer_and_save._video_generator
            # Generate video
            video_output_path = output_folder / f"video_{resolution}_front.mp4"
            logger.info(f"  - Generating video with prompt: {video_prompt}")

            generator.generate(
                semantic_buffer=semantic_buffer_np,
                coordinate_buffer=coordinate_buffer_np,
                prompt=video_prompt,
                seed=0,
                tiled=True,
                output_path=str(video_output_path),
                fps=10,
                quality=8,
            )

            logger.info(f"  - Generated video saved to: {video_output_path.name}")

        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            logger.error("Continuing without video generation...")
            import traceback

            traceback.print_exc()


# ============================================================================
# Mode-Specific Generation Functions
# ============================================================================


def generate_guidance_buffer_trajectory(
    clip,
    extrap_voxel_time,
    extrap_voxel_root,
    output_root,
    resolution,
    offset_setting,
    make_dynamic,
    data_root,
    video_prompt,
    disable_video_generation,
    video_checkpoint_path,
):
    """Generate guidance buffer for trajectory mode."""
    output_folder = output_root / clip
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    semantic_video_file = (
        output_folder / f"guidance_semantic_buffer_{resolution}_front.mp4"
    )
    depth_video_file = output_folder / f"guidance_depth_vis_{resolution}_front.mp4"

    if semantic_video_file.exists() and depth_video_file.exists():
        logger.info(f"skip {clip} since the guidance buffer video already exists")
        return

    print("start generating guidance buffer (trajectory mode): ", clip)

    # Load camera poses. use first frame as origin, align with voxel generation stage
    # The grid cooridnate align with first camera's flu coordinate
    camera_pose_opencv_in_grid, grid_to_world = get_pose_and_transform(clip, data_root)
    camera_pose_opencv_in_grid = camera_pose_opencv_in_grid.float().cuda()
    world_to_grid = np.linalg.inv(grid_to_world.numpy())

    # Setup camera model with rescaled intrinsic
    camera_model = setup_camera_model(clip, data_root, resolution)

    # Sample poses if needed
    if offset_setting is not None:
        offset_unit, offset = offset_setting
        if offset_unit == "meter":
            key_pose_indices = get_key_frames_indices(
                camera_pose_opencv_in_grid, offset
            )
        elif offset_unit == "frame":
            key_pose_indices = list(
                range(0, len(camera_pose_opencv_in_grid), int(offset))
            )
        else:
            raise ValueError(f"offset_unit {offset_unit} is not supported")

        logger.info(f"key pose indices: {key_pose_indices}")
        camera_pose_opencv_in_grid = camera_pose_opencv_in_grid[key_pose_indices]
    else:
        key_pose_indices = list(range(len(camera_pose_opencv_in_grid)))

    # Load object info
    static_object_info, dynamic_object_info = load_object_info(
        clip, data_root, world_to_grid, key_pose_indices, make_dynamic
    )

    # Load voxel
    fvdb_grid, fvdb_semantic, voxel_path = load_voxel(
        extrap_voxel_root, clip, extrap_voxel_time
    )

    # copy this voxel_path to the output_folder
    shutil.copy(voxel_path, output_folder / "voxel.pt")

    # Generate buffer
    generate_guidance_buffer_and_save(
        clip,
        output_folder,
        resolution,
        camera_model,
        camera_pose_opencv_in_grid,
        fvdb_grid,
        fvdb_semantic,
        static_object_info,
        dynamic_object_info,
        video_prompt=video_prompt,
        disable_video_generation=disable_video_generation,
        video_checkpoint_path=video_checkpoint_path,
    )


def generate_guidance_buffer_blocks(
    clip,
    extrap_voxel_time,
    map_extrap_root,
    output_root,
    resolution,
    existing_trajectory_npy,
    interpolate_frame_num,
    make_dynamic,
    data_root,
    video_prompt,
    disable_video_generation,
    video_checkpoint_path,
    use_wan_1pt3b,
):
    """Generate guidance buffer for blocks mode."""
    # Load voxel
    fvdb_grid, fvdb_semantic, voxel_path = load_voxel(
        map_extrap_root, clip, extrap_voxel_time
    )

    # Load grid to waymo world transformation matrix
    _, grid_to_world = get_pose_and_transform(clip, data_root)
    world_to_grid = np.linalg.inv(grid_to_world.numpy())

    # copy this voxel_path to the output_folder's first pass folder
    (output_root / clip / "pass_0").mkdir(parents=True, exist_ok=True)
    shutil.copy(voxel_path, (output_root / clip / "pass_0" / "voxel.pt"))

    # Get camera poses [N_pass, N_frames, 4, 4]
    if existing_trajectory_npy is not None:
        camera_pose_opencv_mulitpass = np.load(existing_trajectory_npy)
        if len(camera_pose_opencv_mulitpass.shape) == 3:
            camera_pose_opencv_mulitpass = [camera_pose_opencv_mulitpass]
    else:
        # Prepare visualization
        voxel_center_world = (
            fvdb_grid.grid_to_world(fvdb_grid.ijk.jdata.float()).jdata.cpu().numpy()
        )
        visualizaiton_color = semantic_to_color(fvdb_semantic)

        camera_pose_opencv_mulitpass = render_point_cloud_with_viser(
            voxel_center_world,
            visualizaiton_color,
            name="voxel",
            interpolate_frame_num=interpolate_frame_num,
        )

        # save multi-pass camera poses to the output_folder
        np.save(
            output_root / clip / "multi_pass_camera_poses.npy",
            np.stack(camera_pose_opencv_mulitpass, axis=0),
        )

    camera_pose_opencv_mulitpass = [
        torch.tensor(camera_pose_opencv).cuda()
        for camera_pose_opencv in camera_pose_opencv_mulitpass
    ]

    # Setup camera model with rescaled intrinsic
    camera_model = setup_camera_model(clip, data_root, resolution)

    cprint(
        f"Pass number: {camera_pose_opencv_mulitpass.__len__()}",
        "yellow",
        attrs=["bold"],
    )

    for pass_idx, camera_pose_opencv_in_grid in enumerate(camera_pose_opencv_mulitpass):
        # Get key pose indices for object info
        key_pose_indices = get_object_info_key_indices(
            clip, data_root, len(camera_pose_opencv_in_grid)
        )

        # Load object info
        static_object_info, dynamic_object_info = load_object_info(
            clip, data_root, world_to_grid, key_pose_indices, make_dynamic
        )

        # Generate buffer
        output_folder = output_root / clip / f"pass_{pass_idx}"
        output_folder.mkdir(parents=True, exist_ok=True)

        generate_guidance_buffer_and_save(
            clip,
            output_folder,
            resolution,
            camera_model,
            camera_pose_opencv_in_grid,
            fvdb_grid,
            fvdb_semantic,
            static_object_info,
            dynamic_object_info,
            video_prompt=video_prompt,
            disable_video_generation=disable_video_generation,
            video_checkpoint_path=video_checkpoint_path,
            use_wan_1pt3b=use_wan_1pt3b,
        )


# ============================================================================
# CLI
# ============================================================================


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["trajectory", "blocks"]),
    required=True,
    help="Generation mode: trajectory or blocks",
)
@click.option("--clip", type=str, required=True, help="the clip name")
@click.option(
    "--extrap_voxel_time",
    type=int,
    default=None,
    help="the cumulative number of iterations of voxel extrapolation. None means the last voxel",
)
@click.option(
    "--extrap_voxel_root", type=str, required=True, help="the root of the voxel world"
)
@click.option(
    "--output_root",
    type=str,
    default="visualization/infinicube_inference/guidance_buffer_generation",
    help="the root of the output",
)
@click.option(
    "--data_root", type=str, default="data", help="the root of the data folder"
)
@click.option(
    "--resolution",
    type=click.Choice(["480p", "720p"]),
    default="480p",
    help="the target height and width of the buffer",
)
@click.option("--make_dynamic", is_flag=True, help="whether to make scene dynamic.")
# Trajectory mode options
@click.option(
    "--offset_unit",
    type=str,
    default="meter",
    help="offset unit (trajectory mode). meter or frame",
)
@click.option(
    "--offset",
    "-p",
    type=float,
    default=None,
    help="the number of offset for sampling (trajectory mode).",
)
# Blocks mode options
@click.option(
    "--existing_trajectory_npy",
    type=str,
    default=None,
    help="the existing trajectory npy file (blocks mode)",
)
@click.option(
    "--interpolate_frame_num",
    "-i",
    type=int,
    default=93,
    help="the number of interpolated frames (blocks mode)",
)
# Video generation options
@click.option(
    "--video_prompt",
    type=str,
    default="The video is capture by a camera mounted on a vehicle. The video is about a driving scene captured at daytime. The weather is clear.",
    help="Text prompt for video generation.",
)
@click.option(
    "--disable_video_generation",
    is_flag=True,
    help="Disable video generation using WanVideoGenerator",
)
@click.option(
    "--video_checkpoint_path",
    type=str,
    default="checkpoints/wan14b-t2v-buffer-step-1200.safetensors",
    help="Path to video generation checkpoint",
)
@click.option(
    "--use_wan_1pt3b",
    is_flag=True,
    help="Use Wan2.1-T2V-1.3B model for video generation",
)
def main(
    mode,
    clip,
    extrap_voxel_time,
    extrap_voxel_root,
    output_root,
    data_root,
    resolution,
    make_dynamic,
    offset_unit,
    offset,
    existing_trajectory_npy,
    interpolate_frame_num,
    video_prompt,
    disable_video_generation,
    video_checkpoint_path,
    use_wan_1pt3b,
):
    output_root_p = Path(output_root)

    if mode == "trajectory" and offset:
        if offset_unit == "meter":
            output_root_p = output_root_p / f"{mode}_pose_sample_{offset:.2f}m"
        elif offset_unit == "frame":
            output_root_p = output_root_p / f"{mode}_pose_sample_{int(offset)}frame"
        else:
            raise ValueError(f"offset_unit {offset_unit} is not supported")

        offset_setting = (offset_unit, offset)
    else:
        output_root_p = output_root_p / mode

        offset_setting = None

    if mode == "trajectory":
        generate_guidance_buffer_trajectory(
            clip,
            extrap_voxel_time,
            extrap_voxel_root,
            output_root_p,
            resolution,
            offset_setting,
            make_dynamic,
            data_root,
            video_prompt,
            disable_video_generation,
            video_checkpoint_path,
            use_wan_1pt3b,
        )
    elif mode == "blocks":
        generate_guidance_buffer_blocks(
            clip,
            extrap_voxel_time,
            extrap_voxel_root,
            output_root_p,
            resolution,
            existing_trajectory_npy,
            interpolate_frame_num,
            make_dynamic,
            data_root,
            video_prompt,
            disable_video_generation,
            video_checkpoint_path,
            use_wan_1pt3b,
        )


if __name__ == "__main__":
    main()
