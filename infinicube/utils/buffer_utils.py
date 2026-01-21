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
from typing import List, Union

import numpy as np
import torch
from decord import VideoReader

from infinicube.camera.pinhole import PinholeCamera
from infinicube.utils.depth_utils import (
    unproject_depth_torch,
)
from infinicube.utils.semantic_utils import generate_rgb_semantic_buffer
from infinicube.utils.wds_utils import get_sample

resolution_anno_to_wh = {
    "480p": (832, 480),
    "720p": (1280, 720),
}


def read_semantic_buffer_from_file(
    semantic_buffer_video_file: Union[str, Path],
    instance_buffer_tar_file: Union[str, Path],
    frame_indices: List[int],
) -> np.ndarray:
    """
    Read semantic_buffer video and instance_buffer, convert them to RGB colors,
    and overlay instance buffer on semantic buffer to get scene-level semantic buffer.

    Args:
        semantic_buffer_video_file: Path to the semantic buffer video file.
        instance_buffer_tar_file: Path to the instance buffer tar file.
        frame_indices: List of frame indices to read, e.g., [0, 1, 2, ...]

    Returns:
        rgb_semantic_buffer_frames: np.ndarray, shape [N, H, W, 3], range [0, 255], dtype uint8
    """
    # Read semantic_buffer video file and instance_buffer tar file
    vr = VideoReader(str(semantic_buffer_video_file))

    # Read specified frames from video
    # VideoReader returns frames in shape [N, H, W, C] where C=3 for RGB video
    # We need to extract the grayscale semantic values from the video
    semantic_rgb_frames = vr.get_batch(
        frame_indices
    ).asnumpy()  # shape [N, H, W, 3]

    # Load instance buffer data from tar
    instance_data = get_sample(instance_buffer_tar_file)

    # Read all instance frames
    instance_buffers = []
    for frame_idx in frame_indices:
        # Construct key name
        instance_key = f"{frame_idx:06d}.instance_buffer.front.png"

        if instance_key not in instance_data:
            raise KeyError(f"Frame {frame_idx} not found in instance buffer tar")

        # Get instance buffer
        instance_buffer = instance_data[instance_key]  # shape (H, W), dtype uint16
        instance_buffers.append(instance_buffer)

    # Stack all instance frames: [N, H, W]
    instance_buffers = np.stack(instance_buffers, axis=0)  # shape [N, H, W]

    return generate_rgb_semantic_buffer(semantic_rgb_frames, instance_buffers)


def read_coordinate_buffer_from_file(
    depth_buffer_tar_file: Union[str, Path],
    intrinsic_tar_file: Union[str, Path],
    pose_tar_file: Union[str, Path],
    frame_indices: List[int],
    resolution_anno: str = "480p",
    percentile: float = 0.05,
) -> np.ndarray:
    """
    Read depth buffer, intrinsic, and pose, convert them to coordinate buffer (point map),
    and normalize the coordinate buffer.

    Args:
        depth_buffer_tar_file: Path to the depth buffer tar file.
        intrinsic_tar_file: Path to the intrinsic tar file.
        pose_tar_file: Path to the pose tar file.
        frame_indices: List of frame indices to read, e.g., [0, 1, 2, ...]
        resolution_anno: Resolution of the annotation, e.g., "480p" or "720p".
    Returns:
        coordinate_buffer: np.ndarray, shape [N, H, W, 3], range [0, 1], dtype float32
    """

    # Get resolution
    target_h, target_w = resolution_anno_to_wh[resolution_anno]

    # Load data from tar files
    depth_data = get_sample(depth_buffer_tar_file)
    intrinsic_data = get_sample(intrinsic_tar_file)
    pose_data = get_sample(pose_tar_file)

    # Read all depth frames for specified indices
    depth_buffers = []
    for frame_idx in frame_indices:
        # Construct key name for depth buffer
        depth_key = f"{frame_idx:06d}.voxel_depth_100.front.png"

        if depth_key not in depth_data:
            raise KeyError(f"Frame {frame_idx} not found in depth buffer tar")

        # Read depth buffer (stored as int16, scaled by 100)
        depth_buffer = depth_data[depth_key]  # shape (H, W), dtype int16
        # Convert back to float depth in meters
        depth_buffer = depth_buffer.astype(np.float32) / 100.0
        depth_buffers.append(depth_buffer)

    # Stack all depth frames: [N, H, W]
    depth_buffers = np.stack(depth_buffers, axis=0)  # shape [N, H, W]

    # Read camera intrinsic for FRONT camera
    intrinsic_key = "intrinsic.front.npy"
    if intrinsic_key not in intrinsic_data:
        raise KeyError("Intrinsic for front camera not found in intrinsic tar")

    camera_intrinsic = intrinsic_data[
        intrinsic_key
    ]  # shape (6,): [fx, fy, cx, cy, w, h]

    # Create camera model and rescale to target resolution
    camera_model = PinholeCamera.from_numpy(camera_intrinsic, device="cpu")
    original_h, original_w = camera_model.height, camera_model.width
    camera_model.rescale(ratio_h=target_h / original_h, ratio_w=target_w / original_w)

    # Get intrinsic matrix
    intrinsic_matrix = camera_model.get_intrinsics_matrix()  # shape (3, 3)

    # Read camera poses for specified frames
    camera_poses = []
    for frame_idx in frame_indices:
        pose_key = f"{frame_idx:06d}.pose.front.npy"

        if pose_key not in pose_data:
            raise KeyError(f"Pose for frame {frame_idx} not found in pose tar")

        camera_pose = pose_data[pose_key]  # shape (4, 4), camera to world
        camera_poses.append(camera_pose)

    # Stack all camera poses: [N, 4, 4]
    camera_poses = np.stack(camera_poses, axis=0)
    camera_poses = torch.from_numpy(camera_poses).float()  # [N, 4, 4]

    # Convert to torch tensors
    depth_torch = torch.from_numpy(depth_buffers).float()  # [N, H, W]

    # camera model
    camera_model = PinholeCamera.from_numpy(camera_intrinsic, device="cpu")
    original_h, original_w = camera_model.height, camera_model.width
    camera_model.rescale(ratio_h=target_h / original_h, ratio_w=target_w / original_w)

    coordinate_buffer = generate_coordinate_buffer_from_memory_global_norm(
        depth_torch, camera_model, camera_poses, percentile=percentile
    )

    return coordinate_buffer


def generate_coordinate_buffer_from_memory_global_norm(
    depth_buffer: torch.Tensor,
    camera_model: PinholeCamera,
    camera_poses: torch.Tensor,
    percentile: float = 0.05,
) -> torch.Tensor:
    """
    Generate coordinate buffer with global normalization across all frames.
    This method is better for forward motion as it uses all frames to compute normalization range.

    Args:
        depth_buffer: torch.Tensor, shape [N, H, W], depth in meters
        camera_model: PinholeCamera, contains camera intrinsic parameters
        camera_poses: torch.Tensor, shape [N, 4, 4], camera to world transformation
        percentile: float, percentile for normalization

    Returns:
        coordinate_buffer: torch.Tensor, shape [N, H, W, 3], range [0, 1], dtype float32
    """
    # Convert depth_buffer to [N, 1, H, W] format
    depth_torch = depth_buffer.unsqueeze(1)  # [N, 1, H, W]

    # Get intrinsic matrix from camera model
    intrinsic_matrix = camera_model.get_intrinsics_matrix()  # [3, 3]
    intrinsic_matrix_torch = intrinsic_matrix.unsqueeze(0).repeat(
        depth_buffer.shape[0], 1, 1
    )  # [N, 3, 3]

    # Identify infinite far mask (depth == 0 means infinite far)
    infinite_far_mask = depth_buffer == 0  # [N, H, W]

    # Compute transformation from each frame to the first frame
    first_camera_pose = camera_poses[0]  # [4, 4]
    first_camera_pose_inv = torch.inverse(first_camera_pose)  # [4, 4]
    camera_to_camera0 = torch.einsum(
        "ij,bjk->bik", first_camera_pose_inv, camera_poses
    )  # [N, 4, 4]

    # Unproject depth to 3D points in the first frame's coordinate
    points_camera0 = unproject_depth_torch(
        depth_torch,  # [N, 1, H, W]
        camera_to_camera0,  # [N, 4, 4]
        intrinsic_matrix_torch,  # [N, 3, 3]
    )  # [N, H, W, 3]

    # Set infinite far points to a large value temporarily
    points_camera0[infinite_far_mask] = 1e7

    # Global normalization: use all frames to compute normalization range
    # Flatten all points from all frames
    all_points_flat = points_camera0.reshape(-1, 3)  # [N*H*W, 3]

    # Filter out infinite far points
    valid_mask = all_points_flat[:, 2] < 1e6  # z < 1e6
    valid_points = all_points_flat[valid_mask]

    if valid_points.shape[0] > 0:
        # Compute global min/max for each dimension using percentile
        # random sample 100000 points for quantile computation
        valid_points_sampled = valid_points[
            torch.randperm(valid_points.shape[0])[:100000]
        ]

        mins = torch.quantile(valid_points_sampled, percentile, dim=0)  # [3]
        maxs = torch.quantile(valid_points_sampled, 1 - percentile, dim=0)  # [3]

        # Compute range
        ranges = maxs - mins
        ranges = torch.clamp(ranges, min=1e-7)  # Avoid division by zero

        # Normalize to [-1, 1] then to [0, 1]
        points_normalized = (points_camera0 - mins) / ranges * 2.0 - 1.0  # [-1, 1]

        # Clip to [-1, 1] range
        points_normalized = torch.clamp(points_normalized, -1.0, 1.0)

        # Convert to [0, 1] range
        coordinate_buffer = (points_normalized + 1.0) / 2.0
    else:
        # Fallback: no valid points
        coordinate_buffer = points_camera0 * 0.5  # Just return zeros

    # Set infinite far points to [1, 1, 1]
    coordinate_buffer[infinite_far_mask] = 1.0

    return coordinate_buffer


if __name__ == "__main__":
    import json

    import imageio.v3 as iio

    clips = json.load(
        open("infinicube/assets/waymo_split/all_w_dynamic_w_ego_motion_gt_30m.json")
    )[:10]

    for clip in clips:
        depth_buffer_tar_file = f"data/voxel_depth_100_480p_front/{clip}.tar"
        intrinsic_tar_file = f"data/intrinsic/{clip}.tar"
        pose_tar_file = f"data/pose/{clip}.tar"
        frame_indices = list(range(100))
        resolution_anno = "480p"
        coordinate_buffer = read_coordinate_buffer_from_file(
            depth_buffer_tar_file,
            intrinsic_tar_file,
            pose_tar_file,
            frame_indices,
            resolution_anno,
        )
        iio.imwrite(
            f"coordinate_buffer_{clip}.mp4",
            (coordinate_buffer.numpy() * 255).astype(np.uint8),
        )
