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
Dynamic scene generation with feedforward gaussian reconstruction.

This script handles reconstruction of dynamic scenes with both static background and dynamic objects.
Ported from SCube to InfiniCube.

Example usage:
# Resume from local config and checkpoint
python infinicube/inference/scene_gaussian_generation.py none \
    --data_folder visualization/infinicube_inference/guidance_buffer_generation/trajectory_pose_sample_1frame/13679757109245957439_4167_170_4187_170 \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 

# Resume from wandb
python infinicube/inference/scene_gaussian_generation.py none \
    --data_folder visualization/infinicube_inference/guidance_buffer_generation/blocks/13679757109245957439_4167_170_4187_170 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator
"""

import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycg import exp
from termcolor import cprint
from torchvision.utils import save_image

from infinicube import get_sample
from infinicube.utils.fileio_utils import read_video_file
from infinicube.utils.gaussian_io_utils import (
    process_gaussian_params_to_dict,
    save_splat_file,
)
from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.utils.box_util import get_points_in_cuboid_torch
from infinicube.voxelgen.utils.common_util import (
    batch2device,
    create_model_from_args,
    create_model_from_local_config,
    get_default_parser,
)
from infinicube.voxelgen.utils.extrap_util import transform_points

# Constants
SKY_CLASS_ID = 10  # Sky class ID in Cityscapes dataset
DEPTH_SCALE_FACTOR = 100.0  # Depth buffer scaling factor


@torch.inference_mode()
def inference_sky_seg(video, model="segformer"):
    """
    Perform sky segmentation on video sequences.

    Args:
        video: Can be one of the following formats:
            - list of numpy arrays with shape [H, W, 3]
            - list of torch.Tensor with shape [H, W, 3]
            - np.ndarray with shape [N, H, W, 3]
            - torch.Tensor with shape [N, H, W, 3]
        model: Segmentation model name, default is 'segformer' (mask2former has compatibility issues)

    Returns:
        sky_masks: np.ndarray with shape [N, H, W], dtype=bool
            True indicates sky, False indicates non-sky
    """
    from mmseg.apis import MMSegInferencer

    from infinicube.data_process.utils import inference_mmseg

    if model == "segformer":
        inferencer = MMSegInferencer(
            model="segformer_mit-b5_8xb1-160k_cityscapes-1024x1024", device="cuda"
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Convert to numpy list
    video_numpy_list = _convert_video_to_numpy_list(video)

    # Ensure uint8 dtype
    if video_numpy_list[0].dtype != np.uint8:
        video_numpy_list = [
            (frame * 255).astype(np.uint8) for frame in video_numpy_list
        ]

    # Run segmentation
    pred_list = inference_mmseg(video_numpy_list, inferencer)

    # Extract sky regions (sky class ID is 10 in Cityscapes)
    sky_mask_list = [pred.astype(np.uint8) == SKY_CLASS_ID for pred in pred_list]
    sky_masks = np.stack(sky_mask_list, axis=0)

    return sky_masks


def _convert_video_to_numpy_list(video):
    """
    Convert various video data formats to a list of numpy arrays.

    Args:
        video: Input video data

    Returns:
        list of np.ndarray: Each element is a numpy array with shape [H, W, 3]
    """
    if isinstance(video, list):
        if isinstance(video[0], np.ndarray):
            return video
        elif isinstance(video[0], torch.Tensor):
            return [frame.cpu().numpy() for frame in video]
    elif isinstance(video, np.ndarray):
        return [frame for frame in video]
    elif isinstance(video, torch.Tensor):
        return [frame.cpu().numpy() for frame in video]
    else:
        raise TypeError(f"Unsupported video type: {type(video)}")


class GSMModelController:
    """
    Controller for managing GSM model branches and modes.

    Attributes:
        net_model_gsm: GSM network model
        dav2_encoder: DAv2 encoder hosted in the controller
    """

    def __init__(self, net_model_gsm):
        self.net_model_gsm = net_model_gsm
        self.dav2_encoder = None

    def update_pixel_branch_depth_source(self):
        """Update the depth source for the pixel branch."""
        depth_priors = (
            self.net_model_gsm.backbone.backbone_2d.backbone_2d_concat_depth_priors
        )
        if "masked_voxel_depth" in depth_priors:
            depth_priors.remove("masked_voxel_depth")
        if "provided_depth" not in depth_priors:
            depth_priors.append("provided_depth")

    def turn_off_pixel_branch(self):
        """Turn off pixel branch, use 3D branch only."""
        self.net_model_gsm.backbone.use_3d = True
        self.net_model_gsm.backbone.use_2d = False
        # Store dav2_encoder in controller and remove from img_encoder
        if "dav2" in self.net_model_gsm.img_encoder.encoders:
            self.dav2_encoder = self.net_model_gsm.img_encoder.encoders.pop("dav2")

    def turn_off_voxel_branch(self):
        """Turn off voxel branch, use 2D branch only."""
        self.net_model_gsm.backbone.use_3d = False
        self.net_model_gsm.backbone.use_2d = True

    def turn_on_dynamic_recon(self):
        """Enable dynamic reconstruction mode, decode all pixels, disable 3d branch"""
        self.net_model_gsm.backbone.use_3d = False
        self.net_model_gsm.backbone.use_2d = True
        # Decode all pixels to reconstruct dynamic objects
        self.net_model_gsm.backbone.backbone_2d.decode_all_pixel2gs = True

        # Restore dav2_encoder from controller to img_encoder if it was popped
        if self.dav2_encoder is not None:
            self.net_model_gsm.img_encoder.encoders["dav2"] = self.dav2_encoder


def get_parser():
    """Create and configure argument parser for the script."""
    parser = exp.ArgumentParserX(
        base_config_path="infinicube/voxelgen/configs/default/param.yaml",
        parents=[get_default_parser()],
    )
    parser.add_argument(
        "--data_folder", type=str, default=None, help="Path to the folder of data."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="visualization/infinicube_inference/gaussian_scene_generation/",
        help="Path to the folder to save the results.",
    )
    # Either local_config and local_checkpoint_path or wandb_config should be provided
    parser.add_argument(
        "--local_config", type=str, required=False, help="Path to local config file."
    )
    parser.add_argument(
        "--local_checkpoint_path",
        type=str,
        required=False,
        help="Path to local checkpoint file.",
    )
    parser.add_argument(
        "--wandb_config",
        type=str,
        required=False,
        help="Wandb experiment name, start with 'wdb:'.",
    )
    parser.add_argument(
        "--wandb_base",
        type=str,
        default="./wandb/",
        help="Path to wandb base directory.",
    )
    parser.add_argument(
        "--start_frame_index",
        type=int,
        default=0,
        help="Starting frame index for feedforward reconstruction.",
    )
    parser.add_argument(
        "--use_frame_interval",
        type=int,
        default=6,
        help="Frame interval for feedforward reconstruction.",
    )
    parser.add_argument(
        "--active_frame_proportion",
        type=float,
        default=1,
        help="Proportion of frames to use in reconstruction. If 1, use all frames.",
    )
    parser.add_argument(
        "--enable_pixel_branch_last_n_frame",
        type=int,
        default=1,
        help="Enable pixel branch for last N frames. If 0, disable pixel branch for all frames.",
    )
    parser.add_argument(
        "--accumulate_multi_frame_for_dynamic",
        action="store_true",
        help="Accumulate multi-frame data for dynamic objects."
        + "If True, the dynamic object will be reconstructed from the accumulated data."
        "If False, the dynamic object will be reconstructed from the single frame with most Gaussians.",
    )
    return parser


def get_data_dict_from_folder(args):
    """
    Load data dictionary from folder.

    This function loads:
    1) Voxel grid data
    2) Camera poses
    3) Video frames
    4) Camera intrinsics
    5) Depth G-buffers

    Returns:
        dict: Dictionary containing all loaded data
    """
    data_folder = Path(args.data_folder)
    data_dict = {}

    # 1) Load the voxel. voxel.pt only exists in pass_0 if multiple passes are used.
    voxel_file = data_folder / "voxel.pt"
    if voxel_file.exists():
        scene_grid = torch.load(voxel_file)["points"]
        scene_semantic = torch.load(voxel_file)["semantics"]  # [N,]
        data_dict["scene_grid"] = scene_grid
        data_dict["scene_semantic"] = scene_semantic

    # 2) Load the poses
    pose_file = data_folder / "pose.tar"
    pose_sample = get_sample(str(pose_file))
    pose_keys = sorted([i for i in pose_sample.keys() if "pose.front.npy" in i])
    poses = torch.from_numpy(np.stack([pose_sample[k] for k in pose_keys])).float()

    # 3) Load the video
    video_vr = read_video_file(str(data_folder / "video_480p_front.mp4"))
    video_array = video_vr.get_batch(range(len(video_vr))).asnumpy()
    video_array = torch.from_numpy(video_array) / 255.0
    height, width = video_array.shape[1], video_array.shape[2]

    # 4) Load the intrinsic
    intrinsics = get_sample(str(data_folder / "intrinsic.tar"))["intrinsic.front.npy"]
    intrinsics = torch.from_numpy(intrinsics).float()

    # 5) Load the depth buffer
    depth_buffer_tar_file = data_folder / "voxel_depth_100_480p_front.tar"
    depth_buffer_sample = get_sample(str(depth_buffer_tar_file))
    depth_buffer_keys = sorted(
        [i for i in depth_buffer_sample.keys() if ".voxel_depth_100.front.png" in i]
    )
    depth_buffers = torch.from_numpy(
        np.stack(
            [depth_buffer_sample[k] / DEPTH_SCALE_FACTOR for k in depth_buffer_keys]
        )
    ).float()

    # 6) Load the instance buffer -> dynamic masks
    instance_buffer_tar_file = data_folder / "instance_buffer_480p_front.tar"
    instance_buffer_sample = get_sample(str(instance_buffer_tar_file))
    instance_buffer_keys = sorted(
        [i for i in instance_buffer_sample.keys() if ".instance_buffer.front.png" in i]
    )
    instance_buffers = torch.from_numpy(
        np.stack([instance_buffer_sample[k] for k in instance_buffer_keys]).astype(
            np.int32
        )
    )
    dynamic_masks = instance_buffers >= 10000  # dynamic objects are counting from 10000
    non_dynamic_masks = ~dynamic_masks

    # 7) Load the dynamic object info
    dynamic_object_info_tar_file = data_folder / "dynamic_object_info.tar"
    dynamic_object_info_sample = get_sample(str(dynamic_object_info_tar_file))
    dynamic_object_info_keys = sorted(
        [
            i
            for i in dynamic_object_info_sample.keys()
            if ".dynamic_object_info.json" in i
        ]
    )
    dynamic_object_infos = [
        dynamic_object_info_sample[k] for k in dynamic_object_info_keys
    ]

    # Select key frames
    frames_generated_by_video = video_array.shape[0]
    key_frame_indices = _determine_key_frame_indices(
        args, data_folder, frames_generated_by_video
    )

    cprint(f"Key frame indices: {key_frame_indices}", "green", attrs=["bold"])
    cprint(f"Key frame count: {len(key_frame_indices)}", "green", attrs=["bold"])

    # Prepare output dictionary
    data_dict.update(
        {
            "key_frame_indices": key_frame_indices,
            "poses": poses[key_frame_indices],
            "intrinsics": intrinsics.expand(len(key_frame_indices), -1),
            "video_array": video_array[key_frame_indices],
            "depth_buffers": depth_buffers[key_frame_indices],
            "dynamic_masks": dynamic_masks[key_frame_indices],
            "non_dynamic_masks": non_dynamic_masks[key_frame_indices],
            "dynamic_object_infos": [
                dynamic_object_infos[k] for k in key_frame_indices
            ],
            "original_dynamic_object_info_file": dynamic_object_info_tar_file,
        }
    )

    # Infer sky masks
    data_dict = _compute_sky_and_foreground_masks(data_dict)

    # Create input masks for GSM
    data_dict = _create_gsm_input_masks(data_dict, args)

    return data_dict


def _determine_key_frame_indices(args, data_folder, total_frames):
    """
    Determine which frames to use as key frames.

    Priority:
    1. key_frame_indices.json if exists
    2. meta.json if exists
    3. Command line arguments
    """
    if (data_folder / "key_frame_indices.json").exists():
        key_frame_indices = json.load(open(data_folder / "key_frame_indices.json"))
        cprint("Key frames loaded from key_frame_indices.json", "magenta")
    elif (data_folder / "meta.json").exists():
        meta = json.load(open(data_folder / "meta.json"))
        active_frame_proportion = meta["active_frame_proportion"]
        use_frame_interval = int(meta["use_frame_interval"])
        start_frame_index = int(meta["start_frame_index"])
        end_frame = int(active_frame_proportion * total_frames)
        key_frame_indices = list(
            range(start_frame_index, end_frame, use_frame_interval)
        )
        cprint("Key frames loaded from meta.json", "magenta")
    else:
        start_idx = int(args.start_frame_index)
        end_idx = min(
            start_idx + int(args.active_frame_proportion * total_frames), total_frames
        )
        key_frame_indices = list(range(start_idx, end_idx, args.use_frame_interval))
        cprint("Key frames selected by default", "magenta")

    return key_frame_indices


def _compute_sky_and_foreground_masks(data_dict):
    """
    Compute sky and foreground masks from segmentation and depth.
    """
    try:
        sky_masks_from_seg = inference_sky_seg(
            data_dict["video_array"],
        )
        sky_masks_from_seg = torch.from_numpy(sky_masks_from_seg)  # [N, H, W]
    except (NotImplementedError, Exception) as e:
        cprint(
            f"Sky segmentation failed: {e}. Using depth buffer only.",
            "yellow",
            attrs=["bold"],
        )
        sky_masks_from_seg = torch.zeros_like(
            data_dict["depth_buffers"], dtype=torch.bool
        )

    # Get sky mask from depth buffer (depth == 0 indicates sky)
    sky_masks_from_grid = data_dict["depth_buffers"] == 0  # [N, H, W]

    data_dict["foreground_mask_from_seg"] = ~sky_masks_from_seg
    data_dict["foreground_mask_from_grid"] = ~sky_masks_from_grid

    return data_dict


def _create_gsm_input_masks(data_dict, args):
    """
    Create input masks for GSM model.

    Mask channels:
    - Channel 0: Foreground from segmentation
    - Channel 1: Non-dynamic mask, voxel branch will use this mask to reconstruct static background
    - Channel 3: Foreground from depth grid. Note that midground = Channel 0 - Channel 3

    if enable_pixel_branch_last_n_frame > 0, we enable pixel branch to model midground by
    setting proper Channel 0 and Channel 3. Otherwise, we directly let Channel 0 = Channel 3,
    thus disable pixel branch.
    """
    shape = (*data_dict["video_array"].shape[:3], 4)
    images_input_mask = torch.ones(shape).to(data_dict["video_array"])

    # from_seg - from_grid = midground area
    images_input_mask[..., 0] = data_dict["foreground_mask_from_seg"]
    images_input_mask[..., 3] = data_dict["foreground_mask_from_grid"]
    # Assume non-dynamic for all frames (updated later in get_dynamic_only_from_folder)
    images_input_mask[..., 1] = torch.ones_like(
        images_input_mask[..., 0], dtype=torch.bool
    )

    # Enable pixel branch for last N frames only
    n_frames = args.enable_pixel_branch_last_n_frame
    if n_frames > 0:
        breakpoint_frame = -n_frames
        # Disable pixel branch before breakpoint by setting Channel 0 = Channel 3
        images_input_mask[:breakpoint_frame, ..., 0] = images_input_mask[
            :breakpoint_frame, ..., 3
        ]
        cprint(
            f"Pixel branch enabled for last {n_frames} frames (total: {images_input_mask.shape[0]})",
            "yellow",
            attrs=["bold"],
        )
    else:
        # Disable pixel branch for all frames by setting Channel 0 = Channel 3
        images_input_mask[:, ..., 0] = images_input_mask[:, ..., 3]
        cprint(
            f"Pixel branch disabled for all {images_input_mask.shape[0]} frames",
            "yellow",
            attrs=["bold"],
        )

    data_dict["gsm_images_input_mask"] = images_input_mask

    return data_dict


def inference_gsm_feedforward(net_model_gsm, args, data_dict):
    """
    Perform GSM model inference in feedforward manner.

    This function handles both static background and dynamic object reconstruction.
    It uses hybrid GSM with 3D branch and pixel-aligned branch.

    Args:
        net_model_gsm: GSM network model
        args: Command line arguments
        data_dict: Dictionary containing scene data

    Returns:
        output_dict: Dictionary containing all model outputs
    """
    gsm_controller = GSMModelController(net_model_gsm)
    gsm_controller.update_pixel_branch_depth_source()

    cprint(
        f"\nTotal frames in GSM feedforward: {data_dict['video_array'].shape[0]}",
        "red",
        "on_cyan",
        attrs=["bold"],
    )

    # Prepare batch for GSM
    batch_gsm = _prepare_gsm_batch(data_dict, net_model_gsm.device)

    start_time = time.time()

    # Perform static background reconstruction
    with torch.no_grad():
        renderer_output, network_output = net_model_gsm.forward(
            batch_gsm, update_grid_mask=False
        )
        gt_package = net_model_gsm.loss.prepare_resized_gt(batch_gsm)
        vis_images_dict = net_model_gsm.loss.assemble_visualization(
            gt_package, renderer_output
        )
        decoded_gaussians = network_output["decoded_gaussians"]
        decoded_gaussian = decoded_gaussians[0]

    # Initialize output dictionary for all model outputs (separate from data_dict)
    output_dict = {
        "rendering": {},
        "decoded_gaussian": {},
    }

    # Save skybox representation if available
    if net_model_gsm.hparams.skybox_target == "mlp_modulator":
        output_dict["skybox_representation"] = network_output["skybox_representation"]

    # Store static reconstruction results
    output_dict["rendering"]["static_pd_images"] = vis_images_dict["pd_images"][0]
    output_dict["rendering"]["static_pd_images_fg"] = vis_images_dict["pd_images_fg"][0]
    output_dict["rendering"]["static_gt_images"] = vis_images_dict["gt_images"][0]
    output_dict["decoded_gaussian"]["static"] = decoded_gaussian

    _reconstruct_dynamic_objects(
        net_model_gsm, gsm_controller, batch_gsm, data_dict, output_dict, args
    )

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    return output_dict


def _prepare_gsm_batch(data_dict, device):
    """
    Prepare input batch for GSM model.

    Args:
        data_dict: Dictionary containing scene data
        device: Device to move data to

    Returns:
        dict: Batch dictionary for GSM model
    """
    batch_gsm = {
        DS.INPUT_PC: data_dict["scene_grid"],
        DS.IMAGES_INPUT: [data_dict["video_array"]],
        DS.IMAGES: [data_dict["video_array"]],
        DS.IMAGES_INPUT_INTRINSIC: [data_dict["intrinsics"]],
        DS.IMAGES_INTRINSIC: [data_dict["intrinsics"]],
        DS.IMAGES_INPUT_POSE: [data_dict["poses"]],
        DS.IMAGES_POSE: [data_dict["poses"]],
        DS.IMAGES_INPUT_MASK: [data_dict["gsm_images_input_mask"]],
        DS.IMAGES_MASK: [data_dict["gsm_images_input_mask"]],
        "provided_depth": [data_dict["depth_buffers"].unsqueeze(-1)],
    }
    return batch2device(batch_gsm, device)


def _reconstruct_dynamic_objects(
    net_model_gsm, gsm_controller, batch_gsm, data_dict, output_dict, args
):
    """
    Reconstruct dynamic objects using pixel branch.

    Args:
        net_model_gsm: GSM network model
        gsm_controller: GSM model controller
        batch_gsm: Batch data for GSM
        data_dict: Dictionary containing scene data
        output_dict: Dictionary to store model outputs
        args: Command line arguments
    """
    dynamic_masks = data_dict["dynamic_masks"]

    # if all zero, this is a static scene, skip dynamic reconstruction.
    if not dynamic_masks.any():
        cprint("Static scene, skipping dynamic reconstruction.", "red", attrs=["bold"])
        return

    gsm_controller.turn_on_dynamic_recon()

    with torch.no_grad():
        imgenc_output = net_model_gsm.img_encoder(batch_gsm)
        network_output = net_model_gsm.backbone(batch_gsm, imgenc_output)
        decoded_gaussians = network_output["decoded_gaussians"]
        decoded_gaussian = decoded_gaussians[0]

    # Verify output shape
    n_frames, height, width = data_dict["video_array"].shape[:3]
    gs_per_pixel = net_model_gsm.backbone.backbone_2d.backbone_2d_gs_per_pixel
    expected_size = n_frames * height * width * gs_per_pixel
    assert decoded_gaussian.shape[0] == expected_size, (
        f"Expected {expected_size} Gaussians, got {decoded_gaussian.shape[0]}"
    )

    # Reshape decoded Gaussians
    decoded_gaussian = decoded_gaussian.view(n_frames, height, width, gs_per_pixel, -1)

    # Extract dynamic object Gaussians
    _extract_and_save_dynamic_gaussians(decoded_gaussian, data_dict, output_dict, args)

    # Render first frame with dynamic objects
    _render_first_frame_with_dynamics(net_model_gsm, batch_gsm, data_dict, output_dict)


def _extract_and_save_dynamic_gaussians(decoded_gaussian, data_dict, output_dict, args):
    """
    Extract dynamic object Gaussians from per-pixel decoded Gaussians.

    Args:
        decoded_gaussian: Decoded Gaussian parameters [N, H, W, K, 14]
        data_dict: Dictionary containing scene data
        output_dict: Dictionary to store model outputs
        args: Command line arguments
    """
    dynamic_masks = data_dict["dynamic_masks"]  # [N, H, W]

    # Extract Gaussians for dynamic regions
    per_frame_dynamic_gaussians = [
        decoded_gaussian[i][dynamic_masks[i]].flatten(0, 1)
        for i in range(decoded_gaussian.shape[0])
    ]
    per_frame_dynamic_gaussians_center = [x[:, :3] for x in per_frame_dynamic_gaussians]

    # Collect all dynamic object IDs
    all_dynamic_gids = [list(x.keys()) for x in data_dict["dynamic_object_infos"]]
    all_dynamic_gids = [item for sublist in all_dynamic_gids for item in sublist]
    all_dynamic_gids = list(set(all_dynamic_gids))
    all_dynamic_gids.sort()

    # Initialize empty Gaussian storage for each dynamic object
    dynamic_gid_to_gaussian = {
        gid: torch.zeros(0, 14).to("cuda") for gid in all_dynamic_gids
    }

    cprint(
        f"Accumulate multi-frame: {args.accumulate_multi_frame_for_dynamic}",
        "red",
        attrs=["bold"],
    )

    # Process each frame and extract Gaussians within cuboids
    for i, dynamic_object_info in enumerate(data_dict["dynamic_object_infos"]):
        for gid, object_info in dynamic_object_info.items():
            points_in_cuboid, inside_mask = get_points_in_cuboid_torch(
                per_frame_dynamic_gaussians_center[i], object_info
            )

            valid_gaussian_in_cuboid = per_frame_dynamic_gaussians[i][inside_mask]
            valid_gaussian_in_cuboid[:, :3] = points_in_cuboid[inside_mask]

            # Accumulate Gaussians across all frames
            if args.accumulate_multi_frame_for_dynamic:
                dynamic_gid_to_gaussian[gid] = torch.cat(
                    [dynamic_gid_to_gaussian[gid], valid_gaussian_in_cuboid], dim=0
                )
            # Keep only the frame with most Gaussians
            else:
                if (
                    valid_gaussian_in_cuboid.shape[0]
                    > dynamic_gid_to_gaussian[gid].shape[0]
                ):
                    dynamic_gid_to_gaussian[gid] = valid_gaussian_in_cuboid

    # Store tensor version for rendering (before dict conversion)
    data_dict["_dynamic_gid_to_gaussian_tensor"] = dynamic_gid_to_gaussian

    # Convert to Gaussian dictionaries for saving
    output_dict["decoded_gaussian"]["object"] = {}
    for gid, gaussian_concat in dynamic_gid_to_gaussian.items():
        gaussian_dict = process_gaussian_params_to_dict(
            *gaussian_concat.split([3, 3, 4, 1, 3], dim=-1)
        )
        output_dict["decoded_gaussian"]["object"][gid] = gaussian_dict


def _render_first_frame_with_dynamics(net_model_gsm, batch_gsm, data_dict, output_dict):
    """
    Render the first frame with dynamic objects placed in the scene.

    Args:
        net_model_gsm: GSM network model
        batch_gsm: Batch data for GSM
        data_dict: Dictionary containing scene data
        output_dict: Dictionary to store model outputs
    """
    # Use the tensor version of dynamic Gaussians stored earlier
    dynamic_gid_to_gaussian = data_dict["_dynamic_gid_to_gaussian_tensor"]

    # Start with static background Gaussians
    first_frame_gaussians = output_dict["decoded_gaussian"]["static"].clone()
    dynamic_object_info_first_frame = data_dict["dynamic_object_infos"][0]

    # Add each dynamic object to the first frame
    for gid, cuboid in dynamic_object_info_first_frame.items():
        if gid not in dynamic_gid_to_gaussian:
            continue

        object_gaussian = dynamic_gid_to_gaussian[gid].clone()

        # Transform object to grid coordinates
        object_to_grid = torch.tensor(cuboid["object_to_world"]).to(
            object_gaussian.device
        )
        object_gaussian_xyz_in_grid = transform_points(
            object_gaussian[:, :3], object_to_grid
        )
        object_gaussian[:, :3] = object_gaussian_xyz_in_grid

        first_frame_gaussians = torch.cat(
            [first_frame_gaussians, object_gaussian], dim=0
        )

    output_dict["decoded_gaussian"]["first_frame"] = first_frame_gaussians

    # Render the first frame
    with torch.no_grad():
        imgenc_output = net_model_gsm.img_encoder(batch_gsm)
        skyenc_output = net_model_gsm.skybox.encode_sky_feature(
            batch_gsm, imgenc_output
        )
        network_output = net_model_gsm.backbone(batch_gsm, imgenc_output)
        network_output = net_model_gsm.skybox(skyenc_output, network_output)
        network_output["decoded_gaussians"] = [first_frame_gaussians]
        renderer_output = net_model_gsm.renderer(
            batch_gsm, network_output, net_model_gsm.skybox
        )
        gt_package = net_model_gsm.loss.prepare_resized_gt(batch_gsm)
        vis_images_dict = net_model_gsm.loss.assemble_visualization(
            gt_package, renderer_output
        )

        output_dict["rendering"]["first_frame_pd_images"] = vis_images_dict[
            "pd_images"
        ][0]
        output_dict["rendering"]["first_frame_images_fg"] = vis_images_dict[
            "pd_images_fg"
        ][0]
        output_dict["rendering"]["first_frame_gt_images"] = vis_images_dict[
            "gt_images"
        ][0]


def visualize_gsm_result(data_dict, output_dict, args, output_folder):
    """
    Visualize GSM reconstruction results.

    Args:
        data_dict: Dictionary containing scene data
        output_dict: Dictionary containing reconstruction results
        args: Command line arguments
        output_folder: Path to save visualizations
    """
    visualize_folder_p = output_folder / "visualize_gsm"
    visualize_folder_p.mkdir(parents=True, exist_ok=True)

    # Save main visualization images
    save_image(
        output_dict["rendering"]["static_pd_images"].permute(0, 3, 1, 2),
        (visualize_folder_p / "static_pd_images.jpg"),
    )
    save_image(
        output_dict["rendering"]["static_pd_images_fg"].permute(0, 3, 1, 2),
        (visualize_folder_p / "static_pd_images_fg.jpg"),
    )
    save_image(
        output_dict["rendering"]["static_gt_images"].permute(0, 3, 1, 2),
        (visualize_folder_p / "static_gt_images.jpg"),
    )

    # Save dynamic scene visualizations if available
    if data_dict["dynamic_masks"].any():
        save_image(
            output_dict["rendering"]["first_frame_pd_images"].permute(0, 3, 1, 2),
            (visualize_folder_p / "first_frame_pd_images.jpg"),
        )
        save_image(
            output_dict["rendering"]["first_frame_images_fg"].permute(0, 3, 1, 2),
            (visualize_folder_p / "first_frame_images_fg.jpg"),
        )
        save_image(
            output_dict["rendering"]["first_frame_gt_images"].permute(0, 3, 1, 2),
            (visualize_folder_p / "first_frame_gt_images.jpg"),
        )

    # Save mask visualizations
    close_range = (
        data_dict["foreground_mask_from_seg"].unsqueeze(1).float()
        * data_dict["foreground_mask_from_grid"].unsqueeze(1).float()
    )
    mid_ground = (
        data_dict["foreground_mask_from_seg"].unsqueeze(1).float() - close_range
    )
    sky_area = 1 - data_dict["foreground_mask_from_seg"].unsqueeze(1).float()

    gt_image = output_dict["rendering"]["static_gt_images"].permute(0, 3, 1, 2)
    gt_image = F.interpolate(
        gt_image,
        (close_range.shape[2], close_range.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).cpu()

    save_image(gt_image * close_range, (visualize_folder_p / "gt_close_range.jpg"))
    save_image(gt_image * mid_ground, (visualize_folder_p / "gt_mid_ground.jpg"))
    save_image(gt_image * sky_area, (visualize_folder_p / "gt_sky_area.jpg"))


def data_loading_handler(args):
    """
    Load data from single or multiple passes.

    If the data folder contains multiple passes (pass_0, pass_1, ...),
    this function will merge data from all passes.

    Args:
        args: Command line arguments with data_folder attribute

    Returns:
        dict: Merged data dictionary from all passes
    """
    sub_folders = list(Path(args.data_folder).glob("pass_*"))

    is_multiple_pass = len(sub_folders) > 0
    if is_multiple_pass:
        assert args.enable_pixel_branch_last_n_frame == 0, (
            "It would be better to disable pixel branch for multiple passes."
        )

        data_dict = None
        data_folder = args.data_folder

        for sub_folder in sub_folders:
            args.data_folder = sub_folder.as_posix()
            data_dict_one_pass = get_data_dict_from_folder(args)

            if data_dict is None:
                data_dict = data_dict_one_pass
            else:
                # Merge data from multiple passes
                data_dict = _merge_pass_data(data_dict, data_dict_one_pass)

        args.data_folder = data_folder
        return data_dict
    else:
        data_dict = get_data_dict_from_folder(args)
        return data_dict


def _merge_pass_data(data_dict, data_dict_one_pass):
    """
    Merge data from two passes.

    Args:
        data_dict: Existing data dictionary
        data_dict_one_pass: New pass data to merge

    Returns:
        dict: Merged data dictionary
    """
    for key, value in data_dict_one_pass.items():
        if key.startswith("scene_"):
            # Scene grid and semantics are the same for every pass
            continue
        elif key == "original_dynamic_object_info_file":
            # they are the same for every pass
            continue
        elif isinstance(value, torch.Tensor):
            data_dict[key] = torch.cat([data_dict[key], value], dim=0)
        elif isinstance(value, list):
            data_dict[key].extend(value)
        elif isinstance(value, bool):
            continue
        else:
            raise TypeError(f"Unknown type {type(value)} for key {key}")

    return data_dict


def main():
    """Main function to run GSM feedforward inference and save results."""
    known_args = get_parser().parse_known_args()[0]

    # Check resume method
    resume_from_local = known_args.local_config is not None
    resume_from_wandb = (
        known_args.wandb_config is not None
        and known_args.wandb_config.startswith("wdb:")
    )
    assert resume_from_local or resume_from_wandb, (
        "Either local_config and local_checkpoint_path or wandb_config (starting with 'wdb:') should be provided"
    )

    hparam_update = {"skybox_forward_sky_only": True}

    # Load GSM model
    if resume_from_local:
        net_model_gsm, args, global_step_gsm = create_model_from_local_config(
            config_path=known_args.local_config,
            checkpoint_path=known_args.local_checkpoint_path,
            hparam_update=hparam_update,
        )
        # Merge command line args into model args
        for key, value in vars(known_args).items():
            if value is not None or not hasattr(args, key):
                setattr(args, key, value)
    else:
        net_model_gsm, args, global_step_gsm = create_model_from_args(
            known_args.wandb_config + ":last",
            known_args,
            get_parser(),
            hparam_update=hparam_update,
        )

    net_model_gsm.to("cuda")
    net_model_gsm.eval()

    # Load and prepare data
    data_dict = data_loading_handler(args)

    # Run GSM inference
    output_dict = inference_gsm_feedforward(net_model_gsm, args, data_dict)

    # Create output folder
    mode, clip_name = args.data_folder.split("/")[-2:]
    output_folder = Path(args.output_folder) / mode / clip_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save static Gaussian splat
    static_gs_path = (output_folder / "decoded_gs_static.pkl").as_posix()
    save_splat_file(output_dict["decoded_gaussian"]["static"], static_gs_path)

    # Save skybox representation
    net_model_gsm.skybox.save_skybox(output_dict, static_gs_path)

    # Save dynamic object Gaussians if available
    if (
        "object" in output_dict["decoded_gaussian"]
        and output_dict["decoded_gaussian"]["object"]
    ):
        dynamic_gs_path = (output_folder / "decoded_gs_object.pkl").as_posix()
        with open(dynamic_gs_path, "wb") as f:
            pickle.dump(output_dict["decoded_gaussian"]["object"], f)
        print(f"Dynamic GS: {dynamic_gs_path}")

    # copy dynamic object info file to output folder
    dynamic_object_info_file = data_dict["original_dynamic_object_info_file"]
    shutil.copy(dynamic_object_info_file, output_folder / "dynamic_object_info.tar")

    # Print output paths
    print(f"Static GS: {static_gs_path}")

    # Visualize results
    visualize_gsm_result(data_dict, output_dict, args, output_folder)


if __name__ == "__main__":
    main()
