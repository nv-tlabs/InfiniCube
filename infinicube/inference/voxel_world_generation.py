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
Unified Voxel World Generation
Support two generation modes:
1. trajectory: Follow the original dataset trajectory
2. blocks: Generate the entire map by blocks

Example usage:

# Trajectory mode - from local checkpoint
python infinicube/inference/voxel_world_generation.py none \
    --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip 13679757109245957439_4167_170_4187_170 \
    --target_pose_num 8 

# Trajectory mode - from wandb
python infinicube/inference/voxel_world_generation.py none \
    --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/diffusion_64x64x64_dense_vs02_map_cond \
    --clip 13679757109245957439_4167_170_4187_170 \
    --target_pose_num 8 

# Blocks mode - from local checkpoint
python infinicube/inference/voxel_world_generation.py none \
    --mode blocks \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip 13679757109245957439_4167_170_4187_170 \
    --overlap_ratio 0.25 

# Blocks mode - from wandb
python infinicube/inference/voxel_world_generation.py none \
    --mode blocks \
    --use_ema --use_ddim --ddim_step 100 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/diffusion_64x64x64_dense_vs02_map_cond \
    --clip 13679757109245957439_4167_170_4187_170 \
    --overlap_ratio 0.25 

The generated voxel is in first camera's flu coordinate!
"""

import sys
from pathlib import Path

import fvdb
import imageio.v3 as imageio
import numpy as np
import torch

sys.path.append(Path(__file__).parent.parent.as_posix())

from copy import deepcopy

from fvdb import JaggedTensor
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from pycg import exp, render, vis
from tqdm import tqdm

from infinicube.camera import opencv_to_flu
from infinicube.voxelgen.utils.common_util import (
    batch2device,
    create_model_from_args,
    create_model_from_local_config,
    get_default_parser,
)
from infinicube.voxelgen.utils.extrap_util import (
    generate_camera_poses_from_batch_trajectory,
    get_relative_transforms,
    get_wds_data,
    transform_grid_and_semantic,
    transform_points,
)
from infinicube.voxelgen.utils.voxel_util import (
    merge_grid2_to_grid1,
    offscreen_map_voxel_render,
    offscreen_voxel_list_to_mesh_renderer,
    single_semantic_voxel_to_mesh,
)


def get_parser():
    parser = exp.ArgumentParserX(
        base_config_path="configs/default/param.yaml", parents=[get_default_parser()]
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["trajectory", "blocks"],
        help="Generation mode: trajectory or blocks.",
    )
    parser.add_argument("--clip", type=str, required=True, help="Clip name.")
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
        "--output_root",
        type=str,
        default="visualization/infinicube_inference/voxel_world_generation",
        help="Output directory.",
    )
    parser.add_argument(
        "--webdataset_root", type=str, default="data/", help="Path to webdataset root."
    )

    # Trajectory mode specific arguments
    parser.add_argument(
        "--target_pose_num",
        type=int,
        default=5,
        help="Target pose number (trajectory mode).",
    )
    parser.add_argument(
        "--pose_distance_ratio",
        type=float,
        default=0.75,
        help="Ratio of grid size for pose distance (trajectory mode).",
    )

    # Blocks mode specific arguments
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.25,
        help="Overlap ratio for blocks (blocks mode).",
    )

    # Diffusion sampling arguments
    parser.add_argument(
        "--use_ddim", action="store_true", help="Use DDIM for diffusion."
    )
    parser.add_argument(
        "--ddim_step", type=int, default=100, help="Number of steps to increase ddim."
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to turn on ema option."
    )
    parser.add_argument(
        "--use_dpm", action="store_true", help="use DPM++ solver or not"
    )
    parser.add_argument(
        "--use_karras", action="store_true", help="use Karras noise schedule or not "
    )
    parser.add_argument(
        "--solver_order",
        type=int,
        default=3,
        help="order of the solver; 3 for unconditional diffusion, 2 for guided sampling",
    )
    parser.add_argument(
        "--h_stride", type=int, default=2, help="Use for anisotropic pooling settting"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Strength of the guidance for classifier-free.",
    )
    return parser


class VoxelWorldGenerator:
    """Unified Voxel World Generator supporting trajectory and blocks modes."""

    def __init__(self, clip, args, net_model_diffusion):
        self.clip = clip
        self.mode = args.mode
        self.output_path = Path(args.output_root) / self.mode / clip
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.args = args
        self.webdataset_root = args.webdataset_root

        self.device = torch.device("cuda")
        self.net_model_diffusion = net_model_diffusion.to("cuda")
        self.net_model_diffusion.eval()

        # Common grid parameters
        self.grid_crop_bbox_min = self.net_model_diffusion.hparams.grid_crop_bbox_min
        self.grid_crop_bbox_max = self.net_model_diffusion.hparams.grid_crop_bbox_max

        # Make grid symmetric in x and y
        x_len = self.grid_crop_bbox_max[0] - self.grid_crop_bbox_min[0]
        y_len = self.grid_crop_bbox_max[1] - self.grid_crop_bbox_min[1]
        self.grid_crop_bbox_min[0] = -x_len / 2
        self.grid_crop_bbox_max[0] = x_len / 2
        self.grid_crop_bbox_min[1] = -y_len / 2
        self.grid_crop_bbox_max[1] = y_len / 2

        # Common scene storage
        self.scene_grid = None
        self.scene_semantic = None
        self.grid_semantic_pairs = []

        # Mode-specific initialization
        if self.mode == "trajectory":
            self._init_trajectory_mode()
        elif self.mode == "blocks":
            self._init_blocks_mode()

    def _init_trajectory_mode(self):
        """Initialize trajectory mode specific parameters."""
        self.target_pose_num = self.args.target_pose_num
        self.use_preset_trajectory = True
        self.pose_distance_interval = (
            self.grid_crop_bbox_max[0] - self.grid_crop_bbox_min[0]
        ) * self.args.pose_distance_ratio
        print(
            f"[Trajectory Mode] pose_distance_interval: {self.pose_distance_interval}"
        )

    def _init_blocks_mode(self):
        """Initialize blocks mode specific parameters."""
        self.overlap_ratio = self.args.overlap_ratio
        self.block_x_len = self.grid_crop_bbox_max[0] - self.grid_crop_bbox_min[0]
        self.block_y_len = self.grid_crop_bbox_max[1] - self.grid_crop_bbox_min[1]

        # Compute latent voxel size
        self.latent_voxel_size = (
            torch.tensor(self.net_model_diffusion.vae.hparams.grid_crop_bbox_max)
            - torch.tensor(self.net_model_diffusion.vae.hparams.grid_crop_bbox_min)
        ) / (
            torch.tensor(
                self.net_model_diffusion.vae.hparams.network.unet.params.neck_bound
            )
            * 2
        )

        self.latent_ijk_hash_table = {}
        print(
            f"[Blocks Mode] block size: ({self.block_x_len}, {self.block_y_len}), overlap_ratio: {self.overlap_ratio}"
        )

    def get_grid_coord(self, camera_pose_flu):
        """
        Get grid coordinate from camera pose.

        Args:
            camera_pose_flu: [4, 4] camera pose in FLU convention
        """
        cam2world_flu = camera_pose_flu
        camera_pos = cam2world_flu[:3, 3]
        camera_front = cam2world_flu[:3, 0]  # unit
        camera_left = cam2world_flu[:3, 1]  # unit
        camera_up = cam2world_flu[:3, 2]  # unit

        new_grid_pos = (
            camera_pos
            + camera_front
            * (self.grid_crop_bbox_min[0] + self.grid_crop_bbox_max[0])
            / 2
            + camera_left
            * (self.grid_crop_bbox_min[1] + self.grid_crop_bbox_max[1])
            / 2
            + camera_up * (self.grid_crop_bbox_min[2] + self.grid_crop_bbox_max[2]) / 2
        )

        grid2world_flu = torch.clone(camera_pose_flu).double()
        grid2world_flu[:3, 3] = new_grid_pos

        return grid2world_flu

    def _filter_healthy_boxes(self, boxes_3d):
        """
        Filter healthy boxes.
        """
        from pytorch3d.ops.iou_box3d import box3d_overlap

        healthy_box_indices = []
        for i in range(boxes_3d.shape[0]):
            single_box = boxes_3d[i : i + 1].float()
            try:
                box3d_overlap(single_box, single_box, eps=1e-1)
            except ValueError:
                continue
            healthy_box_indices.append(i)
        return boxes_3d[healthy_box_indices]

    def _update_scene_grid(self, grid, semantics, transform):
        """
        Update scene grid with new block.

        Args:
            grid: fvdb.GridBatch, current block grid
            semantics: torch.Tensor, semantic labels
            transform: torch.Tensor [4, 4], transformation matrix
        """
        grid_warp, semantics_warp = transform_grid_and_semantic(
            grid, semantics, transform, subdivide=True
        )
        if self.scene_grid is None:
            self.scene_grid = grid_warp
            self.scene_semantic = semantics_warp
        else:
            self.scene_grid, self.scene_semantic = merge_grid2_to_grid1(
                self.scene_grid, grid_warp, self.scene_semantic, semantics_warp
            )

    def _render_map_condition(self, cond_dict, dense_latents, step, output_path):
        """
        Render map condition visualization.

        Args:
            cond_dict: dict, condition dictionary
            dense_latents: fvdb.nn.VDBTensor, dense latents
            step: int, current step
            output_path: Path, output directory
        """
        map_3d_cond = self.net_model_diffusion.map_3d_cond_model(
            cond_dict["maps_3d"], dense_latents.voxel_sizes
        )
        ijk_canonical = map_3d_cond.grid.ijk.jdata.cpu().numpy()

        colors = ["orange", "cyan", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        geometry_list = []
        embedding_dim = map_3d_cond.data.jdata.shape[-1] // len(
            self.net_model_diffusion.hparams.map_types
        )

        for idx, map_type in enumerate(self.net_model_diffusion.hparams.map_types):
            ijk_map_type_valid = (
                (map_3d_cond.data.jdata[:, idx * embedding_dim] != 0).cpu().numpy()
            )
            if ijk_map_type_valid.sum() == 0:
                continue
            cube_v_i, cube_f_i = single_semantic_voxel_to_mesh(
                ijk_canonical[ijk_map_type_valid],
                voxel_size=dense_latents.voxel_sizes[0].cpu().numpy(),
            )
            geometry = vis.mesh(
                cube_v_i,
                cube_f_i,
                np.array(cmap(idx / len(self.net_model_diffusion.hparams.map_types)))[
                    :3
                ]
                .reshape(1, 3)
                .repeat(cube_v_i.shape[0], axis=0),
            )
            geometry_list.append(geometry)

        scene: render.Scene = vis.show_3d(
            geometry_list,
            show=False,
            up_axis="+Z",
            default_camera_kwargs={
                "pitch_angle": 25.0,
                "fill_percent": 0.5,
                "fov": 40.0,
                "plane_angle": 90,
            },
        )
        img = scene.render_filament()
        imageio.imwrite(output_path / f"{step}_map_cond.jpg", img)

    def _run_diffusion_sampling(self, cond_dict, sdedit_dict, dense_latents):
        """
        Run diffusion sampling.

        Args:
            cond_dict: dict, condition dictionary
            sdedit_dict: dict, sdedit dictionary
            dense_latents: fvdb.nn.VDBTensor, dense latents

        Returns:
            res_feature_set: feature set
            out_vdb_tensor: output VDB tensor
        """
        return self.net_model_diffusion.evaluation_api(
            batch=None,
            grids=dense_latents,
            cond_dict=cond_dict,
            sdedit_dict=sdedit_dict,
            use_ddim=self.args.use_ddim,
            ddim_step=self.args.ddim_step,
            use_ema=self.args.use_ema,
            use_dpm=self.args.use_dpm,
            use_karras=self.args.use_karras,
            solver_order=self.args.solver_order,
            h_stride=self.args.h_stride,
            guidance_scale=self.args.guidance_scale,
        )

    def prepare_from_dataset(self):
        """Prepare data from dataset based on the mode."""
        if self.mode == "trajectory":
            self._prepare_trajectory_mode()
        elif self.mode == "blocks":
            self._prepare_blocks_mode()

    def _prepare_trajectory_mode(self):
        """
        Prepare data for trajectory mode.
        Uses get_wds_data utility to load trajectory and map data.
        """
        wds_data = get_wds_data(
            clip=self.clip,
            key_frame_interval=self.pose_distance_interval,
            webdataset_root=self.args.webdataset_root,
        )

        maps = {
            "road_edge": wds_data["road_edge"],
            "road_line": wds_data["road_line"],
            "road_surface": wds_data["road_surface"],
        }

        boxes_3d = wds_data["boxes_3d"]
        preset_trajectory = wds_data["ego_trajectory"]

        # still in waymo open dataset coordinates
        preset_trajectory_flu = opencv_to_flu(preset_trajectory).to(self.device)

        self.maps_3d = {k: torch.tensor(v, device=self.device) for k, v in maps.items()}
        self.boxes_3d = torch.tensor(
            boxes_3d, device=self.device
        ).double()  # double for bbox!
        self.boxes_3d = self._filter_healthy_boxes(self.boxes_3d)

        # still in waymo open dataset coordinates
        self.camera_trajectory_key_poses_flu = (
            generate_camera_poses_from_batch_trajectory(
                target_pose_num=self.target_pose_num,
                pose_distance_interval=self.pose_distance_interval,
                batch_trajectory=preset_trajectory_flu
                if self.use_preset_trajectory
                else preset_trajectory_flu[:1],
            )
        )

        # still in waymo open dataset coordinates
        self.grid_coord_poses_flu = torch.stack(
            [
                self.get_grid_coord(camera_pose_flu)
                for camera_pose_flu in self.camera_trajectory_key_poses_flu
            ]
        )

    def _prepare_blocks_mode(self):
        """
        Prepare data for blocks mode using get_wds_data utility.
        Loads entire map and divides it into blocks.
        """
        # Load data using utility function (no frame interval filtering for blocks mode)
        wds_data = get_wds_data(
            clip=self.clip,
            key_frame_interval=None,  # Load all frames
            webdataset_root=self.args.webdataset_root,
        )

        # Extract data from wds_data and convert to device
        road_edge = torch.tensor(wds_data["road_edge"], device=self.device)
        road_line = torch.tensor(wds_data["road_line"], device=self.device)
        road_surface = torch.tensor(wds_data["road_surface"], device=self.device)
        boxes_3d = torch.tensor(
            wds_data["boxes_3d"], device=self.device
        ).double()  # double for bbox!

        # Get first camera pose (world coordinate)
        first_pose_opencv = wds_data["ego_trajectory"][0]  # [4, 4]
        first_pose_flu = opencv_to_flu(first_pose_opencv).to(
            self.device
        )  # [4, 4], camera2world

        self.camera_trajectory_key_poses_flu_0 = first_pose_flu
        self.grid_coordinate_key_poses_flu_0 = self.get_grid_coord(first_pose_flu)
        self.camera_to_grid_flu_0 = (
            torch.inverse(self.grid_coordinate_key_poses_flu_0)
            @ self.camera_trajectory_key_poses_flu_0
        )

        world2grid_0 = torch.inverse(self.grid_coordinate_key_poses_flu_0)

        # Transform to grid coordinate
        road_edge = transform_points(road_edge, world2grid_0)
        road_line = transform_points(road_line, world2grid_0)
        road_surface = transform_points(road_surface, world2grid_0)
        boxes_3d = transform_points(boxes_3d.flatten(0, 1), world2grid_0.double()).view(
            -1, 8, 3
        )
        boxes_3d = self._filter_healthy_boxes(boxes_3d)

        # Get scene range
        min_values_x = []
        min_values_y = []
        if road_edge.numel() > 0:
            min_values_x.append(road_edge[:, 0].min())
            min_values_y.append(road_edge[:, 1].min())
        if road_line.numel() > 0:
            min_values_x.append(road_line[:, 0].min())
            min_values_y.append(road_line[:, 1].min())
        if road_surface.numel() > 0:
            min_values_x.append(road_surface[:, 0].min())
            min_values_y.append(road_surface[:, 1].min())
        if boxes_3d.numel() > 0:
            min_values_x.append(boxes_3d[..., 0].min())
            min_values_y.append(boxes_3d[..., 1].min())

        scene_x_min = torch.stack(min_values_x).min()
        scene_y_min = torch.stack(min_values_y).min()

        # Shift scene to make x and y start from 0
        road_edge[:, 0] -= scene_x_min
        road_edge[:, 1] -= scene_y_min
        road_line[:, 0] -= scene_x_min
        road_line[:, 1] -= scene_y_min
        road_surface[:, 0] -= scene_x_min
        road_surface[:, 1] -= scene_y_min
        boxes_3d[..., 0] -= scene_x_min
        boxes_3d[..., 1] -= scene_y_min

        # Update camera pose in new world coordinate
        self.camera_pose_in_new_world_flu_0 = self.camera_to_grid_flu_0.clone()
        self.camera_pose_in_new_world_flu_0[:3, 3] -= torch.tensor(
            [scene_x_min, scene_y_min, 0], device=self.device
        )
        self.camera_pose_in_new_world_flu_0 = self.camera_pose_in_new_world_flu_0.cuda()

        # Store the scene
        self.maps_3d = {
            "road_edge": road_edge,
            "road_line": road_line,
            "road_surface": road_surface,
        }
        self.boxes_3d = boxes_3d

        # Get scene max
        max_values_x = []
        max_values_y = []
        if road_edge.numel() > 0:
            max_values_x.append(road_edge[:, 0].max())
            max_values_y.append(road_edge[:, 1].max())
        if road_line.numel() > 0:
            max_values_x.append(road_line[:, 0].max())
            max_values_y.append(road_line[:, 1].max())
        if road_surface.numel() > 0:
            max_values_x.append(road_surface[:, 0].max())
            max_values_y.append(road_surface[:, 1].max())
        if boxes_3d.numel() > 0:
            max_values_x.append(boxes_3d[..., 0].max())
            max_values_y.append(boxes_3d[..., 1].max())

        scene_x_max = torch.stack(max_values_x).max()
        scene_y_max = torch.stack(max_values_y).max()

        # Generate block centers
        block_center_xs = torch.arange(
            self.block_x_len / 2,
            scene_x_max,
            self.block_x_len * (1 - self.overlap_ratio),
        )
        block_center_ys = torch.arange(
            self.block_y_len / 2,
            scene_y_max,
            self.block_y_len * (1 - self.overlap_ratio),
        )

        self.block_center_xs = block_center_xs
        self.block_center_ys = block_center_ys
        self.block_center_z = 0

        # Get valid blocks (skip empty blocks)
        valid_block_center_xy = []
        for block_center_x in block_center_xs:
            for block_center_y in block_center_ys:
                block_min = torch.tensor(
                    [
                        block_center_x - self.block_x_len / 2,
                        block_center_y - self.block_y_len / 2,
                    ]
                )
                block_max = torch.tensor(
                    [
                        block_center_x + self.block_x_len / 2,
                        block_center_y + self.block_y_len / 2,
                    ]
                )
                block_mask_road_edge = (
                    (road_edge[:, 0] >= block_min[0])
                    & (road_edge[:, 0] <= block_max[0])
                    & (road_edge[:, 1] >= block_min[1])
                    & (road_edge[:, 1] <= block_max[1])
                )
                block_mask_road_surface = (
                    (road_surface[:, 0] >= block_min[0])
                    & (road_surface[:, 0] <= block_max[0])
                    & (road_surface[:, 1] >= block_min[1])
                    & (road_surface[:, 1] <= block_max[1])
                )
                if block_mask_road_edge.sum() > 0 and block_mask_road_surface.sum() > 0:
                    valid_block_center_xy.append([block_center_x, block_center_y])
                else:
                    print(f"Skip block {block_center_x}, {block_center_y}")

        self.valid_block_center_xy = valid_block_center_xy

        # Prepare relative poses
        poses = torch.eye(4).unsqueeze(0).repeat(len(valid_block_center_xy), 1, 1)
        poses[:, :2, 3] = torch.tensor(valid_block_center_xy)
        self.each_grid_poses_in_new_world = poses.cuda()

    def create_cond_dict_trajectory(self, ego_pose):
        """
        Create condition dict for trajectory mode.
        Transform conditions to the grid coordinate.

        Args:
            ego_pose: FLU convention
        """
        grid2world = self.get_grid_coord(ego_pose)
        world2grid = torch.inverse(grid2world)

        # Transform maps_3d and boxes_3d to grid coord
        maps_3d_grid = {
            k: transform_points(v, world2grid) for k, v in self.maps_3d.items()
        }
        boxes_3d_grid = transform_points(
            self.boxes_3d.flatten(0, 1), world2grid.double()
        ).reshape(-1, 8, 3)

        # Batch size is 1, list collate
        cond_dict = {}
        cond_dict["maps_3d"] = {k: [v.float()] for k, v in maps_3d_grid.items()}
        cond_dict["boxes_3d"] = [boxes_3d_grid.float()]

        return cond_dict

    def create_cond_and_sdedit_dict_blocks(self, valid_block_idx):
        """
        Create condition and sdedit dict for blocks mode.

        Args:
            valid_block_idx: the index of the valid block
        """
        block_center_x, block_center_y = self.valid_block_center_xy[valid_block_idx]

        # Prepare sdedit dict
        if valid_block_idx == 0:
            sdedit_dict = {}
        else:
            # Get ijk for this block, retrieve the latent from hash table if exists
            dense_latent = self.net_model_diffusion.create_dense_latents(
                batch_size=1, h_stride=self.args.h_stride
            )
            assert (
                dense_latent.voxel_sizes[0] == self.latent_voxel_size.cuda()
            ).all(), "Invalid dense latent voxel size"
            dense_latent_ijk = dense_latent.ijk.jdata  # canonical
            dense_latent_ijk_world = dense_latent_ijk.clone()
            dense_latent_ijk_world[:, 0] += torch.round(
                block_center_x / self.latent_voxel_size[0]
            ).int()
            dense_latent_ijk_world[:, 1] += torch.round(
                block_center_y / self.latent_voxel_size[1]
            ).int()

            assert (dense_latent_ijk_world[:, 0] >= 0).all() and (
                dense_latent_ijk_world[:, 1] >= 0
            ).all(), "Invalid dense latent ijk"

            # Get the latent from the hash table
            latent_ijk = []
            latent_feature = []
            for ijk_canonical, ijk_world in zip(
                dense_latent_ijk, dense_latent_ijk_world
            ):
                ijk_world_key = tuple(ijk_world.tolist())
                if ijk_world_key in self.latent_ijk_hash_table:
                    latent_ijk.append(ijk_canonical)
                    latent_feature.append(self.latent_ijk_hash_table[ijk_world_key])

            if len(latent_ijk) == 0:
                sdedit_dict = {}
            else:
                latent_ijk = torch.stack(latent_ijk)
                latent_feature = torch.stack(latent_feature)

                # Build fvdb.nn.VDBTensor as the previous latents
                prev_latents_grid = fvdb.gridbatch_from_ijk(
                    latent_ijk,
                    voxel_sizes=dense_latent.voxel_sizes,
                    origins=dense_latent.origins,
                )
                latent_ijk_index = prev_latents_grid.ijk_to_inv_index(latent_ijk).jdata
                assert (latent_ijk_index >= 0).all() and len(
                    torch.unique(latent_ijk_index)
                ) == len(latent_ijk), "Invalid latent ijk index"
                prev_latents_feature = latent_feature[latent_ijk_index]

                prev_latents = fvdb.nn.VDBTensor(
                    prev_latents_grid, JaggedTensor([prev_latents_feature])
                )
                spatial_movement = torch.eye(4, device=prev_latents_grid.device)
                sdedit_dict = {
                    "prev_latents": prev_latents,
                    "spatial_movement": spatial_movement,
                }

        # Prepare condition dict for current block
        this_block_center = torch.tensor(
            [block_center_x, block_center_y, 0], device=self.device
        )

        cond_dict = {}
        maps_3d = deepcopy(self.maps_3d)
        maps_3d = {k: v.float() - this_block_center for k, v in maps_3d.items()}
        cond_dict["maps_3d"] = {}
        cond_dict["boxes_3d"] = [self.boxes_3d.float() - this_block_center]

        # Filter maps_3d within the block
        for k, v in maps_3d.items():
            mask = (
                (v[:, 0] >= self.grid_crop_bbox_min[0])
                & (v[:, 0] < self.grid_crop_bbox_max[0])
                & (v[:, 1] >= self.grid_crop_bbox_min[1])
                & (v[:, 1] < self.grid_crop_bbox_max[1])
            )
            cond_dict["maps_3d"][k] = [v[mask]]

        return cond_dict, sdedit_dict

    def fill_latent_ijk_hash_table(self, valid_block_idx, current_latents):
        """
        Fill latent ijk hash table for blocks mode.

        Args:
            valid_block_idx: the index of the valid block
            current_latents: fvdb.nn.VDBTensor, the current latents
        """
        block_center_x, block_center_y = self.valid_block_center_xy[valid_block_idx]

        current_latents_ijk = current_latents.grid.ijk.jdata
        current_latents_feature = current_latents.data.jdata

        current_latents_ijk_world = current_latents_ijk.clone()
        current_latents_ijk_world[:, 0] += torch.round(
            block_center_x / self.latent_voxel_size[0]
        ).int()
        current_latents_ijk_world[:, 1] += torch.round(
            block_center_y / self.latent_voxel_size[1]
        ).int()

        for ijk_world, feature in zip(
            current_latents_ijk_world, current_latents_feature
        ):
            ijk_world_key = tuple(ijk_world.tolist())
            if ijk_world_key not in self.latent_ijk_hash_table:
                self.latent_ijk_hash_table[ijk_world_key] = feature

    @torch.inference_mode()
    def extrapolate(self):
        """Main extrapolation method that dispatches to mode-specific implementation."""
        if self.mode == "trajectory":
            self._extrapolate_trajectory()
        elif self.mode == "blocks":
            self._extrapolate_blocks()

    @torch.inference_mode()
    def _extrapolate_trajectory(self):
        """Extrapolation implementation for trajectory mode."""
        print("[Trajectory Mode] Start extrapolation...")
        self.prepare_from_dataset()
        sdedit_dict = {}

        for step in tqdm(range(len(self.camera_trajectory_key_poses_flu))):
            cond_dict = self.create_cond_dict_trajectory(
                self.camera_trajectory_key_poses_flu[step]
            )
            dense_latents = self.net_model_diffusion.create_dense_latents(
                batch_size=1, h_stride=self.args.h_stride
            )

            # Run diffusion sampling
            res_feature_set, out_vdb_tensor = self._run_diffusion_sampling(
                cond_dict, sdedit_dict, dense_latents
            )

            # Update sdedit dict for next iteration
            if "current_latents" in sdedit_dict:
                sdedit_dict["prev_latents"] = sdedit_dict.pop("current_latents")
                sdedit_dict["spatial_movement"] = get_relative_transforms(
                    self.grid_coord_poses_flu[step : step + 2]
                )[-1]

            # Get current block grid and semantics
            grid = out_vdb_tensor.grid
            semantics = res_feature_set.semantic_features[-1].data.jdata.argmax(dim=-1)
            self.grid_semantic_pairs.append((grid, semantics))

            # Transform and update scene grid to first camera's flu coordinate
            current_grid_to_first_camera_flu = (
                torch.inverse(self.camera_trajectory_key_poses_flu[0])
                @ self.grid_coord_poses_flu[step]
            ).float()
            self._update_scene_grid(grid, semantics, current_grid_to_first_camera_flu)

            # Render voxels
            render_and_save(
                [(self.scene_grid, self.scene_semantic)],
                self.output_path / f"{step}.jpg",
            )

            # Render condition map
            self._render_map_condition(cond_dict, dense_latents, step, self.output_path)

            # Render whole maps (first step only)
            if step == 0:
                rendered = offscreen_map_voxel_render(cond_dict["maps_3d"])
                imageio.imwrite(self.output_path / f"{step}_map.jpg", rendered)

            # Save to pt file
            torch.save(
                {
                    "points": self.scene_grid.to("cpu"),
                    "semantics": self.scene_semantic.to("cpu"),
                },
                self.output_path / f"{step}.pt",
            )

    @torch.inference_mode()
    def _extrapolate_blocks(self):
        """Extrapolation implementation for blocks mode."""
        print("[Blocks Mode] Start extrapolation...")
        self.prepare_from_dataset()

        for step, valid_block_idx in enumerate(
            tqdm(range(len(self.valid_block_center_xy)))
        ):
            cond_dict, sdedit_dict = self.create_cond_and_sdedit_dict_blocks(
                valid_block_idx
            )
            print(
                f"Process block {valid_block_idx} with center {self.valid_block_center_xy[valid_block_idx]}"
            )

            # Prepare the input
            cond_dict = batch2device(cond_dict, self.device)
            sdedit_dict = batch2device(sdedit_dict, self.device)

            # Run diffusion sampling
            dense_latents = self.net_model_diffusion.create_dense_latents(
                batch_size=1, h_stride=self.args.h_stride
            )
            res_feature_set, out_vdb_tensor = self._run_diffusion_sampling(
                cond_dict, sdedit_dict, dense_latents
            )

            # Fill the hash table
            if "current_latents" in sdedit_dict:
                self.fill_latent_ijk_hash_table(
                    valid_block_idx, sdedit_dict.pop("current_latents")
                )

            # Save for visualization
            grid = out_vdb_tensor.grid
            semantics = res_feature_set.semantic_features[-1].data.jdata.argmax(dim=-1)
            self.grid_semantic_pairs.append((grid, semantics))

            # Transform to the first camera's flu coordinate and update scene
            current_grid_to_first_camera_flu = (
                torch.inverse(self.camera_pose_in_new_world_flu_0)
                @ self.each_grid_poses_in_new_world[step].double()
            ).float()
            self._update_scene_grid(grid, semantics, current_grid_to_first_camera_flu)

            # Render voxels
            render_and_save(
                [(self.scene_grid, self.scene_semantic)],
                self.output_path / f"{step}.jpg",
            )

            # Render maps (only this block)
            rendered = offscreen_map_voxel_render(cond_dict["maps_3d"])
            imageio.imwrite(self.output_path / f"{step}_map.jpg", rendered)

            # Save to pt file
            torch.save(
                {
                    "points": self.scene_grid.to("cpu"),
                    "semantics": self.scene_semantic.to("cpu"),
                },
                self.output_path / f"{step}.pt",
            )

            # Render entire map (first step only)
            if step == 0:
                rendered = offscreen_map_voxel_render(deepcopy(self.maps_3d))
                imageio.imwrite(self.output_path / "map.jpg", rendered)


def render_and_save(grid_and_semantic, output_path):
    """Helper function to render and save voxel grid."""
    rendered = offscreen_voxel_list_to_mesh_renderer(
        grid_and_semantic,
        extend_direction="x",
        default_camera_kwargs={
            "pitch_angle": 80.0,
            "fill_percent": 0.9,
            "fov": 40.0,
            "plane_angle": 90,
            "w": 1024,
            "h": 1024,
        },
    )
    imageio.imwrite(output_path, rendered)


if __name__ == "__main__":
    known_args = get_parser().parse_known_args()[0]

    # Check that either local or wandb config is provided
    resume_from_local = known_args.local_config is not None
    resume_from_wandb = (
        known_args.wandb_config is not None
        and known_args.wandb_config.startswith("wdb:")
    )
    assert resume_from_local or resume_from_wandb, (
        "Either local_config and local_checkpoint_path or wandb_config (starting with 'wdb:') should be provided"
    )

    # Load model from local or wandb
    hparam_update = None  # Can add hparam updates if needed
    if resume_from_local:
        logger.info(f"Loading from local config: {known_args.local_config}")
        logger.info(
            f"Loading from local checkpoint: {known_args.local_checkpoint_path}"
        )
        net_model_diffusion, _, _ = create_model_from_local_config(
            config_path=known_args.local_config,
            checkpoint_path=known_args.local_checkpoint_path,
            hparam_update=hparam_update,
        )
        model_identifier = known_args.local_config.split("/")[-1].split(".")[0]
    else:
        logger.info(f"Loading from wandb: {known_args.wandb_config}")
        net_model_diffusion, _, _ = create_model_from_args(
            known_args.wandb_config + ":last",
            known_args,
            get_parser(),
            hparam_update=hparam_update,
        )
        model_identifier = known_args.wandb_config.replace("/", "_").replace("wdb:", "")

    # Create generator and run
    generator = VoxelWorldGenerator(
        clip=known_args.clip, args=known_args, net_model_diffusion=net_model_diffusion
    )
    generator.extrapolate()

    print(f"Output path: {generator.output_path}")
