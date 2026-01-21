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
Generate a single chunk of voxel world by diffusion, conditioned on the map

# example: resume from local config and checkpoint
python infinicube/inference/voxel_generation_single_chunk.py none \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt

# example: resume from wandb
python infinicube/inference/voxel_generation_single_chunk.py none \
    --use_ema --use_ddim --ddim_step 100 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/diffusion_64x64x64_dense_vs02_map_cond
"""

from pathlib import Path

import fvdb
import fvdb.nn
import imageio.v3 as imageio
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from pycg import exp, render, vis
from tqdm import tqdm

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.utils.color_util import semantic_from_points
from infinicube.voxelgen.utils.common_util import (
    batch2device,
    create_model_from_args,
    create_model_from_local_config,
    get_default_parser,
)
from infinicube.voxelgen.utils.voxel_util import (
    offscreen_voxel_to_mesh_render_for_vae_decoded_list,
)

fvdb.nn.SparseConv3d.backend = "igemm_mode1"


def get_parser():
    parser = exp.ArgumentParserX(
        base_config_path="infinicube/voxelgen/configs/default/param.yaml",
        parents=[get_default_parser()],
    )
    # either local_config and local_checkpoint_path or wandb_config should be provided
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
        "--nosync", action="store_true", help="Do not synchronize nas even if forced."
    )
    parser.add_argument(
        "--wandb_base",
        type=str,
        default="./wandb/",
        help="Path to wandb base directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on. test or train",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="visualization/voxel_generation_single_chunk/",
        help="Output directory.",
    )
    parser.add_argument(
        "--val_starting_frame", type=int, default=50, help="Starting frame."
    )

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


@torch.inference_mode()
def diffusion_and_save(net_model_diffusion, dataloader, saving_dir, known_args):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_diffusion.device)

        res_feature_set, out_vdb_tensor = net_model_diffusion.evaluation_api(
            batch,
            grids=None,
            use_ddim=known_args.use_ddim,
            ddim_step=known_args.ddim_step,
            use_ema=known_args.use_ema,
            use_dpm=known_args.use_dpm,
            use_karras=known_args.use_karras,
            solver_order=known_args.solver_order,
            h_stride=known_args.h_stride,
            guidance_scale=known_args.guidance_scale,
        )

        grid = out_vdb_tensor.grid
        semantic_prob = res_feature_set.semantic_features[-1].jdata  # [n_voxel, 23]
        semantic = semantic_prob.argmax(dim=-1)  # [n_voxel, ]

        # save pred
        save_dict = {"points": grid.to("cpu"), "semantics": semantic.to("cpu")}
        torch.save(save_dict, saving_dir / f"{batch_idx}.pt")
        print(f"Save to {saving_dir / f'{batch_idx}.pt'}")

        # save GT
        gt_save_dict = {
            "points": batch[DS.INPUT_PC].to("cpu"),
            "semantics": batch[DS.GT_SEMANTIC][0].to("cpu"),
        }
        torch.save(gt_save_dict, saving_dir / f"{batch_idx}_gt.pt")
        print(f"Save to {saving_dir / f'{batch_idx}_gt.pt'}")

        # render the pred & GT
        vae_decoded_list = []
        vae_decoded_list.append((res_feature_set, out_vdb_tensor))  # pred
        rendered = offscreen_voxel_to_mesh_render_for_vae_decoded_list(vae_decoded_list)
        imageio.imwrite(saving_dir / f"{batch_idx}_pred.jpg", rendered)

        if DS.MAPS_3D in batch:
            # render map
            colors = ["orange", "cyan", "red"]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

            pc_list = []
            maps_3d = batch[
                DS.MAPS_3D
            ]  # dict, the key is map types, the value is [n_points, 3]

            rendered_images = []

            # this is saying to ego, not grid coordinate
            grid_crop_bbox_min = torch.tensor(
                net_model_diffusion.hparams.grid_crop_bbox_min,
                device=net_model_diffusion.device,
            )
            grid_crop_bbox_max = torch.tensor(
                net_model_diffusion.hparams.grid_crop_bbox_max,
                device=net_model_diffusion.device,
            )

            # convert to grid coordinate
            grid_crop_bbox_max = (grid_crop_bbox_max - grid_crop_bbox_min) / 2
            grid_crop_bbox_min = -grid_crop_bbox_max

            # breakpoint()
            # crop the points
            for map_type, map_points in maps_3d.items():
                map_points = map_points[0]  # extract the first sample
                map_points = map_points.to(torch.int32)
                # discard points outside the grid
                map_points = map_points[
                    (map_points >= grid_crop_bbox_min).all(dim=1)
                    & (map_points < grid_crop_bbox_max).all(dim=1)
                ]
                # only keep unique points
                map_points = torch.unique(map_points, dim=0)

                maps_3d[map_type] = map_points.float()

            # prepare pc_list
            for idx, map_type in enumerate(maps_3d):
                if maps_3d[map_type].shape[0] == 0:
                    continue
                map_color = (
                    np.array(cmap(idx / len(maps_3d)))[:3]
                    .reshape(1, 3)
                    .repeat(maps_3d[map_type].shape[0], axis=0)
                )
                map_pc = vis.pointcloud(
                    pc=maps_3d[map_type].to("cpu").numpy(), color=map_color
                )
                pc_list.append(map_pc)

            # render the map
            for plane_angle in [90, 180, 270, 0]:
                scene: render.Scene = vis.show_3d(
                    pc_list,
                    show=False,
                    up_axis="+Z",
                    default_camera_kwargs={
                        "pitch_angle": 45.0,
                        "fill_percent": 0.7,
                        "fov": 40.0,
                        "plane_angle": plane_angle,
                    },
                )
                img = scene.render_filament()
                rendered_images.append(img)

            rendered_images = np.concatenate(rendered_images, axis=1)
            imageio.imwrite(saving_dir / f"{batch_idx}_map.jpg", rendered_images)

            # save the map as fvdb grid as well (for easier visualization)
            origins = grid.origins
            voxel_sizes = grid.voxel_sizes
            ijk_collection = []
            point_collection = []
            semantic_collection = []
            for idx, (map_type, map_points) in enumerate(maps_3d.items()):
                if map_points.shape[0] == 0:
                    continue

                point_collection.append(map_points)
                semantic_collection.append(
                    torch.tensor(
                        [idx] * map_points.shape[0],
                        dtype=torch.int32,
                        device=map_points.device,
                    )
                )

                grid = fvdb.gridbatch_from_points(
                    map_points, voxel_sizes=voxel_sizes, origins=origins
                )
                ijk_collection.append(grid.ijk.jdata)

            ijk_collection = torch.cat(ijk_collection, dim=0)
            merged_grid = fvdb.gridbatch_from_ijk(
                ijk_collection, voxel_sizes=voxel_sizes, origins=origins
            )

            merged_grid_semantic = semantic_from_points(
                merged_grid.grid_to_world(merged_grid.ijk.float()).jdata,
                torch.cat(point_collection, dim=0),
                torch.cat(semantic_collection, dim=0),
            )

            # save pt
            save_dict = {
                "points": merged_grid.to("cpu"),
                "semantics": merged_grid_semantic.to("cpu"),
            }

            saving_dir_map = Path(str(saving_dir) + "_map_grid")
            saving_dir_map.mkdir(parents=True, exist_ok=True)

            torch.save(save_dict, saving_dir_map / f"{batch_idx}.pt")


def main():
    known_args = get_parser().parse_known_args()[0]
    hparam_update = {"batch_size": 1, "batch_size_val": 1, "train_val_num_workers": 0}

    resume_from_local = known_args.local_config is not None
    resume_from_wandb = (
        known_args.wandb_config is not None
        and known_args.wandb_config.startswith("wdb:")
    )
    assert resume_from_local or resume_from_wandb, (
        "Either local_config and local_checkpoint_path or wandb_config (starting with 'wdb:') should be provided"
    )

    if resume_from_local:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.local_config.split("/")[-1].split(".")[0]
            / (known_args.split + f"_starting_at_{known_args.val_starting_frame}")
        )
        net_model_diffusion, _, _ = create_model_from_local_config(
            config_path=known_args.local_config,
            checkpoint_path=known_args.local_checkpoint_path,
            hparam_update=hparam_update,
        )
    else:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.wandb_config.replace("/", "_")
            / (known_args.split + f"_starting_at_{known_args.val_starting_frame}")
        )
        net_model_diffusion, _, _ = create_model_from_args(
            known_args.wandb_config + ":last",
            known_args,
            get_parser(),
            hparam_update=hparam_update,
        )

    saving_dir.mkdir(parents=True, exist_ok=True)
    net_model_diffusion.cuda()

    if known_args.split == "test":
        dataset_kwargs = net_model_diffusion.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_diffusion.hparams.train_kwargs

    dataset_kwargs["split"] = "not_train_to_ensure_reproducibility"

    if known_args.split == "test":
        dataloader = net_model_diffusion.test_dataloader()
    else:
        dataloader = net_model_diffusion.train_dataloader()

    diffusion_and_save(net_model_diffusion, dataloader, saving_dir, known_args)


if __name__ == "__main__":
    main()
