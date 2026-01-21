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
We only consider the second stage, feedfoward model of GSM.
We use ground-truth voxels, and save the decoded gaussians / rendered results


If you trained voxel branch and pixel branch, you can merge them into a single checkpoint by running
>>> python infinicube/voxelgen/utils/model_merge_util.py 

# resume from local config and checkpoint, dual branch
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt

# resume from local config and checkpoint, voxel branch (local checkpoint is merged from voxel branch and pixel branch)
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 


# resume from local config and checkpoint, pixel branch (local checkpoint is merged from voxel branch and pixel branch)
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 


# resume from wandb, voxel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama \
    --skybox_resolution 1024

# resume from wandb, pixel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator 

"""

from pathlib import Path

import fvdb
import fvdb.nn
import imageio.v3 as imageio
import numpy as np
import torch
import torchvision
from pycg import exp
from tqdm import tqdm

from infinicube.utils.gaussian_io_utils import save_splat_file
from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.utils.common_util import (
    batch2device,
    create_model_from_args,
    create_model_from_local_config,
    get_default_parser,
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
        default="visualization/feedforward_reconstruction/",
        help="Output directory.",
    )
    parser.add_argument(
        "--save_img_separately",
        action="store_true",
        help="save pred image separately in one folder",
    )
    parser.add_argument(
        "--save_gs", action="store_true", help="save gaussians to .pkl file"
    )
    parser.add_argument(
        "--input_frame_offsets",
        type=int,
        nargs="+",
        default=None,
        help="Input frame offsets.",
    )
    parser.add_argument(
        "--val_starting_frame", type=int, default=50, help="Starting frame."
    )
    parser.add_argument(
        "--skybox_resolution", type=int, default=512, help="Skybox panorama resolution."
    )
    return parser


@torch.inference_mode()
def render_and_save_gsm(
    net_model_gsm,
    dataloader,
    saving_dir,
    img_reorder,
    save_img_together,
    save_img_separately,
    save_gaussians,
):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_gsm.device)
        renderer_output, network_output = net_model_gsm.forward(batch)
        gt_package = net_model_gsm.loss.prepare_resized_gt(batch)
        vis_images_dict = net_model_gsm.loss.assemble_visualization(
            gt_package, renderer_output
        )

        if save_img_together or save_img_separately:
            gt_images = vis_images_dict["gt_images"][0]  # [N, H, W, 3]
            pd_images = vis_images_dict["pd_images"][0]  # [N, H, W, 3]

            # reorder for better visualization. [N, H, W, 3]
            n_frames = gt_images.shape[0] // len(img_reorder)
            pd_images_reorder = torch.cat(
                [x[img_reorder] for x in torch.chunk(pd_images, n_frames, dim=0)], dim=0
            )
            gt_images_reorder = torch.cat(
                [x[img_reorder] for x in torch.chunk(gt_images, n_frames, dim=0)], dim=0
            )

        if save_img_together:
            pd_images_reorder_resize = pd_images_reorder.permute(0, 3, 1, 2)
            gt_images_reorder_resize = gt_images_reorder.permute(0, 3, 1, 2)
            torchvision.utils.save_image(
                pd_images_reorder_resize,
                saving_dir / f"{batch_idx}_pred_images.jpg",
                nrow=len(img_reorder),
            )
            torchvision.utils.save_image(
                gt_images_reorder_resize,
                saving_dir / f"{batch_idx}_gt_images.jpg",
                nrow=len(img_reorder),
            )

        if save_img_separately:
            out_folder = saving_dir / f"{batch_idx}"
            out_folder.mkdir(parents=True, exist_ok=True)
            pd_images_numpy = (pd_images.clamp(0, 1).cpu().numpy() * 255).astype(
                np.uint8
            )
            gt_images_numpy = (gt_images.clamp(0, 1).cpu().numpy() * 255).astype(
                np.uint8
            )
            sup_frame_idxs = batch[DS.SHAPE_NAME][0].split("_with_sup_frames_")[-1]
            sup_frame_idxs = sup_frame_idxs.split("_")

            for idx, frame_name in enumerate(sup_frame_idxs):
                for view_idx in range(len(img_reorder)):
                    imageio.imwrite(
                        out_folder / f"{frame_name}_{view_idx}.jpg",
                        pd_images_numpy[idx * len(img_reorder) + view_idx],
                    )

            for idx, frame_name in enumerate(sup_frame_idxs):
                for view_idx in range(len(img_reorder)):
                    imageio.imwrite(
                        out_folder / f"{frame_name}_{view_idx}_gt.jpg",
                        gt_images_numpy[idx * len(img_reorder) + view_idx],
                    )

        if save_gaussians:
            decoded_gaussians = network_output["decoded_gaussians"][0]
            assert decoded_gaussians.shape[1] == 14
            output_path = saving_dir / f"{batch_idx}_rgb_gaussians.pkl"
            save_splat_file(decoded_gaussians, output_path.as_posix())

            # save skybox representation
            net_model_gsm.skybox.save_skybox(network_output, output_path)

            # save renderer decoder (no decoder for RGBRenderer)
            net_model_gsm.renderer.save_decoder(output_path)


def main():
    known_args = get_parser().parse_known_args()[0]

    resume_from_local = known_args.local_config is not None
    resume_from_wandb = (
        known_args.wandb_config is not None
        and known_args.wandb_config.startswith("wdb:")
    )
    assert resume_from_local or resume_from_wandb, (
        "Either local_config and local_checkpoint_path or wandb_config (starting with 'wdb:') should be provided"
    )

    hparam_update = {
        "skybox_resolution": known_args.skybox_resolution,
        "skybox_forward_sky_only": True,
        "train_val_num_workers": 0,
    }

    if resume_from_local:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.local_config.split("/")[-1].split(".")[0]
            / f"{known_args.split}_starting_at_{known_args.val_starting_frame}"
        )
        net_model_gsm, _, global_step_gsm = create_model_from_local_config(
            config_path=known_args.local_config,
            checkpoint_path=known_args.local_checkpoint_path,
            hparam_update=hparam_update,
        )
    else:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.wandb_config.replace("/", "_")
            / f"{known_args.split}_starting_at_{known_args.val_starting_frame}"
        )
        net_model_gsm, _, global_step_gsm = create_model_from_args(
            known_args.wandb_config + ":last",
            known_args,
            get_parser(),
            hparam_update=hparam_update,
        )

    saving_dir.mkdir(parents=True, exist_ok=True)
    net_model_gsm.cuda()

    if known_args.split == "test":
        dataset_kwargs = net_model_gsm.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_gsm.hparams.train_kwargs

    # change data set configs if needed
    if known_args.input_frame_offsets is not None:
        input_frame_offsets = known_args.input_frame_offsets
    else:
        input_frame_offsets = dataset_kwargs["input_frame_offsets"]

    # update dataset_kwargs
    dataset_kwargs["split"] = "not_train_to_ensure_reproducibility"
    dataset_kwargs["val_starting_frame"] = known_args.val_starting_frame
    dataset_kwargs["input_frame_offsets"] = input_frame_offsets
    dataset_kwargs["sup_slect_ids"] = dataset_kwargs["sup_slect_ids"]
    dataset_kwargs["sup_frame_offsets"] = dataset_kwargs["sup_frame_offsets"]
    dataset_kwargs["n_image_per_iter_sup"] = None

    # reorder for better visualization
    if len(dataset_kwargs["sup_slect_ids"]) == 3:
        img_reorder = [1, 0, 2]
    elif len(dataset_kwargs["sup_slect_ids"]) == 5:
        img_reorder = [3, 1, 0, 2, 4]
    elif len(dataset_kwargs["sup_slect_ids"]) == 1:
        img_reorder = [0]
    else:
        raise NotImplementedError

    if known_args.split == "test":
        dataloader = net_model_gsm.test_dataloader()
    else:
        dataloader = net_model_gsm.train_dataloader()

    render_and_save_gsm(
        net_model_gsm,
        dataloader,
        saving_dir,
        img_reorder,
        save_img_together=True,
        save_img_separately=known_args.save_img_separately,
        save_gaussians=known_args.save_gs,
    )


if __name__ == "__main__":
    main()
