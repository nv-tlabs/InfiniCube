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
This evaluation script will use a dataloader to get the batch data
and use evalution_api to get the diffused result.

# example: resume from local config and checkpoint
python infinicube/inference/voxel_vae.py none \
    --local_config infinicube/voxelgen/configs/vae_64x64x64_height_down2_vs02_dense_residual.yaml \
    --local_checkpoint_path checkpoints/vae_epoch7_step6250.ckpt

# example: resume from wandb
python infinicube/inference/voxel_vae.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/vae_64x64x64_height_down2_vs02_dense_residual
"""

from pathlib import Path

import fvdb
import fvdb.nn
import imageio.v3 as imageio
import torch
from pycg import exp
from tqdm import tqdm

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.utils.common_util import (
    batch2device,
    create_model_from_args,
    create_model_from_local_config,
    get_default_parser,
)
from infinicube.voxelgen.utils.voxel_util import offscreen_voxel_list_to_mesh_renderer

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
        default="visualization/voxel_vae/",
        help="Output directory.",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for output directory."
    )
    parser.add_argument(
        "--val_starting_frame", type=int, default=50, help="Starting frame."
    )

    return parser


@torch.inference_mode()
def diffusion_and_save(net_model_vae, dataloader, saving_dir, known_args):
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = batch2device(batch, net_model_vae.device)
        output_dict = net_model_vae(batch, {})

        # save / visualize out_vdb_tensor
        grid = output_dict["tree"][0]
        semantic_prob = output_dict["semantic_features"][-1].data.jdata
        semantic = torch.argmax(semantic_prob, dim=-1)

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
        grid_semantic_pairs = [
            (grid, semantic),  # pred
            (batch[DS.INPUT_PC], batch[DS.GT_SEMANTIC][0]),  # GT
        ]
        rendered = offscreen_voxel_list_to_mesh_renderer(grid_semantic_pairs)
        imageio.imwrite(saving_dir / f"{batch_idx}_pred_gt.jpg", rendered)
        print(f"Save visualization to {saving_dir / f'{batch_idx}_pred_gt.jpg'}")


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

    if known_args.suffix != "":
        known_args.suffix = "_" + known_args.suffix

    if resume_from_local:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.local_config.split("/")[-1].split(".")[0]
            / (
                known_args.split
                + f"_starting_at_{known_args.val_starting_frame}"
                + known_args.suffix
            )
        )
        net_model_vae, _, _ = create_model_from_local_config(
            config_path=known_args.local_config,
            checkpoint_path=known_args.local_checkpoint_path,
            hparam_update=hparam_update,
        )
    else:
        saving_dir = (
            Path(known_args.output_root)
            / known_args.wandb_config.replace("/", "_")
            / (
                known_args.split
                + f"_starting_at_{known_args.val_starting_frame}"
                + known_args.suffix
            )
        )
        net_model_vae, _, _ = create_model_from_args(
            known_args.wandb_config + ":last",
            known_args,
            get_parser(),
            hparam_update=hparam_update,
        )

    saving_dir.mkdir(parents=True, exist_ok=True)
    net_model_vae.cuda()

    if known_args.split == "test":
        dataset_kwargs = net_model_vae.hparams.test_kwargs
    else:
        dataset_kwargs = net_model_vae.hparams.train_kwargs

    # update dataset_kwargs if needed
    dataset_kwargs["split"] = "not_train_to_ensure_reproducibility"

    if known_args.split == "test":
        dataloader = net_model_vae.test_dataloader()
    else:
        dataloader = net_model_vae.train_dataloader()

    diffusion_and_save(net_model_vae, dataloader, saving_dir, known_args)


if __name__ == "__main__":
    main()
