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

import os
from collections import OrderedDict

import torch


def merge_dict1_to_dict2(dict1, dict2):
    """
    same key will be overwritten by dict2
    """
    merged_dict = OrderedDict()
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    symm_diff_set = dict1_keys & dict2_keys

    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlapping modules in two state dict=======")
    print(*overlap_module, sep="\n")

    print("=======Exact overlapping params in two state dict=======")
    print(*symm_diff_set, sep="\n")

    print("=======Overlapping but different params in two state dict=======")
    for param in symm_diff_set:
        if not torch.equal(dict1[param], dict2[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in dict1_keys:
        merged_dict[key] = dict1[key]

    for key in dict2_keys:
        merged_dict[key] = dict2[key]

    return merged_dict


def merge_two_checkpoint(ckpt1_path, ckpt2_path, target_ckpt_path):
    if os.path.exists(target_ckpt_path):
        target_ckpt_path_backup = target_ckpt_path + ".backup"
        os.system(f"cp {target_ckpt_path} {target_ckpt_path_backup}")

    ckpt1 = torch.load(ckpt1_path)
    ckpt2 = torch.load(ckpt2_path)

    merged_state_dict = merge_dict1_to_dict2(ckpt1["state_dict"], ckpt2["state_dict"])
    target_ckpt = {
        "state_dict": merged_state_dict,
    }
    torch.save(target_ckpt, target_ckpt_path)


def merge_two_wandb_checkpoints(wandb_exp1, wandb_exp2, target_ckpt_path):
    from infinicube.voxelgen.utils.wandb_util import get_wandb_run

    wandb_exp1 = f"{wandb_exp1}:last"
    wandb_exp2 = f"{wandb_exp2}:last"
    wandb_run1, ckpt1_path = get_wandb_run(wandb_exp1)
    print(f"For {wandb_exp1}, found checkpoint at {ckpt1_path}")
    wandb_run2, ckpt2_path = get_wandb_run(wandb_exp2)
    print(f"For {wandb_exp2}, found checkpoint at {ckpt2_path}")
    merge_two_checkpoint(ckpt1_path, ckpt2_path, target_ckpt_path)
    print(f"Merged checkpoint saved to {target_ckpt_path}")


if __name__ == "__main__":
    wandb_exp1 = "wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama"
    wandb_exp2 = "wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator"
    target_ckpt_path = (
        "checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt"
    )
    merge_two_wandb_checkpoints(wandb_exp1, wandb_exp2, target_ckpt_path)
