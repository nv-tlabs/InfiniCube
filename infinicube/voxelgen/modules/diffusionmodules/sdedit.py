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

import torch
from fvdb import JaggedTensor
from fvdb.nn import VDBTensor


def sdedit_prepare_input(
    noisy_latents, sdedit_dict, noise_scheduler, scale_factor, timestep
):
    """
    Args:
        noisy_latents: fvdb.nn.VDBTensor

        sdedit_dict: dict
            - 'prev_latents': fvdb.nn.VDBTensor, previous latents. Already (1. / scale_factor * prev_latents)
            - 'spatial_movement': torch.Tensor shape [4, 4], current grid in previous grid's coordinate (current2prev)

        noise_scheduler: usually DDPMScheduler
            add noise to prev_latents

        scale_factor: float or None
            a scale factor multiplied to the previous latent before adding noise

        timestep: int

    Returns:
        updated_noisy_latents: fvdb.nn.VDBTensor
            For the overlapping region, we replace the noisy_latents with previous latents
            from sdedit_dict. But we need to shift the previous latents according to the spatial_movement.
    """
    if "spatial_movement" not in sdedit_dict or "prev_latents" not in sdedit_dict:
        # logger.info("no SDEdit is performed at timestep {}".format(timestep))
        return noisy_latents

    # calcuate the ijk offset for previous latents, to fill in the current grid
    current2prev = sdedit_dict["spatial_movement"]
    prev2current = torch.inverse(current2prev).float()
    previous_latents = sdedit_dict["prev_latents"]

    prev_grid_xyz_prev = previous_latents.grid.grid_to_world(
        previous_latents.grid.ijk.jdata.float()
    ).jdata  # [N, 3]
    prev_grid_xyz_prev = torch.cat(
        [prev_grid_xyz_prev, torch.ones_like(prev_grid_xyz_prev[:, :1])], dim=1
    )  # [N, 4]
    prev_grid_xyz_current = (prev2current @ prev_grid_xyz_prev.t()).t()[:, :3]  # [N, 3]
    prev_grid_ijk_current = (
        noisy_latents.grid.world_to_grid(prev_grid_xyz_current)
        .jdata.round()
        .to(torch.int32)
    )  # [N, 3]
    previous_latents_feature = previous_latents.data.jdata

    if scale_factor is not None:
        previous_latents_feature = scale_factor * previous_latents_feature

    # add noise to previous latents
    noise = torch.randn_like(previous_latents_feature)
    previous_latents_feature = noise_scheduler.add_noise(
        previous_latents_feature, noise, timestep
    )

    # in current grid, convert ijk to index to update the noisy_latents feature
    prev_grid_index_current = noisy_latents.grid.ijk_to_index(
        prev_grid_ijk_current
    ).jdata
    prev_grid_index_current_mask = prev_grid_index_current >= 0

    prev_grid_index_current = prev_grid_index_current[prev_grid_index_current_mask]
    previous_latents_feature = previous_latents_feature[prev_grid_index_current_mask]

    # update noisy_latents feature with previous latents
    noisy_latents_feature = noisy_latents.data.jdata

    # prev_grid_index_current is the index, previous_latents_feature is its corresponding feature
    # fill the noisy_latents_feature with previous_latents_feature
    noisy_latents_feature[prev_grid_index_current] = previous_latents_feature

    return VDBTensor(
        grid=noisy_latents.grid, data=JaggedTensor([noisy_latents_feature])
    )
