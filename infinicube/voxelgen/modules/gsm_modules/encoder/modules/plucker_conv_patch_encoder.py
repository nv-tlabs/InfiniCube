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
import torch.nn as nn
from einops import rearrange

from infinicube.voxelgen.utils.render_util import create_plucker_coords_torch


class PluckerConvPatchEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        plucker_conv_patch_params = hparams.encoder.plucker_conv_patch_params

        # determine the input channels
        in_channels = 3
        self.concat_plucker_coords = plucker_conv_patch_params.concat_plucker_coords
        self.concat_depth = plucker_conv_patch_params.concat_depth

        if self.concat_plucker_coords:
            in_channels += 6
        if self.concat_depth:
            self.depth_embed_method = plucker_conv_patch_params.depth_embed_method
            in_channels += 1

        self.patch_size = plucker_conv_patch_params.patch_size
        self.use_LN = plucker_conv_patch_params.use_LN

        self.conv_patchify_out_dim = plucker_conv_patch_params.conv_patchify_out_dim
        self.conv_patchify = nn.Conv2d(
            in_channels=in_channels,  # RGB and plucker
            out_channels=self.conv_patchify_out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.conv_patchify_layernorm = nn.LayerNorm(self.conv_patchify_out_dim)

    def embed_depth(self, depth):
        """
        depth: metric depth map ?
            torch.Tensor of shape (B, N, H, W, 1)
        """
        if self.depth_embed_method == "none":
            return depth
        elif self.depth_embed_method == "normalize_255":
            return depth / 255.0
        elif self.depth_embed_method == "normalize_patch_0_1":
            pass

    def _init_weight(self):
        self.conv_patchify.weight.data.fill_(0)
        self.conv_patchify.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        """
        Args:
            x: image tensor of shape (B*N, C, H, W) or (B, N, C, H, W)
            kwargs:
                - camera_info: dict, include
                    intrinsic: torch.Tensor of shape (B, N, 6)
                    pose: torch.Tensor of shape (B, N, 4, 4)

        Returns:
            imgenc_package: dict, include
                x_patches: tensor of shape (..., conv_patchify_out_dim, H // patch_size, W // patch_size)
                camera_info: dict
        """
        mv_input = False
        assert "camera_info" in kwargs

        if len(x.shape) == 5:
            B, N = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            mv_input = True

        pose = kwargs["camera_info"]["pose"].view(-1, 4, 4)
        intrinsic = kwargs["camera_info"]["intrinsic"].view(-1, 6)

        # prepare input with extra data
        concat_list = [x]

        # generate plucker
        if self.concat_plucker_coords:
            plucker_coords = create_plucker_coords_torch(
                pose, intrinsic
            )  # [B*N, H, W, 6]
            plucker_coords = rearrange(plucker_coords, "b h w c -> b c h w")
            concat_list.append(plucker_coords)

        # concat depth
        if self.concat_depth:
            concat_list.append(self.embed_depth(kwargs["depth"]))

        # cat with rgb channel
        x = torch.cat(concat_list, dim=1)  # [B*N, 9/10, H, W]

        # shape [B*N, conv_patchify_out_dim, H // patch_size, W // patch_size]
        x_patches = self.conv_patchify(x)
        # LN on the channel dimension. duplicate with transformer's pre LN?
        if self.use_LN:
            x_patches = self.conv_patchify_layernorm(
                x_patches.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)

        if mv_input:
            x_patches = rearrange(x_patches, "(b n) c h w -> b n c h w", b=B, n=N)

        return x_patches
