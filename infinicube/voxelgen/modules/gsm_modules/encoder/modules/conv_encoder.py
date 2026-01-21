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

from infinicube.voxelgen.modules.basic_modules import ResBlock


class ConvEncoder(nn.Module):
    def __init__(self, conv_hparams):
        super().__init__()

        n_filter_list = conv_hparams.n_filter_list
        n_stride_list = conv_hparams.n_stride_list
        n_padding_list = conv_hparams.n_padding_list
        n_kernel_list = conv_hparams.n_kernel_list

        if hasattr(conv_hparams, "n_residual_list"):
            n_residual_list = conv_hparams.n_residual_list
            print("if residual, kernel_size must be 3")
        else:
            n_residual_list = [0] * len(n_kernel_list)

        if hasattr(conv_hparams, "n_use_gn_list"):
            n_use_gn_list = conv_hparams.n_use_gn_list
        else:
            n_use_gn_list = [True] * len(n_kernel_list)

        self.project_head = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=n_filter_list[i],
                        out_channels=n_filter_list[i + 1],
                        kernel_size=n_kernel_list[i],
                        stride=n_stride_list[i],
                        padding=n_padding_list[i],
                    ),
                    nn.BatchNorm2d(n_filter_list[i + 1]),
                    nn.ReLU(),
                )
                if not n_residual_list[i]
                else ResBlock(
                    n_filter_list[i],
                    dropout=0.0,
                    out_channels=n_filter_list[i + 1],
                    use_conv=False,
                    dims=2,
                    use_checkpoint=False,
                    up=False,
                    down=True if n_stride_list[i] == 2 else False,
                    use_gn=n_use_gn_list[i],
                )
                for i in range(len(n_filter_list) - 1)
            ]
        )
        self.project_head.add_module(
            "conv_1x1",
            torch.nn.Conv2d(
                in_channels=n_filter_list[-1],
                out_channels=conv_hparams.conv_encoder_out_dim,
                stride=1,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x, **kwargs):
        """
        x: image tensor of shape (B*N, C, H, W) or (B, N, C, H, W)
        """
        mv_input = False

        if len(x.shape) == 5:
            B, N = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            mv_input = True

        x = self.project_head(x)
        if mv_input:
            x = rearrange(x, "(b n) c h w -> b n c h w", b=B, n=N)

        return x
