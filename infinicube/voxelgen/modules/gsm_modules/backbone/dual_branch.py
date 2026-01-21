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

from infinicube.voxelgen.modules.gsm_modules.backbone.pixel_branch import Pure2DUNet
from infinicube.voxelgen.modules.gsm_modules.backbone.voxel_branch import Pure3DUnet

def list2tuple(d):
    """
    d is a dictionary some values of which are lists. This function converts these lists to tuples.
    """
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = tuple(v)
    return d


class DualBranchUNet(nn.Module):
    def __init__(self, params):
        super(DualBranchUNet, self).__init__()
        self.params = params

        # help us training separately
        self.use_3d = params.use_3d
        self.use_2d = params.use_2d
        assert self.use_3d or self.use_2d, (
            "At least one of 3d or 2d should be used, or together"
        )

        if self.use_3d:
            backbone_3d_target = params.backbone_3d_target
            backbone_3d_params = params.backbone_3d_params
            self.backbone_3d = eval(backbone_3d_target)(**backbone_3d_params)
        else:
            self.backbone_3d = None

        if self.use_2d:
            self.backbone_2d = Pure2DUNet(params.backbone_2d_params)
        else:
            self.backbone_2d = None

    def forward(self, batch, imgenc_output):
        """
        Returns:
            network_output = {'decoded_gaussians': decoded_gaussians}
        """

        # 3d backbone
        if self.use_3d:
            network_output_3d = self.backbone_3d(batch, imgenc_output)
            decoded_gs_3d = network_output_3d["decoded_gaussians"]

        if self.use_2d:
            infer_with_3d_branch = self.use_3d and not self.training
            network_output_2d = self.backbone_2d(
                batch, imgenc_output, infer_with_3d_branch=infer_with_3d_branch
            )
            decoded_gs_2d = network_output_2d["decoded_gaussians"]

        if self.use_3d and self.use_2d:
            decoded_gs = [
                torch.cat([decoded_gs_3d[i], decoded_gs_2d[i]], dim=0)
                for i in range(len(decoded_gs_3d))
            ]
        elif self.use_3d:
            decoded_gs = decoded_gs_3d
        elif self.use_2d:
            decoded_gs = decoded_gs_2d
        else:
            raise ValueError("At least one of 3d or 2d should be used, or together")

        network_output = {"decoded_gaussians": decoded_gs}

        return network_output
