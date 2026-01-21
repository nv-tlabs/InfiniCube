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
from omegaconf import OmegaConf
from torchvision import transforms

from infinicube.utils.depth_utils import align_inv_depth_to_depth
from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.basic_modules import ResBlock as ResBlock2D
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.depth_anything_v2_hf import (
    DepthAnythingForDepthEstimation,
)
from infinicube.voxelgen.utils.voxel_util import get_depth_from_voxel

imagenet_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class DAV2Encoder(nn.Module):
    def __init__(self, dav2_hparams):
        super().__init__()

        self.dav2_hparams = dav2_hparams
        depth_anything_model = (
            self.dav2_hparams.depth_anything_model
        )  # 'depth-anything/Depth-Anything-V2-Large-hf'
        self.target_input_size = OmegaConf.to_container(
            self.dav2_hparams.target_input_size
        )
        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            depth_anything_model
        )
        self.freeze = self.dav2_hparams.model_freeze

        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.upsampler = []
        in_dim = self.model.config.fusion_hidden_size
        patch_size = self.model.config.backbone_config.patch_size

        self.patch_num_h = self.target_input_size[0] // patch_size
        self.patch_num_w = self.target_input_size[1] // patch_size

        for i, out_dim in enumerate(self.dav2_hparams.out_dim_list):
            self.upsampler.append(
                ResBlock2D(
                    in_dim,
                    out_channels=out_dim,
                    up=self.dav2_hparams.out_upsample_list[i],
                    down=self.dav2_hparams.out_downsample_list[i],
                    use_gn=self.dav2_hparams.out_use_gn_list[i],
                )
            )
            in_dim = out_dim

        self.upsampler = nn.Sequential(*self.upsampler)

    def forward(self, x, **kwargs):
        """
        x: image tensor of shape (B*N, C, H, W) or (B, N, C, H, W)

        Returns:
            high_level_feature: tensor of shape (B*N, C, H, W) or (B, N, C, H, W)
        """
        mv_input = False

        if self.freeze:
            self.model.eval()

        if len(x.shape) == 5:
            B, N = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            mv_input = True

        # resize to target size
        x = nn.functional.interpolate(
            x, size=self.target_input_size, mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            if x.shape[0] > 10:
                # divide into smaller batches to avoid OOM
                batch_size = 10
                fused_feature_map_list = []
                for i in range(0, x.shape[0], batch_size):
                    fused_feature_map = self.model.forward_fusion(
                        pixel_values=imagenet_normalization(x[i : i + batch_size])
                    )[-1]
                    fused_feature_map_list.append(fused_feature_map)
                fused_feature_map = torch.cat(fused_feature_map_list, dim=0)
            else:
                fused_feature_map = self.model.forward_fusion(
                    pixel_values=imagenet_normalization(x)
                )[-1]  # [B*N, C, patch_h_num * 8, patch_w_num * 8]

        high_level_feature = self.upsampler(fused_feature_map)

        if mv_input:
            high_level_feature = rearrange(
                high_level_feature, "(b n) c h w -> b n c h w", b=B, n=N
            )

        return high_level_feature

    @staticmethod
    def get_voxel_depth(batch, is_input=False):
        # voxel rendered depth
        B = batch[DS.INPUT_PC].grid_count

        voxel_rendered_depth_list = []
        if is_input:
            for b in range(B):
                voxel_rendered_depth = get_depth_from_voxel(
                    batch[DS.IMAGES_INPUT_POSE][b],
                    batch[DS.IMAGES_INPUT_INTRINSIC][b],
                    batch[DS.INPUT_PC][b],
                )

                voxel_rendered_depth_list.append(voxel_rendered_depth)
        else:
            for b in range(B):
                voxel_rendered_depth = get_depth_from_voxel(
                    batch[DS.IMAGES_POSE][b],
                    batch[DS.IMAGES_INTRINSIC][b],
                    batch[DS.INPUT_PC][b],
                )

                voxel_rendered_depth_list.append(voxel_rendered_depth)

        # [B, N, H, W, 1]
        voxel_rendered_depth = torch.stack(voxel_rendered_depth_list, dim=0)

        return voxel_rendered_depth

    @staticmethod
    def rectify_depth_anything_v2_depth_inv(batch):
        """
        align the depth-anything v2 inverse depth to voxel rendered depth

        Note that the voxel depth is very inaccurate. So the aligned depth is also inaccurate.
        """
        voxel_rendered_depth = DAV2Encoder.get_voxel_depth(batch)

        # reference depth, [B, N, H, W, 1]
        source_inv_depth = torch.stack(batch[DS.IMAGES_INPUT_DEPTH])

        images_input_mask = torch.stack(batch[DS.IMAGES_INPUT_MASK])
        foreground_mask_from_seg = images_input_mask[..., 0:1].float()
        foreground_mask_from_grid = images_input_mask[..., 3:4].float()

        # if we come here, we must model the mid-ground
        close_range = foreground_mask_from_seg * foreground_mask_from_grid
        mid_ground = foreground_mask_from_seg - close_range
        mask_for_align = (
            close_range * (source_inv_depth > 0) * (voxel_rendered_depth > 0)
        )

        # rectified depth anything v2 inverse depth to voxel rendered inverse depth
        B, N, H, W = source_inv_depth.shape[:4]

        voxel_rendered_depth = voxel_rendered_depth.view(B * N, H, W)
        mask_for_align = mask_for_align.view(B * N, H, W)
        source_inv_depth = source_inv_depth.view(B * N, H, W)
        rectified_depth = [
            align_inv_depth_to_depth(
                source_inv_depth[i], voxel_rendered_depth[i], mask_for_align[i]
            )
            for i in range(B * N)
        ]
        rectified_depth = torch.stack(rectified_depth)
        rectified_depth = torch.where(source_inv_depth > 0, rectified_depth, 0)
        rectified_depth = rectified_depth.view(B, N, H, W, 1)

        return rectified_depth
