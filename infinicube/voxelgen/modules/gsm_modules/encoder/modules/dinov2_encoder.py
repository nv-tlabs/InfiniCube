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
from torchvision import transforms
from transformers import Dinov2Backbone

from infinicube.voxelgen.modules.basic_modules import ResBlock as ResBlock2D

imagenet_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class DinoWrapper(nn.Module):
    """
    Dino v2 wrapper using huggingface transformer implementation.
    """

    def __init__(self, dino_params):
        super().__init__()
        self.dino_params = dino_params

        model_name = self.dino_params.dino_model_name
        freeze = self.dino_params.dino_freeze

        self.out_dim_list = self.dino_params.out_dim_list
        self.out_upsample_list = self.dino_params.out_upsample_list
        self.out_downsample_list = self.dino_params.out_downsample_list
        self.out_use_gn_list = self.dino_params.out_use_gn_list

        self.freeze = freeze
        self.model = Dinov2Backbone.from_pretrained(model_name)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dino_dim = self.model.encoder.config.hidden_size

        self.upsampler = []
        in_dim = dino_dim
        for i, out_dim in enumerate(self.out_dim_list):
            self.upsampler.append(
                ResBlock2D(
                    in_dim,
                    out_channels=out_dim,
                    up=self.out_upsample_list[i],
                    down=self.out_downsample_list[i],
                    use_gn=self.out_use_gn_list[i],
                )
            )
            in_dim = out_dim
        self.upsampler = nn.Sequential(*self.upsampler)

        if freeze:
            self._freeze()

    def _forward(self, x):
        """
        Args:
            x: image tensor of shape (N_, C, H, W)

        Returns:
            high_level_feature: tensor of shape (N_, C, H, W)
        """
        inputs = self.normalize(x)
        with torch.no_grad():
            dino_outputs = self.model(inputs)
        feature_maps = dino_outputs.feature_maps[-1]
        high_level_feature = self.upsampler(feature_maps)

        return high_level_feature

    def forward(self, x, **kwargs):
        """
        x: image tensor of shape (B*N, C, H, W) or (B, N, C, H, W)

        Returns:
            high_level_feature: tensor of shape (B*N, C, H, W) or (B, N, C, H, W)
            unprojected_mask: tensor of shape (B*N, H, W) or (B, N, H, W) if 'image_mask' is in kwargs
        """
        mv_input = False

        if self.freeze:
            self.model.eval()

        if len(x.shape) == 5:
            B, N = x.shape[:2]
            x = rearrange(x, "b n c h w -> (b n) c h w")
            mv_input = True

        if "image_mask" in kwargs:
            mask = kwargs["image_mask"]
            if mv_input:
                mask = rearrange(mask, "b n c h w -> (b n) c h w")

            # assume masking only happens at height dimension, i.e. each column of the mask is the same
            assert torch.all(
                mask[:, :, :, :1].expand(-1, -1, -1, mask.shape[3]) == mask
            ), "Assume masking is for height dimension"
            assert torch.all(mask[:, :, 0, :] == 1), (
                "Assume masking happens at the bottom"
            )

            mask_start_ys = torch.sum(mask[:, 0, :, 0], dim=1)  # (B*N,)
            mask_start_ys_unique = torch.unique(mask_start_ys)
            mask_start_height_collection = []

            for mask_start_y in mask_start_ys_unique:
                this_height_mask_idx = mask_start_ys == mask_start_y  # [B*N,]
                mask_start_y_patch_size = (
                    mask_start_y // self.model.config.patch_size
                ) * self.model.config.patch_size
                mask_start_y_patch_size = int(mask_start_y_patch_size)
                x_cropped = x[this_height_mask_idx][
                    :, :, :mask_start_y_patch_size
                ]  # [N_this_height, C, H_cropped, W]
                high_level_feature_cropped = self._forward(x_cropped)

                mask_start_height_collection.append(
                    (this_height_mask_idx, high_level_feature_cropped)
                )

            # merge the high level features, according to each height mask idx. Also padding 0 to the bottom
            feature_map_W = high_level_feature_cropped.shape[3]
            # according to feature_map_W, we can infer the maximum feature_map_H
            feature_map_max_H = (mask.shape[2] / mask.shape[3]) * feature_map_W
            feature_map_max_H = int(feature_map_max_H)

            high_level_feature = torch.zeros(
                (
                    mask.shape[0],
                    high_level_feature_cropped.shape[1],
                    feature_map_max_H,
                    feature_map_W,
                ),
                device=x.device,
            )  # [B*N, C, H_, W_]
            unprojected_mask = torch.zeros(
                (mask.shape[0], feature_map_max_H, feature_map_W), device=x.device
            )  # [B*N, H_, W_]

            for (
                this_height_mask_idx,
                high_level_feature_cropped,
            ) in mask_start_height_collection:
                high_level_feature[
                    this_height_mask_idx, :, : high_level_feature_cropped.shape[2], :
                ] = high_level_feature_cropped
                unprojected_mask[
                    this_height_mask_idx, : high_level_feature_cropped.shape[2], :
                ] = 1

            if mv_input:
                high_level_feature = rearrange(
                    high_level_feature, "(b n) c h w -> b n c h w", b=B, n=N
                )
                unprojected_mask = rearrange(
                    unprojected_mask, "(b n) h w -> b n h w", b=B, n=N
                )

            return high_level_feature, unprojected_mask

        else:
            high_level_feature = self._forward(x)

            if mv_input:
                high_level_feature = rearrange(
                    high_level_feature, "(b n) c h w -> b n c h w", b=B, n=N
                )

            return high_level_feature

    def _freeze(self):
        print("======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False


if __name__ == "__main__":
    pass
