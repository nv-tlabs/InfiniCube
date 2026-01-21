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

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.conv_encoder import (
    ConvEncoder,
)
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.dav2_encoder import (
    DAV2Encoder,
)
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.dinov2_encoder import (
    DinoWrapper,
)
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.plucker_conv_patch_encoder import (
    PluckerConvPatchEncoder,
)


class UnifiedEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        assert isinstance(
            self.hparams.encoder.encoder_modules, omegaconf.listconfig.ListConfig
        )
        encoder_modules = self.hparams.encoder.encoder_modules
        self.encoders = nn.ModuleDict()
        # ! project head

        if "convnext" in encoder_modules:
            self.encoders["convnext"] = eval(self.hparams.convnext_model_name)(
                pretrained=self.hparams.convnext_pretrain_weights
            ).features[: self.hparams.convnext_layer]
        if "conv" in encoder_modules:
            self.encoders["conv"] = ConvEncoder(self.hparams.encoder.conv_params)
        if "dino" in encoder_modules:
            self.encoders["dino"] = DinoWrapper(self.hparams.encoder.dino_params)
        if "dav2" in encoder_modules:
            self.encoders["dav2"] = DAV2Encoder(self.hparams.encoder.dav2_params)
        if "plucker_conv_patch" in encoder_modules:
            self.encoders["plucker_conv_patch"] = PluckerConvPatchEncoder(self.hparams)

    def resize_reshape_projection_input(
        self, input_image, input_sky_mask, input_effective_mask, intrinsics
    ):
        """
        input shape: [B, N, H, W, 3], [B, N, H, W, 1], [B, N, H, W, 1]
        return shape: [B, N, 3, H', W'], [B, N, 1, H', W'], [B, N, 1, H', W']
        """
        # cat them together for higher effeciency
        input_chunk = torch.cat(
            [input_image, input_sky_mask, input_effective_mask], dim=4
        )  # [B, N, H, W, 3+1+1]
        b, n, h, w, _ = input_chunk.size()

        # [B, N, H, W, 3+1+1] -> [B * N, 3+1+1, H, W]. Necessary for the project head
        input_chunk = rearrange(input_chunk, "b n h w c -> (b n) c h w")
        aspect_ratio = w / h  # width / height
        assert (
            intrinsics[0, 0, 5].int().item() == h
            and intrinsics[0, 0, 4].int().item() == w
        )

        # note for the scene, h != w, we make the height to be given input_projection_size
        if self.hparams.encoder.resize_projection_input:
            if isinstance(self.hparams.encoder.input_projection_size, int):
                resize_h = self.hparams.encoder.input_projection_size
                resize_w = int(
                    self.hparams.encoder.input_projection_size * aspect_ratio
                )
            else:
                resize_h, resize_w = self.hparams.encoder.input_projection_size

            input_chunk = F.interpolate(
                input_chunk,
                (resize_h, resize_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # the intrinsics should be updated. careful about the float division of w and h.
            downsample_ratio_h = intrinsics[0, 0, 5].int().item() / resize_h
            downsample_ratio_w = intrinsics[0, 0, 4].int().item() / resize_w

            intrinsics_image = intrinsics.clone()
            intrinsics_image[:, :, 0] /= downsample_ratio_w
            intrinsics_image[:, :, 1] /= downsample_ratio_h
            intrinsics_image[:, :, 2] /= downsample_ratio_w
            intrinsics_image[:, :, 3] /= downsample_ratio_h
            intrinsics_image[:, :, 4] = resize_w
            intrinsics_image[:, :, 5] = resize_h
        else:
            self.hparams.input_projection_size = input_image.size(2)
            intrinsics_image = intrinsics

        # [B * N, 3, H, W], [B * N, 1, H, W], [B * N, 1, H, W]
        input_image, input_sky_mask, input_effective_mask = input_chunk.split(
            [3, 1, 1], dim=1
        )

        # rearrange to [B, N, 3, H, W], [B, N, 1, H, W], [B, N, 1, H, W]
        input_image = rearrange(input_image, "(b n) c h w -> b n c h w", n=n)
        input_sky_mask = rearrange(input_sky_mask, "(b n) c h w -> b n c h w", n=n)
        input_effective_mask = rearrange(
            input_effective_mask, "(b n) c h w -> b n c h w", n=n
        )

        return input_image, input_sky_mask, input_effective_mask, intrinsics_image

    def forward(self, batch, skybox=None) -> dict:
        assert self.hparams.use_fvdb_loader is True  # ! v3 version default fvdb loader

        # store unet feature and skybox feature
        imgenc_output = {}

        input_intrinsics = torch.stack(
            batch[DS.IMAGES_INPUT_INTRINSIC]
        )  # [B, N, 6], 6 is fx fy cx cy w h

        input_image = torch.stack(batch[DS.IMAGES_INPUT])  # [B, N, H, W, C]
        input_mask = torch.stack(batch[DS.IMAGES_INPUT_MASK])

        # in the scene, we store the foreground mask in the 0-1 channel
        input_sky_mask = ((input_mask[:, :, :, :, 0:1]) == 0).float()
        # dynamic object mask and the hood mask
        input_effective_mask = (
            (input_mask[:, :, :, :, 1:2] * input_mask[:, :, :, :, 2:3]) > 0
        ).float()

        # !! resize the input image
        # [B, N, 3, H, W], [B, N, 1, H, W], [B, N, 1, H, W], [B, N, 6]
        input_image, input_sky_mask, input_effective_mask, input_intrinsics = (
            self.resize_reshape_projection_input(
                input_image, input_sky_mask, input_effective_mask, input_intrinsics
            )
        )

        # [B, N, 4, 4]
        input_pose = torch.stack(batch[DS.IMAGES_INPUT_POSE])
        camera_info = {"intrinsic": input_intrinsics, "pose": input_pose}

        # !! encode the input image, note that no sky mask applied on the encoded feat!!
        for encoder_name, encoder in self.encoders.items():
            imgenc_output[encoder_name] = encoder(
                input_image, camera_info=camera_info, batch=batch
            )

        # !! add original RGB image
        imgenc_output["original_rgb"] = input_image
        imgenc_output["camera_info"] = camera_info
        imgenc_output["input_effective_mask"] = (
            input_effective_mask  # [B, N, 1, H, W], dynamic object mask and the hood mask
        )

        return imgenc_output
