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

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import functional as F

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.sky_modules.skybox_panorama_full import (
    get_world_directions_latlong,
)
from infinicube.voxelgen.utils.embedder_util import NeRFEncoding, SHEncoding
from infinicube.voxelgen.utils.render_util import (
    create_rays_from_intrinsic_torch_batch,
    to_opengl,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ModulatedLinearLayer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        condition_channels=768,
        out_channels=3,
        view_pos_embedding="mlp",
        view_pos_embedding_config=None,
    ):
        """
        Args:
            hidden_channels: hidden channels inside this module for high frequency features
        """
        super().__init__()

        if view_pos_embedding == "mlp":
            self.pos_emb = nn.Linear(in_channels, hidden_channels)
            self.pos_emb.weight.data.zero_()
        elif view_pos_embedding == "sincos_xyz":
            # assert hidden_channels divisible by 2 and in_channels
            assert hidden_channels % (2 * in_channels) == 0
            self.pos_emb = NeRFEncoding(
                in_channels,
                num_frequencies=int(hidden_channels / 2 / in_channels),
                min_freq_exp=-2,
                max_freq_exp=2,
                include_input=False,
            )
        elif view_pos_embedding == "spherical_harmonics":
            self.pos_emb = SHEncoding(level=view_pos_embedding_config.level)
            assert hidden_channels == self.pos_emb.out_channels
        else:
            raise NotImplementedError(
                f"pos_embedding {view_pos_embedding} not implemented"
            )

        self.norm = nn.LayerNorm(
            hidden_channels, elementwise_affine=False, eps=1e-6
        )  # only feature channel
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, c):
        """
        Args:
            x: [B, N, 3], normalized xyz coordinates of view direction
            c: [B, 768], sky features for batches.
                method1: can be pooling of skybox features
                method2: can use sky token to cross attention with skybox features

        Returns:
            x: [B, N, 3], color of the view direction
        """
        x_emb = self.pos_emb(x)  # high frequency features, [B, N, 3] -> [B, N, 64]
        c = self.condition_mapping(c.squeeze(1))  # sky token. [B, 768] -> [B, 64]
        shift, scale = self.adaLN_modulation(c).chunk(
            2, dim=-1
        )  # [B, 64] -> [B, 64], [B, 64]
        x_emb_shape = x_emb.shape  # [B, N, 64]
        x_emb = modulate(
            self.norm(x_emb.reshape(x_emb_shape[0], -1, x_emb_shape[-1])), shift, scale
        )  # [B, N, 64] -> [B, N, 64]
        x_colors = self.output(x_emb)  # [B, N, 64] -> [B, N, 3]
        x_colors = x_colors.reshape(*x_emb_shape[:-1], -1)
        return x_colors


class SkyboxMlpModulator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # only useful when self.sky_token_encoding_type == 'cross attention'
        self.sky_query = nn.Parameter(
            torch.randn(1, 1, hparams.skybox_mlp_modulator.embed_dim)
        )
        self.modulator = ModulatedLinearLayer(
            in_channels=3,  # normalized xyz coordinates
            hidden_channels=hparams.skybox_mlp_modulator.hidden_channels,
            condition_channels=hparams.skybox_mlp_modulator.embed_dim,
            out_channels=hparams.skybox_mlp_modulator.out_channels,
            view_pos_embedding=hparams.skybox_mlp_modulator.modulator_pos_embedding,
        )

        # encoding the sky token using the whole image or only the sky region
        self.skybox_forward_sky_only = hparams.skybox_forward_sky_only
        # key of feature name. usually conv
        self.skybox_feature_source = hparams.skybox_feature_source

        if self.skybox_feature_source == "original_rgb":
            self.skybox_in_dim = 3
        elif self.skybox_feature_source == "conv":
            self.skybox_in_dim = hparams.encoder.conv_params.conv_encoder_out_dim

        # pooling / attention / transformer
        self.sky_token_encoding_type = (
            hparams.skybox_mlp_modulator.sky_token_encoding_type
        )

        self.patch_size = hparams.skybox_mlp_modulator.patch_size

        self.patch_embedder = nn.Conv2d(
            self.skybox_in_dim,
            hparams.skybox_mlp_modulator.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if self.sky_token_encoding_type == "attention":
            self.mha = nn.MultiheadAttention(
                hparams.skybox_mlp_modulator.embed_dim,
                hparams.skybox_mlp_modulator.num_heads,
                batch_first=True,
            )

        if self.sky_token_encoding_type == "transformer":
            # image positional encoding. 'none', 'sincos_xyz' (use normalized xyz)
            self.sky_image_embedder_type = (
                hparams.skybox_mlp_modulator.sky_image_embedder_type
            )

            if self.sky_image_embedder_type == "none":
                self.pos_embedder = None
            elif self.sky_image_embedder_type == "sincos_xyz":
                assert hparams.skybox_mlp_modulator.embed_dim % 6 == 0
                self.pos_embedder = NeRFEncoding(
                    3,
                    num_frequencies=int(hparams.skybox_mlp_modulator.embed_dim / 2 / 3),
                    min_freq_exp=-2,
                    max_freq_exp=2,
                    include_input=False,
                )
            elif self.sky_image_embedder_type == "mlp":
                self.pos_embedder = nn.Linear(3, hparams.skybox_mlp_modulator.embed_dim)
                # initialize the weight to be 0
                self.pos_embedder.weight.data.zero_()
            else:
                raise NotImplementedError(
                    f"sky_image_embedder_type {self.sky_image_embedder_type} not implemented"
                )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hparams.skybox_mlp_modulator.embed_dim,
                nhead=hparams.skybox_mlp_modulator.num_heads,
                dim_feedforward=hparams.skybox_mlp_modulator.transformer.dim_feedforward,
                activation=hparams.skybox_mlp_modulator.transformer.activation,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=hparams.skybox_mlp_modulator.transformer.num_layers,
            )

    def encode_sky_feature(self, batch, imgenc_output):
        camera_pose_stack = torch.stack(batch[DS.IMAGES_INPUT_POSE])  # [B, N, 4, 4]
        camera_pose_stack[:, :, :3, 3] = 0  # remove translation
        intrinsics = torch.stack(batch[DS.IMAGES_INPUT_INTRINSIC])  # [B, N, 6]

        """
        if model the midground, sky should be pure sky, inverse of foreground_from_seg

        if not model the midground, sky should include sky and midground, inverse of (foreground_from_seg x foreground_from_grid)
        """
        if self.hparams.model_midground:
            sky_mask = (
                torch.stack(batch[DS.IMAGES_INPUT_MASK])[:, :, :, :, 0:1] == 0
            )  # [B, N, H, W, 1]
        else:
            sky_mask = (
                torch.stack(batch[DS.IMAGES_INPUT_MASK])[:, :, :, :, 0:1]
                * torch.stack(batch[DS.IMAGES_INPUT_MASK])[:, :, :, :, 3:4]
            ) == 0  # [B, N, H, W, 1]

        sky_mask = sky_mask.permute(
            0, 1, 4, 2, 3
        ).float()  # [B, N, 1, H, W], 1 is the sky

        image_or_feature = imgenc_output[self.skybox_feature_source]  # [B, N, C, H, W]
        B, N, image_or_feature_dim, H, W = image_or_feature.shape

        # we need to reshape sky_mask to match the shape of image_or_feature
        sky_mask = (
            F.interpolate(
                sky_mask.flatten(0, 1).float(), size=(H, W), mode="nearest"
            ).view(B, N, 1, H, W)
            > 0
        )  # [B, N, 1, H, W]

        # warning, if the shape is not divisible by patch_size, the last row and column will be ignored
        input_patches = self.patch_embedder(
            image_or_feature.flatten(0, 1)
        )  # [B * N, C, H_patch_num, W_patch_num]
        H_patch_num, W_patch_num = input_patches.shape[-2:]
        input_patches_sky_ratio = F.avg_pool2d(
            sky_mask.flatten(0, 1).float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B * N, 1, H_patch_num, W_patch_num]
        valid_sky_patch_mask = (
            input_patches_sky_ratio > 0.75
        )  # [B * N, 1, H_patch_num, W_patch_num]

        input_patches = input_patches.view(B, N, -1, H_patch_num, W_patch_num).permute(
            0, 1, 3, 4, 2
        )  # [B, N, H_patch_num, W_patch_num, C]
        input_patches_flatten = input_patches.reshape(
            B, N * H_patch_num * W_patch_num, -1
        )  # [B, N * H_patch_num * W_patch_num, C]
        valid_sky_patch_mask_flatten = valid_sky_patch_mask.view(
            B, N, H_patch_num, W_patch_num
        ).reshape(
            B, N * H_patch_num * W_patch_num, 1
        )  # [B, N * H_patch_num * W_patch_num, 1]

        if self.sky_token_encoding_type == "pooling":
            # pooling the sky region. If no valid sky patch, sky token is 0
            sky_tokens = (input_patches_flatten * valid_sky_patch_mask_flatten).sum(
                dim=1
            ) / (valid_sky_patch_mask_flatten.sum(dim=1) + 1e-6)  # [B, C]

        elif self.sky_token_encoding_type == "attention":
            # sky query will cross attention to the sky patches using nn.MultiheadAttention
            sky_query = self.sky_query.repeat(B, 1, 1)
            sky_tokens, _ = self.mha(
                sky_query,
                input_patches_flatten,
                input_patches_flatten,
                key_padding_mask=~valid_sky_patch_mask_flatten.squeeze(-1),
            )

        elif self.sky_token_encoding_type == "transformer":
            # do not include non-sky tokens here.
            # use for-loop to fix inconsistent sky patch number

            # get intrinsic for the patchified image
            resize_h, resize_w = (
                intrinsics[:, :, 5] / H_patch_num,
                intrinsics[:, :, 4] / W_patch_num,
            )
            intrinsics_patchified = intrinsics.clone()
            intrinsics_patchified[:, :, [0, 2, 4]] /= resize_w.unsqueeze(-1)
            intrinsics_patchified[:, :, [1, 3, 5]] /= resize_h.unsqueeze(-1)

            # [B*N, 3], [B*N, H_patch_num, W_patch_num, 3]
            camera_origin, ray_d_world = create_rays_from_intrinsic_torch_batch(
                camera_pose_stack.flatten(0, 1), intrinsics_patchified.flatten(0, 1)
            )
            ray_d_opengl = (
                to_opengl(ray_d_world)
                .view(B, N, H_patch_num, W_patch_num, 3)
                .view(B, N * H_patch_num * W_patch_num, 3)
            )
            ray_d_opengl = ray_d_opengl / ray_d_opengl.norm(
                dim=-1, keepdim=True
            )  # [B, N * H_patch_num * W_patch_num, 3]

            # positional encoding using normalized xyz
            if self.pos_embedder is not None:
                ray_d_opengl_embed = self.pos_embedder(
                    ray_d_opengl
                )  # [B, N * H_patch_num * W_patch_num, C]
            input_patches_flatten = input_patches_flatten + ray_d_opengl_embed

            # transformer
            sky_tokens = []
            for i in range(B):
                valid_sky_patches = input_patches_flatten[i][
                    valid_sky_patch_mask_flatten[i].squeeze(-1)
                ]  # [valid_sky_patch_num, C]
                valid_sky_patches = valid_sky_patches.unsqueeze(
                    0
                )  # [1, valid_sky_patch_num, C]
                valid_sky_patches = torch.cat(
                    [self.sky_query, valid_sky_patches], dim=1
                )  # [1, valid_sky_patch_num + 1, C]
                valid_sky_patches = self.transformer(
                    valid_sky_patches
                )  # [1, valid_sky_patch_num + 1, C]
                sky_tokens.append(valid_sky_patches[0, 0])

            sky_tokens = torch.stack(sky_tokens)  # [B, C]

        skyenc_output = {"sky_tokens": sky_tokens}

        return skyenc_output

    def forward(self, skyenc_output, network_output):
        """
        Args:
            skyenc_output: dict, output of encode_sky_feature, sky_tokens. shape [B, C]
            network_output: dict, output of network

        Returns:
            network_output: dict, updated network_output
        """
        sky_tokens = skyenc_output["sky_tokens"]

        network_output.update({"skybox_representation": sky_tokens})

        return network_output

    @staticmethod
    def sample(camera_pose, intrinsic, one_skybox_representation):
        """
        Args:
            camera_pose: [4, 4], camera pose
            intrinsic: [6,], fx fy cx cy w h
            one_skybox_representation: [C], skybox representation

        Returns:
            sampled_image: [H', W', 3], color of the skybox
        """
        pass

    def sample_batch(self, camera_pose_stack, intrinsics, network_output, batch_idx):
        """
        Args:
            camera_pose_stack : torch.tensor
                camera pose matrix, shape [N, 4, 4]
            intrinsics : torch.tensor
                camera intrinsic, shape [N, 6]
            network_output : dict, contains
                skybox_representation : torch.tensor
                    sky_token, shape [B, C]. note that B is not N! select by batch_idx
        Returns:
            sampled_image : torch.tensor
                sampled color given pose_matrice and intrinsic, shape [N, H', W', C]
        """
        N = camera_pose_stack.shape[0]
        sky_token = network_output["skybox_representation"][batch_idx]  # [C,]

        # [N, 3] and [N, H', W', 3]
        camera_origin, ray_d_world = create_rays_from_intrinsic_torch_batch(
            camera_pose_stack, intrinsics
        )
        ray_d_opengl = to_opengl(ray_d_world)  # [N, H', W', 3]
        # normalize ray direction
        ray_d_opengl = ray_d_opengl / ray_d_opengl.norm(dim=-1, keepdim=True)

        N_, H_, W_ = ray_d_opengl.shape[:3]

        # when the pixel number is too large, we need to split the ray_d_opengl to avoid OOM
        if ray_d_opengl.numel() > 1e7 * 3:
            # split N to avoid OOM
            sampled_image_list = []
            for i in range(0, N):
                sampled_image_list.append(
                    self.modulator(
                        ray_d_opengl[i : i + 1].view(1, -1, 3), sky_token[None, ...]
                    ).view(1, H_, W_, 3)
                )
            sampled_image = torch.cat(sampled_image_list, dim=0)

        else:
            # [N, H', W', 3]
            sampled_image = self.modulator(
                ray_d_opengl.view(1, -1, 3), sky_token[None, ...]
            ).view(N_, H_, W_, 3)

        return sampled_image

    @torch.no_grad()
    def visualize(self, network_output):
        """
        visualize the skybox_representation. Now we can only visualize if RGB is stored, not feature

        Args:
            network_output : dict
                output dictionary containing 'skybox_representation', sky tokens shape [B, C]
        Returns:
            skybox_representation : torch.tensor
                skybox color panorama, shape [B, H, W, 3]
        """
        sky_tokens = network_output["skybox_representation"]

        world_sky_direction = get_world_directions_latlong(
            256, 512
        )  # opengl. [256, 512, 3]
        world_sky_direction = world_sky_direction / world_sky_direction.norm(
            dim=-1, keepdim=True
        )
        world_sky_direction = world_sky_direction.to(sky_tokens)

        B = sky_tokens.shape[0]
        sampled_panorama_image = self.modulator(
            world_sky_direction.view(1, -1, 3).repeat(B, 1, 1), sky_tokens
        ).view(B, 256, 512, 3)

        return sampled_panorama_image

    def save_skybox(self, network_output, gaussian_saving_path):
        """
        save the skybox representation for offline rendering.

        We need to save
        1) weights of the modulator
        2) hparams of this module
        3) skybox representation (sky token)

        network_output : dict, contains
            skybox_representation : torch.tensor
                sky_token, shape [B, C]. note that B is not N! select by batch_idx, here should be 1
        """
        sky_tokens = network_output["skybox_representation"]
        assert sky_tokens.shape[0] == 1

        gs_stem = Path(gaussian_saving_path).with_suffix("").as_posix()
        sky_token_path = gs_stem + "_sky_token.pt"
        modulator_weight_path = gs_stem + "_modulator.pt"
        modulator_config_path = gs_stem + "_modulator.yaml"

        torch.save(sky_tokens.cpu(), sky_token_path)
        torch.save(self.state_dict(), modulator_weight_path)
        with open(modulator_config_path, "w") as f:
            OmegaConf.save(config=self.hparams, f=f)
