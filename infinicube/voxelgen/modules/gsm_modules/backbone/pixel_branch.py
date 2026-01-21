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
import torch.nn.functional as F
from omegaconf import OmegaConf
from termcolor import cprint
from x_unet import XUnet

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.dav2_encoder import (
    DAV2Encoder,
)
from infinicube.voxelgen.utils.common_util import mask_image_patches
from infinicube.voxelgen.utils.render_util import create_rays_from_intrinsic_torch_batch


def list2tuple(d):
    """
    d is a dictionary some values of which are lists. This function converts these lists to tuples.
    """
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = tuple(v)
    return d


class Pure2DUNet(nn.Module):
    def __init__(self, backbone_2d_params):
        super(Pure2DUNet, self).__init__()
        # pixel-aligned branch, here we use UNet
        self.backbone_2d_feature_source = backbone_2d_params.feature_source
        self.net = XUnet(
            **list2tuple(OmegaConf.to_container(backbone_2d_params.unet2d_params))
        )

        self.backbone_2d_concat_depth_priors = backbone_2d_params.concat_depth_priors
        self.backbone_2d_gs_per_pixel = backbone_2d_params.gs_per_pixel
        self.backbone_2d_gs_dim = backbone_2d_params.gs_dim
        self.linear_out = nn.Linear(
            backbone_2d_params.unet2d_params.out_dim,
            backbone_2d_params.gs_per_pixel * backbone_2d_params.gs_dim,
        )

        self.znear = backbone_2d_params.znear
        self.zfar = backbone_2d_params.zfar
        self.max_scale = backbone_2d_params.max_scale
        self.interpret_output_depth = backbone_2d_params.interpret_output_depth

        self.masked_voxel_depth_params = backbone_2d_params.masked_voxel_depth_params
        self.decode_all_pixel2gs = False

    def get_mask(self, batch, target_H_W, mask_type="midground"):
        """
        Return the mid-ground mask

        Returns:
            mid_ground: torch.Tensor, shape [B, N, H, W]
        """
        images_input_mask = torch.stack(batch[DS.IMAGES_INPUT_MASK])
        foreground_mask_from_seg = images_input_mask[..., 0:1].float()
        foreground_mask_from_grid = images_input_mask[..., 3:4].float()

        # if we come here, we must model the mid-ground
        close_range = foreground_mask_from_seg * foreground_mask_from_grid
        mid_ground = foreground_mask_from_seg - close_range

        B, N, H, W = mid_ground.shape[:4]

        if mask_type == "midground":
            mask = mid_ground.view(B, N, H, W)
        elif mask_type == "close_range_and_midground":
            mask = foreground_mask_from_seg.view(B, N, H, W)

        target_H, target_W = target_H_W

        if H != target_H or W != target_W:
            mask = F.interpolate(
                mask.view(B * N, 1, H, W).float(),
                size=(target_H, target_W),
                mode="nearest",
            ).view(B, N, target_H, target_W)
        else:
            mask = mask.view(B, N, target_H, target_W)

        return mask

    def reshape_bnhwc_to_target(self, tensor, target_H, target_W):
        """
        Args:
            tensor: torch.Tensor, shape [B, N, H, W, C]
            target_H: int, target height
            target_W: int, target width

        Returns:
            reshaped_tensor: torch.Tensor, shape [B, N, target_H, target_W, C]
        """
        B, N, H, W, C = tensor.shape

        if H != target_H or W != target_W:
            reshaped_tensor = (
                F.interpolate(
                    tensor.view(B * N, H, W, C).permute(0, 3, 1, 2).float(),
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
                .view(B, N, C, target_H, target_W)
                .permute(0, 1, 3, 4, 2)
            )
        else:
            reshaped_tensor = tensor

        return reshaped_tensor

    def forward(self, batch, imgenc_output, infer_with_3d_branch=False):
        """
        Returns:
            network_output = {'decoded_gaussians': decoded_gaussians}
        """
        image_feature = imgenc_output[
            self.backbone_2d_feature_source
        ]  # can be original rgb
        B, N, _, target_H, target_W = image_feature.shape

        depth_priors = []

        if "voxel_depth" in self.backbone_2d_concat_depth_priors:
            # B, N, H, W, 1, get the depth prior, e.g. voxel depth, dav2 depth, etc.
            voxel_depth = DAV2Encoder.get_voxel_depth(batch, is_input=True)
            depth_priors.append(
                self.reshape_bnhwc_to_target(voxel_depth, target_H, target_W)
            )

        if "masked_voxel_depth" in self.backbone_2d_concat_depth_priors:
            # B, N, H, W, 1, get the depth prior, e.g. voxel depth, dav2 depth, etc
            voxel_depth = DAV2Encoder.get_voxel_depth(batch, is_input=True)
            # randomly drop some patches
            voxel_depth = self.reshape_bnhwc_to_target(voxel_depth, target_H, target_W)
            voxel_depth = mask_image_patches(
                voxel_depth,
                self.masked_voxel_depth_params.patch_size,
                self.masked_voxel_depth_params.mask_prob,
            )
            depth_priors.append(voxel_depth)

        if "provided_depth" in self.backbone_2d_concat_depth_priors:
            cprint("using provided depth", "yellow")
            assert "provided_depth" in batch, "provided_depth is not in the batch"
            assert not self.training, (
                "provided_depth is only used in inference time, as an alternative to masked voxel_depth"
            )
            assert "masked_voxel_depth" not in self.backbone_2d_concat_depth_priors, (
                "provided_depth and masked_voxel_depth cannot be used together"
            )

            voxel_depth = torch.stack(batch["provided_depth"])  # B, N, H, W, 1
            voxel_depth = self.reshape_bnhwc_to_target(voxel_depth, target_H, target_W)
            depth_priors.append(voxel_depth)

        if "dav2_feature" in self.backbone_2d_concat_depth_priors:
            # need to make sure we include 'dav2' encoder
            dav2_feature = imgenc_output["dav2"].permute(0, 1, 3, 4, 2)  # B, N, H, W, C
            depth_priors.append(
                self.reshape_bnhwc_to_target(dav2_feature, target_H, target_W)
            )

        if len(depth_priors) > 0:
            depth_priors = torch.cat(depth_priors, dim=-1).permute(
                0, 1, 4, 2, 3
            )  # B, N, #depth_priors, H, W
            image_feature = torch.cat([image_feature, depth_priors], dim=2)
        else:
            image_feature = image_feature

        network_input_2d = image_feature.flatten(0, 1)

        # if network_input_2d is too long, can chunk them. only in inference time
        if not self.training and network_input_2d.shape[0] > 8:
            network_output_2d = []
            for i in range(0, network_input_2d.shape[0], 8):
                network_output_2d.append(self.net(network_input_2d[i : i + 8]))
            network_output_2d = torch.cat(network_output_2d, dim=0)
        else:
            network_output_2d = self.net(network_input_2d)

        network_output_2d = self.linear_out(
            network_output_2d.view(B, N, -1, target_H, target_W).permute(0, 1, 3, 4, 2)
        )  # B, N, H, W, C
        gaussian_params = network_output_2d.view(
            B,
            N,
            target_H,
            target_W,
            self.backbone_2d_gs_per_pixel,
            self.backbone_2d_gs_dim,
        )

        if self.decode_all_pixel2gs:
            # only used in inference time to reconstruct dynamic objects
            mask_for_pixel_aligned = None

        else:
            if infer_with_3d_branch:
                # 2d & 3d branch inference
                mask_for_pixel_aligned = self.get_mask(
                    batch, (target_H, target_W), mask_type="midground"
                )
            else:
                # 2d branch training
                mask_for_pixel_aligned = self.get_mask(
                    batch,
                    (target_H, target_W),
                    mask_type="close_range_and_midground",
                )

        # convert to gaussians
        decoded_gs_2d = self.params2gs(
            gaussian_params,
            imgenc_output["camera_info"]["pose"],
            imgenc_output["camera_info"]["intrinsic"],
            mask_for_pixel_aligned,
        )

        network_output = {"decoded_gaussians": decoded_gs_2d}

        return network_output

    def params2gs(self, gaussian_params, camera_poses, intrinsics, midground_mask=None):
        """
        Args:
            gaussians_params:
                torch.Tensor, shape [B, N, H, W, #gs, #gs_dim],
                #gs_dim = 12, distance 1, scale 3, rotation 4, opacity 1, appearance 3
            camera_poses:
                torch.Tensor, shape [B, N, 4, 4], camera pose
            intrinsics:
                torch.Tensor, shape [B, N, 6], camera intrinsic
            midground_mask:
                torch.Tensor, shape [B, N, H, W], midground mask

        Returns:
            decoded_gs_list:
                list of torch.Tensor, each element is [#all_gs, 14], xyz 3, scale 3, rotation 4, opacity 1, appearance 3 (RGB or feature)
        """
        B, N, H, W = gaussian_params.shape[:4]

        # usually need to reshape the intrinsic
        if (H != intrinsics[..., 5]).all() or (W != intrinsics[..., 4]).all():
            downsample_h = intrinsics[0, 0, 5] / H
            downsample_w = intrinsics[0, 0, 4] / W
            intrinsics[..., [1, 3, 5]] = intrinsics[..., [1, 3, 5]] / downsample_h
            intrinsics[..., [0, 2, 4]] = intrinsics[..., [0, 2, 4]] / downsample_w

        # split the gaussian params, [B, N, H, W, #gs, each_dim]
        depth_raw, scale, rotation, opacity, appearance = gaussian_params.split(
            [1, 3, 4, 1, self.backbone_2d_gs_dim - 9], dim=-1
        )

        # get xyz from distance
        camera_poses = camera_poses.view(-1, 4, 4)  # [B*N, 4, 4]
        camera_intrinsic = intrinsics.view(-1, 6)  # [B*N, 6]
        # [B*N, 3], [B*N, H, W, 3]
        nimg_origins, nimg_directions = create_rays_from_intrinsic_torch_batch(
            camera_poses, camera_intrinsic
        )

        if self.interpret_output_depth == "weight":
            depth = torch.sigmoid(depth_raw) * (self.zfar - self.znear) + self.znear
            distance = depth / torch.sum(
                camera_poses[:, :3, 2].view(B, N, 1, 1, 1, 3)
                * nimg_directions.view(B, N, H, W, 1, 3),
                dim=-1,
                keepdim=True,
            )

        elif self.interpret_output_depth == "inverse_metric_depth":
            metric_disparity = torch.sigmoid(depth_raw)
            depth = 1 / metric_disparity  # [B, N, H, W, #gs, 1]
            distance = depth / torch.sum(
                camera_poses[:, :3, 2].view(B, N, 1, 1, 1, 3)
                * nimg_directions.view(B, N, H, W, 1, 3),
                dim=-1,
                keepdim=True,
            )

        else:
            raise ValueError(
                f"interpret_output_depth {self.interpret_output_depth} not supported"
            )

        # [B, N, H, W, 1, 3]
        nimg_origins = nimg_origins.view(B, N, 1, 1, 1, 3).expand(-1, -1, H, W, -1, -1)
        nimg_directions = nimg_directions.view(B, N, H, W, 1, 3)

        xyz = nimg_origins + distance * nimg_directions

        # activate scale / rotation / opacity
        scale = torch.clamp(torch.exp(scale - 2.3), 0, self.max_scale)
        opacity = torch.sigmoid(opacity - 1)
        rotation = F.normalize(rotation, p=2, dim=-1)

        decoded_gs_list = []
        decoded_gs = torch.cat(
            [xyz, scale, rotation, opacity, appearance], dim=-1
        )  # [B, N, H, W, #gs, 14]

        for b in range(B):
            if midground_mask is not None:
                midground_mask_b = midground_mask[b] > 0
                decoded_gs_list.append(decoded_gs[b][midground_mask_b].flatten(0, 1))
            else:
                decoded_gs_list.append(decoded_gs[b].flatten(0, 3))

        return decoded_gs_list
