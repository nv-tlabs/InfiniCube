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

import fvdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_scatter import scatter_max, scatter_mean, scatter_sum

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.encoders.point_encoder import ResnetBlockFC
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.dinov2_encoder import (
    DinoWrapper,
)


def project_2D_to_3D(depth, camera_intrinsic, camera_to_world):
    """
    Args:
        depth: torch.Tensor, shape (B, 1, H, W)
        camera_intrinsic: torch.Tensor, shape (B, 6), fx, fy, cx, cy, w, h
        camera_to_world: torch.Tensor, shape (B, 4, 4)

    Returns:
        points_3d: torch.Tensor, shape (B, H*W, 3)
    """

    B, _, H, W = depth.shape
    assert H == round(camera_intrinsic[0, 5].item()), (
        f"The height of the depth map and camera intrinsic do not match. H is {H} and camera_intrinsic[0, 5] is {camera_intrinsic[0, 5]}"
    )

    # Create a grid of pixel coordinates
    x = torch.arange(W, dtype=torch.float32, device=depth.device)
    y = torch.arange(H, dtype=torch.float32, device=depth.device)
    x, y = torch.meshgrid(x, y, indexing="xy")  # must pass indexing='xy'

    # Stack pixel coordinates and flatten
    pixel_coords = torch.stack(
        (x.flatten(), y.flatten(), torch.ones_like(x.flatten())), dim=0
    )  # (3, H*W)

    # Inverse of the intrinsic matrix
    camera_intrinsic_K = torch.eye(3).expand(B, -1, -1).to(depth.device)
    camera_intrinsic_K[:, 0, 0] = camera_intrinsic[:, 0]
    camera_intrinsic_K[:, 1, 1] = camera_intrinsic[:, 1]
    camera_intrinsic_K[:, 0, 2] = camera_intrinsic[:, 2]
    camera_intrinsic_K[:, 1, 2] = camera_intrinsic[:, 3]
    K_inv = torch.linalg.inv(camera_intrinsic_K)  # (B, 3, 3)

    # Convert pixel coordinates to normalized camera coordinates
    normalized_camera_coords = K_inv @ pixel_coords.unsqueeze(0)  # (B, 3, H*W)

    # Get the depth values and reshape
    current_depth = depth.view(B, 1, -1)  # (B, 1, H*W)

    # Scale by depth
    camera_coords = normalized_camera_coords * current_depth  # (B, 3, H*W)

    # Convert to homogeneous coordinates
    camera_coords_homogeneous = torch.cat(
        (camera_coords, torch.ones(B, 1, H * W, device=depth.device)), dim=1
    )  # (B, 4, H*W)

    # Transform to world coordinates
    world_coords_homogeneous = (
        camera_to_world @ camera_coords_homogeneous
    )  # (B, 4, H*W)

    # Return the 3D points in world coordinates (X, Y, Z)
    points_3d = world_coords_homogeneous[:, :3, :].transpose(1, 2)  # (B, H*W, 3)

    return points_3d


def get_rank(ijk_from_low_bound, bound_voxel_num, B):
    """
    Args:
        ijk_from_low_bound : torch.Tensor
            N_valid_grid x 4, 4 is [voxel_i, voxel_j, voxel_k, batch_idx], integer

        bound_voxel_num : list of 3 integers, voxel_num in each dimension
            [nx_i, nx_j, nx_k]

    Returns:
        ranks : torch.Tensor
            N_valid_grid, integer
    """
    nx = bound_voxel_num
    # get tensors from the same voxel next to each other
    # Give each point a rank value, points with equal ranks are in the same batch and in the same grid
    # ranks : [N_valid_grid, ]
    ranks = (
        ijk_from_low_bound[:, 0] * (nx[1] * nx[2] * B)
        + ijk_from_low_bound[:, 1] * (nx[2] * B)
        + ijk_from_low_bound[:, 2] * B
        + ijk_from_low_bound[:, 3]
    )

    return ranks


def scatter_sum_by_ranks(features, ranks, bound_voxel_num, B):
    """
    Args:
        features: torch.Tensor
            N_valid_grid x C,

        geom_feats : torch.Tensor
            N_valid_grid x 4, 4 is [voxel_i, voxel_j, voxel_k, batch_idx], integer

        bound_voxel_num : list of 3 integers, voxel_num in each dimension
            [nx_i, nx_j, nx_k]

    Returns:
        final : torch.Tensor
            [B, X, Y, Z, C]
    """
    nx = bound_voxel_num
    # [X, Y, Z, B, C]
    final = scatter_sum(
        features, ranks, dim=0, dim_size=nx[0] * nx[1] * nx[2] * B
    ).view(nx[0], nx[1], nx[2], B, -1)
    # [B, X, Y, Z, C]
    final = final.permute(3, 0, 1, 2, 4)

    return final


def scatter_mean_by_ranks(features, ranks, bound_voxel_num, B):
    """
    Args:
        features: torch.Tensor
            N_valid_grid x C,

        geom_feats : torch.Tensor
            N_valid_grid x 4, 4 is [voxel_i, voxel_j, voxel_k, batch_idx], integer

        bound_voxel_num : list of 3 integers, voxel_num in each dimension
            [nx_i, nx_j, nx_k]

    Returns:
        final : torch.Tensor
            [B, X, Y, Z, C]
    """
    nx = bound_voxel_num
    # [X, Y, Z, B, C]
    final = scatter_mean(
        features, ranks, dim=0, dim_size=nx[0] * nx[1] * nx[2] * B
    ).view(nx[0], nx[1], nx[2], B, -1)
    # [B, X, Y, Z, C]
    final = final.permute(3, 0, 1, 2, 4)

    return final


def get_points3d_ijk_from_lower_bound(
    points_3d, neck_voxel_sizes, low_bound, high_bound
):
    """
    Args:
        points_3d : torch.Tensor
            [B, N*H'*W', 3], 3D points in grid coordinate
        neck_voxel_sizes : torch.Tensor
            [B, 3]
    Returns:
        points_3d_ijk_from_low_bound : torch.Tensor
            [B, N*H'*W', 4], 4 is [voxel_i, voxel_j, voxel_k, batch_idx], integer

        mask_valid : torch.Tensor
            [B, N*H'*W'], whether the ijk is in neck cube
    """
    B, NxH_xW_ = points_3d.shape[:2]
    neck_voxel_sizes = neck_voxel_sizes.view(B, 1, 3)
    bound_length = torch.tensor(high_bound, device=points_3d.device) - torch.tensor(
        low_bound, device=points_3d.device
    )

    # (B, N*H'*W', 3)
    # for example, the 32x32x32 dense grid with height 4x has
    # origins: [1.6m, 1.6m, 0.8m], voxel_sizes: [3.2m, 3.2m, 1.6m], the bbox is [[-16,-16,-16], [15,15,15]]
    # the perception range is -+ 51.2m in x and y axis

    # if the i is -16 in the grid, then it should be 0 when count from the lower bound
    points_3d_ijk = (
        ((points_3d - (neck_voxel_sizes / 2)) / neck_voxel_sizes).round().long()
    )  # ijk from grid origins (voxel centers)
    points_3d_ijk_from_low_bound = points_3d_ijk - torch.tensor(
        low_bound, device=points_3d.device
    )
    batch_ix = torch.stack(
        [
            torch.full([NxH_xW_, 1], ix, device=points_3d.device, dtype=torch.long)
            for ix in range(B)
        ]
    )
    points_3d_ijk_from_low_bound = torch.cat(
        (points_3d_ijk_from_low_bound, batch_ix), 2
    )

    # filter out points that are outside box, (B, N*H'*W')
    mask_valid = (
        (points_3d_ijk_from_low_bound[..., 0] >= 0)
        & (points_3d_ijk_from_low_bound[..., 0] < bound_length[0])
        & (points_3d_ijk_from_low_bound[..., 1] >= 0)
        & (points_3d_ijk_from_low_bound[..., 1] < bound_length[1])
        & (points_3d_ijk_from_low_bound[..., 2] >= 0)
        & (points_3d_ijk_from_low_bound[..., 2] < bound_length[2])
    )

    return points_3d_ijk_from_low_bound, mask_valid


class Lift3DEncoder(nn.Module):
    def __init__(
        self,
        image_resize_shape,  # (H, W)
        cube_bbox_size,  # [voxel_x_max, voxel_y_max, voxel_z_max]
        encoding_points_pos=False,  # not do consider the relation and position of the points
        normalize_cube_feature=True,
        encoder_name="dinov2",
        return_dense_cube=True,
        random_drop_input_frames=False,
        depth_shift_aug=False,
        encoder_params=None,
        **kwargs,
    ):
        super().__init__()
        self.image_resize_shape = (
            image_resize_shape
            if isinstance(image_resize_shape, list)
            else OmegaConf.to_container(image_resize_shape)
        )
        self.encoder_name = encoder_name
        self.encoder_params = encoder_params
        self.return_dense_cube = return_dense_cube
        self.normalize_cube_feature = normalize_cube_feature
        self.encoding_points_pos = encoding_points_pos
        self.depth_shift_aug = depth_shift_aug
        self.random_drop_input_frames = random_drop_input_frames

        if isinstance(cube_bbox_size, int):
            neck_bound = cube_bbox_size // 2
            low_bound = [-neck_bound] * 3
            high_bound = [neck_bound] * 3
        else:
            low_bound = [-int(res / 2) for res in cube_bbox_size]
            high_bound = [int(res / 2) for res in cube_bbox_size]

        self.low_bound = low_bound
        self.high_bound = high_bound
        if self.encoder_name == "dinov2":
            self.encoder = DinoWrapper(self.encoder_params)
        else:
            raise ValueError(f"Unknown encoder name: {self.encoder_name}")

        if self.encoding_points_pos:
            hidden_dim = self.encoder.out_dim_list[-1]
            xyz_dim = 3
            n_blocks = kwargs.get("pointnet_n_blocks", 3)
            self.fc_pos = nn.Linear(xyz_dim + hidden_dim, 2 * hidden_dim)
            self.blocks = nn.ModuleList(
                [ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)]
            )
            self.fc_c = nn.Linear(hidden_dim, hidden_dim)

        self.debug_count = 0

    def get_points3d_for_resized_input(
        self, resized_H, resized_W, depth, camera_pose, camera_intrinsic
    ):
        """
        Args:
            resized_H : int
            resized_W : int
            depth : torch.Tensor
                [B, N, H, W, 1], original resolution
            camera_pose : torch.Tensor
                [B, N, 4, 4]
            camera_intrinsic : torch.Tensor
                [B, N, 6], original resolution
        Returns:
            points_3d : torch.Tensor
                [B, N*H'*W', 3], 3D points in grid coordinate
        """
        H_ = resized_H
        W_ = resized_W
        B, N, H, W = depth.shape[:4]

        # sky is depth = 0
        sky_area = torch.logical_or(depth >= 150, depth <= 0)

        # B*N, 1, H', W'
        sky_area_resized = F.interpolate(
            sky_area.flatten(0, 1).permute(0, 3, 1, 2).float(),
            size=(H_, W_),
            mode="nearest",
        )

        # B*N, 1, H', W'
        depth_resized = F.interpolate(
            depth.flatten(0, 1).permute(0, 3, 1, 2), size=(H_, W_), mode="nearest"
        )

        # set sky area to 1e6
        depth_resized[sky_area_resized == 1] = 1e6

        # resize camera intrinsic accordingly
        camera_intrinsic_resized = torch.clone(camera_intrinsic)
        camera_intrinsic_resized[:, :, [1, 3, 5]] /= H / H_
        camera_intrinsic_resized[:, :, [0, 2, 4]] /= W / W_

        # (B*N, H'*W', 3), already in grid coordinate
        points_3d = project_2D_to_3D(
            depth_resized,
            camera_intrinsic_resized.flatten(0, 1),
            camera_pose.flatten(0, 1),
        )

        # (B, N*H'*W', 3)
        points_3d = points_3d.reshape(B, N * H_ * W_, 3)

        return points_3d

    def get_feature_map_shape(self):
        H, W = self.image_resize_shape
        patch_size = self.encoder.model.config.patch_size
        upsample_time = self.encoder.out_upsample_list  # list of true/false
        downsample_time = self.encoder.out_downsample_list  # list of true/false

        H_ = H / patch_size * 2 ** sum(upsample_time) / 2 ** sum(downsample_time)
        W_ = W / patch_size * 2 ** sum(upsample_time) / 2 ** sum(downsample_time)

        print(f"Feature map shape: {H_}, {W_}")
        return int(H_), int(W_)

    def forward(
        self,
        images,
        unproject_mask,
        depth,
        camera_pose,
        camera_intrinsic,
        neck_voxel_sizes,
    ):
        """
        images: [B, N, H, W, 3]
        unproject_mask: [B, N, H, W, 1]
        depth: [B, N, H, W, 1] or [B, N, H, W, 3]
            when it is [B, N, H, W, 3], it is already a point map. No unprojection is needed.
            This is especially used in testing mode.
        camera_pose: [B, N, 4, 4]
        camera_intrinsic: [B, N, 6]
        neck_voxel_sizes: [B, 3]

        Returns:
            fvdb.VDBTensor
        """
        B, N, H, W, _ = images.shape

        # B*N, C, H, W
        image_resized = F.interpolate(
            images.flatten(0, 1).permute(0, 3, 1, 2),
            size=self.image_resize_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        unproject_mask = F.interpolate(
            unproject_mask.flatten(0, 1).permute(0, 3, 1, 2).float(),
            size=self.image_resize_shape,
            mode="nearest",
        )

        # encode feature, B*N, C, H', W', mask: B*N, H', W'
        image_features, feature_unproject_mask = self.encoder(
            image_resized, image_mask=unproject_mask
        )

        # resize depth to the same size as image feature
        H_, W_ = image_features.shape[-2:]

        # B, N*H'*W', 3
        if depth.shape[-1] == 1:
            if self.depth_shift_aug and self.training:
                depth = depth * (1 + torch.randn_like(depth) * 0.1)
            points_3d = self.get_points3d_for_resized_input(
                H_, W_, depth, camera_pose, camera_intrinsic
            )
        elif depth.shape[-1] == 3:
            assert not self.training, "Point Map mode should not appear in training"
            points_3d = depth.reshape(B, N * H_ * W_, 3)

        # (B, N*H'*W', 4)
        points_3d_ijk_from_low_bound, kept1 = get_points3d_ijk_from_lower_bound(
            points_3d, neck_voxel_sizes, self.low_bound, self.high_bound
        )

        # filter out pixel out of unproject mask
        feature_unproject_mask = feature_unproject_mask.reshape(B, N * H_ * W_)
        kept2 = feature_unproject_mask > 0

        kept = kept1 & kept2

        # if random drop input frames and is training mode, we keep the first frame and drop the rest with 50% probability each.
        if self.random_drop_input_frames and self.training:
            keep_mask = torch.rand(B, N, device=kept.device) > 0.5  # (B, N)
            keep_mask[:, 0] = True  # always keep the first frame
            # make [B, N*H'*W']
            keep_mask = (
                keep_mask.unsqueeze(-1).expand(B, N, H_ * W_).flatten(1, 2)
            )  # (B, N*H'*W')
            kept = kept & keep_mask

        points_3d_ijk_from_low_bound = points_3d_ijk_from_low_bound[
            kept
        ]  # (N_valid, 4)
        image_features_channel_last = image_features.permute(0, 2, 3, 1).reshape(
            B, N * H_ * W_, -1
        )  # (B, N*H_*W_, C)
        image_features_channel_last = image_features_channel_last[kept]  # (N_valid, C)

        bound_length = torch.tensor(
            self.high_bound, device=points_3d.device
        ) - torch.tensor(self.low_bound, device=points_3d.device)
        if not self.encoding_points_pos:
            # (B x X x Y x Z x C)
            ranks = get_rank(points_3d_ijk_from_low_bound, bound_length, B)
            if self.normalize_cube_feature:
                final = scatter_mean_by_ranks(
                    image_features_channel_last, ranks, bound_length, B
                )
            else:
                final = scatter_sum_by_ranks(
                    image_features_channel_last, ranks, bound_length, B
                )

        else:
            # N_valid, C
            ranks = get_rank(points_3d_ijk_from_low_bound, bound_length, B)
            dim_size = bound_length[0] * bound_length[1] * bound_length[2] * B

            points_3d_kept = points_3d[kept]  # (N_valid, 3)
            # mod neck voxel sizes
            neck_voxel_sizes_vec = neck_voxel_sizes[0]
            points_3d_kept_local = (
                points_3d_kept
                - points_3d_kept // neck_voxel_sizes_vec * neck_voxel_sizes_vec
            )
            points_feature = torch.cat(
                (points_3d_kept_local, image_features_channel_last), 1
            )  # (N_valid, 3 + C)

            points_feature = self.fc_pos(points_feature)
            points_feature = self.blocks[0](points_feature)
            for block in self.blocks[1:]:
                pooled = scatter_max(points_feature, ranks, dim=0, dim_size=dim_size)[0]
                pooled = pooled[ranks]
                points_feature = torch.cat([points_feature, pooled], dim=1)
                points_feature = block(points_feature)

            points_feature = self.fc_c(points_feature)  # (N_valid, C)

            # (B x X x Y x Z x C)
            if self.normalize_cube_feature:
                final = scatter_mean_by_ranks(points_feature, ranks, bound_length, B)
            else:
                final = scatter_sum_by_ranks(points_feature, ranks, bound_length, B)

        X, Y, Z = final.shape[1:-1]
        if self.return_dense_cube:
            # note that the origins and voxel sizes are default! we just need the features
            voxel_tensor = fvdb.nn.vdbtensor_from_dense(
                final, ijk_min=[-X // 2, -Y // 2, -Z // 2]
            )
        else:
            raise NotImplementedError

        return voxel_tensor

    def create_cond_dict_from_batch(self, batch):
        cond_dict = {}
        cond_dict["images"] = torch.stack(batch[DS.IMAGES_INPUT])
        cond_dict["depth"] = torch.stack(batch[DS.IMAGES_INPUT_DEPTH])
        cond_dict["camera_pose"] = torch.stack(batch[DS.IMAGES_INPUT_POSE])
        cond_dict["camera_intrinsic"] = torch.stack(batch[DS.IMAGES_INPUT_INTRINSIC])

        # B, N, H, W, 1
        foreground_area_from_seg = (
            torch.stack(batch[DS.IMAGES_INPUT_MASK])[..., 0:1] > 0
        )
        # sky area depth 0
        cond_dict["depth"] = cond_dict["depth"] * foreground_area_from_seg

        # B, N, H, W, 1
        non_hood_or_padding_area = (
            torch.stack(batch[DS.IMAGES_INPUT_MASK])[..., 2:3] > 0
        )
        cond_dict["unproject_mask"] = non_hood_or_padding_area

        return cond_dict

    @torch.no_grad()
    def generate_visualization_items(
        self,
        images,
        unproject_mask,
        depth,
        camera_pose,
        camera_intrinsic,
        neck_voxel_sizes,
    ):
        B, N, H, W, _ = images.shape

        # B*N, C, H, W
        image_resized = F.interpolate(
            images.flatten(0, 1).permute(0, 3, 1, 2),
            size=self.image_resize_shape,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        unproject_mask = F.interpolate(
            unproject_mask.flatten(0, 1).permute(0, 3, 1, 2).float(),
            size=self.image_resize_shape,
            mode="nearest",
        )

        # encode feature, B*N, C, H', W', mask: B*N, H', W'
        image_features, feature_unproject_mask = self.encoder(
            image_resized, image_mask=unproject_mask
        )

        # resize depth to the same size as image feature
        H_, W_ = image_features.shape[-2:]

        # B*N, C, H', W'
        image_resized_feature_map_size = F.interpolate(
            images.flatten(0, 1).permute(0, 3, 1, 2),
            size=(H_, W_),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # (B, N*H'*W', 3)
        points_3d = self.get_points3d_for_resized_input(
            H_, W_, depth, camera_pose, camera_intrinsic
        )

        # (B, N*H'*W', 4), (B, N*H'*W')
        points_3d_ijk_from_low_bound, kept1 = get_points3d_ijk_from_lower_bound(
            points_3d, neck_voxel_sizes, self.low_bound, self.high_bound
        )

        # filter out pixel out of unproject mask
        feature_unproject_mask = feature_unproject_mask.reshape(B, N * H_ * W_)
        kept2 = feature_unproject_mask > 0  # (B, N*H'*W')

        # [B, N*H'*W']
        kept = kept1 & kept2

        # now we have 3d points, image_resized_feature_map_size, kept, we preserve points in kept
        # (B, N*H'*W', 3)
        image_resized_feature_map_size = image_resized_feature_map_size.permute(
            0, 2, 3, 1
        ).reshape(B, N * H_ * W_, -1)

        # (B, N*H'*W', 4)
        points_3d_ijk_from_low_bound = points_3d_ijk_from_low_bound[
            kept
        ]  # (N_valid, 4)
        # (B x X x Y x Z x C)
        bound_length = torch.tensor(
            self.high_bound, device=points_3d.device
        ) - torch.tensor(self.low_bound, device=points_3d.device)
        weigths = torch.ones(
            (points_3d_ijk_from_low_bound.shape[0], 1),
            device=points_3d_ijk_from_low_bound.device,
        )
        ranks = get_rank(points_3d_ijk_from_low_bound, bound_length, B)
        weights = scatter_sum_by_ranks(weigths, ranks, bound_length, B)
        X, Y, Z = weights.shape[1:-1]
        voxel_occupancy = fvdb.nn.vdbtensor_from_dense(
            weights, ijk_min=[-X // 2, -Y // 2, -Z // 2]
        )

        kept_points = {}
        for i in range(B):
            kept_points_xyz = points_3d[i, kept[i]]
            kept_points_color = image_resized_feature_map_size[i, kept[i]]
            kept_points[i] = (
                kept_points_xyz,
                kept_points_color,
                VDBTensor(voxel_occupancy.grid[i], voxel_occupancy.data[i]),
            )

        return kept_points
