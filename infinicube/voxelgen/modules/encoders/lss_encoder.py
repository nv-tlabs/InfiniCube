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
from einops import rearrange
from omegaconf import OmegaConf

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.encoders.lift3d_encoder import (
    get_points3d_ijk_from_lower_bound,
    get_rank,
    scatter_sum_by_ranks,
)
from infinicube.voxelgen.modules.gsm_modules.encoder.modules.dinov2_encoder import (
    DinoWrapper,
)


def get_gt_depth_dist(x, mode, depth_min, depth_max, num_bins):
    """
    Args:
        x: [B*N, H, W], the GT depth map
    Returns:
        onehot_dist: [B*N, D, H, W]

        depth_indices: [B*N, H, W]

        mask: [B*N, H, W]
    """
    # [B*N, H, W], indices (float), value: [0, num_bins)
    depth_indices, mask = get_depth_indices(x, mode, depth_min, depth_max, num_bins)
    onehot_dist = F.one_hot(depth_indices.long(), num_bins).permute(
        0, 3, 1, 2
    )  # [B*N, num_bins, H, W]

    return onehot_dist, depth_indices, mask


def depth_discretization(depth_min, depth_max, num_bins, mode):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        depth_discre = depth_min + bin_size * torch.arange(num_bins)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        depth_discre = (
            depth_min
            + bin_size * (torch.arange(num_bins) * torch.arange(1, 1 + num_bins)) / 2
        )
    else:
        raise NotImplementedError
    return depth_discre


def get_depth_indices(
    depth_map, mode, depth_min, depth_max, num_bins, clamp_outliers=True
):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        clamp_outliers [bool]: Clamp outliers to the first or last bin
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
        mask [torch.Tensor(H, W)]: Mask for valid depth values
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - torch.log(1 + depth_min))
            / (torch.log(1 + depth_max) - torch.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    mask = (indices < 0) | (indices >= num_bins) | (~torch.isfinite(indices))

    if clamp_outliers:
        indices[indices < 0] = 0
        indices[indices >= num_bins] = num_bins - 1
        indices[~torch.isfinite(indices)] = num_bins - 1

    # Convert to integer
    indices = indices.type(torch.int64)

    return indices, ~mask


def create_frustum(H, W, depth_min, depth_max, num_bins, mode):
    """
    We create a 3D frustum grid. Each grid cell has a `3D` coordinate (u, v, depth).
    u, v is the pixel coordinate in the image plane. depth is the z-value of the pixel.

    Args:
        H: height of the image
        W: width of the image
        depth_min: minimum depth value in the frustum
        depth_max: maximum depth value in the frustum
        num_bins: number of depth bins
        mode: discretization mode (See bin_depths for more details)

    Returns:
        frustum: D x H x W x 3
    """
    ds = (
        torch.tensor(
            depth_discretization(depth_min, depth_max, num_bins, mode),
            dtype=torch.float,
        )
        .view(-1, 1, 1)
        .expand(-1, H, W)
    )
    # D: number of grids in the depth direction.
    D, _, _ = ds.shape
    xs = (
        torch.linspace(0, W - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
    )  # e.g. xs: DxHxW(41x12x22)
    ys = (
        torch.linspace(0, H - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)
    )  # e.g. ys: DxHxW(41x12x22)

    # D x H x W x 3
    frustum = torch.stack((xs, ys, ds), -1)  # frustum: D x H x W x 3, 3: (u, v, depth)
    return frustum


def get_points(pose_matrix, intrinsic, frustum):
    """Determine the (x,y,z) locations of the point cloud in the grid coordinate.

    Args:
        pose_matrix (torch.Tensor): B x N x 4 x 4
        intrinsic (torch.Tensor): B x N x 3 x 3 or B x N x 6
        frustum (torch.Tensor): D x H x W x 3

    Returns
        points (torch.Tensor): B x N x D x H x W x 3
    """
    if intrinsic.dim() == 3 and intrinsic.size(-1) == 6:
        # [B, N, 1]
        fx, fy, cx, cy, w, h = intrinsic.split(1, dim=-1)
        intrinsic_matrix = torch.zeros(
            intrinsic.size(0), intrinsic.size(1), 3, 3, device=intrinsic.device
        )
        intrinsic_matrix[:, :, 0, 0] = fx[..., 0]
        intrinsic_matrix[:, :, 1, 1] = fy[..., 0]
        intrinsic_matrix[:, :, 0, 2] = cx[..., 0]
        intrinsic_matrix[:, :, 1, 2] = cy[..., 0]
        intrinsic_matrix[:, :, 2, 2] = 1
    else:
        intrinsic_matrix = intrinsic

    B, N, _, _ = pose_matrix.shape  # B: (batchsize)    N: camera_num
    D, H, W, _ = (
        frustum.shape
    )  # D: number of grids in the depth direction. H: height of the image. W: width of the image.

    points = frustum.view(1, 1, D, H, W, 3).expand(
        B, N, D, H, W, 3
    )  # B x N x D x H x W x 3

    # [u*Z, v*Z, Z]
    points = torch.cat(
        (
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3],
        ),
        dim=5,
    )  # B x N x D x H x W x 3
    # points in camera frame
    points = (
        torch.linalg.inv(intrinsic_matrix.view(B, N, 3, 3))
        .view(B, N, 1, 1, 1, 3, 3)
        .matmul(points.view(B, N, D, H, W, 3, 1))
        .squeeze(-1)
    )  # B x N x D x H x W x 3
    # homogeneous coordinates
    points = torch.cat(
        (points, torch.ones(B, N, D, H, W, 1, device=points.device)), dim=5
    )  # B x N x D x H x W x 4

    # camera to grid frame (ego is the grid origin)
    points = (
        pose_matrix.view(B, N, 1, 1, 1, 4, 4)
        .matmul(points.view(B, N, D, H, W, 4, 1))
        .squeeze(-1)
    )  # B x N x D x H x W x 3
    points = points[..., :3]

    return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3)


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha=0.25, gamma=2.0, reduction="none", smooth_target=False, eps=None
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_target = smooth_target
        self.eps = eps
        if self.smooth_target:
            self.smooth_kernel = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.smooth_kernel.weight = torch.nn.Parameter(
                torch.tensor([[[0.2, 0.9, 0.2]]]), requires_grad=False
            )
            self.smooth_kernel = self.smooth_kernel.to(torch.device("cuda"))

            for p in self.smooth_kernel.parameters():
                p.requires_grad = False

    def forward(self, input, target):
        """
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
        :math:`0 ≤ targets[i] ≤ C−1`.
        """
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        D = input.shape[1]
        if self.smooth_target:
            target_one_hot = (
                F.one_hot(target, num_classes=D).to(input).view(-1, D)
            )  # [N*H*W, D]
            target_one_hot = self.smooth_kernel(
                target_one_hot.float().unsqueeze(1)
            ).squeeze(1)  # [N*H*W, D]
            target_one_hot = target_one_hot.view(*target.shape, D).permute(0, 3, 1, 2)
        else:
            target_one_hot = (
                F.one_hot(target, num_classes=D).to(input).permute(0, 3, 1, 2)
            )
        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum("bc...,bc...->b...", (target_one_hot, focal))

        if self.reduction == "none":
            loss = loss_tmp
        elif self.reduction == "mean":
            loss = torch.mean(loss_tmp)
        elif self.reduction == "sum":
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss


class LssEncoder(nn.Module):
    def __init__(
        self,
        image_resize_shape,
        cube_bbox_size,  # [voxel_x_max, voxel_y_max, voxel_z_max]
        depth_minmax,  # [min_depth, max_depth]
        depth_bin_num,  # number of depth bins
        depth_mode,  # 'UD' or 'LID'
        encoder_name="dinov2",
        return_dense_cube=True,
        encoder_params=None,
        focal_loss_params=None,
        **kwargs,
    ):
        super().__init__()
        self.image_resize_shape = (
            image_resize_shape
            if isinstance(image_resize_shape, list)
            else OmegaConf.to_container(image_resize_shape)
        )
        self.encoder_name = encoder_name
        self.return_dense_cube = return_dense_cube
        self.encoder_params = encoder_params
        self.depth_minmax = depth_minmax
        self.depth_bin_num = depth_bin_num
        self.depth_mode = depth_mode

        if focal_loss_params is None:
            focal_loss_params = {}
        self.focal_loss = FocalLoss(**focal_loss_params)

        if isinstance(cube_bbox_size, int):
            neck_bound = cube_bbox_size // 2
            low_bound = [-neck_bound] * 3
            high_bound = [neck_bound] * 3
        else:
            low_bound = [-int(res / 2) for res in cube_bbox_size]
            high_bound = [int(res / 2) for res in cube_bbox_size]

        self.low_bound = low_bound
        self.high_bound = high_bound

        if encoder_name == "dinov2":
            self.encoder = DinoWrapper(encoder_params)
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not implemented")

        self.depth_estimator = None

        # set up lss stuffs
        feature_map_h, feature_map_w = self.get_feature_map_shape()
        self.frustum = create_frustum(
            feature_map_h,
            feature_map_w,
            depth_minmax[0],
            depth_minmax[1],
            depth_bin_num,
            depth_mode,
        )

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
        depth: [B, N, H, W, 1], here the depth is just for GT depth calculation
        camera_pose: [B, N, 4, 4]
        camera_intrinsic: [B, N, 6]
        neck_voxel_sizes: [B, 3]

        Returns:
            fvdb.VDBTensor
        """

        # This is prepared for evaluation, we don't want to go through the heavy encoder again.
        if (
            getattr(self, "cache_condition_flag", False)
            and getattr(self, "cached_condition", None) is not None
        ):
            return self.cached_condition, {}

        if self.frustum.device != images.device:
            self.frustum = self.frustum.to(images.device)

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

        image_features_, feature_unproject_mask = self.encoder(
            image_resized, image_mask=unproject_mask
        )

        H_, W_ = image_features_.shape[-2:]

        # B*N, H', W'
        depth_GT_resized = F.interpolate(
            depth.flatten(0, 1).permute(0, 3, 1, 2), size=(H_, W_), mode="nearest"
        ).squeeze(1)

        # maybe not image_features.
        image_features, depth_prob_logit = (
            image_features_[:, self.depth_bin_num :],
            image_features_[:, : self.depth_bin_num],
        )
        depth_prob = F.softmax(depth_prob_logit, dim=1)  # [B*N, D, H', W']

        # out product for image features and depth prob. [B, N, D, H', W', C]
        feature_out_product = rearrange(
            image_features, "(b n) c h w -> b n 1 h w c", b=B, n=N
        ) * rearrange(depth_prob, "(b n) d h w -> b n d h w 1", b=B, n=N)

        # resize the camera intrinsic according to the feature map size
        camera_intrinsic = camera_intrinsic.clone()
        camera_intrinsic[:, :, [1, 3, 5]] /= H / depth_prob.shape[2]
        camera_intrinsic[:, :, [0, 2, 4]] /= W / depth_prob.shape[3]

        points3D = get_points(
            camera_pose, camera_intrinsic, self.frustum
        )  # [B, N, D, H', W', 3]
        # [B, N, D, H', W', 3] -> [B, N*D*H'*W', 3] and [B, N*D*H'*W']
        points3D_ijk_from_low_bound, kept1 = get_points3d_ijk_from_lower_bound(
            points3D.view(B, -1, 3), neck_voxel_sizes, self.low_bound, self.high_bound
        )

        feature_unproject_mask = feature_unproject_mask.view(B, N, 1, H_, W_).expand(
            -1, -1, self.depth_bin_num, -1, -1
        )
        feature_unproject_mask = feature_unproject_mask.reshape(B, -1)
        kept2 = feature_unproject_mask > 0

        non_sky_area = depth_GT_resized > 0
        kept3 = (
            non_sky_area.view(B, N, 1, H_, W_)
            .expand(-1, -1, self.depth_bin_num, -1, -1)
            .reshape(B, -1)
        )

        kept = kept1 & kept2 & kept3

        # [B, N*D*H'*W', C]
        feature_out_product = feature_out_product.reshape(
            B, -1, feature_out_product.shape[-1]
        )

        # keep only the points that are inside the neck voxel
        points3D_ijk_from_low_bound = points3D_ijk_from_low_bound[kept]
        feature_out_product = feature_out_product[kept]

        bound_length = torch.tensor(
            self.high_bound, device=points3D.device
        ) - torch.tensor(self.low_bound, device=points3D.device)
        ranks = get_rank(points3D_ijk_from_low_bound, bound_length, B)
        final = scatter_sum_by_ranks(feature_out_product, ranks, bound_length, B)

        X, Y, Z = final.shape[1:-1]
        if self.return_dense_cube:
            # note that the origins and voxel sizes are default! we just need the features
            voxel_tensor = fvdb.nn.vdbtensor_from_dense(
                final, ijk_min=[-X // 2, -Y // 2, -Z // 2]
            )
        else:
            raise NotImplementedError

        if hasattr(self, "cache_condition_flag") and self.cache_condition_flag:
            self.cached_condition = voxel_tensor

        return voxel_tensor, self.prepare_loss_dict(depth_prob_logit, depth_GT_resized)

    def prepare_loss_dict(self, pred_logit, gt_depth):
        """
        Args:
            pred_logit: [B*N, D, H', W']
            gt_depth: [B*N, H', W']
        """
        # sky and padding area are 0-depth, they will be masked out
        onehot_dist, depth_indices, mask = get_gt_depth_dist(
            gt_depth,
            self.depth_mode,
            self.depth_minmax[0],
            self.depth_minmax[1],
            self.depth_bin_num,
        )

        loss_unreduced = self.focal_loss(pred_logit, depth_indices)
        loss = torch.sum(loss_unreduced * mask) / mask.sum()

        return {"depth_focal_loss": loss}

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

    def generate_visualization_items(self, images, unproject_mask, depth):
        """
        images: [B, N, H, W, 3]
        unproject_mask: [B, N, H, W, 1]
        depth: [B, N, H, W, 1], here the depth is just for GT depth calculation
        """
        from infinicube.utils.depth_utils import vis_depth

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

        # B*N, C, H', W' and B*N, H', W'
        image_features_, feature_unproject_mask = self.encoder(
            image_resized, image_mask=unproject_mask
        )

        H_, W_ = image_features_.shape[-2:]
        depth_GT_resized = F.interpolate(
            depth.flatten(0, 1).permute(0, 3, 1, 2), size=(H_, W_), mode="nearest"
        ).squeeze(1)

        # B*N, H', W'
        non_sky_area = depth_GT_resized > 0

        # maybe not image_features.
        image_features, depth_prob_logit = (
            image_features_[:, self.depth_bin_num :],
            image_features_[:, : self.depth_bin_num],
        )
        depth_prob = F.softmax(depth_prob_logit, dim=1)  # [B*N, D, H', W']

        depth_prob_max_indices = torch.argmax(depth_prob, dim=1)  # [B*N, H', W']
        depth_GT_indices = get_depth_indices(
            depth_GT_resized,
            self.depth_mode,
            self.depth_minmax[0],
            self.depth_minmax[1],
            self.depth_bin_num,
        )[0]

        vis_depth_map_comparsion = []
        for i in range(depth_prob_max_indices.shape[0]):
            depth_prob_max_indices_vis_map = vis_depth(
                depth_prob_max_indices[i], minmax=[0, self.depth_bin_num]
            )  # [H, W, 3], 0-255
            depth_GT_indices_vis_map = vis_depth(
                depth_GT_indices[i], minmax=[0, self.depth_bin_num]
            )  # [H, W, 3], 0-255
            unproj_mask = (feature_unproject_mask[i] * non_sky_area[i]).unsqueeze(
                -1
            ).expand(-1, -1, 3) * 255
            unproj_mask = unproj_mask.to(depth_prob_max_indices_vis_map.device)
            vis_depth_map_comparsion.append(
                torch.cat(
                    [
                        depth_prob_max_indices_vis_map,
                        depth_GT_indices_vis_map,
                        unproj_mask,
                    ],
                    dim=0,
                )
            )

        vis_depth_map_comparsion = torch.stack(
            vis_depth_map_comparsion
        )  # [B*N, 2*H, W, 3], 0-255

        vis_depth_map_comparsion = vis_depth_map_comparsion.permute(0, 3, 1, 2) / 255.0

        return vis_depth_map_comparsion
