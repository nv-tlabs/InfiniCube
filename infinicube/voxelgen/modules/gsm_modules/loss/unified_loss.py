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

import lpips
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycg import exp

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.gsm_modules.loss.depth_losses import (
    affine_invariant_loss,
)
from infinicube.voxelgen.modules.gsm_modules.loss.ssim_w_mask import ssim_module


class UnifiedLoss(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.supervise_image_resize = self.hparams.supervise_image_resize

        if self.hparams.supervision.render_weight > 0:
            if self.hparams.perceptual_weight > 0.0:
                # init a perceptual loss
                self.perceptual_loss = lpips.LPIPS(net="vgg")
                self.perceptual_loss.requires_grad_(False)

        if self.hparams.with_render_branch and self.hparams.use_alex_metric:
            self.loss_fn_alex = lpips.LPIPS(net="alex").eval()

        if self.hparams.pixel_loss == "l1":
            self.pixel_loss = F.l1_loss
        elif self.hparams.pixel_loss == "l2":
            self.pixel_loss = F.mse_loss

    def forward(
        self,
        batch,
        renderer_output,
        network_output,
        global_step,
        current_epoch,
        compute_metric,
        optimizer_idx=0,
    ):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()
        latent_dict = exp.TorchLossMeter()

        # prepare the resized gt
        gt_package = self.prepare_resized_gt(batch)
        image_loss_mask = gt_package["image_loss_mask"]
        vis_images_dict = self.assemble_visualization(gt_package, renderer_output)

        if optimizer_idx == 0:
            # compute alpha loss
            if (
                self.hparams.render_alpha
                and self.hparams.supervision.alpha_weight > 0.0
            ):
                pd_alphas = renderer_output["pd_alphas"]  # B, N, H, W, 1
                gt_alphas = gt_package["gt_alphas"]  # B, N, H, W, 1

                B, N, H_pd, W_pd = (
                    pd_alphas.size(0),
                    pd_alphas.size(1),
                    pd_alphas.size(2),
                    pd_alphas.size(3),
                )

                # rasterized alpha can have lower resolution than the image
                if gt_alphas.shape[2] != pd_alphas.shape[2]:
                    if global_step % 500 == 0:
                        print(
                            f"pd alpha {pd_alphas.shape} and gt alpha {gt_alphas.shape} have different resolution, resize gt to match them"
                        )
                    gt_alphas_ = (
                        F.interpolate(
                            gt_alphas.flatten(0, 1).permute(0, 3, 1, 2),
                            size=(H_pd, W_pd),
                            mode="nearest",
                        )
                        .permute(0, 2, 3, 1)
                        .view(B, N, H_pd, W_pd, 1)
                    )

                    image_loss_mask_ = (
                        F.interpolate(
                            image_loss_mask.flatten(0, 1).permute(0, 3, 1, 2),
                            size=(H_pd, W_pd),
                            mode="nearest",
                        )
                        .permute(0, 2, 3, 1)
                        .view(B, N, H_pd, W_pd, 1)
                    )
                else:
                    gt_alphas_ = gt_alphas
                    image_loss_mask_ = image_loss_mask

                alpha_loss = torch.sum(
                    F.l1_loss(pd_alphas, gt_alphas_, reduction="none")
                    * image_loss_mask_
                ) / torch.sum(image_loss_mask_)
                loss_dict.add_loss(
                    "alpha", alpha_loss, self.hparams.supervision.alpha_weight
                )

            # compute render loss
            if self.hparams.with_render_branch:
                pd_images = renderer_output["pd_images"]  # B, N, H, W, 3
                gt_images = gt_package["gt_images"]  # B, N, H, W, 3

                pixel_loss, perceptual_loss = self.image_loss(
                    pd_images.contiguous(),
                    gt_images,
                    current_epoch=current_epoch,
                    mask=image_loss_mask,
                )
                loss_dict.add_loss(
                    "render", pixel_loss, self.hparams.supervision.render_weight
                )

                if perceptual_loss is not None:
                    loss_dict.add_loss(
                        "render_perceptual",
                        perceptual_loss,
                        self.hparams.perceptual_weight,
                    )

                if compute_metric:
                    with torch.no_grad():
                        # compute PSNR
                        mse = F.mse_loss(pd_images, gt_images, reduction="mean")
                        max_pixel_value = 1.0
                        psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
                        metric_dict.add_loss("PSNR", psnr_value)

                        # compute lpips
                        if self.hparams.use_alex_metric:
                            img0 = pd_images * 2 - 1
                            img1 = gt_images * 2 - 1
                            img0 = img0.permute(0, 1, 4, 2, 3).flatten(
                                0, 1
                            )  # B*N, 3, H, W
                            img1 = img1.permute(0, 1, 4, 2, 3).flatten(
                                0, 1
                            )  # B*N, 3, H, W
                            lpips_value = self.loss_fn_alex(img0, img1)
                            metric_dict.add_loss("LPIPS", lpips_value.mean())

            if self.hparams.use_sup_depth and self.hparams.supervision.depth_weight > 0:
                pd_depths = renderer_output["pd_depths"]  # B, N, H, W, 1
                gt_depths = gt_package["gt_depths"]  # B, N, H, W, 1
                depth_loss_mask = gt_package["depth_loss_mask"]  # B, N, H, W, 1

                B, N, H_pd, W_pd = (
                    pd_depths.size(0),
                    pd_depths.size(1),
                    pd_depths.size(2),
                    pd_depths.size(3),
                )

                if pd_depths.shape[2] != gt_depths.shape[2]:
                    if global_step % 500 == 0:
                        print(
                            f"pd depth {pd_depths.shape} and gt depth {gt_depths.shape} have different resolution, resize gt to match them"
                        )

                    gt_depths_ = (
                        F.interpolate(
                            gt_depths_.flatten(0, 1).permute(0, 3, 1, 2),
                            size=(H_pd, W_pd),
                            mode="nearest",
                        )
                        .permute(0, 2, 3, 1)
                        .view(B, N, H_pd, W_pd, 1)
                    )

                    depth_loss_mask_ = (
                        F.interpolate(
                            depth_loss_mask.flatten(0, 1).permute(0, 3, 1, 2),
                            size=(H_pd, W_pd),
                            mode="nearest",
                        )
                        .permute(0, 2, 3, 1)
                        .view(B, N, H_pd, W_pd, 1)
                    )

                    depth_loss_mask_ *= (gt_depths_ > 0.0).float()  # make sure again
                else:
                    gt_depths_ = gt_depths
                    depth_loss_mask_ = depth_loss_mask

                depth_loss = self.depth_loss(
                    pd_depths, gt_depths_, current_epoch, mask=depth_loss_mask_
                )
                loss_dict.add_loss(
                    "disparity", depth_loss, self.hparams.supervision.depth_weight
                )

            if getattr(self.hparams, "log_gaussian_stats", True):
                # ! log statistics
                decoded_gaussians = network_output["decoded_gaussians"]

                batch_idx = 0
                gaussians: torch.Tensor = decoded_gaussians[batch_idx]
                scaling = gaussians[:, 3:6]
                rots = gaussians[:, 6:10]
                opacities = gaussians[:, 10:11]

                metric_dict.add_loss("scaling-mean", scaling.mean())
                metric_dict.add_loss("scaling-max", scaling.max())
                metric_dict.add_loss("scaling-min", scaling.min())

                metric_dict.add_loss("opacities-mean", opacities.mean())
                metric_dict.add_loss("opacities-max", opacities.max())
                metric_dict.add_loss("opacities-min", opacities.min())

        return loss_dict, metric_dict, latent_dict, vis_images_dict

    def image_loss(self, pd_images, gt_images, current_epoch, mask=None):
        # pd_images: B, N, H, W, 3
        # gt_images: B, N, H, W, 3

        if mask is None:
            mask = torch.ones_like(gt_images, dtype=torch.bool)

        pd_images = pd_images.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 3, H, W
        gt_images = gt_images.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 3, H, W
        mask = mask.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 1, H, W
        # pixel loss
        if self.hparams.use_ssim_loss:
            ssim_loss = 1.0 - ssim_module(
                pd_images, gt_images, mask.expand(*pd_images.shape).bool()
            )
            pixel_loss = (
                0.8
                * torch.sum(F.l1_loss(pd_images, gt_images, reduction="none") * mask)
                / torch.sum(mask)
                + 0.2 * ssim_loss
            )
        else:
            pixel_loss = torch.sum(
                F.l1_loss(pd_images, gt_images, reduction="none") * mask
            ) / torch.sum(mask)
        # percetual loss
        if (
            self.hparams.perceptual_weight > 0.0
            and current_epoch >= self.hparams.perceptual_start_epoch
        ):
            # [0,1] -> [-1, 1]
            pd_images = (pd_images * 2 - 1) * mask
            gt_images = (gt_images * 2 - 1) * mask

            aspect_ratio = pd_images.shape[-1] / pd_images.shape[-2]  # W / H
            # reduce the resolution to save memory
            pd_images = F.interpolate(
                pd_images,
                (
                    int(self.hparams.perceputal_resize_height),
                    int(self.hparams.perceputal_resize_height * aspect_ratio),
                ),
                mode="bilinear",
                align_corners=False,
            )
            gt_images = F.interpolate(
                gt_images,
                (
                    int(self.hparams.perceputal_resize_height),
                    int(self.hparams.perceputal_resize_height * aspect_ratio),
                ),
                mode="bilinear",
                align_corners=False,
            )

            perceptual_loss = self.perceptual_loss(pd_images, gt_images).mean()
        else:
            perceptual_loss = None

        return pixel_loss, perceptual_loss

    def depth_loss(self, pd_depths, gt_depths, current_epoch, mask=None):
        # pd_depths: B, N, H, W, 1
        # gt_depths: B, N, H, W, 1

        if mask is None:
            mask = torch.ones_like(gt_depths, dtype=torch.bool)

        pd_depths = pd_depths.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 1, H, W
        gt_depths = gt_depths.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 1, H, W
        mask = mask.permute(0, 1, 4, 2, 3).flatten(0, 1)  # B*N, 1, H, W

        if self.hparams.supervision.depth_supervision_format == "l1":
            depth_loss = torch.sum(
                F.l1_loss(pd_depths, gt_depths, reduction="none") * mask
            ) / torch.sum(mask)
        elif self.hparams.supervision.depth_supervision_format == "weight":
            pd_depths = (pd_depths - self.hparams.supervision.z_near) / (
                self.hparams.supervision.z_far - self.hparams.supervision.z_near
            )
            gt_depths = (gt_depths - self.hparams.supervision.z_near) / (
                self.hparams.supervision.z_far - self.hparams.supervision.z_near
            )
            depth_loss = torch.sum(
                F.l1_loss(pd_depths, gt_depths, reduction="none") * mask
            ) / torch.sum(mask)
        elif self.hparams.supervision.depth_supervision_format == "affine_invariant":
            mask = mask * (pd_depths > 0.0).float()
            depth_loss = affine_invariant_loss(pd_depths, gt_depths, mask)
        elif (
            self.hparams.supervision.depth_supervision_format == "inverse_metric_depth"
        ):
            pd_depths_inv = torch.where(pd_depths > 0, 1 / pd_depths, pd_depths)
            gt_depths_inv = torch.where(gt_depths > 0, 1 / gt_depths, gt_depths)
            depth_loss = torch.sum(
                F.l1_loss(pd_depths_inv, gt_depths_inv, reduction="none") * mask
            ) / torch.sum(mask)
        else:
            raise ValueError(
                f"Unknown depth supervision format: {self.hparams.supervision.depth_supervision_format}, "
                + "can be 'l1', 'weight', 'affine_invariant', 'inverse_metric_depth'"
            )

        return depth_loss

    def prepare_resized_gt(self, batch):
        batch_len = len(batch[DS.IMAGES])

        # [B, N, H, W, 3]
        gt_images = torch.stack(
            batch[DS.IMAGES], dim=0
        )  # only RGB. note list collate_fn
        gt_masks = torch.stack(
            batch[DS.IMAGES_MASK], dim=0
        ).float()  # only mask. note list collate_fn

        B, N, H, W = (
            gt_images.size(0),
            gt_images.size(1),
            gt_images.size(2),
            gt_images.size(3),
        )

        # resize the gt images
        if self.supervise_image_resize[0] != H or self.supervise_image_resize[1] != W:
            assert isinstance(
                self.supervise_image_resize, omegaconf.listconfig.ListConfig
            )
            resize_h = self.supervise_image_resize[0]
            resize_w = self.supervise_image_resize[1]

            gt_images = (
                F.interpolate(
                    gt_images.flatten(0, 1).permute(0, 3, 1, 2),
                    (resize_h, resize_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .reshape(B, N, resize_h, resize_w, -1)
            )
            gt_masks = (
                F.interpolate(
                    gt_masks.flatten(0, 1).permute(0, 3, 1, 2),
                    (resize_h, resize_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .reshape(B, N, resize_h, resize_w, -1)
            )

        # 0-1 channel is foreground mask from seg (fg pixel value > 0)
        foreground_mask_from_seg = gt_masks[..., 0:1]
        # 1-2 channel is dynamic object mask. 0 is dynamic object, 1 is static object. shape [B, N, H, W, 1]
        dynamic_object_mask = gt_masks[..., 1:2]
        # 2-3 channel is padding or hood mask. 1 is valid area, 0 is padding or hood area. shape [B, N, H, W, 1]
        padding_or_hood_mask = gt_masks[..., 2:3]
        # 3-4 channel is foreground mask from grid (fg pixel value > 0)
        foreground_mask_from_grid = gt_masks[..., 3:4]

        if not self.hparams.model_midground:
            gt_alphas_close = foreground_mask_from_seg * foreground_mask_from_grid
            gt_alphas_midground = torch.zeros_like(foreground_mask_from_seg)
        else:
            gt_alphas_close = foreground_mask_from_seg * foreground_mask_from_grid
            gt_alphas_midground = foreground_mask_from_seg - gt_alphas_close

        # gt_alphas is always the inverse of sky. (sky depends on hparam.model_midground)
        gt_alphas = torch.logical_or(gt_alphas_close, gt_alphas_midground).float()
        gt_alphas_sky = 1.0 - gt_alphas

        if self.hparams.use_sup_depth:
            if self.hparams.sup_depth_type == "lidar_depth":
                gt_depths = torch.stack(
                    batch[DS.IMAGES_DEPTH_LIDAR_PROJECT], dim=0
                )  # [B, N, H, W, 1]
                gt_depths_is_inv = False
            elif self.hparams.sup_depth_type == "rectified_metric3d_depth":
                gt_depths = torch.stack(
                    batch[DS.IMAGES_DEPTH_MONO_EST_RECTIFIED], dim=0
                )
                gt_depths_is_inv = False
            elif self.hparams.sup_depth_type == "depth_anything_v2_depth_inv":
                gt_depths = torch.stack(
                    batch[DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV], dim=0
                )
                gt_depths_is_inv = True
            elif self.hparams.sup_depth_type == "voxel_depth":
                gt_depths = batch[DS.IMAGES_DEPTH_VOXEL]
                gt_depths_is_inv = False
            else:
                raise ValueError(f"Unknown depth type: {self.hparams.sup_depth_type}")

            if (
                self.supervise_image_resize[0] != H
                or self.supervise_image_resize[1] != W
            ):
                gt_depths = (
                    F.interpolate(
                        gt_depths.flatten(0, 1).permute(0, 3, 1, 2),
                        (resize_h, resize_w),
                        mode="nearest-exact",
                    )
                    .squeeze(1)
                    .reshape(B, N, resize_h, resize_w, -1)
                )

            depth_loss_mask = (
                padding_or_hood_mask
                * dynamic_object_mask
                * gt_alphas
                * (gt_depths > 0.0).float()
            )

        else:
            gt_depths = None
            depth_loss_mask = None
            gt_depths_is_inv = None

        if self.hparams.only_sup_foreground:
            image_loss_mask = padding_or_hood_mask * dynamic_object_mask * gt_alphas
        else:
            image_loss_mask = padding_or_hood_mask * dynamic_object_mask

        gt_package = {
            "gt_images": gt_images,
            "gt_alphas": gt_alphas,
            "gt_alphas_close": gt_alphas_close,
            "gt_alphas_midground": gt_alphas_midground,
            "dynamic_object_mask": dynamic_object_mask,
            "padding_or_hood_mask": padding_or_hood_mask,
            "image_loss_mask": image_loss_mask,
            "depth_loss_mask": depth_loss_mask,
            "gt_depths": gt_depths,
            "gt_depths_is_inv": gt_depths_is_inv,
        }

        return gt_package

    def assemble_visualization(self, gt_package, renderer_output):
        vis_images_dict = {
            "gt_images": gt_package["gt_images"],
            "pd_images": renderer_output["pd_images"],
            "gt_alphas": gt_package["gt_alphas"],
            "gt_alphas_close": gt_package["gt_alphas_close"],
            "gt_alphas_midground": gt_package["gt_alphas_midground"],
            "pd_alphas": renderer_output["pd_alphas"],
            "pd_images_fg": renderer_output["pd_images_fg"],
            "image_loss_mask": gt_package["image_loss_mask"],
        }

        if self.hparams.use_sup_depth:
            vis_images_dict["gt_depths"] = (
                gt_package["gt_depths"]
                if not gt_package["gt_depths_is_inv"]
                else torch.where(
                    gt_package["gt_depths"] > 0,
                    1 / gt_package["gt_depths"],
                    gt_package["gt_depths"],
                )
            )
            vis_images_dict["pd_depths"] = renderer_output["pd_depths"]
            vis_images_dict["depth_loss_mask"] = gt_package["depth_loss_mask"]

        return vis_images_dict
