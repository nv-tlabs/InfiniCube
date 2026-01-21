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

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.render.gsplat_renderer import (
    GsplatPinholeCamera,
    render_gsplat_func,
)


class RGBRenderer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def save_decoder(self, gaussian_saving_path):
        """
        no decoder to save for RGBRenderer
        """
        pass

    def prepare_rasterizing_params(self, batch):
        target_intrinsics = torch.stack(batch[DS.IMAGES_INTRINSIC])
        rasterizing_target_intrinsics = (
            target_intrinsics / self.hparams.rasterizing_downsample
        )

        target_poses = torch.stack(batch[DS.IMAGES_POSE])

        rasterizing_params = {
            "target_poses": target_poses,
            "rasterizing_target_intrinsics": rasterizing_target_intrinsics,
        }
        return rasterizing_params

    def gsplat_render(self, rasterizing_params: dict, network_output: dict, skybox):
        """
        rasterizing_params:
            target_pose:
                [B, N, 4, 4]
            rasterizing_target_intrinsic:
                [B, N, 6]
        """
        target_poses = rasterizing_params["target_poses"]
        rasterizing_target_intrinsics = rasterizing_params[
            "rasterizing_target_intrinsics"
        ]

        assert (
            self.hparams.with_render_branch
            and "decoded_gaussians" in network_output.keys()
        )

        batch_rendered_images = []
        batch_rendered_images_fg = []
        batch_rendered_alphas = []
        batch_rendered_depth = []

        decoded_gaussians = network_output["decoded_gaussians"]
        renderer_output = {}

        for batch_idx in range(len(decoded_gaussians)):
            # n_dup: int = self.hparams.gsplat_upsample
            gaussians: torch.Tensor = decoded_gaussians[batch_idx]
            target_poses_one_batch = target_poses[batch_idx]
            target_intrinsic = rasterizing_target_intrinsics[batch_idx]

            one_sample_cameras = []
            for i, camera_pose in enumerate(target_poses_one_batch):
                intrinsic = target_intrinsic[i]
                camera = GsplatPinholeCamera(
                    "cuda",
                    int(intrinsic[5]),
                    int(intrinsic[4]),
                    intrinsic[0],
                    intrinsic[1],
                    intrinsic[2],
                    intrinsic[3],
                )
                camera.pose.set_from_torch(camera_pose)
                one_sample_cameras.append(camera)

            # if the number of views is less than 20, we render them all at once
            if len(one_sample_cameras) < 20:
                # render_image shape [N_views, H, W, C] (C can be feature dim or RGB), depth and alpha shape [N_views, H, W, 1]
                render_images, render_depths, render_alphas = render_gsplat_func(
                    one_sample_cameras,
                    None,
                    gaussians[:, :3],
                    gaussians[:, 3:6],
                    gaussians[:, 6:10],
                    gaussians[:, 10:11],
                    gaussians[:, 11:],
                    bg=None,
                    free_space=None,
                    **self.hparams.gsplat_params,
                )
            else:
                # else we render them one by one
                render_images = []
                render_depths = []
                render_alphas = []
                for i, camera in enumerate(one_sample_cameras):
                    render_image, render_depth, render_alpha = render_gsplat_func(
                        [camera],
                        None,
                        gaussians[:, :3],
                        gaussians[:, 3:6],
                        gaussians[:, 6:10],
                        gaussians[:, 10:11],
                        gaussians[:, 11:],
                        bg=None,
                        free_space=None,
                        **self.hparams.gsplat_params,
                    )
                    render_images.append(render_image)
                    render_depths.append(render_depth)
                    render_alphas.append(render_alpha)

                render_images = torch.cat(render_images, dim=0)
                render_depths = torch.cat(render_depths, dim=0)
                render_alphas = torch.cat(render_alphas, dim=0)

            # sky_image. [N_view, H, W, C].
            if self.hparams.use_skybox:
                sky_image = skybox.sample_batch(
                    target_poses_one_batch, target_intrinsic, network_output, batch_idx
                )

                # assert self.hparams.skybox_net == 'identity', "We restrict the skybox_net to be identity now, but it can be changed."
                render_images_with_sky = render_images + (1 - render_alphas) * sky_image
            else:
                render_images_with_sky = render_images

            batch_rendered_images.append(render_images_with_sky)
            batch_rendered_images_fg.append(render_images)
            batch_rendered_depth.append(render_depths)
            batch_rendered_alphas.append(render_alphas)

        torch.cuda.empty_cache()
        # stack for batch
        renderer_output.update({"pd_images": torch.stack(batch_rendered_images, dim=0)})
        renderer_output.update(
            {"pd_images_fg": torch.stack(batch_rendered_images_fg, dim=0)}
        )
        renderer_output.update({"pd_depths": torch.stack(batch_rendered_depth, dim=0)})
        renderer_output.update({"pd_alphas": torch.stack(batch_rendered_alphas, dim=0)})

        return renderer_output

    def forward(self, batch: dict, network_output: dict, skybox):
        rasterizing_params = self.prepare_rasterizing_params(batch)
        renderer_output = self.gsplat_render(rasterizing_params, network_output, skybox)
        return renderer_output
