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

def hparams_handler(hparams):
    if not hasattr(hparams, "keep_surface_voxel"):
        hparams.keep_surface_voxel = False
    if not hasattr(hparams, "use_high_res_grid_for_alpha_mask"):
        hparams.use_high_res_grid_for_alpha_mask = False

    if not hasattr(hparams, "use_skybox"):
        hparams.use_skybox = True
    if not hasattr(hparams, "skybox_type"):
        hparams.skybox_type = "panorama_full"
    if not hasattr(hparams, "skybox_resolution"):
        hparams.skybox_resolution = 1024  # pixel

    hparams.with_render_branch = False
    if not hasattr(hparams.supervision, "render_weight"):
        hparams.supervision.render_weight = 0.0
    if hparams.supervision.render_weight > 0:
        hparams.with_render_branch = True
        if not hasattr(hparams, "perceptual_weight"):
            hparams.perceptual_weight = 0.0

    if not hasattr(hparams, "pixel_loss"):
        hparams.pixel_loss = "l1"

    if not hasattr(hparams.supervision, "depth_weight"):
        hparams.supervision.depth_weight = 0.0

    if hparams.supervision.depth_weight == 0:
        hparams.use_sup_depth = False
    else:
        hparams.use_sup_depth = True
        assert hasattr(hparams, "sup_depth_type"), (
            'must specify sup_depth_type, can be "lidar_depth" or "rectified_metric3d_depth" or "voxel_depth"'
        )

    if not hasattr(hparams, "use_ssim_loss"):
        hparams.use_ssim_loss = False
    if not hasattr(hparams, "gs_free_space"):
        hparams.gs_free_space = "soft"
    if not hasattr(hparams, "use_alex_metric"):  # ! can set to default afterward
        hparams.use_alex_metric = False

    if not hasattr(hparams, "render_alpha"):
        hparams.render_alpha = False
    if not hasattr(hparams, "gt_alpha_from"):
        hparams.gt_alpha_from = "grid"
    if not hasattr(hparams, "only_sup_foreground"):
        hparams.only_sup_foreground = False
    if not hasattr(hparams, "render_target_is_object"):
        hparams.render_target_is_object = True

    if not hasattr(hparams, "gsplat_params"):
        hparams.gsplat_params = {"radius_clip": 0, "rasterize_mode": "classic"}

    if not hasattr(hparams, "rasterizing_downsample"):
        hparams.rasterizing_downsample = 1

    return hparams
