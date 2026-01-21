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

import imageio.v3 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.modules.basic_modules import ResBlock
from infinicube.voxelgen.utils.render_util import (
    create_rays_from_intrinsic_torch,
    create_rays_from_intrinsic_torch_batch,
    from_opengl,
    to_opengl,
)


def world2latlong(xyz):
    """
    https://github.com/yifanlu0227/skylibs/blob/f9bbf0ab30a61a4cb8963a779d379c1b94f022d0/envmap/projections.py#L15C1-L22C16
    Get the (u, v) coordinates of the point defined by (x, y, z) for
    a latitude-longitude map
    (u, v) coordinates are in the [0, 1] interval.

    (0, 0)--------------------> (u=1)
    |
    |
    v (v=1)


    Args:
        xyz: np.ndarray or torch.Tensor, shape [..., 3]. Needs to be OpenGL coordinates
    Returns:
        uv: np.ndarray or torch.Tensor, shape [..., 2]
    """
    if isinstance(xyz, np.ndarray):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = 1 + (1 / np.pi) * np.arctan2(x, -z)
        v = (1 / np.pi) * np.arccos(y)
        u = u / 2
        return np.stack([u, v], axis=-1)

    elif isinstance(xyz, torch.Tensor):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = 1 + (1 / np.pi) * torch.atan2(x, -z)
        v = (1 / np.pi) * torch.acos(y)
        u = u / 2
        return torch.stack([u, v], dim=-1)

    else:
        raise NotImplementedError


def get_image_coordinates(h, w):
    """Returns the (u, v) coordinates in range (0, 1) for each pixel center."""
    assert w == 2 * h
    cols = np.linspace(0, 1, 2 * w + 1)
    rows = np.linspace(0, 1, 2 * h + 1)
    cols = cols[1::2]
    rows = rows[1::2]

    return [d.astype("float32") for d in np.meshgrid(cols, rows)]


def latlong2world(uv):
    """Get the (x, y, z) coordinates of the point defined by (u, v)
    for a latlong map.
    Args:
        uv: np.ndarray or torch.Tensor, shape [..., 2]
    Returns:
        xyz: np.ndarray or torch.Tensor, shape [..., 3]
    """
    if isinstance(uv, np.ndarray):
        u, v = uv[..., 0], uv[..., 1]
        u = u * 2

        # lat-long -> world
        thetaLatLong = np.pi * (u - 1)
        phiLatLong = np.pi * v

        x = np.sin(phiLatLong) * np.sin(thetaLatLong)
        y = np.cos(phiLatLong)
        z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

        return np.stack([x, y, z], axis=-1)

    elif isinstance(uv, torch.Tensor):
        u, v = uv[..., 0], uv[..., 1]
        u = u * 2

        # lat-long -> world
        thetaLatLong = np.pi * (u - 1)
        phiLatLong = np.pi * v

        x = torch.sin(phiLatLong) * torch.sin(thetaLatLong)
        y = torch.cos(phiLatLong)
        z = -torch.sin(phiLatLong) * torch.cos(thetaLatLong)

        return torch.stack([x, y, z], dim=-1)

    else:
        raise NotImplementedError


def get_world_directions_latlong(h, w, type="torch", device="cpu"):
    """Returns the world-space direction in range [-1, 1] for each pixel center."""
    uvs = get_image_coordinates(h, w)
    uvs = np.stack(uvs, axis=-1)  # [H, W, 2]

    if type == "torch":
        uvs = torch.from_numpy(uvs).to(device)

    # Convert to world-space directions with skylatlong2world
    xyz = latlong2world(uvs)
    return xyz


def sample_panorama_full_from_camera(pose_matrice, intrinsic, panorama):
    """
    Args:
        pose_matrice : torch.tensor
            camera pose matrix, shape [4, 4]

        intrinsic : torch.tensor
            camera intrinsic, shape [6, ]

        panorama : torch.tensor
            panorama to sample, shape [H, 2*H, C]

    Returns:
        skybox_color : torch.tensor
            sampled color given pose_matrice and intrinsic, shape [H', W', C]
    """
    camera_origin, ray_d_world = create_rays_from_intrinsic_torch(
        pose_matrice, intrinsic
    )
    ray_d_opengl = to_opengl(ray_d_world)
    uv = world2latlong(ray_d_opengl)  # [H', W', 2], range in [0, 1]

    # sampling using nn.functional.grid_sample
    panorama = panorama.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    grid = uv.unsqueeze(0) * 2 - 1  # [1, H', W', 2], range in [-1, 1]
    skybox_color = nn.functional.grid_sample(
        panorama, grid, align_corners=True
    )  # [1, C, H', W']

    return skybox_color.squeeze(0).permute(1, 2, 0)


def sample_panorama_full_from_camera_batch(pose_matrices, intrinsics, panorama):
    """
    Args:
        pose_matrices : torch.tensor
            camera pose matrix, shape [N, 4, 4]

        intrinsics : torch.tensor
            camera intrinsic, shape [N, 6]

        panorama : torch.tensor
            panorama to sample, shape [H, 4*H, C], note that this is not batched

    Returns:
        skybox_color : torch.tensor
            sampled color given pose_matrice and intrinsic, shape [N, H', W', C]
    """
    N = pose_matrices.shape[0]
    # [N, 3] and [N, H', W', 3]
    camera_origin, ray_d_world = create_rays_from_intrinsic_torch_batch(
        pose_matrices, intrinsics
    )
    ray_d_opengl = to_opengl(ray_d_world)  # [N, H', W', 3]
    uv = world2latlong(ray_d_opengl)  # [N, H', W', 2], range in [0, 1]

    # sampling using nn.functional.grid_sample
    panoramas = panorama.expand(N, -1, -1, -1).permute(0, 3, 1, 2)  # [N, C, H, 4*H]
    grid = uv * 2 - 1  # [N, H', W', 2], range in [-1, 1]

    skybox_color = nn.functional.grid_sample(
        panoramas, grid, align_corners=True
    )  # [N, C, H', W']

    return skybox_color.permute(0, 2, 3, 1)


class SkyboxPanoramaFull(nn.Module):
    def __init__(self, hparams):
        super(SkyboxPanoramaFull, self).__init__()
        self.hparams = hparams
        self.get_world_directions = get_world_directions_latlong
        self.skybox_panorama_height = self.hparams.skybox_resolution
        self.skybox_panorama_width = 2 * self.hparams.skybox_resolution
        world_sky_directions = from_opengl(
            self.get_world_directions(
                self.skybox_panorama_height, self.skybox_panorama_width
            )
        )
        # self.register_buffer('world_sky_directions', world_sky_directions)
        self.world_sky_directions = world_sky_directions
        self.skybox_forward_sky_only = getattr(
            self.hparams, "skybox_forward_sky_only", True
        )
        logger.info(
            f"SkyboxPanoramaFull skybox_forward_sky_only: {self.skybox_forward_sky_only}"
        )

        if self.hparams.skybox_net == "conv-c16":
            # 3x3 convolutions with stride 1 and padding 2
            self.skybox_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.skybox_in_dim, 16, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            )

        elif self.hparams.skybox_net == "conv-c32":
            # 3x3 convolutions with stride 1 and padding 2
            self.skybox_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.skybox_in_dim, 32, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            )

        elif self.hparams.skybox_net == "conv-c64":
            # 3x3 convolutions with stride 1 and padding 2
            self.skybox_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.skybox_in_dim, 64, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            )

        elif self.hparams.skybox_net == "conv-c32-decode-3":
            self.skybox_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.skybox_in_dim, 32, kernel_size=5, stride=1, padding=2
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
            )

        elif self.hparams.skybox_net == "res-up4x-decode-3":
            # resolution of feature panorama and the RGB panorama are different,
            # but it is fine for the grid_sample (which use normalized coordinate)
            self.skybox_net = nn.Sequential(
                ResBlock(
                    self.hparams.skybox_in_dim,
                    out_channels=self.hparams.skybox_in_dim // 2,
                    up=True,
                    use_gn=False,
                ),
                ResBlock(
                    self.hparams.skybox_in_dim // 2,
                    out_channels=self.hparams.skybox_in_dim // 4,
                    up=True,
                    use_gn=False,
                ),
                nn.Conv2d(
                    self.hparams.skybox_in_dim // 4,
                    3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),  # RGB
            )

        elif self.hparams.skybox_net == "conv-up4":
            # 5x5 convolutions with stride 1 and padding 2
            self.skybox_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.skybox_in_dim, 32, kernel_size=5, stride=1, padding=2
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2),
            )
        elif self.hparams.skybox_net == "feature-identity":
            # this is used for distinguish feature input from RGB input.
            self.skybox_net = nn.Identity()
        elif self.hparams.skybox_net == "identity":
            self.skybox_net = nn.Identity()

    def encode_sky_feature(self, batch, imgenc_output):
        """
        This is originally used in Encoder, we encode the sky feature from images, store them in some representation (e.g. panorama).
        Args:
            camera_pose_stack: torch.Tensor
                camera pose matrix, shape [B, N, 4, 4]
            intrinsics: torch.Tensor
                camera intrinsic, shape [B, N, 6]
            image_or_feature: torch.Tensor
                image to be sampled, shape [B, N, C, H, W]. can be RGB image or feature
            sky_mask: torch.Tensor
                sky mask, shape [B, N, 1, H, W]. 1 is sky, 0 is non-sky

        Returns:
            skyenc_output: dict
                we would update the sky panorama (`skybox_feat`) and mask (`skybox_mask`) in the feature_dict
                - skybox_feat: torch.Tensor, shape [B, H', W', C]
                - skybox_mask: torch.Tensor, shape [B, H', W', 1]
        """
        if (
            self.world_sky_directions.device
            != imgenc_output[self.hparams.skybox_feature_source].device
        ):
            self.world_sky_directions = self.world_sky_directions.to(
                imgenc_output[self.hparams.skybox_feature_source].device
            )

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

        sky_mask = sky_mask.permute(0, 1, 4, 2, 3).float()  # [B, N, 1, H, W]
        image_or_feature = imgenc_output[self.hparams.skybox_feature_source]

        sampled_feature_batch = []
        panorama_mask_batch = []
        B, n_imgs = camera_pose_stack.shape[0], camera_pose_stack.shape[1]

        for bidx in range(B):
            cur_pose_inv = torch.linalg.inv(camera_pose_stack[bidx])  # world to camera
            cur_world_sky_directions = self.world_sky_directions.unsqueeze(0).repeat(
                n_imgs, 1, 1, 1
            )  # [N, H', W', 3]
            cur_camera_sky_directions = torch.einsum(
                "nab,nhwb->nhwa", cur_pose_inv[:, :3, :3], cur_world_sky_directions
            )  # we only need rotation part

            cur_incoming_ray_mask = (
                cur_camera_sky_directions[:, :, :, 2] > 0
            ).unsqueeze(1)  # [N, 1, H', W']
            proj_xyz = (
                cur_camera_sky_directions / cur_camera_sky_directions[:, :, :, 2:3]
            )  # on normalized plane

            # we will use the intrinsics to get projected pixel coordinate, and make them in range [-1, 1]
            intrinsic_matrix = torch.zeros((n_imgs, 3, 3), device=intrinsics.device)
            intrinsic_matrix[:, 0, 0] = intrinsics[bidx, :, 0]
            intrinsic_matrix[:, 1, 1] = intrinsics[bidx, :, 1]
            intrinsic_matrix[:, 0, 2] = intrinsics[bidx, :, 2]
            intrinsic_matrix[:, 1, 2] = intrinsics[bidx, :, 3]
            intrinsic_matrix[:, 2, 2] = 1
            # need check the einsum
            proj_pixel = torch.einsum("nab,nhwb->nhwa", intrinsic_matrix, proj_xyz)[
                ..., :2
            ]  # [N, H', W', 2], pixel coordinate
            # if mv images' shapes are different, need to change here.
            W = intrinsics[bidx, :, 4]
            H = intrinsics[bidx, :, 5]  # [N]
            WH = (
                torch.stack([W, H], dim=1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, proj_xyz.shape[1], proj_xyz.shape[2], 1)
            )
            # make the projected proj_pixel in range [-1, 1]
            proj_pixel = (
                proj_pixel / WH * 2
            ) - 1  # [N, H', W', 3], pixel coordinate in [-1, 1]

            # [N, C, H', W']
            sampled_feature = F.grid_sample(
                image_or_feature[bidx],
                proj_pixel,
            )

            # [N, 1, H', W']
            panorama_mask = F.grid_sample(
                sky_mask[bidx],
                proj_pixel,
            )

            sampled_feature *= cur_incoming_ray_mask  # mask out invalid projection
            panorama_mask *= cur_incoming_ray_mask  # mask out invalid projection

            sampled_feature = sampled_feature.permute(0, 2, 3, 1)  # [N, H', W', C]
            panorama_mask = panorama_mask.permute(0, 2, 3, 1)  # [N, H', W', 1]

            # maxpooling N views, [H', W', C]
            sampled_feature = torch.max(sampled_feature, dim=0)[0]  # [H', W', C]
            panorama_mask = torch.max(panorama_mask, dim=0)[0] > 0  # [H', W', 1]

            sampled_feature_batch.append(sampled_feature)
            panorama_mask_batch.append(panorama_mask)

        skyenc_output = {}
        skyenc_output["skybox_feat"] = torch.stack(
            sampled_feature_batch
        )  # [B, H', W', C]
        skyenc_output["skybox_mask"] = torch.stack(
            panorama_mask_batch
        )  # [B, H', W', 1]

        return skyenc_output

    def forward(self, skyenc_output, network_output):
        """
        Args:
            skyenc_output : dict
                feature dictionary, containing 'skybox_feat', shape [B, H, W, 3/C], 'skybox_mask', shape [B, H, W, 1]
            network_output : dict
                output dictionary to store intermediate results from network
        Returns:
            out : dict
                updated output dictionary with 'skybox_representation', shape [B, H, W, 3/C], not necessary RGB
        """
        if self.skybox_forward_sky_only:
            skybox_feat = skyenc_output["skybox_feat"] * skyenc_output["skybox_mask"]
        else:
            skybox_feat = skyenc_output["skybox_feat"]

        skybox_representation = self.skybox_net(
            skybox_feat.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        network_output.update(
            {
                "skybox_representation": skybox_representation,  # [B, H, W, 3/C]
                "skybox_representation_mask": skyenc_output["skybox_mask"],
            }
        )

        return network_output

    @staticmethod
    def sample(camera_pose, intrinsic, one_skybox_representation):
        """
        Args:
            camera_pose : torch.tensor
                camera pose matrix, shape [4, 4]
            intrinsic : torch.tensor
                camera intrinsic, shape [6, ]
            skybox_representation : torch.tensor
                panorama to sample in this case, shape [H, 2*H, C]
        Returns:
            sampled_image : torch.tensor
                sampled color given pose_matrice and intrinsic, shape [H', W', C]
        """
        return sample_panorama_full_from_camera(
            camera_pose, intrinsic, one_skybox_representation
        )

    def sample_batch(self, camera_pose_stack, intrinsics, network_output, batch_idx):
        """
        Args:
            camera_pose_stack : torch.tensor
                camera pose matrix, shape [N, 4, 4]
            intrinsics : torch.tensor
                camera intrinsic, shape [N, 6]
            network_output : dict, contains
                skybox_representation : torch.tensor
                    panorama to sample, shape [B, H, 2*H, C], note it's B not N
        Returns:
            sampled_image : torch.tensor
                sampled color given pose_matrice and intrinsic, shape [N, H', W', C]
        """
        skybox_representation = network_output["skybox_representation"]

        return sample_panorama_full_from_camera_batch(
            camera_pose_stack, intrinsics, skybox_representation[batch_idx]
        )

    def visualize(self, network_output):
        """
        visualize the skybox_representation. Now we can only visualize if RGB is stored, not feature

        Args:
            network_output : dict
                output dictionary containing 'skybox_representation', shape [B, H, W, 3]
        Returns:
            skybox_representation : torch.tensor
                skybox color panorama, shape [B, H, W, 3]
        """
        return network_output["skybox_representation"]

    def save_skybox(self, network_output, gaussian_saving_path):
        """
        save the sky_representation to a file, but following gs_utils's configuration.

        - `{gs_stem}_pano.png` for RGB panorama
        - `{gs_stem}_pano.pt` for feature panorama
        - `{gs_stem}_pano_mask.png` for mask panorama
        - `[H, 2H, 3]` for paranoram and `[H, 2H / 4H]` for mask

        Here we save the panorama and panorama mask.
        """
        panorama = network_output["skybox_representation"]  # [B, H, W, 3/C]
        panorama_mask = network_output["skybox_representation_mask"]  # [B, H, W, 1]

        assert panorama.shape[0] == 1, "support batch size 1 only"
        panorama = panorama.squeeze(0)  # [H, W, 3/C]
        panorama_mask = panorama_mask.squeeze(0).squeeze(-1)  # [H, W]

        is_rgb_panorama = panorama.shape[-1] == 3  # RGB panorama or feature panorama
        gs_stem = Path(gaussian_saving_path).with_suffix("").as_posix()

        if is_rgb_panorama:
            panorama_saving_path = gs_stem + "_pano.png"
            panorama_mask_saving_path = gs_stem + "_pano_mask.png"

            panorama = (
                (panorama * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            )  # [H, W, 3]
            panorama_mask = (
                ((panorama_mask > 0).float() * 255)
                .clamp(0, 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )  # [H, W]

            imageio.imwrite(panorama_saving_path, panorama)
            imageio.imwrite(panorama_mask_saving_path, panorama_mask)

        else:
            panorama_saving_path = gs_stem + "_pano.pt"
            panorama_mask_saving_path = gs_stem + "_pano_mask.png"

            panorama = panorama.cpu()  # [H, W, C]
            panorama_mask = (
                ((panorama_mask > 0).float() * 255)
                .clamp(0, 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )  # [H, W]

            torch.save(panorama, panorama_saving_path)
            imageio.imwrite(panorama_mask_saving_path, panorama_mask)

    def merge_multi_sample(self, multiple_skybox_representation):
        """
        Args:
            multiple_skybox_representation : list
                list of skybox_representation, each with shape [B, H, W, 3]
        """
        assert len(multiple_skybox_representation) > 0
        assert multiple_skybox_representation[0].shape[0] == 1, (
            "support batch size 1 only"
        )

        return torch.stack(multiple_skybox_representation).max(dim=0)[0]


if __name__ == "__main__":
    # test sample_panorama_full_from_camera and sample_panorama_full_from_camera_batch are the same
    import torch

    pose_matrices = torch.eye(4).unsqueeze(0).repeat(2, 1, 1).cuda()
    intrinsics = torch.tensor([200, 200, 400, 300, 800, 600]).expand(2, 6).cuda()
    panorama = torch.randn(600, 800, 3).cuda()

    sampled_panorama = torch.stack(
        [
            sample_panorama_full_from_camera(pose_matrices[i], intrinsics[i], panorama)
            for i in range(2)
        ]
    )
    sampled_panorama_batch = sample_panorama_full_from_camera_batch(
        pose_matrices, intrinsics, panorama
    )

    assert torch.allclose(sampled_panorama, sampled_panorama_batch)
