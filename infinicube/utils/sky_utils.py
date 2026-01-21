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
from PIL import Image
from termcolor import colored


def to_opengl(ray_d):
    """
    transform the ray direction vector into OpenGL convention (-z is FRONT, +x is RIGHT, +y is UP)
    FLU to RUB

    Attention! The waymo dataset processed by Jiahui is RFU, but the waymo_wds are FLU!
    The code still works, but the saved panorama/cubemap is rotated.

            z                        y
            |  x (front)             |
            |/                       |
    y <-----o    ===========>>       o----> x
                                    /
                                 z /
                                (back)
    Args:
        ray_d : torch.tensor
            shape [*, 3]
    """
    return torch.cat([-ray_d[..., 1:2], ray_d[..., 2:3], -ray_d[..., 0:1]], dim=-1)


def from_opengl(ray_d):
    """
    transform the ray direction vector from OpenGL convention to our convention (+y is front, +x is right, +z is up)
    FLU to RFU

    Attention! The waymo dataset processed by Jiahui is RFU, but the waymo_wds are FLU!
    The code still works, but the saved panorama/cubemap is rotated.

            z                        y
            |  x (front)             |
            |/                       |
    y <-----o    <<===========       o----> x
                                    /
                                 z /
                                (back)
    Args:
        ray_d : torch.tensor
            shape [*, 3]
    """
    return torch.cat([-ray_d[..., 2:3], -ray_d[..., 0:1], ray_d[..., 1:2]], dim=-1)


def world2skylatlong(xyz):
    """
    https://github.com/yifanlu0227/skylibs/blob/f9bbf0ab30a61a4cb8963a779d379c1b94f022d0/envmap/projections.py#L15C1-L22C16
    Get the (u, v) coordinates of the point defined by (x, y, z) for
    a sky-latitude-longitude map (the zenith hemisphere of a latlong map).
    (u, v) coordinates are in the [0, 1] interval.

    (0, 0)--------------------> (u=1)
    |
    |
    v (v=1)


    Args:
        xyz: np.ndarray or torch.Tensor, shape [..., 3]
    Returns:
        uv: np.ndarray or torch.Tensor, shape [..., 2]
    """
    if isinstance(xyz, np.ndarray):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = 1 + (1 / np.pi) * np.arctan2(x, -z)
        v = (1 / np.pi) * np.arccos(y) * 2
        u = u / 2
        return np.stack([u, v], axis=-1)

    elif isinstance(xyz, torch.Tensor):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = 1 + (1 / np.pi) * torch.atan2(x, -z)
        v = (1 / np.pi) * torch.acos(y) * 2
        u = u / 2
        return torch.stack([u, v], dim=-1)

    else:
        raise NotImplementedError


def world2latlong(xyz):
    """
    https://github.com/yifanlu0227/skylibs/blob/f9bbf0ab30a61a4cb8963a779d379c1b94f022d0/envmap/projections.py#L15C1-L22C16
    Get the (u, v) coordinates of the point defined by (x, y, z) for
    a latitude-longitude map (the zenith hemisphere of a latlong map).
    (u, v) coordinates are in the [0, 1] interval.

    (0, 0)--------------------> (u=1)
    |
    |
    v (v=1)


    Args:
        xyz: np.ndarray or torch.Tensor, shape [..., 3]
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
    assert w == 4 * h
    cols = np.linspace(0, 1, 2 * w + 1)
    rows = np.linspace(0, 1, 2 * h + 1)
    cols = cols[1::2]
    rows = rows[1::2]

    return [d.astype("float32") for d in np.meshgrid(cols, rows)]


def skylatlong2world(uv):
    """Get the (x, y, z) coordinates of the point defined by (u, v)
    for a sky latlong map.
    Args:
        uv: np.ndarray, shape [..., 2]
    Returns:
        xyz: np.ndarray, shape [..., 3]
    """
    u, v = uv[..., 0], uv[..., 1]
    u = u * 2

    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2
    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    xyz = np.stack([x, y, z], axis=-1)

    return xyz


def latlong2world(uv):
    """Get the (x, y, z) coordinates of the point defined by (u, v)
    for a latlong map.
    Args:
        uv: np.ndarray, shape [..., 2]
    Returns:
        xyz: np.ndarray, shape [..., 3]
    """
    u, v = uv[..., 0], uv[..., 1]
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    xyz = np.stack([x, y, z], axis=-1)

    return xyz


def get_world_directions(h, w):
    """Returns the world-space direction in range [-1, 1] for each pixel center."""
    uvs = get_image_coordinates(h, w)
    uvs = np.stack(uvs, axis=-1)  # [H, W, 2]

    if h * 4 == w:
        # Convert to world-space directions with skylatlong2world
        xyz = skylatlong2world(uvs)
    elif h * 2 == w:
        # Convert to world-space directions with latlong2world
        xyz = latlong2world(uvs)

    return xyz


def create_rays_from_intrinsic(pose_matric, intrinsic):
    """
    Args:
        pose_matric: (4, 4)
        intrinsic: (6, ), [fx, fy, cx, cy, w, h]
    Returns:
        camera_origin: (3, )
        d: (H, W, 3)
    """
    if isinstance(pose_matric, torch.Tensor):
        pose_matric = pose_matric.cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()

    camera_origin = pose_matric[:3, 3]
    fx, fy, cx, cy, w, h = intrinsic
    ii, jj = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")  # ! fix error
    uu, vv = (ii - cx) / fx, (jj - cy) / fy
    local_xyz = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    local_xyz = np.concatenate([local_xyz, np.ones((int(h), int(w), 1))], axis=-1)
    pixel_xyz = np.einsum("ij, hwj->hwi", pose_matric, local_xyz)[
        :, :, :3
    ]  # (H, W, 3) # ! fix error

    d = pixel_xyz - camera_origin
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    return camera_origin, d


def sample_panorama_hemi_from_camera(pose_matrice, intrinsic, panorama):
    """
    Args:
        pose_matrice : torch.tensor
            camera pose matrix, shape [4, 4]

        intrinsic : torch.tensor
            camera intrinsic, shape [6, ], fx fy cx cy w h

        panorama : torch.tensor
            panorama to sample, shape [H, 4*H, C]

    Returns:
        skybox_color : torch.tensor
            sampled color given pose_matrice and intrinsic, shape [H', W', C]
    """
    camera_origin, ray_d_world = create_rays_from_intrinsic(pose_matrice, intrinsic)
    ray_d_world = torch.from_numpy(ray_d_world).to(panorama)
    ray_d_opengl = to_opengl(ray_d_world)
    uv = world2skylatlong(ray_d_opengl)  # [H', W', 2], range in [0, 1]

    # sampling using nn.functional.grid_sample
    panorama = panorama.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    grid = uv.unsqueeze(0) * 2 - 1  # [1, H', W', 2], range in [-1, 1]
    skybox_color = nn.functional.grid_sample(
        panorama, grid, align_corners=True
    )  # [1, C, H', W']

    return skybox_color.squeeze(0).permute(1, 2, 0)


def sample_panorama_full_from_camera(pose_matrice, intrinsic, panorama):
    """
    Args:
        pose_matrice : torch.tensor
            camera pose matrix, shape [4, 4]

        intrinsic : torch.tensor
            camera intrinsic, shape [6, ]

        panorama : torch.tensor
            panorama to sample, shape [H, 4*H, C]

    Returns:
        skybox_color : torch.tensor
            sampled color given pose_matrice and intrinsic, shape [H', W', C]
    """
    camera_origin, ray_d_world = create_rays_from_intrinsic(pose_matrice, intrinsic)
    ray_d_world = torch.from_numpy(ray_d_world).to(panorama)
    ray_d_opengl = to_opengl(ray_d_world)
    uv = world2latlong(ray_d_opengl)  # [H', W', 2], range in [0, 1]

    # sampling using nn.functional.grid_sample
    panorama = panorama.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    grid = uv.unsqueeze(0) * 2 - 1  # [1, H', W', 2], range in [-1, 1]
    skybox_color = nn.functional.grid_sample(
        panorama, grid, align_corners=True
    )  # [1, C, H', W']

    return skybox_color.squeeze(0).permute(1, 2, 0)


def sample_cubemap_from_camera(pose_matrice, intrinsic, cubemap):
    """
    Args:
        pose_matrice : torch.tensor
            camera pose matrix, shape [4, 4]

        intrinsic : torch.tensor
            camera intrinsic, shape [6, ]

        cubemap : torch.tensor
            cube map to sample, shape [6, N, N, C]
    """

    def sample_ray_d(ray_d, cubemap):
        """
        Args:
            ray_d : torch.tensor
                ray direction tensor in world, shape [*, 3]

            cubemap : torch.tensor
                cube map to sample, shape [6, N, N, C]

        Returns:
            sampled : torch.tensor
                sampled pixel value from cube map. shape [*, cubemap_dim]
        """
        import nvdiffrast.torch as dr

        ray_d = ray_d.contiguous()
        prefix = ray_d.shape[:-1]

        # reshape to [minibatch_size, height, width, 3]
        if len(prefix) != 3:
            ray_d = ray_d.reshape(1, 1, -1, ray_d.shape[-1])

        # texture input should be [minibatch_size、6、tex_height、tex_width、tex_channels]
        sampled = dr.texture(
            cubemap[None, ...], ray_d, filter_mode="linear", boundary_mode="cube"
        )
        sampled = sampled.view(*prefix, -1)

        return sampled

    camera_origin, ray_d_world = create_rays_from_intrinsic(pose_matrice, intrinsic)
    ray_d_world = torch.from_numpy(ray_d_world).to(cubemap)
    ray_d_opengl = to_opengl(ray_d_world)

    sampled_img = sample_ray_d(ray_d_opengl, cubemap)  # [H, W, embedding_dim]

    return sampled_img


def render_sky_panorama_hemi(skybox, camera_to_world, height, width, FoVy, FoVx):
    assert skybox.shape[0] * 4 == skybox.shape[1]
    pose_matrix = camera_to_world
    w = width
    h = height
    fy = h / (2 * np.tan(FoVy / 2))
    fx = w / (2 * np.tan(FoVx / 2))
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([fx, fy, cx, cy, w, h])

    # panorama (hemisphere)
    if isinstance(skybox, torch.Tensor):
        skybox_tensor = skybox
    elif isinstance(skybox, np.ndarray):
        skybox_tensor = torch.tensor(skybox).float().cuda()
    else:
        raise NotImplementedError("skybox should be torch.Tensor or np.ndarray")

    sampled_color = sample_panorama_hemi_from_camera(
        pose_matrix, intrinsic, skybox_tensor
    )  # [h, w, 3]

    if isinstance(skybox, torch.Tensor):
        return sampled_color
    else:
        return sampled_color.cpu().numpy()


def render_sky_panorama_full(skybox, camera_to_world, height, width, FoVy, FoVx):
    assert skybox.shape[0] * 2 == skybox.shape[1]
    pose_matrix = camera_to_world
    w = width
    h = height
    fy = h / (2 * np.tan(FoVy / 2))
    fx = w / (2 * np.tan(FoVx / 2))
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([fx, fy, cx, cy, w, h])

    # panorama (sphere)
    if isinstance(skybox, torch.Tensor):
        skybox_tensor = skybox
    elif isinstance(skybox, np.ndarray):
        skybox_tensor = torch.tensor(skybox).float().cuda()
    else:
        raise NotImplementedError("skybox should be torch.Tensor or np.ndarray")

    sampled_color = sample_panorama_full_from_camera(
        pose_matrix, intrinsic, skybox_tensor
    )  # [h, w, 3]

    if isinstance(skybox, torch.Tensor):
        return sampled_color
    else:
        return sampled_color.cpu().numpy()


def convert_cubemap(img, view_name):
    """
    Args:
        img : np.ndarray
            [res, res, 3]
        view_name : str
            one of ['front', 'back', 'left', 'right', 'top', 'bottom']

    if view name is in ['front', 'back', 'left', 'right'], horizontal flip the image!
    if view name is in ['top', 'bottom'], vertical flip the image!
    """
    if view_name in ["front", "back", "left", "right"]:
        img = np.flip(img, axis=1)
    elif view_name in ["top", "bottom"]:
        img = np.flip(img, axis=0)
    return img


def render_sky_cubemap(skybox, camera_to_world, height, width, FoVy, FoVx):
    assert skybox.shape[0] * 4 == skybox.shape[1] * 3
    skybox = skybox[..., :3]
    N = skybox.shape[0] // 3
    cubemap_vis = np.zeros((6, N, N, 3))
    cubemap_store = np.zeros((6, N, N, 3))
    cubemap_vis[2] = skybox[:N, N : 2 * N]  # top
    cubemap_vis[1] = skybox[N : 2 * N, :N]  # left
    cubemap_vis[5] = skybox[N : 2 * N, N : 2 * N]  # front
    cubemap_vis[0] = skybox[N : 2 * N, 2 * N : 3 * N]  # right
    cubemap_vis[4] = skybox[N : 2 * N, 3 * N : 4 * N]  # back
    cubemap_vis[3] = skybox[2 * N : 3 * N, N : 2 * N]  # bottom

    view_names = ["right", "left", "top", "bottom", "back", "front"]
    for idx, view_name in enumerate(view_names):
        cubemap_store[idx] = convert_cubemap(cubemap_vis[idx], view_name)

    pose_matrix = camera_to_world
    w = width
    h = height
    fy = h / (2 * np.tan(FoVy / 2))
    fx = w / (2 * np.tan(FoVx / 2))
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([fx, fy, cx, cy, w, h])

    if isinstance(cubemap_store, torch.Tensor):
        cubemap_store = cubemap_store
    elif isinstance(cubemap_store, np.ndarray):
        cubemap_store = torch.tensor(cubemap_store).float().cuda()
    else:
        raise NotImplementedError("skybox should be torch.Tensor or np.ndarray")

    sampled_color = sample_cubemap_from_camera(
        pose_matrix, intrinsic, cubemap_store
    )  # [h, w, 3]

    if isinstance(skybox, torch.Tensor):
        return sampled_color
    else:
        return sampled_color.cpu().numpy()


def build_pose_and_intrinsic_tensor(camera_to_world, height, width, FoVy, FoVx):
    """
    Args:
        camera_to_world: np.ndarray, shape [4, 4]
        height: int
        width: int
        FoVy: float in radians
        FoVx: float in radians

    Returns:
        pose: torch.Tensor shape [4, 4]
        intrinsic: torch.Tensor shape [6, ]
    """
    pose_matrix = camera_to_world
    # shape [6, ], fx fy cx cy w h
    # calcuate fx, fy, cx, cy from fov and aspect
    w = width
    h = height
    fy = h / (2 * np.tan(FoVy / 2))
    fx = w / (2 * np.tan(FoVx / 2))
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([fx, fy, cx, cy, w, h])

    return torch.tensor(pose_matrix).float(), torch.tensor(intrinsic).float()


def read_skybox(gaussian_model_path, white_bg=False):
    """
    read the skybox representation along with the 3d gaussians.
    Also return the rendering API for each skybox type.
    """
    gaussian_model_path = Path(gaussian_model_path)
    gaussian_model_stem = gaussian_model_path.with_suffix("").as_posix()

    # (1) if 360 / 180 RGB panorama
    rgb_panorama_path = Path(gaussian_model_stem + "_pano.png")

    # (2) if sky modulator
    modulator_path = Path(gaussian_model_stem + "_modulator.yaml")
    modulator_weight = Path(gaussian_model_stem + "_modulator.pt")
    sky_token_path = Path(gaussian_model_stem + "_sky_token.pt")

    if white_bg:
        skybox_dict = {
            "type": "rgb_panorama",
            "panorama": np.ones((64, 128, 3), dtype=np.float32),
            "panorama_mask": np.zeros((64, 128), dtype=np.float32),
        }

        return skybox_dict

    # (1) if 360 / 180 RGB panorama
    if rgb_panorama_path.exists():
        panorama_path = rgb_panorama_path
        panorama_mask_path = Path(gaussian_model_stem + "_pano_mask.png")

        panorama = imageio.imread(panorama_path.as_posix()) / 255.0

        try:
            panorama_mask = imageio.imread(panorama_mask_path.as_posix()) / 255.0
        except FileNotFoundError:
            panorama_mask = np.zeros(
                (panorama.shape[0], panorama.shape[1]), dtype=np.float32
            )

        # resize the mask if not the same size
        if (
            panorama_mask.shape[0] != panorama.shape[0]
            or panorama_mask.shape[1] != panorama.shape[1]
        ):
            panorama_mask = np.array(
                Image.fromarray(panorama_mask).resize(
                    (panorama.shape[1], panorama.shape[0]), Image.Resampling.BILINEAR
                )
            )

        skybox_dict = {
            "panorama": panorama,
            "panorama_mask": panorama_mask,
            "type": "rgb_panorama",
        }

    # (4) if sky modulator
    elif (
        modulator_path.exists()
        and modulator_weight.exists()
        and sky_token_path.exists()
    ):
        from omegaconf import OmegaConf

        from infinicube.voxelgen.modules.sky_modules import SkyboxMlpModulator

        modulator_config = OmegaConf.load(modulator_path.as_posix())
        modulator_weight = torch.load(modulator_weight, map_location="cpu")
        sky_token = torch.load(sky_token_path)

        modulator = SkyboxMlpModulator(modulator_config)
        modulator.load_state_dict(modulator_weight)
        modulator.eval()
        modulator = modulator.cuda()
        sky_token = sky_token.cuda()

        skybox_dict = {
            "type": "sky_modulator",
            "modulator": modulator,
            "sky_token": sky_token,
        }

    else:
        print(colored("No skybox representation found. Leave it black!", "red"))
        skybox_dict = {
            "type": "rgb_panorama",
            "panorama": np.zeros((64, 128, 3), dtype=np.float32),
            "panorama_mask": np.zeros((64, 128), dtype=np.float32),
        }

    return skybox_dict


def render_sky_api(skybox_dict, camera_to_world, height, width, vfov, hfov):
    """
    Define the rendering API for each skybox type.

    Args:
        skybox_dict: dict, skybox representation.
        camera_to_world: np.ndarray, [4, 4], camera pose
        height: int, height of the rendering image.
        width: int, width of the rendering image.
        vfov: float, vertical field of view in radians.
        hfov: float, horizontal field of view in radians.

    Returns:
        np.ndarray, [H, W, 3], rendered skybox image.
    """
    apply_skybox_mask = skybox_dict.get("apply_skybox_mask", False)

    if skybox_dict["type"] == "rgb_panorama":
        if apply_skybox_mask:
            panorama = skybox_dict["panorama"] * (
                skybox_dict["panorama_mask"][:, :, None] > 0
            )
        else:
            panorama = skybox_dict["panorama"]

        if panorama.shape[0] * 4 == panorama.shape[1]:
            return render_sky_panorama_hemi(
                panorama, camera_to_world, height, width, vfov, hfov
            )
        elif panorama.shape[0] * 2 == panorama.shape[1]:
            return render_sky_panorama_full(
                panorama, camera_to_world, height, width, vfov, hfov
            )
        else:
            raise ValueError(colored("Panorama shape is not supported.", "red"))

    elif skybox_dict["type"] == "rgb_cubemap":
        return render_sky_cubemap(
            skybox_dict["cubemap"], camera_to_world, height, width, vfov, hfov
        )

    elif skybox_dict["type"] == "sky_modulator":
        camera_pose, intrinsic = build_pose_and_intrinsic_tensor(
            camera_to_world, height, width, vfov, hfov
        )
        camera_pose_stack = camera_pose.unsqueeze(0).cuda()
        intrinsics = intrinsic.unsqueeze(0).cuda()
        network_output = {"skybox_representation": skybox_dict["sky_token"]}
        with torch.no_grad():
            rendered_rgb = skybox_dict["modulator"].sample_batch(
                camera_pose_stack,
                intrinsics,
                network_output,
                batch_idx=0,
            )
        # [N, H, W ,3] -> [H, W, 3]
        rendered_rgb = rendered_rgb.squeeze(0).cpu().numpy().clip(0, 1)
        return rendered_rgb

    else:
        raise NotImplementedError
