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

from typing import List, Optional, Union

import numpy as np
import torch
from fvdb import GridBatch
from gsplat import rasterization
from pycg.isometry import Isometry
from torch.cuda.amp import autocast


class IsoTransform:
    """An isometry transformation backended in PyTorch"""

    def __init__(self, device: Union[str, torch.device]) -> None:
        self.rotation = torch.eye(3, device=device)
        self.translation = torch.zeros(3, device=device)

    @property
    def matrix(self):
        upper = torch.cat([self.rotation, self.translation[:, None]], dim=1)
        lower = torch.tensor([0, 0, 0, 1], device=self.rotation.device)
        return torch.cat([upper, lower[None, :]], dim=0)

    @property
    def inverse(self):
        inv_rot = self.rotation.transpose(0, 1)
        inv_tran = -inv_rot @ self.translation
        res = IsoTransform(self.rotation.device)
        res.rotation = inv_rot
        res.translation = inv_tran
        return res

    @property
    def device(self):
        return self.rotation.device

    def set_from_numpy(self, matrix: np.ndarray):
        device = self.device
        self.rotation = torch.from_numpy(matrix[:3, :3]).to(device).float()
        self.translation = torch.from_numpy(matrix[:3, 3]).to(device).float()

    def set_from_torch(self, matrix: torch.Tensor):
        self.rotation = matrix[:3, :3].clone().float()
        self.translation = matrix[:3, 3].clone().float()

    def set_from_pycg(self, iso: Isometry):
        self.set_from_numpy(iso.matrix)


class GsplatPinholeCamera:
    def __init__(
        self,
        device: Union[str, torch.device],
        h: int = 800,
        w: int = 600,
        fx: float = 800,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> None:
        self.pose = IsoTransform(device)
        self.h = h
        self.w = w
        self.fx = fx
        self.fy = fy if fy is not None else fx
        self.cx = cx if cx is not None else w / 2
        self.cy = cy if cy is not None else h / 2
        self.znear = 0.001
        self.zfar = 10.0

    def to(self, device: Union[str, torch.device]):
        self.pose.rotation = self.pose.rotation.to(device)
        self.pose.translation = self.pose.translation.to(device)
        return self

    @property
    def intrinsic_matrix(self):
        K = torch.zeros(3, 3, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1
        return K

    @property
    def intrinsic_list(self):
        return [self.fx, self.fy, self.cx, self.cy, self.w, self.h]

    @property
    def device(self):
        return self.pose.device

    @property
    def camera_center(self):
        return self.pose.translation

    @property
    def world_view_transform(self):
        return self.pose.inverse.matrix.transpose(0, 1)

    @property
    def projection_matrix_naive(self):
        tan_half_fov_x = self.w / (2 * self.fx)
        tan_half_fov_y = self.h / (2 * self.fy)

        top = tan_half_fov_y * self.znear
        bottom = -top
        right = tan_half_fov_x * self.znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)  # 2 * fx / w
        P[1, 1] = 2.0 * self.znear / (top - bottom)  # 2 * fy / h
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)

        return P.to(self.device).transpose(0, 1)

    @property
    def projection_matrix(self):
        """
        we will use fx fy cx cy w h to calculate the projection matrix.
        check projection_matrix == projection_matrix_naive when cx = w/2 and cy = h/2
        """
        z_sign = 1.0

        P = torch.zeros(4, 4)
        P[0, 0] = 2 * self.fx / self.w
        P[1, 1] = 2 * self.fy / self.h
        P[0, 2] = 2 * self.cx / self.w - 1
        P[1, 2] = 2 * self.cy / self.h - 1
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P.to(self.device).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return self.world_view_transform @ self.projection_matrix


def render_gsplat_backend_batch(
    viewpoint_cameras: List[GsplatPinholeCamera],
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    active_sh_degree: int,
    bg_color,
    scaling_modifier=1.0,
    override_color: Optional[torch.Tensor] = None,
    **kwargs,
):
    # assert cameras have the same resolution
    assert all(
        [
            viewpoint_camera.w == viewpoint_cameras[0].w
            and viewpoint_camera.h == viewpoint_cameras[0].h
            for viewpoint_camera in viewpoint_cameras
        ]
    )

    Ks = [viewpoint_camera.intrinsic_matrix for viewpoint_camera in viewpoint_cameras]
    Ks = torch.stack(Ks)

    viewmats = [
        viewpoint_camera.world_view_transform.transpose(0, 1)
        for viewpoint_camera in viewpoint_cameras
    ]
    viewmats = torch.stack(viewmats)

    means3D = xyz
    opacity = opacity
    scales = scaling
    rotations = rotation
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = color
        sh_degree = active_sh_degree

    if "force_float32" in kwargs and kwargs["force_float32"]:
        with autocast(enabled=False):
            render_colors, render_alphas, info = rasterization(
                means=means3D.float(),  # [N, 3]
                quats=rotations.float(),  # [N, 4]
                scales=scales.float(),  # [N, 3]
                opacities=opacity.squeeze(-1).float(),  # [N,]
                colors=colors.float(),  # [N, C]
                viewmats=viewmats.float(),  # [N_view, 4, 4]
                Ks=Ks.float(),  # [N_view, 3, 3]
                backgrounds=bg_color,
                width=int(viewpoint_cameras[0].w),
                height=int(viewpoint_cameras[0].h),
                packed=False,
                sh_degree=sh_degree,
                radius_clip=kwargs.get("radius_clip", 0.0),
                render_mode="RGB+ED",  # expected depth.
                rasterize_mode=kwargs.get("rasterize_mode", "classic"),
                absgrad=kwargs.get("absgrad", False),
            )
    else:
        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,  # [N, C]
            viewmats=viewmats,  # [N_view, 4, 4]
            Ks=Ks,  # [N_view, 3, 3]
            backgrounds=bg_color,
            width=int(viewpoint_cameras[0].w),
            height=int(viewpoint_cameras[0].h),
            packed=False,
            sh_degree=sh_degree,
            radius_clip=kwargs.get("radius_clip", 0.0),
            render_mode="RGB+ED",  # expected depth.
            rasterize_mode=kwargs.get("rasterize_mode", "classic"),
            absgrad=kwargs.get("absgrad", False),
        )

    # [N_view, H, W, 4] -> [N_view, H, W, 3] and [N_view, H, W, 1]
    rendered_image = render_colors[..., :-1]  # [N_view, H, W, 3]
    rendered_depth = render_colors[..., -1:]  # [N_view, H, W, 1]
    rendered_alpha = render_alphas  # [N_view, H, W, 1]
    radii = info["radii"]

    try:
        info["means2d"].retain_grad()  # [N_views, N, 2]
    except:
        pass

    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
    }


def render_gsplat_func(
    camera: List[GsplatPinholeCamera],
    grid: GridBatch,
    _rel_xyz,
    _scaling,
    _rots,
    _opacities,
    _color,
    bg=None,
    free_space="hard",
    **kwargs,
):
    """
    Returns
    -------
    image : torch.tensor
        [N, H, W, 3]
    depth : torch.tensor
        [N, H, W, 1]
    alpha : torch.tensor
        [N, H, W, 1]

    N = len(camera)
    """
    base_pos = _rel_xyz
    scaling = _scaling

    rotation = _rots
    opacity = _opacities

    color = _color

    render = render_gsplat_backend_batch

    render_pkg = render(
        camera,
        base_pos,
        scaling,
        rotation,
        opacity,
        None,
        0,
        bg,
        scaling_modifier=1.0,
        override_color=color,
        **kwargs,
    )

    # [N, H, W, 3], [N, H, W, 1], [N, H, W, 1]
    image, depth, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["alpha"]

    return image, depth, alpha
