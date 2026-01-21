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

from collections import OrderedDict

import numpy as np
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def transform2tensor(x: OrderedDict):
    for key, value in x.items():
        if isinstance(value, np.ndarray):
            x[key] = torch.from_numpy(value).float()
            if torch.cuda.is_available():
                x[key] = x[key].cuda()
    return x


def rasterization_inria_backend(
    gaussians_tensors,
    image_height,
    image_width,
    tanfovx,
    tanfovy,
    bg_tensor,
    scale_modifier,
    world_view_transform,
    full_proj_transform,
    camera_center,
    render_alpha,
):
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height),
        image_width=int(image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_tensor,
        scale_modifier=scale_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=gaussians_tensors["sh_degree"],
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = gaussians_tensors["xyz"]
    means2D = torch.zeros_like(
        means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
    )
    opacity = gaussians_tensors["opacity"]
    scales = gaussians_tensors["scaling"]
    rotations = gaussians_tensors["rotation"]
    shs = gaussians_tensors["features"]

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    if render_alpha:
        rendered_alpha, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones_like(means3D),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        return {"image": rendered_image, "alpha": rendered_alpha}

    return {"image": rendered_image}


def rasterization_gsplat_backend(
    gaussians_tensors,
    image_height,
    image_width,
    tanfovx,
    tanfovy,
    scale_modifier,
    world_view_transform,
    render_alpha,
):
    from gsplat import rasterization

    focal_length_x = image_width / (2 * tanfovx)
    focal_length_y = image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, image_width / 2.0],
            [0, focal_length_y, image_height / 2.0],
            [0, 0, 1],
        ],
    ).to(gaussians_tensors["xyz"])

    means3D = gaussians_tensors["xyz"]
    opacity = gaussians_tensors["opacity"]
    scales = gaussians_tensors["scaling"] * scale_modifier
    rotations = gaussians_tensors["rotation"]
    shs = gaussians_tensors["features"]
    sh_degree = gaussians_tensors["sh_degree"]

    viewmat = torch.tensor(world_view_transform).to(gaussians_tensors["xyz"])

    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=None,
        width=int(image_width),
        height=int(image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode="RGB",
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    rendered_alpha = render_alphas[0].permute(2, 0, 1)

    if render_alpha:
        return {"image": rendered_image, "alpha": rendered_alpha}

    return {"image": rendered_image}


def standard_3dgs_rendering_func(
    camera_to_world,
    height,
    width,
    vfov,
    hfov,
    gaussians_tensors,
    scale_modifier=1,
    skybox_dict=None,
):
    """
    Args:
        camera_to_world: np.ndarray
            4x4 matrix, camera to world transformation

        height: int
            height of the image

        width: int
            width of the image

        vfov: float
            vertical field of view, radians

        hfov: float
            horizontal field of view, radians. Take care of this one.
            If real-time renderer, use camera.aspect to get aspect ratio, and get hfov from vfov & aspect ratio
            If recording, use width/height to calculate aspect ratio, and get hfov from vfov & aspect ratio. (we make fx=fy)

        gaussians_tensors: dict
            dictionary storing cuda tensor of gaussians

        scale_modifier: float
            scale modifier for the gaussian splatting

    """
    from infinicube.utils.sky_utils import render_sky_api

    world_to_camera = np.linalg.inv(camera_to_world)
    FoVy = vfov
    FoVx = hfov
    render_alpha = True

    # render fg (foreground, 3d gaussians)
    with torch.no_grad():
        # [3, H, W], range 0-1
        rendered_output = rasterization_gsplat_backend(
            gaussians_tensors,
            height,
            width,
            np.tan(FoVx / 2),
            np.tan(FoVy / 2),
            scale_modifier,
            world_to_camera,
            render_alpha,
        )

    fg_image = (
        rendered_output["image"].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    )  # [H, W, 3]
    fg_alpha = (
        torch.mean(rendered_output["alpha"], dim=0, keepdim=True)
        .clamp(0, 1)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )  # [H, W, 1]

    # render bg (skybox, depends on its representation)
    if skybox_dict is not None:
        sky_color = render_sky_api(
            skybox_dict, camera_to_world, height, width, vfov, hfov
        )
        full_image = fg_image + sky_color * (1 - fg_alpha)
    else:
        full_image = fg_image

    # clamp and to uint8
    full_image = (full_image * 255).clip(0, 255).astype(np.uint8)

    return full_image


def client_rendering_and_set_background(
    client,
    rendering_func,
    rendering_kwargs,
):
    """
    Args:
        vfov: read from slider (gui_fovy_modifier)
        scale_modifier: read from slider (gui_scale_modifier)
    """
    from infinicube.utils.viser_gui_utils import build_camera_to_world

    # provide camera_to_world to rendering_func
    camera_to_world = build_camera_to_world(client.camera)
    rendering_kwargs["camera_to_world"] = camera_to_world

    # client rendering, we get hfov from client.camera.aspect instead of width/height
    client.camera.fov = rendering_kwargs["vfov"]  # first set vfov to client
    aspect_ratio = client.camera.aspect
    vfov = client.camera.fov
    hfov = 2 * np.arctan(np.tan(vfov / 2) * aspect_ratio)
    rendering_kwargs["hfov"] = hfov

    full_image = rendering_func(**rendering_kwargs)

    client.set_background_image(full_image)
