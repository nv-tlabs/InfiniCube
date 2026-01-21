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

import pickle
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm
from viser import transforms as tf

from infinicube.utils.gaussian_render_utils import RGB2SH


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_gaussian(path, max_sh_degree=3):
    if path.endswith(".ply") or path.endswith(".PLY"):
        return load_gaussian_ply(path, max_sh_degree)
    elif path.endswith(".splat"):
        return load_gaussian_splat(path)
    elif path.endswith(".pkl"):
        return load_gaussian_pkl(path)


def load_gaussian_ply(path, max_sh_degree=3):
    """
    https://github.com/graphdeco-inria/gaussian-splatting/blob/b17ded92b56ba02b6b7eaba2e66a2b0510f27764/scene/gaussian_model.py#L215

    Note: In the PLY file, the attributes are as below. We will read these attributes, process them and return them as a dictionary.
        x (N,)   # position
        y (N,)
        z (N,)
        nx (N,)  # normal, all zeros
        ny (N,)
        nz (N,)
        f_dc_0 (N,)   # SH coefficients at degree 0, Red
        f_dc_1 (N,)   # SH coefficients at degree 0, Green
        f_dc_2 (N,)   # SH coefficients at degree 0, Blue
        f_rest_0 (N,)    # rest of SH coefficients
        ...
        f_rest_44 (N,)   # rest of SH coefficients
        opacity (N,)
        scale_0 (N,)
        scale_1 (N,)
        scale_2 (N,)
        rot_0 (N,)
        rot_1 (N,)
        rot_2 (N,)
        rot_3 (N,)

    Returns:
        gaussians: OrderedDict
        _xyz: (N, 3) float32
            xyz coordinates

        _features_dc: (N, 1, 3) float32
            Direct color of SH coefficient.

        _features_rest: (N, (max_sh_degree + 1) ** 2 - 1, 3) float32
            Rest of SH coefficients.
            Note the final color will be:
                shs_view = torch.cat((features_dc, features_rest), dim=1).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        _opacity: (N, 1) float32
            positive or negative. need sigmoid actiation function to convert to 0-1 to get real opacity

        _scaling: (N, 3) float32
            positive or negative scaling factor for each axis, need exp activation function to convert to real scales

        _rotation: (N, 4) float32
            quaternion for rotation, need normalize activation function to convert to unit quaternion


        active_sh_degree: int
            usually 3

    """
    gaussians = OrderedDict()
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gaussians["_xyz"] = xyz
    gaussians["_features_dc"] = features_dc.transpose([0, 2, 1])
    gaussians["_features_rest"] = features_extra.transpose([0, 2, 1])
    gaussians["_opacity"] = opacities
    gaussians["_scaling"] = scales
    gaussians["_rotation"] = rots
    gaussians["sh_degree"] = max_sh_degree

    # explicit interface
    gaussians["xyz"] = xyz
    gaussians["opacity"] = sigmoid(gaussians["_opacity"])
    gaussians["scaling"] = np.exp(scales)
    gaussians["rotation"] = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    gaussians["features"] = np.concatenate(
        [gaussians["_features_dc"], gaussians["_features_rest"]], axis=1
    )  # (N, (max_sh_degree + 1) ** 2, 3)

    gaussians["ply_format"] = True

    return gaussians


def load_gaussian_splat(path):
    """
    Read .splat gaussian splatting file, which has compressed size and no view-dependent effect (SH degree=0)
    https://github.com/nerfstudio-project/viser/blob/brent%2Fsplatting/examples/21_gaussian_splats.py
    """
    path = Path(path)
    splat_buffer = path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian
    print(f"{num_gaussians=}")

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = np.array([tf.SO3(wxyz).as_matrix() for wxyz in wxyzs])
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )

    gaussians = OrderedDict()

    # explicit interface
    gaussians["xyz"] = splat_uint8[:, 0:12].copy().view(np.float32)  # (N, 3)
    gaussians["opacity"] = splat_uint8[:, 27:28] / 255.0  # (N, 1)
    gaussians["scaling"] = scales  # consistent with .ply file.
    gaussians["rotation"] = wxyzs  # consistent with .ply file.
    # only diffuse color, no SH.
    # to align with .ply file format, we contruct the degree 0 SH coefficients.
    gaussians["rgbs"] = splat_uint8[:, 24:27] / 255.0  # (N, 3)
    gaussians["features"] = RGB2SH(gaussians["rgbs"]).reshape(-1, 1, 3)  # (N, 1, 3)
    gaussians["sh_degree"] = 0

    gaussians["ply_format"] = False

    return gaussians


def load_gaussian_pkl(path):
    """
    Read .pkl gaussian splatting file, customized by Xuanchi
    It is a dict, with keys: xyz, opacity, scaling, rotation, rgbs

    Note that the sh_degree is specified to be 0
    """
    with open(path, "rb") as f:
        gaussians = pickle.load(f)

    gaussians["rgbs"] = np.clip(gaussians["rgbs"], 0, 1)

    if "features" not in gaussians:
        gaussians["features"] = RGB2SH(gaussians["rgbs"]).reshape(-1, 1, 3)  # (N, 1, 3)
        gaussians["sh_degree"] = 0

    gaussians["ply_format"] = False

    return gaussians


def process_gaussian_params_to_splat(xyz, scaling, rotation, opacity, color):
    """
    xyz, scaling, rotation, opacity, color are all explicit parameters of the gaussian.

    xyz: (N, 3) float32
    scaling: (N, 3) float32
    rotation: (N, 4) float32
    opacity: (N,) or (N, 1) float32, range [0, 1]
    color: (N, 3) float32, range [0, 1]
    """
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        opacity = opacity.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

    # sorted_indices = np.argsort(-scaling[:,0] * scaling[:,1] * scaling[:,2] * opacity)
    sorted_indices = np.arange(xyz.shape[0])  # no need to sort

    xyz = xyz.astype(np.float32)
    rotation = rotation.astype(np.float32)
    scaling = scaling.astype(np.float32)
    opacity = opacity.astype(np.float32)
    color = color.astype(np.float32)

    if len(opacity.shape) == 2:
        opacity = opacity[:, 0]

    buffer = BytesIO()
    for idx in tqdm(sorted_indices):
        position = xyz[idx]
        scales = scaling[idx]
        rot = rotation[idx]
        rgba = np.array(
            [
                color[idx][0],
                color[idx][1],
                color[idx][2],
                opacity[idx],
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((rgba * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    return buffer.getvalue()


def process_gaussian_params_to_dict(xyz, scaling, rotation, opacity, color):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        opacity = opacity.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

    gaussians = OrderedDict()
    gaussians["xyz"] = xyz
    gaussians["opacity"] = opacity
    gaussians["scaling"] = scaling
    gaussians["rotation"] = rotation
    gaussians["rgbs"] = color

    return gaussians


def _save_splat_file(xyz, scaling, rotation, opacity, color, output_path):
    if output_path.endswith(".splat"):
        splat_data = process_gaussian_params_to_splat(
            xyz, scaling, rotation, opacity, color
        )
        with open(output_path, "wb") as f:
            f.write(splat_data)

    elif output_path.endswith(".pkl"):
        gaussians = process_gaussian_params_to_dict(
            xyz, scaling, rotation, opacity, color
        )
        with open(output_path, "wb") as f:
            pickle.dump(gaussians, f)


def save_splat_file(gaussian_params, output_path):
    _save_splat_file(*gaussian_params.split([3, 3, 4, 1, 3], dim=-1), output_path)
