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

import numpy as np
import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(
        self,
        include_input=True,
        input_dims=3,
        max_freq_log2=10,
        num_freqs=10,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    ):
        super().__init__()
        embed_fns = []
        d = input_dims
        out_dim = 0
        if include_input:
            out_dim += d

        max_freq = max_freq_log2
        N_freqs = num_freqs

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for _ in periodic_fns:
                out_dim += d

        self.include_input = include_input
        self.freq_bands = freq_bands
        self.periodic_fns = periodic_fns

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        output_list = [inputs]
        for fn in self.embed_fns:
            output_list.append(fn(inputs))

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                output_list.append(p_fn(inputs * freq))

        return torch.cat(output_list, -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embedder_obj = Embedder(
        max_freq_log2=multires - 1, num_freqs=multires, input_dims=input_dims
    )
    return embedder_obj, embedder_obj.out_dim


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                   Sine/Cosine Positional Embedding Functions (NeRF)           #
#################################################################################
# https://github.com/nerfstudio-project/nerfstudio/blob/fc4fc5cb15ad994ea82d8c651c9d42172d890de1/nerfstudio/field_components/encodings.py#L35

from abc import abstractmethod
from typing import Literal, Optional

from jaxtyping import Float, Shaped
from torch import Tensor, nn


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


def components_from_spherical_harmonics(
    levels: int, directions: Float[Tensor, "*batch 3"]
) -> Float[Tensor, "*batch components"]:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels**2
    components = torch.zeros(
        (*directions.shape[:-1], num_components), device=directions.device
    )

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, (
        f"Direction input should have three dimensions. Got {directions.shape[-1]}"
    )

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (
            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
        )

    return components


class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(
        self, in_dim: Optional[int] = None, out_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between -1 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = torch.pi * (in_tensor + 1)  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device
        )
        scaled_inputs = (
            scaled_in_tensor[..., None] * freqs
        )  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(
            *scaled_inputs.shape[:-2], -1
        )  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
            )
        else:
            input_var = (
                torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None]
                * freqs[None, :] ** 2
            )
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
                torch.cat(2 * [input_var], dim=-1),
            )
        return encoded_inputs

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        encoded_inputs = self.pytorch_fwd(in_tensor, covs)

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(
        self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch"
    ) -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(
                f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}"
            )

        self.levels = levels

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def pytorch_fwd(
        self, in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return components_from_spherical_harmonics(
            levels=self.levels, directions=in_tensor
        )

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        """
        in_tensor: Input normalized xyz coordinate of view direction
        """
        return self.pytorch_fwd(in_tensor)
