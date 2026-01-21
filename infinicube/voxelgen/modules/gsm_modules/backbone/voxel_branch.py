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

import math

import fvdb
import fvdb.nn as fvnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from fvdb import JaggedTensor
from fvdb.nn import VDBTensor
from loguru import logger
from torch_scatter import scatter_mean

from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.utils.render_util import (
    camera_intrinsic_list_to_matrix,
    create_rays_from_intrinsic_torch_batch,
    get_rel_pos,
)
from infinicube.voxelgen.utils.voxel_util import get_occ_front_voxel, project_points


class depth_wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return self.module(*args), 0


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str,
        num_groups: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        for i, char in enumerate(order):
            if char == "r":
                self.add_module("ReLU", fvnn.ReLU(inplace=True))
            elif char == "s":
                self.add_module("SiLU", fvnn.SiLU(inplace=True))
            elif char == "c":
                self.add_module(
                    "Conv",
                    fvnn.SparseConv3d(
                        in_channels, out_channels, kernel_size, 1, bias="g" not in order
                    ),
                )
            elif char == "g":
                num_channels = in_channels if i < order.index("c") else out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module(
                    "GroupNorm",
                    fvnn.GroupNorm(
                        num_groups=num_groups, num_channels=num_channels, affine=True
                    ),
                )
            else:
                raise NotImplementedError


class SparseHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, num_groups, enhanced="None"):
        super().__init__()
        self.add_module(
            "SingleConv", ConvBlock(in_channels, in_channels, order, num_groups)
        )
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        if enhanced == "three":
            self.add_module("OutConv-1", fvnn.Linear(in_channels, mid_channels))
            self.add_module("ReLU-1", fvnn.LeakyReLU(inplace=True))
            self.add_module("OutConv-2", fvnn.Linear(mid_channels, mid_channels))
            self.add_module("ReLU-2", fvnn.LeakyReLU(inplace=True))
            self.add_module(
                "OutConv", fvnn.Linear(mid_channels, out_channels)
            )  # !: final linear keep name consistent
        elif enhanced == "upsample":
            self.add_module("upsample", fvnn.UpsamplingNearest(2))  # !: upsample
            self.add_module(
                "OutConv-1",
                SparseResBlock(
                    in_channels,  # ! add back skip connection
                    mid_channels,
                    order,
                    num_groups,
                    False,
                    None,
                    return_feat_depth=False,
                ),
            )
            self.add_module(
                "OutConv", fvnn.Linear(mid_channels, out_channels)
            )  # !: final linear keep name consistent
        else:
            self.add_module(
                "OutConv", fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True)
            )


class LinearHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, num_groups, enhanced="None"):
        super().__init__()
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        if enhanced == "three":
            self.add_module("OutConv-1", fvnn.Linear(in_channels, mid_channels))
            self.add_module("ReLU-1", fvnn.LeakyReLU(inplace=True))
            self.add_module("OutConv-2", fvnn.Linear(mid_channels, mid_channels))
            self.add_module("ReLU-2", fvnn.LeakyReLU(inplace=True))
            self.add_module(
                "OutConv", fvnn.Linear(mid_channels, out_channels)
            )  # !: final linear keep name consistent
        elif enhanced == "original":
            self.add_module(
                "SingleConv", ConvBlock(in_channels, in_channels, order, num_groups)
            )
            self.add_module(
                "OutConv", fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True)
            )


class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str,
        num_groups: int,
        encoder: bool,
        pooling=None,
        use_checkpoint: bool = False,
        return_feat_depth: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.return_feat_depth = return_feat_depth

        self.use_pooling = pooling is not None and encoder

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == "max":
                self.maxpooling = fvnn.MaxPool(2)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups)
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups)

        if conv1_in_channels != conv2_out_channels:
            self.skip_connection = fvnn.SparseConv3d(
                conv1_in_channels, conv2_out_channels, 1, 1
            )
        else:
            self.skip_connection = nn.Identity()

    def _forward(self, input, hash_tree=None, feat_depth: int = 0):
        if self.use_pooling:
            if hash_tree is not None:
                feat_depth += 1
                input = self.maxpooling(input, hash_tree[feat_depth])
            else:
                input = self.maxpooling(input)

        h = input
        h = self.conv1(h)
        h = self.conv2(h)
        input = self.skip_connection(input)

        return h + input, feat_depth

    def forward(self, input, hash_tree=None, feat_depth: int = 0):
        if self.use_checkpoint:
            # !: we need to set use_reentrant = False
            input, feat_depth = checkpoint.checkpoint(
                self._forward, input, hash_tree, feat_depth, use_reentrant=False
            )
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)

        if self.return_feat_depth:
            return input, feat_depth
        else:
            return input


class SparseDoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str,
        num_groups: int,
        encoder: bool,
        pooling=None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == "max":
                self.add_module("MaxPool", fvnn.MaxPool(2))
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module(
            "SingleConv1",
            ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups),
        )
        self.add_module(
            "SingleConv2",
            ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups),
        )

    def _forward(self, input, hash_tree=None, feat_depth: int = 0):
        for module in self:
            if module._get_name() == "MaxPool" and hash_tree is not None:
                feat_depth += 1
                input = module(input, hash_tree[feat_depth])
            else:
                input = module(input)
        return input, feat_depth

    def forward(self, input, hash_tree=None, feat_depth: int = 0):
        if self.use_checkpoint:
            # !: we need to set use_reentrant = False
            input, feat_depth = checkpoint.checkpoint(
                self._forward, input, hash_tree, feat_depth, use_reentrant=False
            )
        else:
            input, feat_depth = self._forward(input, hash_tree, feat_depth)
        return input, feat_depth


class AttentionBlock(nn.Module):
    """
    A for loop version with flash attention
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = fvnn.Linear(channels, channels)

    def _attention(self, qkv: torch.Tensor):
        # conduct attention for each batch
        length, width = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3)  # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1)  # (1, num_heads, length, ch)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[
                0
            ]  # (1, num_heads, length, ch)
        values = values.permute(1, 0, 2)  # (length, num_heads, ch)
        values = values.reshape(length, -1)
        return values

    def attention(self, qkv: VDBTensor):
        values = []
        for batch_idx in range(qkv.grid.grid_count):
            values.append(self._attention(qkv.data[batch_idx].jdata))
        return fvdb.JaggedTensor(values)

    def forward(self, x: VDBTensor):
        return self._forward(x), None  # !: return None for feat_depth

    def _forward(self, x: VDBTensor):
        qkv = self.qkv(self.norm(x))
        feature = self.attention(qkv)
        feature = VDBTensor(x.grid, feature, x.kmap)
        feature = self.proj_out(feature)
        return feature + x


class Pure3DUnet(nn.Module):
    def __init__(
        self,
        lifter_params,
        img_feature_source,
        in_channels,
        num_blocks,
        f_maps=64,
        order="gcs",
        num_groups=8,
        neck_dense_type="UNCHANGED",
        neck_bound=4,
        with_render_branch=True,
        gsplat_upsample=1,
        gs_enhanced="None",
        use_attention=False,
        use_residual=True,
        apply_gs_init: bool = True,
        addtional_gs_constraint="None",
        use_checkpoint=False,
        gs_init_scale=0.5,
        gs_dim=14,
        # ! joint 2D and 3D
        f_maps_2d=32,
        feature_pooling_2d="max",
        gs_free_space="hard",
        max_return=1,
        drop_invisible=False,
        occ_upsample=2,
        max_scaling=0.0,
        **kwargs,
    ):
        super().__init__()
        self.lifter = Lifter(**lifter_params)
        self.img_feature_source = img_feature_source

        n_features = [in_channels] + [f_maps * 2**k for k in range(num_blocks)]
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        self.pre_kl_bottleneck = nn.ModuleList()
        self.post_kl_bottleneck = nn.ModuleList()

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.num_blocks = num_blocks

        if not use_residual:
            basic_block = SparseDoubleConv
        else:
            basic_block = SparseResBlock

        # Attention setup
        self.use_attention = use_attention

        # Encoder
        self.pre_conv = fvnn.SparseConv3d(
            in_channels, in_channels, 1, 1
        )  # a MLP to smooth the input
        for layer_idx in range(num_blocks):
            self.encoders.add_module(
                f"Enc{layer_idx}",
                basic_block(
                    n_features[layer_idx],
                    n_features[layer_idx + 1],
                    order,
                    num_groups,
                    True,  # if encoder branch
                    None,
                    use_checkpoint,
                ),
            )
        for layer_idx in range(1, num_blocks):
            self.downsamplers.add_module(f"Down{layer_idx}", fvnn.MaxPool(2))

        # Bottleneck
        self.pre_kl_bottleneck.add_module(
            "pre_kl_bottleneck_0",
            basic_block(
                n_features[-1],
                n_features[-1],
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )
        if use_attention:
            self.pre_kl_bottleneck.add_module(
                "pre_kl_attention",
                AttentionBlock(n_features[-1], use_checkpoint=use_checkpoint),
            )
            # ! bug here -> forget to move this out
            self.pre_kl_bottleneck.add_module(
                "pre_kl_bottleneck_1",
                basic_block(
                    n_features[-1],
                    n_features[-1],
                    order,
                    num_groups,
                    False,
                    use_checkpoint=use_checkpoint,
                ),
            )

        self.post_kl_bottleneck.add_module(
            "post_kl_bottleneck_0",
            basic_block(
                n_features[-1],
                n_features[-1],
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )
        if use_attention:
            self.post_kl_bottleneck.add_module(
                "post_kl_attention",
                AttentionBlock(n_features[-1], use_checkpoint=use_checkpoint),
            )
        self.post_kl_bottleneck.add_module(
            "post_kl_bottleneck_1",
            basic_block(
                n_features[-1],
                n_features[-1],
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )

        # Decoder
        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_convs.add_module(
                f"Struct{layer_idx}",
                SparseHead(n_features[layer_idx], 2, order, num_groups),
            )
            if layer_idx < -1:
                self.decoders.add_module(
                    f"Dec{layer_idx}",
                    basic_block(
                        n_features[layer_idx + 1]
                        + n_features[layer_idx],  # ! add back skip connection
                        n_features[layer_idx],
                        order,
                        num_groups,
                        False,
                        None,
                        use_checkpoint=use_checkpoint,
                    ),
                )
                self.upsamplers.add_module(f"Up{layer_idx}", fvnn.UpsamplingNearest(2))
        self.up_sample0 = fvnn.UpsamplingNearest(1)

        # check the type of neck_bound
        if isinstance(neck_bound, int):
            self.low_bound = [-neck_bound] * 3
            self.voxel_bound = [neck_bound * 2] * 3
        else:
            self.low_bound = [-res for res in neck_bound]
            self.voxel_bound = [res * 2 for res in neck_bound]
        # self.neck_bound = neck_bound
        self.neck_dense_type = neck_dense_type

        self.with_render_branch = with_render_branch
        if with_render_branch:
            # ! hybrid head
            print(f"{n_features[1]=}")
            print(f"{f_maps_2d=}")
            self.render_head_hybrid = LinearHead(
                n_features[1] + f_maps_2d,
                gsplat_upsample * gs_dim,
                order,
                num_groups,
                enhanced=gs_enhanced,
            )
            if apply_gs_init:
                self.render_head_hybrid.OutConv.weight.data.zero_()
                init_value = self.render_head_hybrid.OutConv.bias.data.view(
                    gsplat_upsample, gs_dim
                )
                init_value[:, :3] = 0.0
                if gsplat_upsample > 1:
                    init_value[:, :3] = torch.randn_like(init_value[:, :3]) * 0.5
                init_value[:, 3:6] = math.log(gs_init_scale)
                init_value[:, 6] = 1.0
                init_value[:, 7:10] = 0.0
                init_value[:, 10] = math.log(0.1 / (1 - 0.1))
                if gs_dim == 14:  # rgb
                    init_value[:, 11:14] = 0.5
                self.render_head_hybrid.OutConv.bias.data = init_value.view(-1)
            # ! 3D only head
            self.render_head_3D = LinearHead(
                n_features[1],
                gsplat_upsample * gs_dim,
                order,
                num_groups,
                enhanced=gs_enhanced,
            )
            if apply_gs_init:
                self.render_head_3D.OutConv.weight.data.zero_()
                init_value = self.render_head_3D.OutConv.bias.data.view(
                    gsplat_upsample, gs_dim
                )
                init_value[:, :3] = 0.0
                if gsplat_upsample > 1:
                    init_value[:, :3] = torch.randn_like(init_value[:, :3]) * 0.5
                init_value[:, 3:6] = math.log(gs_init_scale)
                init_value[:, 6] = 1.0
                init_value[:, 7:10] = 0.0
                init_value[:, 10] = math.log(0.1 / (1 - 0.1))
                if gs_dim == 14:  # rgb
                    init_value[:, 11:14] = 0.5
                self.render_head_3D.OutConv.bias.data = init_value.view(-1)

        self.gsplat_upsample = gsplat_upsample
        self.addtional_gs_constraint = addtional_gs_constraint
        self.gs_dim = gs_dim

        self.padding = fvnn.FillFromGrid()  # FillToGrid -> FillFromGrid
        # ! prepare for upsapmle occ-only part
        self.occ_upsample = fvnn.UpsamplingNearest(occ_upsample)
        self.feature_pooling_2d = feature_pooling_2d
        self.gs_free_space = gs_free_space
        self.max_return = max_return
        self.drop_invisible = drop_invisible
        self.max_scaling = max_scaling

    @classmethod
    def sparse_zero_padding(cls, in_x: fvnn.VDBTensor, target_grid: fvdb.GridBatch):
        source_grid = in_x.grid
        source_feature = in_x.data.jdata
        assert torch.allclose(source_grid.origins, target_grid.origins)
        assert torch.allclose(source_grid.voxel_sizes, target_grid.voxel_sizes)
        out_feat = torch.zeros(
            (target_grid.total_voxels, source_feature.size(1)),
            device=source_feature.device,
            dtype=source_feature.dtype,
        )
        in_idx = source_grid.ijk_to_index(target_grid.ijk).jdata
        in_mask = in_idx != -1
        out_feat[in_mask] = source_feature[in_idx[in_mask]]
        return fvnn.VDBTensor(target_grid, target_grid.jagged_like(out_feat))

    @classmethod
    def struct_to_mask(cls, struct_pred: fvnn.VDBTensor):
        # 0 is exist, 1 is non-exist
        mask = struct_pred.data.jdata[:, 0] > struct_pred.data.jdata[:, 1]
        return struct_pred.grid.jagged_like(mask)

    @classmethod
    def cat(cls, x: fvnn.VDBTensor, y: fvnn.VDBTensor):
        assert x.grid == y.grid
        return fvnn.VDBTensor(
            x.grid, x.grid.jagged_like(torch.cat([x.data.jdata, y.data.jdata], dim=1))
        )

    def build_normal_hash_tree(self, input_grid):
        hash_tree = {}

        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        _origins = input_grid.origins[0]
        _voxel_size = input_grid.voxel_sizes[0]

        for depth in range(self.num_blocks):
            voxel_size = [sv * 2**depth for sv in _voxel_size]
            origins = [
                _origins[idx] + 0.5 * _voxel_size[idx] * (2**depth - 1)
                for idx in range(3)
            ]

            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.gridbatch_from_nearest_voxels_to_points(
                    input_xyz, voxel_sizes=voxel_size, origins=origins
                )
        return hash_tree

    def build_fit_neck(self, sparse_grid, neck_expand: int = 1):
        sparse_coords = sparse_grid.ijk
        n_padding = (neck_expand - 1) // 2
        all_coords = []
        for b in range(sparse_grid.grid_count):
            min_bound = (
                torch.min(sparse_coords[b].jdata, dim=0).values.cpu().numpy()
                - n_padding
            )
            max_bound = (
                torch.max(sparse_coords[b].jdata, dim=0).values.cpu().numpy()
                + 1
                + n_padding
            )
            cx = torch.arange(
                min_bound[0],
                max_bound[0],
                dtype=torch.int32,
                device=sparse_coords.device,
            )
            cy = torch.arange(
                min_bound[1],
                max_bound[1],
                dtype=torch.int32,
                device=sparse_coords.device,
            )
            cz = torch.arange(
                min_bound[2],
                max_bound[2],
                dtype=torch.int32,
                device=sparse_coords.device,
            )
            coords = torch.stack(torch.meshgrid(cx, cy, cz, indexing="ij"), dim=3).view(
                -1, 3
            )
            all_coords.append(coords)
        all_coords = fvdb.JaggedTensor(all_coords)
        neck_grid = fvdb.gridbatch_from_ijk(
            all_coords,
            voxel_sizes=sparse_grid.voxel_sizes[0],
            origins=sparse_grid.origins[0],
        )
        return neck_grid

    class FeaturesSet:
        def __init__(self):
            self.encoder_features = {}
            self.structure_features = {}
            self.structure_grid = {}
            self.render_features = {}

    def _encode(self, x: fvnn.VDBTensor, hash_tree: dict, is_forward: bool = True):
        feat_depth = 0
        res = self.FeaturesSet()
        x = self.pre_conv(x)

        encoder_features = {}
        for module, downsampler in zip(self.encoders, [None] + list(self.downsamplers)):
            if downsampler is not None:
                x = downsampler(x, ref_coarse_data=hash_tree[feat_depth + 1])
                feat_depth += 1
            x, _ = module(x)
            encoder_features[feat_depth] = x

        if self.neck_dense_type == "UNCHANGED":
            pass
        elif self.neck_dense_type == "HAND_CRAFTED":
            voxel_size = x.grid.voxel_sizes[0]  # !: modify for remain h
            origins = x.grid.origins[0]  # !: modify for remain h
            neck_grid = fvdb.gridbatch_from_dense(
                x.grid.grid_count,
                self.voxel_bound,
                self.low_bound,  # type: ignore
                device="cpu",
                voxel_sizes=voxel_size,
                origins=origins,
            ).to(x.device)
            x = fvnn.VDBTensor(neck_grid, neck_grid.fill_from_grid(x.data, x.grid, 0.0))
        elif self.neck_dense_type == "FIT":
            neck_grid = self.build_fit_neck(x.grid)
            x = self.padding(x, neck_grid)
        else:
            raise NotImplementedError

        for module in self.pre_kl_bottleneck:
            x, _ = module(x)
        return res, x, encoder_features

    def encode(self, x: fvnn.VDBTensor, hash_tree: dict):
        return self._encode(x, hash_tree, True)

    def decode(
        self,
        res: FeaturesSet,
        x: fvnn.VDBTensor,
        hash_tree: dict,
        encoder_features: dict,
        img_features_batch,
        camera_pose,
        intrinsics,
        n_imgs,
    ):
        for module in self.post_kl_bottleneck:
            x, _ = module(x)

        struct_decision = None
        feat_depth = self.num_blocks - 1
        for module, upsampler, struct_conv in zip(
            [None] + list(self.decoders),
            [None] + list(self.upsamplers),
            self.struct_convs,
        ):
            if module is not None:
                x = upsampler(x, struct_decision)
                feat_depth -= 1

                enc_feat = self.padding(encoder_features[feat_depth], x)
                x = fvdb.jcat([enc_feat, x], dim=1)
                x, _ = module(x)
            # guided setting do not need to predict structure
            res.structure_features[feat_depth] = None
            # get the guided structure
            target_struct = hash_tree[feat_depth]
            struct_decision = target_struct.ijk_to_index(x.grid.ijk).jdata > -1
            res.structure_grid[feat_depth] = self.up_sample0(x, struct_decision).grid

        x = self.up_sample0(x, struct_decision)

        if self.with_render_branch:
            if x.grid.total_voxels > 0:
                decoded_gaussians = []
                h, w = img_features_batch.shape[3:5]
                grid = x.grid

                world_to_camera = torch.inverse(camera_pose)  # [B, N, 4, 4]
                input_camera_K = camera_intrinsic_list_to_matrix(
                    intrinsics, normalize_pixel=True
                )  # [B, N, 3, 3]
                world_to_image = torch.einsum(
                    "bnij,bnjk->bnik", input_camera_K, world_to_camera[..., :3, :4]
                )  # [B, N, 3, 4]

                for bidx in range(grid.grid_count):
                    cur_grid = grid[bidx]
                    image_features = img_features_batch[bidx]  # N, C, H, W
                    torch.cuda.empty_cache()

                    occ_front_voxel_mask, occ_front_per_camera = get_occ_front_voxel(
                        cur_grid,
                        camera_pose[bidx : bidx + 1],
                        intrinsics[bidx : bidx + 1],
                        max_height=h,
                        max_voxels=self.max_return,
                        return_per_cam_occ=True,
                    )
                    occ_front_voxel_mask = occ_front_voxel_mask.jdata.flatten()
                    occ_front_voxel_mask_jagged = JaggedTensor([occ_front_voxel_mask])
                    occluded_voxel_mask_jagged = JaggedTensor([~occ_front_voxel_mask])

                    # ! process occ-only part
                    ## use voxel center xyz to query image feature and then concat with voxel feature
                    cur_occ_tensor = self.occ_upsample(
                        VDBTensor(cur_grid, x.data[bidx]), occ_front_voxel_mask_jagged
                    )
                    cur_occ_per_camera_mask = self.occ_upsample(
                        VDBTensor(
                            cur_grid, JaggedTensor([occ_front_per_camera.jdata.float()])
                        ),
                        occ_front_voxel_mask_jagged,
                    ).data
                    cur_occ_per_camera_mask = cur_occ_per_camera_mask.jdata.to(
                        torch.bool
                    )

                    cur_occ_grid = cur_occ_tensor.grid
                    cur_occ_xyz = cur_occ_grid.grid_to_world(cur_occ_grid.ijk.float())

                    # camera coordinate
                    reference_points_cam, per_image_visibility_mask = project_points(
                        cur_occ_xyz, world_to_image[bidx : bidx + 1]
                    )

                    # !! since we are in batch = 1
                    reference_points_cam = (
                        reference_points_cam.jdata
                    )  # [N_voxel, N_view, 1, 2]
                    reference_points_cam = reference_points_cam.permute(
                        1, 0, 2, 3
                    )  # [N_view, N_voxel, 1, 2]. pseduo height = N_voxel, width = 1
                    grid_to_sample = (
                        2 * reference_points_cam - 1
                    )  # [N_view, N_voxel, 1, 2]

                    if self.feature_pooling_2d == "max":
                        # if input view is greater than 30, which is too large, we can not infer them in one forward
                        if reference_points_cam.shape[0] < 30:
                            sampled_features = F.grid_sample(
                                image_features, grid_to_sample
                            )  # [N_view, C, N_voxel, 1]
                            sampled_features = sampled_features[..., 0].permute(
                                2, 0, 1
                            )  # [N_voxel, N_view, C]

                            # mask out invisible points in some camera, using occ_front_per_camera
                            sampled_features.mul_(cur_occ_per_camera_mask.unsqueeze(-1))
                            occ_voxel_2D_feature = torch.max(sampled_features, dim=1)[0]

                        # since we use max pooling for the features, we can use for loop to get the max value and save memory
                        else:
                            logger.info("Too many views, use for loop to get max value")
                            occ_voxel_2D_feature = torch.zeros(
                                cur_occ_tensor.grid.total_voxels,
                                image_features.shape[1],
                                device=image_features.device,
                            )
                            for idx in range(reference_points_cam.shape[0]):
                                cur_mask = cur_occ_per_camera_mask[
                                    :, idx : idx + 1
                                ]  # [N_voxel, 1]
                                cur_feature = F.grid_sample(
                                    image_features[idx : idx + 1],
                                    grid_to_sample[idx : idx + 1],
                                )  # [1, C, N_voxel, 1]
                                cur_feature = cur_feature[..., 0].permute(
                                    2, 0, 1
                                )  # [N_voxel, 1, C]
                                cur_feature.mul_(cur_mask.unsqueeze(-1)).squeeze_(
                                    1
                                )  # [N_voxel, C]
                                # update the max value with inplace operation
                                occ_voxel_2D_feature = torch.max(
                                    occ_voxel_2D_feature,
                                    cur_feature,
                                    out=occ_voxel_2D_feature,
                                )
                    else:
                        raise NotImplementedError

                    occ_voxel_3D_feature = cur_occ_tensor.data.jdata
                    occ_voxel_hybrid_feature = torch.cat(
                        [occ_voxel_2D_feature, occ_voxel_3D_feature], dim=1
                    )
                    occ_render_feature = self.render_head_hybrid(
                        VDBTensor(
                            cur_occ_grid,
                            cur_occ_grid.jagged_like(occ_voxel_hybrid_feature),
                        )
                    )
                    occ_gaussians = self.feature2gs(
                        cur_occ_grid, occ_render_feature.data.jdata
                    )

                    # ! prcess non-occ part
                    if not self.drop_invisible:
                        cur_non_occ_tensor = self.up_sample0(
                            VDBTensor(x.grid[bidx], x.data[bidx]),
                            occluded_voxel_mask_jagged,
                        )
                        cur_non_occ_tensor = self.render_head_3D(cur_non_occ_tensor)
                        non_occ_gaussians = self.feature2gs(
                            cur_non_occ_tensor.grid, cur_non_occ_tensor.data.jdata
                        )
                        decoded_gaussians.append(
                            torch.cat([occ_gaussians, non_occ_gaussians], dim=0)
                        )
                    else:
                        decoded_gaussians.append(occ_gaussians)

        return decoded_gaussians

    def feature2gs(self, grid, feature):
        feature = feature.view(-1, self.gsplat_upsample, self.gs_dim)
        _rel_xyz = feature[:, :, :3]
        _scaling = feature[:, :, 3:6]
        _rots = feature[:, :, 6:10]
        _opacities = feature[:, :, 10:11]
        _color = feature[:, :, 11 : self.gs_dim]

        rel_pos = get_rel_pos(_rel_xyz, self.gs_free_space, grid)
        abs_pos = (
            grid.grid_to_world(grid.ijk.float() - 0.5).jdata.unsqueeze(1) + rel_pos
        )
        abs_pos = abs_pos.view(-1, 3)
        scaling = (torch.exp(_scaling) * grid.voxel_sizes[0, 0]).view(-1, 3)

        if self.max_scaling > 0.0:
            scaling = torch.clamp(scaling, max=self.max_scaling)

        rotation = torch.nn.functional.normalize(_rots.view(-1, 4), dim=1)
        opacity = torch.sigmoid(_opacities.view(-1, 1))

        color_dim = self.gs_dim - 11
        color = _color.view(-1, color_dim)  # rgb or feature

        gs_feature = torch.cat([abs_pos, scaling, rotation, opacity, color], dim=1)
        return gs_feature

    def forward(self, batch, imgenc_output):
        x_grid = batch[DS.INPUT_PC]
        voxel_features = self.lifter(batch, imgenc_output)
        x = fvnn.VDBTensor(x_grid, x_grid.jagged_like(voxel_features))

        img_features = imgenc_output[self.img_feature_source]
        img_features_effective_mask = imgenc_output["input_effective_mask_resized"] > 0

        # directly use the input effective mask
        img_features = img_features * img_features_effective_mask

        camera_pose = torch.stack(batch[DS.IMAGES_INPUT_POSE], dim=0)
        intrinsics = torch.stack(batch[DS.IMAGES_INPUT_INTRINSIC], dim=0)
        n_imgs, H, W = img_features.size(1), img_features.size(3), img_features.size(4)

        # update the batch[DS.IMAGES_INPUT_INTRINSIC]
        if (H != intrinsics[..., 5]).all() or (W != intrinsics[..., 4]).all():
            downsample_h = intrinsics[0, 0, 5] / H
            downsample_w = intrinsics[0, 0, 4] / W
            intrinsics[..., [1, 3, 5]] = intrinsics[..., [1, 3, 5]] / downsample_h
            intrinsics[..., [0, 2, 4]] = intrinsics[..., [0, 2, 4]] / downsample_w

        # build a hash tree
        hash_tree = self.build_normal_hash_tree(x.grid)
        res, x, encoder_features = self.encode(x, hash_tree)
        decoded_gaussians = self.decode(
            res,
            x,
            hash_tree,
            encoder_features,
            img_features,
            camera_pose,
            intrinsics,
            n_imgs,
        )

        network_output = {"decoded_gaussians": decoded_gaussians}

        return network_output


class Lifter(nn.Module):
    def __init__(self, img_feature_source, img_in_dim, voxel_out_dim):
        super().__init__()
        self.img_feature_source = img_feature_source
        self.mix_fc = nn.Linear(img_in_dim, voxel_out_dim)

    def build_ray_casting_feature(self, batch, imgenc_output):
        """
        This is previous `build_occulusion_feature_cube`, I use the new name for more accurate meaning

        We unproject the image pixels to the voxel grid, and assign the pixel feature to the voxel grid,
        then return the voxel feature for each voxel.

        Args:
            grid: len(grid) == B
            img_features: [B, N, C, H, W]
            camera_pose: [B, N, 4, 4]
            intrinsics: [B, N, 6], 6 is fx fy cx cy w h

        Returns:
            voxel_features: JaggedTensor
        """
        img_features = imgenc_output[self.img_feature_source]
        grid = batch[DS.INPUT_PC]
        camera_pose = torch.stack(batch[DS.IMAGES_INPUT_POSE], dim=0)
        intrinsics = torch.stack(batch[DS.IMAGES_INPUT_INTRINSIC], dim=0)

        voxel_features = []
        n_imgs, H, W = img_features.size(1), img_features.size(3), img_features.size(4)
        # update the batch[DS.IMAGES_INPUT_INTRINSIC]
        if (H != intrinsics[..., 5]).all() or (W != intrinsics[..., 4]).all():
            downsample_h = intrinsics[0, 0, 5] / H
            downsample_w = intrinsics[0, 0, 4] / W
            intrinsics[..., [1, 3, 5]] = intrinsics[..., [1, 3, 5]] / downsample_h
            intrinsics[..., [0, 2, 4]] = intrinsics[..., [0, 2, 4]] / downsample_w

        # get input effective mask
        input_effective_mask = imgenc_output["input_effective_mask"]  # [B, N, 1, h, w]

        # if shape mismatch, we need to resize the mask
        if input_effective_mask.shape[2] != H or input_effective_mask.shape[3] != W:
            B, N, _, input_effective_mask_h, input_effective_mask_w = (
                input_effective_mask.shape
            )
            input_effective_mask = F.interpolate(
                input_effective_mask.view(
                    B * N, 1, input_effective_mask_h, input_effective_mask_w
                ),
                size=(H, W),
                mode="nearest",
            )

            input_effective_mask = input_effective_mask.view(B, N, 1, H, W)

        imgenc_output["input_effective_mask_resized"] = input_effective_mask.view(
            B, N, 1, H, W
        )

        for bidx in range(grid.grid_count):
            cur_grid = grid[bidx]
            cur_pose = camera_pose[bidx]  # N, 4, 4
            cur_intrinsics = intrinsics[bidx]

            # [N, 3], [N, H, W, 3] -> [N * H * W, 3]
            nimg_origins, nimg_directions = create_rays_from_intrinsic_torch_batch(
                cur_pose, cur_intrinsics
            )
            nimg_origins = (
                nimg_origins.view(n_imgs, 1, 1, 3).expand(-1, H, W, -1).reshape(-1, 3)
            )
            nimg_directions = nimg_directions.reshape(-1, 3)

            nimg_features = img_features[bidx]  # N, C, H, W
            nimg_features = nimg_features.permute(0, 2, 3, 1).view(
                n_imgs * H * W, -1
            )  # N, C, H, W -> N, H, W, C -> N * H * W, C
            effective_feature_mask = input_effective_mask[bidx].view(
                n_imgs * H * W
            )  # N, H, W, 1 -> N * H * W

            if fvdb.__version__ == "0.0.0":
                pack_info, out_voxel_ijk, _ = cur_grid.voxels_along_rays(
                    JaggedTensor([nimg_origins]),
                    JaggedTensor([nimg_directions]),
                    max_voxels=1,
                )
                out_voxel_ids = cur_grid.ijk_to_index(out_voxel_ijk)
                pixel_feature = nimg_features[pack_info.jdata[:, 1] > 0, :]
            else:
                out_voxel_ids, ray_start_end = cur_grid.voxels_along_rays(
                    JaggedTensor([nimg_origins]),
                    JaggedTensor([nimg_directions]),
                    max_voxels=1,
                    return_ijk=False,
                )

                mask = (
                    ray_start_end.joffsets[1:] - ray_start_end.joffsets[:-1]
                ).bool()  # [N_ray]
                pixel_feature = nimg_features[mask, :]  # [N_ray_hit, C]
                out_voxel_ids = out_voxel_ids.jdata.to(torch.int64)
                effective_feature_mask = effective_feature_mask[mask]

            # if any effective_feature_mask has 0 value
            if (effective_feature_mask == 0).any():
                pixel_feature = pixel_feature[effective_feature_mask > 0]
                out_voxel_ids = out_voxel_ids[effective_feature_mask > 0]

            out_voxel_features = torch.zeros(
                (cur_grid.total_voxels, nimg_features.shape[1]), device=cur_pose.device
            )
            out_voxel_features = scatter_mean(
                pixel_feature, out_voxel_ids, out=out_voxel_features, dim=0
            )

            voxel_features.append(out_voxel_features)

        voxel_features = torch.cat(voxel_features, dim=0)

        return voxel_features

    def forward(self, batch, imgenc_output):
        voxel_features = self.build_ray_casting_feature(batch, imgenc_output)
        voxel_features = self.mix_fc(voxel_features)
        return voxel_features
