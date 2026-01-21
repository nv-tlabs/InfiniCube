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

import fvdb
import torch
import torch.nn as nn
from torch_scatter import scatter_sum


class MapEncoder(nn.Module):
    def __init__(
        self,
        cube_bbox_size,  # [voxel_x_max, voxel_y_max, voxel_z_max]
        return_dense_cube=True,
        use_embedding=False,
        map_types=[],
        embedding_dim=1,
    ):
        super().__init__()
        self.cube_bbox_size = cube_bbox_size
        self.return_dense_cube = return_dense_cube
        self.use_embedding = use_embedding

        if isinstance(cube_bbox_size, int):
            neck_bound = cube_bbox_size // 2
            low_bound = [-neck_bound] * 3
            high_bound = [neck_bound] * 3
        else:
            low_bound = [-int(res / 2) for res in cube_bbox_size]
            high_bound = [int(res / 2) for res in cube_bbox_size]

        self.low_bound = low_bound
        self.high_bound = high_bound

        if self.use_embedding:
            self.embedding = nn.Embedding(len(map_types), embedding_dim)
            self.embedding.weight.data.normal_(0, 1)

    def forward(self, map_3d_dict, neck_voxel_sizes):
        map_flag_dict = {}
        bound_length = torch.tensor(
            self.high_bound, device=neck_voxel_sizes.device
        ) - torch.tensor(self.low_bound, device=neck_voxel_sizes.device)

        for map_type_idx, map_type in enumerate(map_3d_dict.keys()):
            map_merged_points = map_3d_dict[
                map_type
            ]  # list of torch.Tensor, shape[N_map_point, 3]
            B = len(map_merged_points)
            max_length = max(
                [len(map_merged_points[i]) for i in range(len(map_merged_points))]
            )
            # create a padded tensor
            map_merged_points_padded = torch.full(
                (len(map_merged_points), max_length, 3),
                1e6,
                device=map_merged_points[0].device,
            )
            for i in range(len(map_merged_points)):
                if map_merged_points[i].shape[0] == 0:
                    continue
                map_merged_points_padded[i, : len(map_merged_points[i]), :] = (
                    map_merged_points[i]
                )

            # [B, max_length, 3]
            neck_voxel_sizes = neck_voxel_sizes.view(B, 1, 3)
            low_bound_range = (
                torch.tensor(self.low_bound, device=neck_voxel_sizes.device)
                * neck_voxel_sizes
            )

            points_3d_ijk = (
                ((map_merged_points_padded - (neck_voxel_sizes / 2)) / neck_voxel_sizes)
                .round()
                .long()
            )  # ijk from grid origins (voxel centers)
            points_3d_ijk_from_low_bound = points_3d_ijk - torch.tensor(
                self.low_bound, device=map_merged_points_padded.device
            )
            batch_ix = torch.stack(
                [
                    torch.full(
                        [max_length, 1],
                        ix,
                        device=points_3d_ijk_from_low_bound.device,
                        dtype=torch.long,
                    )
                    for ix in range(B)
                ]
            )
            points_3d_ijk_from_low_bound = torch.cat(
                [points_3d_ijk_from_low_bound, batch_ix], dim=-1
            )

            # filter out points that are outside box, (B, N*H'*W')
            kept = (
                (points_3d_ijk_from_low_bound[..., 0] >= 0)
                & (points_3d_ijk_from_low_bound[..., 0] < bound_length[0])
                & (points_3d_ijk_from_low_bound[..., 1] >= 0)
                & (points_3d_ijk_from_low_bound[..., 1] < bound_length[1])
                & (points_3d_ijk_from_low_bound[..., 2] >= 0)
                & (points_3d_ijk_from_low_bound[..., 2] < bound_length[2])
            )

            # (N_valid, 4)
            points_3d_ijk_from_low_bound = points_3d_ijk_from_low_bound[kept]

            # filter out duplicate points
            points_3d_ijk_from_low_bound_unique = torch.unique(
                points_3d_ijk_from_low_bound, dim=0
            )

            # scatter to cube
            x = torch.ones(
                points_3d_ijk_from_low_bound_unique.shape[0],
                1,
                device=points_3d_ijk_from_low_bound_unique.device,
            )
            nx = bound_length
            ijk_from_low_bound = points_3d_ijk_from_low_bound_unique
            ranks = (
                ijk_from_low_bound[:, 0] * (nx[1] * nx[2] * B)
                + ijk_from_low_bound[:, 1] * (nx[2] * B)
                + ijk_from_low_bound[:, 2] * B
                + ijk_from_low_bound[:, 3]
            )

            # [X, Y, Z, B, 1]
            final = scatter_sum(
                x, ranks, dim=0, dim_size=nx[0] * nx[1] * nx[2] * B
            ).view(nx[0], nx[1], nx[2], B, -1)
            # [B, X, Y, Z, 1]
            final = final.permute(3, 0, 1, 2, 4)

            if self.use_embedding:
                map_type_idx_tensor = torch.tensor(
                    [map_type_idx], device=neck_voxel_sizes.device, dtype=torch.long
                )
                emb = self.embedding(map_type_idx_tensor)  # [1, embedding_dim]
                # [B, X, Y, Z, embedding_dim]
                final = final * emb

            map_flag_dict[map_type] = final

        map_flag = torch.cat(
            [map_flag_dict[map_type] for map_type in map_flag_dict.keys()], dim=-1
        )
        X, Y, Z = map_flag.shape[1:-1]

        if self.return_dense_cube:
            # note that the origins and voxel sizes are default! we just need the features
            voxel_tensor = fvdb.nn.vdbtensor_from_dense(
                map_flag, ijk_min=[-X // 2, -Y // 2, -Z // 2]
            )
        else:
            raise NotImplementedError

        return voxel_tensor
