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
from fvdb import JaggedTensor
from fvdb.nn import VDBTensor
from pytorch3d.ops import box3d_overlap


class Box3dEncoder(nn.Module):
    def __init__(
        self,
        cube_bbox_size,  # [voxel_x_max, voxel_y_max, voxel_z_max]
        add_occupancy_flag=False,
        return_dense_cube=True,
    ):
        super().__init__()
        self.cube_bbox_size = cube_bbox_size

        if isinstance(cube_bbox_size, int):
            neck_bound = cube_bbox_size // 2
            low_bound = [-neck_bound] * 3
            high_bound = [neck_bound] * 3
        else:
            low_bound = [-int(res / 2) for res in cube_bbox_size]
            high_bound = [int(res / 2) for res in cube_bbox_size]

        self.low_bound = low_bound
        self.high_bound = high_bound

        self.add_occupancy_flag = add_occupancy_flag
        self.cond_dim = 2 if not add_occupancy_flag else 3

    def forward(self, corners3d_list, neck_voxel_sizes):
        """
        corners3d_list:
            list of torch.Tensor, shape [N_i, 8, 3]
        neck_voxel_sizes:
            torch.Tensor, shape [B, 3]
        """
        bound_length = torch.tensor(
            self.high_bound, device=neck_voxel_sizes.device
        ) - torch.tensor(self.low_bound, device=neck_voxel_sizes.device)
        encode_cond_vdb_tensor_list = []

        # build dense cube
        grid = fvdb.gridbatch_from_dense(
            num_grids=1,
            ijk_min=self.low_bound,
            dense_dims=bound_length,
            origins=neck_voxel_sizes[0] / 2,
            voxel_sizes=neck_voxel_sizes[0],
            device="cuda",
        )

        # get ijk and indices
        ijk = grid.ijk.jdata
        indices = grid.ijk_to_index(ijk).jdata
        neck_voxel_sizes_vector = neck_voxel_sizes[0]
        neck_voxel_volume = neck_voxel_sizes_vector.prod()

        # now we can use ijk, origins and voxel_sizes to create 3d bbox for IoU calculation
        offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.int32,
            device=neck_voxel_sizes.device,
        )

        dense_grid_corners = torch.stack(
            [ijk + offsets[i] for i in range(8)], dim=1
        )  # [N_dense_voxel, 8, 3]

        # to world coordinates
        dense_grid_corners = dense_grid_corners * neck_voxel_sizes_vector

        for corners3d in corners3d_list:
            corners3d = corners3d.float()
            if corners3d.shape[0] == 0:
                dense_grid_box_cond = torch.zeros(
                    (len(dense_grid_corners), 2), device=neck_voxel_sizes.device
                )
            else:
                # we first encode the corners3d, we get its orientation's sin and cos. heading is from vertex 3 to vertex 0
                heading_bev = torch.atan2(
                    corners3d[:, 0, 1] - corners3d[:, 3, 1],
                    corners3d[:, 0, 0] - corners3d[:, 3, 0],
                )
                heading_bev_sin = torch.sin(heading_bev)
                heading_bev_cos = torch.cos(heading_bev)
                if self.add_occupancy_flag:
                    heading_bev_enc = torch.stack(
                        [
                            heading_bev_sin,
                            heading_bev_cos,
                            torch.ones_like(heading_bev_sin),
                        ],
                        dim=-1,
                    )
                else:
                    heading_bev_enc = torch.stack(
                        [heading_bev_sin, heading_bev_cos], dim=-1
                    )  # [N, 3]

                max_height = torch.max(corners3d[:, :, 2])
                min_height = torch.min(corners3d[:, :, 2])

                # dense_grid_corners_outside, [N_dense_voxel]
                dense_grid_corners_outside = torch.all(
                    dense_grid_corners[:, :, 2] > max_height, dim=1
                ) | torch.all(dense_grid_corners[:, :, 2] < min_height, dim=1)
                dense_grid_corners_inside = ~dense_grid_corners_outside

                if dense_grid_corners_inside.sum() == 0:
                    dense_grid_box_cond = torch.zeros(
                        (len(dense_grid_corners), 2), device=neck_voxel_sizes.device
                    )

                else:
                    # corners3d: [N, 8, 3], intersection_vol [N_dense_voxel, N], intersection_vol [N_dense_voxel, N]
                    # we don't need to calculate all the dense_grid_corners. some of them are too high or too low,
                    # since corners3d just appear on the road surface. Consider reducing the number in box3d_overlap
                    intersection_vol = torch.zeros(
                        (len(dense_grid_corners), len(corners3d)),
                        device=neck_voxel_sizes.device,
                    )
                    iou_3d = torch.zeros(
                        (len(dense_grid_corners), len(corners3d)),
                        device=neck_voxel_sizes.device,
                    )
                    try:
                        (
                            intersection_vol[dense_grid_corners_inside],
                            iou_3d[dense_grid_corners_inside],
                        ) = box3d_overlap(
                            dense_grid_corners[dense_grid_corners_inside],
                            corners3d,
                            eps=1e-1,
                        )
                    except ValueError as e:
                        raise RuntimeError(
                            f"Error in box3d_overlap: {e}, we will skip this sample"
                        )

                    # for each voxel in N_dense_voxel, we get the max-iou box and use its heading encoding.
                    # if all intersection_vol < 50% neck_voxel_volume, we ignore this box
                    max_iou, max_iou_corner_idx = iou_3d.max(
                        dim=1
                    )  # [N_dense_voxel], [N_dense_voxel]
                    dense_grid_box_cond = torch.zeros(
                        (len(dense_grid_corners), self.cond_dim),
                        device=neck_voxel_sizes.device,
                    )
                    mask = (
                        torch.gather(
                            intersection_vol,
                            dim=1,
                            index=max_iou_corner_idx.unsqueeze(-1),
                        )
                        > 0.5 * neck_voxel_volume
                    )
                    mask = mask.squeeze()
                    dense_grid_box_cond[mask] = heading_bev_enc[
                        max_iou_corner_idx[mask]
                    ]

            encode_cond_vdb_tensor_list.append(
                VDBTensor(grid, JaggedTensor(dense_grid_box_cond))
            )

        return fvdb.jcat(encode_cond_vdb_tensor_list)

    def generate_visualization_items(self, corners3d_list, neck_voxel_sizes):
        """
        corners3d_list:
            list of torch.Tensor, shape [N_i, 8, 3]
        neck_voxel_sizes:
            torch.Tensor, shape [B, 3]


        """
        return self.forward(corners3d_list, neck_voxel_sizes)
