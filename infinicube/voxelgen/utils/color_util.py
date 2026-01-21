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

import torch

from infinicube.voxelgen.ext import common


def color_from_points(target_pcs, ref_pcs, ref_colors, k=8):
    """
    Compute the color of each point in the target point cloud by weighted average of the colors of its k nearest neighbors in the reference point cloud.

    Parameters:
        target_pcs (torch.Tensor): Coordinates of the points in the target point cloud, size (N, 3).
        ref_pcs (torch.Tensor): Coordinates of the points in the reference point cloud, size (M, 3).
        ref_colors (torch.Tensor): Colors of each point in the reference point cloud, size (M, 3).
        k (int): Number of nearest neighbors to consider for each point in the target point cloud.

    Returns:
        torch.Tensor: Calculated colors for each point in the target point cloud, size (N, 3).
    """
    if target_pcs.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=target_pcs.device)
    torch.cuda.empty_account()
    dist, idx = common.knn_query_fast(target_pcs.contiguous(), ref_pcs.contiguous(), k)
    dist = dist.sqrt()

    knn_color = ref_colors[idx.long()]
    weight = 1 / (dist + 1e-8)  # N, K, inverse distance weighting
    weight = weight / weight.sum(
        dim=1, keepdim=True
    )  # normalize weights across each point's k neighbors
    target_color = (weight.unsqueeze(-1) * knn_color).sum(
        dim=1
    )  # weighted average of colors

    return target_color


def semantic_from_points(target_pcs, ref_pcs, ref_semantic):
    if target_pcs.shape[0] == 0:
        return torch.zeros((0), dtype=torch.int64, device=target_pcs.device)
    torch.cuda.empty_cache()
    dist, idx = common.knn_query_fast(target_pcs.contiguous(), ref_pcs.contiguous(), 1)
    dist = dist.sqrt()

    knn_color = ref_semantic[idx.long()]
    return knn_color[:, 0].long()
