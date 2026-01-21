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

"""
FVDB Voxel Creation Mechanism Explanation
==========================================

When using fvdb.gridbatch_from_points() to create voxel grids, the number of voxels depends on:
1. Point distribution range
2. voxel_size (voxel size)
3. origin (origin position)

Voxel index formula: voxel_idx = floor((point - origin) / voxel_size)

Two examples below:

Example 1: grid1 (origin=[0,0,0], voxel_size=1) -> 8 voxels
===========================================================
Point range: [0, 1) x [0, 1) x [0, 1)


2D view (x-y plane, showing voxel boundaries):

         y
    1.5  +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |          *  * |  *            |
         |          * *  | * *           |
         |           *   | * **          |
    0.5  +---------------+---------------+  <- voxel boundary
         |           * * |* *  *         |
         |            *  |  *  *         |
         |               |               |
         |       o       |               |
         |               |               |
         |               |               |
   -0.5  +---------------+---------------+  <- voxel boundary
        -0.5            0.5              1.5
                         x


Example 2: grid2 (origin=[0.5,0.5,0.5], voxel_size=1) -> 1 voxel
================================================================
Point range (world coords): [0, 1) x [0, 1) x [0, 1)
Offset from origin: [-0.5, 0.5) x [-0.5, 0.5) x [-0.5, 0.5)

2D view (top view, z=0.5 plane):

    y
    1 +---------------+   <- voxel boundary
      |               |
      |   ******      |
      |  ******* **   |
      |   ******      |
      |               |
    0 +---------------+   <- voxel boundary
      0               1  x


Total: 1 voxel

Conclusion
==========
grid_coordinate = floor((point - origin) / voxel_size)
ijk_coordinate = round(grid_coordinate)
"""

import fvdb
import torch
from fvdb import JaggedTensor
from termcolor import cprint

points = torch.rand(100, 3).cuda()  # range in (0, 0, 0) to (1, 1, 1)
jagged_points = JaggedTensor([points])

grid_kwargs1 = {
    "voxel_sizes": 1,
    "origins": [0.0, 0.0, 0.0],
}

grid_kwargs2 = {
    "voxel_sizes": 1,
    "origins": [0.5, 0.5, 0.5],
}

grid1 = fvdb.gridbatch_from_points(points, **grid_kwargs1)
grid2 = fvdb.gridbatch_from_points(points, **grid_kwargs2)
new_point = [1.1, 0.6, 0.6]
new_point = torch.tensor(new_point).cuda().reshape(1, 3)
new_jagged_points = JaggedTensor([new_point])


def show(grid, new_point):
    new_point = torch.tensor(new_point).cuda().reshape(1, 3)
    new_jagged_points = JaggedTensor([new_point])
    total_voxel = grid.total_voxels
    grid_coordinate = grid.world_to_grid(new_jagged_points).jdata
    ijk_coordinate = (
        grid.world_to_grid(new_jagged_points).jdata.round().long()
    )  # you should use round to get the ijk coordinate
    print(
        f"voxel number {total_voxel} for grid with voxel size {grid.voxel_sizes} and origin {grid.origins}"
    )
    print(
        f"point {new_point}'s grid coordinate {grid_coordinate}, ijk coordinate {ijk_coordinate}"
    )

    if torch.all(
        grid.ijk_to_index(JaggedTensor([ijk_coordinate])).jdata >= 0
    ):  # -1 is out of voxels
        cprint(f"point {new_point} is in the grid", "green")
    else:
        cprint(f"point {new_point} is out of the grid", "red")


cprint("grid1:", "blue", attrs=["bold"])
show(grid1, new_point)
cprint("grid2:", "blue", attrs=["bold"])
show(grid2, new_point)
