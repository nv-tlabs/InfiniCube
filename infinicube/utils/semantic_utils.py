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
from pycg import color

from infinicube.utils.instance_utils import coloring_instance_map

WAYMO_CATEGORY_NAMES = [
    "UNDEFINED",
    "CAR",
    "TRUCK",
    "BUS",
    "OTHER_VEHICLE",
    "MOTORCYCLIST",
    "BICYCLIST",
    "PEDESTRIAN",
    "SIGN",
    "TRAFFIC_LIGHT",
    "POLE",
    "CONSTRUCTION_CONE",
    "BICYCLE",
    "MOTORCYCLE",
    "BUILDING",
    "VEGETATION",
    "TREE_TRUNK",
    "CURB",
    "ROAD",
    "LANE_MARKER",
    "OTHER_GROUND",
    "WALKABLE",
    "SIDEWALK",
]

WAYMO_VISUALIZATION_TYPES_BLUE_SKY = {
    0: ["SIGN", "TRAFFIC_LIGHT", "CONSTRUCTION_CONE"],
    1: ["MOTORCYCLIST", "BICYCLIST", "PEDESTRIAN", "BICYCLE", "MOTORCYCLE"],
    2: ["WALKABLE", "SIDEWALK"],
    3: ["CAR", "TRUCK", "BUS", "OTHER_VEHICLE"],
    4: ["VEGETATION", "TREE_TRUNK"],
    5: ["CURB", "LANE_MARKER"],
    6: ["BUILDING"],
    7: ["ROAD", "OTHER_GROUND"],
    8: ["UNDEFINED"],
    9: ["POLE"],
}


def build_waymo_mapping_and_palette():
    waymo_mapping = np.zeros(23, dtype=np.int32)
    for (
        palette_idx,
        waymo_semantic_categories,
    ) in WAYMO_VISUALIZATION_TYPES_BLUE_SKY.items():
        for waymo_semantic_name in waymo_semantic_categories:
            waymo_mapping[WAYMO_CATEGORY_NAMES.index(waymo_semantic_name)] = palette_idx

    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    waymo_palette = np.zeros((10, 3), dtype=np.float32)
    waymo_palette[:8] = color.get_cmap_array("Set2")
    # Change the purple and green color
    waymo_palette[3] = color.get_cmap_array("Set3")[9]
    waymo_palette[4] = color.get_cmap_array("Set1")[2]
    waymo_palette[8] = color.get_cmap_array("Paired")[1]
    waymo_palette[9] = color.get_cmap_array("Set3")[10]

    return waymo_mapping, waymo_palette


# category index -> WAYMO_MAPPING -> palette index
# palette index -> WAYMO_PALETTE -> color
WAYMO_MAPPING, WAYMO_PALETTE = build_waymo_mapping_and_palette()


def semantic_to_color(semantics):
    """
    Args:
        semantics (np.ndarray): shape [N,]

    Returns:
        colors (np.ndarray): shape [N, 3], range [0, 1]
    """
    if isinstance(semantics, torch.Tensor):
        semantics = semantics.cpu().numpy()

    colors_index = WAYMO_MAPPING[semantics]
    colors = WAYMO_PALETTE[colors_index]
    return colors


def generate_rgb_semantic_buffer(
    semantics_rgb: np.ndarray,
    instance_buffer: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """
    coloring the instance buffer on the semantics rgb buffer,
    then overlay the instance buffer on the semantics rgb buffer.

    Args:
        semantics_rgb: np.ndarray, shape [N, H, W, 3] uint8
        instance_buffer: np.ndarray, shape [N, H, W] uint16

    Returns:
        semantic_buffer_rgb: np.ndarray, shape [N, H, W, 3] uint8
    """
    instance_colored = coloring_instance_map(
        instance_buffer
    )  # shape [N, H, W, 3], range [0, 1]
    instance_colored = (instance_colored * 255).astype(np.uint8)  # Convert to [0, 255]

    # Overlay: use instance color where instance exists
    # (instance_buffer > 0), otherwise use semantic color
    instance_mask = (instance_buffer > 0)[..., np.newaxis]  # shape [N, H, W, 1]
    semantic_buffer_rgb = np.where(
        instance_mask, instance_colored, semantics_rgb
    )  # shape [N, H, W, 3]

    return semantic_buffer_rgb
