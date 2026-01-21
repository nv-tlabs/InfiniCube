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

import matplotlib as mpl
import numpy as np
import torch


def create_instance_mapping(
    unique_instance_ids, color_map_for_vechile="PuRd", color_map_for_pedestrian="YlOrBr"
):
    """
    Create the instance mapping for the given unique instance ids.

    Returns:
        dict: The instance mapping, key is the instance id, value is the color.
    """
    cmap_for_vechile = mpl.colormaps[color_map_for_vechile]
    cmap_for_pedestrian = mpl.colormaps[color_map_for_pedestrian]

    unique_vehicle_instance_ids = unique_instance_ids[unique_instance_ids < 2**15]
    unique_pedestrian_instance_ids = unique_instance_ids[unique_instance_ids >= 2**15]

    # for vehicle
    instance_colors_vehicle = np.random.rand(len(unique_vehicle_instance_ids))
    instance_colors_vehicle = [
        cmap_for_vechile(x)[:3] for x in instance_colors_vehicle
    ]  # get the color (rgba -> rgb) for each instance
    instance_color_map_vehicle = dict(
        zip(unique_vehicle_instance_ids, instance_colors_vehicle)
    )
    # for pedestrian
    instance_colors_pedestrian = np.random.rand(len(unique_pedestrian_instance_ids))
    instance_colors_pedestrian = [
        cmap_for_pedestrian(x)[:3] for x in instance_colors_pedestrian
    ]  # get the color (rgba -> rgb) for each instance
    instance_color_map_pedestrian = dict(
        zip(unique_pedestrian_instance_ids, instance_colors_pedestrian)
    )

    # merge the two instance dict into one
    instance_color_map = {**instance_color_map_vehicle, **instance_color_map_pedestrian}

    return instance_color_map


def coloring_instance_map_with_mapping(instance_maps, instance_to_rgb_mapping):
    """
    Color the instance maps with the given color map. Need to consider all of the instances in N images.

    Args:
        instance_maps (np.ndarray / torch.Tensor): The instance maps to be colored.
            shape [N, H, W, 1] or [N, H, W], pixel value indicates a unique instance.
        instance_to_rgb_mapping (dict): The instance to rgb mapping.
            key is the instance id, value is the color int [0, 1].

    Returns:
        colored_instance_maps (np.ndarray / torch.Tensor): The colored instance maps, shape [N, H, W, 3]. range in [0, 1].
    """
    if len(instance_maps.shape) == 4:
        assert instance_maps.shape[-1] == 1
        instance_maps = instance_maps[..., 0]

    is_torch = isinstance(instance_maps, torch.Tensor)
    if is_torch:
        device = instance_maps.device
        instance_maps = instance_maps.cpu().numpy()

    # get the unique instance ids
    unique_instance_ids = list(instance_to_rgb_mapping.keys())

    # color the instance maps
    colored_instance_maps = np.zeros(instance_maps.shape + (3,))  # shape [N, H, W, 3]
    for i, instance_id in enumerate(unique_instance_ids):
        colored_instance_maps[instance_maps == instance_id] = np.array(
            instance_to_rgb_mapping[instance_id]
        )

    if is_torch:
        colored_instance_maps = torch.from_numpy(colored_instance_maps).to(device)

    return colored_instance_maps


def coloring_instance_map(
    instance_maps, color_map_for_vechile="PuRd", color_map_for_pedestrian="YlOrBr"
):
    """
    Color the instance maps with the given color map. Need to consider all of the instances in N images.

    Args:
        instance_maps (np.ndarray / torch.Tensor): The instance maps to be colored.
            shape [N, H, W, 1] or [N, H, W], pixel value indicates a unique instance.

        color_map_for_vechile (str): The color map to be used for coloring the vehicle instances.

        color_map_for_pedestrian (str): The color map to be used for coloring the pedestrian instances.
            typically, we let the instance id of pedestrian start from 2**15.

    Returns:
        np.ndarray: The colored instance maps, shape [N, H, W, 3]. range in [0, 1].
    """
    if len(instance_maps.shape) == 4:
        assert instance_maps.shape[-1] == 1
        instance_maps = instance_maps[..., 0]

    is_torch = isinstance(instance_maps, torch.Tensor)
    if is_torch:
        device = instance_maps.device
        instance_maps = instance_maps.cpu().numpy()

    # Get the unique instances in the instance maps
    unique_instance_ids = np.unique(instance_maps)

    # remove 0, which is the background
    unique_instance_ids = unique_instance_ids[unique_instance_ids != 0]

    # create the instance color map
    instance_color_map = create_instance_mapping(
        unique_instance_ids, color_map_for_vechile, color_map_for_pedestrian
    )

    # color the instance maps
    colored_instance_maps = coloring_instance_map_with_mapping(
        instance_maps, instance_color_map
    )

    if is_torch:
        colored_instance_maps = torch.from_numpy(colored_instance_maps).to(device)

    return colored_instance_maps
