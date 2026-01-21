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

import copy
import io

import numpy as np
from pyquaternion import Quaternion


############ Encoding Related ############
def encode_dict_to_npz_bytes(data_dict):
    buffer = io.BytesIO()
    np.savez(buffer, **data_dict)
    buffer.seek(0)

    return buffer.getvalue()


def imageencoder_imageio_png(image):
    """Compress an image using PIL and return it as a string.

    Can handle 16-bit images.

    :param image: ndarray representing an image

    """
    import imageio.v3 as imageio

    with io.BytesIO() as result:
        imageio.imwrite(result, image, extension=".png")
        return result.getvalue()


############ Object Related ############
def object_info_to_canonical_cuboid(object_info):
    """
    Do not transform the cuboid to the world coordinate, just return the cuboid in the object coordinate
    """
    size = np.array(object_info["object_lwh"])
    corners_obj = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
    )
    corners_obj = corners_obj * size
    corners_obj = corners_obj - size / 2

    return corners_obj


def object_info_to_cuboid(object_info):
    """

        bbox format:
            h
            ^  w
            | /
            |/
            o -------> l (heading)

       3 ---------------- 0
      /|                 /|
     / |                / |
    2 ---------------- 1  |
    |  |               |  |
    |  7 ------------- |- 4
    | /                | /
    6 ---------------- 5


        Args:
        - object_info: dict, object information, with the following keys:
            - translation: list, x, y, z
            - size: list, l, w, h
            - rotation: list, w, x, y, z

        Returns:
        - corners_world: np.ndarray, shape=(8, 3), the 8 corners of the object in the world coordinate
    """
    try:
        size = np.array(object_info["size"])
    except:
        size = np.array(object_info["object_lwh"])

    corners_obj = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
    )
    corners_obj = corners_obj * size
    corners_obj = corners_obj - size / 2
    # pad 1 for homogeneous coordinates
    corners_obj = np.concatenate([corners_obj, np.ones((8, 1))], axis=1)

    # quaternion to rotation matrix
    object_to_world = object_info_to_object2world(object_info)

    corners_world = np.einsum("ij,kj->ki", object_to_world, corners_obj)[:, :3]

    return corners_world


def object_info_to_object2world(object_info):
    try:
        object_to_world = object_info["object_to_world"]
    except:
        # quaternion to rotation matrix
        T = Quaternion(object_info["rotation"]).transformation_matrix
        T[:3, 3] = object_info["translation"]
        object_to_world = T

    return object_to_world


def get_points_in_cuboid(lidar_points, lidar_to_world, object_info):
    """
    Args:
    - lidar_points: np.ndarray, shape=(N, 3), lidar points in the ego car coordinate
    - lidar_to_world: np.ndarray, shape=(4, 4), the transformation matrix from lidar coordinate to world
    - object_info: dict, object information

    Returns:
    - points_in_cuboid: np.ndarray, shape=(M, 3), the points in the cuboid
    """
    box_l, box_w, box_h = object_info["object_lwh"]

    object_to_world = object_info_to_object2world(object_info)
    world_to_object = np.linalg.inv(object_to_world)

    # transform lidar points to world coordinate then to object coordinate
    lidar_to_object = world_to_object @ lidar_to_world

    # transform lidar points to object coordinate
    lidar_points_padded = np.concatenate(
        [lidar_points, np.ones((lidar_points.shape[0], 1))], axis=1
    )
    points_in_object = np.einsum("ij,kj->ki", lidar_to_object, lidar_points_padded)[
        :, :3
    ]  # shape=(N, 3)

    # keep points in the cuboid
    points_in_cuboid = points_in_object[
        (points_in_object[:, 0] >= -box_l / 2)
        & (points_in_object[:, 0] <= box_l / 2)
        & (points_in_object[:, 1] >= -box_w / 2)
        & (points_in_object[:, 1] <= box_w / 2)
        & (points_in_object[:, 2] >= -box_h / 2)
        & (points_in_object[:, 2] <= box_h / 2)
    ]

    return points_in_cuboid


def classify_static_dynamic_objects(
    all_object_info_sample,
    all_object_points_canonical_data,
    static_object_info_sample,
    dynamic_object_info_sample,
    dynamic_object_points_canonical_data,
):
    """
    Classify objects into static and dynamic based on all frames.
    Only objects that are static in ALL frames are classified as static.

    Args:
        all_object_info_sample: dict, contains all object info for each frame
        all_object_points_canonical_data: dict, contains canonical points data for each object
        static_object_info_sample: dict, will be filled with static objects for each frame
        dynamic_object_info_sample: dict, will be filled with dynamic objects for each frame
        dynamic_object_points_canonical_data: dict, will be filled with canonical points data for dynamic objects
    """
    # Collect all object IDs and their moving status across all frames
    object_ever_moved = {}  # object_id -> bool (True if ever moved)
    all_frames_data = {}  # frame_idx -> frame_data

    # First pass: collect all object IDs and determine if they ever moved
    for key, frame_data in all_object_info_sample.items():
        if key == "__key__":
            continue

        frame_idx = key.split(".")[0]
        all_frames_data[frame_idx] = frame_data

        for object_id, object_info in frame_data.items():
            if object_id not in object_ever_moved:
                object_ever_moved[object_id] = False

            # If this object is moving in this frame, mark it as ever moved
            if object_info["object_is_moving"]:
                object_ever_moved[object_id] = True

    # Separate static and dynamic object IDs
    static_object_ids = sorted([obj_id for obj_id, moved in object_ever_moved.items() if not moved])
    dynamic_object_ids = sorted([obj_id for obj_id, moved in object_ever_moved.items() if moved])
    
    # Create object_id to object_id_int mapping
    # Static objects: counting from 1
    # Dynamic objects: counting from 10000
    object_id_to_int = {}
    for idx, obj_id in enumerate(static_object_ids):
        object_id_to_int[obj_id] = idx + 1  # Start from 1
    for idx, obj_id in enumerate(dynamic_object_ids):
        object_id_to_int[obj_id] = 10000 + idx  # Start from 10000

    # Second pass: classify objects for each frame and add object_id_int
    for frame_idx, frame_data in all_frames_data.items():
        static_objects = {}
        dynamic_objects = {}

        for object_id, object_info in frame_data.items():
            # Add object_id_int to object_info
            object_info["object_id_int"] = object_id_to_int[object_id]

            if object_ever_moved[object_id]:
                # This object moved at some point, so it's dynamic
                dynamic_objects[object_id] = object_info
            else:
                # This object never moved in any frame, so it's static
                static_objects[object_id] = object_info

        static_object_info_sample[f"{frame_idx}.static_object_info.json"] = (
            static_objects
        )
        dynamic_object_info_sample[f"{frame_idx}.dynamic_object_info.json"] = (
            dynamic_objects
        )

    # distribute object info across all frames
    distribute_object_info_across_all_frames(static_object_info_sample)

    # populate dynamic object points canonical data
    for object_id in dynamic_object_ids:
        dynamic_object_points_canonical_data[object_id + "_xyz"] = (
            all_object_points_canonical_data[object_id + "_xyz"]
        )
        dynamic_object_points_canonical_data[object_id + "_semantic"] = (
            all_object_points_canonical_data[object_id + "_semantic"]
        )
        dynamic_object_points_canonical_data[object_id + "_corner"] = (
            all_object_points_canonical_data[object_id + "_corner"]
        )


def keep_car_only_in_object_info(object_info_all_frames):
    """
    Keep only car objects in the object_info_all_frames.

    object_info_all_frames: dict
        {
            "{frame_idx:06d}.xxx_object_info.json": {
                "object_gid1": {
                    "object_to_world": (4, 4),
                    "object_lwh": (l, w, h),
                    "object_is_moving": False,
                    "object_type": "car",
                    "object_id_int": int
                },
            }
        }

    Returns:
        new_object_info_all_frames: dict, with only car objects inside
    """
    new_object_info_all_frames = {}
    for frame_idx_key, frame_object_data in object_info_all_frames.items():
        if frame_idx_key.endswith(".json"):
            new_frame_object_data = {}
            for object_id, object_info in frame_object_data.items():
                if object_info["object_type"] == "car":
                    new_frame_object_data[object_id] = object_info
            new_object_info_all_frames[frame_idx_key] = new_frame_object_data
        else:
            new_object_info_all_frames[frame_idx_key] = frame_object_data

    return new_object_info_all_frames


def distribute_object_info_across_all_frames(object_info_dict):
    """
    Distribute object info across all frames.
    Static or dynamic objects may not appear in all frames (because they requires at
    least 1 hit LiDAR points to be labeled), but we want them to be so.

    So we use this function to distribute static / dynamic object info across all frames.
    We will do two passes:
        1. from the latest to the earliest
        2. from the earliest to the latest

    In each pass, we will distribute the object info to the frames that do not have it.

    Args:
        object_info_dict: dict, can be static or dynamic object info
            {
                "{frame_idx:06d}.xxx_object_info.json": {
                    "object_gid1": {
                        "object_to_world": (4, 4),
                        "object_lwh": (l, w, h),
                        "object_is_moving": False,
                        "object_type": "car",
                        "object_id_int": int
                    },
                    "object_gid2": {
                        "object_to_world": (4, 4),
                        "object_lwh": (l, w, h),
                        "object_is_moving": False,
                        "object_type": "car",
                        "object_id_int": int
                    },
                    ...
                }
            }
    """

    def distribute_forward_pass(ordered_object_info_dict_keys):
        for i in range(1, len(ordered_object_info_dict_keys)):
            current_object_info = object_info_dict[ordered_object_info_dict_keys[i]]
            previous_object_info = object_info_dict[
                ordered_object_info_dict_keys[i - 1]
            ]
            for object_id, object_info in previous_object_info.items():
                if object_id not in current_object_info:
                    current_object_info[object_id] = copy.deepcopy(object_info)

    object_info_dict_keys = list(object_info_dict.keys())
    object_info_dict_keys = [
        key for key in object_info_dict_keys if key.endswith(".json")
    ]

    object_info_dict_keys.sort(reverse=True)  # descending order.
    distribute_forward_pass(object_info_dict_keys)

    object_info_dict_keys.sort()
    distribute_forward_pass(object_info_dict_keys)
