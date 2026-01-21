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


def create_bbox_line_segments(corners):
    """
    Convert 8 corners of a bounding box to line segments.

    Args:
        corners: np.ndarray, shape (8, 3), the 8 corners of the bbox

    Returns:
        points: np.ndarray, shape (12, 2, 3), line segments for the 12 edges of the bbox
    """
    # Define the 12 edges of a bounding box
    # Bottom 4 edges, Top 4 edges, Vertical 4 edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    # Build line segments array
    line_segments = []
    for start_idx, end_idx in edges:
        line_segments.append([corners[start_idx], corners[end_idx]])

    return np.array(line_segments)
