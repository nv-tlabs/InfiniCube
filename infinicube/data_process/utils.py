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

import sys

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from infinicube.utils.interpolate_utils import interpolate_polyline_to_points


############ Segmentation Related ############
def load_mmseg_inferencer():
    """
    Load MMSeg inferencer for sky mask generation.

    Returns:
        MMSegInferencer: The loaded inferencer instance

    Raises:
        SystemExit: If MMCV version is incompatible with mmseg
    """
    try:
        from mmseg.apis import MMSegInferencer
    except AssertionError as e:
        if "MMCV" in str(e) and "incompatible" in str(e):
            logger.error("=" * 80)
            logger.error("MMCV Version Incompatibility Error!")
            logger.error(f"Error message: {e}")
            logger.error("")
            logger.error(
                "Solution: You need to manually modify mmseg's __init__.py file"
            )
            logger.error("")
            logger.error("Please follow these steps:")
            logger.error("1. Locate mmseg's __init__.py file, typically at:")
            logger.error(
                "   $CONDA_PREFIX/lib/python3.10/site-packages/mmseg/__init__.py"
            )
            logger.error("")
            logger.error("2. Modify line 11, change:")
            logger.error("   MMCV_MAX = '2.2.0'")
            logger.error("   to:")
            logger.error("   MMCV_MAX = '2.3.0'")
            logger.error("")
            logger.error("3. Save the file and re-run this script")
            logger.error("=" * 80)
            sys.exit(1)
        else:
            raise

    inferencer = MMSegInferencer(
        model="segformer_mit-b5_8xb1-160k_cityscapes-1024x1024", device="cuda"
    )
    return inferencer


@torch.inference_mode()
def inference_mmseg(video_numpy_list, inferenecer):
    """
    Inference the video numpy list using mmseg

    Args:
        video_numpy_list: list,
            list of numpy array with shape [H, W, 3],
            each numpy array is a frame of the video

        inferencer: MMSegInferencer

    Returns:
        list: list of semantic segmentation results for each frame
    """
    pred_list = []

    # video_numpy_list can be too long for one inference, so we split it into several parts
    chunking_num = 15
    chunk_index = torch.linspace(0, len(video_numpy_list), chunking_num + 1).long()
    print("Inference mmseg...")

    for idx in range(chunking_num):
        # print(f"Processing segformer chunk {idx}/{chunking_num}")
        if chunk_index[idx] == chunk_index[idx + 1]:
            continue
        video_numpy_chunk = video_numpy_list[chunk_index[idx] : chunk_index[idx + 1]]
        semantic_map = inferenecer(video_numpy_chunk)["predictions"]
        if isinstance(semantic_map, np.ndarray):
            semantic_map = [semantic_map]

        pred_list.extend(semantic_map)

    return pred_list


############ Depth Related ############
@torch.inference_mode()
def inference_metric3dv2(
    video_tensor, max_depth=199.9, metric3d_model="metric3d_vit_large", chunking_num=100
):
    """
    Args:
        video_tensor: torch.Tensor, shape (T, C, H, W), normalized to [0, 1]
        max_depth: float, the maximum depth value, used to filter out invalid depth values to be 0. Note that Metric3D itself set it to 200!
        metric3d_model: str, the name of the metric3d model, e.g. metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2

    Returns:
        depth: torch.Tensor, shape (T, 1, H, W)
        normal: torch.Tensor, shape (T, 3, H, W)
    """
    chunk_index = torch.linspace(0, video_tensor.shape[0], chunking_num + 1).long()

    pred_depth_chunks = []
    pred_normal = None

    print(
        f"Inference metric3d with maximum batch size {(chunk_index[1:] - chunk_index[:-1]).max()}"
    )
    model = (
        torch.hub.load("yvanyin/metric3d", metric3d_model, pretrain=True).cuda().eval()
    )  # can not continue with different batchsize

    for idx in tqdm(range(chunking_num)):
        # print(f"Processing metric3d chunk {idx}/{chunking_num}")
        if chunk_index[idx] == chunk_index[idx + 1]:
            continue
        pred_depth, confidence, output_dict = model.inference(
            {"input": video_tensor[chunk_index[idx] : chunk_index[idx + 1]]}
        )
        pred_depth[pred_depth > max_depth] = 0
        pred_depth_chunks.append(pred_depth)  # shape (T, 1, H, W)
        del pred_depth

        del confidence
        del output_dict

    pred_depth = torch.cat(pred_depth_chunks, dim=0)
    pred_depth = torch.nn.functional.interpolate(
        pred_depth, size=video_tensor.shape[2:], mode="bilinear", align_corners=False
    )

    return pred_depth, pred_normal


def align_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    return_scale: bool = False,
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    source_invalid = source_depth == 0
    source_mask = source_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    outlier_quantiles = torch.tensor([0.1, 0.9], device=source_depth.device)

    try:
        source_data_low, source_data_high = torch.quantile(
            source_depth[source_mask], outlier_quantiles
        )
        target_data_low, target_data_high = torch.quantile(
            target_depth[target_mask], outlier_quantiles
        )
        source_mask = (source_depth > source_data_low) & (
            source_depth < source_data_high
        )
        target_mask = (target_depth > target_data_low) & (
            target_depth < target_data_high
        )

        mask = torch.logical_and(source_mask, target_mask)

        source_data = source_depth[mask].view(-1, 1)
        target_data = target_depth[mask].view(-1, 1)

        # TODO: Maybe use RANSAC or M-estimators to make it more robust
        ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
        source_data_h = torch.cat([source_data, ones], dim=1)
        transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

        scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
        aligned_depth = source_depth * scale + bias

        # invalid still invalid
        aligned_depth[source_invalid] = 0

        print(f"Scale: {scale}, Bias: {bias}")

    except Exception:
        if return_scale:
            return 1, 0
        else:
            return source_depth

    if return_scale:
        return scale, bias
    else:
        return aligned_depth


def align_depth_to_depth_batch(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    return_scale: bool = False,
) -> torch.Tensor:
    """
    Apply affine transformation to align source depth to target depth.

    Args:
        source_inv_depth: Depth map to be aligned. Shape: (B, H, W).
        target_depth: Target depth map. Shape: (B, H, W).
        target_mask: Mask of valid target pixels. Shape: (B, H, W).

    Returns:
        Aligned Depth map. Shape: (B, H, W).
    """
    assert return_scale == False, "return_scale is not supported for batch version"

    B = source_depth.shape[0]
    aligned_depth = []
    for i in range(B):
        aligned_depth.append(
            align_depth_to_depth(
                source_depth[i], target_depth[i], target_mask=target_mask[i]
            )
        )

    return torch.stack(aligned_depth)


def project_points_to_depth_image(
    points, points_coordinate_to_camera, camera_intrinsic, width, height
):
    """
    Fast implementation to project the 3D points to image plane to get the depth.

    Args:
        points: torch.Tensor, shape (N, 3), the 3D points
        points_coordinate_to_camera: torch.Tensor, shape (4, 4), the transformation matrix from the points coordinate to the camera coordinate
        camera_intrinsic: torch.Tensor, shape (3, 3), the camera intrinsic matrix

    Returns:
        depth_image: torch.Tensor, shape (H, W)
    """
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1).to(points)], dim=1
    )  # shape (N, 4)
    points_cam = points @ points_coordinate_to_camera.T  # shape (N, 4)
    points_cam = points_cam[:, :3]  # shape (N, 3)

    valid_depth_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth_mask]

    points_depth = points_cam[:, 2]
    points_cam = points_cam[:, :3] / points_cam[:, 2:3]  # shape (N, 3)
    points_uv = points_cam @ camera_intrinsic.T  # shape (N, 3)

    u_round = torch.round(points_uv[:, 0]).long()
    v_round = torch.round(points_uv[:, 1]).long()

    valid_uv_mask = (
        (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)
    )

    u_valid = u_round[valid_uv_mask]
    v_valid = v_round[valid_uv_mask]
    z_valid = points_depth[valid_uv_mask]

    indices = v_valid * width + u_valid

    depth_image = torch.full((height, width), float("inf")).to(points_depth).flatten()
    depth_image = depth_image.scatter_reduce_(0, indices, z_valid, "amin")
    depth_image = depth_image.view(height, width)
    depth_mask = torch.isfinite(depth_image)

    # change inf to 0
    depth_image[~depth_mask] = 0

    return depth_image, depth_mask


############ Road Surface Estimation ############
def estimate_road_surface_in_grid(
    road_edge_full,
    lane_full,
    block_x_idx,
    block_y_idx,
    blocks_x_start,
    blocks_x_end,
    blocks_y_start,
    blocks_y_end,
    voxel_sizes,
    visualize_debug=False,
):
    """
    Estimate road surface in a single grid block.

    Args:
        road_edge_full: np.ndarray, shape (N_1, 3), road_edge points in world coordinates
        lane_full: np.ndarray, shape (N_2, 3), lane points in world coordinates
        block_x_idx: int, block index in x direction
        block_y_idx: int, block index in y direction
        blocks_x_start: np.ndarray, start x coordinates of all blocks
        blocks_x_end: np.ndarray, end x coordinates of all blocks
        blocks_y_start: np.ndarray, start y coordinates of all blocks
        blocks_y_end: np.ndarray, end y coordinates of all blocks
        voxel_sizes: list, voxel sizes [x, y, z] in meters
        visualize_debug: bool, whether to save debug visualization images

    Returns:
        road_surface_points: np.ndarray, shape (N, 3), estimated road surface points
    """
    import random

    from skimage import measure
    from skspatial.objects import Plane

    # retrieve exact block boundary coordinates
    block_x_start, block_x_end = blocks_x_start[block_x_idx], blocks_x_end[block_x_idx]
    block_y_start, block_y_end = blocks_y_start[block_y_idx], blocks_y_end[block_y_idx]

    # get block size
    block_size_x = block_x_end - block_x_start
    block_size_y = block_y_end - block_y_start

    # filter road edge that is out of the grid
    mask = (
        (road_edge_full[:, 0] >= block_x_start)
        & (road_edge_full[:, 0] <= block_x_end)
        & (road_edge_full[:, 1] >= block_y_start)
        & (road_edge_full[:, 1] <= block_y_end)
    )
    road_edge = road_edge_full[mask]

    # filter lane that is out of the grid
    mask = (
        (lane_full[:, 0] >= block_x_start)
        & (lane_full[:, 0] <= block_x_end)
        & (lane_full[:, 1] >= block_y_start)
        & (lane_full[:, 1] <= block_y_end)
    )
    lane = lane_full[mask]

    # expand to 3x3 block, get the road_edge and lane for distance comparison
    mask = (
        (road_edge_full[:, 0] >= block_x_start - voxel_sizes[0])
        & (road_edge_full[:, 0] <= block_x_end + voxel_sizes[0])
        & (road_edge_full[:, 1] >= block_y_start - voxel_sizes[1])
        & (road_edge_full[:, 1] <= block_y_end + voxel_sizes[1])
    )
    road_edge_full = road_edge_full[mask]

    mask = (
        (lane_full[:, 0] >= block_x_start - voxel_sizes[0])
        & (lane_full[:, 0] <= block_x_end + voxel_sizes[0])
        & (lane_full[:, 1] >= block_y_start - voxel_sizes[1])
        & (lane_full[:, 1] <= block_y_end + voxel_sizes[1])
    )
    lane_full = lane_full[mask]

    # if points are too few, we will not estimate the road surface for that block
    if road_edge.shape[0] < 3 or lane.shape[0] < 3:
        print(f"block {block_x_idx}_{block_y_idx} has too few points")
        return np.zeros((0, 3))

    # create a BEV grid
    bev_w = round(block_size_x / voxel_sizes[0])
    bev_h = round(block_size_y / voxel_sizes[1])

    bev_rasterize_map = np.zeros((bev_h, bev_w), dtype=np.uint8)

    # road edge points is dense, we can use it to rasterize the road edge
    road_edge_u = ((road_edge[:, 0] - block_x_start) // voxel_sizes[0]).astype(int)
    road_edge_v = ((road_edge[:, 1] - block_y_start) // voxel_sizes[1]).astype(int)
    road_edge_uv = np.stack([road_edge_u, road_edge_v], axis=1)
    road_edge_uv = np.unique(road_edge_uv, axis=0)
    road_edge_u, road_edge_v = road_edge_uv[:, 0], road_edge_uv[:, 1]

    # also for lane points
    lane_u = ((lane[:, 0] - block_x_start) // voxel_sizes[0]).astype(int)
    lane_v = ((lane[:, 1] - block_y_start) // voxel_sizes[1]).astype(int)
    lane_uv = np.stack([lane_u, lane_v], axis=1)
    lane_uv = np.unique(lane_uv, axis=0)
    lane_u, lane_v = lane_uv[:, 0], lane_uv[:, 1]

    # draw road edge
    bev_rasterize_map[road_edge_v, road_edge_u] = (
        255  # road edge is 255 while the road surface will be 0
    )

    # find the connected connected components
    background_value = 255
    cc_image = measure.label(
        bev_rasterize_map, background=background_value, connectivity=1
    )

    # find the connected component that contains the lanes
    lane_cc = cc_image[lane_v, lane_u]
    lane_cc_unique = np.unique(lane_cc)
    lane_cc_mask = np.isin(cc_image, lane_cc_unique)

    # if lane_cc_mask is too large (>70% of the whole map), possibly it is near the boundary.
    # we create a convex hull to estimate the road surface
    if lane_cc_mask.mean() > 0.7:
        print(f"block {block_x_idx}_{block_y_idx} has too large lane_cc_mask")
        # we further subdivde the block into sub-blocks, keep those has lane_uv and road_edge_uv inside the sub-block
        subdivide_num = 4
        sub_bev_w_idx = np.linspace(0, bev_w, subdivide_num + 1).astype(int)
        sub_bev_h_idx = np.linspace(0, bev_h, subdivide_num + 1).astype(int)

        for j in range(subdivide_num):
            for i in range(subdivide_num):
                lane_uv_in_sub_block = (
                    (lane_u >= sub_bev_w_idx[i])
                    & (lane_u < sub_bev_w_idx[i + 1])
                    & (lane_v >= sub_bev_h_idx[j])
                    & (lane_v < sub_bev_h_idx[j + 1])
                )
                road_edge_uv_in_sub_block = (
                    (road_edge_u >= sub_bev_w_idx[i])
                    & (road_edge_u < sub_bev_w_idx[i + 1])
                    & (road_edge_v >= sub_bev_h_idx[j])
                    & (road_edge_v < sub_bev_h_idx[j + 1])
                )

                if (
                    lane_uv_in_sub_block.sum() > 0
                    or road_edge_uv_in_sub_block.sum() > 0
                ) and lane_cc_mask[
                    sub_bev_h_idx[j] : sub_bev_h_idx[j + 1],
                    sub_bev_w_idx[i] : sub_bev_w_idx[i + 1],
                ].any():
                    # we can keep this sub block, but we need more fine-grained mask.
                    # criteria: valid points must have nearer distance to (full) lane points than to (full) road edge points
                    sub_block_v = np.arange(sub_bev_h_idx[j], sub_bev_h_idx[j + 1])
                    sub_block_u = np.arange(sub_bev_w_idx[i], sub_bev_w_idx[i + 1])

                    sub_block_x = sub_block_u * voxel_sizes[0] + block_x_start
                    sub_block_y = sub_block_v * voxel_sizes[1] + block_y_start
                    sub_block_xy = np.stack(
                        np.meshgrid(sub_block_x, sub_block_y), axis=-1
                    )  # [sub_bev_h, sub_bev_w, 2], it stores xy
                    sub_block_xy = sub_block_xy.reshape(-1, 2)[np.newaxis, :, :]

                    # calculate minimum distance to lane points
                    lane_xy = lane_full[:, :2]
                    lane_xy = lane_xy[:, np.newaxis, :]

                    lane_dist = np.linalg.norm(sub_block_xy - lane_xy, axis=-1).min(
                        axis=0
                    )

                    # calculate minimum distance to road edge points
                    road_edge_xy = road_edge_full[:, :2]
                    road_edge_xy = road_edge_xy[:, np.newaxis, :]
                    road_edge_dist = np.linalg.norm(
                        sub_block_xy - road_edge_xy, axis=-1
                    ).min(axis=0)

                    # mask
                    mask = lane_dist < road_edge_dist
                    mask = mask.reshape(
                        sub_bev_h_idx[j + 1] - sub_bev_h_idx[j],
                        sub_bev_w_idx[i + 1] - sub_bev_w_idx[i],
                    )

                    # keep the points that are closer to lane points
                    lane_cc_mask[
                        sub_bev_h_idx[j] : sub_bev_h_idx[j + 1],
                        sub_bev_w_idx[i] : sub_bev_w_idx[i + 1],
                    ] &= mask

                else:
                    lane_cc_mask[
                        sub_bev_h_idx[j] : sub_bev_h_idx[j + 1],
                        sub_bev_w_idx[i] : sub_bev_w_idx[i + 1],
                    ] = False

    random_road_edge_pts = random.sample(
        range(road_edge.shape[0]), min(1500, road_edge.shape[0])
    )
    random_road_edge_pts = road_edge[random_road_edge_pts]
    random_lane_pts = random.sample(range(lane.shape[0]), min(1500, lane.shape[0]))
    random_lane_pts = lane[random_lane_pts]

    random_sample = np.concatenate([random_road_edge_pts, random_lane_pts], axis=0)
    plane = Plane.best_fit(random_sample)
    a, b, c, d = plane.cartesian()  # ax + by + cz + d = 0

    # use the 2D BEV mask ego_cc_mask, to sample the 3D points on the plane
    vv, uu = np.where(lane_cc_mask)
    # 2D to 3D transformationm, get z one the plane
    x = uu * voxel_sizes[0] + block_x_start
    y = vv * voxel_sizes[1] + block_y_start
    z = -(a * x + b * y + d) / c

    road_surface_points = np.stack([x, y, z], axis=1)  # shape (N, 3)

    return road_surface_points


def estimate_road_surface_in_world(
    road_edge, lane, block_size=[40, 40], voxel_sizes=[0.4, 0.4, 0.2]
):
    """
    Estimate road surface in world coordinates using lane and road edge.
    Divides the map into blocks and estimates the 3D plane for each block separately.

    Args:
        road_edge: np.ndarray, shape (N_1, 3), 3D road_edge points in world coordinates
        lane: np.ndarray, shape (N_2, 3), 3D lane points in world coordinates
        block_size: list, block size [x, y] in meters
        voxel_sizes: list, voxel sizes [x, y, z] in meters

    Returns:
        road_surface_points: np.ndarray, shape (N, 3), estimated road surface points
    """
    from math import ceil

    # Step 1: Divide the map into blocks.
    map_x_min, map_x_max, map_y_min, map_y_max = (
        lane[:, 0].min(),
        lane[:, 0].max(),
        lane[:, 1].min(),
        lane[:, 1].max(),
    )

    block_x_num = ceil((map_x_max - map_x_min) / block_size[0])
    block_y_num = ceil((map_y_max - map_y_min) / block_size[1])

    print(f"block_x_num: {block_x_num}, block_y_num: {block_y_num}")

    blocks_x_start = map_x_min + np.arange(block_x_num) * block_size[0]
    blocks_y_start = map_y_min + np.arange(block_y_num) * block_size[1]
    blocks_x_end = blocks_x_start + block_size[0]
    blocks_y_end = blocks_y_start + block_size[1]

    # not all blocks have lane points / road edge points inside, if not, we will not estimate the road surface for that block
    valid_mask = np.zeros((block_y_num, block_x_num), dtype=bool)
    for j in range(block_y_num):
        for i in range(block_x_num):
            mask_lane = (
                (lane[:, 0] >= blocks_x_start[i])
                & (lane[:, 0] < blocks_x_end[i])
                & (lane[:, 1] >= blocks_y_start[j])
                & (lane[:, 1] < blocks_y_end[j])
            )
            mask_road_edge = (
                (road_edge[:, 0] >= blocks_x_start[i])
                & (road_edge[:, 0] < blocks_x_end[i])
                & (road_edge[:, 1] >= blocks_y_start[j])
                & (road_edge[:, 1] < blocks_y_end[j])
            )
            if mask_lane.sum() > 0 and mask_road_edge.sum() > 0:
                valid_mask[j, i] = True

    print(f"valid blocks: {valid_mask.sum()}")

    # Step 2: Estimate the road surface for each block
    road_surface_points = []
    for j in range(block_y_num):
        for i in range(block_x_num):
            if not valid_mask[j, i]:
                continue

            road_surface_points_block = estimate_road_surface_in_grid(
                road_edge,
                lane,
                i,
                j,
                blocks_x_start,
                blocks_x_end,
                blocks_y_start,
                blocks_y_end,
                voxel_sizes,
            )
            road_surface_points.append(road_surface_points_block)

    road_surface_points = np.concatenate(road_surface_points, axis=0)

    return road_surface_points


def polylines_to_discrete_points(polylines, segment_interval):
    """
    Args:
        polylines: list of polylines, each polyline is a list of points (x, y, z)
        segment_interval: float, segment interval

    Returns:
        points: numpy.ndarray, shape (N, 3) discrete points
    """
    points = []
    for polyline in polylines:
        points.extend(interpolate_polyline_to_points(polyline, segment_interval))
    return np.array(points)


def add_interpolated_maps_to_voxel(
    map_name_to_polylines,
    map_name_to_semantic,
    interpolate_voxel_size=0.025,
):
    """
    Args:
        map_name_to_polylines: dict, map name to polylines, each polyline is a list of points (x, y, z)
        map_name_to_semantic_category: dict, map name to semantic category
        interpolate_voxel_size: float, voxel size for interpolation

    Returns:
        voxel_points_with_map_points: torch.Tensor, shape (M, 3), dtype=torch.float32, points in voxel with map
        voxel_semantics_with_map_points: torch.Tensor, shape (M, ), dtype=int, semantics of the points in voxel with map
    """
    map_names1 = list(map_name_to_polylines.keys())
    map_names2 = list(map_name_to_semantic.keys())
    assert map_names1 == map_names2, (
        "map names in map_name_to_polylines and map_name_to_semantics must be the same"
    )
    map_names = map_names1

    map_interpolated_points = []
    map_interpolated_semantics = []

    for map_name in map_names:
        polylines = map_name_to_polylines[
            map_name
        ]  # list of polylines, each polyline is a list of points (x, y, z)
        semantic = map_name_to_semantic[map_name]  # int, semantic category

        # Interpolate each polyline separately and concatenate
        polyline_interpolated_points_list = []
        for polyline in polylines:
            polyline_interpolated_points_list.append(
                interpolate_polyline_to_points(polyline, interpolate_voxel_size)
            )
        polyline_interpolated_points = (
            np.concatenate(polyline_interpolated_points_list, axis=0)
            if polyline_interpolated_points_list
            else np.empty((0, 3))
        )

        # augment polyline_interpolated_points with z-axis
        z_shifts = [0, -interpolate_voxel_size, interpolate_voxel_size]
        polyline_interpolated_points_aug = np.concatenate(
            [
                polyline_interpolated_points + np.array([0, 0, z_shift])
                for z_shift in z_shifts
            ],
            axis=0,
        )

        map_interpolated_points.append(polyline_interpolated_points_aug)
        map_interpolated_semantics.append(
            np.ones((polyline_interpolated_points_aug.shape[0],), dtype=np.int32)
            * semantic
        )

    map_interpolated_points = np.concatenate(map_interpolated_points, axis=0)
    map_interpolated_semantics = np.concatenate(map_interpolated_semantics, axis=0)

    return map_interpolated_points, map_interpolated_semantics
