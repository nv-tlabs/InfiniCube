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


def vis_depth(depth, minmax=None, valid_farthest=300):
    """
    Args:
        depth: np.ndarray or torch.Tensor,
            shape (H, W)

        minmax:
            if None, use adaptive minmax according to the depth values
            if [d_min, d_max], use the provided minmax

        valid_farthest: float
            the farthest valid depth value, used to filter out invalid depth values like inf

        cmap:
            https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            e.g. cv2.COLORMAP_JET

    Returns:
        colored_depth: (H, W, 3), uint8
    """
    import matplotlib as mpl
    from matplotlib import cm

    is_tensor = False
    if isinstance(depth, torch.Tensor):
        device = depth.device
        depth = depth.detach().cpu().numpy()
        is_tensor = True

    depth = np.nan_to_num(depth)  # change nan to 0
    depth_valid_count = (depth < valid_farthest).sum()

    if minmax is None:
        constant_max = np.percentile(depth[depth < valid_farthest], 99.5)
        constant_min = (
            np.percentile(depth, 0.5) if np.percentile(depth, 0.5) < constant_max else 0
        )
    else:
        constant_min, constant_max = minmax

    normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="magma_r")

    colored_depth = mapper.to_rgba(depth)[:, :, :3]  # range [0, 1]
    colored_depth = (colored_depth * 255).astype(np.uint8)  # range [0, 255]

    if is_tensor:
        colored_depth = torch.from_numpy(colored_depth).to(device)

    return colored_depth


def least_square_fit(
    relative_dense_depth, absolute_sparse_depth, sparse_depth_mask, return_scale=False
):
    """
    Args:
        relative_dense_depth: torch.Tensor, shape (H, W), the relative depth map
        absolute_sparse_depth: torch.Tensor, shape (H, W), the absolute depth map
        sparse_depth_mask: torch.Tensor, shape (H, W), the mask of the sparse depth map

    Returns:
        absolute_dense_depth: torch.Tensor, shape (H, W), the absolute depth map

    Note that if 3 inputs have the same shape, they can not be directly passed to this function.
    """
    relative_depths = relative_dense_depth[sparse_depth_mask]
    absolute_depths = absolute_sparse_depth[sparse_depth_mask]

    # only keep quantile 0.1 to 0.9
    relative_depths_min, relative_depths_max = torch.quantile(
        relative_depths, torch.tensor([0.1, 0.9]).to(relative_depths)
    )
    absolute_depths_min, absolute_depths_max = torch.quantile(
        absolute_depths, torch.tensor([0.1, 0.9]).to(absolute_depths)
    )

    relative_depths_mask = (relative_depths >= relative_depths_min) & (
        relative_depths <= relative_depths_max
    )
    absolute_depths_mask = (absolute_depths >= absolute_depths_min) & (
        absolute_depths <= absolute_depths_max
    )

    quantile_mask = relative_depths_mask & absolute_depths_mask

    relative_depths = relative_depths[quantile_mask]
    absolute_depths = absolute_depths[quantile_mask]

    numerator = torch.sum(relative_depths * absolute_depths)
    denominator = torch.sum(relative_depths**2)

    if denominator == 0:
        raise ValueError(
            "The denominator is zero, cannot perform the least square fit."
        )

    scale = numerator / denominator
    print(f"Rescaling factor for least square fit: {scale}")

    # apply scale to the relative depth
    absolute_dense_depth = relative_dense_depth * scale

    if return_scale:
        return scale
    else:
        return absolute_dense_depth


def least_square_fit_np(
    relative_dense_depth, absolute_sparse_depth, sparse_depth_mask, return_scale=False
):
    """
    Args:
        relative_dense_depth: np.ndarray, shape (H, W), the relative depth map
        absolute_sparse_depth: np.ndarray, shape (H, W), the absolute depth map
        sparse_depth_mask: np.ndarray, shape (H, W), the mask of the sparse depth map

    Returns:
        absolute_dense_depth: np.ndarray, shape (H, W), the absolute depth map

    Note that if 3 inputs have the same shape, they can not be directly passed to this function.
    """
    relative_depths = relative_dense_depth[sparse_depth_mask]
    absolute_depths = absolute_sparse_depth[sparse_depth_mask]

    try:
        # only keep quantile 0.1 to 0.9
        relative_depths_min, relative_depths_max = np.quantile(
            relative_depths, [0.1, 0.9]
        )
        absolute_depths_min, absolute_depths_max = np.quantile(
            absolute_depths, [0.1, 0.9]
        )

        relative_depths_mask = (relative_depths >= relative_depths_min) & (
            relative_depths <= relative_depths_max
        )
        absolute_depths_mask = (absolute_depths >= absolute_depths_min) & (
            absolute_depths <= absolute_depths_max
        )

        quantile_mask = relative_depths_mask & absolute_depths_mask
    except:
        # if there is any error, use the whole data
        quantile_mask = np.ones_like(relative_depths).astype(bool)

    relative_depths = relative_depths[quantile_mask]
    absolute_depths = absolute_depths[quantile_mask]

    numerator = np.sum(relative_depths * absolute_depths)
    denominator = np.sum(relative_depths**2)

    if denominator == 0:
        raise ValueError(
            "The denominator is zero, cannot perform the least square fit."
        )

    scale = numerator / denominator
    print(f"Rescaling factor for least square fit: {scale}")

    # apply scale to the relative depth
    absolute_dense_depth = relative_dense_depth * scale

    if return_scale:
        return scale
    else:
        return absolute_dense_depth


def least_square_fit_batch(
    relative_dense_depth, absolute_sparse_depth, sparse_depth_mask, return_scale=False
):
    """
    Calculate the scaling factor to align the relative depth map to the absolute depth map for each sample in the batch seperately.
    Note that sparse_depth_mask is not the same of each sample in the batch.

    Args:
        relative_dense_depth: torch.Tensor, shape (B, H, W), the relative depth map
        absolute_sparse_depth: torch.Tensor, shape (B, H, W), the absolute depth map
        sparse_depth_mask: torch.Tensor, shape (B, H, W), the mask of the sparse depth map

    Returns:
        absolute_dense_depth: torch.Tensor, shape (B, H, W), the absolute depth map
    """
    from joblib import Parallel, delayed

    relative_dense_depth = relative_dense_depth.cpu().numpy()
    absolute_sparse_depth = absolute_sparse_depth.cpu().numpy()
    sparse_depth_mask = sparse_depth_mask.cpu().numpy()

    # it is very slow for pytorch. use numpy instead
    results = Parallel(n_jobs=16)(
        delayed(least_square_fit_np)(
            relative_dense_depth[i],
            absolute_sparse_depth[i],
            sparse_depth_mask[i],
            return_scale,
        )
        for i in range(relative_dense_depth.shape[0])
    )

    return torch.from_numpy(np.stack(results))


def align_depth_to_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor = None,
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
    target_mask: torch.Tensor = None,
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


def align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Apply affine transformation to align source inverse depth to target depth.
    https://github.com/LiheYoung/Depth-Anything/issues/72

    true depth = 1 / (A + B * depth_anything_result)

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    outlier_quantiles = torch.tensor(
        [0.1, 0.9], device=source_inv_depth.device, dtype=source_inv_depth.dtype
    )
    try:
        source_data_low, source_data_high = torch.quantile(
            source_inv_depth[source_mask], outlier_quantiles
        )
        source_mask = (source_inv_depth > source_data_low) & (
            source_inv_depth < source_data_high
        )
    except:
        source_mask = torch.ones_like(source_inv_depth, dtype=torch.bool)

    try:
        target_data_low, target_data_high = torch.quantile(
            target_inv_depth[target_mask], outlier_quantiles
        )
        target_mask = (target_inv_depth > target_data_low) & (
            target_inv_depth < target_data_high
        )
    except:
        target_mask = torch.ones_like(target_inv_depth, dtype=torch.bool)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    # TODO: Maybe use RANSAC or M-estimators to make it more robust
    ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)
    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth * scale + bias

    aligned_depth = 1.0 / aligned_inv_depth

    # make < 0 to 0
    aligned_depth[aligned_depth < 0] = 0

    return aligned_depth


def unproject_depth_torch(
    depth: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Unproject depth map to 3D points in world coordinate.
    Compatible with InfiniCube camera convention (OpenCV convention).

    Args:
        depth: torch.Tensor, shape (B, 1, H, W) or (B, H, W), depth map
        camera_to_world: torch.Tensor, shape (B, 4, 4), camera to world transformation
        intrinsic_matrix: torch.Tensor, shape (B, 3, 3), camera intrinsic matrix

    Returns:
        points_world: torch.Tensor, shape (B, H, W, 3), 3D points in world coordinate
    """
    # Handle both (B, 1, H, W) and (B, H, W) depth formats
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

    b, c, h, w = depth.shape
    assert depth.shape == (b, 1, h, w), (
        f"Expected depth shape (B, 1, H, W), got {depth.shape}"
    )
    assert camera_to_world.shape == (b, 4, 4), (
        f"Expected camera_to_world shape (B, 4, 4), got {camera_to_world.shape}"
    )
    assert intrinsic_matrix.shape == (b, 3, 3), (
        f"Expected intrinsic_matrix shape (B, 3, 3), got {intrinsic_matrix.shape}"
    )

    device = depth.device

    # Create pixel coordinates
    x1d = torch.arange(0, w, device=device)[None]  # (1, W)
    y1d = torch.arange(0, h, device=device)[:, None]  # (H, 1)
    x2d = x1d.repeat([h, 1])  # (H, W)
    y2d = y1d.repeat([1, w])  # (H, W)
    ones_2d = torch.ones(size=(h, w), device=device)  # (H, W)
    ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (B, H, W, 1, 1)
    pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
        None, :, :, :, None
    ]  # (1, H, W, 3, 1)

    # Compute inverse intrinsics
    intrinsic_inv = torch.inverse(intrinsic_matrix)  # (B, 3, 3)
    intrinsic_inv_4d = intrinsic_inv[:, None, None]  # (B, 1, 1, 3, 3)

    depth_4d = depth[:, 0][:, :, :, None, None]  # (B, H, W, 1, 1)

    # Unproject to camera space
    unnormalized_pos = torch.matmul(
        intrinsic_inv_4d, pos_vectors_homo
    )  # (B, H, W, 3, 1)
    camera_points = depth_4d * unnormalized_pos  # (B, H, W, 3, 1)

    # Transform to world space
    camera_points_homo = torch.cat([camera_points, ones_4d], dim=3)  # (B, H, W, 4, 1)
    trans_4d = camera_to_world[:, None, None]  # (B, 1, 1, 4, 4)
    world_points_homo = torch.matmul(trans_4d, camera_points_homo)  # (B, H, W, 4, 1)
    world_points = world_points_homo[:, :, :, :3]  # (B, H, W, 3, 1)
    world_points = world_points.squeeze(dim=-1)  # (B, H, W, 3)

    return world_points


def normalize_pointmap_pytorch(
    points: torch.Tensor,
    ref_points: torch.Tensor,
    max_distance: float = 150,
    final_scale_clip: float = 2.0,
    percentile: float = 0.01,
) -> torch.Tensor:
    """
    Normalize a point map by scaling its Z-axis based on a reference point map,
    then clips the result within a spherical radius. X and Y axes are not translated or scaled.

    Args:
        points: torch.Tensor, shape (N, 3) or (B, H, W, 3), point map to normalize.
        ref_points: torch.Tensor, shape (M, 3), reference point map to compute normalization stats.
        max_distance: float, maximum distance from origin for reference points to be considered valid.
        final_scale_clip: float, the radius of the sphere to which the final points are clipped.
        percentile: float, percentile used to compute a robust Z-range from reference points.

    Returns:
        norm_points: torch.Tensor, same shape as input 'points', with normalized Z and clipped coordinates.
    """
    EPS = 1e-7

    # --- 1. Handle Input Shape ---
    # Reshape (B, H, W, 3) to (N, 3) for vectorized operations, but keep original shape for final output.
    original_shape = points.shape
    if points.dim() == 4:
        points_flat = points.reshape(-1, 3)
    else:
        points_flat = points

    # --- 2. Calculate Z-Axis Normalization Parameters from Reference Points ---
    # Filter reference points to include only those within a specified distance from the origin.
    ref_distances = torch.linalg.norm(ref_points, dim=-1)
    valid_ref_mask = ref_distances <= max_distance

    if not torch.any(valid_ref_mask):
        print(
            "==> Warning: No valid reference points within max_distance. Skipping Z-normalization."
        )
        # If no valid points, proceed directly to clipping with original data.
        norm_points_flat = points_flat
    else:
        # Use the valid points to compute a robust Z-range using percentiles.
        usable_ref_points = ref_points[valid_ref_mask]
        z_coords_ref = usable_ref_points[:, 2]
        source_z_min = torch.quantile(z_coords_ref, percentile)
        source_z_max = torch.quantile(z_coords_ref, 1 - percentile)
        source_z_range = max(source_z_max - source_z_min, EPS)

        # --- 3. Apply Z-Axis Normalization ---
        # Define the target Z-range for normalization.
        TARGET_Z_MIN = -0.8
        TARGET_Z_MAX = 0.8
        target_z_range = TARGET_Z_MAX - TARGET_Z_MIN

        # Linearly map the Z-coordinates of all input points from the source range to the target range.
        # Formula: new_z = ((old_z - old_min) / old_range) * new_range + new_min
        points_z = points_flat[:, 2]
        normalized_z = (
            (points_z - source_z_min) / source_z_range
        ) * target_z_range + TARGET_Z_MIN

        # Reconstruct the points with the newly normalized Z-axis.
        norm_points_flat = points_flat.clone()
        norm_points_flat[:, 2] = normalized_z

    # --- 4. Apply Spherical Clipping ---
    # Any point outside a sphere of radius `final_scale_clip` is pulled back onto its surface.
    distances = torch.linalg.norm(norm_points_flat, dim=-1)
    clip_mask = distances > final_scale_clip

    if torch.any(clip_mask):
        points_to_clip = norm_points_flat[clip_mask]
        distances_to_clip = distances[clip_mask]

        # Calculate the scaling factor to bring points onto the sphere's surface.
        scale_factor = final_scale_clip / (distances_to_clip + EPS)
        scaled_points = points_to_clip * scale_factor.unsqueeze(-1)
        norm_points_flat[clip_mask] = scaled_points

    # --- 5. Restore Original Shape ---
    # If the input was a 4D tensor, reshape the output back to its original dimensions.
    if len(original_shape) == 4:
        norm_points = norm_points_flat.reshape(original_shape)
    else:
        norm_points = norm_points_flat

    return norm_points
