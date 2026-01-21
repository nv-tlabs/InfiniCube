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

from abc import abstractmethod
from typing import List, Union

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

from infinicube.utils.dtype_utils import make_sure_numpy, make_sure_torch
from infinicube.utils.interpolate_utils import interpolate_polyline_to_points


def opencv_to_flu(
    camera_pose: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert camera pose from opencv convention to flu convention

    (opencv)
      z
     /
    o ---->x
    |
    v y

    (FLU)
            z
            |  x
            | /
    y <---- o

    Args:
        camera_pose: (N, 4, 4) or (4, 4)
    Returns:
        camera_pose: (N, 4, 4) or (4, 4)
    """
    if isinstance(camera_pose, np.ndarray):
        return np.concatenate(
            [
                camera_pose[..., 2:3],
                -camera_pose[..., 0:1],
                -camera_pose[..., 1:2],
                camera_pose[..., 3:4],
            ],
            axis=-1,
        )
    else:
        return torch.cat(
            [
                camera_pose[..., 2:3],
                -camera_pose[..., 0:1],
                -camera_pose[..., 1:2],
                camera_pose[..., 3:4],
            ],
            dim=-1,
        )


def flu_to_opencv(
    camera_pose: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    (opencv)
      z
     /
    o ---->x
    |
    v y

    (FLU)
            z
            |  x
            | /
    y <---- o

    Args:
        camera_pose: (N, 4, 4) or (4, 4)
    Returns:
        camera_pose: (N, 4, 4) or (4, 4)
    """
    if isinstance(camera_pose, np.ndarray):
        return np.concatenate(
            [
                -camera_pose[..., 1:2],
                -camera_pose[..., 2:3],
                camera_pose[..., 0:1],
                camera_pose[..., 3:4],
            ],
            axis=-1,
        )
    else:
        return torch.cat(
            [
                -camera_pose[..., 1:2],
                -camera_pose[..., 2:3],
                camera_pose[..., 0:1],
                camera_pose[..., 3:4],
            ],
            dim=-1,
        )


class CameraBase:
    def __init__(self):
        pass

    @abstractmethod
    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pixel2ray_torch(self, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def pixel2ray_np(self, pixels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ray2pixel(
        self, rays: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            rays: (M, 3), camera rays in camera coordinate (opencv convention)
        Returns:
            pixel_coords: (M, 2), pixel coordinates, not normalized

             z (front)
            /
            o ------> x (right)
            |
            v y (down)
        """
        if isinstance(rays, torch.Tensor):
            return self.ray2pixel_torch(rays)
        else:
            return self.ray2pixel_np(rays)

    def pixel2ray(
        self, pixels: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            pixels: (M, 2), pixel coordinates, not normalized to (0, 1)
        Returns:
            rays: (M, 3), camera rays in camera coordinate (opencv convention)

             z (front)
            /
            o ------> x (right)
            |
            v y (down)
        """
        if isinstance(pixels, torch.Tensor):
            return self.pixel2ray_torch(pixels)
        else:
            return self.pixel2ray_np(pixels)

    def pixel2uv(
        self, pixels: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            pixels: (M, 2), pixel coordinates, not normalized to (0, 1)
        Returns:
            uv_coords: (M, 2), pixel coordinates, normalized to (0, 1)
        """
        pixels_normalized = (
            pixels.copy() if isinstance(pixels, np.ndarray) else pixels.clone()
        )
        pixels_normalized[:, 0] = pixels[:, 0] / self.width
        pixels_normalized[:, 1] = pixels[:, 1] / self.height
        return pixels_normalized

    @abstractmethod
    def _get_rays_impl(self) -> torch.Tensor:
        raise NotImplementedError

    def get_rays(self) -> torch.Tensor:
        """
        Returns:
            rays: (H, W, 3), normalized camera rays camera coordinate (opencv convention)
        """
        if not hasattr(self, "rays_cached"):
            self.rays_cached = self._get_rays_impl()
        return self.rays_cached

    def get_rays_posed(self, camera_poses: torch.Tensor):
        """
        Args:
            camera_poses: (N, 4, 4)
        Returns:
            ray_o: (N, H, W, 3), camera origin
            ray_d: (N, H, W, 3), camera rays
        """
        rays_in_cam = self.get_rays()  # shape (H, W, 3)
        rays_d_in_world = torch.einsum(
            "bij,hwj->bhwi", camera_poses[:, :3, :3], rays_in_cam
        )  # shape (N, H, W, 3)
        rays_o_in_world = (
            camera_poses[:, :3, 3]
            .unsqueeze(-2)
            .unsqueeze(-2)
            .expand_as(rays_d_in_world)
        )  # shape (N, H, W, 3)

        return rays_o_in_world, rays_d_in_world

    @staticmethod
    def transform_points_torch(points: torch.Tensor, tfm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].unsqueeze(-1)
        return transformed_points.T

    @staticmethod
    def transform_points_np(points: np.ndarray, tfm: np.ndarray) -> np.ndarray:
        """
        Args:
            points: (M, 3)
            tfm: (4, 4)
        Returns:
            points_transformed: (M, 3)
        """
        transformed_points = tfm[:3, :3] @ points.T + tfm[:3, 3].reshape(-1, 1)
        return transformed_points.T

    @staticmethod
    def transform_points(
        points: Union[torch.Tensor, np.ndarray], tfm: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        # assert points and tfm are the same type
        assert isinstance(points, type(tfm)), (
            f"points and tfm must be the same type, but got {type(points)} and {type(tfm)}"
        )

        if isinstance(points, torch.Tensor):
            return CameraBase.transform_points_torch(points, tfm)
        else:
            return CameraBase.transform_points_np(points, tfm)

    @staticmethod
    def _clip_polyline_to_image_plane(
        points_in_cam: np.ndarray, eps: float = 5e-2
    ) -> np.ndarray:
        """
        Args:
            points_in_cam: np.ndarray
                shape: (M, 3), a polyline, they are connected.
        Returns:
            points: np.ndarray
                shape: (M', 3), a polyline, but we clip the points to positive z if the points are behind the camera.
        """
        depth = points_in_cam[:, 2]
        # go through all the edges of the polyline.

        cam_coords_cliped = []
        for i in range(len(points_in_cam) - 1):
            pt1 = points_in_cam[i]
            pt2 = points_in_cam[i + 1]

            if depth[i] >= 0 and depth[i + 1] >= 0:
                cam_coords_cliped.append(pt1)
            elif depth[i] < 0 and depth[i + 1] < 0:
                continue
            else:
                # clip the line to the image boundary
                if depth[i] >= 0:
                    # add the first point
                    cam_coords_cliped.append(pt1)

                    # calculate the intersection point and add it
                    t = (-pt2[2]) / (pt1[2] - pt2[2]) + eps
                    inter_pt = pt2 + t * (pt1 - pt2)
                    cam_coords_cliped.append(inter_pt)
                else:
                    # calculate the intersection point and add it
                    t = (-pt1[2]) / (pt2[2] - pt1[2]) + eps
                    inter_pt = pt1 + t * (pt2 - pt1)
                    cam_coords_cliped.append(inter_pt)

        # handle the last point, if its depth > 0 and not already added
        if depth[-1] >= 0:
            cam_coords_cliped.append(points_in_cam[-1])

        cam_coords_cliped = np.stack(cam_coords_cliped, axis=0)  # shape (M', 3)

        return cam_coords_cliped

    def get_xy_and_depth(
        self,
        points_in_world: Union[torch.Tensor, np.ndarray],
        camera_pose: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            points_in_world: (M, 3)
            camera_pose: (4, 4)
        Returns:
            xy_and_depth: (M', 3)
        """
        assert isinstance(points_in_world, type(camera_pose)), (
            f"points_in_world and camera_pose must be the same type, but got {type(points_in_world)} and {type(camera_pose)}"
        )

        if isinstance(camera_pose, torch.Tensor):
            world_to_camera = torch.linalg.inv(camera_pose)
        else:
            world_to_camera = np.linalg.inv(camera_pose)

        points_in_cam = CameraBase.transform_points(points_in_world, world_to_camera)
        depth = points_in_cam[:, 2:3]
        xy = self.ray2pixel(points_in_cam)

        if isinstance(points_in_world, torch.Tensor):
            xy_and_depth = torch.cat([xy, depth.unsqueeze(-1)], dim=-1)
        else:
            xy_and_depth = np.concatenate([xy, depth.reshape(-1, 1)], axis=-1)

        return xy_and_depth

    """
    Projection related functions
    """

    def distance_to_zdepth(
        self, distance_map: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            distance_map: (N, H, W) or (H, W), distance to the camera
        Returns:
            zdepth_map: (N, H, W) or (H, W), 0 means no depth value
        """
        rays = self.get_rays()  # normalized camera rays, shape (H, W, 3)

        return distance_map * rays[..., 2].expand_as(distance_map)

    def zdepth_to_distance(
        self, zdepth_map: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            zdepth_map: (N, H, W) or (H, W), depth value
        Returns:
            distance_map: (N, H, W) or (H, W), distance to the camera
        """
        rays = self.get_rays()

        return zdepth_map / rays[..., 2].expand_as(zdepth_map)

    def get_zdepth_map_from_points(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        points: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(camera_poses, np.ndarray):
            return self.get_zdepth_map_from_points_np(camera_poses, points)
        else:
            return self.get_zdepth_map_from_points_torch(camera_poses, points)

    def get_zdepth_map_from_points_torch(
        self, camera_poses: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            zdepth_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_images = []
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        points = make_sure_torch(points).to(self.device).to(self.dtype)

        single_camera_pose = False
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)
            single_camera_pose = True

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_torch(
                points, torch.inverse(camera_to_world)
            )
            uv_coords = self.ray2pixel_torch(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = torch.round(uv_coords[:, 0]).long()
            v_round = torch.round(uv_coords[:, 1]).long()

            valid_uv_mask = (
                (u_round >= 0)
                & (u_round < self.width)
                & (v_round >= 0)
                & (v_round < self.height)
            )
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]

            indices = v_valid * self.width + u_valid

            depth_image = (
                torch.full((self.height, self.width), float("inf")).to(depth).flatten()
            )
            depth_image = depth_image.scatter_reduce_(0, indices, z_valid, "amin")
            depth_image = depth_image.view(self.height, self.width)
            depth_mask = torch.isfinite(depth_image)

            # change inf to 0
            depth_image[~depth_mask] = 0

            depth_images.append(depth_image)

        depth_images = torch.stack(depth_images, dim=0)

        if single_camera_pose:
            depth_images = depth_images.squeeze(0)

        return depth_images

    def get_zdepth_map_from_points_np(
        self, camera_poses: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            zdepth_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_images = []
        camera_poses = make_sure_numpy(camera_poses)
        points = make_sure_numpy(points)

        single_camera_pose = False
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
            single_camera_pose = True

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_np(
                points, np.linalg.inv(camera_to_world)
            )
            uv_coords = self.ray2pixel_np(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = np.round(uv_coords[:, 0]).astype(np.int32)
            v_round = np.round(uv_coords[:, 1]).astype(np.int32)

            valid_uv_mask = (
                (u_round >= 0)
                & (u_round < self.width)
                & (v_round >= 0)
                & (v_round < self.height)
            )
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]

            depth_image = np.full((self.height, self.width), np.inf)
            # use np.minimum.at to update the depth image
            np.minimum.at(depth_image, (v_valid, u_valid), z_valid)
            depth_image = np.where(np.isfinite(depth_image), depth_image, 0)

            depth_images.append(depth_image)

        depth_images = np.stack(depth_images, axis=0)

        if single_camera_pose:
            depth_images = depth_images[0]

        return depth_images

    def get_distance_map_from_points(
        self, camera_poses: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            points: (M, 3), in world coordinate
        Returns:
            distance_map: (N, H, W) or (H, W), 0 means no depth value
        """
        depth_map = self.get_zdepth_map_from_points(
            camera_poses, points
        )  # shape (N, H, W) or (H, W)
        return self.zdepth_to_distance(depth_map)

    def get_distance_map_from_voxel(
        self, camera_poses: torch.Tensor, voxel_grid
    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch

        Returns:
            distance_map: (N, H, W) or (H, W), 0 means no depth value
        """
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        voxel_grid = voxel_grid.to(self.device)

        single_camera_pose = False
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)
            single_camera_pose = True

        rays_o, rays_d = self.get_rays_posed(camera_poses)
        N, H, W = rays_o.shape[:3]

        segment = voxel_grid.segments_along_rays(
            rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-1
        )
        pixel_hit = segment.joffsets[1:] - segment.joffsets[:-1]
        pixel_hit = pixel_hit.view(N, H, W).float()
        distance = segment.jdata[:, 0]  # [N_hit,]

        distance_map = torch.zeros((N, H, W)).to(distance.device)
        distance_map[pixel_hit > 0] = distance

        if single_camera_pose:
            distance_map = distance_map.squeeze(0)

        return distance_map

    def get_zdepth_map_from_voxel(
        self, camera_poses: torch.Tensor, voxel_grid
    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch

        Returns:
            zdepth_map: (N, H, W) or (H, W), 0 means no depth value
        """
        distance_map = self.get_distance_map_from_voxel(camera_poses, voxel_grid)
        return self.distance_to_zdepth(distance_map)

    def get_semantic_map_from_voxel(
        self,
        camera_poses: torch.Tensor,
        voxel_grid,
        voxel_semantic: torch.Tensor,
        background_semantic: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            camera_pose: (N, 4, 4) or (4, 4)
            voxel_grid: fvdb.GridBatch
            voxel_semantic: torch.Tensor, (#voxel, )

        Returns:
            semantic_map: (N, H, W) or (H, W)
        """
        camera_poses = make_sure_torch(camera_poses).to(self.device).to(self.dtype)
        voxel_grid = voxel_grid.to(self.device)
        voxel_semantic = voxel_semantic.to(self.device)

        single_camera_pose = False
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses.unsqueeze(0)
            single_camera_pose = True

        rays_o, rays_d = self.get_rays_posed(camera_poses)
        N, H, W = rays_o.shape[:3]

        vox, times = voxel_grid.voxels_along_rays(
            rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), 1, eps=1e-2, return_ijk=False
        )

        pixel_hit = times.joffsets[1:] - times.joffsets[:-1]
        pixel_hit = pixel_hit.view(N, H, W).bool()  # (N, H, W) 0,1 mask

        # semantic
        hit_voxel_semantic = voxel_semantic[vox.jdata]  # [N_hit, C]

        # fill the semantic to the image plane. 0 refers to UNDEFINED, which is the sky / background.
        semantic_rasterize = torch.full((N, H, W), background_semantic).to(
            hit_voxel_semantic
        )
        semantic_rasterize[pixel_hit] = hit_voxel_semantic

        if single_camera_pose:
            semantic_rasterize = semantic_rasterize.squeeze(0)  # (H, W)

        return semantic_rasterize

    """
    Drawing related functions
    """

    def draw_points(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        points: Union[torch.Tensor, np.ndarray],
        colors: Union[torch.Tensor, np.ndarray, None] = None,
        radius: int = 1,
        fast_impl_when_radius_gt_1: bool = True,
    ) -> np.ndarray:
        """
        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            points: torch.Tensor or np.ndarray
                shape: (M, 3)
            colors: torch.Tensor or np.ndarray or None
                shape: (M, 3) in uint8
            radius: int,
                radius of the point
            fast_impl_when_radius_gt_1: bool,
                if True, use cv2.circle to draw the point when radius > 1
        Returns:
            canvas: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)
        points = make_sure_numpy(points)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        if colors is not None:
            colors = make_sure_numpy(colors)
        else:
            colors = np.tile([[255, 0, 0]], (points.shape[0], 1))

        for camera_to_world in camera_poses:
            points_in_cam = self.transform_points_np(
                points, np.linalg.inv(camera_to_world)
            )
            uv_coords = self.ray2pixel_np(points_in_cam)
            depth = points_in_cam[:, 2]
            valid_depth_mask = depth > 0

            u_round = np.round(uv_coords[:, 0]).astype(np.int32)
            v_round = np.round(uv_coords[:, 1]).astype(np.int32)

            valid_uv_mask = (
                (u_round >= 0)
                & (u_round < self.width)
                & (v_round >= 0)
                & (v_round < self.height)
            )
            valid_mask = valid_depth_mask & valid_uv_mask

            u_valid = u_round[valid_mask]
            v_valid = v_round[valid_mask]
            z_valid = depth[valid_mask]
            colors_valid = colors[valid_mask]

            sorted_indices = np.argsort(z_valid, axis=0)[::-1]
            u_valid = u_valid[sorted_indices]
            v_valid = v_valid[sorted_indices]
            colors_valid = colors_valid[sorted_indices]

            if radius > 1 and fast_impl_when_radius_gt_1 is False:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for u, v, color in zip(u_valid, v_valid, colors_valid):
                    cv2.circle(canvas, (u.item(), v.item()), radius, color.tolist(), -1)
                canvas = np.array(canvas, dtype=np.uint8)

            # radius = 1 or we want fast impl
            else:
                canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                canvas[v_valid, u_valid] = (
                    colors_valid  # fill from the farthest point to the nearest point
                )

                # use fast impl when radius > 1
                if radius > 1:
                    canvas_accum = np.zeros_like(canvas)
                    i_shifts = np.arange(-radius // 2, radius // 2 + 1)
                    j_shifts = np.arange(-radius // 2, radius // 2 + 1)
                    for i in i_shifts:
                        for j in j_shifts:
                            # use torch.roll to shift the canvas
                            canvas_shifted = np.roll(canvas, shift=(i, j), axis=(0, 1))
                            canvas_accum = np.maximum(canvas_accum, canvas_shifted)
                    canvas = canvas_accum

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1:
            draw_images = draw_images[0]

        return draw_images

    """
    Auxiliary functions for drawing lines
    """

    def project_line_depth(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        polylines: List,
        segment_interval: float = 0,
        depth_max: float = 122.5,
    ) -> List:
        """
        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            polylines: list of list of points,
                each point is in 3D (x, y, z)
            radius: int,
                radius of the drawn circle
            colors: np.ndarray,
                shape: (3, ), dtype: np.uint8
            segment_interval: float,
                if > 0, the polyline is segmented into segments with the interval

        Returns:
            xy_and_depth_all_frames: list of 2D polyline sets to be drawn on the image
                len(xy_and_depths) = N
                xy_and_depths[i] is a list of 2D line segments to be drawn on the i-th image
                each line segment [(u1, v1, depth1), (u2, v2, depth2)]. just 2 points for each line segment
        """
        xy_and_depth_all_frames = []
        camera_poses = make_sure_numpy(camera_poses)
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
        world_to_cameras = np.linalg.inv(camera_poses)

        for world_to_camera in world_to_cameras:
            xy_and_depth_current_frame = []

            for polyline in polylines:
                if len(polyline) < 2:
                    continue

                if isinstance(polyline, list):
                    polyline = np.array(polyline)

                points_in_cam = self.transform_points(polyline, world_to_camera)
                if (points_in_cam[:, 2] < 0).all():
                    continue

                if segment_interval > 0:
                    polyline = interpolate_polyline_to_points(
                        polyline, segment_interval
                    )

                points_in_cam = self.transform_points(polyline, world_to_camera)

                uv_coords = self.ray2pixel(points_in_cam)
                depth = points_in_cam[:, 2]

                valid_uv_mask = (
                    (uv_coords[:, 0] >= 0)
                    & (uv_coords[:, 0] < self.width)
                    & (uv_coords[:, 1] >= 0)
                    & (uv_coords[:, 1] < self.height)
                )

                # filter out the polyline if all points are out of the image boundary
                if (~valid_uv_mask).all():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > depth_max:
                    continue

                for i in range(len(uv_coords) - 1):
                    if depth[i] < 0 and depth[i + 1] < 0:
                        continue

                    if depth[i] * depth[i + 1] < 0:
                        # if the two points are on different sides of the camera, we first clip the 3d point in the back to the camera plane + epsilon
                        # and then reproject it to the image plane, calculate the uv coordinate
                        pt1 = points_in_cam[i]
                        pt2 = points_in_cam[i + 1]

                        # make sure pt1 is in front of the camera, pt2 is behind the camera
                        if depth[i] < 0:
                            pt1, pt2 = pt2, pt1

                        # clip the line to the image boundary
                        eps = 2e-1
                        t = (-pt2[2]) / (pt1[2] - pt2[2]) + eps
                        pt2 = t * pt1 + (1 - t) * pt2

                        # project the point to the image plane
                        pt1_norm = pt1[:3] / pt1[2]
                        pt2_norm = pt2[:3] / pt2[2]

                        pixel1 = self.ray2pixel(pt1_norm)[0]
                        pixel2 = self.ray2pixel(pt2_norm)[0]
                    else:
                        pixel1 = uv_coords[i]
                        pixel2 = uv_coords[i + 1]

                    x1 = float(pixel1[0])
                    y1 = float(pixel1[1])
                    x2 = float(pixel2[0])
                    y2 = float(pixel2[1])
                    depth1 = float(depth[i])
                    depth2 = float(depth[i + 1])

                    xy_and_depth_current_frame.append(
                        [(x1, y1, depth1), (x2, y2, depth2)]
                    )

            xy_and_depth_all_frames.append(xy_and_depth_current_frame)

        return xy_and_depth_all_frames

    def draw_line_depth(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        polylines: List,
        radius: int = 8,
        colors: np.ndarray = None,
        segment_interval: float = 0,
        depth_max: float = 122.5,
    ) -> np.ndarray:
        """
        draw lines on the image, and the drawed pixel value is related to the depth of the points.
        The polyline can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            polylines: list of list of points,
                each point is in 3D (x, y, z)
            radius: int,
                radius of the drawn circle
            colors: np.ndarray,
                shape: (3, ), dtype: np.uint8
            segment_interval: float,
                if > 0, the polyline is segmented into segments with the interval
            depth_max: float,
                depth max value

        Returns:
            draw_images: np.ndarray
                shape: (N, H, W, 3) or (H, W, 3)
                dtype: np.uint8
        """
        draw_images = []
        camera_poses = make_sure_numpy(camera_poses)

        if colors is None:
            colors = np.array([255, 255, 255])

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        xy_and_depths = self.project_line_depth(
            camera_poses, polylines, segment_interval
        )

        for xy_and_depth_current_frame in xy_and_depths:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for line_segment in xy_and_depth_current_frame:
                u1, v1, depth1 = line_segment[0]
                u2, v2, depth2 = line_segment[1]
                pixel1 = np.array([u1, v1])
                pixel2 = np.array([u2, v2])

                try:
                    clipped, pixel1, pixel2 = cv2.clipLine(
                        (0, 0, self.width, self.height),
                        pixel1.astype(np.int32),
                        pixel2.astype(np.int32),
                    )
                except:
                    breakpoint()

                depth_mean = (depth1 + depth2) / 2
                depth_mean = np.clip(depth_mean, 0, depth_max)
                fill_weight = (2 * (depth_max - depth_mean)) / 255
                fill_value = (fill_weight * colors).astype(np.uint8).tolist()

                cv2.line(canvas, tuple(pixel1), tuple(pixel2), fill_value, radius)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if draw_images.shape[0] == 1 and len(draw_images.shape) == 3:
            draw_images = draw_images[0]

        return draw_images

    def project_hull_depth(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        hulls: List,
        depth_max: float = 122.5,
    ) -> List:
        """
        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            hulls: list of list of points or np.ndarray [N, M, 3]
                each point is in 3D (x, y, z)
            depth_max: float,
                depth max value
        Returns:
            xy_and_depths_all_frames: list of 2D hull sets to be drawn on the image
                len(xy_and_depths_all_frames) = N
                xy_and_depths_all_frames[i] is a list of 2D hull segments to be drawn on the i-th image
                each hull segment [(u1, v1, depth1), (u2, v2, depth2), ...]
        """
        xy_and_depths_all_frames = []
        camera_poses = make_sure_numpy(camera_poses)

        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]

        for camera_to_world in camera_poses:
            xy_and_depth_current_frame = []

            for hull in hulls:
                if len(hull) < 3:
                    continue

                points_in_cam = self.transform_points_np(
                    hull, np.linalg.inv(camera_to_world)
                )

                if (points_in_cam[:, 2] < 0).all():
                    continue

                uv_coords = self.ray2pixel_np(points_in_cam).astype(np.int32)
                depth = points_in_cam[:, 2]
                valid_depth_mask = depth > 0

                u_round = uv_coords[:, 0]
                v_round = uv_coords[:, 1]
                valid_uv_mask = (
                    (u_round >= 0)
                    & (u_round < self.width)
                    & (v_round >= 0)
                    & (v_round < self.height)
                )
                valid_mask = valid_depth_mask & valid_uv_mask

                # filter out the polyline if all points are out of the image boundary
                if not valid_mask.any():
                    continue

                # if depth all greater than DEPTH_MAX, skip
                if depth.min() > depth_max:
                    continue

                # project again with clipped points
                points_in_cam_clipped = self._clip_polyline_to_image_plane(
                    points_in_cam
                )
                uv_coords = self.ray2pixel_np(points_in_cam_clipped).astype(np.int32)

                xy_and_depth_this_hull = np.concatenate(
                    [uv_coords, points_in_cam_clipped[:, -1:]], axis=1
                )
                xy_and_depth_current_frame.append(xy_and_depth_this_hull)

            xy_and_depths_all_frames.append(xy_and_depth_current_frame)

        return xy_and_depths_all_frames

    def draw_hull_depth(
        self,
        camera_poses: Union[torch.Tensor, np.ndarray],
        hulls: Union[List, np.ndarray],
        colors: np.ndarray = None,
        depth_max: float = 122.5,
    ) -> torch.Tensor:
        """
        draw hulls on the image, and the drawed pixel value is related to the depth of the points.
        The hull can be out of boundary, use cv2.clipLine to clip the line to the image boundary, or abandon the line.
        Then use cv2.line to draw the line.

        Args:
            camera_poses: torch.Tensor or np.ndarray
                shape: (N, 4, 4) or (4, 4)
            hulls: list of list of points, or can be stored as np.ndarray [N, M, 3]
                each point is in 3D (x, y, z)
            colors: np.ndarray,
                shape: (3, ), dtype: np.uint8
            depth_max: float,
                depth max value

        Returns:
            draw_images: (N, H, W, 3) or (H, W, 3), image with hulls drawn
        """
        draw_images = []

        make_sure_numpy(camera_poses)

        single_camera_pose = False
        if len(camera_poses.shape) == 2:
            camera_poses = camera_poses[np.newaxis, ...]
            single_camera_pose = True

        xy_and_depths_all_frames = self.project_hull_depth(camera_poses, hulls)

        for xy_and_depth_current_frame in xy_and_depths_all_frames:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for hull in xy_and_depth_current_frame:
                uv_coords = hull[:, :2].astype(np.int32)
                depth = hull[:, 2]

                depth_mean = depth.mean()

                # create convex hull for uv_coords, update the uv_coords
                uv_coords = cv2.convexHull(uv_coords).reshape(-1, 2)

                # maybe degrade to a line
                if uv_coords.shape[0] < 3:
                    continue

                polygon = Polygon(uv_coords)

                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                boundary = Polygon(
                    [
                        (0, 0),
                        (self.width, 0),
                        (self.width, self.height),
                        (0, self.height),
                    ]
                )
                clipped_polygon = polygon.intersection(boundary)

                if clipped_polygon.is_empty or clipped_polygon.geom_type != "Polygon":
                    continue

                clipped_points = list(clipped_polygon.exterior.coords)

                fill_weight = (2 * (depth_max - depth_mean)) / 255
                fill_value = (fill_weight * colors).astype(np.uint8).tolist()

                clipped_points = np.array(clipped_points, dtype=np.int32)
                cv2.fillPoly(canvas, [clipped_points], fill_value)

            draw_images.append(canvas)

        draw_images = np.stack(draw_images, axis=0)

        if single_camera_pose:
            draw_images = draw_images[0]

        return draw_images
