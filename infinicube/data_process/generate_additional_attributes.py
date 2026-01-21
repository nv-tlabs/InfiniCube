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

import os
import sys
import traceback
from pathlib import Path

import click
import numpy as np
import torch

sys.path.append(Path(__file__).parent.parent.as_posix())

from loguru import logger
from tqdm import tqdm

from infinicube.camera.pinhole import PinholeCamera
from infinicube.data_process.utils import (
    add_interpolated_maps_to_voxel,
    estimate_road_surface_in_world,
    inference_mmseg,
    load_mmseg_inferencer,
    polylines_to_discrete_points,
)
from infinicube.data_process.waymo_utils import imageencoder_imageio_png
from infinicube.utils.fileio_utils import write_video_file
from infinicube.utils.semantic_utils import (
    WAYMO_CATEGORY_NAMES,
    WAYMO_VISUALIZATION_TYPES_BLUE_SKY,
    semantic_to_color,
)
from infinicube.utils.wds_utils import get_sample, write_to_tar

DEBUG = False

CAMERA_NAMES = ["FRONT"]  # infinicube only uses front camera

RESOLUTION_ANNO = {
    "480p": (480, 832),
    "720p": (720, 1280),
}


def setup(local_rank, world_size):
    """Bind the process to a GPU"""
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    print(
        f"Process {local_rank} / {world_size} is using GPU {local_rank % torch.cuda.device_count()} in its node."
    )


def generate_skymask_for_clip(clip_id, inferenecer, output_root, resolution="480p"):
    """Generate sky mask from existing image tar files"""
    output_root_p = Path(output_root)

    logger.info(f"Processing {clip_id} for skymask generation")

    for camera_name in CAMERA_NAMES:
        image_tar_file = (
            output_root_p
            / f"image_{resolution}_{camera_name.lower()}"
            / f"{clip_id}.tar"
        )
        output_file = (
            output_root_p
            / f"skymask_{resolution}_{camera_name.lower()}"
            / f"{clip_id}.tar"
        )

        if output_file.exists():
            print(f"Skip {clip_id} {camera_name} for skymask, already exists")
            continue

        if not image_tar_file.exists():
            print(f"Skip {clip_id} {camera_name}, image tar file not found")
            continue

        try:
            # Read images from tar
            images = get_sample(image_tar_file)
            video_numpy_list = [
                image for name, image in images.items() if name.endswith(".jpg")
            ]

            # Run segmentation inference
            segmentation_numpy_list = inference_mmseg(video_numpy_list, inferenecer)

            # Create sample
            sample = {}
            sample["__key__"] = f"{clip_id}"
            for idx, segmentation_numpy in enumerate(segmentation_numpy_list):
                sky_mask = segmentation_numpy.astype(np.uint8) == 10
                sample[f"{idx:06d}.skymask.{camera_name.lower()}.png"] = sky_mask

            # Write to tar file
            write_to_tar(sample, output_file)

        except Exception:
            print(
                f"Error processing {clip_id} {camera_name} for skymask: {traceback.format_exc()}"
            )


def generate_lidar_depth_for_clip(clip_id, output_root, resolution="480p"):
    """Generate lidar depth tar files from NPZ point cloud files"""
    output_root_p = Path(output_root)
    resize_hw = RESOLUTION_ANNO[resolution]

    logger.info(f"Processing {clip_id} for lidar depth generation")

    # Check lidar NPZ files exist
    lidar_folder = output_root_p / "lidars" / clip_id / "lidar_TOP"
    if not lidar_folder.exists():
        print(f"Skip {clip_id}, lidar folder not found")
        return

    lidar_files = sorted(list(lidar_folder.glob("*.npz")))
    if len(lidar_files) == 0:
        print(f"Skip {clip_id}, no lidar files found")
        return

    # Load intrinsic, extrinsic and pose info
    intrinsic_tar_file = output_root_p / "intrinsic" / f"{clip_id}.tar"
    extrinsic_tar_file = output_root_p / "extrinsic" / f"{clip_id}.tar"
    pose_tar_file = output_root_p / "pose" / f"{clip_id}.tar"

    if (
        not intrinsic_tar_file.exists()
        or not extrinsic_tar_file.exists()
        or not pose_tar_file.exists()
    ):
        print(f"Skip {clip_id}, intrinsic/extrinsic/pose tar not found")
        return

    try:
        # Load intrinsics
        intrinsics = get_sample(intrinsic_tar_file)

        # Load extrinsics
        extrinsics = get_sample(extrinsic_tar_file)

        # Load poses
        poses = get_sample(pose_tar_file)

        for camera_name in CAMERA_NAMES:
            output_file = (
                output_root_p
                / f"lidar_depth_100_{resolution}_{camera_name.lower()}"
                / f"{clip_id}.tar"
            )
            sample_lidar_depth = {}
            sample_lidar_depth["__key__"] = f"{clip_id}"

            if output_file.exists():
                print(f"Skip {clip_id} {camera_name} for lidar depth, already exists")
                continue

            # Get intrinsic for this camera
            intrinsic_key = f"intrinsic.{camera_name.lower()}.npy"
            extrinsic_key = f"extrinsic.{camera_name.lower()}.npy"
            lidar_extrinsic_key = "extrinsic.lidar_top.npy"

            # Get poses for this camera
            pose_keys = [
                k
                for k in poses.keys()
                if k.endswith(f".pose.{camera_name.lower()}.npy")
            ]
            pose_keys.sort()

            if len(pose_keys) != len(lidar_files):
                print(
                    f"Warning: {clip_id} {camera_name} pose count {len(pose_keys)} != lidar count {len(lidar_files)}"
                )

            if intrinsic_key not in intrinsics or extrinsic_key not in extrinsics:
                print(f"Skip {clip_id} {camera_name}, intrinsic or extrinsic not found")
                continue

            camera_model = PinholeCamera.from_numpy(intrinsics[intrinsic_key])
            camera_model.rescale(
                ratio_h=resize_hw[0] / camera_model.height,
                ratio_w=resize_hw[1] / camera_model.width,
            )

            camera_to_vehicle = extrinsics[extrinsic_key]
            lidar_to_vehicle = extrinsics[lidar_extrinsic_key]

            camera_to_lidar = np.linalg.inv(lidar_to_vehicle) @ camera_to_vehicle

            for idx, lidar_file in enumerate(lidar_files):
                # Load lidar points from NPZ (in vehicle frame)
                lidar_data = np.load(lidar_file)
                lidar_points = lidar_data["points"]  # [N, 3], in vehicle frame

                # Convert to torch for projection
                points_torch = torch.from_numpy(lidar_points).float().cuda()
                lidar_depth = camera_model.get_zdepth_map_from_points_torch(
                    camera_to_lidar, points_torch
                )
                lidar_depth_np = lidar_depth.cpu().numpy()

                # Encode as PNG16 (depth * 100)
                sample_lidar_depth[
                    f"{idx:06d}.lidar_depth_100.{camera_name.lower()}.png"
                ] = imageencoder_imageio_png((lidar_depth_np * 100).astype(np.uint16))

            # Write to tar file
            write_to_tar(sample_lidar_depth, output_file)

    except Exception:
        print(f"Error processing {clip_id} for lidar depth: {traceback.format_exc()}")


def generate_discrete_map_points_for_clip(clip_id, output_root):
    """
    Generate discrete map points from polylines for 3d_lane, 3d_road_edge, and 3d_road_line.
    Also generate 3d_road_surface using road surface estimation.
    They are just used in latent conditioning so the resolution don't need to be high.

    Args:
        clip_id: str, the clip id
        output_root: str or Path, the root folder of the dataset
    """

    segment_interval = 0.25
    block_size = [40, 40]
    voxel_sizes = [0.4, 0.4, 0.2]
    output_root_p = Path(output_root)

    logger.info(f"Processing {clip_id} for discrete map points generation")

    # Step 1: Process each map feature type (3d_lane, 3d_road_edge, 3d_road_line)
    # Interpolate polylines to discrete points
    map_feature_types = [
        (("3d_lane", "lane.json"), ("3d_lane_voxelsize_025", "lane.npy")),
        (
            ("3d_road_edge", "road_edge.json"),
            ("3d_road_edge_voxelsize_025", "road_edge.npy"),
        ),
        (
            ("3d_road_line", "road_line.json"),
            ("3d_road_line_voxelsize_025", "road_line.npy"),
        ),
    ]

    for (in_wds_dir, in_wds_key), (out_wds_dir, out_wds_key) in map_feature_types:
        input_tar_file = output_root_p / in_wds_dir / f"{clip_id}.tar"
        output_file = output_root_p / out_wds_dir / f"{clip_id}.tar"

        if output_file.exists():
            print(f"Skip {clip_id} {out_wds_dir}, already exists")
            continue

        if not input_tar_file.exists():
            print(f"Skip {clip_id} {out_wds_dir}, input tar file not found")
            continue

        try:
            # Read polylines from tar
            map_sample = get_sample(input_tar_file)

            polylines = map_sample[
                in_wds_key
            ]  # List of polylines, each is a list of [x, y, z] points

            # Interpolate each polyline to discrete points
            interpolated_polylines = polylines_to_discrete_points(
                polylines, segment_interval
            )

            # Create output sample
            output_sample = {
                "__key__": f"{clip_id}",
                out_wds_key: interpolated_polylines,
            }

            # Write to tar file
            write_to_tar(output_sample, output_file)

        except Exception:
            print(f"Error processing {clip_id} {out_wds_dir}: {traceback.format_exc()}")

    # Generate road surface from discrete lane and road_edge points
    output_road_surface_file = (
        output_root_p / "3d_road_surface_voxelsize_04" / f"{clip_id}.tar"
    )

    if output_road_surface_file.exists():
        print(f"Skip {clip_id} 3d_road_surface, already exists")
        return

    # Read discrete lane and road_edge points from tar files
    lane_tar_file = output_root_p / "3d_lane_voxelsize_025" / f"{clip_id}.tar"
    road_edge_tar_file = output_root_p / "3d_road_edge_voxelsize_025" / f"{clip_id}.tar"

    # Load lane discrete points
    lane_sample = get_sample(lane_tar_file)
    lane_polyline_discrete_points = lane_sample["lane.npy"]

    # Load road_edge discrete points
    road_edge_sample = get_sample(road_edge_tar_file)
    road_edge_polylines_discrete_points = road_edge_sample["road_edge.npy"]

    if (
        len(lane_polyline_discrete_points) > 0
        and len(road_edge_polylines_discrete_points) > 0
    ):
        logger.info(f"Processing {clip_id} for road surface generation")

        # Estimate road surface
        road_surface_points = estimate_road_surface_in_world(
            road_edge_polylines_discrete_points,
            lane_polyline_discrete_points,
            block_size=block_size,
            voxel_sizes=voxel_sizes,
        )

        print(f"Generated road surface points: {len(road_surface_points)}")

        # Create output sample
        road_surface_sample = {
            "__key__": f"{clip_id}",
            "road_surface.npy": road_surface_points,
        }

        # Write to tar file
        write_to_tar(road_surface_sample, output_road_surface_file)
    else:
        print(f"Skip {clip_id} road surface generation: insufficient points")


def generate_map_augmentated_car_removed_voxel_for_clip(clip_id, output_root):
    """
    To prepare a better static point cloud for stage 1 training, we will do the following:
    1. add interpolated maps points to the static point cloud to strengthen the map voxels
    2. remove car / pedestrian points in the static point cloud even they are static (and add them back in Dataloader with CAD model)
    """

    output_root_p = Path(output_root)
    static_point_cloud_file = output_root_p / "pc_voxelsize_01" / f"{clip_id}.tar"
    static_data = get_sample(static_point_cloud_file)["pcd.vs01.pth"]
    static_points = static_data["points"]  # first vehicle coordinate
    static_category = static_data["semantics"]

    road_line_tar_file = output_root_p / "3d_road_line" / f"{clip_id}.tar"
    new_static_point_cloud_file = (
        output_root_p / "pc_with_map_without_car_voxelsize_01" / f"{clip_id}.tar"
    )

    map_name_to_polylines = {
        "3d_road_line": get_sample(road_line_tar_file)["road_line.json"],
    }
    map_name_to_semantic_category = {
        "3d_road_line": WAYMO_CATEGORY_NAMES.index("LANE_MARKER"),
    }

    map_voxel_points, map_voxel_semantics = add_interpolated_maps_to_voxel(
        map_name_to_polylines,
        map_name_to_semantic_category,
    )

    pc_to_world = static_data["pc_to_world"]
    world_to_pc = np.linalg.inv(pc_to_world)
    map_voxel_points = PinholeCamera.transform_points(map_voxel_points, world_to_pc)

    map_voxel_points_tensor = torch.from_numpy(map_voxel_points).to(static_points.dtype)
    map_voxel_semantics_tensor = torch.from_numpy(map_voxel_semantics).to(
        static_category.dtype
    )

    static_with_map_points = torch.cat([static_points, map_voxel_points_tensor], dim=0)
    static_with_map_semantics = torch.cat(
        [static_category, map_voxel_semantics_tensor], dim=0
    )

    # Get category names to remove (pedestrians/cyclists and cars/vehicles)
    semantic_category_names_to_remove = (
        WAYMO_VISUALIZATION_TYPES_BLUE_SKY[1] + WAYMO_VISUALIZATION_TYPES_BLUE_SKY[3]
    )
    # Convert category names to indices
    semantic_category_to_remove = [
        WAYMO_CATEGORY_NAMES.index(name) for name in semantic_category_names_to_remove
    ]
    # remove car / pedestrian points
    remove_mask = torch.isin(
        static_with_map_semantics, torch.tensor(semantic_category_to_remove)
    )
    static_with_map_without_car_points = static_with_map_points[~remove_mask]
    static_with_map_without_car_semantics = static_with_map_semantics[~remove_mask]

    new_data_sample = {
        "__key__": f"{clip_id}",
        "pcd.vs01.pth": {
            "points": static_with_map_without_car_points,
            "semantics": static_with_map_without_car_semantics,
            "pc_to_world": static_data["pc_to_world"],
        },
    }

    write_to_tar(new_data_sample, new_static_point_cloud_file)


def generate_buffer_for_clip(clip_id, output_root, resolution="480p"):
    """
    generate
    - semantic buffer (with static object only)
    - depth buffer (with static and dynamc objects)
    - instance buffer (with static and dynamc objects)
    """
    from pathlib import Path

    from infinicube.utils.fvdb_utils import generate_infinicube_buffer_from_fvdb_grid

    output_root_p = Path(output_root)
    resize_hw = RESOLUTION_ANNO[resolution]

    # Define output tar files
    depth_buffer_tar_file = (
        output_root_p / f"voxel_depth_100_{resolution}_front" / f"{clip_id}.tar"
    )
    semantic_buffer_tar_file = (
        output_root_p / f"semantic_buffer_{resolution}_front" / f"{clip_id}.tar"
    )
    semantic_buffer_video_file = (
        output_root_p
        / f"semantic_buffer_video_{resolution}_front"
        / f"{clip_id}.mp4"
    )
    instance_buffer_tar_file = (
        output_root_p / f"instance_buffer_{resolution}_front" / f"{clip_id}.tar"
    )

    # Check if already processed
    if (
        depth_buffer_tar_file.exists()
        and semantic_buffer_video_file.exists()
        and instance_buffer_tar_file.exists()
    ):
        print(f"Skip {clip_id} for buffer generation, already exists")
        return

    logger.info(f"Processing {clip_id} for buffer generation")

    # Load required data
    static_point_cloud_tar_file = output_root_p / "pc_voxelsize_01" / f"{clip_id}.tar"
    pose_tar_file = output_root_p / "pose" / f"{clip_id}.tar"
    intrinsic_tar_file = output_root_p / "intrinsic" / f"{clip_id}.tar"
    static_object_info_tar_file = (
        output_root_p / "static_object_info" / f"{clip_id}.tar"
    )
    dynamic_object_info_tar_file = (
        output_root_p / "dynamic_object_info" / f"{clip_id}.tar"
    )
    dynamic_object_points_canonical_tar_file = (
        output_root_p / "dynamic_object_points_canonical" / f"{clip_id}.tar"
    )

    # Check if all required files exist
    if not all(
        [
            static_point_cloud_tar_file.exists(),
            pose_tar_file.exists(),
            intrinsic_tar_file.exists(),
            static_object_info_tar_file.exists(),
            dynamic_object_info_tar_file.exists(),
            dynamic_object_points_canonical_tar_file.exists(),
        ]
    ):
        print(f"Skip {clip_id} for buffer generation: missing required files")
        return

    # Load data
    static_point_cloud_with_map_data = get_sample(static_point_cloud_tar_file)[
        "pcd.vs01.pth"
    ]
    static_points = static_point_cloud_with_map_data["points"].cuda()
    static_semantics = static_point_cloud_with_map_data["semantics"].cuda()
    static_pc_to_world = static_point_cloud_with_map_data["pc_to_world"].cuda()

    pose_data = get_sample(pose_tar_file)
    intrinsic_data = get_sample(intrinsic_tar_file)
    static_object_info = get_sample(static_object_info_tar_file)
    dynamic_object_info = get_sample(dynamic_object_info_tar_file)
    dynamic_object_points_canonical_data = get_sample(
        dynamic_object_points_canonical_tar_file
    )["dynamic_object_points_canonical.npz"]

    # Get camera intrinsic for FRONT camera
    camera_intrinsic = intrinsic_data["intrinsic.front.npy"]
    camera_model = PinholeCamera.from_numpy(camera_intrinsic, device="cuda")
    camera_model.rescale(
        ratio_h=resize_hw[0] / camera_model.height,
        ratio_w=resize_hw[1] / camera_model.width,
    )

    # Get all pose keys for FRONT camera
    pose_keys = [k for k in pose_data.keys() if k.endswith(".pose.front.npy")]
    pose_keys.sort()

    # Stack all camera poses
    camera_poses = torch.from_numpy(np.stack([pose_data[k] for k in pose_keys])).cuda()

    # Generate buffers using the fvdb_utils function
    depth_buffer, semantic_buffer, instance_buffer = (
        generate_infinicube_buffer_from_fvdb_grid(
            camera_model=camera_model,
            camera_poses_in_world=camera_poses,
            fvdb_scene_grid_or_points=static_points,
            fvdb_scene_semantic=static_semantics,
            fvdb_grid_to_world=static_pc_to_world,
            static_object_info=static_object_info,
            dynamic_object_info=dynamic_object_info,
            dynamic_object_points_canonical_data=dynamic_object_points_canonical_data,
            cad_model_for_dynamic_objects=False,
            voxel_sizes=[0.2, 0.2, 0.2],
        )
    )

    # Create samples for each buffer type
    depth_sample = {"__key__": f"{clip_id}"}
    semantic_sample = {"__key__": f"{clip_id}"}
    instance_sample = {"__key__": f"{clip_id}"}

    # Process and encode each frame
    for idx in range(depth_buffer.shape[0]):
        # Depth buffer: multiply by 100 and convert to uint16
        depth_np = depth_buffer[idx].cpu().numpy()
        depth_uint16 = (depth_np * 100).astype(np.uint16)
        depth_sample[f"{idx:06d}.voxel_depth_100.front.png"] = imageencoder_imageio_png(
            depth_uint16
        )

        # Semantic buffer: convert to uint8
        semantic_np = semantic_buffer[idx].cpu().numpy().astype(np.uint8)
        semantic_sample[f"{idx:06d}.semantic_buffer.front.png"] = (
            imageencoder_imageio_png(semantic_np)
        )

        # Instance buffer: convert to uint16
        instance_np = instance_buffer[idx].cpu().numpy().astype(np.uint16)
        instance_sample[f"{idx:06d}.instance_buffer.front.png"] = (
            imageencoder_imageio_png(instance_np)
        )

    # Write to tar files
    write_to_tar(depth_sample, depth_buffer_tar_file)
    write_to_tar(semantic_sample, semantic_buffer_tar_file)
    write_to_tar(instance_sample, instance_buffer_tar_file)

    # write videos for static
    semantic_buffer_rgb = semantic_to_color(
        semantic_buffer.cpu().numpy().astype(np.uint8)
    )  # (N, H, W, 3) in range [0, 1]
    semantic_buffer_rgb = (semantic_buffer_rgb * 255).astype(
        np.uint8
    )  # (N, H, W, 3) in range [0, 255]
    write_video_file(
        semantic_buffer_rgb, semantic_buffer_video_file, fps=10
    )

    logger.info(f"Successfully generated buffers for {clip_id}")


@click.command()
@click.option(
    "--input_root",
    type=str,
    default="data/",
    help="The root folder of the input webdataset",
)
@click.option(
    "--node_split",
    default="0,1",
    help="The node split. For example, 0,1 means there is 1 node and we are processing in first node",
)
@click.option("--generate_skymask", is_flag=True, help="Generate sky mask")
@click.option(
    "--generate_discrete_map",
    is_flag=True,
    help="Generate discrete map points and road surface",
)
@click.option(
    "--generate_map_augmentated_car_removed_voxel",
    is_flag=True,
    help="Generate map augmentated car removed voxel",
)
@click.option("--generate_buffer", is_flag=True, help="Generate buffer")
def main(
    input_root,
    node_split,
    generate_skymask,
    generate_discrete_map,
    generate_map_augmentated_car_removed_voxel,
    generate_buffer,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup(local_rank, world_size)

    # Initialize models as needed
    inferenecer = None
    if generate_skymask:
        inferenecer = load_mmseg_inferencer()

    # Load clip list
    if DEBUG:
        clip_id_list = ["10107710434105775874_760_000_780_000"]
    else:
        clip_id_list = (Path(input_root) / "pose").rglob("*.tar")
        clip_id_list = [c.stem for c in clip_id_list]
        this_node, all_nodes = (
            int(node_split.split(",")[0]),
            int(node_split.split(",")[1]),
        )
        clip_id_list = clip_id_list[this_node::all_nodes]  # split by node
        clip_id_list = clip_id_list[local_rank::world_size]  # split by process

    # Process each clip
    for clip_id in tqdm(clip_id_list):
        if generate_skymask and inferenecer is not None:
            generate_skymask_for_clip(
                clip_id, inferenecer, input_root, resolution="480p"
            )

        if generate_discrete_map:
            generate_discrete_map_points_for_clip(clip_id, input_root)

        if generate_map_augmentated_car_removed_voxel:
            generate_map_augmentated_car_removed_voxel_for_clip(clip_id, input_root)

        if generate_buffer:
            generate_buffer_for_clip(clip_id, input_root, resolution="480p")


if __name__ == "__main__":
    main()
