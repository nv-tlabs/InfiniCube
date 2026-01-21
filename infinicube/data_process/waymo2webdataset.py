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

import argparse
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import imageio.v3 as iio
import numpy as np
import tensorflow as tf
from google.protobuf import json_format
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils

sys.path.append(Path(__file__).parent.parent.as_posix())

from infinicube.data_process.waymo_utils import (
    classify_static_dynamic_objects,
    encode_dict_to_npz_bytes,
    get_points_in_cuboid,
    object_info_to_canonical_cuboid,
    object_info_to_cuboid,
    object_info_to_object2world,
)
from infinicube.utils.fileio_utils import write_video_file
from infinicube.utils.wds_utils import write_to_tar

DEBUG = False
CAMERA_NAMES = [
    "FRONT"
]  # use front camera only in infinicube, available cameras are ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

if int(tf.__version__.split(".")[0]) < 2:
    tf.enable_eager_execution()


def flu_to_opencv(camera_pose: np.ndarray) -> np.ndarray:
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
    return np.concatenate(
        [
            -camera_pose[..., 1:2],
            -camera_pose[..., 2:3],
            camera_pose[..., 0:1],
            camera_pose[..., 3:4],
        ],
        axis=-1,
    )


class WaymoToWebdatasetConverter:
    RETURN_OK = 0
    RETURN_SKIP = 1

    MIN_MOVING_DISTANCE_AT_10FPS = 0.05  # minimum moving distance in 0.1 seconds

    def __init__(
        self,
        waymo_root: Union[Path, str],
        num_workers: int,
        node_split: str,
        skip_lidar: bool = False,
        skip_video: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.waymo_root = Path(waymo_root)
        self.num_workers = num_workers
        self.node_split = node_split
        self.this_node_index, self.total_nodes = map(int, node_split.split(","))
        self.skip_lidar = skip_lidar
        self.skip_video = skip_video
        self.overwrite = overwrite

        self._box_type_to_str = {
            label_pb2.Label.Type.TYPE_UNKNOWN: "unknown",
            label_pb2.Label.Type.TYPE_VEHICLE: "car",
            label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
            label_pb2.Label.Type.TYPE_SIGN: "sign",
            label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
        }

        self.WAYMO_CATEGORY_NAMES = [
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

        self.lidar_anno_mapping = {
            "car": self.WAYMO_CATEGORY_NAMES.index("CAR"),
            "pedestrian": self.WAYMO_CATEGORY_NAMES.index("PEDESTRIAN"),
            "cyclist": self.WAYMO_CATEGORY_NAMES.index("BICYCLIST"),
            "sign": self.WAYMO_CATEGORY_NAMES.index("SIGN"),
            "unknown": self.WAYMO_CATEGORY_NAMES.index("UNDEFINED"),
        }

    def list_segments(self) -> List[str]:
        return list(self.waymo_root.glob("*.tfrecord"))

    def extract_sensor_params(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        out = {}
        for camera_calib in frame.context.camera_calibrations:
            camera_name = self.get_camera_name(camera_calib.name)

            intrinsic = camera_calib.intrinsic
            fx, fy, cx, cy = intrinsic[:4]
            distortion = intrinsic[4:]

            extrinsic = np.array(camera_calib.extrinsic.transform).reshape((4, 4))

            out[camera_name] = {
                "camera_intrinsic": [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                "camera_D": distortion,
                "extrinsic": extrinsic,  # camera to vehicle
                "width": camera_calib.width,
                "height": camera_calib.height,
            }

        for lidar_calib in frame.context.laser_calibrations:
            lidar_name = self.get_lidar_name(lidar_calib.name)
            extrinsic = np.array(lidar_calib.extrinsic.transform).reshape((4, 4))
            out[lidar_name] = {"extrinsic": extrinsic}

        return out

    def extract_frame_images(
        self,
        frame: dataset_pb2.Frame,
        sensor_params,
        image_samples,
        pose_sample,
        camera_timestamps,
    ):
        lidar_timestamp: int = frame.timestamp_micros

        for image_data in frame.images:
            camera_name = self.get_camera_name(image_data.name)
            if camera_name.upper() not in CAMERA_NAMES:
                continue

            # Track timestamps and get index
            camera_timestamps[camera_name].append(lidar_timestamp)
            idx = len(camera_timestamps[camera_name]) - 1

            # Save image
            image_samples[camera_name][f"{idx:06d}.image.{camera_name.lower()}.jpg"] = (
                image_data.image
            )

            # Calculate pose (ego_pose @ camera_extrinsic = camera2world in FLU, then convert to OpenCV)
            ego_pose = np.array(image_data.pose.transform).reshape((4, 4))
            camera2world_flu = ego_pose @ sensor_params[camera_name]["extrinsic"]
            pose_opencv = flu_to_opencv(camera2world_flu)

            pose_sample[f"{idx:06d}.pose.{camera_name.lower()}.npy"] = pose_opencv

    def extract_frame_annotations(
        self,
        frame: dataset_pb2.Frame,
        frame_idx: int,
        out_dir: Path,
        segment_name: str,
        all_object_info_sample,
        all_object_points_canonical_data,
    ):
        """Extract lidar points, save as NPZ, and process annotations in one pass"""
        lidar_timestamp = frame.timestamp_micros
        timestamp = lidar_timestamp / 1.0e6

        if self.skip_lidar:
            lidar_points = np.zeros((0, 3), dtype=np.float32)
        else:
            # Parse lidar data ONCE for both NPZ saving and annotation processing
            (range_images, camera_projections, _, range_image_top_pose) = (
                frame_utils.parse_range_image_and_camera_projection(frame)
            )

            # occupy most of the time
            points, _ = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose
            )
            # occupy most of the time
            points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1,
            )
            points = [np.concatenate([p1, p2]) for p1, p2 in zip(points, points_ri2)]

            # Get TOP lidar points and save as compressed NPZ
            lidar_ids = [calib.name for calib in frame.context.laser_calibrations]
            lidar_ids.sort()

            for lidar_id, lidar_pts in zip(lidar_ids, points):
                lidar_name = self.get_lidar_name(lidar_id)
                if lidar_name == "lidar_TOP":
                    lidar_points = lidar_pts

                    # Save TOP lidar as compressed NPZ (much better compression than PCD)
                    lidar_folder = out_dir / "lidars" / segment_name / lidar_name
                    lidar_folder.mkdir(parents=True, exist_ok=True)
                    lidar_path = lidar_folder / f"{lidar_timestamp}.npz"
                    np.savez_compressed(lidar_path, points=lidar_pts.astype(np.float16))
                    break

        lidar_to_world = np.array(frame.pose.transform).reshape((4, 4))

        # Get previous frame's object info for movement detection
        prev_frame_objects = None
        if frame_idx > 0:
            prev_frame_key = f"{frame_idx - 1:06d}.all_object_info.json"
            if prev_frame_key in all_object_info_sample:
                prev_frame_objects = all_object_info_sample[prev_frame_key]

        # Process objects and collect all object info for this frame
        dynamic_bboxes = np.zeros((0, 8, 3), dtype=np.float32)
        transformation_current_timestamp = {}
        current_frame_all_objects = {}

        for label in frame.laser_labels:
            # Calculate object info
            center_vcs = np.array(
                [label.box.center_x, label.box.center_y, label.box.center_z, 1]
            )
            center_wcs = lidar_to_world @ center_vcs
            heading = label.box.heading
            rotation_vcs = R.from_euler(
                "xyz", [0, 0, heading], degrees=False
            ).as_matrix()
            rotation_wcs = lidar_to_world[:3, :3] @ rotation_vcs
            rotation_wcs = R.from_matrix(rotation_wcs).as_quat()

            object_to_world = object_info_to_object2world(
                {
                    "rotation": [
                        rotation_wcs[3],
                        rotation_wcs[0],
                        rotation_wcs[1],
                        rotation_wcs[2],
                    ],
                    "translation": center_wcs[:3].tolist(),
                }
            )

            # Determine if object is moving by comparing with previous frame
            is_moving = False
            if prev_frame_objects is not None and label.id in prev_frame_objects:
                # Get previous frame's center position
                prev_object_info = prev_frame_objects[label.id]
                prev_center_wcs = np.array(prev_object_info["object_to_world"])[:3, 3]

                # Calculate L2 distance between current and previous frame
                distance = np.linalg.norm(center_wcs[:3] - prev_center_wcs)

                # Use distance threshold to determine if moving
                is_moving = distance > self.MIN_MOVING_DISTANCE_AT_10FPS

            object_info = {
                "object_to_world": object_to_world.tolist(),
                "object_lwh": [label.box.length, label.box.width, label.box.height],
                "object_is_moving": bool(is_moving),
                "object_type": self._box_type_to_str[label.type],
            }

            # Store in current frame all objects
            current_frame_all_objects[label.id] = object_info

            # compute bbox and extract points
            bbox = object_info_to_cuboid(object_info)
            dynamic_bboxes = np.concatenate([dynamic_bboxes, bbox[np.newaxis]], axis=0)
            transformation_current_timestamp[label.id] = object_to_world

            if lidar_points is not None:
                points_in_bbox = get_points_in_cuboid(
                    lidar_points, lidar_to_world, object_info
                )

                obj_xyz_name = label.id + "_xyz"
                obj_semantic_name = label.id + "_semantic"
                obj_corner_name = label.id + "_corner"

                if obj_xyz_name not in all_object_points_canonical_data:
                    all_object_points_canonical_data[obj_xyz_name] = points_in_bbox
                    all_object_points_canonical_data[obj_semantic_name] = (
                        self.lidar_anno_mapping[object_info["object_type"]]
                    )
                    all_object_points_canonical_data[obj_corner_name] = (
                        object_info_to_canonical_cuboid(object_info)
                    )
                else:
                    all_object_points_canonical_data[obj_xyz_name] = np.concatenate(
                        [
                            all_object_points_canonical_data[obj_xyz_name],
                            points_in_bbox,
                        ],
                        axis=0,
                    )

        # Store current frame all objects info
        all_object_info_sample[f"{frame_idx:06d}.all_object_info.json"] = (
            current_frame_all_objects
        )

    def extract_map_data(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        """
        frame.map_features have many MapFeature item
        message MapFeature {
            // A unique ID to identify this feature.
            optional int64 id = 1;

            // Type specific data.
            oneof feature_data {
                LaneCenter lane = 3; # polyline
                RoadLine road_line = 4; # polyline
                RoadEdge road_edge = 5; # polyline
                StopSign stop_sign = 7;
                Crosswalk crosswalk = 8; # polygon
                SpeedBump speed_bump = 9; # polygon
                Driveway driveway = 10; # polygon
            }
        }

        Returns:
            map_data: Dict
                'lane': list of polylines, each polyline is noted by several vertices.
                'road_line': list of polylines, each polyline is noted by several vertices.
                ...
        """

        def hump_to_underline(hump_str):
            import re

            return re.sub(r"([a-z])([A-Z])", r"\1_\2", hump_str).lower()

        map_features_list = json_format.MessageToDict(frame)["mapFeatures"]
        feature_names = [
            "lane",
            "road_line",
            "road_edge",
            "crosswalk",
            "speed_bump",
            "driveway",
        ]
        map_data = dict(zip(feature_names, [[] for _ in range(len(feature_names))]))

        for feature in map_features_list:
            feature_name = list(feature.keys())
            feature_name.remove("id")
            feature_name = feature_name[0]
            feature_name_lower = hump_to_underline(feature_name)

            feature_content = feature[feature_name]
            if feature_name_lower in ["lane", "road_line", "road_edge"]:
                polyline = feature_content[
                    "polyline"
                ]  # [{'x':..., 'y':..., 'z':...}, {'x':..., 'y':..., 'z':...}, ...]
            elif feature_name_lower in ["crosswalk", "speed_bump", "driveway"]:
                polyline = feature_content[
                    "polygon"
                ]  # [{'x':..., 'y':..., 'z':...}, {'x':..., 'y':..., 'z':...}, ...]
            else:
                continue

            polyline = [
                [point["x"], point["y"], point["z"]] for point in polyline
            ]  # [[x, y, z], [x, y, z], ...]
            map_data[hump_to_underline(feature_name)].append(polyline)

        return map_data

    def get_camera_name(self, name_int) -> str:
        return dataset_pb2.CameraName.Name.Name(name_int)

    def get_lidar_name(self, name_int) -> str:
        return "lidar_" + dataset_pb2.LaserName.Name.Name(name_int)

    def extract_all(self, specify_segments: List[str], out_root: Union[Path, str]):
        out_root = Path(out_root)
        if not out_root.exists():
            out_root.mkdir(parents=True)

        all_segments = self.list_segments()
        all_segments = all_segments[self.this_node_index :: self.total_nodes]

        def find_segement(partial_segment_name: str, segments: List[Path]):
            for seg in segments:
                if partial_segment_name in seg.as_posix():
                    return seg
            return None

        inexist_segs, task_segs = [], []
        if specify_segments:
            for specify_segment in specify_segments:
                seg = find_segement(specify_segment, all_segments)
                if seg is None:
                    inexist_segs.append(specify_segment)
                else:
                    task_segs.append(seg)
        else:
            task_segs = all_segments

        if inexist_segs:
            print(f"{len(inexist_segs)} segments not found:")
            for seg in inexist_segs:
                print(seg)

        def print_error(e):
            print("ERROR:", e)

        fail_tasks, skip_tasks, succ_tasks = [], [], []

        if DEBUG:
            self.extract_one(task_segs[0], out_root)
            return

        with Pool(processes=self.num_workers) as pool:
            results = [
                pool.apply_async(
                    func=self.extract_one,
                    args=(seg, out_root),
                    error_callback=print_error,
                )
                for seg in task_segs
            ]

            print(f"Processing {len(results)} segments (it can be slow)")
            # Use tqdm to track progress as tasks complete
            with tqdm(total=len(results), desc="Processing segments") as pbar:
                completed = 0
                while completed < len(results):
                    # Count how many tasks have completed
                    new_completed = sum(1 for r in results if r.ready())
                    if new_completed > completed:
                        pbar.update(new_completed - completed)
                        completed = new_completed
                    # Small sleep to avoid busy waiting
                    import time

                    time.sleep(0.1)

            for segment, result in zip(task_segs, results):
                if not result.successful():
                    fail_tasks.append(segment)
                elif result.get() == WaymoToWebdatasetConverter.RETURN_SKIP:
                    skip_tasks.append(segment)
                elif result.get() == WaymoToWebdatasetConverter.RETURN_OK:
                    succ_tasks.append(segment)

        print(
            f"""{len(task_segs)} tasks total, {len(fail_tasks)} tasks failed, """
            f"""{len(skip_tasks)} tasks skipped, {len(succ_tasks)} tasks success"""
        )
        print("Failed tasks:")
        for seg in fail_tasks:
            print(seg.as_posix())
        print("Skipped tasks:")
        for seg in skip_tasks:
            print(seg.as_posix())

    def extract_one(self, segment_tfrecord: Path, out_dir: Path) -> int:
        dataset = tf.data.TFRecordDataset(
            segment_tfrecord.as_posix(), compression_type=""
        )
        segment_name = None
        sensor_params = None

        # Prepare tar samples
        image_samples = {camera_name: {} for camera_name in CAMERA_NAMES}
        lane_sample = {}
        road_edge_sample = {}
        road_line_sample = {}
        pose_sample = {}
        intrinsic_sample = {}
        extrinsic_sample = {}
        all_object_info_sample = {}
        static_object_info_sample = {}
        dynamic_object_info_sample = {}
        dynamic_object_points_canonical_sample = {}

        # Frame data for processing
        camera_timestamps = {camera_name: [] for camera_name in CAMERA_NAMES}
        all_object_points_canonical_data = {}
        dynamic_object_points_canonical_data = {}

        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            segment_name = frame.context.name

            if frame_idx == 0:
                # Check if already processed
                output_file = out_dir / "pose" / f"{segment_name}.tar"
                if output_file.exists() and not self.overwrite:
                    print(f"Skip {segment_name}, already processed")
                    return WaymoToWebdatasetConverter.RETURN_SKIP

                sensor_params = self.extract_sensor_params(frame)

                # Save intrinsic and extrinsic data
                for camera_name in CAMERA_NAMES:
                    intrinsic_matrix = sensor_params[camera_name]["camera_intrinsic"]
                    fx, fy, cx, cy = (
                        intrinsic_matrix[0][0],
                        intrinsic_matrix[1][1],
                        intrinsic_matrix[0][2],
                        intrinsic_matrix[1][2],
                    )
                    w = sensor_params[camera_name]["width"]
                    h = sensor_params[camera_name]["height"]
                    intrinsic = np.array([fx, fy, cx, cy, w, h])
                    intrinsic_sample[f"intrinsic.{camera_name.lower()}.npy"] = intrinsic

                    # Save extrinsic (camera to vehicle, in FLU convention)
                    extrinsic_sample[f"extrinsic.{camera_name.lower()}.npy"] = (
                        sensor_params[camera_name]["extrinsic"]
                    )

                # also save lidar extrinsic
                lidar_extrinsic = sensor_params["lidar_TOP"]["extrinsic"]
                extrinsic_sample["extrinsic.lidar_TOP.npy"] = lidar_extrinsic

                # save map data
                map_data = self.extract_map_data(frame)
                lane_sample["lane.json"] = map_data["lane"]
                road_edge_sample["road_edge.json"] = map_data["road_edge"]
                road_line_sample["road_line.json"] = map_data["road_line"]

            # Extract images and poses
            self.extract_frame_images(
                frame, sensor_params, image_samples, pose_sample, camera_timestamps
            )

            # Extract lidar NPZ and annotations in one pass (merged to avoid duplicate parsing)
            self.extract_frame_annotations(
                frame,
                frame_idx,
                out_dir,
                segment_name,
                all_object_info_sample,
                all_object_points_canonical_data,
            )

        # Post-process: classify objects into static and dynamic based on all frames
        classify_static_dynamic_objects(
            all_object_info_sample,
            all_object_points_canonical_data,
            static_object_info_sample,
            dynamic_object_info_sample,
            dynamic_object_points_canonical_data,
        )

        dynamic_object_points_canonical_sample[
            "dynamic_object_points_canonical.npz"
        ] = encode_dict_to_npz_bytes(dynamic_object_points_canonical_data)

        # Write all tar files
        for camera_name in CAMERA_NAMES:
            if not self.skip_video:
                # write to video
                video_name = (
                    out_dir / f"video_{camera_name.lower()}" / f"{segment_name}.mp4"
                )
                images_decoded = [
                    iio.imread(io.BytesIO(image_samples[camera_name][key]))
                    for key in image_samples[camera_name].keys()
                ]
                write_video_file(images_decoded, video_name.as_posix(), fps=10)

                # write to video 480p, resize to height=480, width=832 from images_decoded
                video_name_480p = (
                    out_dir
                    / f"video_480p_{camera_name.lower()}"
                    / f"{segment_name}.mp4"
                )
                images_decoded_480p = [
                    cv2.resize(image, (832, 480), interpolation=cv2.INTER_AREA)
                    for image in images_decoded
                ]
                write_video_file(
                    images_decoded_480p, video_name_480p.as_posix(), fps=10
                )

                # write to tar
                output_file = (
                    out_dir / f"image_{camera_name.lower()}" / f"{segment_name}.tar"
                )
                write_to_tar(
                    image_samples[camera_name], output_file, __key__=segment_name
                )

                # write to tar 480p
                output_file = (
                    out_dir
                    / f"image_480p_{camera_name.lower()}"
                    / f"{segment_name}.tar"
                )
                image_480p_samples = {
                    f"{idx:06d}.image.{camera_name.lower()}.jpg": image
                    for idx, image in enumerate(images_decoded_480p)
                }
                write_to_tar(image_480p_samples, output_file, __key__=segment_name)

        # Write other tar files (using dict for cleaner code)
        tar_outputs = {
            "pose": pose_sample,
            "intrinsic": intrinsic_sample,
            "extrinsic": extrinsic_sample,
            "static_object_info": static_object_info_sample,
            "dynamic_object_info": dynamic_object_info_sample,
            "3d_lane": lane_sample,
            "3d_road_edge": road_edge_sample,
            "3d_road_line": road_line_sample,
        }

        if not self.skip_lidar:
            tar_outputs["dynamic_object_points_canonical"] = (
                dynamic_object_points_canonical_sample
            )

        for dir_name, sample_data in tar_outputs.items():
            output_file = out_dir / dir_name / f"{segment_name}.tar"
            write_to_tar(sample_data, output_file, __key__=segment_name)

        return WaymoToWebdatasetConverter.RETURN_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Root directory of waymo dataset (tfrecord files).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory of webdataset tar files.",
    )
    parser.add_argument(
        "--split",
        "-s",
        default=None,
        const=None,
        nargs="?",
        choices=["training", "testing", "validation", None],
        help="Split of the dataset. If a specific split is provided, it will search input_dir / split, \
                            If None, it will assume you have all tfrecord files in the input directory without split.",
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip video data parsing, which will skip the generation of video (image) related attributes in the output webdataset.",
    )
    parser.add_argument(
        "--skip_lidar",
        action="store_true",
        help="Skip lidar data parsing, which will skip the generation of 'dynamic_object_points_canonical' attribute in the output webdataset.",
    )
    parser.add_argument(
        "--node_split",
        default="0,1",
        help="The node split. For example, 0,1 means there is 1 node and we are processing in first node",
    )
    parser.add_argument(
        "--specify_segments",
        default=[],
        nargs="+",
        help="Specify segments to process. If None, it will process all segments.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers to process the dataset. If not specified, defaults to CPU core count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )

    args = parser.parse_args()

    # Auto-detect CPU cores if num_workers not specified
    if args.num_workers is None:
        cpu_count = os.cpu_count()
        args.num_workers = os.cpu_count() // 2
        print(
            f"\n\nAuto-detected {cpu_count} CPU cores, using {args.num_workers} workers\n\n"
        )

    converter = WaymoToWebdatasetConverter(
        args.input_dir,
        args.num_workers,
        args.node_split,
        args.skip_lidar,
        args.skip_video,
        args.overwrite,
    )
    converter.extract_all(args.specify_segments, args.output_dir)
