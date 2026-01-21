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
Generate metadata.csv file for DiffSynth-Studio training.

Collects video files and their corresponding caption files, then generates
a metadata.csv file in the format required by DiffSynth-Studio.
"""

import argparse
import csv
import json
import os
from typing import List, Tuple


def load_caption(caption_path: str) -> str:
    """Load caption text from JSON file."""
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            original_caption = data.get("weather", list(data.values())[0] if data else "")
            caption = "This video is capture by a camera mounted on a vehicle. " + original_caption 
            return caption
    except Exception as e:
        print(f"Warning: Failed to read {caption_path}: {e}")
        return ""


def load_allowed_clips(json_path: str) -> set:
    """Load allowed clip names from JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            clips = json.load(f)
            return set(clips)
    except Exception as e:
        print(f"Warning: Failed to read allowed clips from {json_path}: {e}")
        return set()


def get_video_caption_pairs(video_dir: str, caption_dir: str, allowed_clips: set) -> List[Tuple[str, str]]:
    """Match video files with their corresponding caption files."""
    video_exts = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm"}
    video_files = [
        f
        for f in os.listdir(video_dir)
        if any(f.lower().endswith(ext) for ext in video_exts)
    ]

    print(f"Found {len(video_files)} video files")
    print(f"Allowed clips: {len(allowed_clips)}")

    pairs = []
    matched = unmatched = 0
    filtered = 0

    for video_file in sorted(video_files):
        base_name = os.path.splitext(video_file)[0]
        
        # Check if clip is in allowed list
        if allowed_clips and base_name not in allowed_clips:
            filtered += 1
            continue
        
        caption_file = os.path.join(caption_dir, f"{base_name}.json")

        if os.path.exists(caption_file):
            caption = load_caption(caption_file)
            semantic_buffer_video = (
                f"../semantic_buffer_video_480p_front/{base_name}.mp4"
            )
            depth_buffer_tar = f"../voxel_depth_100_480p_front/{base_name}.tar"
            intrinsic_tar = f"../intrinsic/{base_name}.tar"
            pose_tar = f"../pose/{base_name}.tar"

            if caption:
                pairs.append(
                    (
                        video_file,
                        caption,
                        semantic_buffer_video,
                        depth_buffer_tar,
                        intrinsic_tar,
                        pose_tar,
                    )
                )
                matched += 1
            else:
                print(f"Warning: Empty caption for {video_file}")
                unmatched += 1
        else:
            print(f"Warning: Caption not found for {video_file}")
            unmatched += 1

    print(f"\nMatched: {matched} | Unmatched: {unmatched} | Filtered: {filtered}")
    return pairs


def generate_metadata_csv(pairs: List[Tuple[str, str]], output_path: str) -> None:
    """Generate metadata.csv file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video",
                "prompt",
                "semantic_buffer_video",
                "depth_buffer_tar",
                "camera_intrinsic",
                "camera_pose",
            ]
        )
        writer.writerows(pairs)

    print(f"\nSuccessfully generated: {output_path}")
    print(f"Total records: {len(pairs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate metadata.csv for DiffSynth-Studio training"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/",
        help="Root directory containing video and caption data",
    )
    parser.add_argument(
        "--video_subdir",
        type=str,
        default="video_480p_front",
        help="Subdirectory name or absolute path for videos (default: video_480p_front)",
    )
    parser.add_argument(
        "--caption_subdir",
        type=str,
        default="caption",
        help="Subdirectory name or absolute path for captions (default: caption)",
    )
    parser.add_argument(
        "--allowed_clips_json",
        type=str,
        default="infinicube/assets/waymo_split/video_training_good.json",
        help="Path to JSON file containing allowed clip names (default: infinicube/assets/waymo_split/video_training_good.json)",
    )

    args = parser.parse_args()

    video_dir = os.path.join(args.data_root, args.video_subdir)
    caption_dir = os.path.join(args.data_root, args.caption_subdir)
    output_path = os.path.join(video_dir, "metadata.csv")
    
    # Load allowed clips
    allowed_clips = load_allowed_clips(args.allowed_clips_json)

    print("=" * 60)
    print("DiffSynth-Studio Metadata Generator")
    print("=" * 60)
    print(f"\nVideo directory: {video_dir}")
    print(f"Caption directory: {caption_dir}")
    print(f"Allowed clips JSON: {args.allowed_clips_json}")
    print(f"Output file: {output_path}\n")

    if not os.path.exists(video_dir):
        print(f"Error: Video directory does not exist: {video_dir}")
        return

    if not os.path.exists(caption_dir):
        print(f"Error: Caption directory does not exist: {caption_dir}")
        return
    
    if not allowed_clips:
        print(f"Warning: No allowed clips loaded from {args.allowed_clips_json}")
        print("Proceeding without filtering...\n")

    pairs = get_video_caption_pairs(video_dir, caption_dir, allowed_clips)

    if not pairs:
        print("\nError: No valid video-caption pairs found")
        return

    generate_metadata_csv(pairs, output_path)

    print("\nFirst 5 records:")
    print("-" * 60)
    for i, (video, caption, _, _, _, _) in enumerate(pairs[:5], 1):
        print(f"{i}. {video}")
        print(f"   Prompt: {caption[:100]}{'...' if len(caption) > 100 else ''}\n")


if __name__ == "__main__":
    main()
