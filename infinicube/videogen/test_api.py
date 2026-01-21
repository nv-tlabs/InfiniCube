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
Test script for WanVideoGenerator API
Using the same inputs as the original script
"""

import cv2
import numpy as np
import torch

from infinicube.videogen import WanVideoGenerator


def load_video_as_numpy(video_path):
    """Load video file as numpy array"""
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")

    video_array = np.stack(frames, axis=0)  # (N, H, W, 3)
    print(f"  Loaded {video_array.shape[0]} frames, shape: {video_array.shape}")

    return video_array


def main():
    """Test the WanVideoGenerator API with original script inputs"""

    # Paths from original script (DiffSynth-Studio directory)
    base_path = "/home/yiflu/holodeck/yiflu/DiffSynth-Studio"

    checkpoint_path = (
        f"{base_path}/models/wan_buffer_trained_14B/baselr_2e-5_bufferlr_2e-4_zero_init/step-1050.safetensors"
    )
    semantic_buffer_video_path = (
        f"{base_path}/assets/semantic_buffer_video_480p_front.mp4"
    )
    coordinate_buffer_video_path = (
        f"{base_path}/assets/coordinate_buffer_video_480p_front.mp4"
    )

    # Prompt from original script
    prompt = (
        "The video is about a driving scene captured at daytime. The weather is clear."
    )
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    print("=" * 80)
    print("Testing WanVideoGenerator API")
    print("=" * 80)

    # Load buffer videos
    print("\n[1/4] Loading buffer videos...")
    semantic_buffer = load_video_as_numpy(semantic_buffer_video_path)
    coordinate_buffer = load_video_as_numpy(coordinate_buffer_video_path)

    # Verify buffer shapes match
    assert semantic_buffer.shape == coordinate_buffer.shape, (
        f"Buffer shapes don't match: {semantic_buffer.shape} vs {coordinate_buffer.shape}"
    )

    # Crop to 93 frames as in original script
    num_frames = 93
    if semantic_buffer.shape[0] >= num_frames:
        print(f"\nCropping to {num_frames} frames...")
        semantic_buffer = semantic_buffer[:num_frames]
        coordinate_buffer = coordinate_buffer[:num_frames]
    else:
        print(
            f"\nWarning: Video has only {semantic_buffer.shape[0]} frames, expected {num_frames}"
        )

    print(f"Final buffer shape: {semantic_buffer.shape}")

    # Initialize generator
    print("\n[2/4] Initializing WanVideoGenerator...")
    generator = WanVideoGenerator(
        checkpoint_path=checkpoint_path,
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        buffer_channels=16,
        enable_vram_management=True,
    )

    # Generate video
    print("\n[3/4] Generating video...")
    output_path = "output_buffer_controlled_video_test.mp4"

    video_frames = generator.generate(
        semantic_buffer=semantic_buffer,
        coordinate_buffer=coordinate_buffer,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=0,
        tiled=True,
        output_path=output_path,
        fps=10,
        quality=8,
    )

    print("\n[4/4] Test complete!")
    print(f"✓ Generated {len(video_frames)} frames")
    print(f"✓ Video saved to: {output_path}")
    print("\n" + "=" * 80)
    print("SUCCESS: WanVideoGenerator API test passed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Test failed with exception:")
        print(f"{type(e).__name__}: {e}")
        print("=" * 80)
        raise
