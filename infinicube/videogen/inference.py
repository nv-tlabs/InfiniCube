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
Video Generation Inference API
Based on DiffSynth-Studio's Wan2.1-14B-Buffer-Control model
"""

from typing import List, Optional

import numpy as np
import torch
from diffsynth import load_state_dict, save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from PIL import Image


class WanVideoGenerator:
    """
    Wan Video Generator - Video generation based on semantic and coordinate buffers

    Args:
        checkpoint_path: Path to trained model checkpoint (contains buffer_embedder and dit weights)
        device: Device to run on, default "cuda:0"
        torch_dtype: Torch data type, default torch.bfloat16
        buffer_channels: Number of channels for buffer embedder, default 16
        enable_vram_management: Whether to enable VRAM management, default True
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        buffer_channels: int = 16,
        enable_vram_management: bool = True,
        use_wan_1pt3b: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.buffer_channels = buffer_channels

        # Load base model
        if use_wan_1pt3b:
            print("Loading Wan2.1-T2V-1.3B base model...")
        else:
            print("Loading Wan2.1-T2V-14B base model...")

        if use_wan_1pt3b:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", skip_download=True),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", skip_download=True),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", skip_download=True),
                ],
            )
        else:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch_dtype,
                device=device,
                model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", skip_download=True),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", skip_download=True),
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", skip_download=True),
                ],
            )


        # Initialize buffer embedder
        print(f"Initializing buffer embedder (channels={buffer_channels})...")
        self.pipe.initialize_buffer_embedder(
            buffer_channels=buffer_channels, zero_init=True
        )

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self._load_checkpoint()

        # Enable VRAM management
        if enable_vram_management:
            print("Enabling VRAM management...")
            self.pipe.enable_vram_management()

        print("✓ WanVideoGenerator initialization complete")

    def _load_checkpoint(self):
        """Load trained checkpoint"""
        state_dict = load_state_dict(self.checkpoint_path)

        # Load buffer_embedder weights
        if self.pipe.buffer_embedder is not None:
            buffer_embedder_state = {
                k.replace("buffer_embedder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("buffer_embedder.")
            }
            if buffer_embedder_state:
                self.pipe.buffer_embedder.load_state_dict(buffer_embedder_state)
                print(
                    f"  ✓ Buffer embedder weights loaded, {len(buffer_embedder_state)} parameters"
                )
            else:
                print("  ⚠ Warning: buffer_embedder weights not found in checkpoint")

        # Load DiT weights (if included in checkpoint)
        dit_state = {
            k.replace("dit.", ""): v
            for k, v in state_dict.items()
            if k.startswith("dit.")
        }
        if dit_state:
            self.pipe.dit.load_state_dict(dit_state, strict=False)
            print(f"  ✓ DiT weights loaded, {len(dit_state)} parameters")

    def _ndarray_to_pil_list(self, buffer_array: np.ndarray) -> List[Image.Image]:
        """
        Convert numpy array to list of PIL Images

        Args:
            buffer_array: Numpy array with shape (N, H, W, 3) and dtype=uint8

        Returns:
            List[Image.Image]: List of PIL Images
        """
        if not isinstance(buffer_array, np.ndarray):
            raise TypeError(
                f"buffer_array must be numpy.ndarray, got {type(buffer_array)}"
            )

        if buffer_array.ndim != 4 or buffer_array.shape[-1] != 3:
            raise ValueError(
                f"buffer_array shape must be (N, H, W, 3), got {buffer_array.shape}"
            )

        if buffer_array.dtype != np.uint8:
            raise TypeError(
                f"buffer_array dtype must be uint8, got {buffer_array.dtype}"
            )

        # Convert each frame to PIL Image
        pil_list = []
        for i in range(buffer_array.shape[0]):
            frame = buffer_array[i]  # (H, W, 3)
            pil_image = Image.fromarray(frame, mode="RGB")
            pil_list.append(pil_image)

        return pil_list

    def generate(
        self,
        semantic_buffer: np.ndarray,
        coordinate_buffer: np.ndarray,
        prompt: str = "The video is about a driving scene captured at daytime. The weather is clear.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed: int = 0,
        tiled: bool = True,
        output_path: Optional[str] = None,
        fps: int = 10,
        quality: int = 8,
    ) -> List[Image.Image]:
        """
        Generate video

        Args:
            semantic_buffer: Semantic buffer, numpy array with shape (N, H, W, 3) and dtype=uint8
            coordinate_buffer: Coordinate buffer, numpy array with shape (N, H, W, 3) and dtype=uint8
            prompt: Text prompt
            negative_prompt: Negative prompt
            seed: Random seed
            tiled: Whether to use tiled mode
            output_path: Output video path (optional), will auto-save if provided
            fps: Video framerate (only used when output_path is provided)
            quality: Video quality (only used when output_path is provided)

        Returns:
            List[Image.Image]: List of generated video frames (PIL Images)
        """
        # Validate input
        if semantic_buffer.shape != coordinate_buffer.shape:
            raise ValueError(
                f"semantic_buffer and coordinate_buffer must have the same shape, "
                f"got {semantic_buffer.shape} and {coordinate_buffer.shape}"
            )

        num_frames, height, width, channels = semantic_buffer.shape

        print("\nStarting video generation...")
        print(f"  - Prompt: {prompt}")
        print(f"  - Frames: {num_frames}")
        print(f"  - Resolution: {height}x{width}")
        print(f"  - Seed: {seed}")
        print(f"  - Tiled: {tiled}")

        # Convert buffers to PIL Image lists
        print("Converting buffer data...")
        semantic_buffer_list = self._ndarray_to_pil_list(semantic_buffer)
        coordinate_buffer_list = self._ndarray_to_pil_list(coordinate_buffer)

        # Execute inference
        print("Executing video generation...")
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            semantic_buffer_video=semantic_buffer_list,
            coordinate_buffer_video=coordinate_buffer_list,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            tiled=tiled,
        )

        # Save video if output path is provided
        if output_path is not None:
            print(f"Saving video to: {output_path}")
            save_video(video, output_path, fps=fps, quality=quality)
            print("✓ Video saved")

        print(f"✓ Video generation complete ({len(video)} frames)")

        return video

    def __call__(self, *args, **kwargs):
        """Make instance callable like a function"""
        return self.generate(*args, **kwargs)
