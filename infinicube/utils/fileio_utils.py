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

import io
from pathlib import Path
from typing import Dict, List, Union

import imageio.v3 as iio
import numpy as np
from decord import VideoReader


def read_video_file(
    video_path: str,
    indices: Union[int, list, str, None] = None,
):
    """
    Read video file.
    Args:
        video_path: Path to the video file.
        indices:
            1) int, index of the frame to read.
            2) list, indices of the frames to read.
            3) str, 'all', read the whole video.
            4) None, return the VideoReader object.
    Returns:
        VideoReader: VideoReader object. If indices is not None, return the frames as numpy array.
    """
    vr = VideoReader(video_path)

    if indices is None:
        return vr

    if isinstance(indices, int):
        frames = vr[indices]
    elif isinstance(indices, list):
        frames = vr.get_batch(indices)
    elif indices == "all":
        frames = vr.get_batch(range(len(vr)))
    else:
        raise ValueError(f"Invalid indices: {indices}")

    return frames.asnumpy()


def write_video_file(
    frames: Union[np.ndarray, List, Dict],
    output_file: Union[str, Path],
    fps: int = 30,
    use_jiahui_params: bool = True,
):
    """
    Args:
    frames: can be
        1) list of np.ndarray shape [H, W, 3]
        2) np.ndarray shape [N, H, W, 3]
        3) dict, key is about frame index, value is np.ndarray shape [H, W, 3]

    Returns:
        output: bytes
    """
    if isinstance(output_file, Path):
        output_file = output_file.as_posix()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if not output_file.endswith(".mp4"):
        output_file = Path(output_file).with_suffix(".mp4").as_posix()

    # as list like ['-option1', 'value1', '-option2', 'value2'].
    if use_jiahui_params:
        output_params = [
            "-preset",
            "veryslow",
            "-crf",
            "23.5",
            "-g",
            "250",
            "-bf",
            "3",
            "-sc_threshold",
            "60",
            "-qcomp",
            "0.5",
            "-psy-rd",
            "0.3:0",
            "-aq-mode",
            "2",
            "-aq-strength",
            "0.8",
            "-me_method",
            "umh",
            "-flags",
            "+cgop",
            "-movflags",
            "+faststart",
        ]
    else:
        output_params = []

    assert len(frames) > 0

    if isinstance(frames, np.ndarray):
        frames = [frame for frame in frames]

    if isinstance(frames, dict):
        new_frames = []
        keys = sorted(frames.keys())
        if "__key__" in keys:
            keys.remove("__key__")  # remove __key__ if exists

        for key in keys:
            if type(frames[key]) == bytes:
                new_frames.append(iio.imread(io.BytesIO(frames[key])))
            else:
                new_frames.append(frames[key])
        frames = new_frames

    # use imageio to write video, it will automatically use ffmpeg plugin
    iio.imwrite(
        output_file,
        frames,
        plugin="FFMPEG",
        fps=fps,
        codec="libx264",
        output_params=output_params,
    )


def read_fvdb_grid_and_semantic(path):
    """
    Args:
        path (str): path to the fvdb grid file
        coarsen (int): coarsen the grid by this factor. if 1, no coarsening

    Returns:
        {
            'voxel_centers': np.ndarray, shape [N, 3],
            'voxel_corners': np.ndarray, shape [N, 2, 3], note that voxels must be axis aligned! so 2 corners are enough
            'semantics': np.ndarray, shape [N,]
         }
    """
    import pickle

    import torch

    from infinicube import get_sample

    if path.endswith(".pt"):
        data = torch.load(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".tar"):
        sample = get_sample(path)
        if "pcd.vs01.pth" in sample:
            data = sample["pcd.vs01.pth"]
        else:
            raise ValueError("Only support pcd.vs01.pth file format for now.")
    else:
        raise ValueError(f"Unknown file format: {path}")

    if isinstance(data, dict):
        if "points" in data:
            fvdb_grid = data["points"]
        elif "grid" in data:
            fvdb_grid = data["grid"]
        else:
            raise ValueError(f"Unknown fvdb grid key name: {data.keys()}")

        voxel_centers = (
            fvdb_grid.grid_to_world(fvdb_grid.ijk.float()).jdata.cpu().numpy()
        )
        voxel_corners = np.stack(
            [
                fvdb_grid.grid_to_world(fvdb_grid.ijk.float() - 0.5)
                .jdata.cpu()
                .numpy(),
                fvdb_grid.grid_to_world(fvdb_grid.ijk.float() + 0.5)
                .jdata.cpu()
                .numpy(),
            ],
            axis=1,
        )
        N = voxel_centers.shape[0]

        if "semantics" in data:
            semantics = data["semantics"].cpu().numpy()
        elif "semantic" in data:
            semantics = data["semantic"].cpu().numpy()
        else:
            semantics = np.zeros(N, dtype=np.int32)

        voxel_dict = {
            "ijk": fvdb_grid.ijk.jdata.cpu().numpy(),
            "voxel_centers": voxel_centers,
            "voxel_corners": voxel_corners,
            "semantics": semantics,
        }

    print("reading fvdb grid done")

    return voxel_dict
