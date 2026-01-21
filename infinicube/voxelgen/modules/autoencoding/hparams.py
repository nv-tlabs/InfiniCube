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


def hparams_handler(hparams):
    # !: anisotropic setting
    if not hasattr(hparams, "remain_h"):
        hparams.remain_h = False
    if isinstance(hparams.voxel_size, int) or isinstance(hparams.voxel_size, float):
        hparams.voxel_size = [hparams.voxel_size] * 3

    # !: pretrain weight
    if not hasattr(hparams, "pretrained_weight"):
        hparams.pretrained_weight = None

    hparams.use_input_color = False
    hparams.with_color_branch = False
    hparams.supervision.color_weight = 0.0

    hparams.with_normal_branch = False
    if not hasattr(hparams.supervision, "normal_weight"):
        hparams.supervision.normal_weight = 0.0
    if hparams.supervision.normal_weight > 0:
        hparams.with_normal_branch = True

    hparams.with_semantic_branch = False
    if not hasattr(hparams.supervision, "semantic_weight"):
        hparams.supervision.semantic_weight = 0.0
    if hparams.supervision.semantic_weight > 0:
        hparams.with_semantic_branch = True

    if not hasattr(hparams.supervision, "adaptive_structure_weight"):
        hparams.supervision.adaptive_structure_weight = False

    return hparams
