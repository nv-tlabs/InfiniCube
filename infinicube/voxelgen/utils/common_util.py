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
import importlib
import os
from pathlib import Path

import omegaconf
import pytorch_lightning as pl
import torch
from fvdb import GridBatch
from loguru import logger
from omegaconf import OmegaConf
from pycg import exp

from infinicube.voxelgen.utils import wandb_util


def batch2device(batch, device):
    """Send a batch to GPU"""
    if batch is None:
        return None
    for k, v in batch.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            batch[k] = [v[i].to(device) for i in range(len(v))]
        elif isinstance(v, GridBatch):
            batch[k] = v.to(device)
        elif isinstance(v, dict):
            batch[k] = batch2device(v, device)
    return batch


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser


def create_model_from_args(
    args_ckpt, known_args, parser, ckpt_name=None, hparam_update=None
):
    wdb_run, args_ckpt = wandb_util.get_wandb_run(
        args_ckpt, wdb_base=known_args.wandb_base, default_ckpt="test_auto"
    )
    logger.info(f"Use wandb run: {wdb_run.name}")
    assert args_ckpt is not None, "Please specify checkpoint version!"
    assert args_ckpt.exists(), "Selected checkpoint does not exist!"

    model_args = omegaconf.OmegaConf.create(
        wandb_util.recover_from_wandb_config(wdb_run.config)
    )
    args = parser.parse_args(additional_args=model_args)
    if hasattr(args, "nosync"):
        # Force not to sync to shorten bootstrap time.
        os.environ["NO_SYNC"] = "1"

    net_module = importlib.import_module(
        "infinicube.voxelgen.models." + args.model
    ).Model
    ckpt_path = args_ckpt

    if ckpt_name is not None:
        ckpt_path = str(ckpt_path).replace("last", ckpt_name)
        logger.info(f"Use ckpt: {ckpt_path}")

    print(f"Load model from {ckpt_path}")
    # change model config here
    if hparam_update is not None:
        for k, v in hparam_update.items():
            OmegaConf.update(args, k, v)
    net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args, strict=False)

    # get net_state_dict
    net_state_dict = torch.load(ckpt_path, map_location="cpu")
    # get global_step
    global_step = net_state_dict["global_step"]

    return net_model.eval(), args, global_step


def create_model_from_local_config(
    config_path,
    checkpoint_path=None,
    hparam_update=None,
    base_config_path="infinicube/voxelgen/configs/default/param.yaml",
):
    """
    从本地config.yaml和checkpoint加载模型，不依赖wandb

    Args:
        config_path (str): 本地config.yaml文件路径
        checkpoint_path (str, optional): 本地checkpoint文件路径。如果为None，则只实例化模型而不加载权重
        hparam_update (dict, optional): 需要更新的超参数字典
        base_config_path (str): 基础config路径，默认为default/param.yaml

    Returns:
        tuple: (net_model, args, global_step)
            - net_model: 加载好的模型实例（已设为eval模式）
            - args: 解析后的模型参数
            - global_step: checkpoint中的全局步数（如果没有提供checkpoint则为0）
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    # 如果提供了checkpoint_path，检查其是否存在
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
    else:
        logger.info("No checkpoint provided, will instantiate model.")

    model_parser = exp.ArgumentParserX(base_config_path=base_config_path)
    model_args = model_parser.parse_args([str(config_path)])

    if "hyper" in model_args:
        del model_args["hyper"]

    if hparam_update is not None:
        for k, v in hparam_update.items():
            OmegaConf.update(model_args, k, v)
            logger.info(f"Updated hparam: {k} = {v}")

    net_module = importlib.import_module(
        "infinicube.voxelgen.models." + model_args.model
    ).Model

    if checkpoint_path is not None:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            net_model = net_module.load_from_checkpoint(
                str(checkpoint_path), hparams=model_args, strict=False
            )
            net_state_dict = torch.load(checkpoint_path, map_location="cpu")
            global_step = net_state_dict.get("global_step", 0)

        # if checkpoint file only stores state_dict, it can not be parsed by load_from_checkpoint
        except KeyError:
            logger.info(
                "Checkpoint file only stores state_dict, will instantiate model and load state_dict"
            )
            net_model = net_module(model_args)
            net_model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["state_dict"]
            )
            global_step = 0

        logger.info(f"Model loaded successfully. Global step: {global_step}")
    else:
        logger.info("Instantiating model with random weights")
        net_model = net_module(model_args)
        global_step = 0
        logger.info("Model instantiated successfully")

    return net_model.eval(), model_args, global_step


def mask_image_patches(images: torch.Tensor, P: int, p_mask: float) -> torch.Tensor:
    """
    Masks patches of images with a specified probability.

    Parameters:
        images (torch.Tensor): Input tensor of shape [B, N, H, W, 1].
        P (int): Size of each patch.
        p_mask (float): Probability of masking each patch.

    Returns:
        torch.Tensor: Masked images of the same shape as input.
    """
    B, N, H, W, _ = images.shape

    # Calculate number of patches in height and width
    num_patches_h = H // P
    num_patches_w = W // P

    # Create a random mask for patches
    # Shape [B, N, num_patches_h * num_patches_w]
    random_mask = torch.rand(B, N, num_patches_h, num_patches_w) < p_mask
    random_mask = torch.repeat_interleave(random_mask, P, dim=2)
    random_mask = torch.repeat_interleave(random_mask, P, dim=3)
    random_mask = random_mask.unsqueeze(-1)

    return images * random_mask.to(images.device)
