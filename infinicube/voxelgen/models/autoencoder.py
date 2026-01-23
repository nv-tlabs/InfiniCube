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

import gc

import fvdb
import fvdb.nn as fvnn
import numpy as np
import torch
from torch.autograd import Variable
from loguru import logger
from torch.utils.data import DataLoader
import torch.distributed as dist

from infinicube.voxelgen.models.base_model import BaseModel
from infinicube.voxelgen.data.base import DatasetSpec as DS
from infinicube.voxelgen.data.base import list_collate
from infinicube.voxelgen.data import build_dataset
from infinicube.voxelgen.modules.autoencoding.hparams import hparams_handler
from infinicube.voxelgen.modules.autoencoding.base_encoder import Encoder
from infinicube.voxelgen.modules.autoencoding.losses.base_loss import Loss
from infinicube.voxelgen.utils.voxel_util import offscreen_voxel_list_to_mesh_renderer
from infinicube.voxelgen.modules.autoencoding.sunet import StructPredictionNet

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def lambda_lr_wrapper(it, lr_config, batch_size, accumulate_grad_batches=1):
    return max(
        lr_config["decay_mult"]
        ** (int(it * batch_size * accumulate_grad_batches / lr_config["decay_step"])),
        lr_config["clip"] / lr_config["init"],
    )


class Model(BaseModel):
    def __init__(self, hparams):
        hparams = hparams_handler(hparams)  # set up hparams automatically
        super().__init__(hparams)
        self.encoder = Encoder(self.hparams)
        self.unet = eval(self.hparams.network.unet.target)(
            cut_ratio=self.hparams.cut_ratio,
            with_normal_branch=self.hparams.with_normal_branch,
            with_semantic_branch=self.hparams.with_semantic_branch,
            **self.hparams.network.unet.params,
        )
        self.loss = Loss(self.hparams)

        # load pretrained weight
        if self.hparams.pretrained_weight is not None:
            logger.info(f"load pretrained weight from {self.hparams.pretrained_weight}")
            checkpoint = torch.load(self.hparams.pretrained_weight, map_location="cpu")
            missing_keys, unexpected_keys = self.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            logger.info(f"missing_keys: {missing_keys}")
            logger.info(f"unexpected_keys: {unexpected_keys}")

        # using for testing time
        self.reconstructor = None

    def build_hash_tree(self, input_xyz):
        if self.hparams.use_fvdb_loader:
            if isinstance(input_xyz, dict):
                return input_xyz
            return self.build_hash_tree_from_grid(input_xyz)

        return self.build_hash_tree_from_points(input_xyz)

    def build_hash_tree_from_points(self, input_xyz):
        if isinstance(input_xyz, torch.Tensor):
            input_xyz = fvdb.JaggedTensor(input_xyz)
        elif isinstance(input_xyz, fvdb.JaggedTensor):
            pass
        else:
            raise NotImplementedError

        hash_tree = {}
        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break
            voxel_size = [sv * 2**depth for sv in self.hparams.voxel_size]
            origins = [sv / 2.0 for sv in voxel_size]
            hash_tree[depth] = fvdb.gridbatch_from_nearest_voxels_to_points(
                input_xyz, voxel_sizes=voxel_size, origins=origins
            )
        return hash_tree

    def build_hash_tree_from_grid(self, input_grid):
        hash_tree = {}
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())

        for depth in range(self.hparams.tree_depth):
            if depth != 0 and not self.hparams.use_hash_tree:
                break
            voxel_size = [sv * 2**depth for sv in self.hparams.voxel_size]
            origins = [sv / 2.0 for sv in voxel_size]

            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.gridbatch_from_nearest_voxels_to_points(
                    input_xyz, voxel_sizes=voxel_size, origins=origins
                )
        return hash_tree

    def forward(self, batch, out: dict):
        self.generate_fvdb_grid_on_the_fly(batch)

        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({"input_grid": input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        unet_res, unet_output, dist_features = self.unet(batch, unet_feat, hash_tree)

        out.update({"tree": unet_res.structure_grid})
        out.update(
            {
                "structure_features": unet_res.structure_features,
                "dist_features": dist_features,
            }
        )
        out.update({"gt_grid": input_grid})
        out.update({"gt_tree": hash_tree})

        if self.hparams.with_normal_branch:
            out.update(
                {
                    "normal_features": unet_res.normal_features,
                }
            )
        if self.hparams.with_semantic_branch:
            out.update(
                {
                    "semantic_features": unet_res.semantic_features,
                }
            )
        if self.hparams.with_color_branch:
            out.update(
                {
                    "color_features": unet_res.color_features,
                }
            )
        return out

    def on_validation_epoch_start(self):
        pass

    def train_val_step(self, batch, batch_idx, is_val):
        out = {"idx": batch_idx}
        out = self(batch, out)

        if out is None and not is_val:
            return None

        loss_dict, metric_dict, latent_dict = self.loss(
            batch,
            out,
            compute_metric=is_val,
            global_step=self.global_step,
            current_epoch=self.current_epoch,
        )

        if not is_val:
            self.log_dict_prefix("train_loss", loss_dict)
            self.log_dict_prefix("train_loss", latent_dict)
            if self.hparams.enable_anneal:
                self.log("anneal_kl_weight", self.loss.get_kl_weight(self.global_step))
        else:
            self.log_dict_prefix("val_metric", metric_dict)
            self.log_dict_prefix("val_loss", loss_dict)
            self.log_dict_prefix("val_loss", latent_dict)

        # log the image
        if self.trainer.global_rank == 0 and batch_idx % 500 == 0 and self.hparams.voxel_size[0] >= 0.2:  # only log the image on rank 0
            pick_b = np.random.randint(0, batch[DS.INPUT_PC].grid_count)

            pred_grid = out["tree"][0][pick_b]
            if pred_grid.num_voxels.item() == 0:
                logger.warning(f"pred_grid has no voxels at batch index {batch_idx} when {'validation' if is_val else 'training'}")
            else:
                semantic_prob = out["semantic_features"][-1].data[pick_b].jdata
                pred_semantic = torch.argmax(semantic_prob, dim=-1)

                gt_grid = batch[DS.INPUT_PC][pick_b]
                gt_semantic = batch[DS.GT_SEMANTIC][pick_b]

                # hack to fix runtime error: voxelSizes array does not have the same size as the number of grids, got 1 expected 4
                gt_grid_ = fvdb.GridBatch().to(gt_grid.device)
                (
                    gt_grid_.set_from_ijk(
                        gt_grid.ijk.jdata,
                        voxel_sizes=gt_grid.voxel_sizes,
                        origins=gt_grid.origins,
                    ),
                )

                pred_grid_ = fvdb.GridBatch().to(pred_grid.device)
                (
                    pred_grid_.set_from_ijk(
                        pred_grid.ijk.jdata,
                        voxel_sizes=pred_grid.voxel_sizes,
                        origins=pred_grid.origins,
                    ),
                )

                rendered = offscreen_voxel_list_to_mesh_renderer(
                    [(gt_grid_, gt_semantic), (pred_grid_, pred_semantic)],
                    palette="waymo"
                )

                if is_val:
                    self.log_image("img/sample_val", rendered)
                else:
                    self.log_image("img/sample_train", rendered)

        loss_sum = loss_dict.get_sum()
        self.log("val_loss" if is_val else "train_loss/sum", loss_sum)
        self.log("val_step", self.global_step)

        return loss_sum

    def test_step(self, batch, batch_idx):
        self.log("source", batch[DS.SHAPE_NAME][0])
        out = {"idx": batch_idx}
        out = self(batch, out)
        loss_dict, metric_dict, latent_dict = self.loss(
            batch,
            out,
            compute_metric=True,
            global_step=self.trainer.global_step,
            current_epoch=self.current_epoch,
        )
        self.log_dict(loss_dict)
        self.log_dict(metric_dict)
        self.log_dict(latent_dict)

    def train_dataloader(self):
        

        train_set = build_dataset(
            self.hparams.train_dataset,
            self.get_dataset_spec(),
            self.hparams,
            self.hparams.train_kwargs,
        )
        torch.manual_seed(0)
        return DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True
            if not isinstance(train_set, torch.utils.data.IterableDataset)
            else False,
            num_workers=self.hparams.train_val_num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        val_set = build_dataset(
            self.hparams.val_dataset,
            self.get_dataset_spec(),
            self.hparams,
            self.hparams.val_kwargs,
        )
        return DataLoader(
            val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self):
        self.hparams.test_kwargs.resolution = (
            self.hparams.resolution
        )  # ! use for testing when training on X^3 but testing on Y^3

        test_set = build_dataset(
            self.hparams.test_dataset,
            self.get_dataset_spec(),
            self.hparams,
            self.hparams.test_kwargs,
        )
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        return DataLoader(
            test_set,
            batch_size=1,
            shuffle=self.hparams.test_set_shuffle,
            num_workers=0,
            collate_fn=self.get_collate_fn(),
        )

    def get_dataset_spec(self):
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC, DS.GT_DENSE_PC]
        if self.hparams.use_input_normal:
            all_specs.append(DS.TARGET_NORMAL)
            all_specs.append(DS.GT_DENSE_NORMAL)
        if self.hparams.use_input_semantic or self.hparams.with_semantic_branch:
            all_specs.append(DS.GT_SEMANTIC)
        if self.hparams.use_input_intensity:
            all_specs.append(DS.INPUT_INTENSITY)
        return all_specs

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [("val_loss", True)]

    def configure_optimizers(self):
        # overwrite this from base model to fix pretrained vae layer
        lr_config = self.hparams.learning_rate
        # parameters = list(self.parameters())
        parameters = list(self.encoder.parameters()) + list(self.unet.parameters())

        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                lr=lr_config["init"],
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "Adam":
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(
                parameters,
                lr=lr_config["init"],
                weight_decay=self.hparams.weight_decay,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        # build scheduler
        import functools

        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=functools.partial(
                lambda_lr_wrapper,
                lr_config=lr_config,
                batch_size=self.hparams.batch_size,
                accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            ),
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @torch.no_grad()
    def _encode(self, batch, use_mode=False):
        self.generate_fvdb_grid_on_the_fly(batch)
        input_xyz = batch[DS.INPUT_PC]
        hash_tree = self.build_hash_tree(input_xyz)
        input_grid = hash_tree[0]
        batch.update({"input_grid": input_grid})

        if not self.hparams.use_hash_tree:
            hash_tree = None

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        _, x, mu, log_sigma = self.unet.encode(
            unet_feat, hash_tree=hash_tree, batch=batch
        )
        if use_mode:
            sparse_feature = mu
        else:
            sparse_feature = reparametrize(mu, log_sigma)

        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(sparse_feature))
