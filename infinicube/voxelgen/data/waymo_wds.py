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

import numpy as np
import webdataset as wds
import json
import torch
import random
import os
import traceback
import copy
from omegaconf import OmegaConf
from torch.distributed import get_rank, get_world_size
from infinicube.voxelgen.data.base import DatasetSpec as DS
from webdataset.tariterators import url_opener, tarfile_samples
from webdataset import autodecode
from termcolor import colored, cprint
from infinicube.utils.wds_utils import my_imagehandler, ply_decoder
from infinicube.utils.semantic_utils import WAYMO_CATEGORY_NAMES
from infinicube.utils.mesh_utils import build_scene_mesh_from_all_object_info
from infinicube.utils.bbox_utils import build_scene_bounding_boxes_from_object_info
from infinicube.camera.base import flu_to_opencv, opencv_to_flu, CameraBase
from infinicube.camera.pinhole import PinholeCamera
from pytorch3d.ops.iou_box3d import box3d_overlap

CAMERA_ID_TO_NAME = {
    0: 'front',
    1: 'front_left',
    2: 'front_right',
    3: 'side_left',
    4: 'side_right'
}


class WaymoWdsDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, wds_root_url, wds_scene_list_file, attr_subfolders=['image_front'],
        spec=None, transforms=None, dino_slect_ids=[], dino_data_root=None, split='train', 
        frame_start_num=30, frame_end_num=170, front_view_input_wh=(832, 480),
        grid_crop_bbox_min=[-10.24, -51.2, -12.8], grid_crop_bbox_max=[92.16, 51.2, 38.4],
        input_slect_ids=[0,1,2], input_frame_offsets=[0], offset_unit='frame', # frame or meter
        sup_slect_ids=[0,1,2], sup_frame_offsets=[0], n_image_per_iter_sup=None,
        fvdb_grid_type='vs01', finest_voxel_size_goal='vs01',
        val_starting_frame=50, map_types=None,
        random_seed=0, shuffle_buffer=128, input_depth_type='voxel_depth_100',
        hparams=None,
        grid_crop_augment=False, grid_crop_augment_range=[12.8, 12.8, 3.2],
        replace_all_car_with_cad=False,
        **kwargs
    ):
        super().__init__()

        self.split = split
        self.wds_root_url = wds_root_url
        self.attr_subfolders = attr_subfolders
        self.shuffle_buffer = shuffle_buffer
        self.hparams = hparams
        self.front_view_input_wh = front_view_input_wh
        self.input_slect_ids = input_slect_ids
        self.input_frame_offsets = input_frame_offsets
        self.offset_unit = offset_unit
        self.sup_slect_ids = sup_slect_ids
        self.sup_frame_offsets = sup_frame_offsets
        self.dino_slect_ids = dino_slect_ids
        self.dino_data_root = dino_data_root
        self.input_depth_type = input_depth_type
        self.map_types = map_types
        
        self.grid_crop_bbox_min = grid_crop_bbox_min if isinstance(grid_crop_bbox_min, list) else \
            OmegaConf.to_container(grid_crop_bbox_min)
        self.grid_crop_bbox_max = grid_crop_bbox_max if isinstance(grid_crop_bbox_max, list) else \
            OmegaConf.to_container(grid_crop_bbox_max)

        self.grid_length_in_meter = [self.grid_crop_bbox_max[i] - self.grid_crop_bbox_min[i] for i in range(3)]
        self.grid_half_diagonal = (self.grid_length_in_meter[0]**2 + self.grid_length_in_meter[1]**2)**0.5 / 2
        
        self.val_starting_frame = val_starting_frame
        self.frame_start_num = frame_start_num

        assert self.offset_unit == 'frame', "Only support frame offset unit for now"
        self.last_starting_frame = frame_end_num - max(max(input_frame_offsets), max(sup_frame_offsets)) - 1  # in the case of offset_unit='frame'

        self.sample_time_from_shard = self.last_starting_frame if split == 'train' else 1

        self.spec = spec
        self.fvdb_grid_type = fvdb_grid_type
        self.finest_voxel_size_goal = finest_voxel_size_goal

        if n_image_per_iter_sup is None:
            self.n_image_per_iter_sup = len(sup_slect_ids) * len(sup_frame_offsets)
        else:
            self.n_image_per_iter_sup = n_image_per_iter_sup
        
        # should be .json file. otherwise only for debug use.
        if wds_scene_list_file.endswith('.json'):
            wds_scene_list = json.load(open(wds_scene_list_file, 'r'))
        else: 
            wds_scene_list = [wds_scene_list_file]

        if self.split == 'train':
            # shuffle the wds_scene_list using the random seed
            print(f"Shuffle the wds_scene_list using the random seed {random_seed}")
            random.seed(random_seed)
            random.shuffle(wds_scene_list)

        self.grid_crop_augment = grid_crop_augment
        self.grid_crop_augment_range = grid_crop_augment_range
        self.replace_all_car_with_cad = replace_all_car_with_cad

        self.wds_scene_list = wds_scene_list
        self.CAMERA_ID_TO_NAME = CAMERA_ID_TO_NAME
        
        self.prepare_pre_pipeline()


    def prepare_pre_pipeline(self):
        self.url_opener_custom = url_opener
        self.wds_url_collection = [
            os.path.join(self.wds_root_url, self.attr_subfolders[0], f"{scene}.tar") 
            for scene in self.wds_scene_list
        ]
        print(f"Number of shards: {len(self.wds_url_collection)}")
        
        decode_handlers = [my_imagehandler('npraw'), ply_decoder]
        self.decode_func = autodecode.Decoder(decode_handlers)

    def __len__(self):
        _, world_size, _ = self.get_rank_worker()
        
        if self.split == 'train':
            return len(self.wds_scene_list) * (self.last_starting_frame - self.frame_start_num) // world_size
        else:
            return len(self.wds_scene_list) # each worker reads all the data
        
    def get_rank_worker(self, string=None):
        from torch.utils.data import get_worker_info
        try:
            rank = get_rank()
            world_size = get_world_size()
        except ValueError:  # single gpu
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else: # single process
            worker_id = 0
            
        if string is not None:
            print(f"[{colored('rank='+str(rank), 'cyan')} {colored('worker='+str(worker_id), 'red')}] {string}")
        
        return rank, world_size, worker_id

    
    def _assemble_data(self, attr_subfolder, use_frame_idxs, reassemble_sample, sample):
        # Case 1: per frame data with frame indexing
        if 'image' in attr_subfolder or 'skymask' in attr_subfolder or \
            'pose' in attr_subfolder or 'depth' in attr_subfolder or \
            'buffer' in attr_subfolder or 'object_info' in attr_subfolder:
            for use_frame_idx in use_frame_idxs:
                key_names = [x for x in sample.keys() if f"{use_frame_idx:06d}" in x]
                for key_name in key_names:
                    reassemble_sample[key_name] = sample[key_name]

        # Case 2: per scene data without frame indexing
        elif 'grid' in attr_subfolder:
            key_name = [x for x in sample.keys() if 'grid' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif 'pc' in attr_subfolder:
            key_name = [x for x in sample.keys() if 'pcd' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif 'intrinsic' in attr_subfolder:
            key_names = [x for x in sample.keys() if 'intrinsic' in x]
            for key_name in key_names:
                reassemble_sample[key_name] = sample[key_name]
        
        elif attr_subfolder.startswith('3d_') and "voxelsize" in attr_subfolder:
            key_name = [x for x in sample.keys() if '.npy' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        elif attr_subfolder == 'dynamic_object_points_canonical':
            key_name = [x for x in sample.keys() if '.npz' in x][0]
            reassemble_sample[key_name] = sample[key_name]

        else:
            raise NotImplementedError(f"attr_subfolder {attr_subfolder} is not implemented yet.")

    def get_depth_images(self, sample, frame_idxs, select_view_ids, sup_image_indices=None, depth_type='depth_buffer_100'):
        """
        Returns:
            torch.Tensor, shape (N, H, W, 1)
        """
        all_depth_images_tensor = []

        if sup_image_indices is None:
            sup_image_indices = np.arange(len(frame_idxs) * len(select_view_ids))

        use_frame_idxs = np.array(frame_idxs)[np.unique(sup_image_indices // len(select_view_ids))]
        
        for ii, frame_idx in enumerate(use_frame_idxs):
            for jj, select_id in enumerate(select_view_ids):
                if ii * len(select_view_ids) + jj not in sup_image_indices:
                    continue

                depth_image = sample[f"{frame_idx:06d}.{depth_type}.{self.CAMERA_ID_TO_NAME[select_id]}.png"] / 100.0
                depth_image = torch.tensor(depth_image).float().unsqueeze(-1)
                
                if select_id > 2: # side view, pad height
                    depth_image_padded = torch.zeros(self.front_view_input_wh[1], self.front_view_input_wh[0], 1)
                    depth_image_padded[:depth_image.shape[0], :, :] = depth_image
                    depth_image = depth_image_padded

                all_depth_images_tensor.append(depth_image)

        return torch.stack(all_depth_images_tensor)

    def set_input_sup_frame_idx(self, reassemble_sample, starting_frame):
        """
            set the input and sup frame idx (json), and return all needed frame idxs for assembling data
        """
        if self.offset_unit == 'frame':
            use_frame_idxs = [starting_frame + offset for offset in self.input_frame_offsets] + \
                    [starting_frame + offset for offset in self.sup_frame_offsets]
            use_frame_idxs = sorted(list(set(use_frame_idxs)))

            # prepare for auto decoding
            reassemble_sample['input_frame_idx.json'] = json.dumps([starting_frame + offset for offset in self.input_frame_offsets]).encode('utf-8')
            reassemble_sample['sup_frame_idx.json'] = json.dumps([starting_frame + offset for offset in self.sup_frame_offsets]).encode('utf-8')

            return use_frame_idxs
        else:
            raise NotImplementedError(f"Unknown offset unit {self.offset_unit} !.")

    def __iter__(self):
        # mannually split by node (rank), different GPU need to load different data.
        rank, world_size, worker_id = self.get_rank_worker()
        if self.split == 'train':
            shards_this_rank = [tar for i, tar in enumerate(self.wds_url_collection) if i % world_size == rank] # divide by node mannually
            dataset_this_rank = wds.DataPipeline([
                wds.SimpleShardList(shards_this_rank * self.sample_time_from_shard), # repeat the shard for multiple times
                wds.shuffle(self.shuffle_buffer),
                wds.split_by_worker, # divide by worker mannually
                tarfile_samples
            ])
        else:
            shards_this_rank = self.wds_url_collection # avoid empty list
            dataset_this_rank = wds.DataPipeline([
                wds.SimpleShardList(shards_this_rank), # each worker reads all the data
                # wds.split_by_worker,                  
                tarfile_samples
            ])

        # since each shard is a complete clip, we can sample it many times (with different starting frame)
        for _ in range(self.sample_time_from_shard):
            for sample in dataset_this_rank:
                # self.get_rank_worker(sample['__key__'])
                if self.split == 'train':
                    starting_frame = random.randint(self.frame_start_num, self.last_starting_frame)
                else:
                    starting_frame = self.val_starting_frame

                reassemble_sample = {}
                reassemble_sample['__key__'] = sample['__key__']
                reassemble_sample['__url__'] = sample['__url__']
                
                try:
                    use_frame_idxs = self.set_input_sup_frame_idx(reassemble_sample, starting_frame)
                except ValueError:
                    print(f"{colored('Skip this sample:', 'red')} {sample['__key__']} " + \
                          f"starting at {starting_frame} seeking offsets {self.input_frame_offsets} & {self.sup_frame_offsets}")
                    continue

                # 1) gather the primary image data
                primary_attr_subfolder = self.attr_subfolders[0]
                self._assemble_data(primary_attr_subfolder, use_frame_idxs, reassemble_sample, sample)
                
                # gather data. 
                for attr_subfolder in self.attr_subfolders[1:]:
                    tarfile_path = reassemble_sample['__url__'].replace(primary_attr_subfolder, attr_subfolder)
                    sample = next(iter(wds.DataPipeline(
                        wds.SimpleShardList(tarfile_path), tarfile_samples
                    )))
                    self._assemble_data(attr_subfolder, use_frame_idxs, reassemble_sample, sample)

                # 2) decode the reassemble_sample data.
                reassemble_sample = self.decode_func(reassemble_sample)

                # 3) data transform
                try:
                    reassemble_sample = self.data_transform(reassemble_sample)
                except Exception as e:
                    print("Error in data transform:")
                    traceback.print_exc()
                    print(f"{colored('Skip this sample:', 'red')} {sample['__key__']}")
                    continue
                
                yield reassemble_sample


    def data_transform(self, sample):
        data = {}
        data[DS.SHAPE_NAME] = sample['__url__'] + \
            "_with_input_frames_" + '_'.join([str(x) for x in sample['input_frame_idx.json']]) + \
            "_with_sup_frames_" + '_'.join([str(x) for x in sample['sup_frame_idx.json']])
        
        grid_to_world = self.crop_pcd_and_generate_grid_raw(sample)

        # data[DS.GRID_TO_FIRST_CAMERA_FLU] = grid_to_first_camera_flu
        data[DS.GRID_TO_WORLD] = grid_to_world
        data[DS.GRID_CROP_RANGE] = torch.tensor([self.grid_crop_bbox_min, self.grid_crop_bbox_max])

        data[DS.INPUT_PC] = "Generate on the fly from DS.INPUT_PC_RAW"
        data[DS.INPUT_PC_RAW] = sample[f"grid_raw.{self.fvdb_grid_type}.pth"] # real pc data to be voxelized
        data[DS.GT_SEMANTIC] = "Generate on the fly from DS.INPUT_PC_RAW"

        if DS.IMAGES_INPUT in self.spec:
            input_image, input_mask, input_pose, input_intrinsic = \
                self.get_images(sample, sample['input_frame_idx.json'],
                                self.input_slect_ids)
            
            data[DS.IMAGES_INPUT] = input_image
            data[DS.IMAGES_INPUT_MASK] = input_mask
            data[DS.IMAGES_INPUT_POSE] = input_pose
            data[DS.IMAGES_INPUT_INTRINSIC] = input_intrinsic

        all_sup_img_num = len(sample['sup_frame_idx.json']) * len(self.sup_slect_ids)
        if self.n_image_per_iter_sup < all_sup_img_num:
            sup_image_indices = np.random.choice(all_sup_img_num, self.n_image_per_iter_sup, replace=False)
        else:
            sup_image_indices = np.arange(all_sup_img_num)

        if DS.IMAGES in self.spec:
            sup_image, sup_mask, sup_pose, sup_intrinsic = \
                self.get_images(sample, sample['sup_frame_idx.json'],
                                self.sup_slect_ids,
                                sup_image_indices)
            
            data[DS.IMAGES] = sup_image
            data[DS.IMAGES_MASK] = sup_mask
            data[DS.IMAGES_POSE] = sup_pose
            data[DS.IMAGES_INTRINSIC] = sup_intrinsic

        # input depht
        if DS.IMAGES_INPUT_DEPTH in self.spec:
            assert self.input_depth_type is not None, 'input depth type is not set'
            depth = self.get_depth_images(sample, sample['input_frame_idx.json'],
                                          self.input_slect_ids,
                                          None, depth_type=self.input_depth_type)
            data[DS.IMAGES_INPUT_DEPTH] = depth

        # supervision depth
        if DS.IMAGES_DEPTH_VOXEL in self.spec:
            depth = self.get_depth_images(sample, sample['sup_frame_idx.json'],
                                          self.sup_slect_ids,
                                          sup_image_indices, depth_type='voxel_depth_100')
            data[DS.IMAGES_DEPTH_VOXEL] = depth

        if DS.MAPS_3D in self.spec:
            data[DS.MAPS_3D] = {}
            grid_to_world = sample['grid_to_world']
            world_to_grid = torch.inverse(grid_to_world)
            for map_type in self.map_types:
                # transform to grid coordinate
                map_points = torch.from_numpy(sample[f"{map_type}.npy"]).float()
                if map_points.shape[0] != 0:
                    data[DS.MAPS_3D][map_type] = CameraBase.transform_points(map_points, world_to_grid)
                else:
                    data[DS.MAPS_3D][map_type] = torch.zeros((0, 3), dtype=torch.float32)

        if DS.BOXES_3D in self.spec:
            data[DS.BOXES_3D] = {}
            grid_to_world = sample['grid_to_world']
            world_to_grid = torch.inverse(grid_to_world)

            # create 3d bounding boxes from all_object_info 
            dynamic_object_dict_this_frame = sample[f"{sample['input_frame_idx.json'][0]:06d}.dynamic_object_info.json"]
            static_object_dict_this_frame = sample[f"{sample['input_frame_idx.json'][0]:06d}.static_object_info.json"]
            all_object_dict = copy.deepcopy(dynamic_object_dict_this_frame)
            all_object_dict.update(static_object_dict_this_frame)

            crop_half_range_canonical = (torch.tensor(self.grid_crop_bbox_max) - torch.tensor(self.grid_crop_bbox_min)) / 2
            bounding_box_in_grid = build_scene_bounding_boxes_from_object_info(
                all_object_dict, apply_object_to_world=True, world_to_target_coord=world_to_grid, aabb_half_range=crop_half_range_canonical
            )

            data[DS.BOXES_3D] = torch.tensor(bounding_box_in_grid).float()
            healthy_box_indices = []
            for i in range(data[DS.BOXES_3D].shape[0]):
                single_box = data[DS.BOXES_3D][i:i+1]
                try:
                    box3d_overlap(single_box, single_box, eps=1e-1)
                except ValueError as e:
                    continue
                healthy_box_indices.append(i)

            data[DS.BOXES_3D] = data[DS.BOXES_3D][healthy_box_indices]
    
        return data

    def crop_pcd_and_generate_grid_raw(self, sample):
        """
        Rather than reading scene-level fvdb grid, here we read from pc and generate the grid on the fly.

        Args:
            cam2world: the front camera c2w (note that opencv convention, RDF) matrix
                in the first frame from input frames

            grid's coordinate is grid_to_world (FLU convention)

        """
        pc_names = [x for x in sample.keys() if 'pcd' in x]
        assert len(pc_names) == 1, 'only need to put the finest pc in the scene'
        assert 'vs01' in pc_names[0], 'I think we only have this type of pc...'
        pc_name = pc_names[0]

        cam2world_opencv = sample[f"{sample['input_frame_idx.json'][0]:06d}.pose.front.npy"].astype(np.float32)
        cam2world_opencv = torch.from_numpy(cam2world_opencv)
        cam2world_flu = opencv_to_flu(cam2world_opencv)
        camera_pos = cam2world_flu[:3, 3]
        camera_front = cam2world_flu[:3, 0] # unit 
        camera_left = cam2world_flu[:3, 1] # unit 
        camera_up = cam2world_flu[:3, 2] # unit 

        new_grid_pos = camera_pos + \
                        camera_front * (self.grid_crop_bbox_min[0] + self.grid_crop_bbox_max[0]) / 2 + \
                        camera_left * (self.grid_crop_bbox_min[1] + self.grid_crop_bbox_max[1]) / 2 + \
                        camera_up * (self.grid_crop_bbox_min[2] + self.grid_crop_bbox_max[2]) / 2

        if self.grid_crop_augment and self.split == 'train':
            new_grid_pos += torch.tensor([random.uniform(-self.grid_crop_augment_range[0], self.grid_crop_augment_range[0]),
                                          random.uniform(-self.grid_crop_augment_range[1], self.grid_crop_augment_range[1]),
                                          random.uniform(-self.grid_crop_augment_range[2], self.grid_crop_augment_range[2])])
        
        grid2world = torch.clone(cam2world_flu)
        grid2world[:3, 3] = new_grid_pos
        world2grid = torch.inverse(grid2world)
        
        crop_half_range_canonical = (torch.tensor(self.grid_crop_bbox_max) - torch.tensor(self.grid_crop_bbox_min)) / 2
        
        # retrieve the point cloud data
        points_static = sample[pc_name]['points']
        semantics_static = sample[pc_name]['semantics']

        pc_to_world = sample[pc_name]['pc_to_world']
        pc2grid = world2grid @ pc_to_world
        points = CameraBase.transform_points(points_static, pc2grid)
        semantics = semantics_static

        extra_car_meshes_vertices = np.zeros((0,3))
        extra_car_meshes_faces = np.zeros((0,3))

        if self.replace_all_car_with_cad:
            dynamic_object_info_this_frame = sample[f"{sample['input_frame_idx.json'][0]:06d}.dynamic_object_info.json"]
            static_object_info_this_frame = sample[f"{sample['input_frame_idx.json'][0]:06d}.static_object_info.json"]

            all_object_dict = copy.deepcopy(dynamic_object_info_this_frame)
            all_object_dict.update(static_object_info_this_frame)

            extra_car_meshes_vertices, extra_car_meshes_faces = build_scene_mesh_from_all_object_info(
                all_object_dict, world2grid, crop_half_range_canonical,
            )

            # we need remove all the car points
            non_car_mask = (semantics != WAYMO_CATEGORY_NAMES.index('CAR')) & \
                           (semantics != WAYMO_CATEGORY_NAMES.index('TRUCK')) & \
                            (semantics != WAYMO_CATEGORY_NAMES.index('BUS')) & \
                            (semantics != WAYMO_CATEGORY_NAMES.index('OTHER_VEHICLE'))

            points = points[non_car_mask]
            semantics = semantics[non_car_mask]

        else:
            """
            else, we use accumulated LiDAR points for each dynamic object. (since static object is already accumulated in the points_static)
            """
            points_dynamic = torch.zeros((0,3))
            semantics_dynamic = torch.zeros((0,))

            dynamic_object_info_this_frame = sample[f"{sample['input_frame_idx.json'][0]:06d}.dynamic_object_info.json"]
            for gid, object_dict in dynamic_object_info_this_frame.items():
                tfm_to_world = torch.tensor(object_dict['object_to_world']).float()

                points_in_canonical = sample['dynamic_object_points_canonical.npz'][f'{gid}_xyz']
                point_semantic = sample['dynamic_object_points_canonical.npz'][f'{gid}_semantic'].item()
                points_in_canonical = torch.from_numpy(points_in_canonical).float()
                points_in_world = CameraBase.transform_points(points_in_canonical, tfm_to_world)

                points_dynamic = torch.cat([points_dynamic, points_in_world], dim=0)
                semantics_dynamic = torch.cat([semantics_dynamic, torch.tensor([point_semantic] * points_in_world.shape[0])], dim=0)

            # ! note that these points are in original waymo world coordinate, while static points are in first vehicle frame
            points_dynamic = CameraBase.transform_points(points_dynamic, world2grid)

            # merge static and dynamic points
            points = torch.cat([points, points_dynamic], dim=0)
            semantics = torch.cat([semantics, semantics_dynamic], dim=0).to(torch.int32)

        # crop the point cloud
        crop_mask = (points[:,0] > -crop_half_range_canonical[0]) & \
                    (points[:,0] < crop_half_range_canonical[0]) & \
                    (points[:,1] > -crop_half_range_canonical[1]) & \
                    (points[:,1] < crop_half_range_canonical[1]) & \
                    (points[:,2] > -crop_half_range_canonical[2]) & \
                    (points[:,2] < crop_half_range_canonical[2])

        cropped_points = points[crop_mask]
        cropped_semantics = semantics[crop_mask]

        # create the grid on the fly
        if self.fvdb_grid_type == 'vs01':
            voxel_sizes_target = torch.tensor([0.1, 0.1, 0.1])
        elif self.fvdb_grid_type == 'vs02':
            voxel_sizes_target = torch.tensor([0.2, 0.2, 0.2])
        elif self.fvdb_grid_type == 'vs04':
            voxel_sizes_target = torch.tensor([0.4, 0.4, 0.4])
        else:
            raise ValueError(f"Unknown fvdb grid type: {self.fvdb_grid_type}")
        origins_target = voxel_sizes_target / 2
        grid_batch_kwargs_target = {'voxel_sizes': voxel_sizes_target, 'origins': origins_target}

        # we may need finest goal voxel size for supervision
        if self.finest_voxel_size_goal == 'vs01':
            voxel_sizes_finest = torch.tensor([0.1, 0.1, 0.1])
        elif self.finest_voxel_size_goal == 'vs02':
            voxel_sizes_finest = torch.tensor([0.2, 0.2, 0.2])
        elif self.finest_voxel_size_goal == 'vs04':
            voxel_sizes_finest = torch.tensor([0.4, 0.4, 0.4])
        else:
            raise ValueError(f"Unknown finest voxel size goal: {self.finest_voxel_size_goal}")

        origins_finest = voxel_sizes_finest / 2 
        grid_batch_kwargs_finest = {'voxel_sizes': voxel_sizes_finest, 'origins': origins_finest}

        sample['grid_to_world'] = grid2world

        assert cropped_points.shape[0] == cropped_semantics.shape[0], 'points and semantics should have the same length'
        assert cropped_points.shape[0] > 0, 'no points in the cropped point cloud'

        grid_name_raw = f'grid_raw.{self.fvdb_grid_type}.pth'
        sample[grid_name_raw] = {'points_finest': cropped_points, 'semantics_finest': cropped_semantics, 
                                 'grid_batch_kwargs_target': grid_batch_kwargs_target,
                                 'grid_batch_kwargs_finest': grid_batch_kwargs_finest}

        # already the grid coordinate
        sample[grid_name_raw]['extra_meshes'] = {
            'vertices': torch.tensor(extra_car_meshes_vertices).float(), 
            'faces': torch.tensor(extra_car_meshes_faces).int()
        }
        
        return grid2world


    def get_images(self, sample, frame_idxs, select_view_ids, sup_image_indices=None):
        """Get images, masks, poses and intrinsics for specified frames and views."""
        all_images_tensor = []
        all_masks_tensor = []
        all_poses_tensor = []
        all_intrinsics_tensor = []

        if sup_image_indices is None:
            sup_image_indices = np.arange(len(frame_idxs) * len(select_view_ids))
        
        for ii, frame_idx in enumerate(frame_idxs):
            for jj, select_id in enumerate(select_view_ids):
                if ii * len(select_view_ids) + jj not in sup_image_indices:
                    continue

                # - camera pose (camera to grid)
                camera_to_world = sample[f"{frame_idx:06d}.pose.{CAMERA_ID_TO_NAME[select_id]}.npy"].astype(np.float32)
                camera_to_world = torch.from_numpy(camera_to_world)
                grid_to_world = sample['grid_to_world']
                cam2grid = torch.inverse(grid_to_world) @ camera_to_world
                all_poses_tensor.append(cam2grid)

                
                # - camera image
                image = sample[f"{frame_idx:06d}.image.{CAMERA_ID_TO_NAME[select_id]}.jpg"]
                image = torch.tensor(image) / 255.0
                intrinsic = sample[f"intrinsic.{CAMERA_ID_TO_NAME[select_id]}.npy"].astype(np.float32)

                camera_model = PinholeCamera.from_numpy(intrinsic, device='cpu')
                camera_model.rescale(
                    ratio_h=image.shape[0] / intrinsic[5], 
                    ratio_w=image.shape[1] / intrinsic[4]
                )
                intrinsic_tensor = torch.tensor(camera_model.intrinsics).float()

                if select_id > 2: # side view, we pad the image [H, W, 3] -> [1280, W, 3]
                    image_padded = torch.zeros((self.front_view_input_wh[1], self.front_view_input_wh[0], 3), dtype=torch.float32)
                    image_padded[:image.shape[0], :, :] = image
                    image = image_padded
                    intrinsic_tensor[5] = self.front_view_input_wh[1] # update height for intrisic
                
                all_images_tensor.append(image)
                all_intrinsics_tensor.append(intrinsic_tensor)

                # - camera mask, all kinds of masks. (1) foreground mask from seg (2) dynamic mask (3) hood & padding mask (4) foreground mask from grid
                # 1) foreground mask from segmentation: 0 for background, 1 for foreground, very precise
                # 2) non dynamic mask: leave it all 1. Do not use dynamic scene in GSM training. Diffusion is OK!
                # 3) non hood & padding mask: hand-craft a padding for side views
                # 4) foreground mask from grid: 0 for background, 1 for foreground, less precise. 

                # We leave [foreground mask from grid] generation in the model part.
                # The difference between [foreground mask from grid] & [foreground mask from seg] 
                # we consider it is mid-ground.
                mask = torch.ones(self.front_view_input_wh[1], self.front_view_input_wh[0], 4, dtype=torch.bool)

                # 1) foreground mask from segmentation
                skymask = torch.tensor(sample[f"{frame_idx:06d}.skymask.{CAMERA_ID_TO_NAME[select_id]}.png"])
                foreground_mask_from_seg = skymask == 0
                mask[:image.shape[0], :, 0] = foreground_mask_from_seg # the first channel stores foreground mask

                # 2) non dynamic mask, leave it all 1 is fine because we will not use dynamic scene in GSM training
                pass

                # 3) hood & padding mask
                if select_id > 2: # side view, padding mask.
                    mask[image.shape[0]:, :, 2] = 0
                    
                all_masks_tensor.append(mask)

        return torch.stack(all_images_tensor), torch.stack(all_masks_tensor), torch.stack(all_poses_tensor), torch.stack(all_intrinsics_tensor)


