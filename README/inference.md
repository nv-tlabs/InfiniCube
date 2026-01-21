# InfiniCube Inference Guide

This guide provides detailed instructions for using the inference scripts in `infinicube/inference/`. The scripts are organized into two categories: single module inference and sequential pipeline inference.

## Table of Contents

- [Single Module Inference](#single-module-inference)
  - [Voxel VAE Inference](#1-voxel-vae-inference)
  - [Voxel Generation (Single Chunk)](#2-voxel-generation-single-chunk)
  - [Feedforward Reconstruction](#3-feedforward-reconstruction)
- [Sequential Pipeline Inference](#sequential-pipeline-inference)
  - [Voxel World Generation](#1-voxel-world-generation)
  - [Guidance Buffer Generation](#2-guidance-buffer-generation)
  - [Scene Gaussian Generation](#3-scene-gaussian-generation)
- [Complete Pipeline Example](#complete-pipeline-example)

---

## Single Module Inference

These scripts allow you to test individual modules independently.

### 1. Voxel VAE Inference

**Script:** `infinicube/inference/voxel_vae.py`

**Purpose:** Evaluate the Voxel VAE model on input voxel data. This script reconstructs voxel grids using the autoencoder.

**Usage:**

```bash
# Resume from local config and checkpoint
python infinicube/inference/voxel_vae.py none \
    --local_config infinicube/voxelgen/configs/vae_64x64x64_height_down2_vs02_dense_residual.yaml \
    --local_checkpoint_path checkpoints/vae_epoch7_step6250.ckpt

# Resume from wandb
python infinicube/inference/voxel_vae.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/vae_64x64x64_height_down2_vs02_dense_residual
```

**Key Arguments:**
- `--local_config`: Path to local config file
- `--local_checkpoint_path`: Path to local checkpoint file
- `--wandb_config`: Wandb experiment name (start with 'wdb:')
- `--split`: Dataset split to evaluate on (test or train, default: test)
- `--output_root`: Output directory (default: `visualization/voxel_vae/`)
- `--val_starting_frame`: Starting frame index (default: 50)

**Output:**
- `{batch_idx}.pt`: Predicted voxel grid and semantics
- `{batch_idx}_gt.pt`: Ground truth voxel grid and semantics
- `{batch_idx}_pred_gt.jpg`: Visualization comparing prediction and ground truth

---

### 2. Voxel Generation (Single Chunk)

**Script:** `infinicube/inference/voxel_generation_single_chunk.py`

**Purpose:** Generate a single chunk of voxel world using diffusion models, conditioned on map information.

**Usage:**

```bash
# Resume from local config and checkpoint
python infinicube/inference/voxel_generation_single_chunk.py none \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt

# Resume from wandb
python infinicube/inference/voxel_generation_single_chunk.py none \
    --use_ema --use_ddim --ddim_step 100 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/diffusion_64x64x64_dense_vs02_map_cond
```

**Key Arguments:**
- `--local_config` / `--local_checkpoint_path` / `--wandb_config`: Model loading options
- `--ddim_step`: Number of DDIM steps (default: 100)
- `--guidance_scale`: Classifier-free guidance scale (default: 1.0)
- `--output_root`: Output directory (default: `visualization/voxel_generation_single_chunk/`)

**Output:**
- `{batch_idx}.pt`: Generated voxel grid and semantics
- `{batch_idx}_gt.pt`: Ground truth voxel grid and semantics
- `{batch_idx}_pred.jpg`: Visualization of generated voxels
- `{batch_idx}_map.jpg`: Visualization of map conditions

---

### 3. Feedforward Reconstruction

**Script:** `infinicube/inference/feedforward_reconstruction.py`

**Purpose:** Feedforward reconstruction using the Gaussian Splatting Model (GSM). This script uses ground-truth voxels to generate 3D Gaussians and render images. It supports both dual-branch (voxel + pixel) and single-branch modes.

If you have trained voxel branch and pixel branch, you first merge them into a single checkpoint by running
```bash
# you need to update the wandb config name inside
python infinicube/voxelgen/utils/model_merge_util.py 
```

**Usage:**

```bash
# Dual branch
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt

# Voxel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 

# Pixel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 

# From wandb, voxel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama \
    --skybox_resolution 1024

# From wandb, pixel branch only
python infinicube/inference/feedforward_reconstruction.py none \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator 
```

**Key Arguments:**
- `--local_config` / `--local_checkpoint_path` / `--wandb_config`: Model loading options
- `--skybox_resolution`: Skybox panorama resolution (default: 512)
- `--save_img_separately`: Save prediction images separately
- `--save_gs`: Save Gaussians to .pkl file
- `--output_root`: Output directory (default: `visualization/feedforward_reconstruction/`)

**Output:**
- `{batch_idx}_pred_images.jpg`: Predicted rendered images
- `{batch_idx}_gt_images.jpg`: Ground truth images
- `{batch_idx}_rgb_gaussians.pkl`: 3D Gaussian splat file (if `--save_gs` is enabled)

---

## Sequential Pipeline Inference of InfiniCube

These scripts form a complete pipeline for generating novel view synthesis from map data. They must be executed sequentially as each step depends on the output of the previous one. Note that not all the waymo clips have good map alignment. If a clip has bad map alignment, the trajectory mode will not work well because the generated voxel world will misalign with original ego trajectory. We filter good clips in `infinicube/assets/waymo_split/good_map_and_motion_alignment.json`.

### 1. Voxel World Generation

**Script:** `infinicube/inference/voxel_world_generation.py`

**Purpose:** Generate a complete voxel world using diffusion models. This is the first step in the pipeline. It supports two modes:
- **Trajectory mode**: Follow the original dataset trajectory
- **Blocks mode**: Generate the entire map by blocks

**Usage:**

#### Trajectory Mode

```bash
# From local checkpoint
python infinicube/inference/voxel_world_generation.py none \
    --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip 13679757109245957439_4167_170_4187_170 \
    --target_pose_num 8

# From wandb
python infinicube/inference/voxel_world_generation.py none \
    --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/diffusion_64x64x64_dense_vs02_map_cond \
    --clip 13679757109245957439_4167_170_4187_170 \
    --target_pose_num 8
```

#### Blocks Mode

```bash
# From local checkpoint
python infinicube/inference/voxel_world_generation.py none \
    --mode blocks \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip 13679757109245957439_4167_170_4187_170 \
    --overlap_ratio 0.25 
```

**Key Arguments:**
- `--mode`: Generation mode (trajectory or blocks)
- `--clip`: Clip name to process
- `--use_ema` / `--use_ddim` / `--ddim_step`: Diffusion sampling options
- `--guidance_scale`: Classifier-free guidance scale (default: 1.0)
- `--webdataset_root`: Path to webdataset root (default: `data/`)
- `--output_root`: Output directory (default: `visualization/infinicube_inference/voxel_world_generation`)

**Trajectory Mode Specific:**
- `--target_pose_num`: Target number of poses (default: 5)
- `--pose_distance_ratio`: Ratio of grid size for pose distance (default: 0.75)

**Blocks Mode Specific:**
- `--overlap_ratio`: Overlap ratio between blocks (default: 0.25)

**Output:**
- `{step}.pt`: Voxel grid and semantics at each step
- `{step}.jpg`: Visualization of voxel world at each step
- `{step}_map_cond.jpg`: Visualization of map conditions
- Final output is in the first camera's FLU coordinate system

---

### 2. Guidance Buffer Generation

**Script:** `infinicube/inference/guidance_buffer_generation.py`

**Purpose:** Generate guidance buffers and video from the voxel world. This is the second step in the pipeline and depends on the output from voxel world generation.

**Usage:**

#### Trajectory Mode

```bash
# Sample every 1 frame from the original trajectory
python infinicube/inference/guidance_buffer_generation.py \
    --mode trajectory \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --make_dynamic --offset_unit frame --offset 1

# Sample every 2 meters from the original trajectory
python infinicube/inference/guidance_buffer_generation.py \
    --mode trajectory \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --make_dynamic --offset 2
```

#### Blocks Mode (with interactive GUI)

```bash
python infinicube/inference/guidance_buffer_generation.py \
    --mode blocks \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/blocks
```

**Note:** In blocks mode, the script launches an interactive Viser GUI where you can:
1. Record keyframes by clicking the "Record Keyframe" button while navigating the scene
2. Save the current pass with "Save Current Pass"
3. Optinally Reset to the first frame with "Force Reset to first frame position"
4. Repeat 1-2 to record multiple passes
5. Close the GUI with "Kill" when done

**Key Arguments:**
- `--mode`: Generation mode (trajectory or blocks)
- `--clip`: Clip name to process
- `--extrap_voxel_root`: Path to voxel world directory (from step 1)
- `--data_root`: Path to data folder (default: `data`)
- `--output_root`: Output directory (default: `visualization/infinicube_inference/guidance_buffer_generation`)
- `--video_prompt`: Text prompt for video generation (default: `The video is about a driving scene captured at daytime. The weather is clear.`)
- `--disable_video_generation`: Disable video generation
- `--video_checkpoint_path`: Path to video generation checkpoint (default: `checkpoints/video_generation.safetensors`)
- `--use_wan_1pt3b`: Use Wan2.1-T2V-1.3B model for video generation

**Trajectory Mode Specific:**
- `--make_dynamic`: Include dynamic objects in the scene
- `--offset_unit`: Offset unit (meter or frame, default: meter)
- `--offset`: Sampling offset value

**Blocks Mode Specific:**
- `--interpolate_frame_num`: Number of interpolated frames (default: 150)
- `--existing_trajectory_npy`: Path to pre-recorded trajectory (skip GUI)

**Output:**
- `voxel_depth_100_{resolution}_front.tar`: Depth buffer (scaled by 100, uint16)
- `instance_buffer_{resolution}_front.tar`: Instance segmentation buffer
- `pose.tar`: Camera poses for each frame
- `intrinsic.tar`: Camera intrinsics
- `dynamic_object_info.tar`: Dynamic object information
- `semantic_buffer_video_{resolution}_front.mp4`: Semantic buffer visualization
- `depth_vis_video_{resolution}_front.mp4`: Depth visualization
- `coordinate_buffer_video_{resolution}_front.mp4`: Coordinate buffer visualization
- `voxel.pt`: Copy of the input voxel grid
- `video_{resolution}_front.mp4`: Generated video from text and buffers

---

### 3. Scene Gaussian Generation

**Script:** `infinicube/inference/scene_gaussian_generation.py`

**Purpose:** Generate 3D Gaussian representations of the scene with both static background and dynamic objects. This is the final step in the pipeline and depends on the output from guidance buffer generation.

**Usage:**

```bash
# From local config and checkpoint
python infinicube/inference/scene_gaussian_generation.py none \
    --data_folder visualization/infinicube_inference/guidance_buffer_generation/trajectory_pose_sample_1frame/13679757109245957439_4167_170_4187_170 \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt 

# From wandb
python infinicube/inference/scene_gaussian_generation.py none \
    --data_folder visualization/infinicube_inference/guidance_buffer_generation/blocks/13679757109245957439_4167_170_4187_170 \
    --wandb_config wdb:nvidia-toronto/infinicube-release/waymo_wds/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama
```

**Key Arguments:**
- `--data_folder`: Path to guidance buffer directory (from step 2)
- `--local_config` / `--local_checkpoint_path` / `--wandb_config`: Model loading options
- `--output_folder`: Output directory (default: `visualization/infinicube_inference/gaussian_scene_generation/`)
- `--start_frame_index`: Starting frame index (default: 0)
- `--use_frame_interval`: Frame interval for reconstruction (default: 6)
- `--active_frame_proportion`: Proportion of frames to use (default: 1.0)
- `--enable_pixel_branch_last_n_frame`: Enable pixel branch for last N frames (default: 1)
- `--accumulate_multi_frame_for_dynamic`: Accumulate multi-frame data for dynamic objects

**Output:**
- `decoded_gs_static.pkl`: Static background 3D Gaussians
- `decoded_gs_object.pkl`: Dynamic object 3D Gaussians (if dynamic objects present)
- `dynamic_object_info.tar`: Dynamic object transformation information
- `visualize_gsm/`: Folder containing visualization images
  - `static_pd_images.jpg`: Static scene predictions
  - `static_pd_images_fg.jpg`: Static foreground predictions
  - `first_frame_pd_images.jpg`: First frame with dynamic objects
  - Mask visualizations for different scene regions

**Multi-Pass Support:**
The script automatically detects and handles multiple passes (e.g., `pass_0/`, `pass_1/`, ...) from blocks mode, merging them into a single reconstruction.

---

## Complete Pipeline Example

Here's a complete example of running the entire pipeline:

```bash
# Step 1: Generate voxel world (trajectory mode)
python infinicube/inference/voxel_world_generation.py none \
    --mode trajectory \
    --use_ema --use_ddim --ddim_step 100 \
    --local_config infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml \
    --local_checkpoint_path checkpoints/voxel_diffusion.ckpt \
    --clip 13679757109245957439_4167_170_4187_170 \
    --target_pose_num 12 \
    --guidance_scale 2.0

# Step 2: Generate guidance buffers
python infinicube/inference/guidance_buffer_generation.py \
    --mode trajectory \
    --clip 13679757109245957439_4167_170_4187_170 \
    --extrap_voxel_root visualization/infinicube_inference/voxel_world_generation/trajectory \
    --make_dynamic --offset_unit frame --offset 1

# Step 3: Generate 3D Gaussians
python infinicube/inference/scene_gaussian_generation.py none \
    --data_folder visualization/infinicube_inference/guidance_buffer_generation/trajectory_pose_sample_1frame/13679757109245957439_4167_170_4187_170 \
    --local_config infinicube/voxelgen/configs/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.yaml \
    --local_checkpoint_path checkpoints/gsm_vs02_res512_view1_dual_branch_sky_mlp_modulator.ckpt
```

The final output will be a complete 3D Gaussian scene representation that can be rendered from novel viewpoints.

---

## Notes

1. **Coordinate Systems**: 
   - The voxel world is generated in the first camera's FLU (Front-Left-Up) coordinate system
   - All subsequent steps maintain this coordinate system for consistency

2. **Model Checkpoints**: 
   - You can use either local checkpoints or load from Weights & Biases (wandb)
   - For wandb, use the format: `wdb:organization/project/experiment_name`


3. **Dynamic Objects**:
   - Dynamic objects are optional and can be controlled with the `--make_dynamic` flag
   - We usually disable dynamic objects in block mode.

5. **Blocks Mode GUI**:
   - The Viser GUI in blocks mode allows interactive trajectory generation
   - You can record multiple passes for comprehensive scene coverage
   - Saved trajectories can be reused with `--existing_trajectory_npy`

6. **Output Formats**:
   - `.pt` files contain PyTorch tensors (voxel grids, semantics)
   - `.pkl` files contain 3D Gaussian parameters
   - `.tar` files contain webdataset archives for buffers
   - `.jpg`/`.mp4` files are for visualization

For more details about the data format and model architectures, please refer to the main README and the paper.
