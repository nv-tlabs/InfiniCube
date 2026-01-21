# Training Guide

This document describes how to train different components of InfiniCube using either single-node or multi-node configurations.

## WandB Setup 
Our training script highly relies on [WandB](https://wandb.ai/). Please register an account for [WandB](https://wandb.ai/) first and get your `API_key`. Then you can setup for your machine by running this command in the terminal:
```bash
# requires your API key
wandb login 

# or store your API key in the environment variable `WANDB_API_KEY` in bashrc or zshrc
echo "export WANDB_API_KEY=<your_api_key>" >> ~/.bashrc
```

Then you edit `infinicube/voxelgen/train.py`, find the following line
```python
check_wandb_name = "wdb:nvidia-toronto/infinicube-%s/%s:last" % (project_name, run_name)
```
and replace `nvidia-toronto` with your own WandB account organization name.

---

## Single Node Training

For training on a single node (8 GPUs), use the following commands directly. 
The default logger type is wandb, but if you have multiple jobs on the wandb with same name, they will conflict. You can add `--logger_type none` to the command to disable logging when debugging, and use the default `--logger_type wandb` in final training.

### VAE

```bash
python infinicube/voxelgen/train.py infinicube/voxelgen/configs/vae_64x64x64_height_down2_vs02_dense_residual.yaml --gpus 8 --eval_interval 1 --wname vae_64x64x64_height_down2_vs02_dense_residual
```

### Voxel Diffusion

```bash
python infinicube/voxelgen/train.py infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml --gpus 8 --eval_interval 1 --wname diffusion_64x64x64_dense_vs02_map_cond
```

### Feed-forward Reconstruction

#### 3D Branch

```bash
python infinicube/voxelgen/train.py infinicube/voxelgen/configs/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama.yaml --gpus 8 --eval_interval 1 --wname gsm_vs02_res512_view1_voxel_branch_only_sky_panorama
```

#### 2D Branch

```bash
python infinicube/voxelgen/train.py infinicube/voxelgen/configs/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator.yaml --gpus 8 --eval_interval 1 --wname gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator
```

After training these two branches, you need to merge them into a single checkpoint by running
```bash
python infinicube/voxelgen/utils/model_merge_util.py
```
editing the wandb experiment names to yours inside the script.

---

## Multi-Node Training

For distributed training across multiple nodes, use the SLURM wrapper script. The script automatically handles job submission and supports auto-resume for jobs with 4-hour time limits.

### VAE

```bash
# 2 nodes (16 GPUs total)
sh slurm/train-slurm.sh -n 2 -s 6006 -i 100 -c 'python infinicube/voxelgen/train.py infinicube/voxelgen/configs/vae_64x64x64_height_down2_vs02_dense_residual.yaml --gpus 8 --eval_interval 1 --wname vae_64x64x64_height_down2_vs02_dense_residual'
```

### Voxel Diffusion

```bash
# 2 nodes (16 GPUs total)
sh slurm/train-slurm.sh -n 4 -s 6006 -i 100 -c 'python infinicube/voxelgen/train.py infinicube/voxelgen/configs/diffusion_64x64x64_dense_vs02_map_cond.yaml --gpus 8 --eval_interval 1 --wname diffusion_64x64x64_dense_vs02_map_cond'
```

### Feed-forward Reconstruction

#### 3D Branch

```bash
# 2 nodes (16 GPUs total)
sh slurm/train-slurm.sh -n 2 -s 6006 -i 100 -c 'python infinicube/voxelgen/train.py infinicube/voxelgen/configs/gsm_vs02_res512_view1_voxel_branch_only_sky_panorama.yaml --gpus 8 --eval_interval 1 --wname gsm_vs02_res512_view1_voxel_branch_only_sky_panorama'
```

#### 2D Branch

```bash
# 2 nodes (16 GPUs total)
sh slurm/train-slurm.sh -n 2 -s 6006 -i 100 -c 'python infinicube/voxelgen/train.py infinicube/voxelgen/configs/gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator.yaml --gpus 8 --eval_interval 1 --wname gsm_vs02_res512_view1_pixel_branch_only_sky_mlp_modulator'
```

After training these two branches, you need to merge them into a single checkpoint by running
```bash
python infinicube/voxelgen/utils/model_merge_util.py
```
editing the wandb experiment names to yours inside the script.

---

## SLURM Script Parameters

When using `slurm/train-slurm.sh`:

- `-n`: Number of nodes (e.g., `-n 2` for 2 nodes)
- `-s`: Random seed (e.g., `-s 6006`)
- `-i`: Number of iterations (each ~4 hours, e.g., `-i 100` for auto-resume)
- `-c`: Training command to execute

The script automatically handles:
- Job submission and resubmission (for 4-hour time limit)
- Auto-resume from checkpoints
- Log management


## Video Generation Training
We reimplement video generation stage with Wan2.1 1.3B (text-to-video) in [diffsynth](https://github.com/yifanlu0227/DiffSynth-Studio-InfiniCube). If you want to train the video generation model, you need extra data preparation.

### Captioning
You can use the script in `infinicube/data_process/generate_caption.py` to generate captions for the videos. To caption the videos in `data/video_480p_front`, run:
```bash
# single gpu
python infinicube/data_process/generate_caption.py

# multi gpu
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py

# multi node
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py -s 0,2
torchrun --nproc_per_node=8 infinicube/data_process/generate_caption.py -s 1,2
```

### Create Metadata CSV
[Diffsynth](https://github.com/yifanlu0227/DiffSynth-Studio-InfiniCube) requires metadata.csv for training. You can use the script in `infinicube/data_process/generate_metadata_csv.py` to generate metadata.csv for the videos. To generate metadata.csv for the videos in `data/video_480p_front`, run:
```bash
python infinicube/data_process/generate_metadata_csv.py --data_root data/ --input_attribute video_480p_front --output_attribute metadata.csv
```

### Training
You can use the script in `diffsynth/train_wan_with_buffer.sh` to train the video generation model. To train the video generation model, run:
```bash
sh train_wan_with_buffer.sh
```