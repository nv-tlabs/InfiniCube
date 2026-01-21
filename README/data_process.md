# Waymo Dataset Conversion Pipeline

This document describes the pipeline for converting Waymo Open Dataset tfrecord files to webdataset tar files.

## Data Download
### Waymo Dataset Downloading
Download the all the training & validation clips from [waymo perception dataset v1.4.2](https://waymo.com/open/download/) to the `/path/to/waymo/tfrecords`. 

If you have `sudo`, you can use [gcloud](https://cloud.google.com/storage/docs/discover-object-storage-gcloud) to download them from terminal.
<details>
<summary><span style="font-weight: bold;">gcloud installation (need sudo) and downloading from terminal</span></summary>

```bash
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Then you can login your google account and download the above tfrecords via
```bash
# or use `gcloud init --no-launch-browser` if you are in a remote terminal session
gcloud init 
bash infinicube/data_process/download_waymo.sh infinicube/data_process/waymo_all.json /path/to/waymo/tfrecords
```
</details>

After downloading tfrecord files, we expect a folder structure as follows:
```bash
/path/to/waymo/tfrecords
    |-- segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord
    |-- segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord
    |-- ...
    `-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels.tfrecord
```

## Data Processing

The conversion process is split into two stages:

1. **Primary Conversion** (`waymo2webdataset.py`): Directly extracts essential data from tfrecord files and saves to tar files
2. **Additional Attributes** (`generate_additional_attributes.py`): Generates computed attributes (sky masks, depth buffers, etc.) from existing tar files

## Stage 1: Primary Conversion (waymo2webdataset.py)

This script directly converts Waymo tfrecord files to webdataset tar files without intermediate storage.

### Output generated:

**Tar files:**
- `image_{camera_name.lower()}` - Camera images (JPEG)
- `image_480p_{camera_name.lower()}` - Camera images (JPEG, 480p: 480x832)
- `video_{camera_name.lower()}` - Video files (MP4, original resolution)
- `video_480p_{camera_name.lower()}` - Video files (MP4, 480p: 480x832)
- `pose` - Camera poses in OpenCV convention
- `intrinsic` - Camera intrinsic parameters
- `extrinsic` - Extrinsic parameters (camera-to-vehicle, lidar-to-vehicle transforms)
- `static_object_info` - Static object information
- `dynamic_object_info` - Dynamic object information (determined by inter-frame displacement)
- `dynamic_object_points_canonical` - LiDAR points in canonical object coordinates
- `3d_lane` - 3D lane polylines
- `3d_road_edge` - 3D road edge polylines
- `3d_road_line` - 3D road line polylines

**Non-tar files (high compression):**
- `lidars/{segment_name}/lidar_TOP/*.npz` - LiDAR point clouds in compressed NPZ format (numpy savez_compressed, float16)

### Usage:

```bash
# activate the waymo processing environment
source .venv-waymo/bin/activate

python infinicube/data_process/waymo2webdataset.py \
    --input_dir /path/to/waymo/tfrecords \
    --output_dir data/

# slurm job submission. We provide a template for you, please modify the script to your own needs.
sbatch slurm/data-process.sh
```

### Arguments:
- `--input_dir, -i`: Root directory of waymo dataset (tfrecord files)
- `--output_dir, -o`: Output directory of webdataset tar files
- `--node_split, -n`: Node split format "node_index,total_nodes" (default: '0,1')
- `--specify_segments`: Specific segment names to process (optional, default: process all segments)
- `--num_workers`: Number of parallel workers (default: 16, recommended: set to your CPU core count)
- `--skip_lidar`: Skip lidar point cloud extraction
- `--skip_video`: Skip video extraction
- `--overwrite`: not skip existing files, overwrite them

### Notes:
- Default camera: FRONT only (can be extended to FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT)
- Dynamic objects are identified by inter-frame displacement threshold (MIN_MOVING_DISTANCE_AT_10FPS = 0.05m)
- All camera poses are converted from FLU to OpenCV convention
- Video files are generated at both original resolution and 480p (480x832)

### Download Ground-Truth Voxels
Ground-truth voxels can be found on [Hugging Face](https://huggingface.co/datasets/xrenaa/SCube-Data/tree/main/pc_voxelsize_01). Download and put them in `data/pc_voxelsize_01` along with other attributes.

## Stage 2: Generate Additional Attributes (generate_additional_attributes.py)

This script reads existing tar files and generates additional computed attributes using GPU acceleration (PyTorch distributed).

### Output tar files generated:
- `skymask_{camera_name.lower()}` - Sky segmentation masks (PNG, generated using mmsegmentation)
- `3d_lane_voxelsize_025` - Discretized lane polylines (0.25m voxel size)
- `3d_road_edge_voxelsize_025` - Discretized road edge polylines (0.25m voxel size)
- `3d_road_line_voxelsize_025` - Discretized road line polylines (0.25m voxel size)
- `3d_road_surface_voxelsize_04` - Estimated road surface points (0.4m voxel size)
- `pc_with_map_without_car_voxelsize_01` - Static point cloud with augmented maps, dynamic objects removed (0.1m voxel size)
- `voxel_depth_buffer_100_{resolution}_front` - Depth buffer (PNG16, INT16, scaled by 100, 0.2m voxel size)
- `semantic_buffer_{resolution}_front` - Semantic buffer (PNG, UINT8, 0.2m voxel size)
- `semantic_buffer_video_{resolution}_front` - Semantic buffer video (MP4, 10fps)
- `instance_buffer_{resolution}_front` - Instance buffer (PNG, UINT16, 0.2m voxel size)

### Usage:

**Generate sky masks:**
```bash
# single node with 8 gpus
# --generate_skymask, --generate_discrete_map, --generate_map_augmentated_car_removed_voxel, --generate_buffer can be run separately. But they are all required.
torchrun --nproc_per_node=8 infinicube/data_process/generate_additional_attributes.py \
    --input_root data/ \
    --generate_skymask \
    --generate_discrete_map \
    --generate_map_augmentated_car_removed_voxel \
    --generate_buffer
```

**Multi-node execution** (example with 2 nodes):
```bash
# Node 0
torchrun --nproc_per_node=8 infinicube/data_process/generate_additional_attributes.py \
    --input_root data/ \
    --generate_skymask \
    --node_split 0,2

# Node 1
torchrun --nproc_per_node=8 infinicube/data_process/generate_additional_attributes.py \
    --input_root data/ \
    --generate_skymask \
    --node_split 1,2
```

**Slurm job submission** 
```bash
# slurm job submission. We provide a template for you, please modify the script to your own needs.
sbatch slurm/additional-data-process.sh
```

### Arguments:
- `--input_root`: Root folder of the webdataset (default: 'data/')
- `--node_split`: Node split format "node_index,total_nodes" (default: '0,1')
- `--generate_skymask`: Flag to generate sky masks
- `--generate_discrete_map`: Flag to generate discrete map points and road surface
- `--generate_map_augmentated_car_removed_voxel`: Flag to generate map-augmented, car-removed voxel
- `--generate_buffer`: Flag to generate depth/semantic/instance buffers

### Notes:
- All generation tasks support multi-GPU and multi-node distributed execution
- Default resolution for buffers and depth maps: 480p (480x832)
- Buffers are generated using FVDB grid with voxel size [0.2, 0.2, 0.2]

