# Environment Setup

InfiniCube uses a dual-environment strategy to resolve dependency conflicts between `waymo-open-dataset-tf-2-11-0` (requires TensorFlow) and the main project dependencies (PyTorch ecosystem).

## Architecture

- **Main Environment (conda `infini` environment)**: PyTorch, PyTorch3D, gsplat, and other core dependencies
- **Waymo Environment (`.venv-waymo`)**: Only waymo-open-dataset and its dependencies (TensorFlow, etc.)

---

## Setup

### Prerequisites
```bash
conda env create -f environment.yaml
```

### Create Main Environment
```bash
conda activate infini # make sure in the conda environment
uv pip install -e .[main]

# Install pytorch3d from source. takes ~2min
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# Install mmcv using mim, then ensure numpy<2.0.0 is respected
mim install "mmcv>=2.0.0"
uv pip install "mmsegmentation>=1.0.0" "numpy<2.0.0"
```


### Create Waymo Environment
This is for data processing. Since waymo-open-dataset has strict version requirements, we create a separate environment to make things easier.
```bash
conda activate infini # make sure in the conda environment
uv venv .venv-waymo --python $(which python) # use the same python interpreter as the main environment
uv pip install -e .[waymo] --python .venv-waymo/bin/python # install to the virtual environment
```

---

## Usage

### Activate Main Environment
```bash
conda activate infini
```

### Use Waymo Environment
You will only need to activate the waymo environment when you are processing waymo data.
```bash
conda activate infini && source .venv-waymo/bin/activate
```
Deactivate the waymo environment (the virtual environment created by uv) when you are done.
```bash
deactivate
```

### Install Other Packages
You should use `uv pip install` to install other packages. 
It is **not recommended** to use `conda install` for python packages now because we are using uv to parse dependencies!
