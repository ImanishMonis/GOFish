# GOFish: usage on this machine

This guide covers the `gofish` conda environment on this workstation and how to run the pipeline on
your own data. For the method itself and the paper benchmarks, see [README.md](README.md).

## 1. Activate the environment

```bash
conda activate gofish
cd /ipb70/home_domuser2/s49imoni/GOFish
```

Activating the environment also sets the following. The hooks live in
`$CONDA_PREFIX/etc/conda/activate.d/gofish.sh` and are undone on `conda deactivate`.

| Variable | Value | Why |
|---|---|---|
| `PATH` | gets `$CONDA_PREFIX/opt/slang/bin` (slangc 2026.5.2) | the renderer compiles Slang kernels |
| `CUDA_HOME` | `$CONDA_PREFIX` (CUDA 12.8.1 toolkit inside the env) | there is no system `nvcc` |
| `CPATH` | the env's CUDA and CGAL headers | C++ extensions find their headers |
| `TORCH_CUDA_ARCH_LIST` | `7.5` | builds kernels for the RTX 2080 SUPER only, which is faster. If you move to another GPU, change this or `unset` it. |

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.8.0+cu128 True
slangc -v                                                                       # 2026.5.2
```

### What is installed

- Python 3.11, PyTorch 2.8.0 + torchvision 0.23.0 (cu128), and every Python package pinned in
  `requirements-paper.txt`. GOFish itself is installed editable (`pip install -e ".[eval]"`), so edits
  to the code take effect without reinstalling.
- `fused-ssim` at the pinned commit (`requirements_extra.txt`).
- From conda: the CUDA 12.8.1 toolkit, cmake, CGAL, GMP, MPFR and Eigen.
- GOF's `tetra-triangulation` extension (needed by `mesh.py`), built from
  `/ipb70/home_domuser2/s49imoni/gaussian-opacity-fields` at commit `5245b20`. The env imports it from
  there, so **do not delete that folder**.

GOFish's own CUDA/Slang renderer is JIT-compiled on first use and cached under `~/.cache/torch_extensions`.
The first run of a new config therefore pauses for a few minutes with "Compiling native code..".

## 2. What your data must look like

GOFish reads one folder per scene, in COLMAP format:

```
my_scene/
├── sparse/
│   └── 0/
│       ├── cameras.bin        # or cameras.txt
│       ├── images.bin         # or images.txt
│       └── points3D.bin       # or points3D.txt; required, it initializes the Gaussians
├── images_4/                  # images_<k>/: your images downsampled by k (k = dataset.downsample_factor)
│   ├── frame_00001.jpg        # names must match the image names in images.bin (subfolders are fine)
│   ├── frame_00001_mask.png   # optional per-image mask: same name + "_mask.png", white (>127) = use pixel
│   └── ...
└── test_list.txt              # optional: one COLMAP image name per line = held-out test views
```

Rules the loader enforces:

- **Image folder.** With `downsample_factor=k` the loader reads `images_<k>/`. With `k=1` it reads
  `images/`. You downsample the images yourself (any tool). The COLMAP intrinsics are rescaled to match
  the image size automatically.
- **Camera models.** Use `OPENCV_FISHEYE` (fisheye lenses) or `PINHOLE`. The loader also accepts
  `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` and `FULL_OPENCV`. However, the renderer's
  complete-admission table only covers `OPENCV_FISHEYE` and distortion-free `PINHOLE`, so undistort other
  models first:
  `colmap image_undistorter --image_path images --input_path sparse/0 --output_path undistorted`.
- **Train/test split.** By default every 8th frame is held out (`dataset.test_split_interval=8`).
  Give `dataset.test_list=<file>` to choose the test views instead.
- **EXIF exposure** is read from the images if present (it feeds the PPISP exposure prior). It is
  optional.

### New camera: register it once

The renderer has a compiled table with one row per camera calibration and resolution. On a camera it
has not seen, `train.py` stops before training and prints the exact command to add the row. That command
looks like this:

```bash
python -m gofish.tools.build_calibration_table --colmap my_scene/sparse/0/cameras.bin --downsample 4
```

Run it once per new camera or resolution, then start training again. The renderer recompiles
automatically. This **edits a tracked header** in `threedgut_tracer/include/3dgut/kernels/`, so commit and push it to
your fork if you want to keep that camera.

## 3. Run the pipeline on one scene

```bash
conda activate gofish
cd /ipb70/home_domuser2/s49imoni/GOFish
SCENE=/path/to/my_scene
NAME=my_scene
RUN=runs/$NAME
```

**a) Train.** Choose the preset that matches your camera.

```bash
# Fisheye (OPENCV_FISHEYE), images in images_4/; add geometry=sof if you want a mesh
python train.py --config-name apps/fiord path=$SCENE out_dir=runs experiment_name=$NAME \
    dataset.camera_domains=[] dataset.downsample_factor=4 geometry=sof

# Pinhole (PINHOLE), images in images_2/ (turns off Wide-Angle Appearance Modeling)
python train.py --config-name apps/tnt path=$SCENE out_dir=runs experiment_name=$NAME \
    dataset.downsample_factor=2 geometry=sof
```

- `dataset.camera_domains=[]` is required for your own fisheye data. The FIORD preset otherwise loads
  FIORD's measured lens apertures.
- `geometry=sof` enables the Geometry Objective, which gives much better meshes. Leave it out if you
  only want rendering.
- Training runs 35,000 steps: 30k scene steps plus 5k appearance-controller steps. TensorBoard logs go
  to `$RUN`; view them with `tensorboard --logdir runs`.
- Any config value can be overridden on the command line the same way. For example, `n_iterations=7000`
  gives a quick test run. `python train.py --config-name apps/fiord --help` lists the config groups.

**b) Render the held-out views.**

```bash
python render.py --checkpoint $RUN/ckpt_last.pt --out $RUN/heldout                       # real-time renderer
python render.py --checkpoint $RUN/ckpt_last.pt --out $RUN/heldout --renderer exact --metrics  # paper renderer + PSNR/SSIM/LPIPS
```

**c) Measure FPS** (optional).

```bash
python -m threedgrut.benchmark --checkpoint $RUN/ckpt_last.pt --out $RUN/fps.json
```

**d) Extract the mesh.** Use the step-30000 geometry checkpoint, not `ckpt_last.pt`.

```bash
python mesh.py --checkpoint $RUN/ours_30000/ckpt_30000.pt --out $RUN/mesh
```

### Outputs

```
runs/my_scene/
├── ckpt_last.pt              # rendering checkpoint (step 35000)
├── ours_<step>/ckpt_<step>.pt  # intermediate checkpoints; ours_30000 is the geometry checkpoint
├── heldout/                  # rendered test views (pred/), masked ground truth, metrics.json
├── fps.json
└── mesh/
    ├── mesh.ply              # final mesh
    ├── mesh_sof.ply, mesh_raw.ply
    └── mesh_info.json
```

If the dataset moves after training, pass its new location with `render.py --data <scene>` or
`mesh.py --data <scene>`.

## 4. Paper datasets (FIORD, FullCircle, ScanNet++, Tanks & Temples)

The `benchmark/*.sh` scripts prepare, train, render, mesh and evaluate every scene of a dataset. See
the [README](README.md#data-preparation) for the expected download layouts. For example, to prepare a
single FIORD scene:

```bash
python -m gofish.data.fiord scene /data/FIORD/Kitchen_In runs/_data/Kitchen_In
python train.py --config-name apps/fiord path=runs/_data/Kitchen_In out_dir=runs experiment_name=Kitchen_In
```

To run a whole benchmark, use `bash benchmark/fiord.sh --data /data/FIORD [--lidar ...]`.
`bash benchmark/fiord.sh --help` lists the options.

## 5. Notes for this GPU (RTX 2080 SUPER, 8 GB)

- The paper used a 24 GB A10G. On 8 GB, large scenes may run out of memory once densification grows
  the Gaussian count. If that happens, try these in order:
  1. Downsample the images more, e.g. `images_8/` with `dataset.downsample_factor=8`.
  2. Use fewer or smaller training images.
  3. Densify less by raising the gradient thresholds, e.g.
     `strategy.densify.clone_grad_threshold=0.0004 strategy.densify.split_grad_threshold=0.0004`
     (default 0.0002, see `configs/strategy/gs.yaml`). This trades some quality for memory.
- The first training run compiles the renderer, which takes several minutes. Later runs reuse the cache.

## 6. Git remotes

```
origin    https://github.com/ImanishMonis/GOFish.git   # your fork: push here
upstream  https://github.com/tuallen/GOFish.git        # original: git fetch upstream && git merge upstream/main
```

## 7. Rebuilding the environment from scratch

```bash
conda create -y -n gofish -c nvidia/label/cuda-12.8.1 -c conda-forge --override-channels \
    python=3.11 cuda-toolkit=12.8.1 cmake cgal-cpp gmp mpfr eigen pip setuptools wheel
conda activate gofish
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[eval]" -c requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu128
# Slang compiler into the env (then recreate the activate.d hook from section 1)
mkdir -p $CONDA_PREFIX/opt/slang && curl -L https://github.com/shader-slang/slang/releases/download/v2026.5.2/slang-2026.5.2-linux-x86_64.tar.gz | tar -xz -C $CONDA_PREFIX/opt/slang
export PATH=$CONDA_PREFIX/opt/slang/bin:$PATH CUDA_HOME=$CONDA_PREFIX
python -m pip install --no-build-isolation -r requirements_extra.txt
# GOF tetra-triangulation (mesh extraction)
git clone https://github.com/autonomousvision/gaussian-opacity-fields ../gaussian-opacity-fields
git -C ../gaussian-opacity-fields checkout 5245b20e5d11acd6d1ff5af4b890dc2bedd99693
cd ../gaussian-opacity-fields/submodules/tetra-triangulation
sed -i 's/_GLIBCXX_USE_CXX11_ABI=0/_GLIBCXX_USE_CXX11_ABI=1/' cmake/FindTorch.cmake
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');$CONDA_PREFIX" . && make -j8
python -m pip install -e . && cd -
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is needed because conda's cmake 4.x refuses GOF's old bundled
pybind11 otherwise. The original README does not mention this.
