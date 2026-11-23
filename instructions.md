# Running the GOFish pipeline on your own data

How to go from a COLMAP scene to a trained GOFish model and a mesh on this machine
(RTX 2080 SUPER 8 GB, 62 GB RAM, conda env `gofish`). The commands use the
`may_session_v3_edit` scene as the example; swap the paths for another scene.

Pipeline: **prepare data → (register camera, once) → train → extract mesh → render held-out views**.

---

## 0. Environment

```bash
conda activate gofish        # NOT 3dgrut: that env has a different torch
cd /ipb70/home_domuser2/s49imoni/GOFish
```

Activating `gofish` puts the Slang compiler and the env's CUDA 12.8 toolkit on PATH
(see `usage.md` §1). Quick check: `slangc -v` should print `2026.5.2`.

---

## 1. Data folder structure

### 1a. What your raw scene looks like (input to the prep step)

```
may_session_v3_edit/                      # e.g. /cache/s49imoni/SLAM_repo/data_V3/may_session_v3_edit
├── sparse/
│   └── 0/
│       ├── cameras.bin                   # OPENCV_FISHEYE (or PINHOLE), one camera per lens
│       ├── images.bin                    # image names are paths like  images/cam0/<frame>.jpg
│       ├── points3D.bin                  # REQUIRED: initializes the Gaussians
│       └── frames.bin, rigs.bin          # optional (COLMAP rig), copied along, not used
├── images/
│   ├── cam0/<frame>.jpg
│   └── cam1/<frame>.jpg
└── masks/                                # optional, one per image, same sub-path, .png
    ├── cam0/<frame>.png                  # white (>127) = train on this pixel, black = ignore
    └── cam1/<frame>.png
```

### 1b. What GOFish actually reads (output of the prep step)

```
may_session_v3_edit_gofish/               # e.g. /cache/s49imoni/data/may_session_v3_edit_gofish
├── sparse/0/...                          # copied unchanged
└── images_2/                             # images_<k>/  with k = dataset.downsample_factor
    └── images/                           # = the image names from images.bin, below images_<k>/
        ├── cam0/<frame>.jpg              # downsampled by k
        ├── cam0/<frame>_mask.png         # mask NEXT TO its image, same stem + "_mask.png"
        ├── cam1/<frame>.jpg
        └── cam1/<frame>_mask.png
```

Rules the loader enforces:

- **Image path** = `<scene>/images_<k>/<name in images.bin>` (with `k=1`: `<scene>/images/<name>`).
- **Mask path** = the image path with `.jpg` replaced by `_mask.png`. A mask is optional per image;
  white (>127) means the pixel is used. Masks may change from frame to frame (yours do).
- **Camera models**: `OPENCV_FISHEYE` or distortion-free `PINHOLE`. Undistort anything else first:
  `colmap image_undistorter --image_path images --input_path sparse/0 --output_path undistorted`.
- **Intrinsics** are rescaled to the downsampled image size automatically. Don't edit `cameras.bin`.
- **Train/test split**: every 8th registered image is held out by default
  (`dataset.test_split_interval=8`). To pick the test views yourself, pass
  `dataset.test_list=<file>` with one COLMAP image name per line.

### 1c. Convert raw → GOFish layout

```bash
python -m gofish.tools.prepare_scene \
    --source /cache/s49imoni/SLAM_repo/data_V3/may_session_v3_edit \
    --target /cache/s49imoni/data/may_session_v3_edit_gofish \
    --factor 2
```

It never modifies the source. It copies `sparse/0`, downsamples every registered image
(bicubic, JPEG quality 95, EXIF kept) and every mask (nearest neighbour, 0/255), and puts
each mask next to its image. It reports images that have no mask. It takes about 5 seconds
for 1,568 images. `--masks-dir` sets a different mask folder name.

**Resolution:** the 1472×1440 originals at `--factor 2` become 736×720. That is the setting
tested on this 8 GB GPU (~2.9 GB GPU memory at peak). Full resolution (`--factor 1`) is
untested here and will be slower and need much more memory.

> `/cache/s49imoni/data/may_session_v3_edit_gofish` is already prepared. Skip 1c for that scene.

---

## 2. Register a new camera (once per camera + resolution)

The renderer has a compiled table with one row per camera calibration and image size. On a camera
it hasn't seen, `train.py` **stops before training** and prints one command per missing camera:

```bash
python -m gofish.tools.build_calibration_table --camera=OPENCV_FISHEYE,736,720,0x1.d14d...
```

Run each printed command exactly as printed, then start training again. The renderer recompiles
itself on that next run (a few minutes). You can also register straight from COLMAP:

```bash
python -m gofish.tools.build_calibration_table \
    --colmap /cache/s49imoni/data/may_session_v3_edit_gofish/sparse/0/cameras.bin --downsample 2
```

> Your rig's two cameras at 736×720 are **already registered**. You only need this for a new
> rig/calibration or a different `--factor`. It edits
> `threedgut_tracer/include/3dgut/kernels/cuda/renderers/certifiedRemainderPackedKbTable.cuh`.
> Commit that file if you want the camera to stay registered.

---

## 3. Train

```bash
SCENE=/cache/s49imoni/data/may_session_v3_edit_gofish
RUNS=/cache/s49imoni/runs
NAME=gofish_v1                      # use a new name per run, or delete $RUNS/$NAME first

python train.py --config-name apps/fiord \
    path=$SCENE out_dir=$RUNS experiment_name=$NAME \
    dataset.camera_domains=[] dataset.downsample_factor=2 geometry=sof
```

What the arguments mean:

| Argument | Why |
|---|---|
| `--config-name apps/fiord` | fisheye preset (`OPENCV_FISHEYE`). For `PINHOLE` cameras use `apps/tnt`. |
| `dataset.camera_domains=[]` | required for your own data (otherwise it loads FIORD's lens apertures) |
| `dataset.downsample_factor=2` | must match the `--factor` you prepared with (reads `images_2/`) |
| `geometry=sof` | enables the Geometry Objective, giving much better meshes. Drop it if you only want renders. |

**Schedule.** `n_iterations` defaults to 35,000: **30,000 scene steps** (geometry checkpoint at
30k), then 5,000 steps that train only the colour/exposure controller. Checkpoints are saved at steps
650, 15,000 and 30,000, plus `ckpt_last.pt` at the end. For a quick smoke test add `n_iterations=7000`.

**Time.** ~10 steps/s at first, ~4 steps/s after densification (about 330k Gaussians).
**About 2 hours in total**, plus a few minutes of compiling on the first run.

### Run it so it survives closing the terminal

Training takes about 2 hours, so start it detached:

```bash
tmux new -s gofish                  # then run the train command inside; Ctrl-b d detaches
tmux attach -t gofish               # come back later
```

or, without tmux:

```bash
nohup python train.py --config-name apps/fiord path=$SCENE out_dir=$RUNS experiment_name=$NAME \
    dataset.camera_domains=[] dataset.downsample_factor=2 geometry=sof > $RUNS/$NAME.log 2>&1 &
tail -f $RUNS/$NAME.log
```

`/cache/s49imoni/runs/gofish_v1_run.sh` does train + mesh in one go for `gofish_v1`; you can copy
and edit it. With `nohup`/`tmux` the console progress bar isn't in the log. Watch progress
with TensorBoard instead:

```bash
tensorboard --logdir /cache/s49imoni/runs/$NAME
```

**Loss spikes are normal.** From step 15,000 the geometry losses switch on, and
`loss/distortion_loss` occasionally spikes (up to ~8) for a single step before dropping back.
Watch `loss/l1_loss` instead. It should stay around 0.015–0.03.

---

## 4. Extract the mesh

Use the **step-30,000 geometry checkpoint**, not `ckpt_last.pt`:

```bash
python mesh.py --checkpoint $RUNS/$NAME/ours_30000/ckpt_30000.pt --out $RUNS/$NAME/mesh
```

This writes `mesh/mesh.ply` (final), `mesh_sof.ply`, `mesh_raw.ply` and `mesh_info.json`.
These are the only `.ply` files GOFish produces. Unlike 3DGRUT, it does **not** export the
Gaussians as a splat `.ply`.

---

## 5. Render the held-out views (optional)

```bash
# real-time renderer
python render.py --checkpoint $RUNS/$NAME/ckpt_last.pt --out $RUNS/$NAME/heldout
# exact (paper) renderer + PSNR / SSIM / LPIPS in heldout/metrics.json
python render.py --checkpoint $RUNS/$NAME/ckpt_last.pt --out $RUNS/$NAME/heldout --renderer exact --metrics
```

If the dataset folder has moved since training, add `--data <scene>` (same for `mesh.py`).

---

## 6. Outputs

```
/cache/s49imoni/runs/<NAME>/
├── parsed.yaml                     # full resolved config of the run
├── events.out.tfevents.*           # TensorBoard logs
├── ours_650/ckpt_650.pt
├── ours_15000/ckpt_15000.pt
├── ours_30000/ckpt_30000.pt        # geometry checkpoint → mesh.py
├── ckpt_last.pt                    # final checkpoint (step 35,000) → render.py
├── mesh/{mesh.ply, mesh_sof.ply, mesh_raw.ply, mesh_info.json}
└── heldout/                        # render.py output (+ metrics.json)
```

---

## 7. Troubleshooting on this machine

| Symptom | Cause / fix |
|---|---|
| `TableError: complete admission needs a curvature-table row` | New camera or resolution. Run the printed `build_calibration_table` commands (§2). |
| `cusolverDnCreate ... status 7` a few hundred steps in | GPU ran out of memory. With per-frame masks this was a cache bug, fixed in `threedgrut/model/losses.py`. Keep that change. |
| Run killed with no error shortly after step 30,000 | Host RAM ran out in the controller phase (~56 MB cached per frame). Fixed in `threedgrut/trainer.py`: the cache is capped at half the free RAM. Lower it with `post_processing.controller_cache_max_gb=10` if other jobs need RAM. |
| GPU out of memory on a bigger scene | Downsample more (`--factor 4` + `dataset.downsample_factor=4`), or densify less: `strategy.densify.clone_grad_threshold=0.0004 strategy.densify.split_grad_threshold=0.0004`. |
| `ImportError` / wrong torch version | You're in the `3dgrut` env. Run `conda activate gofish`. |

### Local changes this pipeline depends on (not yet committed)

- `threedgrut/model/losses.py`: per-frame mask supports are no longer cached forever (GPU memory fix).
- `threedgrut/trainer.py` + `configs/base_gs.yaml`: RAM cap for the controller-phase render cache
  (`post_processing.controller_cache_max_gb`).
- `gofish/tools/prepare_scene.py`: the data-prep tool from §1c.
- `certifiedRemainderPackedKbTable.cuh`: calibration rows for your two rig cameras at 736×720.

Commit these (`git add -A && git commit`) so a fresh checkout of your fork keeps working.
