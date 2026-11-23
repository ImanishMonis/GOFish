# Setup issues and fixes

This file covers the problems we hit getting GOFish to run on our own scene
(`may_session_v3_edit`) on this machine, and what was changed to fix each one.
All of these changes are in commit `7cdfdc1`.

- Machine: RTX 2080 SUPER (8 GB GPU memory), 62 GB RAM, conda env `gofish`.
- Data: 1,568 registered images from a two-camera fisheye rig, 1472×1440, downsampled to
  736×720. Every image has its own mask. Every 8th image is held out for testing, which
  leaves about 1,370 training frames.

The three memory problems all have the same root cause. **The code kept a copy of something
for every image (frame).** The upstream code was written for datasets that have a few masks
per camera. Ours has a different mask for every image and ~1.4k frames, so these per-frame
copies grew until the GPU or RAM was full.

| # | Problem | Where it broke | File(s) changed |
|---|---|---|---|
| 1 | GPU out of memory a few hundred steps into training | `train.py` | `threedgrut/model/losses.py` |
| 2 | Process killed with no error just after step 30,000 | `train.py` (controller phase) | `threedgrut/trainer.py`, `configs/base_gs.yaml` |
| 3 | GPU memory used per frame instead of per camera during mesh extraction | `mesh.py` | `gofish/mesh/bisection.py`, `gofish/mesh/native/.../opacityFieldBinding.inl` |
| 4 | Training stops: camera not in the calibration table | `train.py` (startup) | `threedgut_tracer/.../certifiedRemainderPackedKbTable.cuh` |
| 5 | Our data layout is not what GOFish reads | data prep | new `gofish/tools/prepare_scene.py` |
| 6 | No way to view the Gaussians in a splat viewer | after training | new `gofish/tools/export_splat_ply.py` |

---

## 1. GPU out of memory during training (masked-loss cache)

**Symptom.** A few hundred steps into training, the run crashed with
`cusolverDnCreate ... status 7`. That is the GPU running out of memory, not a cuSOLVER bug.

**Cause.** `MaskedLossCache` in `threedgrut/model/losses.py` builds a "support" for the SSIM
loss from each frame's mask. A support holds the mask, the SSIM-window support map and flat
gather indices, about **17 MB per frame** at 736×720. The cache stored every support it built
and never removed any. Upstream that's fine: masks are usually per camera, so there are only a
few entries. With a different mask per image, each new training frame adds an entry:

```
~17 MB × ~1,370 frames ≈ 23 GB  >  8 GB GPU
```

Memory grew with every new frame drawn until the GPU was full.

**Fix.** Only per-camera supports (`frame_idx < 0`) are cached now. A per-frame support is
built, used for that step, and then freed:

```python
# threedgrut/model/losses.py, MaskedLossCache
if frame_idx < 0:
    self._support[key] = entry
return entry
```

**Cost.** Building a per-frame support takes a small amount of extra time each step. The
loss values don't change.

---

## 2. Run killed after step 30,000 (controller render cache uses up RAM)

**Symptom.** Training reached step 30,000, then the process was killed with no Python error.
That is the Linux OOM killer, because host RAM was full.

**Cause.** After 30,000 steps, training switches to a 5,000-step "controller-only" phase
(PPISP colour/exposure controller). The scene is frozen in this phase, so `threedgrut/trainer.py`
renders each training frame once and stores every render output in host RAM
(`_controller_render_cache`), about **56 MB per frame**. The cache had no size limit:

```
~56 MB × ~1,370 frames ≈ 77 GB  >  62 GB RAM
```

**Fix.** The cache now has a size budget:

- `_host_available_bytes()` reads `MemAvailable` from `/proc/meminfo`.
- `_controller_cache_has_room()` sets the budget the first time it's called: half the RAM
  that's free then, unless you set it yourself.
- New frames are cached only while they fit in the budget. Frames that don't fit are
  re-rendered whenever they're needed. The scene is frozen, so the result is the same, only slower.
- The log prints the budget once: `controller-phase render cache budget: X GB of host RAM`.

New config option in `configs/base_gs.yaml`:

```yaml
post_processing:
  controller_cache_max_gb: null   # null = half of available RAM; or a number in GB
```

Set a lower limit if other jobs need RAM, for example `post_processing.controller_cache_max_gb=10`.

---

## 3. Mesh extraction kept GPU buffers per frame

**Cause.** In `mesh.py`, the bisection step (`gofish/mesh/bisection.py` → native
`stageBisectionCamera`) adds every training camera to a table on the GPU. For each frame, the
table kept its own copy of the ray-origin raster and the mask. All frames from the same
physical camera have identical buffers, so we were storing the same data ~1,370 times instead
of 2 times (two physical cameras). The batch-size estimate in `bisection.py` also counted
memory per frame. That made it think there was much less free GPU memory than there really was.

**Fix.**

- `opacityFieldBinding.inl`: new `opacityFieldShareTableBuffer()`. Before keeping a buffer
  for the table, it checks the buffers already kept (same dtype, same shape, `torch::equal`).
  If one matches, it reuses that buffer. GPU memory is now per physical camera, not per frame.
  The staging tensors (`currentRayOrigin`, `currentPhysicalMask`) are also released right after
  each camera is staged.
- `bisection.py`: the memory estimate counts each distinct origin raster and each distinct
  mask once, plus room for the camera currently being staged (`3 × largest raster`). That
  gives a realistic batch size.

**Note.** This needs the native mesh extension to be rebuilt. That happens automatically on
the next `mesh.py` run.

---

## 4. "complete admission needs a curvature-table row" (new camera)

**Symptom.** `train.py` stops before training starts with
`TableError: complete admission needs a curvature-table row` and prints a
`build_calibration_table` command for each camera.

**Cause.** The renderer uses a compiled table with one row per camera calibration and image
size. Our two rig cameras at 736×720 weren't in it.

**Fix.** We ran the printed `python -m gofish.tools.build_calibration_table --camera=...`
commands. They added two rows (selectors 76 and 77) to
`threedgut_tracer/include/3dgut/kernels/cuda/renderers/certifiedRemainderPackedKbTable.cuh`.
The table grew from 51,936 to 53,392 bytes. Keep this file committed, or a fresh checkout
will stop with the same error. You'll need to do this again for a new rig or a different
`--factor`.

---

## 5. Data layout (new tool `prepare_scene.py`)

**Problem.** Our scene has `images/<cam>/<frame>.jpg` and a separate `masks/<cam>/<frame>.png`
folder. GOFish reads `images_<k>/<image name>` and expects each mask **next to** its image,
named `<frame>_mask.png`.

**Fix.** New `gofish/tools/prepare_scene.py`. It:

- copies `sparse/0` unchanged
- downsamples each registered image by `k` (bicubic, JPEG quality 95, EXIF kept)
- downsamples each mask (nearest neighbour), sets it to 0/255, and writes it next to its image
- reports images that have no mask, and stops if a registered image is missing
- never modifies the source folder

```bash
python -m gofish.tools.prepare_scene \
    --source /cache/s49imoni/SLAM_repo/data_V3/may_session_v3_edit \
    --target /cache/s49imoni/data/may_session_v3_edit_gofish --factor 2
```

---

## 6. Exporting a splat .ply (new tool `export_splat_ply.py`)

GOFish only writes mesh `.ply` files. To look at the trained Gaussians in a normal 3DGS viewer
(SuperSplat, PlayCanvas, etc.), we added `gofish/tools/export_splat_ply.py`. It writes the
standard INRIA 3DGS layout. It changes the higher-order SH coefficients from GOFish's
coefficient-major order to INRIA's channel-major order. It also bakes the 3D Mip filter into
the scales, so the viewer draws what GOFish's renderer draws.

```bash
python -m gofish.tools.export_splat_ply \
    --checkpoint /cache/s49imoni/runs/<NAME>/ckpt_last.pt \
    --out /cache/s49imoni/runs/<NAME>/<NAME>.ply
# options: --no-filter (raw scales), --min-opacity 0.05 (drop faint Gaussians)
```

---

## Summary of changed files

| File | Change |
|---|---|
| `threedgrut/model/losses.py` | Don't cache per-frame mask supports (GPU memory) |
| `threedgrut/trainer.py` | RAM budget for controller-phase render cache |
| `configs/base_gs.yaml` | New option `post_processing.controller_cache_max_gb` |
| `gofish/mesh/bisection.py` | Memory estimate counts each distinct buffer once |
| `gofish/mesh/native/include/3dgut/kernels/cuda/common/opacityFieldBinding.inl` | Bisection table shares identical buffers |
| `threedgut_tracer/include/3dgut/kernels/cuda/renderers/certifiedRemainderPackedKbTable.cuh` | Calibration rows for our two cameras at 736×720 |
| `gofish/tools/prepare_scene.py` | New: convert our COLMAP + masks layout to GOFish layout |
| `gofish/tools/export_splat_ply.py` | New: export Gaussians as a standard splat `.ply` |
| `instructions.md`, `commands.txt` | How to run the pipeline on this machine |
