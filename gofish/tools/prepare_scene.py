#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convert a COLMAP scene with a separate mask folder into the layout GOFish reads.

Source layout (the image names stored in ``sparse/0/images.bin`` are paths
relative to the scene folder, e.g. ``images/cam0/0001.jpg``)::

    <source>/
    ├── sparse/0/{cameras,images,points3D}.bin   (+ frames.bin, rigs.bin if present)
    ├── images/<cam>/<frame>.jpg
    └── masks/<cam>/<frame>.png                  optional; white (>127) = use pixel

Target layout written by this tool (the source is never modified)::

    <target>/
    ├── sparse/0/...                             copied unchanged
    └── images_<k>/images/<cam>/<frame>.jpg      downsampled by k (bicubic, JPEG q95, EXIF kept)
                 images/<cam>/<frame>_mask.png   downsampled by k (nearest), 0/255

Example::

    python -m gofish.tools.prepare_scene \
        --source /cache/s49imoni/SLAM_repo/data_V3/may_session_v3_edit \
        --target /cache/s49imoni/data/may_session_v3_edit_gofish --factor 2
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image

from gofish.eval.common.colmap import read_images


def _mask_source(source: Path, name: str, masks_dir: str) -> Path:
    """``images/cam0/x.jpg`` -> ``<source>/masks/cam0/x.png``."""
    relative = Path(name)
    if relative.parts and relative.parts[0] == "images":
        relative = Path(*relative.parts[1:])
    return source / masks_dir / relative.with_suffix(".png")


def _convert(name: str, source: Path, target: Path, factor: int, masks_dir: str) -> str:
    src = source / name
    dst = target / f"images_{factor}" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        return f"missing image {src}"
    if not dst.exists():
        with Image.open(src) as image:
            size = (image.width // factor, image.height // factor)
            out = image.resize(size, Image.Resampling.BICUBIC) if factor > 1 else image.copy()
            kwargs = {"quality": 95}
            if image.info.get("exif"):
                kwargs["exif"] = image.info["exif"]
            out.save(dst, **kwargs)
    mask_src = _mask_source(source, name, masks_dir)
    mask_dst = dst.with_name(dst.stem + "_mask.png")
    if not mask_src.is_file():
        return f"no mask {mask_src}"
    if not mask_dst.exists():
        with Image.open(mask_src) as mask:
            mask = mask.convert("L")
            if factor > 1:
                mask = mask.resize((mask.width // factor, mask.height // factor), Image.Resampling.NEAREST)
            mask.point(lambda v: 255 if v > 127 else 0).save(mask_dst)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="scene folder with sparse/0, images/, masks/")
    parser.add_argument("--target", type=Path, required=True, help="output folder (created)")
    parser.add_argument("--factor", type=int, default=2, help="downsample factor k -> images_<k>/ (default 2)")
    parser.add_argument("--masks-dir", default="masks", help="mask folder name inside --source (default masks)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    model = args.source / "sparse" / "0"
    if not (model / "images.bin").is_file():
        raise SystemExit(f"{model}/images.bin not found")
    if args.source.resolve() == args.target.resolve():
        raise SystemExit("--target must differ from --source")
    sparse = args.target / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for f in model.iterdir():
        if f.is_file():
            shutil.copyfile(f, sparse / f.name)

    names = sorted(image.name for image in read_images(model / "images.bin"))
    work = partial(_convert, source=args.source, target=args.target, factor=args.factor, masks_dir=args.masks_dir)
    with ProcessPoolExecutor(args.workers) as pool:
        problems = [p for p in pool.map(work, names, chunksize=16) if p]
    missing_images = [p for p in problems if p.startswith("missing image")]
    missing_masks = [p for p in problems if p.startswith("no mask")]
    print(f"{args.target}: {len(names)} registered images at 1/{args.factor}")
    if missing_masks:
        print(f"  {len(missing_masks)} images have no mask (trained on all pixels), e.g. {missing_masks[0]}")
    if missing_images:
        raise SystemExit(f"  {len(missing_images)} registered images not found, e.g. {missing_images[0]}")


if __name__ == "__main__":
    main()
