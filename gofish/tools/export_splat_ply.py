"""Export the Gaussians of a GOFish checkpoint as a standard 3DGS (INRIA layout) splat .ply.

The file can be opened in any Gaussian-splat web viewer (SuperSplat, antimatter15 splat,
PlayCanvas, ...). The 3D Mip filter is baked into the scales (``sqrt(s^2 + filter_3D^2)``),
matching what GOFish's own renderer draws; opacity is left unchanged, as in the renderer.

    python -m gofish.tools.export_splat_ply --checkpoint RUN/ckpt_last.pt --out RUN/splat.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def export(checkpoint: Path, out: Path, bake_filter: bool = True, min_opacity: float = 0.0) -> int:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    conf = ckpt["config"].model
    if conf.density_activation != "sigmoid" or conf.scale_activation != "exp":
        raise ValueError(f"expected sigmoid/exp activations, got {conf.density_activation}/{conf.scale_activation}")

    pos = ckpt["positions"].detach().float()
    n = pos.shape[0]
    rot = torch.nn.functional.normalize(ckpt["rotation"].detach().float(), dim=1)  # w, x, y, z
    log_scale = ckpt["scale"].detach().float()
    opacity_logit = ckpt["density"].detach().float()
    f_dc = ckpt["features_albedo"].detach().float()
    # GOFish/3DGRUT store higher-order SH coefficient-major [N, K, 3]; INRIA stores channel-major [N, 3, K].
    spec = ckpt["features_specular"].detach().float()
    k = spec.shape[1] // 3
    f_rest = spec.reshape(n, k, 3).transpose(1, 2).reshape(n, 3 * k)

    filt = ckpt.get("filter_3D")
    if bake_filter and filt is not None:
        s = torch.exp(log_scale)
        f = filt.detach().float().reshape(n, 1)
        log_scale = torch.log(torch.sqrt(s * s + f * f))

    keep = torch.ones(n, dtype=torch.bool)
    if min_opacity > 0:
        keep = torch.sigmoid(opacity_logit[:, 0]) >= min_opacity

    cols = [pos, torch.zeros_like(pos), f_dc, f_rest, opacity_logit, log_scale, rot]
    data = torch.cat(cols, dim=1)[keep].numpy().astype("<f4")

    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert len(names) == data.shape[1]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {data.shape[0]}\n"
    header += "".join(f"property float {nm}\n" for nm in names)
    header += "end_header\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(data).tobytes())
    return int(data.shape[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-filter", action="store_true", help="export raw scales without the 3D Mip filter")
    p.add_argument("--min-opacity", type=float, default=0.0, help="drop Gaussians below this opacity")
    a = p.parse_args()
    count = export(a.checkpoint, a.out, bake_filter=not a.no_filter, min_opacity=a.min_opacity)
    print(f"wrote {count} Gaussians to {a.out}")


if __name__ == "__main__":
    main()
