"""Train a Noise2Void model on photosite planes and save it under models/.

  train_n2v.py random  models/n2v_random.pt           # 2 random 512px crops per frame
  train_n2v.py fish    models/n2v_fish.pt             # crop centred on the labelled fish (+40% random)
  train_n2v.py fish    models/n2v_fish_l2.pt --levels 2

Seeded (20260903) so the same frames give equivalent weights; CUDA convolutions
are not bit-deterministic across runs, so not identical ones.
"""
import argparse
import sys
import time

import numpy as np
import rawpy
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import enhancement_eval  # noqa: F401
from enhancement_eval.mosaic import split_cfa
from enhancement_eval.n2v import N2VConfig, PLANE_ORDER, train
from scripts._common import DATA, MODELS, ROOT, rows_of

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["random", "fish"]); ap.add_argument("out")
ap.add_argument("--levels", type=int, default=3); ap.add_argument("--mask-width", type=int, default=5)
ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--base", type=int, default=32)
ap.add_argument("--cross-plane", type=int, default=1)
a = ap.parse_args()
CROP = 512
rows = rows_of(DATA / ("n2v_random.csv" if a.mode == "random" else "n2v_fish.csv"))
rng = np.random.default_rng(20260903 if a.mode == "random" else 7)
crops, scale, dives, n_fish, t0 = [], None, set(), 0, time.time()
for i, row in enumerate(rows):
    path = ROOT / row["image_path"]
    if not path.exists(): continue
    with rawpy.imread(str(path)) as raw:
        black = float(np.mean(raw.black_level_per_channel)); white = float(raw.white_level)
        planes = split_cfa(raw.raw_image_visible.astype(np.float32) - black, raw.raw_pattern, raw.color_desc.decode())
    scale = scale or (white - black)
    stack = np.stack([planes[k] for k in PLANE_ORDER]).astype(np.float32) / scale
    _, h, w = stack.shape
    if a.mode == "random":
        for _ in range(2):
            y, x = int(rng.integers(0, h - CROP)), int(rng.integers(0, w - CROP))
            crops.append(stack[:, y:y + CROP, x:x + CROP].copy())
    else:
        cx = (float(row["head_x"]) + float(row["tail_x"])) / 4; cy = (float(row["head_y"]) + float(row["tail_y"])) / 4
        y, x = int(np.clip(cy - CROP / 2, 0, h - CROP)), int(np.clip(cx - CROP / 2, 0, w - CROP))
        crops.append(stack[:, y:y + CROP, x:x + CROP].copy()); n_fish += 1
        if rng.random() < 0.4:
            y, x = int(rng.integers(0, h - CROP)), int(rng.integers(0, w - CROP))
            crops.append(stack[:, y:y + CROP, x:x + CROP].copy())
    dives.add(row["dive_id"])
    if i % 20 == 0: print(f"  [{i}/{len(rows)}] {len(crops)} crops {time.time() - t0:.0f}s", flush=True)
data = np.stack(crops)
print(f"dataset {data.shape} from {len(dives)} dives ({n_fish} fish-centred)", flush=True)
cfg = N2VConfig(steps=a.steps, patch=64, batch=32, base=a.base, levels=a.levels, mask_width=a.mask_width,
                cross_plane=bool(a.cross_plane), seed=20260903)
t0 = time.time()
model = train(data, cfg, log=lambda s, l: print(f"  step {s:5d} loss {l:.6f} {time.time() - t0:5.0f}s", flush=True)
              if s % 2000 == 0 or s == a.steps - 1 else None)
MODELS.mkdir(exist_ok=True)
torch.save({"state_dict": model.state_dict(), "base": cfg.base, "levels": cfg.levels, "mask_width": cfg.mask_width,
            "cross_plane": cfg.cross_plane, "scale": scale, "mode": a.mode, "dives": sorted(dives),
            "n_crops": len(crops), "n_fish_crops": n_fish, "seed": cfg.seed, "steps": cfg.steps}, a.out)
print(f"saved {a.out}. The loss above is against noisy targets and proves nothing.\nDONE", flush=True)
