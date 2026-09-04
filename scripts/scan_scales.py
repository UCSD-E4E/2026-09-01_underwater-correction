"""Which held-out big-fish frames resolve a scale peak on the shipped decode?
Keeps those whose baseline scale-band peak clears texture.SIGNIFICANCE.
  scan_scales.py START STRIDE WANT   -> data/scaled_START.csv
Run four with stride 4, then merge_scaled below."""
import csv
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import enhancement_eval  # noqa: F401
from enhancement_eval.texture import SIGNIFICANCE, texture_power
from scripts._common import DATA, ROOT, camera, decode, fish_patch, rows_of

if sys.argv[1] == "merge":
    rows = []
    for i in range(4): rows += rows_of(DATA / f"scaled_{i}.csv")
    rows.sort(key=lambda r: -float(r["peak_sigma"]))
    with open(DATA / "scaled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"merged {len(rows)} -> scaled.csv", [(r["dive_id"], r["peak_sigma"]) for r in rows]); sys.exit()
START, STRIDE, WANT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cands = rows_of(DATA / "bigfish.csv")[START::STRIDE]
keep, fields = [], None
for i, row in enumerate(cands):
    path = ROOT / row["image_path"]
    if not path.exists(): continue
    K, D = camera(row)
    lum = decode(path, K, D).astype(np.float64).mean(axis=2)
    sl, _, side = fish_patch(row, *lum.shape)
    prom, noise = texture_power(lum[sl]); sig = prom / noise if noise > 0 else 0
    row = dict(row, peak_sigma=round(sig, 1)); fields = fields or list(row.keys())
    if sig >= SIGNIFICANCE: keep.append(row)
    print(f"  [{i + 1}/{len(cands)}] dive {row['dive_id']:>4} peak {sig:6.1f} sigma {'KEEP' if sig >= SIGNIFICANCE else ''}", flush=True)
    if len(keep) >= WANT: break
with open(DATA / f"scaled_{START}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(keep)
print(f"kept {len(keep)} -> scaled_{START}.csv\nDONE", flush=True)
