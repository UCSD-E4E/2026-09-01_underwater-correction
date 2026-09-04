"""Scale retention, grain and displacement for N2V models and BM3D variants
against the shipped decode, on a frames CSV. Writes one CSV and 1:1 crops.

  eval_scales.py data/heldout.csv data/eval_heldout.csv \
      --n2v models/n2v_random.pt models/n2v_fish.pt --bm3d 0.5:0 0.5:1 1.0:1

BM3D runs on a 1024 px crop around the fish (full-frame three-channel BM3D
is ~13 min per frame); its grain figure processes the water region on its
own against its own PSD, which is the ideal case and reads ~1.4x optimistic
next to N2V's full-frame figure. Both stated in MEASUREMENTS.md.
"""
import argparse
import csv
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".pylib"))
import enhancement_eval  # noqa: F401
from skimage.color import lab2rgb, rgb2lab

from enhancement_eval.bm3d_arm import BM3DConfig, denoise_luminance
from enhancement_eval.texture import grain_reduction, texture_power, texture_retention
from scripts._common import DATA, ROOT, camera, decode, fish_patch, has_headtail, load_model, rows_of, water_patch

ap = argparse.ArgumentParser()
ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--n2v", nargs="*", default=[]); ap.add_argument("--bm3d", nargs="*", default=[])
a = ap.parse_args()
models = {m.split("/")[-1].replace(".pt", ""): load_model(m.split("/")[-1]) for m in a.n2v}
bm3d_variants = {f"bm3d{s}" + ("+chroma" if float(c) > 0 else ""): (float(s), float(c)) for s, c in (v.split(":") for v in a.bm3d)}
crops = DATA / "crops"; crops.mkdir(exist_ok=True)
hann = cv2.createHanningWindow((512, 512), cv2.CV_64F)


def bm3d_run(rgb, water_rgb, strength, chroma):
    lab = rgb2lab(rgb.astype(np.float64) / 255.0); wlab = rgb2lab(water_rgb.astype(np.float64) / 255.0)
    lab[..., 0] = denoise_luminance(lab[..., 0] * 2.55, wlab[..., 0] * 2.55, BM3DConfig(strength=strength)) / 2.55
    if chroma > 0:
        for c in (1, 2):
            lab[..., c] = denoise_luminance(lab[..., c] + 128.0, wlab[..., c] + 128.0, BM3DConfig(strength=chroma)) - 128.0
    return np.clip(np.rint(lab2rgb(lab) * 255), 0, 255).astype(np.uint8)


def shift(a, b):
    (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), hann); return float(np.hypot(dx, dy))


results = []
for row in rows_of(a.frames):
    if not has_headtail(row): continue
    path = ROOT / row["image_path"]
    if not path.exists(): continue
    K, D = camera(row)
    base = decode(path, K, D); H, W = base.shape[:2]; base_l = base.astype(np.float64).mean(axis=2)
    sl, (cx, cy), side = fish_patch(row, H, W); wsl = water_patch(H, W)
    prom, noise = texture_power(base_l[sl])
    rec = {"dive": row["dive_id"], "image": row["image_id"], "fish_px": side, "peak_sigma": round(prom / noise, 1)}
    line = f"  dive {row['dive_id']:>4} peak {prom / noise:5.1f}s"
    cy0, cx0 = int(np.clip(cy - 320, 0, H - 640)), int(np.clip(cx - 320, 0, W - 640))
    for name, (m, sc) in models.items():
        t0 = time.time(); enh = decode(path, K, D, m, sc); enh_l = enh.astype(np.float64).mean(axis=2)
        r = texture_retention(base_l[sl], enh_l[sl]); g = grain_reduction(base_l[wsl], enh_l[wsl])
        s = shift(base_l[H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256], enh_l[H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256])
        rec.update({f"{name}:texture": round(r, 3), f"{name}:grain": round(g, 2), f"{name}:shift_px": round(s, 3)})
        line += f"  {name}: ret {r:.2f} /{g:.2f} {s:.3f}px ({time.time() - t0:.0f}s)"
        cv2.imwrite(str(crops / f"dive{row['dive_id']}_{name}.png"), np.hstack([base[cy0:cy0 + 640, cx0:cx0 + 640], enh[cy0:cy0 + 640, cx0:cx0 + 640]]))
    if bm3d_variants:
        x0, y0 = int(np.clip(cx - 512, 0, W - 1024)), int(np.clip(cy - 512, 0, H - 1024))
        crop = base[y0:y0 + 1024, x0:x0 + 1024]; water_rgb = base[wsl][:, :, ::-1]
        fy, fx = sl[0].start - y0, sl[1].start - x0; fsl = np.s_[fy:fy + side, fx:fx + side]
        crop_l = rgb2lab(crop[:, :, ::-1].astype(np.float64) / 255.0)[..., 0] * 2.55
        for name, (s_, c_) in bm3d_variants.items():
            t0 = time.time()
            enh = bm3d_run(crop[:, :, ::-1], water_rgb, s_, c_)[:, :, ::-1]
            w_enh = bm3d_run(water_rgb, water_rgb, s_, c_)[:, :, ::-1]
            enh_l = rgb2lab(enh[:, :, ::-1].astype(np.float64) / 255.0)[..., 0] * 2.55
            r = texture_retention(crop_l[fsl], enh_l[fsl]); g = grain_reduction(base_l[wsl], w_enh.astype(np.float64).mean(axis=2))
            s = shift(crop.astype(np.float64).mean(axis=2)[256:768, 256:768], enh.astype(np.float64).mean(axis=2)[256:768, 256:768])
            rec.update({f"{name}:texture": round(r, 3), f"{name}:grain": round(g, 2), f"{name}:shift_px": round(s, 3)})
            line += f"  {name}: ret {r:.2f} /{g:.2f} {s:.3f}px ({time.time() - t0:.0f}s)"
            cv2.imwrite(str(crops / f"dive{row['dive_id']}_{name}.png"), np.hstack([crop[192:832, 192:832], enh[192:832, 192:832]]))
    results.append(rec); print(line, flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in results for k in r})); w.writeheader(); w.writerows(results)
print("\nmeans:")
for k in sorted({k for r in results for k in r if k.endswith(":texture")}):
    t = [r[k] for r in results if k in r and not np.isnan(r[k])]; g = [r[k.replace(":texture", ":grain")] for r in results if k in r]
    print(f"  {k.split(':')[0]:<20} scale retention {np.mean(t):.2f} (n={len(t)})   grain /{np.mean(g):.2f}")
print("DONE", flush=True)
