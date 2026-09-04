"""Shared glue for the evaluation scripts: the production JPEG chain, the
mosaic-domain N2V decode, and the fish/water patch conventions every number
in MEASUREMENTS.md uses. Lives in the repo because a reboot took the first
version with the session scratchpad."""
from __future__ import annotations

import ast
import math
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "data"
MODELS = REPO / "models"
ROOT = pathlib.Path.home() / "mnt/fishsense_data/REEF/data"
POOL = {"65", "71", "77", "80", "83"}
RAW_KW = dict(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16, user_flip=0)


def camera(row):
    return (np.asarray(ast.literal_eval(row["camera_matrix"]), dtype=float),
            np.asarray(ast.literal_eval(row["distortion_coefficients"]), dtype=float))


def finish(rgb16, K, D):
    """Production's JPEG chain with the shipped L* stretch: uint8 BGR, rectified."""
    import cv2
    from skimage.exposure import adjust_gamma
    from skimage.util import img_as_float, img_as_ubyte

    from enhancement_eval.decode import DecodeConfig
    from enhancement_eval.stages import apply_stretch

    rec = DecodeConfig(stretch_mode="luminance", clahe_enabled=False)
    img = img_as_float(rgb16)
    val = cv2.split(cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV))[2]
    toned = adjust_gamma(np.clip(img, 0, 1),
                         gamma=1 / (math.log(20 * 255) / math.log(max(float(np.mean(val)), 1.01))))
    bgr = img_as_ubyte(np.clip(apply_stretch(toned, rec), 0, 1))[:, :, ::-1]
    return cv2.undistort(bgr, K, D)


def decode(path, K, D, model=None, scale=None):
    """Shipped decode, or the same with an N2V model applied to the mosaic first."""
    import rawpy

    from enhancement_eval.mosaic import merge_cfa, split_cfa
    from enhancement_eval.n2v import PLANE_ORDER, denoise_planes

    with rawpy.imread(str(path)) as raw:
        if model is not None:
            black = float(np.mean(raw.black_level_per_channel))
            visible = raw.raw_image_visible
            pattern, desc = np.asarray(raw.raw_pattern), raw.color_desc.decode()
            planes = split_cfa(visible.astype(np.float32) - black, pattern, desc)
            stack = np.stack([planes[k] for k in PLANE_ORDER]).astype(np.float32) / scale
            clean = denoise_planes(model, stack) * scale
            merged = merge_cfa({k: clean[i] for i, k in enumerate(PLANE_ORDER)}, pattern, desc, visible.shape) + black
            info = np.iinfo(visible.dtype)
            visible[:] = np.clip(np.rint(merged), info.min, info.max).astype(visible.dtype)
        return finish(raw.postprocess(**RAW_KW), K, D)


def load_model(name):
    import torch

    from enhancement_eval.n2v import BlindSpotUNet

    ck = torch.load(MODELS / name, weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = BlindSpotUNet(channels=4, base=ck["base"], levels=ck.get("levels", 3)).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["scale"]


def fish_patch(row, H, W):
    """Square on the fish body between the head/tail keypoints; 0.6x the
    head-tail distance, clamped to [128, 384]."""
    hx, hy, tx, ty = (float(row[k]) for k in ("head_x", "head_y", "tail_x", "tail_y"))
    cx, cy = (hx + tx) / 2, (hy + ty) / 2
    side = int(np.clip(0.6 * np.hypot(hx - tx, hy - ty), 128, 384))
    y0, x0 = int(np.clip(cy - side / 2, 0, H - side)), int(np.clip(cx - side / 2, 0, W - side))
    return np.s_[y0:y0 + side, x0:x0 + side], (cx, cy), side


def water_patch(H, W):
    """Top-left fifth: the open-water convention for every noise figure."""
    return np.s_[0:H // 5, 0:W // 5]


def has_headtail(row):
    return row.get("head_x") not in (None, "", "None")


def rows_of(csv_path):
    import csv

    return list(csv.DictReader(open(csv_path, newline="")))
