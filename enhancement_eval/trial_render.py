"""Render one JPEG per allocated frame, each through its own arm's decode.

Separate module because it is the only part of the trial that needs the heavy
imports, and because keeping the pool's worker function at module scope is what
lets `spawn` pickle it.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["render_allocation"]

#: Production's JPEG quality, from `Image.to_jpeg_bytes`.
JPEG_QUALITY = 95


def _one(task):
    row, config, raw_root, outdir = task
    import ast

    import cv2

    from enhancement_eval.paths import resolve_under_root
    from enhancement_eval.stages import decode_rectified_stage, rectify

    src = resolve_under_root(raw_root, row["image_path"])
    if src is None:
        return f"MISSING {row['checksum']}"
    try:
        bgr = decode_rectified_stage(src.read_bytes(), config)
        k = row["camera_matrix"]
        d = row["distortion_coefficients"]
        if isinstance(k, str):
            k, d = ast.literal_eval(k), ast.literal_eval(d)
        bgr = rectify(bgr, k, d)
    except Exception as exc:  # pylint: disable=broad-except
        return f"FAILED  {row['checksum']}: {type(exc).__name__}: {exc}"

    # Same filename in both arms. The arm lives in allocation.csv, not in the
    # name, so nothing downstream -- and no labeler glancing at a URL -- can
    # read the condition off the file itself.
    path = Path(outdir) / f"{row['checksum']}.JPG"
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return f"ok      {row['checksum']} [{row['arm']}]"


def render_allocation(
    allocation: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Any],
    raw_root: Path,
    outdir: Path,
    *,
    jobs: int = 6,
) -> None:
    """Render every allocated frame through its assigned arm's decode."""
    tasks = [
        (dict(row), arms[row["arm"]], str(raw_root), str(outdir)) for row in allocation
    ]
    print(f"\nrendering {len(tasks)} JPEGs on {jobs} processes")
    failed = 0
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        for i, msg in enumerate(pool.map(_one, tasks, chunksize=1), 1):
            if not msg.startswith("ok"):
                failed += 1
                print(f"  {msg}")
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    print(f"done: {len(tasks) - failed} rendered, {failed} failed -> {outdir}")
