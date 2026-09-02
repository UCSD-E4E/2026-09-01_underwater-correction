"""Run the per-image sweep across processes.

Sizing, measured rather than assumed: on a 4014x3016 frame the decode splits as
`rawpy.postprocess` 2.2 s, auto-gamma 0.9 s, **`equalize_adapthist` 10.2 s**,
final conversion 0.6 s. CLAHE is 73% of the cost.

Two consequences. First, caching the `postprocess` output across arms -- the
obvious optimization, since every `clahe-jpeg` arm shares one white balance --
would save 16%, and the CLAHE sweep varies precisely the expensive part, so
there is nothing to reuse. Second, the work is CPU-bound and single-threaded
per frame, so processes are the whole win: 8 workers turn a 13-hour sweep into
under two.

Each worker builds its own consumers. Model objects do not survive a fork
cleanly (torch and the ONNX session both hold native state), and rebuilding is
a one-off cost amortized over hundreds of frames.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Sequence

__all__ = ["run_sweep"]

# Per-process state. Built once in `_init_worker`, reused for every frame that
# process handles.
_ARMS: list = []
_LASER = None
_HEADTAIL = None
_SLATE = None
_TEMPLATES: dict = {}
_OPTS: dict = {}


def _init_worker(experiment: str, opts: dict) -> None:
    """Build this process's arms and consumers."""
    global _ARMS, _LASER, _HEADTAIL, _SLATE, _OPTS  # pylint: disable=global-statement

    from enhancement_eval.evaluate import (
        HeadTailConsumer,
        LaserConsumer,
        SlateConsumer,
        Stage,
    )
    from enhancement_eval.experiments import build_arms

    _OPTS = opts
    _ARMS = build_arms(experiment)
    stages = {arm.stage for arm in _ARMS}

    if Stage.LINEAR in stages:
        _LASER = LaserConsumer(opts.get("checkpoint"))
    if Stage.RECTIFIED in stages:
        if opts.get("fish"):
            _HEADTAIL = HeadTailConsumer()
        if opts.get("slate"):
            _SLATE = SlateConsumer(use_board_mask=opts.get("board_mask", False))


def _template_for(row: dict):
    """This process's cached template for the row's dive slate."""
    from enhancement_eval.evaluate import SlateTemplate
    from enhancement_eval.paths import resolve_under_root

    slate_id = row.get("dive_slate_id")
    if not slate_id or not row.get("slate_template_points"):
        return None
    slate_id = int(float(slate_id))
    if slate_id in _TEMPLATES:
        return _TEMPLATES[slate_id]

    pdf = None
    slate_dir = _OPTS.get("slate_dir")
    if slate_dir and os.path.exists(os.path.join(slate_dir, f"{slate_id}.pdf")):
        pdf = os.path.join(slate_dir, f"{slate_id}.pdf")
    else:
        found = resolve_under_root(_OPTS.get("slate_root"), row.get("slate_path"))
        pdf = str(found) if found else None

    if pdf is None:
        _TEMPLATES[slate_id] = None
        return None

    with open(pdf, "rb") as handle:
        _TEMPLATES[slate_id] = SlateTemplate.from_pdf(
            slate_id,
            str(row.get("slate_name") or ""),
            int(float(row.get("slate_dpi") or 300)),
            handle.read(),
            row["slate_template_points"],
        )
    return _TEMPLATES[slate_id]


def _score_one(row: dict) -> tuple[str, list[dict]]:
    """Score one frame. Returns (status, records).

    Never raises: one unreadable frame among several hundred must not take the
    sweep down, but a silent skip would quietly change the population being
    measured, so the reason comes back with the result and is counted.
    """
    from enhancement_eval.cli import _load_raw
    from enhancement_eval.evaluate import evaluate_image

    raw = _load_raw(row, _OPTS.get("raw_dir_path"), _OPTS.get("raw_root_path"))
    if raw is None:
        return "missing_raw", []

    try:
        records = evaluate_image(
            row,
            raw,
            _ARMS,
            laser=_LASER,
            headtail=_HEADTAIL,
            slate=_SLATE,
            slate_template=_template_for(row) if _SLATE is not None else None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"failed:{type(exc).__name__}:{exc}", []

    for record in records:
        arm = next(a for a in _ARMS if a.label == record["arm"])
        record["probe_displacement_px"] = arm.probe.displacement_px if arm.probe else None
    return "ok", records


def run_sweep(
    rows: Sequence[dict], experiment: str, opts: dict, jobs: int
) -> Iterator[tuple[dict, str, list[dict]]]:
    """Yield `(row, status, records)` per frame, in input order.

    Ordered rather than as-completed so the results CSV is deterministic: a
    report that reorders when re-run invites the reader to wonder what else
    changed.
    """
    if jobs <= 1:
        _init_worker(experiment, opts)
        for row in rows:
            status, records = _score_one(row)
            yield row, status, records
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # 'spawn' rather than fork: the parent has torch loaded, and forking a
    # process with an initialized torch/OpenMP runtime into one that then runs
    # native inference is the same class of hazard as the rawpy ordering bug.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_init_worker,
        initargs=(experiment, opts),
    ) as pool:
        # chunksize=1: frames differ by an order of magnitude in cost (a
        # declined estimate is seconds, a full ECC refine is not), so batching
        # would leave workers idle at the tail.
        for row, (status, records) in zip(rows, pool.map(_score_one, rows, chunksize=1)):
            yield row, status, records
