"""Command-line entry point. Three phases, deliberately separable.

    # 1. what is there to test? (the only step that touches a database)
    python -m enhancement_eval manifest --out manifest.csv --per-dive 40

    # 2. decode + infer -- the slow step, cached by checksum
    python -m enhancement_eval evaluate --manifest manifest.csv \
        --experiment wb-linear --raw-dir ./raw_cache --out results.csv

    # 3. read the answer, re-runnably, without re-decoding anything
    python -m enhancement_eval report --results results.csv

The split matters because phase 2 costs roughly 15 s per frame per arm on CPU
and phase 3 is where the thinking happens. Anyone who has to re-run a
several-hour sweep to re-read a table will stop re-reading tables.

Everything here is READ-ONLY. Phase 1 opens a read-only transaction and issues
one SELECT; phases 2 and 3 touch no database at all. Nothing writes to
fishsense-api, Label Studio, Garage, or Postgres.

Raw bytes
---------
Phase 2 needs the `.ORF`, not the JPEG -- white balance cannot be redone from a
frame that has already been through camera WB, auto-gamma, CLAHE and 8-bit
quantization. The raws live on the NAS: Garage's `raw/` prefix is scratch that
`cleanup_raw_bytes_for_dive_activity` deletes once a dive is processed, so it
holds only whatever is mid-flight.

`--raw-dir` is therefore addressed by checksum, exactly like
`validate_headtail_predictions.py`'s JPEG cache: a directory of
`{checksum}.ORF` fetched by any means works, and `evaluate` never needs
credentials of its own.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import random
from collections import defaultdict
import sys
import time
from pathlib import Path
from typing import Any

from enhancement_eval.manifest import MANIFEST_FIELDS, sample_per_dive
from enhancement_eval.parallel import run_sweep
from enhancement_eval.paths import resolve_under_root
from enhancement_eval.sites import Site


def _die(message: str) -> None:
    sys.exit(message)


# --------------------------------------------------------------------------
# phase 1: manifest
# --------------------------------------------------------------------------


def cmd_manifest(args: argparse.Namespace) -> int:
    """SELECT the oracle set and write it to CSV. Read-only."""
    from enhancement_eval.manifest import fetch_manifest

    dsn = os.environ.get("FISHSENSE_DSN")
    if not dsn:
        _die(
            "Set FISHSENSE_DSN, e.g.\n"
            "  export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'\n"
            "A restored backup is preferable to prod: this is the only step that\n"
            "touches a database, it is pure SELECT, and a local restore removes\n"
            "the network path entirely."
        )

    rows = fetch_manifest(dsn)
    total = len(rows)

    if args.dive_id:
        wanted = set(args.dive_id)
        rows = [r for r in rows if r["dive_id"] in wanted]
    if args.require_headtail:
        rows = [r for r in rows if r.get("head_x") is not None]
    if args.require_depth:
        rows = [r for r in rows if r.get("range_m") is not None]
    rows = sample_per_dive(rows, per_dive=args.per_dive)
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.checksums_out:
        Path(args.checksums_out).write_text(
            "".join(f"{r['checksum']}\n" for r in rows), encoding="utf-8"
        )
        print(f"wrote {len(rows)} checksums -> {args.checksums_out}")

    dives = sorted({r["dive_id"] for r in rows})
    print(f"{total} canonical images carry at least one usable oracle")
    print(f"wrote {len(rows)} rows across {len(dives)} dives -> {out}")
    with_laser = sum(1 for r in rows if r.get("laser_points"))
    with_ht = sum(1 for r in rows if r.get("head_x") is not None)
    with_depth = sum(1 for r in rows if r.get("range_m") is not None)
    with_slate = sum(1 for r in rows if r.get("human_slate_points"))
    slate_ready = sum(
        1
        for r in rows
        if r.get("human_slate_points")
        and r.get("slate_template_points")
        and r.get("camera_matrix")
    )
    print(f"  laser dots      {with_laser}")
    print(f"  head/tail       {with_ht}   (fish arm oracle)")
    print(f"  slate points    {with_slate}   -> {slate_ready} with a resolvable "
          f"template (slate arm oracle)")
    print(f"  LaserDepth      {with_depth}   (metric range; Deliverable 3 anchor)")
    return 0


# --------------------------------------------------------------------------
# labeler effort
# --------------------------------------------------------------------------

#: Label tables that carry Label Studio annotation payloads.
EFFORT_TABLES = {
    "laser": "laserlabel",
    "headtail": "headtaillabel",
    "species": "specieslabel",
    "slate": "diveslatelabel",
}


def cmd_effort(args: argparse.Namespace) -> int:
    """Characterize labeler effort from Label Studio's own timing record.

    Read-only. Answers, in order: how much of labeling time is the image rather
    than the labeler; whether per-image difficulty is a reliable construct at
    all; and how large a trial would have to be.
    """
    from enhancement_eval.effort import (
        EFFORT_SQL,
        image_level_reliability,
        labeler_residuals,
        required_per_arm,
        usable_annotations,
        variance_explained,
    )

    dsn = os.environ.get("FISHSENSE_DSN")
    if not dsn:
        _die("Set FISHSENSE_DSN (a restored backup is preferable to prod).")

    import psycopg

    table = EFFORT_TABLES[args.task]
    with psycopg.connect(dsn) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(EFFORT_SQL.format(table=table))
            columns = [c.name for c in cur.description]
            raw = [dict(zip(columns, r)) for r in cur.fetchall()]

    total = len(raw)
    skipped = sum(1 for r in raw if r.get("cancelled"))
    rows = usable_annotations(raw, max_seconds=args.max_seconds)
    if not rows:
        _die("no usable annotations")
    enriched = labeler_residuals(rows)

    times = sorted(float(r["lead_time"]) for r in rows)
    print(f"\n=== {args.task}: {total} annotations, {len(rows)} usable ===")
    print(f"  skipped (was_cancelled)   {skipped}  ({100*skipped/total:.1f}%)")
    print(f"  dropped as > {args.max_seconds:g}s        {total - skipped - len(rows)}")
    print(f"  labelers                  {len({r['labeler'] for r in rows})}")
    print(
        f"  lead time  p25 {times[len(times)//4]:.1f}s   median "
        f"{times[len(times)//2]:.1f}s   p90 {times[int(0.9*len(times))]:.1f}s"
    )

    print("\n  --- what explains the variation in log labeling time? ---")
    for key, label in (("labeler", "labeler identity"), ("dive_id", "dive")):
        share = variance_explained(enriched, key=key, value="y")
        print(f"    {label:20s} {100*share:5.1f}%")
    for row in enriched:
        row["labeler_dive"] = (row["labeler"], row["dive_id"])
    joint = variance_explained(enriched, key="labeler_dive", value="y")
    print(f"    {'labeler x dive':20s} {100*joint:5.1f}%")
    print(f"    {'residual':20s} {100*(1-joint):5.1f}%   <- all image-level effects live here")

    reliability = image_level_reliability(enriched)
    residuals = [r["resid"] for r in enriched]
    sd = math.sqrt(sum(v * v for v in residuals) / len(residuals))
    print("\n  --- is per-image difficulty a reliable construct? ---")
    print(f"    residual SD of log lead time      {sd:.3f}  ({math.exp(sd):.2f}x typical swing)")
    print(f"    images labelled by >=2 labelers   {reliability.n_pairs}")
    if reliability.n_pairs >= 4:
        print(
            f"    labeler-vs-labeler agreement      r = {reliability.r:+.3f}  "
            f"95% CI [{reliability.ci_low:+.3f}, {reliability.ci_high:+.3f}]"
        )
        if reliability.ci_high < 0.3:
            print(
                "    >> Per-image difficulty is weak. Predicting labeling difficulty\n"
                "       from pixels has almost nothing to explain -- do not build that\n"
                "       model. It does NOT rule out a real speedup: a uniform gain\n"
                "       shifts the mean without any image-level structure, and the\n"
                "       randomized trial below detects that."
            )
    else:
        print("    too few doubly-labelled images to estimate (need >=4)")

    print("\n  --- randomized within-labeler A/B: annotations per arm ---")
    print("      (each labeler gets a random mix of original and enhanced frames)")
    median = times[len(times) // 2]
    for pct in (5, 10, 15, 20, 30):
        n = required_per_arm(pct / 100, sd)
        minutes = 2 * n * median / 60
        print(f"    {pct:3d}% reduction   {int(n):6d} per arm   (~{minutes:.0f} min of labeling total)")
    return 0


# --------------------------------------------------------------------------
# trial: build the labeling A/B
# --------------------------------------------------------------------------


def cmd_trial(args: argparse.Namespace) -> int:
    """Select backlog frames, allocate them to arms, and render the JPEGs.

    Writes local files only. Nothing here touches Label Studio, Garage or the
    API -- loading the trial is a deliberate, separate, human step.
    """
    from enhancement_eval.recommended import PRODUCTION, RECOMMENDED, RECOMMENDED_WITH_DOT
    from enhancement_eval.trial import BACKLOG_SQL, assign_arms, summarize_allocation

    dsn = os.environ.get("FISHSENSE_DSN")
    if not dsn:
        _die("Set FISHSENSE_DSN (a restored backup is preferable to prod).")

    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(BACKLOG_SQL)
            cols = [c.name for c in cur.description]
            backlog = [dict(zip(cols, r)) for r in cur.fetchall()]
    print(f"{len(backlog)} frames in the species backlog")

    raw_root = Path(args.raw_root).expanduser() if args.raw_root else None
    if raw_root is None:
        _die("--raw-root is required to render the trial JPEGs")

    per_dive = defaultdict(list)
    for row in backlog:
        per_dive[row["dive_id"]].append(row)

    n_arms = len([a for a in args.arms.split(",") if a])
    wanted = args.per_arm * n_arms
    # Draw several frames from each dive rather than one frame from very many.
    #
    # With one frame per dive, every dive lands in exactly one arm: within-dive
    # blocking does nothing, and -- worse -- the stratified question this
    # project most wants to ask ("did it help more on the murkier dives?")
    # becomes unanswerable, because no dive is represented in both arms.
    # `--per-dive` frames from `wanted // per_dive` dives keeps coverage broad
    # while making each dive its own little balanced experiment.
    per_dive_target = max(n_arms, args.per_dive)

    # Availability is checked per candidate as it is drawn, never over the
    # whole backlog: the mount is slow enough that stat-ing 27k paths dominates
    # the run, and only a few hundred are ever needed.
    chosen: list[dict] = []
    checked = 0
    dives = sorted(per_dive)
    rng_dives = list(dives)
    random.Random(args.seed).shuffle(rng_dives)
    for dive in rng_dives:
        if len(chosen) >= wanted:
            break
        taken = 0
        for candidate in per_dive[dive]:
            if taken >= per_dive_target or len(chosen) >= wanted:
                break
            checked += 1
            if resolve_under_root(raw_root, candidate["image_path"]):
                chosen.append(candidate)
                taken += 1
    print(f"drew {len(chosen)} frames from "
          f"{len({r['dive_id'] for r in chosen})} dives "
          f"({checked} candidates checked for a readable .ORF)")

    ARMS = {"control": PRODUCTION, "recommended": RECOMMENDED,
            "recommended-dot": RECOMMENDED_WITH_DOT}
    arm_names = tuple(a for a in args.arms.split(",") if a in ARMS)
    if len(arm_names) < 2:
        _die(f"--arms must name at least two of: {', '.join(ARMS)}")

    allocation = assign_arms(chosen, arms=arm_names, seed=args.seed)
    print()
    print(summarize_allocation(allocation))

    out = Path(args.out)
    (out / "jpeg").mkdir(parents=True, exist_ok=True)
    with (out / "allocation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_id", "dive_id", "dive_name", "checksum", "arm",
                        "decode", "jpeg", "image_path"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in allocation:
            writer.writerow(dict(row, decode=ARMS[row["arm"]].label,
                                 jpeg=f"jpeg/{row['checksum']}.JPG"))
    print(f"\nwrote allocation -> {out / 'allocation.csv'}")

    if args.no_render:
        print("--no-render: skipping JPEG rendering")
        return 0

    from enhancement_eval.trial_render import render_allocation

    render_allocation(allocation, ARMS, raw_root, out / "jpeg", jobs=args.jobs)
    return 0


# --------------------------------------------------------------------------
# phase 2: evaluate
# --------------------------------------------------------------------------


def _literal(value: Any) -> Any:
    """CSV round-trips JSON columns as their Python repr; restore them."""
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return value or None


def _load_raw(row: dict, raw_dir: Path | None, raw_root: Path | None) -> bytes | None:
    """The frame's raw bytes, from the checksum cache or the NAS share.

    Cache first: it is content-addressed, so it needs no mount and cannot be
    wrong about which frame it holds. The share is the fallback, resolving
    `Image.path` under the configured root.
    """
    checksum = row.get("checksum")
    if raw_dir is not None and checksum:
        for suffix in (".ORF", ".orf"):
            candidate = raw_dir / f"{checksum}{suffix}"
            if candidate.exists():
                return candidate.read_bytes()

    found = resolve_under_root(raw_root, row.get("image_path"))
    return found.read_bytes() if found is not None else None


def cmd_evaluate(args: argparse.Namespace) -> int:
    from enhancement_eval.evaluate import (
        HeadTailConsumer,
        LaserConsumer,
        SlateConsumer,
        Stage,
        evaluate_image,
    )
    from enhancement_eval.experiments import build_arms

    from enhancement_eval.experiments import EXPERIMENTS

    experiment = EXPERIMENTS.get(args.experiment)
    arms = build_arms(args.experiment)

    if experiment is not None:
        print(f"\nexperiment: {args.experiment}")
        print(f"  reaches: {', '.join(experiment.consumers)}")
        if experiment.note:
            print(f"  {experiment.note}")
        if experiment.diagnostic_only:
            print(
                "\n  *** DIAGNOSTIC ONLY -- NOT A SHIPPING CANDIDATE ***\n"
                "  These arms place enhancement upstream of the JPEG, where the\n"
                "  laser detector and classify_laser_color read. Shipping this\n"
                "  would make laser detection depend on enhancement. The laser\n"
                "  supplies depth structure to this project: it is an input, not\n"
                "  a consumer. Results here measure the cost of the constraint;\n"
                "  they are not a proposal to relax it.\n"
            )
    print()
    for arm in arms:
        flag = "  SUSPICIOUS" if arm.probe and arm.probe.suspicious else ""
        print(
            f"  arm {arm.label:20s} stage={arm.stage.value:9s} "
            f"decode={arm.decode.label:22s} probe={arm.probe.displacement_px:.3f}px{flag}"
        )

    rows = list(csv.DictReader(Path(args.manifest).open(newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        _die(f"{args.manifest} has no rows")

    stages = {arm.stage for arm in arms}
    laser = LaserConsumer(args.checkpoint) if Stage.LINEAR in stages else None
    headtail = (
        HeadTailConsumer()
        if Stage.RECTIFIED in stages and not args.no_fish
        else None
    )
    slate_dir = Path(args.slate_pdf_dir) if args.slate_pdf_dir else None
    slate_root = Path(args.slate_root).expanduser() if args.slate_root else None
    slate = (
        SlateConsumer(use_board_mask=args.board_mask)
        if Stage.RECTIFIED in stages and (slate_dir or slate_root)
        else None
    )
    templates: dict[int, Any] = {}

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    raw_root = Path(args.raw_root).expanduser() if args.raw_root else None
    if raw_dir is None and raw_root is None:
        _die(
            "Phase 2 needs the .ORF originals: white balance cannot be redone from\n"
            "an already-processed JPEG, and the Garage raw/ prefix is scratch that\n"
            "is deleted once a dive is processed. Supply either:\n"
            "  --raw-root ~/mnt/fishsense_data/REEF/data   (resolves Image.path)\n"
            "  --raw-dir  ./raw_cache                      ({checksum}.ORF files)"
        )
    if raw_root is not None and not raw_root.is_dir():
        _die(f"--raw-root {raw_root} is not a directory")

    out = Path(args.out)
    written = missing = failed = 0
    started = time.monotonic()

    opts = {
        "checkpoint": args.checkpoint,
        "fish": not args.no_fish,
        "slate": bool(slate_dir or slate_root),
        "board_mask": args.board_mask,
        "slate_dir": str(slate_dir) if slate_dir else None,
        "slate_root": str(slate_root) if slate_root else None,
        "raw_dir_path": raw_dir,
        "raw_root_path": raw_root,
    }

    parsed_rows = [
        {k: _literal(v) if k in _JSON_COLUMNS else v for k, v in row.items()}
        for row in rows
    ]
    jobs = args.jobs if args.jobs > 0 else min(8, (os.cpu_count() or 1))
    print(f"running {len(parsed_rows)} frames x {len(arms)} arms on {jobs} process(es)")

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter | None = None
        for index, (row, status, records) in enumerate(
            run_sweep(parsed_rows, args.experiment, opts, jobs), 1
        ):
            if status == "missing_raw":
                missing += 1
            elif status != "ok":
                failed += 1
                print(f"  image {row.get('image_id')}: {status}")

            for record in records:
                if writer is None:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(record), extrasaction="ignore"
                    )
                    writer.writeheader()
                writer.writerow(record)
            if records:
                written += 1

            if index % args.progress_every == 0:
                handle.flush()
                rate = (time.monotonic() - started) / index
                print(
                    f"  {index}/{len(parsed_rows)}  {rate:.1f}s/frame  "
                    f"eta {rate * (len(parsed_rows) - index) / 60:.0f} min",
                    flush=True,
                )

    print(f"\nscored {written} images -> {out}")
    if missing:
        print(f"  {missing} skipped: no .ORF in {raw_dir}")
    if failed:
        print(f"  {failed} failed during evaluation (listed above)")
    return 0


_JSON_COLUMNS = {
    "camera_matrix",
    "distortion_coefficients",
    "slate_rectangle",
    "human_slate_points",
    "slate_template_points",
}


def _slate_template(row: dict, slate_dir, slate_root, cache: dict):
    """Render (and cache) the dive's slate template.

    Cached by slate id, not by image: one template serves every frame of every
    dive using that slate, and a PDF render at full dpi is not cheap.
    """
    from enhancement_eval.evaluate import SlateTemplate

    slate_id = row.get("dive_slate_id")
    if not slate_id or not row.get("slate_template_points"):
        return None
    slate_id = int(float(slate_id))
    if slate_id in cache:
        return cache[slate_id]

    pdf = None
    if slate_dir is not None and (slate_dir / f"{slate_id}.pdf").exists():
        pdf = slate_dir / f"{slate_id}.pdf"
    else:
        pdf = resolve_under_root(slate_root, row.get("slate_path"))
    if pdf is None:
        cache[slate_id] = None
        print(
            f"  no slate PDF for slate {slate_id} "
            f"({row.get('slate_path') or 'no path'}); slate scoring skipped"
        )
        return None

    cache[slate_id] = SlateTemplate.from_pdf(
        slate_id,
        str(row.get("slate_name") or ""),
        int(float(row.get("slate_dpi") or 300)),
        pdf.read_bytes(),
        row["slate_template_points"],
    )
    return cache[slate_id]


# --------------------------------------------------------------------------
# phase 3: report
# --------------------------------------------------------------------------


#: Columns that stay text, and columns that must stay integral. Ids in
#: particular: read back as floats they render as "65.0" in the per-dive table
#: and, worse, stop comparing equal to the integer keys in `KNOWN_POOL_DIVE_IDS`,
#: which would silently drop every documented pool dive back to "inferred".
_TEXT_COLUMNS = frozenset(
    {"arm", "stage", "decode", "checksum", "dive_name", "dive_path", "laser_color",
     "ht_status"}
)
_INT_COLUMNS = frozenset({"image_id", "dive_id", "n_human_lasers", "ht_n_instances"})

#: Columns written as Python bools. CSV has no types, so they come back as the
#: strings "True"/"False" -- and `bool("False")` is **True**. Left unhandled,
#: every coverage and gate-acceptance figure reads 100%, McNemar sees zero
#: discordant pairs, and the report says "no coverage change" no matter what
#: actually happened. Parsed explicitly, and pinned by a regression test.
_BOOL_COLUMNS = frozenset({"laser_detected", "slate_gate_passed"})
_TRUTHY = frozenset({"true", "1", "yes"})


def _numeric(row: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in ("", None):
            out[key] = None
        elif key in _BOOL_COLUMNS:
            out[key] = (
                value if isinstance(value, bool) else str(value).strip().lower() in _TRUTHY
            )
        elif key in _TEXT_COLUMNS:
            out[key] = value
        elif key in _INT_COLUMNS:
            try:
                out[key] = int(float(value))
            except (TypeError, ValueError):
                out[key] = value
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = value
    return out


def cmd_report(args: argparse.Namespace) -> int:
    from enhancement_eval.report import render_report

    rows = [
        _numeric(r)
        for r in csv.DictReader(Path(args.results).open(newline="", encoding="utf-8"))
    ]
    if not rows:
        _die("no rows")

    overrides = {}
    for spec in args.pool_dive or []:
        overrides[int(spec)] = Site.POOL
    for spec in args.field_dive or []:
        overrides[int(spec)] = Site.FIELD

    render_report(rows, overrides=overrides)
    return 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="enhancement_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="SELECT the human-labelled oracle set")
    m.add_argument("--out", default="manifest.csv")
    m.add_argument("--dive-id", type=int, action="append")
    m.add_argument("--per-dive", type=int, default=0, help="cap rows per dive")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--require-headtail", action="store_true")
    m.add_argument("--require-depth", action="store_true")
    m.add_argument("--checksums-out", help="bare checksum list, for fetching raws")
    m.set_defaults(func=cmd_manifest)

    e = sub.add_parser("evaluate", help="decode both arms and score every consumer")
    e.add_argument("--manifest", default="manifest.csv")
    e.add_argument("--experiment", required=True)
    e.add_argument(
        "--raw-dir",
        default=None,
        help="checksum-addressed cache of {checksum}.ORF; takes precedence",
    )
    e.add_argument(
        "--raw-root",
        default=None,
        help="NAS share root that Image.path is relative to, e.g. "
        "~/mnt/fishsense_data/REEF/data",
    )
    e.add_argument("--out", default="results.csv")
    e.add_argument("--checkpoint", default=None, help="laser detector .pt")
    e.add_argument(
        "--slate-pdf-dir",
        default=None,
        help="directory of {slate_id}.pdf templates; enables slate scoring",
    )
    e.add_argument(
        "--slate-root",
        default=None,
        help="NAS share root that DiveSlate.path is relative to (usually the "
        "same as --raw-root); enables slate scoring",
    )
    e.add_argument(
        "--board-mask",
        action="store_true",
        help="use the learned BoardMasker (off by default: it is a second "
        "learned model and would confound the variable under test)",
    )
    e.add_argument(
        "--no-fish", action="store_true", help="skip segmentation / head-tail"
    )
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--progress-every", type=int, default=10)
    e.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="worker processes (default: min(8, cpu count)). The sweep is "
        "CPU-bound and single-threaded per frame -- equalize_adapthist alone is "
        "73%% of a decode -- so this is the difference between a 13-hour run "
        "and a 2-hour one.",
    )
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser(
        "effort", help="characterize labeler effort from Label Studio timings"
    )
    f.add_argument("--task", choices=sorted(EFFORT_TABLES), default="laser")
    f.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="discard longer annotations as abandoned tabs (default 300; the "
        "corpus max is 868537, i.e. ten days)",
    )
    f.set_defaults(func=cmd_effort)

    tr = sub.add_parser(
        "trial", help="build the labeling A/B: select, allocate, render (local files only)"
    )
    tr.add_argument("--out", default="trial")
    tr.add_argument("--raw-root", default=None, help="NAS root Image.path is relative to")
    tr.add_argument("--per-arm", type=int, default=127,
                    help="frames per arm (127 detects a 20%% effect on species work)")
    tr.add_argument("--arms", default="control,recommended")
    tr.add_argument("--per-dive", type=int, default=4,
                    help="frames drawn per dive (default 4). One per dive would "
                         "leave every dive in a single arm, disabling both the "
                         "blocking and the per-dive stratified comparison.")
    tr.add_argument("--seed", type=int, default=20260902)
    tr.add_argument("--jobs", type=int, default=6)
    tr.add_argument("--no-render", action="store_true")
    tr.set_defaults(func=cmd_trial)

    rec = sub.add_parser(
        "recommend", help="print the recommended configurations and what settles them"
    )
    rec.set_defaults(func=lambda _a: (print(__import__(
        "enhancement_eval.recommended", fromlist=["describe"]).describe()), 0)[1])

    r = sub.add_parser("report", help="summarize a results CSV")
    r.add_argument("--results", default="results.csv")
    r.add_argument("--pool-dive", action="append", help="force a dive to pool")
    r.add_argument("--field-dive", action="append", help="force a dive to field")
    r.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
