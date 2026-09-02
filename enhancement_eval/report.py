"""Turn per-image records into an answer somebody can act on.

The reporting is opinionated in three ways, each of which is a direct response
to how the slate detector's acceptance gate went wrong:

* **Per-image before per-corpus.** The gate that failed looked fine on the
  aggregate. Every section here leads with a distribution and a win/loss count,
  and the corpus median is reported with a bootstrap interval so a 30-image
  pilot cannot be quoted as a result.
* **Split by dive and by site.** The failure was dive-shaped and site-shaped --
  pool dives produced the false fits. A corpus number that is not also broken
  out by pool-vs-field is the exact artefact that shipped.
* **Coverage separate from accuracy, always.** An arm that abstains on the
  hardest frames and is marginally better on the rest reports a better median
  error while being strictly worse. Only the discordant coverage counts show it.

No perceptual quality metric appears anywhere in this module, and that is
deliberate. UIQM, PSNR and SSIM going up is not evidence that the pipeline got
better; the `image_quality` numbers exist only to stratify inputs, and are
labelled as covariates rather than results.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from enhancement_eval.metrics import (
    bootstrap_median_ci,
    coverage_flips,
    paired_delta,
    quantiles,
    spearman,
    stratify_by_quartile,
)
from enhancement_eval.sites import Site, classify_site

__all__ = ["render_report"]

#: The stratification axis. Attenuation is the mechanism enhancement claims to
#: undo, and it acts on contrast and on the red channel, so a frame's baseline
#: RMS contrast is the most direct available proxy for "how degraded is this
#: input" without assuming the answer.
QUALITY_KEY = "rms_contrast"


def _fmt(value: float | None, spec: str = "+8.2f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'-':>8s}"
    return format(value, spec)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def _pair(rows: Sequence[dict], arm: str, metric: str) -> tuple[list, list]:
    """Baseline and arm values for `metric`, aligned image-for-image.

    Alignment is by image id rather than by position: a sweep that skipped an
    image on one arm (a failed decode, a missing raw) would otherwise shift
    every subsequent pair by one and silently compare unrelated frames.
    """
    by_image: dict[Any, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_image[row["image_id"]][row["arm"]] = row

    baseline_values, arm_values = [], []
    for image_id in sorted(by_image):
        arms = by_image[image_id]
        if "baseline" not in arms or arm not in arms:
            continue
        baseline_values.append(arms["baseline"].get(metric))
        arm_values.append(arms[arm].get(metric))
    return baseline_values, arm_values


def _report_metric(rows: Sequence[dict], arm: str, metric: str, unit: str) -> None:
    baseline, enhanced = _pair(rows, arm, metric)
    delta = paired_delta(baseline, enhanced)
    if delta.n == 0:
        print(f"  {metric:22s} no paired observations")
        return

    lo, hi = bootstrap_median_ci(list(delta.values))
    print(f"\n  --- {metric} ({unit}) ---")
    print(
        f"    baseline p50 {_fmt(quantiles([v for v in baseline if v is not None])[50])}"
        f"   arm p50 {_fmt(quantiles([v for v in enhanced if v is not None])[50])}"
    )
    print(
        f"    paired delta   median {_fmt(delta.median)}  "
        f"[95% CI {_fmt(lo)}, {_fmt(hi)}]   p10 {_fmt(delta.p10)}  p90 {_fmt(delta.p90)}"
    )
    print(
        f"    per image      better {delta.improved:4d}   worse {delta.worsened:4d}   "
        f"same {delta.unchanged:4d}   (n={delta.n})"
    )
    if delta.n_baseline_only or delta.n_enhanced_only:
        print(
            f"    unpaired       baseline-only {delta.n_baseline_only}   "
            f"arm-only {delta.n_enhanced_only}   (excluded; see coverage)"
        )
    # The interval is the guard against over-reading. Say so in words rather
    # than leaving a reader to compare two numbers and a sign.
    if not (math.isnan(lo) or math.isnan(hi)) and lo <= 0 <= hi:
        print(
            "    >> the 95% interval spans zero: this sample does not support a "
            "directional claim"
        )


def _report_coverage(rows: Sequence[dict], arm: str) -> None:
    baseline, enhanced = _pair(rows, arm, "laser_detected")
    pairs = [
        (bool(b), bool(e))
        for b, e in zip(baseline, enhanced)
        if b is not None and e is not None
    ]
    if not pairs:
        return
    flips = coverage_flips([b for b, _ in pairs], [e for _, e in pairs])
    print("\n  --- laser detector coverage ---")
    print(
        f"    both {flips.both:4d}   neither {flips.neither:4d}   "
        f"gained {flips.gained:4d}   lost {flips.lost:4d}   "
        f"net {flips.net:+d}   McNemar p={flips.p_value:.4f}"
    )
    if flips.gained or flips.lost:
        print(
            "    (gains and losses are reported separately on purpose: a net of "
            "zero over\n     many discordant pairs means the arm is trading "
            "frames, not improving)"
        )


def _report_tripwire(rows: Sequence[dict], arm: str) -> None:
    """`classify_laser_color` before and after. Expected to collapse."""
    from enhancement_eval.scoring import laser_color_agreement

    baseline, enhanced = _pair(rows, arm, "laser_color")
    outcomes = Counter(
        laser_color_agreement(b, e) for b, e in zip(baseline, enhanced)
    )
    total = sum(outcomes.values())
    if not total:
        return
    print("\n  --- tripwire: classify_laser_color ---")
    for outcome in ("agree", "flipped", "abstained", "gained", "both_abstained"):
        count = outcomes.get(outcome, 0)
        if count:
            print(f"    {outcome:16s} {count:5d}  {_pct(count, total)}")
    flipped = outcomes.get("flipped", 0)
    if flipped:
        print(
            f"    >> {flipped} frames changed laser colour. The discriminator "
            "reads R vs G at the\n       dot and every underwater enhancement "
            "model exists to restore red, so this\n       is the expected "
            "result -- and the reason enhancement belongs at the JPEG\n"
            "       stage, downstream of where this samples."
        )


def _report_stratified(rows: Sequence[dict], arm: str, metric: str) -> None:
    """The claim worth testing: enhancement helps bad frames, hurts good ones."""
    by_image: dict[Any, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_image[row["image_id"]][row["arm"]] = row

    paired_rows = []
    for image_id, arms in by_image.items():
        if "baseline" not in arms or arm not in arms:
            continue
        base_value, arm_value = arms["baseline"].get(metric), arms[arm].get(metric)
        if base_value is None or arm_value is None:
            continue
        paired_rows.append(
            {
                QUALITY_KEY: arms["baseline"].get(QUALITY_KEY),
                "delta": float(arm_value) - float(base_value),
            }
        )

    buckets = stratify_by_quartile(paired_rows, key=QUALITY_KEY)
    if not any(buckets.values()):
        return
    print(f"\n  --- {metric} delta by baseline contrast quartile ---")
    print(f"    {'bucket':14s} {'n':>5s} {'median delta':>14s} {'better':>8s}")
    for label, bucket in buckets.items():
        if not bucket:
            continue
        deltas = [r["delta"] for r in bucket]
        better = sum(1 for d in deltas if d < 0)
        print(
            f"    {label:14s} {len(bucket):5d} {_fmt(quantiles(deltas)[50], '+14.2f')} "
            f"{_pct(better, len(deltas))}"
        )
    print(
        "    (Q1 is the lowest-contrast, most degraded input. The literature's "
        "claim is that\n     enhancement helps Q1 and hurts Q4; a corpus mean "
        "over both describes neither.)"
    )


def _report_by_group(rows: Sequence[dict], arm: str, metric: str, overrides) -> None:
    """Break the headline number out by dive and by site."""
    by_image: dict[Any, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_image[row["image_id"]][row["arm"]] = row

    per_dive: dict[tuple, list[float]] = defaultdict(list)
    per_site: dict[Site, list[float]] = defaultdict(list)
    site_confidence: Counter = Counter()

    for arms in by_image.values():
        if "baseline" not in arms or arm not in arms:
            continue
        base_value, arm_value = arms["baseline"].get(metric), arms[arm].get(metric)
        if base_value is None or arm_value is None:
            continue
        delta = float(arm_value) - float(base_value)
        row = arms["baseline"]
        key = (row.get("dive_id"), row.get("dive_name") or "")
        per_dive[key].append(delta)
        site = classify_site(
            row.get("dive_id"),
            name=row.get("dive_name"),
            path=row.get("dive_path"),
            overrides=overrides,
        )
        per_site[site.site].append(delta)
        site_confidence[site.confident] += 1

    if per_site:
        print(f"\n  --- {metric} delta by site ---")
        print(f"    {'site':10s} {'n':>5s} {'median delta':>14s} {'better':>8s}")
        for site in (Site.POOL, Site.FIELD, Site.UNKNOWN):
            deltas = per_site.get(site, [])
            if not deltas:
                continue
            better = sum(1 for d in deltas if d < 0)
            print(
                f"    {site.value:10s} {len(deltas):5d} "
                f"{_fmt(quantiles(deltas)[50], '+14.2f')} {_pct(better, len(deltas))}"
            )
        documented = site_confidence.get(True, 0)
        print(
            f"    ({documented} of {sum(site_confidence.values())} classifications "
            "are documented;\n     the rest are inferred from dive name and path "
            "-- there is no site column on Dive)"
        )

    if per_dive:
        print(f"\n  --- {metric} delta by dive ---")
        print(f"    {'dive':34s} {'n':>5s} {'median delta':>14s} {'better':>8s}")
        for (dive_id, name), deltas in sorted(
            per_dive.items(), key=lambda kv: (kv[0][0] is None, kv[0][0])
        ):
            better = sum(1 for d in deltas if d < 0)
            label = f"{dive_id} {name}"[:34]
            print(
                f"    {label:34s} {len(deltas):5d} "
                f"{_fmt(quantiles(deltas)[50], '+14.2f')} {_pct(better, len(deltas))}"
            )


def _report_slate_gate(rows: Sequence[dict], arm: str, overrides) -> None:
    """Does the ECC gate track correctness? The question that retired the stage.

    The slate detector shipped behind `ECC >= 0.80` and died the next day
    because pool dives produced high-ECC *false* fits. That gate used a
    confidence score as a correctness proxy without anyone having measured
    whether it was one. `DiveSlateLabel.reference_points` makes it measurable,
    so this section reports it directly -- per site, because per site is where
    the failure lived and where the corpus aggregate hid it.

    The precedent is `LaserDepth.residual_m`, which looked like a quality
    signal, measured rho = -0.026 against actual error, and is now recorded and
    deliberately never gated on. If ECC behaves the same way, no threshold on it
    could ever have worked -- which is a more useful conclusion than "the
    threshold was wrong", because it rules out re-tuning as a fix.
    """
    by_image: dict[Any, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_image[row["image_id"]][row["arm"]] = row

    for label in ("baseline", arm):
        scored = [
            arms[label]
            for arms in by_image.values()
            if label in arms
            and arms[label].get("slate_score_status") == "scored"
            and arms[label].get("slate_ecc") is not None
        ]
        if not scored:
            continue

        eccs = [float(r["slate_ecc"]) for r in scored]
        errors = [float(r["slate_median_px"]) for r in scored]
        rho = spearman(eccs, errors)

        print(f"\n  --- slate: does ECC predict correctness? [{label}] ---")
        print(f"    n={len(scored)}   Spearman rho(ECC, median point error) = {rho:+.3f}")
        if not math.isnan(rho) and abs(rho) < 0.2:
            print(
                "    >> ECC does not track correctness on this sample. A gate on it\n"
                "       cannot be fixed by re-tuning the threshold -- the same shape\n"
                "       of result that retired LaserDepth.residual_m as a quality gate."
            )

        passed = [r for r in scored if r.get("slate_gate_passed")]
        if passed:
            q = quantiles([float(r["slate_median_px"]) for r in passed], (50, 90, 95))
            print(
                f"    among the {len(passed)} fits the ECC>=0.80 gate ACCEPTED: "
                f"error p50 {q[50]:.1f} px  p90 {q[90]:.1f}  p95 {q[95]:.1f}"
            )

        per_site: dict[Site, list[tuple[float, float, bool]]] = defaultdict(list)
        for record in scored:
            site = classify_site(
                record.get("dive_id"),
                name=record.get("dive_name"),
                path=record.get("dive_path"),
                overrides=overrides,
            )
            per_site[site.site].append(
                (
                    float(record["slate_ecc"]),
                    float(record["slate_median_px"]),
                    bool(record.get("slate_gate_passed")),
                )
            )

        print(
            f"    {'site':10s} {'n':>5s} {'accepted':>9s} {'err p50':>9s} "
            f"{'err p90':>9s} {'rho':>7s}"
        )
        for site in (Site.POOL, Site.FIELD, Site.UNKNOWN):
            entries = per_site.get(site, [])
            if not entries:
                continue
            site_errors = [e for _, e, _ in entries]
            accepted = [e for _, e, ok in entries if ok]
            q = quantiles(site_errors, (50, 90))
            site_rho = spearman([c for c, _, _ in entries], site_errors)
            print(
                f"    {site.value:10s} {len(entries):5d} "
                f"{_pct(len(accepted), len(entries)):>9s} "
                f"{q[50]:9.1f} {q[90]:9.1f} {site_rho:+7.3f}"
            )
        # The specific failure: accepted on pool, and wrong.
        pool_accepted = [e for _, e, ok in per_site.get(Site.POOL, []) if ok]
        if pool_accepted:
            pq = quantiles(pool_accepted, (50, 90))
            print(
                f"    pool fits the gate ACCEPTED: n={len(pool_accepted)}  "
                f"error p50 {pq[50]:.1f} px  p90 {pq[90]:.1f} px"
            )


def _report_slate(rows: Sequence[dict], arm: str, overrides) -> None:
    """Coverage, accuracy, and the gate analysis, for the slate estimator."""
    baseline, enhanced = _pair(rows, arm, "slate_gate_passed")
    pairs = [
        (bool(b), bool(e))
        for b, e in zip(baseline, enhanced)
        if b is not None and e is not None
    ]
    if pairs:
        flips = coverage_flips([b for b, _ in pairs], [e for _, e in pairs])
        print("\n  --- slate estimator: gate acceptance ---")
        print(
            f"    both {flips.both:4d}   neither {flips.neither:4d}   "
            f"gained {flips.gained:4d}   lost {flips.lost:4d}   "
            f"net {flips.net:+d}   McNemar p={flips.p_value:.4f}"
        )
        print(
            "    (acceptance is not correctness -- see the ECC section below "
            "before reading\n     a coverage gain as an improvement)"
        )

    _report_metric(rows, arm, "slate_median_px", "pixels, lower is better")
    _report_stratified(rows, arm, "slate_median_px")
    _report_by_group(rows, arm, "slate_median_px", overrides)
    _report_slate_gate(rows, arm, overrides)


def render_report(
    rows: Sequence[dict],
    *,
    overrides: dict[int, Site] | None = None,
) -> None:
    """Print the full comparison for every non-baseline arm in `rows`."""
    arms = sorted({r["arm"] for r in rows} - {"baseline"})
    images = len({r["image_id"] for r in rows})
    stages = sorted({r.get("stage") for r in rows})

    print(f"\n{'=' * 76}")
    print(f"{images} images   stage(s): {', '.join(str(s) for s in stages)}")
    print(f"{'=' * 76}")

    if "rectified" in stages:
        print(
            "\nNOTE on the rectified (JPEG) stage: `predict_laser_image` decodes the\n"
            ".ORF into a LinearRawImage and never reads the JPEG, so an enhancer\n"
            "placed here cannot move the laser detector's numbers at all. That is\n"
            "structural, not a sample-size problem -- see tests/test_decode_parity.py::\n"
            "test_gamma_and_clahe_do_not_reach_the_linear_stage. Laser results are\n"
            "reported only for the linear stage."
        )

    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] in ("baseline", arm)]
        print(f"\n{'-' * 76}\narm: {arm}\n{'-' * 76}")
        probe = next(
            (r.get("probe_displacement_px") for r in rows if r["arm"] == arm), None
        )
        if probe is not None:
            print(f"  geometry probe: {probe:.3f} px displacement (limit 0.5)")

        stage_values = {r.get("stage") for r in arm_rows}
        if "linear" in stage_values:
            _report_coverage(arm_rows, arm)
            _report_metric(arm_rows, arm, "laser_err_px", "pixels, lower is better")
            _report_stratified(arm_rows, arm, "laser_err_px")
            _report_by_group(arm_rows, arm, "laser_err_px", overrides)
            _report_tripwire(arm_rows, arm)
        if "rectified" in stage_values:
            if any(r.get("slate_status") for r in arm_rows):
                _report_slate(arm_rows, arm, overrides)
            _report_metric(arm_rows, arm, "ht_err_head_pct", "% of fish length")
            _report_metric(arm_rows, arm, "ht_len_err_pct", "% of fish length, signed")
            _report_stratified(arm_rows, arm, "ht_err_head_pct")
            _report_by_group(arm_rows, arm, "ht_err_head_pct", overrides)
