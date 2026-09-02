"""Labeler effort: the oracle for "does this make labelers' lives easier?".

Label Studio stamps every annotation with `lead_time` (seconds spent),
`was_cancelled` (skipped) and `completed_by`, and all of it is already in
`*.label_studio_json`. Coverage on this corpus is 100%: 47,501 laser and
39,556 head/tail annotations, every one with a lead time, across 88 and 66
labelers respectively.

That matters more than any pixel statistic, for one reason: **a real A/B trial
needs no new instrumentation.** Re-populate a project with enhanced JPEGs for
a randomized subset of frames, let labeling proceed normally, and the outcome
variable records itself.

What this module computes, in the order the questions have to be asked:

1. **Variance decomposition.** How much of the variation in labeling time is
   the image, versus who happened to label it? On the laser corpus: labeler
   identity 40.3%, dive 9.5%, labeler x dive 49.7%.
2. **Image-level reliability.** Do two labelers agree about which image was
   slow? This is the question the whole direction depends on, and on the laser
   corpus the answer is *barely*: r = +0.065 over 755 doubly-labelled images,
   95% CI [-0.006, +0.136].
3. **Power.** Given that noise, how big does a trial have to be?

Read (2) carefully, because it cuts one way and not the other:

* It **rules out** predicting per-image labeling difficulty from pixels. Only
  ~6% of labeler-adjusted variance is a stable property of the image, so even a
  perfect image-quality model has almost nothing to explain. Do not build that.
* It does **not** rule out a real improvement. A uniform speedup shifts the
  mean without needing any image-level variance structure at all, and a
  randomized trial detects that fine -- 94 annotations per arm for a 20%
  effect, which is about forty minutes of labeling.

The cheap, decisive experiment is therefore the randomized trial, not an
observational model. That conclusion is the deliverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "LEAD_TIME_MAX_S",
    "EFFORT_SQL",
    "usable_annotations",
    "variance_explained",
    "Reliability",
    "image_level_reliability",
    "required_per_arm",
    "labeler_residuals",
]

#: Annotations longer than this are discarded as abandoned tabs rather than
#: labeling. Chosen from the observed distribution, not guessed: laser lead
#: times run p50 13.0 s, p90 33.0 s, p95 54.2 s, p99 406 s, max 868,537 s
#: (ten days). The cut at 300 s removes 1.2% of annotations and everything
#: above it is implausible as continuous attention. `usable_annotations` takes
#: an override so the sensitivity of any result to this choice can be checked.
LEAD_TIME_MAX_S = 300.0

#: One row per annotation. `label_studio_json` holds an array of them, so the
#: lateral join is what turns "one row per label" into "one row per act of
#: labeling" -- which is the unit effort is actually measured in.
EFFORT_SQL = """
SELECT l.image_id,
       i.dive_id,
       COALESCE(d.name, '')            AS dive_name,
       COALESCE(d.path, '')            AS dive_path,
       (ann->>'lead_time')::float      AS lead_time,
       (ann->>'completed_by')          AS labeler,
       (ann->>'was_cancelled')::bool   AS cancelled,
       (ann->>'created_at')            AS annotated_at
FROM {table} l
JOIN image i ON i.id = l.image_id
JOIN dive d  ON d.id = i.dive_id
CROSS JOIN LATERAL jsonb_array_elements((l.label_studio_json::jsonb)->'annotations') ann
WHERE l.label_studio_json IS NOT NULL
  AND i.is_canonical
"""


def usable_annotations(
    rows: Iterable[Mapping[str, Any]], *, max_seconds: float = LEAD_TIME_MAX_S
) -> list[Mapping[str, Any]]:
    """Annotations that represent an act of labeling.

    Two exclusions, both deliberate:

    * **Cancelled.** A skip records the time spent deciding not to label, which
      is a different quantity. Skips are their own outcome (7.4% of head/tail
      annotations), not slow labels.
    * **Implausibly long.** See `LEAD_TIME_MAX_S`.
    """
    kept = []
    for row in rows:
        if row.get("cancelled"):
            continue
        value = row.get("lead_time")
        if value in (None, ""):
            continue
        seconds = float(value)
        if 0 < seconds <= max_seconds:
            kept.append(row)
    return kept


def variance_explained(
    rows: Sequence[Mapping[str, Any]], *, key: str, value: str
) -> float:
    """Fraction of variance in `value` lying between levels of `key`.

    A one-way decomposition, not a model: it answers "how much of this spread
    is just *which* group" for one grouping at a time. NaN when `value` does
    not vary, because zero would claim a measurement that was not made.
    """
    values = [float(row[value]) for row in rows]
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    total = sum((v - mean) ** 2 for v in values)
    if total <= 0:
        return float("nan")

    groups: dict[Any, list[float]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(float(row[value]))
    between = sum(
        len(g) * ((sum(g) / len(g)) - mean) ** 2 for g in groups.values()
    )
    return between / total


def labeler_residuals(
    rows: Sequence[Mapping[str, Any]],
    *,
    labeler_key: str = "labeler",
    value_key: str = "lead_time",
) -> list[dict[str, Any]]:
    """Attach `y` (log lead time) and `resid` (log time minus that labeler's mean).

    Log because labeling time is multiplicative and right-skewed -- a slow
    labeler is slow by a *factor*, and differences in log space are the ratios
    a report should quote.

    Removing each labeler's own mean is what makes times comparable at all:
    labeler identity explains 40.3% of the raw variance on this corpus, with a
    6x spread between the fastest and slowest.
    """
    enriched = [dict(row) for row in rows]
    for row in enriched:
        row["y"] = math.log(float(row[value_key]))

    sums: dict[Any, list[float]] = {}
    for row in enriched:
        sums.setdefault(row[labeler_key], []).append(row["y"])
    means = {k: sum(v) / len(v) for k, v in sums.items()}
    for row in enriched:
        row["resid"] = row["y"] - means[row[labeler_key]]
    return enriched


@dataclass(frozen=True)
class Reliability:
    """How much of labeler-adjusted time is a stable property of the image."""

    r: float
    ci_low: float
    ci_high: float
    n_pairs: int


def image_level_reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_key: str = "image_id",
    labeler_key: str = "labeler",
    resid_key: str = "resid",
    seed: int = 0,
) -> Reliability:
    """Correlate two *different* labelers' adjusted times on the same image.

    This is an intraclass-correlation estimate: the share of labeler-adjusted
    variance that is a stable property of the frame. It is the number that
    decides whether "this image is hard" is a measurable thing at all.

    Two labelers, never two annotations from one. One labeler labeling the same
    frame twice measures their own consistency -- a different question, and one
    that would inflate this estimate with within-labeler autocorrelation.

    One pair per image, drawn deterministically, so an image labelled by five
    people does not contribute ten correlated pairs and overstate the sample.
    """
    import random

    rng = random.Random(seed)
    by_image: dict[Any, dict[Any, list[float]]] = {}
    for row in rows:
        by_image.setdefault(row[image_key], {}).setdefault(
            row[labeler_key], []
        ).append(float(row[resid_key]))

    pairs: list[tuple[float, float]] = []
    for labelers in by_image.values():
        if len(labelers) < 2:
            continue
        first, second = rng.sample(sorted(labelers, key=str), 2)
        pairs.append(
            (
                sum(labelers[first]) / len(labelers[first]),
                sum(labelers[second]) / len(labelers[second]),
            )
        )

    n = len(pairs)
    if n < 4:
        return Reliability(float("nan"), float("nan"), float("nan"), n)

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return Reliability(float("nan"), float("nan"), float("nan"), n)
    r = sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(var_x * var_y)

    # Fisher z interval. A bare point estimate near zero invites both
    # over- and under-reading; the interval is what says "weak, and consistent
    # with nothing at all".
    if n > 3 and abs(r) < 1:
        z = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / math.sqrt(n - 3)
        to_r = lambda v: (math.exp(2 * v) - 1) / (math.exp(2 * v) + 1)  # noqa: E731
        return Reliability(r, to_r(z - 1.96 * se), to_r(z + 1.96 * se), n)
    return Reliability(r, float("nan"), float("nan"), n)


#: z(0.975) + z(0.80): two-sided alpha 0.05 at 80% power.
_Z_SUM = 1.959964 + 0.8416212


def required_per_arm(effect: float, residual_sd: float, *, z_sum: float = _Z_SUM) -> float:
    """Annotations per arm to detect a fractional reduction in labeling time.

    Assumes a **randomized within-labeler** design: each labeler receives a
    random mix of original and enhanced frames. That design is what makes
    `residual_sd` the right noise term -- randomization balances image
    difficulty across arms, so the weak image-level reliability costs power but
    does not bias anything.

    A between-labeler design would instead have to fight the 40.3% of variance
    that is labeler identity, and is not worth costing out.
    """
    if not 0 <= effect < 1:
        raise ValueError(
            f"effect must be a fraction of the current time in [0, 1), got {effect}"
        )
    if effect == 0:
        return float("inf")
    delta = -math.log(1 - effect)
    return math.ceil(2 * z_sum**2 * residual_sd**2 / delta**2)
