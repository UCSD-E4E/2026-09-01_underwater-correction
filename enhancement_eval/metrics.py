"""Paired statistics over per-image results.

Every image is scored twice against the same human label -- once through the
baseline decode, once through the enhanced one. Pairing is what makes a small
effect detectable here at all: between-image variance in this corpus is far
larger than any plausible enhancement effect, so comparing two unpaired means
would need an implausible sample to say anything.

Three reporting choices are deliberate and worth defending:

* **Signed deltas, `enhanced - baseline`, on error metrics.** Negative is
  better. Stated in one place because a sign flip in a report is the cheapest
  way for this work to recommend the wrong thing.
* **Distribution before centre.** A median delta of zero across a wide spread
  means the enhancement helps some frames and hurts others in equal measure,
  which is an entirely different finding from "no effect" and leads to a
  different decision (per-image gating, rather than ship or drop).
* **Accuracy and coverage never mix.** An image the baseline missed and the
  enhancement caught is a coverage gain, not a small error. Folding it into
  the error distribution counts one improvement twice and, worse, silently
  drops the hardest frames from the error statistics on the arm that missed
  them -- which flatters whichever arm has the worse coverage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "quantiles",
    "PairedDelta",
    "paired_delta",
    "bootstrap_median_ci",
    "CoverageFlips",
    "coverage_flips",
    "stratify_by_quartile",
    "spearman",
]

#: Below this absolute change, an image counts as unchanged rather than as a
#: win or a loss. Without it, floating-point noise makes every image
#: discordant and the win/loss counts read as a decisive result.
DEFAULT_DEAD_BAND = 1e-6

QUARTILE_LABELS = ("Q1 (worst)", "Q2", "Q3", "Q4 (best)")


def quantiles(
    values: Sequence[float], qs: Sequence[int] = (50, 90, 95)
) -> dict[int, float]:
    """Linearly-interpolated quantiles. NaN for an empty input rather than an
    exception, so one empty dive cannot abort a corpus-wide report.

    Interpolated rather than nearest-rank so that `quantiles(...)[50]` is the
    ordinary median -- the same statistic `bootstrap_median_ci` resamples. The
    nearest-rank median of an even-sized sample is one of the two middle
    values, so the reported centre and its confidence interval would be
    computed two different ways and could disagree on the sign of the effect.
    """
    if not values:
        return {q: float("nan") for q in qs}
    ordered = sorted(float(v) for v in values)
    last = len(ordered) - 1
    out: dict[int, float] = {}
    for q in qs:
        position = (q / 100.0) * last
        lower = int(math.floor(position))
        upper = min(last, lower + 1)
        weight = position - lower
        out[q] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return out


@dataclass(frozen=True)
class PairedDelta:
    """The per-image signed change, summarized. Negative is better."""

    n: int
    median: float
    mean: float
    p10: float
    p90: float
    improved: int
    worsened: int
    unchanged: int
    #: Images only one arm scored. Reported, never folded into the deltas --
    #: they are a coverage story, and `coverage_flips` tells it.
    n_baseline_only: int
    n_enhanced_only: int
    values: tuple[float, ...]

    @property
    def win_rate(self) -> float:
        """Fraction of scored pairs that improved. NaN when nothing paired."""
        decided = self.improved + self.worsened + self.unchanged
        return self.improved / decided if decided else float("nan")


def paired_delta(
    baseline: Sequence[float | None],
    enhanced: Sequence[float | None],
    *,
    dead_band: float = DEFAULT_DEAD_BAND,
) -> PairedDelta:
    """Per-image `enhanced - baseline` over images BOTH arms scored.

    A pair where either arm abstained is excluded and counted separately. That
    exclusion is the load-bearing part: the frames an arm misses are its
    hardest, so silently dropping them from the other arm's error statistics
    flatters whichever arm has the worse coverage.
    """
    if len(baseline) != len(enhanced):
        raise ValueError(
            f"arms must be paired image-for-image: got {len(baseline)} baseline "
            f"and {len(enhanced)} enhanced values"
        )

    deltas: list[float] = []
    baseline_only = enhanced_only = 0
    for base, enh in zip(baseline, enhanced):
        base_ok = base is not None and not (
            isinstance(base, float) and math.isnan(base)
        )
        enh_ok = enh is not None and not (isinstance(enh, float) and math.isnan(enh))
        if base_ok and enh_ok:
            deltas.append(float(enh) - float(base))
        elif base_ok:
            baseline_only += 1
        elif enh_ok:
            enhanced_only += 1

    q = quantiles(deltas, (10, 50, 90))
    return PairedDelta(
        n=len(deltas),
        median=q[50],
        mean=(sum(deltas) / len(deltas)) if deltas else float("nan"),
        p10=q[10],
        p90=q[90],
        improved=sum(1 for d in deltas if d < -dead_band),
        worsened=sum(1 for d in deltas if d > dead_band),
        unchanged=sum(1 for d in deltas if abs(d) <= dead_band),
        n_baseline_only=baseline_only,
        n_enhanced_only=enhanced_only,
        values=tuple(deltas),
    )


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the median.

    Present so nobody quotes a median delta from a 30-image pilot as though it
    were a corpus result. The median is used rather than the mean because a
    handful of frames where an enhancement catastrophically fails would
    otherwise dominate, and those belong in the per-image tail, not the centre.

    Deterministic for a given seed -- a report that changes when re-run is not
    evidence.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(iterations):
        sample = sorted(values[rng.randrange(n)] for _ in range(n))
        mid = n // 2
        medians.append(
            sample[mid] if n % 2 else (sample[mid - 1] + sample[mid]) / 2.0
        )
    medians.sort()
    tail = (1.0 - confidence) / 2.0
    lo = medians[min(iterations - 1, int(tail * iterations))]
    hi = medians[min(iterations - 1, int((1.0 - tail) * iterations))]
    return lo, hi


@dataclass(frozen=True)
class CoverageFlips:
    """Whether each arm produced an answer at all, as a 2x2 contingency."""

    both: int
    neither: int
    gained: int
    lost: int
    p_value: float

    @property
    def net(self) -> int:
        return self.gained - self.lost


def coverage_flips(
    baseline: Sequence[bool], enhanced: Sequence[bool]
) -> CoverageFlips:
    """Discordant-pair analysis of coverage, with an exact McNemar p-value.

    Net coverage change is not enough. A net of zero can be 0 gains and 0
    losses -- a stable pipeline -- or 40 and 40, which means the enhancement
    is trading one set of frames for another and nothing about the result is
    reliable. Only the discordant counts distinguish them, which is exactly
    what McNemar's test is for.
    """
    if len(baseline) != len(enhanced):
        raise ValueError(
            f"arms must be paired image-for-image: got {len(baseline)} baseline "
            f"and {len(enhanced)} enhanced values"
        )

    both = sum(1 for b, e in zip(baseline, enhanced) if b and e)
    neither = sum(1 for b, e in zip(baseline, enhanced) if not b and not e)
    gained = sum(1 for b, e in zip(baseline, enhanced) if not b and e)
    lost = sum(1 for b, e in zip(baseline, enhanced) if b and not e)

    # Exact binomial two-sided test on the discordant pairs. Exact rather than
    # the chi-square approximation because the discordant counts here are
    # routinely single-digit, which is where the approximation is worst.
    n = gained + lost
    if n == 0:
        p_value = 1.0
    else:
        k = min(gained, lost)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
        p_value = min(1.0, 2.0 * tail)

    return CoverageFlips(
        both=both, neither=neither, gained=gained, lost=lost, p_value=p_value
    )


def stratify_by_quartile(
    rows: Iterable[Mapping[str, Any]], *, key: str
) -> dict[str, list[Mapping[str, Any]]]:
    """Split rows into quartiles of `key`, worst first.

    This is the stratification the underwater-enhancement literature demands
    and dataset-level reporting omits: enhancement is expected to help
    degraded inputs and to *degrade* good ones. A single mean over both
    populations describes neither, and would justify shipping something that
    makes the pipeline's easy frames worse.

    Fewer than one row per bucket returns empty buckets rather than a
    stratification that is really just four samples -- reporting that as
    quartiles invites precisely the over-reading this harness exists to stop.
    """
    buckets: dict[str, list[Mapping[str, Any]]] = {q: [] for q in QUARTILE_LABELS}

    usable = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        usable.append((numeric, row))

    if len(usable) < len(QUARTILE_LABELS):
        return buckets

    usable.sort(key=lambda pair: pair[0])
    size = len(usable)
    for index, (_, row) in enumerate(usable):
        bucket = min(len(QUARTILE_LABELS) - 1, index * len(QUARTILE_LABELS) // size)
        buckets[QUARTILE_LABELS[bucket]].append(row)
    return buckets


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation. NaN when the question cannot be answered.

    Exists for one question: does a confidence score track actual correctness?
    That is what the slate detector's `ECC >= 0.80` gate assumed and nobody
    measured, and it is the same question already asked and answered for
    `LaserDepth.residual_m` against measurement error (rho = -0.026, which is
    why the residual is recorded and never gated on).

    Rank-based rather than Pearson because there is no reason for a correlation
    score and a pixel distance to be linearly related, and Pearson would
    understate a real monotonic dependence between them.

    A constant input returns NaN rather than 0.0. Zero would read as "measured,
    no relationship"; NaN reads as "this sample cannot answer it", which is the
    honest report when a variable never varies.
    """
    if len(xs) != len(ys):
        raise ValueError(
            f"spearman needs two series of the same length, got {len(xs)} and {len(ys)}"
        )
    if len(xs) < 3:
        return float("nan")

    def _ranks(values: Sequence[float]) -> list[float]:
        # Average ranks for ties, which is what makes the coefficient correct
        # when several frames share an ECC score.
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (
                stop + 1 < len(order)
                and values[order[stop + 1]] == values[order[index]]
            ):
                stop += 1
            shared = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                ranks[order[position]] = shared
            index = stop + 1
        return ranks

    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    n = len(rx)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)
