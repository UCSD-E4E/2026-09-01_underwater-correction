"""Paired statistics over per-image results.

The design is paired by construction: every image is scored twice, once
through the baseline decode and once through the enhanced one, and the two
share a human label. That pairing is the whole reason this harness can detect
a small effect at all -- between-image variance in this corpus dwarfs any
plausible enhancement effect, so an unpaired comparison of two means would
need an implausible sample size to say anything.

It also drives what gets reported. The literature on underwater enhancement is
consistent that dataset-level means hide large image-level variance, and that
enhancement helps degraded inputs while *degrading* good ones. A single mean
delta averages those two populations into a number that describes neither.
"""

import math

import pytest

from enhancement_eval.metrics import (
    bootstrap_median_ci,
    coverage_flips,
    paired_delta,
    quantiles,
    stratify_by_quartile,
)


# --------------------------------------------------------------------------
# quantiles
# --------------------------------------------------------------------------


def test_quantiles_on_a_known_sequence():
    q = quantiles(list(range(1, 101)), (50, 90))
    assert q[50] == pytest.approx(50.5)
    assert q[90] == pytest.approx(90.1)


def test_quantiles_median_is_interpolated_so_it_matches_the_bootstrap():
    """The reported centre and its confidence interval must be the same
    statistic, or they can disagree on the sign of the effect."""
    assert quantiles([-8.0, -4.0, 4.0, 8.0], (50,))[50] == pytest.approx(0.0)


def test_quantiles_of_empty_is_nan_not_an_exception():
    """A dive with no scored images must not abort a corpus-wide report."""
    q = quantiles([], (50,))
    assert math.isnan(q[50])


def test_quantiles_of_a_single_value():
    assert quantiles([7.0], (10, 50, 90)) == {10: 7.0, 50: 7.0, 90: 7.0}


# --------------------------------------------------------------------------
# paired_delta
# --------------------------------------------------------------------------


def test_paired_delta_signs_improvement_as_negative_for_an_error_metric():
    """Deltas are enhanced - baseline on an error metric, so negative is
    better. Stated once, here, because a sign flip in the report is the
    easiest way for this harness to recommend the wrong thing."""
    d = paired_delta(baseline=[10.0, 10.0, 10.0], enhanced=[8.0, 8.0, 8.0])
    assert d.median < 0
    assert d.improved == 3
    assert d.worsened == 0


def test_paired_delta_counts_wins_losses_and_ties_with_a_dead_band():
    """A sub-threshold wobble is not a win. Without a dead band, floating
    point noise makes every image a 'win' or a 'loss' and the counts read as
    a decisive result."""
    d = paired_delta(
        baseline=[10.0, 10.0, 10.0, 10.0],
        enhanced=[5.0, 15.0, 10.0001, 9.9999],
        dead_band=0.01,
    )
    assert (d.improved, d.worsened, d.unchanged) == (1, 1, 2)


def test_paired_delta_requires_equal_length_arms():
    with pytest.raises(ValueError, match="paired"):
        paired_delta(baseline=[1.0, 2.0], enhanced=[1.0])


def test_paired_delta_drops_pairs_where_either_arm_is_missing():
    """An image only contributes if BOTH arms scored it. Scoring an image the
    baseline missed but the enhancement caught, as though it were an
    improvement in *error*, double-counts a coverage gain as an accuracy gain.
    Coverage changes are reported separately, by `coverage_flips`."""
    d = paired_delta(baseline=[10.0, None, 8.0], enhanced=[9.0, 4.0, None])
    assert d.n == 1
    assert d.median == pytest.approx(-1.0)
    assert d.n_baseline_only == 1
    assert d.n_enhanced_only == 1


def test_paired_delta_of_nothing_is_empty_not_an_exception():
    d = paired_delta(baseline=[], enhanced=[])
    assert d.n == 0
    assert math.isnan(d.median)


def test_paired_delta_reports_the_distribution_not_just_the_centre():
    """The spread is the point: a median of zero over a wide spread means the
    enhancement is helping some images and hurting others equally, which is a
    completely different finding from 'no effect'."""
    d = paired_delta(
        baseline=[10.0] * 4, enhanced=[2.0, 6.0, 14.0, 18.0]
    )
    assert d.median == pytest.approx(0.0, abs=1e-9)
    assert d.p10 < -3
    assert d.p90 > 3


# --------------------------------------------------------------------------
# bootstrap_median_ci
# --------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_median_of_a_tight_sample():
    lo, hi = bootstrap_median_ci([-2.0] * 50, iterations=500, seed=1)
    assert lo == pytest.approx(-2.0)
    assert hi == pytest.approx(-2.0)


def test_bootstrap_ci_of_a_noisy_zero_sample_spans_zero():
    """The guard against over-reading a 30-image pilot: if the CI spans zero,
    the sample does not support a directional claim."""
    values = [float(v) for v in range(-15, 16)]
    lo, hi = bootstrap_median_ci(values, iterations=1000, seed=1)
    assert lo < 0 < hi


def test_bootstrap_ci_is_deterministic_for_a_given_seed():
    a = bootstrap_median_ci([1.0, 5.0, -3.0, 2.0], iterations=200, seed=7)
    b = bootstrap_median_ci([1.0, 5.0, -3.0, 2.0], iterations=200, seed=7)
    assert a == b


def test_bootstrap_ci_of_too_few_samples_is_nan():
    lo, hi = bootstrap_median_ci([1.0], iterations=100, seed=1)
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------
# coverage_flips
# --------------------------------------------------------------------------


def test_coverage_flips_counts_gains_and_losses_separately():
    """A net-zero coverage change can hide 40 gains and 40 losses, which is a
    much worse outcome than 0 and 0 -- the pipeline would be churning, not
    stable. McNemar's discordant pairs are the honest summary."""
    flips = coverage_flips(
        baseline=[True, True, False, False],
        enhanced=[True, False, True, False],
    )
    assert flips.gained == 1
    assert flips.lost == 1
    assert flips.both == 1
    assert flips.neither == 1


def test_coverage_flips_reports_a_p_value_for_the_discordant_pairs():
    """Twenty losses against one gain is not luck."""
    flips = coverage_flips(
        baseline=[True] * 20 + [False], enhanced=[False] * 20 + [True]
    )
    assert flips.gained == 1 and flips.lost == 20
    assert flips.p_value < 0.001


def test_coverage_flips_with_no_discordant_pairs_has_p_value_one():
    flips = coverage_flips(baseline=[True, False], enhanced=[True, False])
    assert flips.p_value == pytest.approx(1.0)


def test_coverage_flips_requires_equal_length_arms():
    with pytest.raises(ValueError, match="paired"):
        coverage_flips(baseline=[True], enhanced=[True, False])


# --------------------------------------------------------------------------
# stratify_by_quartile
# --------------------------------------------------------------------------


def test_stratify_splits_into_four_labelled_buckets_by_the_keyed_value():
    """The stratification the literature demands: enhancement is expected to
    help the worst inputs and hurt the best ones, and a single mean over both
    populations describes neither."""
    rows = [{"q": float(i), "d": -float(i)} for i in range(100)]
    buckets = stratify_by_quartile(rows, key="q")
    assert list(buckets) == ["Q1 (worst)", "Q2", "Q3", "Q4 (best)"]
    assert sum(len(v) for v in buckets.values()) == 100
    assert max(r["q"] for r in buckets["Q1 (worst)"]) < min(
        r["q"] for r in buckets["Q4 (best)"]
    )


def test_stratify_skips_rows_missing_the_key():
    """A frame the harness could not compute a quality proxy for is dropped,
    not defaulted into a bucket -- defaulting would concentrate every
    unmeasurable frame into one quartile and make that quartile mean nothing.
    """
    rows = [{"q": 1.0}, {"q": None}, {"q": 2.0}, {}, {"q": 3.0}, {"q": 4.0}]
    buckets = stratify_by_quartile(rows, key="q")
    assert sum(len(v) for v in buckets.values()) == 4


def test_stratify_of_too_few_rows_returns_empty_buckets_rather_than_lying():
    """Four quartiles over three images is not a stratification, and reporting
    it as one invites exactly the over-reading this harness exists to prevent."""
    buckets = stratify_by_quartile([{"q": 1.0}, {"q": 2.0}, {"q": 3.0}], key="q")
    assert all(len(v) == 0 for v in buckets.values())
