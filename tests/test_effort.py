"""Labeler effort: the oracle for "does this make labelers' lives easier?".

Label Studio stamps every annotation with `lead_time` (seconds spent),
`was_cancelled` (skipped), and `completed_by`. That is a direct behavioural
record of labeler effort across the whole corpus, and it is already collected
-- which is the single most important fact about it, because it means a real
A/B trial needs no new instrumentation.

The statistics here exist to answer three questions, in order:

1. How much of the variation in labeling time is the *image*, rather than who
   happened to label it? (Variance decomposition.)
2. Is per-image difficulty a reliable construct at all -- do two labelers agree
   about which image was slow? (Image-level reliability.)
3. Given the answers, how big does a trial have to be? (Power.)

Getting (2) wrong in the optimistic direction is the expensive mistake: it
would send someone off to build a model predicting labeling difficulty from
pixels, against an outcome that barely varies with the image.
"""

import math

import pytest

from enhancement_eval.effort import (
    LEAD_TIME_MAX_S,
    image_level_reliability,
    required_per_arm,
    usable_annotations,
    variance_explained,
)


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def test_usable_drops_cancelled_annotations():
    """A skipped task records the time spent deciding to skip, which is a
    different quantity from the time spent labeling. Skips are their own
    outcome, not a slow label."""
    rows = [
        {"lead_time": 10.0, "cancelled": False},
        {"lead_time": 10.0, "cancelled": True},
    ]
    assert len(usable_annotations(rows)) == 1


def test_usable_drops_the_left_the_tab_open_tail():
    """The corpus contains lead_times up to 868,537 s -- ten days. Those are
    browser tabs, not labeling. They are 1.2% of laser annotations and would
    dominate any mean."""
    rows = [
        {"lead_time": 12.0, "cancelled": False},
        {"lead_time": 868537.0, "cancelled": False},
    ]
    kept = usable_annotations(rows)
    assert len(kept) == 1
    assert LEAD_TIME_MAX_S < 868537


def test_usable_drops_nonpositive_times():
    rows = [{"lead_time": 0.0, "cancelled": False}, {"lead_time": -3, "cancelled": False}]
    assert usable_annotations(rows) == []


def test_the_cap_is_configurable_so_sensitivity_can_be_checked():
    """A threshold chosen from a distribution deserves a sensitivity check, so
    it must not be welded in."""
    rows = [{"lead_time": 400.0, "cancelled": False}]
    assert len(usable_annotations(rows, max_seconds=600)) == 1
    assert len(usable_annotations(rows, max_seconds=300)) == 0


# --------------------------------------------------------------------------
# variance decomposition
# --------------------------------------------------------------------------


def test_variance_explained_is_total_when_groups_are_perfectly_separated():
    rows = [{"g": "a", "y": 1.0}, {"g": "a", "y": 1.0},
            {"g": "b", "y": 5.0}, {"g": "b", "y": 5.0}]
    assert variance_explained(rows, key="g", value="y") == pytest.approx(1.0)


def test_variance_explained_is_zero_when_groups_are_identical():
    rows = [{"g": "a", "y": 1.0}, {"g": "a", "y": 5.0},
            {"g": "b", "y": 1.0}, {"g": "b", "y": 5.0}]
    assert variance_explained(rows, key="g", value="y") == pytest.approx(0.0, abs=1e-9)


def test_variance_explained_of_a_constant_column_is_nan():
    """Zero would claim 'measured, explains nothing'. NaN is the honest report
    when there is no variance to explain."""
    rows = [{"g": "a", "y": 3.0}, {"g": "b", "y": 3.0}]
    assert math.isnan(variance_explained(rows, key="g", value="y"))


# --------------------------------------------------------------------------
# image-level reliability -- the question the direction depends on
# --------------------------------------------------------------------------


def test_reliability_is_high_when_labelers_agree_about_which_image_was_slow():
    rows = []
    for image_id, effect in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0] * 8):
        for labeler in ("A", "B"):
            rows.append({"image_id": image_id, "labeler": labeler,
                         "resid": effect + (0.01 if labeler == "A" else -0.01)})
    result = image_level_reliability(rows)
    assert result.r > 0.9
    assert result.n_pairs == 40


def test_reliability_is_near_zero_when_times_are_independent_noise():
    """The null this measurement has to be able to return -- and, on the real
    laser corpus, does."""
    import random

    rng = random.Random(7)
    rows = []
    for image_id in range(400):
        for labeler in ("A", "B"):
            rows.append({"image_id": image_id, "labeler": labeler,
                         "resid": rng.gauss(0, 1)})
    assert abs(image_level_reliability(rows).r) < 0.15


def test_reliability_reports_a_confidence_interval():
    """r = +0.065 over 755 pairs is not distinguishable from a small positive
    effect OR from zero, and a bare point estimate hides that."""
    import random

    rng = random.Random(1)
    rows = []
    for image_id in range(300):
        for labeler in ("A", "B"):
            rows.append({"image_id": image_id, "labeler": labeler,
                         "resid": rng.gauss(0, 1)})
    result = image_level_reliability(rows)
    assert result.ci_low < result.r < result.ci_high


def test_reliability_needs_two_distinct_labelers_not_two_annotations():
    """One labeler labeling the same image twice measures their own
    consistency, which is a different question and would inflate the estimate."""
    rows = [{"image_id": 1, "labeler": "A", "resid": 0.5},
            {"image_id": 1, "labeler": "A", "resid": 0.6}]
    assert image_level_reliability(rows).n_pairs == 0


def test_reliability_of_too_few_pairs_is_nan():
    rows = [{"image_id": 1, "labeler": "A", "resid": 0.1},
            {"image_id": 1, "labeler": "B", "resid": 0.2}]
    assert math.isnan(image_level_reliability(rows).r)


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------


def test_required_sample_shrinks_as_the_effect_grows():
    assert required_per_arm(0.05, 0.546) > required_per_arm(0.20, 0.546)


def test_required_sample_grows_with_noise():
    assert required_per_arm(0.10, 0.752) > required_per_arm(0.10, 0.546)


def test_required_sample_matches_the_standard_formula():
    """20% reduction, sd 0.546, alpha .05 two-sided, power .80 -> 94/arm."""
    assert required_per_arm(0.20, 0.546) == 94


def test_a_zero_effect_needs_an_infinite_sample():
    assert math.isinf(required_per_arm(0.0, 0.5))


def test_effect_must_be_a_fraction_below_one():
    with pytest.raises(ValueError, match="fraction"):
        required_per_arm(1.5, 0.5)
