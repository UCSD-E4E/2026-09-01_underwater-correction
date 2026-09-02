"""Slate scoring, and the ECC-vs-truth analysis the old gate needed.

The slate detector shipped 2026-08-02 behind an `ECC >= 0.80` acceptance gate
and was shut down the next day: pool dives produced high-ECC (0.93-0.97)
*false* fits that sailed through it. The gate was a confidence score being used
as a correctness proxy, and nobody had measured whether it was one.

`DiveSlateLabel.reference_points` makes that measurable. They are stored in
photo-frame pixels (the sync activity shifts the composite's PDF-panel width
off the x coords), which is the same space `BoardEstimate.image_points` lands
in -- so human and predicted points are directly comparable, positionally
paired.

The decisive number is the rank correlation between ECC and actual point
error. This mirrors the analysis already done for `LaserDepth.residual_m`
against measurement error, which found Spearman rho = -0.026 and killed the
idea of gating on it. If ECC correlates with truth, the gate was merely
mis-tuned; if it does not, no threshold on it could ever have worked, and that
is a different and more useful conclusion.
"""

import math

import pytest

from enhancement_eval.metrics import spearman
from enhancement_eval.scoring import slate_point_errors


# --------------------------------------------------------------------------
# spearman
# --------------------------------------------------------------------------


def test_spearman_of_a_perfect_monotonic_relationship_is_one():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_spearman_of_a_perfect_inverse_relationship_is_minus_one():
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_so_it_survives_a_nonlinear_transform():
    """ECC is a correlation score and point error is a distance; there is no
    reason for their relationship to be linear, so Pearson would understate a
    real monotonic dependence."""
    xs = [1, 2, 3, 4, 5]
    assert spearman(xs, [math.exp(x) for x in xs]) == pytest.approx(1.0)


def test_spearman_handles_ties_without_blowing_up():
    value = spearman([1, 1, 2, 2, 3], [5, 5, 3, 3, 1])
    assert -1.0 <= value <= 1.0


def test_spearman_of_a_constant_series_is_nan_not_zero():
    """Zero would read as 'measured, no relationship'. NaN reads as 'this
    sample cannot answer the question', which is the truth when one variable
    never varies."""
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_spearman_of_too_few_points_is_nan():
    assert math.isnan(spearman([1.0], [2.0]))


def test_spearman_requires_equal_lengths():
    with pytest.raises(ValueError, match="same length"):
        spearman([1, 2], [1])


# --------------------------------------------------------------------------
# slate_point_errors
# --------------------------------------------------------------------------


def test_point_errors_are_paired_positionally():
    """Stage 13 pairs reference points by position, so the scoring must too.
    Matching nearest-neighbour instead would hide exactly the failure that
    matters -- a fit that found the board but resolved its 4-fold corner
    ambiguity wrong, which yields small nearest-neighbour distances and
    completely wrong correspondences."""
    human = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    predicted = [(0.0, 0.0), (10.0, 0.0), (10.0, 13.0)]
    result = slate_point_errors(predicted, human)
    assert result["n_points"] == 3
    assert result["median_px"] == pytest.approx(0.0)
    assert result["max_px"] == pytest.approx(3.0)


def test_a_rotated_correspondence_is_reported_as_large_error():
    """The 4-fold ambiguity failure: same points, wrong order. Nearest-
    neighbour scoring would call this perfect."""
    human = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    rotated = human[2:] + human[:2]
    result = slate_point_errors(rotated, human)
    assert result["median_px"] > 10


def test_mismatched_point_counts_abstain_rather_than_truncate():
    """A partial set breaks stage-13's positional pairing, and silently
    zipping to the shorter list would score a wrong-template fit as a good
    one on its first few points."""
    result = slate_point_errors([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])
    assert result["status"] == "point_count_mismatch"
    assert result["median_px"] is None


def test_no_prediction_abstains():
    result = slate_point_errors(None, [(0.0, 0.0)])
    assert result["status"] == "no_estimate"
    assert result["median_px"] is None


def test_no_human_labels_abstains():
    result = slate_point_errors([(0.0, 0.0)], [])
    assert result["status"] == "no_human_label"


def test_a_good_fit_reports_status_scored():
    result = slate_point_errors([(1.0, 1.0)], [(0.0, 0.0)])
    assert result["status"] == "scored"
    assert result["median_px"] == pytest.approx(math.sqrt(2))
