"""Per-image scoring primitives.

Kept pure and separate from the model calls so the decisions that shape a
result -- which human dot a prediction is compared against, what counts as
coverage, what "quality" means for stratification -- are testable without a
GPU or a 15 MB raw file.
"""

import math

import numpy as np
import pytest

from enhancement_eval.scoring import (
    image_quality,
    laser_color_agreement,
    nearest_error,
)


# --------------------------------------------------------------------------
# nearest_error
# --------------------------------------------------------------------------


def test_nearest_error_is_the_distance_to_the_closest_human_dot():
    """461 prod images carry two valid laser labels and there is no basis in
    the data for deciding which is 'the' dot. Comparing against the nearest
    avoids reporting a real hit as a large miss; the cost is that it is
    slightly generous, equally, to both arms."""
    assert nearest_error((10.0, 10.0), [(13.0, 14.0), (100.0, 100.0)]) == pytest.approx(5.0)


def test_nearest_error_of_a_non_detection_is_none_not_infinity():
    """A miss is a coverage fact, not a very large error. Folding it in as
    `inf` would let a coverage collapse masquerade as an accuracy result --
    and worse, the median would hide it entirely."""
    assert nearest_error(None, [(1.0, 1.0)]) is None


def test_nearest_error_with_no_human_dots_is_none():
    assert nearest_error((1.0, 1.0), []) is None


def test_nearest_error_is_exact_on_a_direct_hit():
    assert nearest_error((7.0, 7.0), [(7.0, 7.0)]) == 0.0


# --------------------------------------------------------------------------
# image_quality -- the stratification axis
# --------------------------------------------------------------------------


def test_quality_reports_luminance_and_contrast():
    """The literature's claim is that enhancement helps degraded inputs and
    hurts good ones. Testing that claim needs an axis to sort inputs along,
    computed on the BASELINE frame so both arms are stratified identically."""
    dark = np.full((20, 20, 3), 10, dtype=np.uint8)
    bright = np.full((20, 20, 3), 200, dtype=np.uint8)
    assert image_quality(dark)["mean_luminance"] < image_quality(bright)["mean_luminance"]


def test_quality_contrast_separates_a_flat_frame_from_a_textured_one():
    flat = np.full((20, 20, 3), 128, dtype=np.uint8)
    textured = np.zeros((20, 20, 3), dtype=np.uint8)
    textured[::2] = 255
    assert image_quality(flat)["rms_contrast"] < image_quality(textured)["rms_contrast"]


def test_quality_reports_the_red_to_green_ratio():
    """The attenuation signature, and the number Deliverable 2's white-balance
    experiments are supposed to move. Red falls off fastest underwater, so a
    low ratio is the direct symptom."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # G
    img[:, :, 2] = 50   # R
    assert image_quality(img)["red_green_ratio"] == pytest.approx(0.25)


def test_quality_is_scale_normalized_across_dtypes():
    """The rectified stage is uint8 and the linear stage is uint16. A quality
    proxy that reported raw levels would put every linear frame in the 'good'
    quartile purely because its container is bigger."""
    u8 = np.full((10, 10, 3), 128, dtype=np.uint8)
    u16 = np.full((10, 10, 3), 128 * 257, dtype=np.uint16)
    assert image_quality(u8)["mean_luminance"] == pytest.approx(
        image_quality(u16)["mean_luminance"], rel=0.01
    )


def test_quality_of_a_black_frame_does_not_divide_by_zero():
    q = image_quality(np.zeros((10, 10, 3), dtype=np.uint8))
    assert all(math.isfinite(v) for v in q.values())


# --------------------------------------------------------------------------
# the classify_laser_color tripwire
# --------------------------------------------------------------------------


def test_color_agreement_counts_a_match():
    assert laser_color_agreement("red", "red") == "agree"


def test_color_agreement_distinguishes_a_flip_from_an_abstention():
    """These are not the same failure. An abstention costs a vote; a flip
    casts a WRONG vote, and populate takes the dive-level majority -- so
    enough flips silently relabel an entire dive's laser colour."""
    assert laser_color_agreement("red", "green") == "flipped"
    assert laser_color_agreement("red", None) == "abstained"


def test_color_agreement_records_a_recovered_opinion():
    """The enhancement making a previously-undecidable dot decidable is a real
    outcome and deserves its own name rather than being folded into 'agree'."""
    assert laser_color_agreement(None, "red") == "gained"


def test_color_agreement_when_neither_arm_has_an_opinion():
    assert laser_color_agreement(None, None) == "both_abstained"
