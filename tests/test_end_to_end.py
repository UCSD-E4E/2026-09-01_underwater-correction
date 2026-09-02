"""End-to-end smoke: a real .ORF through both arms, with the real detector.

This is the test that proves the harness is wired to the actual pipeline
rather than to a plausible-looking mock of it. It is slow (a full rawpy decode
per arm, plus tiled CPU inference) and needs the laser-detector checkpoint, so
it is marked `integration` and skipped when either is unavailable.

It deliberately asserts almost nothing about the *values* -- the fixture has no
human label and no real intrinsics, so there is no ground truth to score
against. What it asserts is that every seam holds: the decode variants produce
distinct frames, the detector accepts an enhanced uint16 frame alongside the
original Bayer-excess, the colour tripwire samples in the right coordinate
frame, and the geometry guard survives contact with real data.
"""

import numpy as np
import pytest

from enhancement_eval.contract import IDENTITY
from enhancement_eval.decode import DecodeConfig, WhiteBalance
from enhancement_eval.evaluate import Arm, LaserConsumer, Stage, evaluate_image

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def laser_consumer():
    try:
        return LaserConsumer()
    except Exception as exc:  # pragma: no cover - depends on local checkpoint
        pytest.skip(f"laser detector unavailable: {exc}")


@pytest.fixture(scope="module")
def synthetic_row(orf_bytes):
    """A manifest row for the fixture.

    Intrinsics are a plausible pinhole with zero distortion, which makes
    `rectify_output` an identity and keeps the smoke test about plumbing rather
    than about a calibration the fixture does not have.
    """
    from enhancement_eval.stages import decode_linear_stage

    bgr, _ = decode_linear_stage(orf_bytes, DecodeConfig())
    height, width = bgr.shape[:2]
    return {
        "image_id": 1,
        "dive_id": 1,
        "dive_name": "fixture",
        "checksum": "fixture",
        "laser_points": f"{width / 2},{height / 2}",
        "camera_matrix": [
            [float(width), 0.0, width / 2.0],
            [0.0, float(width), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_linear_arm_runs_the_real_detector_and_the_colour_tripwire(
    synthetic_row, orf_bytes, laser_consumer
):
    records = evaluate_image(
        synthetic_row,
        orf_bytes,
        [Arm("baseline", Stage.LINEAR)],
        laser=laser_consumer,
    )
    (record,) = records
    assert record["arm"] == "baseline"
    # The detector always answers; what matters is that it answered at all and
    # that the coverage flag and the error agree with each other.
    assert "laser_detected" in record
    assert np.isfinite(record["laser_confidence"])
    if record["laser_detected"]:
        assert record["laser_err_px"] is not None
    else:
        assert record["laser_err_px"] is None
    # The tripwire produced a verdict (or an honest abstention).
    assert "laser_color" in record
    # Quality covariates rode along for stratification.
    assert 0.0 <= record["mean_luminance"] <= 1.0
    assert record["red_green_ratio"] >= 0.0


def test_a_white_balance_arm_moves_the_detector_input_and_the_colour_margin(
    synthetic_row, orf_bytes, laser_consumer
):
    """The experiment that matters most, end to end: white balance is the one
    class of change the detector's scale-invariant chromaticity channels CAN
    see, and it is also what the colour tripwire keys on."""
    records = evaluate_image(
        synthetic_row,
        orf_bytes,
        [
            Arm("baseline", Stage.LINEAR),
            Arm(
                "grayworld",
                Stage.LINEAR,
                decode=DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD),
            ),
        ],
        laser=laser_consumer,
    )
    baseline, grayworld = records
    assert baseline["laser_color_margin"] != grayworld["laser_color_margin"], (
        "a gray-world white balance left the R-G margin at the dot unchanged; "
        "either the user_wb did not take effect or the tripwire is not reading "
        "the enhanced frame"
    )


def test_an_enhancer_that_moves_pixels_is_refused_on_real_data(
    synthetic_row, orf_bytes
):
    """Constraint #1, exercised against a real frame rather than a 64x64
    synthetic one."""
    from enhancement_eval.contract import GeometryViolation

    with pytest.raises(GeometryViolation):
        evaluate_image(
            synthetic_row,
            orf_bytes,
            [Arm("roll", Stage.LINEAR, enhancer=lambda a: np.roll(a, 1, axis=1))],
        )


def test_identity_enhancer_leaves_the_linear_frame_untouched(
    synthetic_row, orf_bytes, laser_consumer
):
    """The control arm must be a true control: an identity enhancer and no
    enhancer at all must produce identical detector input, or every delta is
    measured against a moving baseline."""
    from enhancement_eval.stages import decode_linear_stage

    plain, _ = decode_linear_stage(orf_bytes, DecodeConfig())
    through_identity = IDENTITY(plain)
    np.testing.assert_array_equal(plain, through_identity)
