"""The load-bearing test: the baseline arm IS production.

Every number this harness reports is a difference against the baseline decode.
If the baseline only resembles `fishsense_core`, each reported delta is the sum
of the effect being studied and an uncontrolled reimplementation error -- and
the two are indistinguishable in the output. So the baseline is asserted equal
to `RawImage` and `LinearRawImage` byte-for-byte, on a real `.ORF`, not
approximately and not on a synthetic array.

Marked `integration`: needs the 15 MB fixture from the data-worker's test suite
and a few seconds of rawpy + CLAHE per decode.
"""

import numpy as np
import pytest

from enhancement_eval.decode import DecodeConfig, WhiteBalance
from enhancement_eval.stages import decode_linear_stage, decode_rectified_stage

pytestmark = pytest.mark.integration


def test_baseline_rectified_decode_matches_fishsense_core_exactly(orf_bytes):
    """`RawImage` is the JPEG chain: rawpy -> auto-gamma -> CLAHE."""
    from fishsense_core.image.raw_image import RawImage

    expected = RawImage(orf_bytes).data
    actual = decode_rectified_stage(orf_bytes, DecodeConfig())

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_baseline_linear_decode_matches_fishsense_core_exactly(orf_bytes):
    """`LinearRawImage` is the laser detector's chain: linear, no CLAHE,
    sensor coordinates."""
    from fishsense_core.image.linear_raw_image import LinearRawImage

    expected = LinearRawImage(orf_bytes)
    actual, bayer = decode_linear_stage(orf_bytes, DecodeConfig())

    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected.data)
    np.testing.assert_array_equal(bayer, expected.bayer_excess)


def test_a_white_balance_change_actually_changes_the_pixels(orf_bytes):
    """Guards against the experiment silently being a no-op -- a `user_wb`
    that fails to take effect would report 'white balance does not matter',
    which is exactly the wrong conclusion to reach by accident."""
    baseline = decode_rectified_stage(orf_bytes, DecodeConfig())
    grayworld = decode_rectified_stage(
        orf_bytes, DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)
    )
    assert not np.array_equal(baseline, grayworld)


def test_white_balance_moves_the_linear_stage_too(orf_bytes):
    """The asymmetry that decides where enhancement can be placed: WB lives
    inside `rawpy.postprocess`, so it moves the laser detector's input as well
    as the labeler's JPEG. Gamma and CLAHE do not."""
    baseline, _ = decode_linear_stage(orf_bytes, DecodeConfig())
    grayworld, _ = decode_linear_stage(
        orf_bytes, DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)
    )
    assert not np.array_equal(baseline, grayworld)


def test_gamma_and_clahe_do_not_reach_the_linear_stage(orf_bytes):
    """The converse, and the reason a CLAHE sweep cannot possibly move the
    laser detector's numbers."""
    baseline, _ = decode_linear_stage(orf_bytes, DecodeConfig())
    tweaked, _ = decode_linear_stage(
        orf_bytes,
        DecodeConfig(auto_gamma_target=60, clahe_enabled=False, clahe_clip_limit=0.002),
    )
    np.testing.assert_array_equal(baseline, tweaked)


def test_disabling_clahe_changes_the_jpeg_stage(orf_bytes):
    baseline = decode_rectified_stage(orf_bytes, DecodeConfig())
    no_clahe = decode_rectified_stage(orf_bytes, DecodeConfig(clahe_enabled=False))
    assert not np.array_equal(baseline, no_clahe)


def test_every_variant_preserves_frame_geometry(orf_bytes):
    """Constraint #1 applied to the decode arms themselves. A decode variant
    that changed frame size would invalidate every label coordinate, and
    `user_flip` is one keyword away from doing exactly that."""
    baseline = decode_rectified_stage(orf_bytes, DecodeConfig())
    for config in (
        DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD),
        DecodeConfig(white_balance=WhiteBalance.WHITE_PATCH),
        DecodeConfig(auto_gamma_target=40),
        DecodeConfig(clahe_clip_limit=0.003),
        DecodeConfig(clahe_kernel_size=64),
    ):
        got = decode_rectified_stage(orf_bytes, config)
        assert got.shape == baseline.shape, f"{config.label} changed frame shape"
        assert got.dtype == baseline.dtype, f"{config.label} changed dtype"
