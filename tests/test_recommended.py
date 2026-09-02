"""The recommended configurations, pinned.

These are what the visual screen and the measurements converged on. They are
named and tested so a result can be quoted against a specific configuration
rather than a description, and so a later change to the defaults cannot quietly
move what "recommended" means.

None of this is deployed. The decisive evidence is the labeling trial, which
has not run; these are the arms it would run.
"""

import numpy as np
import pytest

from enhancement_eval.decode import DecodeConfig
from enhancement_eval.recommended import (
    PRODUCTION,
    RECOMMENDED,
    RECOMMENDED_WITH_DOT,
    GENTLE_DENOISE,
    describe,
)


def test_production_is_the_current_decode_unchanged():
    """The control arm. If this drifts, every reported delta drifts with it."""
    assert PRODUCTION == DecodeConfig()
    assert PRODUCTION.clahe_enabled is True
    assert PRODUCTION.stretch_mode == "off"


def test_recommended_is_luminance_stretch_with_clahe_off():
    assert RECOMMENDED.stretch_mode == "luminance"
    assert RECOMMENDED.clahe_enabled is False
    assert RECOMMENDED.white_balance is PRODUCTION.white_balance, (
        "the recommendation deliberately does NOT change white balance: every "
        "variant that gave red its own gain failed on field frames"
    )


def test_recommended_with_dot_adds_only_the_adaptive_red_lift():
    """Same scene treatment, plus the laser lift. Kept as a separate config so
    the two can be trialled independently -- the dot lift is a labeling aid and
    deserves its own decision."""
    assert RECOMMENDED_WITH_DOT.stretch_mode == RECOMMENDED.stretch_mode
    assert RECOMMENDED_WITH_DOT.clahe_enabled == RECOMMENDED.clahe_enabled
    assert RECOMMENDED_WITH_DOT.red_boost > 0
    assert RECOMMENDED_WITH_DOT.red_boost_sigmas >= 6.0, (
        "the threshold must be referenced to the frame's own noise: the dot "
        "sits at the 99.999th red percentile on a reef and the 58.8th in a pool"
    )


def test_gentle_denoise_is_offered_but_not_default():
    """Denoising is a real trade -- fish texture is only ~1.6x the grain in the
    same band -- so it is a separate, opt-in arm rather than part of the
    recommendation."""
    assert GENTLE_DENOISE is not RECOMMENDED
    assert RECOMMENDED.denoise == "off"


def test_every_recommended_config_is_hashable_and_labelled():
    for cfg in (PRODUCTION, RECOMMENDED, RECOMMENDED_WITH_DOT):
        assert isinstance(hash(cfg), int)
        assert cfg.label


def test_the_recommendations_have_distinct_labels():
    labels = {c.label for c in (PRODUCTION, RECOMMENDED, RECOMMENDED_WITH_DOT)}
    assert len(labels) == 3


def test_describe_states_what_each_arm_changes_and_what_it_does_not():
    text = describe()
    assert "white balance" in text.lower()
    assert "trial" in text.lower()


@pytest.mark.parametrize("cfg", [RECOMMENDED, RECOMMENDED_WITH_DOT])
def test_recommended_configs_move_no_pixels(cfg):
    """Constraint #1 against the actual shipping candidates, not a stand-in."""
    from enhancement_eval.contract import probe_geometry
    from enhancement_eval.stages import apply_clahe, apply_red_boost, apply_stretch

    def enhance(a):
        f = a.astype(np.float64) / 255.0
        f = apply_stretch(f, cfg)
        f = apply_clahe(f, cfg)
        f = apply_red_boost(f, cfg)
        return (np.clip(f, 0, 1) * 255).astype(a.dtype)

    result = probe_geometry(enhance, name=cfg.label)
    assert result.displacement_px < 0.5


def test_recommended_reaches_no_laser_dependency():
    """The recommendations are JPEG-stage. Asserted against the same guard the
    experiment table uses, so the two cannot disagree."""
    from enhancement_eval.evaluate import Arm, Stage
    from enhancement_eval.experiments import assert_no_laser_dependency

    assert_no_laser_dependency(
        [Arm("recommended", Stage.RECTIFIED, decode=RECOMMENDED),
         Arm("recommended-dot", Stage.RECTIFIED, decode=RECOMMENDED_WITH_DOT)]
    )
