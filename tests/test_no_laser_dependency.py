"""The guarantee: enhancement must not create a dependency for laser detection.

The laser's value to this project is the *depth structure* it provides --
`LaserDepth.range_m` is the metric anchor a physics-based enhancement fits
against. It is an input. Laser *detection* must not become a downstream
consumer of enhancement, because that would couple a working, calibrated,
98%-accurate measurement path to an experimental one.

That guarantee is free if enhancement stays at the JPEG stage, because
`predict_laser_image` decodes the .ORF to a `LinearRawImage` and never reads
the JPEG. But "free if nobody moves it" is not a guarantee, so it is enforced
here: a shipping-candidate experiment cannot contain a linear-stage arm, and
the check is structural rather than a comment somebody has to notice.
"""

import pytest

from enhancement_eval.evaluate import Arm, Stage
from enhancement_eval.experiments import (
    EXPERIMENTS,
    SHIPPING_CANDIDATES,
    assert_no_laser_dependency,
    build_arms,
)


def test_every_shipping_candidate_experiment_is_jpeg_stage_only():
    """The whole guarantee, in one assertion over the real experiment table."""
    for name in SHIPPING_CANDIDATES:
        for arm in build_arms(name):
            assert arm.stage is Stage.RECTIFIED, (
                f"experiment {name!r} arm {arm.label!r} runs at the linear stage, "
                "which is upstream of the JPEG and would make laser detection "
                "depend on enhancement"
            )


def test_the_guard_rejects_a_linear_arm_in_a_shipping_candidate():
    with pytest.raises(ValueError, match="laser"):
        assert_no_laser_dependency([Arm("sneaky", Stage.LINEAR)])


def test_the_guard_accepts_jpeg_stage_arms():
    assert_no_laser_dependency([Arm("fine", Stage.RECTIFIED)])


def test_diagnostic_experiments_are_not_shipping_candidates():
    """Linear-stage arms still exist -- they are how the cost of the placement
    constraint gets measured. They just must not be mistakable for proposals."""
    diagnostics = set(EXPERIMENTS) - set(SHIPPING_CANDIDATES)
    assert diagnostics, "the diagnostic arms should still be available"
    for name in diagnostics:
        assert EXPERIMENTS[name].diagnostic_only


def test_every_experiment_declares_which_consumers_it_can_move():
    """A reader must be able to tell, without reading the pipeline, that a
    CLAHE sweep cannot move laser detection. Stating it per experiment is what
    stops the next person re-running the measurement to find out."""
    for name, experiment in EXPERIMENTS.items():
        assert experiment.consumers, f"{name} declares no consumers"
        if not experiment.diagnostic_only:
            assert "laser detection" not in experiment.consumers
