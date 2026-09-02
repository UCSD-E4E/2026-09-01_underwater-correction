"""The named arm sets, and the guarantee that none of them touches the laser.

Experiments are predefined rather than assembled at the command line so a
result can be quoted by name and reproduced exactly -- "wb-jpeg, 40 frames per
dive" is a claim somebody else can re-run; "I swept some white balances" is not.

----------------------------------------------------------------------------
The laser is an input, not a consumer
----------------------------------------------------------------------------
The laser's value to this project is the depth structure it provides:
`LaserDepth.range_m` is a metric range at a known pixel in every laser-labelled
frame, which is the anchor a physics-based enhancement fits against. That makes
it an **input**.

Laser *detection* must not become a downstream consumer of enhancement. It is a
working, calibrated path, and coupling it to an experimental one trades a known
quantity for an unknown. That guarantee is free as long as enhancement stays at
the JPEG stage, because `predict_laser_image` decodes the `.ORF` into a
`LinearRawImage` and never reads the JPEG -- but "free unless somebody moves it"
is not a guarantee, so `assert_no_laser_dependency` enforces it and
`SHIPPING_CANDIDATES` is checked against it by the test suite.

Linear-stage arms still exist. They are diagnostics: the only way to measure
what the placement constraint costs is to measure the thing it forbids. They
are marked `diagnostic_only` so they cannot be mistaken for proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from enhancement_eval.decode import DecodeConfig, WhiteBalance
from enhancement_eval.evaluate import Arm, Stage

__all__ = [
    "Experiment",
    "EXPERIMENTS",
    "SHIPPING_CANDIDATES",
    "assert_no_laser_dependency",
    "build_arms",
]


@dataclass(frozen=True)
class Experiment:
    """One named arm set, plus what it is allowed to conclude about."""

    build: Callable[[], list[Arm]]
    #: Which pipeline consumers this experiment's stage can actually reach.
    #: Stated per experiment so a reader can tell, without tracing the
    #: pipeline, that a CLAHE sweep cannot move laser detection -- which is
    #: what stops the next person re-running the measurement to find out.
    consumers: tuple[str, ...]
    #: True for arms that would place enhancement upstream of the JPEG. Those
    #: measure the cost of the placement constraint; they are never proposals.
    diagnostic_only: bool = False
    note: str = ""


def assert_no_laser_dependency(arms: Sequence[Arm]) -> None:
    """Refuse an arm set that would make laser detection depend on enhancement.

    A linear-stage arm sits upstream of the JPEG, where `LaserDetector` and
    `classify_laser_color` both read. Enhancement there would couple the
    measurement path to the enhancement path -- exactly the dependency this
    project is meant to avoid creating.
    """
    offenders = [arm.label for arm in arms if arm.stage is Stage.LINEAR]
    if offenders:
        raise ValueError(
            f"arms {offenders} run at the linear stage, upstream of the JPEG. "
            "That would make laser detection (and classify_laser_color) depend "
            "on enhancement. The laser supplies depth structure to this project; "
            "it is an input, not a consumer. Keep shipping candidates at the "
            "JPEG stage."
        )


def _wb_arms(stage: Stage) -> list[Arm]:
    """White-balance candidates, in increasing order of scene knowledge.

    SLATE abstains on images with no slate label; that shows up as unpaired
    coverage rather than being silently dropped.
    """
    return [
        Arm("baseline", stage),
        Arm("grayworld", stage, decode=DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD)),
        Arm("whitepatch", stage, decode=DecodeConfig(white_balance=WhiteBalance.WHITE_PATCH)),
        Arm("rawpyauto", stage, decode=DecodeConfig(white_balance=WhiteBalance.RAWPY_AUTO)),
        Arm("slate", stage, decode=DecodeConfig(white_balance=WhiteBalance.SLATE)),
    ]


def _clahe_arms() -> list[Arm]:
    """The CLAHE / auto-gamma sweep. JPEG stage only, by construction.

    Clip limits bracket skimage's 0.01 default downward, because the concern is
    that the default is too *permissive* on frames auto-gamma has just lifted
    out of the dark. `noclahe` is the floor: nobody has measured whether CLAHE
    is helping at all, and the slate estimator's first step segments a *bright
    quad*, which local histogram equalization is well placed to disrupt.
    """
    stage = Stage.RECTIFIED
    return [
        Arm("baseline", stage),
        Arm("noclahe", stage, decode=DecodeConfig(clahe_enabled=False)),
        Arm("clip0.003", stage, decode=DecodeConfig(clahe_clip_limit=0.003)),
        Arm("clip0.005", stage, decode=DecodeConfig(clahe_clip_limit=0.005)),
        Arm("clip0.02", stage, decode=DecodeConfig(clahe_clip_limit=0.02)),
        Arm("kernel64", stage, decode=DecodeConfig(clahe_kernel_size=64)),
        Arm("kernel256", stage, decode=DecodeConfig(clahe_kernel_size=256)),
        # Auto-gamma and CLAHE interact -- a brighter lift leaves CLAHE less to
        # amplify -- so the target sweep belongs in this experiment, not its own.
        Arm("gamma40", stage, decode=DecodeConfig(auto_gamma_target=40)),
        Arm("gamma60", stage, decode=DecodeConfig(auto_gamma_target=60)),
        Arm("gamma40-clip0.005", stage,
            decode=DecodeConfig(auto_gamma_target=40, clahe_clip_limit=0.005)),
    ]


_JPEG_CONSUMERS = ("fish segmentation", "head/tail", "slate estimator", "human labelers")

EXPERIMENTS: dict[str, Experiment] = {
    # ---------------- shipping candidates: JPEG stage only ----------------
    "wb-jpeg": Experiment(
        build=lambda: _wb_arms(Stage.RECTIFIED),
        consumers=_JPEG_CONSUMERS,
        note=(
            "Deliverable 2, suspect 1. use_camera_wb=True is a topside daylight "
            "preset applied to a scene lit through metres of water."
        ),
    ),
    "clahe-jpeg": Experiment(
        build=_clahe_arms,
        consumers=_JPEG_CONSUMERS,
        note=(
            "Deliverable 2, suspect 2. Auto-gamma to a mean V of 20 followed by "
            "unclipped CLAHE amplifies whatever noise the lift produced."
        ),
    ),
    # ---------------- diagnostics: never proposals ----------------
    "wb-linear": Experiment(
        build=lambda: _wb_arms(Stage.LINEAR),
        consumers=("laser detection", "classify_laser_color"),
        diagnostic_only=True,
        note=(
            "Measures what the JPEG-stage constraint costs, by doing the thing "
            "it forbids. White balance is the only Deliverable-2 knob the "
            "detector's scale-invariant chromaticity channels can see. Running "
            "this is not a proposal to ship it: it would make laser detection "
            "depend on enhancement, and the checkpoint was trained on "
            "use_camera_wb=True, so it is also a covariate shift."
        ),
    ),
    "insulation-control": Experiment(
        build=lambda: [
            Arm("baseline", Stage.LINEAR),
            Arm("jpeg-knobs-only", Stage.LINEAR,
                decode=DecodeConfig(auto_gamma_target=60, clahe_enabled=False)),
        ],
        consumers=("laser detection",),
        diagnostic_only=True,
        note=(
            "The proof of non-dependency. Runs the laser detector across a "
            "JPEG-chain-only change; every delta must be exactly zero. A "
            "non-zero result means the harness is miswired, not that "
            "enhancement helped."
        ),
    ),
}

#: Experiments whose results could justify a pipeline change. Every one of
#: these is asserted JPEG-stage-only by `tests/test_no_laser_dependency.py`.
SHIPPING_CANDIDATES = tuple(
    name for name, exp in EXPERIMENTS.items() if not exp.diagnostic_only
)


def build_arms(name: str) -> list[Arm]:
    """Materialize an experiment's arms, enforcing the laser guarantee."""
    if name not in EXPERIMENTS:
        raise SystemExit(
            f"unknown experiment {name!r}; choose from: {', '.join(sorted(EXPERIMENTS))}"
        )
    experiment = EXPERIMENTS[name]
    arms = experiment.build()
    if not experiment.diagnostic_only:
        assert_no_laser_dependency(arms)
    return arms
