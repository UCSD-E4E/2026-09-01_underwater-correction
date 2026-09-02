"""Per-image scoring primitives.

Kept pure and separate from the model calls, so the decisions that shape a
result are visible and testable without a GPU: which human dot a prediction is
compared against, what counts as coverage rather than accuracy, and what
"quality" means when the report stratifies by it.

The single most important rule here is that **coverage and accuracy never
mix**. A non-detection returns `None`, not a large error. Folding a miss in as
`inf` (or as the frame diagonal, which is the tempting version) lets a coverage
collapse masquerade as an accuracy result, and the median hides it completely:
an arm that detects nothing on the hardest 30% of frames and is slightly better
on the rest would report a *better* median error than the baseline while being
strictly worse. `metrics.coverage_flips` carries the coverage story separately.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "nearest_error",
    "image_quality",
    "laser_color_agreement",
    "slate_point_errors",
]


def nearest_error(
    prediction: tuple[float, float] | None,
    human_points: Sequence[tuple[float, float]],
) -> float | None:
    """Distance from a prediction to the closest human laser dot, in pixels.

    `None` when either side is absent -- see the module docstring on why that
    is not an error value.

    *Nearest* rather than first, because 461 prod images carry two valid laser
    labels and there is no basis in the data for deciding which is "the" dot.
    Picking arbitrarily would report a real hit against the other label as a
    large miss. The cost is that this is slightly generous -- but it is
    generous to both arms identically, and the harness reports differences.
    """
    if prediction is None or not human_points:
        return None
    px, py = prediction
    return min(math.hypot(px - hx, py - hy) for hx, hy in human_points)


def image_quality(bgr: np.ndarray) -> dict[str, float]:
    """Cheap descriptive statistics for one frame, normalized to 0-1.

    These are **not** quality metrics in the UIQM/PSNR sense and must not be
    reported as evidence that an enhancement worked -- the entire premise of
    this project is that a perceptual score going up says nothing about whether
    the pipeline got better. They exist for one purpose: to give the report an
    axis to sort inputs along, because the claim worth testing is that
    enhancement helps degraded frames and *degrades* good ones, and a single
    corpus mean over both populations describes neither.

    Always computed on the BASELINE frame, so both arms are stratified by the
    same number and a bucket means the same thing on each side.

    Scale-normalized so a uint16 linear frame and a uint8 JPEG are comparable;
    otherwise every linear frame would sort into the "good" quartile purely
    because its container is larger.
    """
    if np.issubdtype(bgr.dtype, np.integer):
        full_scale = float(np.iinfo(bgr.dtype).max)
    else:
        full_scale = 1.0
    data = bgr.astype(np.float64) / full_scale

    blue, green, red = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    luminance = data.mean(axis=2)
    green_mean = float(green.mean())

    return {
        "mean_luminance": float(luminance.mean()),
        # Population standard deviation of luminance -- the standard RMS
        # contrast. Low means a flat, washed-out frame, which is what
        # attenuation and backscatter produce.
        "rms_contrast": float(luminance.std()),
        "mean_red": float(red.mean()),
        "mean_green": green_mean,
        "mean_blue": float(blue.mean()),
        # The attenuation signature: red falls off fastest underwater, so a low
        # ratio is the direct symptom and the number a white-balance change is
        # supposed to move. Guarded against an all-black frame.
        "red_green_ratio": (float(red.mean()) / green_mean) if green_mean > 0 else 0.0,
        # Fraction of pixels at the container ceiling. A "successful"
        # enhancement that simply clips the highlights shows up here rather
        # than in a contrast number that would call it an improvement.
        "clipped_fraction": float((data.max(axis=2) >= 0.999).mean()),
    }


def laser_color_agreement(baseline: str | None, enhanced: str | None) -> str:
    """Compare `classify_laser_color` verdicts across the two arms.

    Four outcomes, kept distinct because they carry different costs:

    * `agree` -- unchanged.
    * `abstained` -- the enhancement destroyed a usable opinion. Costs a vote.
    * `gained` -- the enhancement made an undecidable dot decidable. A real
      benefit, and it would be dishonest to fold it into `agree`.
    * `flipped` -- the enhancement produced the *opposite* colour. This is the
      expensive one, and the reason the tripwire exists at all: populate takes
      the dive-level majority, so enough flips silently relabel a whole dive's
      laser colour, and the pre-annotation then points labelers at the wrong
      thing for every frame in it.

    Note the baseline verdict is treated as the reference, not as truth -- it
    is 98.19% accurate over 332 labelled dots, not 100%. A flip is evidence the
    enhancement moved the discriminator, which is the question, but a small
    number of flips may be the baseline's own errors being corrected.
    """
    if baseline is None and enhanced is None:
        return "both_abstained"
    if baseline is None:
        return "gained"
    if enhanced is None:
        return "abstained"
    return "agree" if baseline == enhanced else "flipped"


def slate_point_errors(
    predicted: Sequence[tuple[float, float]] | None,
    human: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Per-point error of a board estimate against the human slate label.

    Both sides are in photo-frame pixels: `BoardEstimate.image_points` lands
    there directly, and the sync activity shifts the LS composite's PDF-panel
    width off `DiveSlateLabel.reference_points` before persisting them. So they
    are comparable without any further transform.

    **Paired positionally, never by nearest neighbour.** Stage 13 consumes
    these points by position, and the estimator's characteristic failure is
    resolving the board's 4-fold corner ambiguity wrong -- which produces
    exactly the same *set* of points in the wrong order. Nearest-neighbour
    matching would score that catastrophe as a perfect fit. Positional pairing
    reports it as the large error it is.

    A point-count mismatch abstains rather than zipping to the shorter list: a
    partial set already breaks stage-13's pairing, and truncating would score a
    wrong-template fit on whichever points happened to line up.
    """
    if predicted is None:
        return {"status": "no_estimate", "median_px": None, "max_px": None,
                "n_points": 0}
    if not len(human):
        return {"status": "no_human_label", "median_px": None, "max_px": None,
                "n_points": 0}
    if len(predicted) != len(human):
        return {
            "status": "point_count_mismatch",
            "median_px": None,
            "max_px": None,
            "n_points": 0,
            "n_predicted": len(predicted),
            "n_human": len(human),
        }

    errors = [
        math.hypot(float(px) - float(hx), float(py) - float(hy))
        for (px, py), (hx, hy) in zip(predicted, human)
    ]
    ordered = sorted(errors)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "status": "scored",
        "median_px": median,
        "max_px": max(errors),
        "mean_px": sum(errors) / len(errors),
        "n_points": len(errors),
    }
