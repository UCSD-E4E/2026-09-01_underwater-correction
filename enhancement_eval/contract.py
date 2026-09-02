"""What an enhancement function is, and what it is not allowed to do.

Hard constraint #1: a shipped enhancer is strictly pixel-wise. Measurements
in this pipeline come from pixel *coordinates*, not pixel values -- Label
Studio percentages resolve through `original_width/height` into rectified
pixels, which stage 14 back-projects against the laser ray. A value-only
enhancer is therefore geometrically free. A generator that shifts an edge by
one pixel is not: 1 px of laser-dot error is 0.75% length error, and head/tail
keypoints are clicked on edges.

That constraint is mechanically checkable, so `guard` checks it on every call
rather than trusting each enhancer's author. It is cheap (two attribute reads)
and it fires at the moment of the violation, naming the enhancer, instead of
surfacing as an unexplained bias in the report.

What is deliberately *not* forbidden is value locality. CLAHE reads a
neighbourhood and CLAHE is a live candidate. "Pixel-wise" means no pixel
moves, not that no pixel may consult its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

#: An enhancement function. Takes a BGR array, returns one of identical shape
#: and dtype. Called with uint8 at the rectified/JPEG stage and uint16 at the
#: linear-raw stage, so an enhancer that cares must branch on `dtype`.
Enhancer = Callable[[np.ndarray], np.ndarray]

__all__ = ["Enhancer", "GeometryViolation", "guard", "probe_geometry", "ProbeResult", "IDENTITY"]


class GeometryViolation(RuntimeError):
    """An enhancer changed the frame's geometry or dtype.

    Raised rather than warned: a geometry change invalidates every coordinate
    comparison the harness is about to make, so continuing would produce
    numbers that look fine and mean nothing.
    """


def IDENTITY(image: np.ndarray) -> np.ndarray:  # noqa: N802 - a sentinel, not a class
    """The baseline arm. Named so reports can print something better than
    `<lambda>`."""
    return image


def guard(enhancer: Enhancer, *, name: str) -> Enhancer:
    """Wrap `enhancer` so it cannot silently change geometry or dtype.

    Also defends the *baseline* arm: both arms score from the same decoded
    array, so an enhancer that mutates its input in place would corrupt the
    control. That failure mode reads as the enhancement helping, which is the
    single most expensive way for this harness to be wrong, so the input is
    passed as a read-only view and restored if the enhancer writes through a
    copy of its own.
    """

    def _guarded(image: np.ndarray) -> np.ndarray:
        source = image
        # A read-only view makes an in-place write raise instead of corrupting
        # the control arm. Enhancers that legitimately need to write must copy
        # first, which is the correct discipline anyway. numpy's own error for
        # this is a bare ValueError from deep inside the enhancer, so it is
        # translated here into the rule that was actually broken.
        view = source.view()
        view.flags.writeable = False
        try:
            out = enhancer(view)
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise
            raise GeometryViolation(
                f"enhancer {name!r} wrote to its input in place; both arms score "
                "from the same decoded array, so this corrupts the baseline "
                "control and reads as the enhancement helping. Copy first."
            ) from exc

        if not isinstance(out, np.ndarray):
            raise GeometryViolation(
                f"enhancer {name!r} returned {type(out).__name__}, not an ndarray"
            )
        if out.shape != source.shape:
            raise GeometryViolation(
                f"enhancer {name!r} changed shape {source.shape} -> {out.shape}; "
                "enhancement must be strictly pixel-wise (no warp, resize, crop, "
                "transpose, or channel change) because every measurement in this "
                "pipeline is a pixel coordinate"
            )
        if out.dtype != source.dtype:
            raise GeometryViolation(
                f"enhancer {name!r} changed dtype {source.dtype} -> {out.dtype}; "
                "the container's full scale is load-bearing (chromaticity_norm "
                "picks 255 vs 65535 off it, and cv2.imencode needs uint8)"
            )

        # Hand the consumer a writeable array it can safely own.
        #
        # This is not tidiness -- it is a segfault. The read-only view above is
        # what lets `guard` catch an in-place enhancer, but an enhancer that
        # legitimately returns its input unchanged (IDENTITY, i.e. every
        # baseline arm) returns that same read-only view. OpenCV does not
        # respect numpy's WRITEABLE flag and writes into the buffer anyway, so
        # the frame goes on to crash the process inside `LaserDetector.predict`
        # -- verified: passing a read-only uint16 frame to the detector dumps
        # core. Copying here costs one array per call on the arms that alias
        # their input, and nothing on the arms that already allocate.
        if not out.flags.writeable or out.base is source or out.base is view:
            out = out.copy()
        return out

    _guarded.__name__ = f"guarded[{name}]"
    return _guarded


# --------------------------------------------------------------------------
# The locality probe
# --------------------------------------------------------------------------

#: Displacement above which the probe refuses the enhancer outright, in pixels.
#: Calibrated against measurement, not chosen: over identity, a halve, a 5px
#: and a 31px Gaussian, and CLAHE, the measured displacement is <= 0.074 px --
#: a symmetric neighbourhood filter preserves the response centroid however
#: wide it is. A one-pixel roll measures 1.01, a square transpose 7.07, a
#: horizontal flip 20.99. Nothing legitimate observed lands between 0.1 and 1.0.
MAX_DISPLACEMENT_PX = 0.5

#: Displacement above which the probe reports the enhancer as suspicious but
#: still allows it. Resample round trips (downsample and back) land here --
#: 0.07 px at 0.5x, 0.16 at 0.33x, 0.34 at 0.7x. Those are a loss of detail
#: rather than a displacement of it, so they are surfaced rather than refused,
#: and the number goes in the report next to the enhancer's name.
WARN_DISPLACEMENT_PX = 0.1


@dataclass(frozen=True)
class ProbeResult:
    """What the locality probe measured. Carried into the report so a reader
    can see the geometric assurance behind each arm's numbers."""

    name: str
    displacement_px: float
    suspicious: bool


def probe_geometry(
    enhancer: Enhancer,
    *,
    name: str,
    max_displacement_px: float = MAX_DISPLACEMENT_PX,
    size: int = 64,
    dtype: "np.dtype | type" = np.uint8,
) -> ProbeResult:
    """Assert `enhancer` does not move pixels. Run once per enhancer, not per frame.

    `guard` compares shape and dtype, which catches a resize, a crop, and a
    transpose of a non-square frame. It structurally cannot catch the three
    that matter most -- a square-frame transpose, a one-pixel roll, and a
    resample round trip -- all of which preserve shape and dtype exactly while
    moving labelled coordinates. One pixel of laser-dot displacement is 0.75%
    length error, so "the shape is right" is not enough assurance to put
    something in front of this pipeline.

    The probe perturbs a small off-centre patch of an otherwise textured input
    and measures where the output changed in response. A value-only or
    symmetric-neighbourhood enhancer answers at the perturbation; a warp
    answers somewhere else. It works for any enhancer whose output depends on
    its input, and it is indifferent to what the enhancer does to values --
    only to where the consequences land.

    What it does *not* measure is resampling loss. A downsample-and-back blurs
    without displacing, so it passes with a warning rather than an error; if
    detail matters for the consumer being scored, that shows up in that
    consumer's numbers, which is the harness's whole point.

    Returns:
        ProbeResult, whose `displacement_px` belongs in the report.

    Raises:
        GeometryViolation: if the response is displaced past
            `max_displacement_px`.
    """
    rng = np.random.default_rng(0xF15E)
    info = np.iinfo(dtype)
    # Mid-grey with light texture. A perfectly flat field makes some enhancers
    # degenerate (a histogram method has nothing to stretch) and leaves a warp
    # no gradient to reveal itself against.
    base = rng.integers(
        int(info.max * 0.40), int(info.max * 0.60), size=(size, size, 3)
    ).astype(dtype)

    # Off-centre, and off both diagonals: a centred patch survives a transpose
    # and a 180-degree rotation, and a patch on the main diagonal survives a
    # transpose. Either placement would let a real warp pass.
    cy, cx, half = size // 4, size // 3, 3
    perturbed = base.copy()
    perturbed[cy - half : cy + half + 1, cx - half : cx + half + 1] = info.max

    guarded = guard(enhancer, name=name)
    before = guarded(base).astype(np.int64)
    after = guarded(perturbed).astype(np.int64)

    response = np.abs(after - before).sum(axis=2)
    peak = float(response.max())
    if peak == 0:
        # The enhancer's output does not depend on its input here -- a constant
        # or a no-op on this synthetic field. Nothing to conclude about
        # geometry, and not this function's business to complain.
        return ProbeResult(name=name, displacement_px=0.0, suspicious=False)

    # Magnitude-weighted centroid of the response, against where the
    # perturbation actually was. The 10% floor drops the far tail of a wide
    # kernel, which is noise-dominated and would jitter the centroid.
    strong = response >= peak * 0.10
    ys, xs = np.nonzero(strong)
    weights = response[ys, xs].astype(float)
    got_y = float((ys * weights).sum() / weights.sum())
    got_x = float((xs * weights).sum() / weights.sum())

    displacement = float(np.hypot(got_y - cy, got_x - cx))
    if displacement > max_displacement_px:
        raise GeometryViolation(
            f"enhancer {name!r} moved pixels: perturbing the input at "
            f"({cx}, {cy}) changed the output around ({got_x:.2f}, {got_y:.2f}), "
            f"a displacement of {displacement:.2f} px (limit "
            f"{max_displacement_px}). Enhancement must be strictly pixel-wise -- "
            "1 px of laser-dot displacement is 0.75% length error, and head/tail "
            "keypoints are clicked on edges."
        )
    return ProbeResult(
        name=name,
        displacement_px=displacement,
        suspicious=displacement > WARN_DISPLACEMENT_PX,
    )
