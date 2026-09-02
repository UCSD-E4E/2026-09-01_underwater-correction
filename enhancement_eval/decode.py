"""Decode variants: Deliverable 2's one-line experiments, made measurable.

The production chain, in `fishsense_core.image.raw_image.RawImage`, is::

    rawpy.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
                      output_bps=16, user_flip=0)
      -> auto-gamma targeting a mean V of 20
      -> skimage.equalize_adapthist() at defaults
      -> (RectifiedImage) cv2.undistort

and the laser detector's chain, in `LinearRawImage`, is the same `postprocess`
call with no gamma and no CLAHE, kept in sensor coordinates.

Two knobs are worth a controlled experiment before anything learned is
considered, and this module makes both switchable while holding everything
else fixed:

* **White balance.** `use_camera_wb=True` is the camera's *topside daylight*
  white balance, applied to a scene lit through several metres of water. It is
  the most likely single cause of red-channel loss, and it is a four-number
  change.
* **Auto-gamma then unclipped CLAHE.** Lifting a dark frame to a mean V of 20
  and then running `equalize_adapthist` at defaults amplifies whatever noise
  the lift produced, locally and without a ceiling.

**The baseline arm must reproduce `fishsense_core` exactly.** Every number the
harness reports is a difference against it, so a baseline that merely resembles
production makes every delta the sum of the effect being studied and an
uncontrolled reimplementation error. `tests/test_decode_parity.py` asserts
byte-for-byte equality against both `RawImage` and `LinearRawImage` on the real
`.ORF` fixture, and that test is the reason to trust anything downstream.

One asymmetry to keep in mind while reading results: white balance is applied
inside `postprocess`, so it moves BOTH stages -- the labeler-facing JPEG *and*
the laser detector's linear input. Gamma and CLAHE exist only in the JPEG
chain. See `stages.py`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "WhiteBalance",
    "DecodeConfig",
    "gray_world_gains",
    "white_patch_gains",
    "slate_patch_gains",
    "normalize_gains",
]

Gains = Tuple[float, float, float]


class WhiteBalance(enum.Enum):
    """Where the white point comes from.

    `CAMERA` is production. The rest are the Deliverable-2 candidates, in
    rough order of how much scene knowledge they assume.
    """

    #: As-shot camera WB -- a topside daylight preset. Production.
    CAMERA = "camera"
    #: Scene averages to neutral. Cheap, and wrong underwater in a specific
    #: way: the water column genuinely is blue-green, so gray-world reads the
    #: water as a cast and over-corrects red. Measured, not assumed.
    GRAY_WORLD = "grayworld"
    #: Brightest percentile is neutral. Better than gray-world when a genuinely
    #: white object is in frame, and the dive slate usually is one.
    WHITE_PATCH = "whitepatch"
    #: Fitted from the dive slate -- a physical, in-scene, printed-white
    #: reference at a known place in the frame. The most defensible of the
    #: four, and the only one anchored to something real.
    SLATE = "slate"
    #: rawpy's own auto WB, for reference. Included because it is free and
    #: because "we tried the obvious library flag" is a question that will be
    #: asked.
    RAWPY_AUTO = "rawpyauto"


def normalize_gains(gains: Gains) -> Gains:
    """Scale gains so green is 1.0.

    rawpy's `user_wb` multiplies raw channel values, so a common factor across
    all three is an *exposure* change, not a white-balance change. Pinning
    green separates the two: without it, a "white balance" experiment also
    changes overall brightness, the auto-gamma stage then partly compensates,
    and no observed difference can be attributed to either cause.

    Green is the reference because it has twice the photosites of red or blue
    and therefore the least noise.
    """
    red, green, blue = gains
    if not green or not math.isfinite(green):
        return (1.0, 1.0, 1.0)
    return (red / green, 1.0, blue / green)


def _channel_stat(image: np.ndarray, reducer) -> Gains:
    """Reduce a BGR array to per-channel (R, G, B) statistics."""
    blue = float(reducer(image[:, :, 0]))
    green = float(reducer(image[:, :, 1]))
    red = float(reducer(image[:, :, 2]))
    return red, green, blue


def _gains_from_levels(levels: Gains) -> Gains:
    """Turn per-channel levels into gains that equalize them.

    A channel measuring zero is a broken decode, not a licence to return `inf`
    and blow the frame out -- it is passed through at unity gain so the
    resulting frame is visibly wrong rather than invisibly saturated.
    """
    red, green, blue = levels
    reference = green if green > 0 else max(red, blue, 1.0)
    out = []
    for level in (red, green, blue):
        out.append(reference / level if level > 0 else 1.0)
    return normalize_gains((out[0], out[1], out[2]))


def gray_world_gains(image: np.ndarray) -> Gains:
    """Gains that equalize the per-channel means. `image` is BGR."""
    return _gains_from_levels(_channel_stat(image, np.mean))


def white_patch_gains(image: np.ndarray, *, percentile: float = 99.0) -> Gains:
    """Gains that equalize the bright tail of each channel. `image` is BGR.

    A percentile rather than the maximum: one hot pixel or one specular glint
    off a fin must not set the white point for the whole frame.
    """
    return _gains_from_levels(
        _channel_stat(image, lambda plane: np.percentile(plane, percentile))
    )


def slate_patch_gains(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    percentile: float = 90.0,
) -> Gains:
    """Gains fitted from the dive slate's printed white. `image` is BGR.

    The dive slate is a physical reference target present in many dives, at a
    known place in the frame (`DiveSlateLabel.slate_rectangle`), lit by the
    same water column as everything else. That makes it the only white
    reference here that is anchored to a real object rather than to an
    assumption about scene statistics.

    The bright tail *within the slate quad* is used rather than the quad's
    mean, because the slate is white paper carrying black markings -- the mean
    is a paper/ink mixture that depends on how much of the artwork is in view,
    while the bright tail is the paper.

    Note what this does and does not assume. It assumes the slate's paper is
    spectrally flat (a reasonable assumption for white paper, and the reason to
    prefer it over the artwork), and that the slate is not blown out. It does
    NOT assume anything about the water column, which is the point.
    """
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)], 255)
    inside = mask.astype(bool)
    if inside.sum() < 16:
        raise ValueError(
            f"slate quad covers only {int(inside.sum())} px; too small to fit a "
            "white balance from"
        )

    levels = []
    for channel in (2, 1, 0):  # R, G, B
        plane = image[:, :, channel][inside]
        levels.append(float(np.percentile(plane, percentile)))
    return _gains_from_levels((levels[0], levels[1], levels[2]))


@dataclass(frozen=True)
class DecodeConfig:
    """One decode arm. Defaults reproduce production exactly.

    Frozen and hashable so it can key the per-image result cache: decoding a
    15 MB `.ORF` dominates the run, and a report has to be re-runnable without
    paying for it again.
    """

    #: Where the white point comes from. Affects BOTH the JPEG and the laser
    #: detector's linear input, because it lives inside `rawpy.postprocess`.
    white_balance: WhiteBalance = WhiteBalance.CAMERA

    #: Target mean V for the auto-gamma lift, in 0-255. Production is 20 --
    #: deliberately dark, and the reason the subsequent CLAHE has so much
    #: noise to amplify. JPEG chain only.
    auto_gamma_target: int = 20

    #: JPEG chain only.
    clahe_enabled: bool = True
    #: How CLAHE is applied across colour channels.
    #:
    #: ``"per_channel"`` is production, and is production by accident rather
    #: than by choice: `skimage.exposure.equalize_adapthist` has no RGB branch,
    #: so it treats an ``HxWx3`` array as a 3-D volume and defaults
    #: ``kernel_size`` to ``(H//8, W//8, max(3//8, 1))``. That trailing 1 is a
    #: tile depth of one across the channel axis, i.e. each channel's histogram
    #: is stretched to full range independently.
    #:
    #: Underwater that is the worst available allocation of gain. Red is the
    #: most attenuated channel, so it has the narrowest and noisiest histogram,
    #: so it receives the *largest* stretch. On a synthetic flat water patch
    #: this raises luminance SD from 0.0023 to 0.1621 -- about 70x -- and
    #: decorrelates the channels into chroma speckle.
    #:
    #: ``"luminance"`` equalizes CIELAB L* alone and leaves a* and b* untouched,
    #: which is the conventional way to run CLAHE on a colour image: the local
    #: contrast that helps a labeler see a fish outline is a luminance property,
    #: and nothing is gained by rewriting hue to get it.
    clahe_mode: str = "per_channel"
    #: `None` means skimage's default (0.01). Lower clips harder and amplifies
    #: less noise.
    clahe_clip_limit: float | None = None
    #: `None` means skimage's default (1/8 of each axis). JPEG chain only.
    clahe_kernel_size: int | None = None

    #: Global contrast stretch, applied after the auto-gamma lift and before
    #: CLAHE. ``"off"`` is production.
    #:
    #: This is the tool the underwater problem actually calls for. Attenuation
    #: and backscatter leave the scene occupying a narrow slice of the
    #: container's range -- which is why everything reads flat -- and backscatter
    #: in particular is roughly *additive*, a veiling floor over the whole
    #: frame. Subtracting a black point and rescaling removes that floor; a
    #: gain-only white balance cannot, because it has no offset term.
    #:
    #: Unlike CLAHE it is ONE affine map over the whole frame, so a flat region
    #: stays flat relative to the scene instead of being handed its own local
    #: histogram and stretched on its own. That is the difference between
    #: recovering contrast and manufacturing it out of sensor noise.
    #:
    #: ``"per_channel"`` maps each channel independently, which also removes the
    #: colour cast (a white balance *with* a black point). ``"luminance"`` maps
    #: only CIELAB L*, expanding contrast while leaving the cast alone -- two
    #: separate decisions, kept separable.
    stretch_mode: str = "off"
    #: Percentile mapped to black. Percentiles rather than min/max so one hot
    #: pixel or one specular glint cannot set the whole frame's mapping.
    #:
    #: Accepts a single value for all channels, or an ``(R, G, B)`` triple.
    #: The triple exists because underwater the three channels are in
    #: completely different states: red is a narrow, noise-dominated band while
    #: blue is broad, so one pair of percentiles for all three is the wrong
    #: shape of knob. Anchoring red's black point *above* its noise floor is
    #: what lets a laser dot survive a stretch that discards the speckle.
    stretch_low: float | tuple[float, float, float] = 1.0
    #: Percentile mapped to white. Same single-or-triple rule.
    stretch_high: float | tuple[float, float, float] = 99.0

    #: Additive gain applied to red where the *spatially coherent* red excess
    #: is high. 0.0 is off.
    #:
    #: This synthesises emphasis rather than restoring signal, which makes it a
    #: different kind of thing from every other knob here. Legitimate as a
    #: target indicator on species / head-tail frames, where the laser marks
    #: which fish is being measured; circular if it were ever pointed at the
    #: laser-labeling task itself.
    red_boost: float = 0.0
    #: Local red-excess *contrast* above which the boost starts to apply.
    #:
    #: Deliberately a local contrast, not an absolute level. In the production
    #: decode the laser dot's absolute red excess `R - max(G, B)` is still
    #: **negative** (about -0.07, against -0.49 in the surrounding water),
    #: because the whole scene is cyan -- so any absolute threshold either never
    #: fires or fires everywhere. What marks the dot is being redder than its
    #: neighbourhood.
    red_boost_threshold: float = 0.02
    #: Blur applied to the red-excess map before thresholding. This is the part
    #: that separates a coherent dot from pixel-scale speckle.
    red_boost_sigma: float = 1.5
    #: Radius of the local baseline the dot is measured against. Together with
    #: `red_boost_sigma` this is a difference-of-Gaussians blob detector on the
    #: red-excess map: small blur keeps the dot, large blur estimates the water
    #: around it, and the difference is what the ramp thresholds.
    red_boost_background_sigma: float = 12.0
    #: Threshold expressed in robust sigmas of the blob response itself. When
    #: non-zero this replaces the absolute `red_boost_threshold`.
    #:
    #: A fixed level cannot work across these scenes. On the showcase frames
    #: the laser dot sits at the 99.999th percentile of the red channel on a
    #: reef -- it is the reddest thing there -- and at the **58.8th** in a pool,
    #: where a white shirt, skin and a painted model are all redder. Referencing
    #: the cut to the response's own noise makes it scene-adaptive: the dot
    #: measures tens of sigmas above the local baseline in both.
    red_boost_sigmas: float = 0.0
    #: Red-excess span, above the threshold, over which the boost ramps to
    #: full. Set from the quantity's real range, not from 1.0: red excess
    #: `R - max(G, B)` on these frames runs about -0.4 in open water to +0.1 at
    #: a laser dot, so normalising over [threshold, 1.0] leaves the ramp
    #: permanently near zero and the boost does nothing.
    red_boost_span: float = 0.05

    #: Optional denoise, applied to the linear decode before the auto-gamma
    #: lift ("pre") or at the end of the chain ("post"). "off" is production.
    #:
    #: Kept opt-in rather than folded into the recommendation because it is a
    #: real trade, not a free win: on these frames the fish's scale texture
    #: measures only ~1.6x the open-water grain *in the same frequency band*,
    #: so any single-frame filter strong enough to remove the grain takes most
    #: of the texture with it. That is an information limit, not a tuning
    #: failure, and it is why the choice belongs to whoever is looking at the
    #: frames rather than to a default.
    denoise: str = "off"
    #: Strength for the total-variation denoiser.
    denoise_weight: float = 0.02

    #: Percentile for WHITE_PATCH and SLATE gain estimation.
    wb_percentile: float = 99.0

    #: Estimate WB gains from a half-resolution decode. Gains are a global
    #: statistic, so the half-size pass costs about a quarter of the time and
    #: changes them negligibly; the full decode then runs once with the
    #: resulting `user_wb`. Turn off to prove that to yourself.
    wb_estimate_half_size: bool = True

    def __post_init__(self) -> None:
        if self.auto_gamma_target <= 0:
            raise ValueError(
                f"auto_gamma_target must be positive, got {self.auto_gamma_target}: "
                "it is compared in log space, so a non-positive target makes "
                "math.log raise inside the decode"
            )
        if self.clahe_clip_limit is not None and not 0 < self.clahe_clip_limit <= 1:
            raise ValueError(
                f"clahe_clip_limit must be in (0, 1], got {self.clahe_clip_limit}"
            )
        if self.clahe_kernel_size is not None and self.clahe_kernel_size < 2:
            raise ValueError(
                f"clahe_kernel_size must be at least 2, got {self.clahe_kernel_size}"
            )
        if self.stretch_mode not in ("off", "per_channel", "luminance"):
            raise ValueError(
                f"stretch_mode must be 'off', 'per_channel' or 'luminance', "
                f"got {self.stretch_mode!r}"
            )
        lows = self.stretch_low if isinstance(self.stretch_low, tuple) else (self.stretch_low,)
        highs = self.stretch_high if isinstance(self.stretch_high, tuple) else (self.stretch_high,)
        for name, value in (("stretch_low", self.stretch_low), ("stretch_high", self.stretch_high)):
            if isinstance(value, tuple) and len(value) != 3:
                raise ValueError(
                    f"{name} must be a number or an (R, G, B) triple, got {value!r}"
                )
        if len(lows) != len(highs) and 1 not in (len(lows), len(highs)):
            raise ValueError("stretch_low and stretch_high must broadcast together")
        for low, high in zip(
            lows * (3 if len(lows) == 1 else 1), highs * (3 if len(highs) == 1 else 1)
        ):
            if not 0.0 <= low < high <= 100.0:
                raise ValueError(
                    f"stretch percentiles must satisfy 0 <= low < high <= 100, got "
                    f"low={low}, high={high}"
                )
        if self.red_boost < 0:
            raise ValueError(f"red_boost must be >= 0, got {self.red_boost}")
        if self.denoise not in ("off", "tv-pre", "tv-post"):
            raise ValueError(
                f"denoise must be 'off', 'tv-pre' or 'tv-post', got {self.denoise!r}"
            )
        if self.red_boost_sigmas < 0:
            raise ValueError(
                f"red_boost_sigmas must be >= 0, got {self.red_boost_sigmas}"
            )
        if self.clahe_mode not in ("per_channel", "luminance"):
            raise ValueError(
                f"clahe_mode must be 'per_channel' or 'luminance', "
                f"got {self.clahe_mode!r}"
            )

    @property
    def label(self) -> str:
        """Short, stable, filename-safe name for reports and cache keys."""
        parts = []
        if self.white_balance is not WhiteBalance.CAMERA:
            parts.append(self.white_balance.value)
        if self.auto_gamma_target != 20:
            parts.append(f"gamma{self.auto_gamma_target}")
        if not self.clahe_enabled:
            parts.append("noclahe")
        if self.stretch_mode != "off":
            fmt = lambda v: (  # noqa: E731
                "/".join(f"{x:g}" for x in v) if isinstance(v, tuple) else f"{v:g}"
            )
            parts.append(
                f"stretch{self.stretch_mode[0]}{fmt(self.stretch_low)}-{fmt(self.stretch_high)}"
            )
        if self.denoise != "off":
            parts.append(f"{self.denoise}{self.denoise_weight:g}")
        if self.red_boost:
            parts.append(
                f"redboost{self.red_boost:g}"
                + (f"@{self.red_boost_sigmas:g}s" if self.red_boost_sigmas else "")
            )
        if self.clahe_mode != "per_channel":
            parts.append("lumaclahe" if self.clahe_mode == "luminance" else self.clahe_mode)
        if self.clahe_clip_limit is not None:
            parts.append(f"clip{self.clahe_clip_limit:g}")
        if self.clahe_kernel_size is not None:
            parts.append(f"kernel{self.clahe_kernel_size}")
        return "-".join(parts) if parts else "baseline"
