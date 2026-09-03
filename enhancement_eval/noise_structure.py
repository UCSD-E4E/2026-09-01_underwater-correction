"""Does this sensor's noise satisfy what Noise2Void assumes?

Noise2Void (Krull et al. 2019) trains a blind-spot network: it hides a pixel
and predicts it from the surrounding neighbourhood, with the noisy pixel itself
as the target. That converges on the clean signal only because of two
properties of the noise, and neither is guaranteed:

  1. **Spatial independence given the signal.** If neighbouring photosites
     share a noise component, the network reads that shared part as signal and
     reproduces it. The failure is silent -- the loss curve looks healthy and
     the output looks denoised, because the *independent* part still goes.
  2. **Zero mean per photosite.** A fixed-pattern offset is the same in every
     frame, so a blind-spot network cannot distinguish it from scene content
     and cannot remove it.

This module measures both before a model exists. It is the same rule the rest
of the project runs on -- build the metric first -- applied to the one learned
method worth trying here, and it is cheap enough that skipping it would be
indefensible: a few frames against days of training.

**How the noise is isolated.** Reusing the G1-G2 identity behind
`mosaic.green_noise_sigma`: the two green photosites of a Bayer quad sample
nearly the same scene point, so their difference cancels the signal and leaves
noise. No assumption is needed about what fraction of the frame is flat, and it
cannot collapse on a dark or quantised channel the way a median-absolute-
deviation does.

**The caveat that decides where to sample.** G1 and G2 sit at different
positions, so on high-frequency scene content their difference is not purely
noise -- real detail leaks in and reads as spatial correlation. Sampling open
water rather than the whole frame keeps scene structure out of the estimate,
which is the same open-water convention as every other noise figure in
MEASUREMENTS.md.

**Why this is measured on the mosaic, not on RGB.** Per Chris, and measured
earlier in this project: by the time there is an image, demosaicing has already
mixed each photosite into its neighbours, taking the cross-channel noise
correlation from 0.029 to 0.328. Any independence assumption tested after
demosaicing would be testing the interpolator, not the sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_LAGS",
    "FIXED_PATTERN_MAX",
    "NOISE_INDEPENDENCE_MAX",
    "NoiseStructure",
    "assess_n2v_viability",
    "fixed_pattern_fraction",
    "noise_field",
    "recommended_mask_width",
    "spatial_autocorrelation",
]

#: Displacements probed, in *plane* pixels. A plane is half resolution, so a
#: lag of 1 here is two photosites on the sensor -- which is the spacing that
#: matters, because a per-plane blind-spot network predicts from same-colour
#: neighbours.
DEFAULT_LAGS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0),
)

#: Above this correlation at any non-zero lag, a blind-spot network can predict
#: a meaningful part of a pixel's noise from its neighbours, and so preserves
#: it. Chosen as the point where the recoverable fraction stops being
#: negligible rather than from any result: at r = 0.1 the neighbourhood
#: explains about 1% of the noise variance.
NOISE_INDEPENDENCE_MAX = 0.10

#: Correlations below this are at the sampling noise floor for the frame sizes
#: used here, and ratios formed from them carry no direction.
MEANINGFUL_CORRELATION = 0.01

#: Above this ratio of horizontal to vertical lag-1 correlation, the noise is
#: directional and a single-pixel blind spot is the wrong shape: the correlated
#: neighbour stays visible for the network to copy from. StructN2V's answer is
#: to mask a line segment along the correlated axis instead.
ANISOTROPY_LIMIT = 2.0

#: Above this share of noise variance sitting at fixed photosite positions,
#: N2V is addressing a minority of the problem and a dark-frame or flat-field
#: correction is the right tool instead.
FIXED_PATTERN_MAX = 0.30


def noise_field(planes: dict[str, np.ndarray]) -> np.ndarray:
    """Signal-free noise, at per-photosite scale.

    ``(G1 - G2) / sqrt(2)``: the difference cancels the shared signal, and the
    sqrt(2) undoes the variance doubling so the result reads back as a single
    photosite's sigma rather than the difference's.
    """
    return (np.asarray(planes["G1"], dtype=np.float64)
            - np.asarray(planes["G2"], dtype=np.float64)) / np.sqrt(2.0)


def spatial_autocorrelation(
    field: np.ndarray, lags: tuple[tuple[int, int], ...] = DEFAULT_LAGS
) -> dict[tuple[int, int], float]:
    """Normalized autocorrelation of ``field`` at each ``(dy, dx)`` lag.

    Computed by explicit overlap rather than through an FFT: the lag set is
    tiny, and cropping to the genuine overlap avoids the wraparound that a
    circular correlation would introduce -- wraparound would show up as a small
    spurious correlation, which is exactly the quantity being judged.
    """
    arr = np.asarray(field, dtype=np.float64)
    arr = arr - arr.mean()
    denominator = float(np.mean(arr * arr))
    out: dict[tuple[int, int], float] = {}
    for dy, dx in lags:
        if denominator <= 0:
            out[(dy, dx)] = 1.0 if (dy, dx) == (0, 0) else 0.0
            continue
        a, b = arr, arr
        if dy:
            a, b = a[:-dy or None, :], b[dy:, :]
        if dx > 0:
            a, b = a[:, :-dx], b[:, dx:]
        elif dx < 0:
            a, b = a[:, -dx:], b[:, :dx]
        out[(dy, dx)] = float(np.mean(a * b) / denominator) if a.size else 0.0
    return out


def fixed_pattern_fraction(fields: list[np.ndarray]) -> float:
    """Share of noise variance that sits at fixed photosite positions.

    Averaging N independent noise fields leaves variance ``sigma^2 / N``, not
    zero, so a raw ratio of the mean field's variance to the per-frame variance
    would report ``1/N`` of fixed pattern where there is none. The estimate is
    bias-corrected for that:

        ``var_fixed = (var(mean) - var_frame / N) / (1 - 1/N)``

    which lands near 0 for independent noise and near 1 when a pattern
    dominates. Returned clipped to [0, 1]; the raw estimate can go slightly
    negative from sampling error.
    """
    if len(fields) < 2:
        raise ValueError("need at least two noise fields to separate fixed pattern from random")
    stack = np.stack([np.asarray(f, dtype=np.float64) for f in fields])
    per_frame_variance = float(np.mean([np.var(f) for f in stack]))
    if per_frame_variance <= 0:
        return 0.0
    n = len(stack)
    mean_variance = float(np.var(stack.mean(axis=0)))
    fixed_variance = (mean_variance - per_frame_variance / n) / (1.0 - 1.0 / n)
    return float(np.clip(fixed_variance / per_frame_variance, 0.0, 1.0))


@dataclass(frozen=True)
class NoiseStructure:
    """What the sensor's noise looks like, and whether N2V can act on it."""

    sigma: float
    autocorrelation: dict[tuple[int, int], float]
    max_offdiagonal: float
    fixed_pattern_fraction: float
    horizontal_anisotropy: float
    """|r(0,1)| / |r(1,0)|. Above 1 the noise correlates along rows more than
    down columns, which points at the readout chain rather than the scene."""
    n_fields: int
    viable: bool
    verdict: str

    def __str__(self) -> str:  # pragma: no cover - presentation
        return self.verdict


def assess_n2v_viability(
    fields: list[np.ndarray],
    *,
    independence_max: float = NOISE_INDEPENDENCE_MAX,
    fixed_pattern_max: float = FIXED_PATTERN_MAX,
) -> NoiseStructure:
    """Judge both assumptions and say plainly whether training is worth starting.

    The verdict quotes the numbers it judged on so a run's output can go
    straight into MEASUREMENTS.md without being re-derived.
    """
    stack = [np.asarray(f, dtype=np.float64) for f in fields]
    # Averaged per field rather than pooled: frames differ in exposure and so
    # in noise level, and pooling would let the noisiest frame set the answer.
    per_field = [spatial_autocorrelation(f) for f in stack]
    autocorrelation = {
        lag: float(np.mean([ac[lag] for ac in per_field])) for lag in DEFAULT_LAGS
    }
    max_offdiagonal = max(abs(v) for lag, v in autocorrelation.items() if lag != (0, 0))
    horizontal = abs(autocorrelation[(0, 1)])
    vertical = abs(autocorrelation[(1, 0)])
    # A ratio of two noise-floor numbers is not a direction. White noise gives
    # r(0,1)=0.0008 and r(1,0)=0.00008, a ratio of 10 that means nothing at
    # all, so the ratio is only formed once the horizontal term is large enough
    # to be a real correlation.
    anisotropy = horizontal / max(vertical, 1e-6) if horizontal > MEANINGFUL_CORRELATION else 1.0
    fpn = fixed_pattern_fraction(stack) if len(stack) >= 2 else 0.0
    sigma = float(np.mean([f.std() for f in stack]))

    problems = []
    if max_offdiagonal >= independence_max:
        problems.append(
            f"noise is spatially correlated (max off-diagonal r = {max_offdiagonal:.3f}, "
            f"limit {independence_max:.2f}); a blind-spot network predicts that shared "
            f"component from the neighbourhood and preserves it"
        )
    if fpn >= fixed_pattern_max:
        problems.append(
            f"fixed-pattern noise dominates ({fpn:.3f} of variance, limit "
            f"{fixed_pattern_max:.2f}); it is identical every frame, so a blind-spot "
            f"network cannot see it as noise -- a dark-frame correction is the tool"
        )

    partial = NoiseStructure(
        sigma=sigma, autocorrelation=autocorrelation, max_offdiagonal=max_offdiagonal,
        fixed_pattern_fraction=fpn, horizontal_anisotropy=anisotropy,
        n_fields=len(stack), viable=not problems, verdict="",
    )
    width = recommended_mask_width(partial)
    shape = (
        f"noise is directional (horizontal/vertical lag-1 ratio {anisotropy:.1f}); "
        f"use a {width}-wide horizontal blind-spot mask, not a single pixel"
        if anisotropy >= ANISOTROPY_LIMIT and horizontal > MEANINGFUL_CORRELATION
        else "noise is isotropic; a single-pixel blind spot is the right shape"
    )

    if problems:
        verdict = (
            "Noise2Void NOT viable on this sensor: " + "; ".join(problems)
            + f". (sigma {sigma:.2f} DN, max off-diagonal r {max_offdiagonal:.3f}, "
            f"fixed-pattern {fpn:.3f}, {len(stack)} fields). If overridden: {shape}."
        )
    else:
        verdict = (
            f"Noise2Void viable: noise is spatially independent (max off-diagonal "
            f"r = {max_offdiagonal:.3f} < {independence_max:.2f}) and mostly random "
            f"rather than fixed-pattern ({fpn:.3f} < {fixed_pattern_max:.2f}). "
            f"sigma {sigma:.2f} DN over {len(stack)} fields. {shape}."
        )
    return NoiseStructure(
        sigma=sigma,
        autocorrelation=autocorrelation,
        max_offdiagonal=max_offdiagonal,
        fixed_pattern_fraction=fpn,
        horizontal_anisotropy=anisotropy,
        n_fields=len(stack),
        viable=not problems,
        verdict=verdict,
    )


def recommended_mask_width(structure: NoiseStructure) -> int:
    """Blind-spot width along the correlated axis, in plane pixels.

    A single pixel is right only for isotropic noise. When the noise correlates
    along one axis, the network can read the correlated neighbour just outside
    a point mask and copy the shared component straight through -- so the mask
    has to reach past where the correlation has decayed. This walks out along
    the row until |r| falls under a tenth of its lag-1 value, then returns the
    odd width that covers that span on both sides of the centre.

    Always odd, because the mask is centred on the pixel being predicted.
    """
    horizontal = abs(structure.autocorrelation.get((0, 1), 0.0))
    if structure.horizontal_anisotropy < ANISOTROPY_LIMIT or horizontal <= MEANINGFUL_CORRELATION:
        return 1
    reach = 1
    for lag in (2, 3, 4, 5, 6, 7):
        value = abs(structure.autocorrelation.get((0, lag), 0.0))
        if value < horizontal * 0.1:
            break
        reach = lag
    return 2 * reach + 1
