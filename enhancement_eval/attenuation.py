"""Fitting water attenuation coefficients against the dive slate.

The obstacle to Sea-thru everywhere else is that it needs a per-pixel metric
range map. This corpus has a metric range at one pixel per frame --
`LaserDepth.range_m` -- and, in slate frames, a **known constant-reflectance
target** at a known place in the image.

That combination is what makes absolute coefficients recoverable. For a target
of fixed reflectance rho_c seen at range z,

    log(I_c) = log(J_c * rho_c) - beta_c * z + (backscatter)

so regressing log intensity on range across frames of one dive estimates
beta_c itself. The earlier annulus-around-the-dot method could not do this: it
sampled a different scene in every frame, so rho_c varied and only *ratios*
between channels were interpretable. That method recovered the right sign on
all three field dives and the right null in the pool, but magnitudes varied
3.7x with r2 = 0.18-0.34, and the variance was exactly the reflectance term.

Assumptions, stated because they bound what the number means:

* **Reflectance is constant.** The slate is one printed sheet, so this holds
  across frames of a dive far better than it holds across scenes -- but the
  bright tail inside the quad is used rather than its mean, because the mean is
  a paper/ink mixture that shifts with how much artwork is in view.
* **Illumination is constant within a dive.** It is not exactly: ambient light
  falls with depth and shifts with time of day. Residual illumination drift
  inflates the scatter and, if it correlates with range, biases the slope.
* **Backscatter is neglected.** The full model adds a veiling term that
  saturates with range, which flattens the curve at distance. Over the 1-3.7 m
  the slate frames span this is a second-order effect; over longer ranges it is
  not, and a straight line would understate beta.

**The pool dives are the negative control.** Clear water has almost no
attenuation over 2 m, so a correct method must return approximately zero there.
Seven of the eight dives currently carrying slate frames with a metric range
are Pool Calibration, which makes that control free -- and it is the only
validation available until field slate frames exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "SLATE_ATTENUATION_SQL",
    "as_polygon",
    "LinearFit",
    "fit_loglinear",
    "sample_slate_patch",
    "AttenuationFit",
    "fit_attenuation",
]

#: Slate frames that also carry a computed metric range. `slate_rectangle` is
#: stored in photo-frame pixels -- the sync activity shifts the Label Studio
#: composite's PDF-panel width off the x coordinates before persisting -- so it
#: indexes the rectified frame directly.
SLATE_ATTENUATION_SQL = """
SELECT DISTINCT ON (ld.image_id)
       ld.image_id,
       i.dive_id,
       COALESCE(d.name, '')       AS dive_name,
       COALESCE(d.path, '')       AS dive_path,
       i.path                     AS image_path,
       ld.range_m,
       s.slate_rectangle          AS slate_rectangle,
       ci.camera_matrix           AS camera_matrix,
       ci.distortion_coefficients AS distortion_coefficients
FROM laserdepth ld
JOIN image i           ON i.id = ld.image_id
JOIN dive d            ON d.id = i.dive_id
JOIN diveslatelabel s  ON s.image_id = ld.image_id
                      AND s.completed
                      AND NOT COALESCE(s.superseded, false)
                      AND s.slate_rectangle IS NOT NULL
LEFT JOIN camera c     ON c.id = COALESCE(i.camera_id, d.camera_id)
LEFT JOIN cameraintrinsics ci ON ci.camera_id = c.id
WHERE ld.range_m IS NOT NULL
  AND ld.depth_m > 0
  AND i.is_canonical
  AND ci.camera_matrix IS NOT NULL
ORDER BY ld.image_id, s.updated_at DESC NULLS LAST, s.id DESC
"""

#: Minimum range spread, in metres, for a slope to mean anything. Attenuation
#: is a slope against distance; without a lever there is nothing to measure and
#: any number reported is noise dressed as a coefficient.
MIN_RANGE_SPAN_M = 0.4
MIN_SAMPLES = 8


@dataclass(frozen=True)
class LinearFit:
    """One ordinary least-squares slope, with enough to judge it."""

    slope: float
    intercept: float
    stderr: float
    r2: float
    n: int
    ci_low: float
    ci_high: float

    @property
    def spans_zero(self) -> bool:
        """True when the data cannot distinguish this slope from none.

        The property that lets the method say "no attenuation here" instead of
        inventing a correction for clear water.
        """
        return self.ci_low <= 0.0 <= self.ci_high


def fit_loglinear(z: Sequence[float], y: Sequence[float]) -> LinearFit | None:
    """OLS of `y` on `z`. None when the data cannot support a slope."""
    n = len(z)
    if n < MIN_SAMPLES or n != len(y):
        return None
    if max(z) - min(z) < 1e-9:
        return None

    mz = sum(z) / n
    my = sum(y) / n
    szz = sum((v - mz) ** 2 for v in z)
    if szz <= 0:
        return None
    slope = sum((a - mz) * (b - my) for a, b in zip(z, y)) / szz
    intercept = my - slope * mz

    residuals = [b - (intercept + slope * a) for a, b in zip(z, y)]
    sse = sum(e * e for e in residuals)
    sst = sum((b - my) ** 2 for b in y)
    stderr = math.sqrt(sse / (n - 2) / szz) if n > 2 else float("nan")
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    # 1.96 rather than a t quantile: n is comfortably above 30 in every real
    # cohort here, and pretending to more precision than the model deserves
    # would be false comfort.
    return LinearFit(slope, intercept, stderr, r2, n,
                     slope - 1.96 * stderr, slope + 1.96 * stderr)


def as_polygon(
    rectangle: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]] | None:
    """Normalize a `DiveSlateLabel.slate_rectangle` into a fillable polygon.

    The column stores **two opposite corners**, not a polygon:

        [[1708.4, 1079.6], [2298.0, 1534.5]]

    Handing that to `cv2.fillPoly` does not fail -- it fills a degenerate
    two-point polygon, i.e. a one-pixel-wide diagonal line. On a real frame
    that line runs several hundred pixels, so it clears a minimum-pixel guard
    and looks like a valid sample while containing almost none of the slate.
    Every statistic taken from it is then a statistic of a streak.

    Returns None for anything that cannot bound an area, so a caller that
    forgets to check gets nothing rather than a line.
    """
    # Explicit None/len rather than truthiness: a numpy array raises on
    # `not array`, and callers legitimately pass one.
    if rectangle is None or len(rectangle) == 0:
        return None
    points = [(float(p[0]), float(p[1])) for p in rectangle]
    if len(points) >= 3:
        return points
    if len(points) != 2:
        return None
    (x0, y0), (x1, y1) = points
    if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
        return None
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    return [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]


def sample_slate_patch(
    bgr: np.ndarray,
    quad: Sequence[Sequence[float]],
    *,
    percentile: float = 85.0,
    min_pixels: int = 400,
) -> tuple[float, float, float] | None:
    """Mean (R, G, B) of the slate's paper inside `quad`, on a LINEAR frame.

    The bright tail rather than the mean: the slate is white paper carrying
    black markings, so its mean is a paper/ink mixture that shifts with how
    much artwork is in view, while the tail is the paper -- and paper being one
    constant reflectance is the assumption the whole method rests on.

    Returns None when the quad is too small to sample or the slate is
    saturated. A blown-out slate sits at the container ceiling no matter how
    much water it was seen through, so it carries no attenuation information at
    all; including it would flatten every slope toward zero.
    """
    import cv2

    polygon = as_polygon(quad)
    if polygon is None:
        return None
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)], 255)
    inside = mask.astype(bool)
    if int(inside.sum()) < min_pixels:
        return None

    full_scale = float(np.iinfo(bgr.dtype).max) if np.issubdtype(bgr.dtype, np.integer) else 1.0
    pixels = bgr[inside].astype(np.float64)
    usable = pixels.max(axis=1) < full_scale * 0.98
    if usable.sum() < min_pixels // 2:
        return None

    pixels = pixels[usable]
    luminance = pixels.mean(axis=1)
    cutoff = float(np.percentile(luminance, percentile))
    tail = pixels[luminance >= cutoff]
    if len(tail) < 8:
        return None
    # BGR in, (R, G, B) out.
    return float(tail[:, 2].mean()), float(tail[:, 1].mean()), float(tail[:, 0].mean())


@dataclass(frozen=True)
class AttenuationFit:
    """Per-dive attenuation, per channel, with the fits that produced it."""

    beta_r: float
    beta_g: float
    beta_b: float
    beta_r_fit: LinearFit
    beta_g_fit: LinearFit
    beta_b_fit: LinearFit
    #: Differential coefficients, directly comparable with the earlier
    #: annulus-around-the-dot result, which could only measure these.
    d_rg: float
    d_bg: float
    n: int
    range_span: float

    @property
    def ordering_ok(self) -> bool:
        """Whether the result obeys the expected underwater ordering.

        Red is absorbed fastest and blue least, so beta_r > beta_g > beta_b is
        what physics predicts. A fit that violates it is reporting something
        other than water -- illumination drift, a mis-sampled quad, or noise.
        """
        return self.beta_r > self.beta_g > self.beta_b

    @property
    def trustworthy(self) -> bool:
        """Enough lever, enough samples, and a red slope distinguishable from zero."""
        return (
            self.range_span >= MIN_RANGE_SPAN_M
            and self.n >= 20
            and not self.beta_r_fit.spans_zero
        )


def fit_attenuation(samples: Sequence[Mapping[str, Any]]) -> AttenuationFit | None:
    """Fit beta per channel from slate observations of one dive.

    `samples` carry `range_m` and linear `R`, `G`, `B`. The sign convention:
    beta is positive for light that is lost with distance, so it is the
    *negation* of the slope of log intensity on range.
    """
    usable = [
        s for s in samples
        if s.get("range_m") and min(float(s["R"]), float(s["G"]), float(s["B"])) > 0
    ]
    if len(usable) < MIN_SAMPLES:
        return None

    z = [float(s["range_m"]) for s in usable]
    fits = {}
    for key in ("R", "G", "B"):
        fits[key] = fit_loglinear(z, [math.log(float(s[key])) for s in usable])
    if any(f is None for f in fits.values()):
        return None

    beta = {k: -f.slope for k, f in fits.items()}
    return AttenuationFit(
        beta_r=beta["R"], beta_g=beta["G"], beta_b=beta["B"],
        beta_r_fit=fits["R"], beta_g_fit=fits["G"], beta_b_fit=fits["B"],
        d_rg=beta["R"] - beta["G"], d_bg=beta["B"] - beta["G"],
        n=len(usable), range_span=max(z) - min(z),
    )
