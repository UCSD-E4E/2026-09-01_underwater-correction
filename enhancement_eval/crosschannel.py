"""Cross-channel denoising: filter the starved channel using the clean one.

Underwater the three channels are in very different states, and that asymmetry
is exploitable. Red is photon-starved, so it is the noisiest channel --
water-region high-frequency energy measures 9.50 against green's 6.88 on dive
370. But luminance is 0.21R + 0.72G + 0.07B, so the structure a labeler reads
(scale texture, fin edges, the outline a keypoint is clicked on) lives mostly
in **green**. Red noise therefore surfaces as chroma speckle: visually loud and
structurally useless.

Two modes follow from that:

* ``red-only`` filters red and leaves green and blue bit-identical. No
  luminance structure can be lost, because the channel carrying 72% of
  luminance is never touched.
* ``guided`` uses green as a guide to filter red and blue (He et al.'s guided
  filter). Edges are shared across channels because they are edges of the same
  objects, so the guide supplies structure that is genuinely present in the
  same exposure.

**Why this is not the generative case.** A GAN denoiser invents detail that is
not in the data, and invented detail is indistinguishable from recovered detail
in the output. A guided filter transfers detail from *another measurement of
the same scene at the same instant*, and its output is an affine function of
the guide within each window: where the guide is flat, the result is flat. It
cannot synthesise an edge that was not measured, which is the property
`test_guided_filter_does_not_invent_an_edge_the_guide_lacks` pins.

Implemented here rather than called from OpenCV because `cv2.ximgproc` in this
build ships without `guidedFilter`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["guided_filter", "denoise_channels", "estimate_noise"]

#: Guide channel index in an RGB array. Green: it collects twice the photosites
#: of red or blue on a Bayer sensor, so it is the least noisy measurement of the
#: same scene available.
GUIDE = 1


def estimate_noise(plane: np.ndarray) -> float:
    """Robust high-frequency noise estimate for one channel.

    MAD of the residual after a 3x3 median, scaled to a Gaussian sigma. Robust
    rather than a plain standard deviation because real edges are exactly the
    outliers a std would absorb, which would inflate the estimate until the
    filter stopped doing anything.

    Computed in float. An earlier version quantised to uint8 for the median,
    which is fine for a mid-grey channel and catastrophic for a starved one:
    linear red occupies roughly the bottom 5% of the range underwater, so x255
    leaves ~13 levels, the MAD lands on exactly zero, and every caller that
    derives a threshold from this silently becomes a no-op.
    """
    from scipy.ndimage import median_filter

    resid = plane - median_filter(plane, size=3, mode="nearest")
    centred = np.abs(resid - np.median(resid))
    sigma = 1.4826 * float(np.median(centred))
    if sigma > 1e-9:
        return sigma
    # MAD collapsed. That happens on a starved, coarsely-quantised channel --
    # linear red in open water averages 0.0129, so most pixels sit exactly on
    # their own 3x3 median and over half the residuals are exactly zero. Fall
    # back to a high percentile, which still sees the pixels that did move.
    return 1.4826 * float(np.percentile(centred, 75)) or float(resid.std())


def _box(img: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    k = 2 * radius + 1
    return cv2.boxFilter(img, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(
    guide: np.ndarray, src: np.ndarray, *, radius: int = 4, eps: float = 1e-3
) -> np.ndarray:
    """He et al. guided filter: smooth `src` while following `guide`'s edges.

    Within each window the output is an affine function of the guide,
    ``q = a * guide + b``, with ``a`` shrinking toward zero as the guide's local
    variance falls below `eps`. That is what makes it edge-preserving where the
    guide has an edge and smoothing where it does not — and what makes it
    unable to invent structure the guide never measured.
    """
    guide = guide.astype(np.float64, copy=False)
    src = src.astype(np.float64, copy=False)

    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    cov_gs = _box(guide * src, radius) - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def denoise_channels(
    rgb: np.ndarray, *, mode: str, strength: float = 0.5, radius: int = 4
) -> np.ndarray:
    """Denoise per channel. `rgb` is float in [0, 1].

    `strength` maps to the guided filter's `eps`: larger smooths harder. It is
    exposed on a 0-1 scale rather than as a raw epsilon because epsilon is in
    units of squared guide variance, which is not a quantity anyone tuning this
    should have to think in.
    """
    if mode not in ("red-only", "guided"):
        raise ValueError(f"mode must be 'red-only' or 'guided', got {mode!r}")

    # eps is scaled to the filtered channel's OWN measured noise, not fixed.
    #
    # eps is in units of squared guide variance, so a constant means different
    # things on different frames: below the noise variance the filter follows
    # the speckle and does nothing, above the edge variance it flattens
    # everything. Setting eps = (k * sigma)^2 makes `strength` mean something
    # stable -- "smooth structure weaker than k noise-sigmas" -- which is the
    # same lesson the laser-dot threshold taught: absolute cuts do not transfer
    # across frames whose noise differs.
    k = 0.5 + 3.5 * float(np.clip(strength, 0.0, 1.0))
    out = rgb.copy()
    guide = rgb[:, :, GUIDE]

    if mode == "red-only":
        # Self-guided: the channel guides its own filtering, so this is an
        # edge-preserving smooth of red alone. Green and blue are untouched,
        # which is the whole point.
        eps = (k * estimate_noise(rgb[:, :, 0])) ** 2
        out[:, :, 0] = guided_filter(rgb[:, :, 0], rgb[:, :, 0], radius=radius, eps=eps)
        return out

    # Guided mode: eps is set from the GUIDE's noise, because eps is compared
    # against the guide's local variance, not the filtered channel's.
    eps = (k * estimate_noise(guide)) ** 2
    for channel in (0, 2):
        out[:, :, channel] = guided_filter(
            guide, rgb[:, :, channel], radius=radius, eps=eps
        )
    return out
