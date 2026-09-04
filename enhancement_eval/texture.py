"""Does a denoiser keep the scale texture on the fish?

Chris intends to identify individual fish from their scale patterns, so
scale-level texture is a future *input*, not cosmetic detail. The Noise2Void
run made the gap in our measurements concrete: it cut open-water grain 5.2x
and fish "detail" 4.6x, and the reticulated scales on the dive 223 angelfish
were gone -- but the fish-detail figure could not have said which had been
removed. It is high-frequency energy, and on a raw frame that energy is mostly
grain. A denoiser that erases scales and noise scores the same on it as one
that erases only the noise.

This module separates them by what scales *are* spectrally. Looked at on a
real frame (dive 223, the angelfish flank), scales are a sharp peak in the
radial power spectrum at a period of about 4.5 px, standing well above a
smooth envelope; noise is that smooth envelope. After Noise2Void the peak is
gone and the envelope is lower -- the spectrum shows the scale loss directly.

So texture is measured as **peak prominence above the fish patch's own smooth
spectral envelope**, over the band where scales live:

    texture = sum_band max(0, P(f) - envelope(f))
    retention = texture_after / texture_before

where the envelope is a running median of log-power across frequency, wide
enough to pass under a peak. A denoiser that removes only noise lowers the
envelope and leaves the peak, scoring ~1.0; a blur flattens the peak and
scores near 0; a linear filter with gain g at the scale frequency scores g^2,
the fraction of scale-band power it kept.

An earlier design subtracted the open-water spectrum as the noise floor. It
failed on real frames twice over: shot noise scales with brightness so the
water's floor is the wrong level for the fish, and the fish flank's scale
texture is only modestly above noise, so a subtraction with its own sampling
error mostly returned "undefined". Measuring the peak needs no second patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SCALE_PERIOD_MAX_PX",
    "SCALE_PERIOD_MIN_PX",
    "TextureReport",
    "assess",
    "grain_reduction",
    "radial_power_spectrum",
    "spectral_envelope",
    "texture_power",
    "texture_retention",
]

#: Spatial periods, in output pixels, over which scale texture is measured. At
#: the ranges the rig works, scale pitch on a labelled fish runs from a few
#: pixels to a couple of dozen. Below 3 px is the demosaic's own limit and is
#: nearly all grain; above 24 px is body shading rather than scales.
SCALE_PERIOD_MIN_PX = 3.0
SCALE_PERIOD_MAX_PX = 24.0


def radial_power_spectrum(
    patch: np.ndarray, *, with_counts: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radially averaged power spectrum. Returns (frequency in cycles/px, power),
    plus the number of spectral cells in each annulus when ``with_counts``.

    The mean is removed and a Hann window applied before the FFT, so a bright
    patch does not read as low-frequency power and the patch edges do not leak
    across the spectrum. Power is averaged in annular bins of radial frequency,
    which makes the result independent of texture orientation -- scales run
    whichever way the fish is facing.

    Power is normalized by the window's energy, so white noise of variance
    sigma^2 reads as sigma^2 per bin *whatever the patch size*. Without that,
    |FFT|^2 grows with pixel count, and a 320 px water patch read 4x above a
    200 px fish patch at the same noise level -- an error the synthetic tests
    could not see because every patch in them was 128x128.
    """
    arr = np.asarray(patch, dtype=np.float64)
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr * win))) ** 2 / float((win ** 2).sum())
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    n_bins = min(h, w) // 2
    edges = np.linspace(0, 0.5 * np.sqrt(2), n_bins + 1)
    which = np.clip(np.digitize(radius, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(which.ravel(), minlength=n_bins)
    sums = np.bincount(which.ravel(), weights=spectrum.ravel(), minlength=n_bins)
    power = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    centres = (edges[:-1] + edges[1:]) / 2
    return (centres, power, counts) if with_counts else (centres, power)


def spectral_envelope(power: np.ndarray, window: int | None = None) -> np.ndarray:
    """Smooth baseline under a spectrum: a running median of log-power.

    A median passes under a narrow peak rather than following it up, which is
    what makes the peak's prominence measurable. The window is a fraction of
    the bin count so it scales with patch size; it has to be wider than a
    scale peak (a few bins) and narrower than the envelope's own curvature.
    """
    from scipy.ndimage import median_filter

    n = len(power)
    window = window or max(9, n // 8)
    if window % 2 == 0:
        window += 1
    logp = np.log(np.maximum(power, 1e-30))
    return np.exp(median_filter(logp, size=window, mode="nearest"))


def _scale_band(freqs: np.ndarray) -> np.ndarray:
    return (freqs >= 1.0 / SCALE_PERIOD_MAX_PX) & (freqs <= 1.0 / SCALE_PERIOD_MIN_PX)


def texture_power(patch: np.ndarray) -> tuple[float, float]:
    """(peak prominence in the scale band, its sampling noise under no peak).

    The second number is the yardstick for the first. It is *not* the
    envelope's total -- comparing a peak to that penalizes narrow peaks, and
    real scales are narrow: the dive 223 angelfish is six discrete lattice
    peaks that the radial average spreads over a handful of bins. Against the
    whole band's envelope those read as 10-20% and were thrown out as noise.
    Against the noise of the estimate itself they are many sigma.

    Under no peak, each bin's power is a mean over its annulus's cells, so its
    standard deviation is about ``envelope / sqrt(cells)``; the band sum's
    noise is the root-sum-square of those.
    """
    freqs, power, counts = radial_power_spectrum(patch, with_counts=True)
    envelope = spectral_envelope(power)
    band = _scale_band(freqs)
    # Summed before clipping, for the same reason as everywhere else in this
    # module: each bin's residual above the envelope carries sampling noise,
    # and clipping the negative half leaves a positive bias that grows with
    # the noise level. At sigma 25 it made a perfect denoiser read well under
    # 1.0. Summed first, noise residuals cancel and the peak remains.
    prominence = max(float((power[band] - envelope[band]).sum()), 0.0)
    noise = float(np.sqrt((envelope[band] ** 2 / np.maximum(counts[band], 1)).sum()))
    return prominence, noise


#: A peak has to stand this many sigma above the estimate's own noise to be
#: measured. The retention ratio's error is roughly 1/significance on each
#: side, so this bounds it near +-15%; below the line the honest answer is
#: "undefined", not a number that will be averaged into a table.
SIGNIFICANCE = 6.0


def texture_retention(fish_before: np.ndarray, fish_after: np.ndarray) -> float:
    """Fraction of scale-band peak power that survived a denoiser.

    Returns NaN when the input carried no measurable peak in the band -- 0/0
    is not a retention figure and must not be averaged into a table.
    """
    before, noise = texture_power(fish_before)
    if before <= 0 or before < SIGNIFICANCE * noise:
        return float("nan")
    after, _ = texture_power(fish_after)
    return after / before


def _grain(patch: np.ndarray) -> float:
    """High-frequency residual after a 3x3 median -- the grain measure every
    other sweep in MEASUREMENTS.md uses, so the figures line up. A plain
    standard deviation was tried first and read /1.2 where this reads /5:
    it was counting the water's illumination gradient, which no denoiser
    removes and which is not noise."""
    from scipy.ndimage import median_filter

    arr = np.asarray(patch, dtype=np.float64)
    return float(np.std(arr - median_filter(arr, size=3)))


def grain_reduction(water_before: np.ndarray, water_after: np.ndarray) -> float:
    """Ratio of open-water grain before to after; >1 means quieter."""
    return _grain(water_before) / max(_grain(water_after), 1e-9)


@dataclass(frozen=True)
class TextureReport:
    """Grain reduction and texture retention, reported together on purpose.

    Either number on its own can be gamed. A blur wins on grain; the identity
    wins on retention. A denoiser is only a win when it moves the first without
    moving the second.
    """

    grain_reduction: float
    texture_retained: float

    def __str__(self) -> str:
        retained = "undefined (no texture in band)" if np.isnan(self.texture_retained) \
            else f"{self.texture_retained:.2f}"
        return f"grain /{self.grain_reduction:.2f}, scale texture retained {retained}"


def assess(fish_before, water_before, fish_after, water_after) -> TextureReport:
    return TextureReport(
        grain_reduction=grain_reduction(water_before, water_after),
        texture_retained=texture_retention(fish_before, fish_after),
    )
