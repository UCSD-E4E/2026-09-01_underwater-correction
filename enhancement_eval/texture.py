"""Does a denoiser keep the scale texture on the fish?

Chris intends to identify individual fish from their scale patterns, so
scale-level texture is a future *input*, not cosmetic detail. The Noise2Void
run made the gap in our measurements concrete: it cut open-water grain 5.2x
and fish "detail" 4.6x, and the reticulated scales on the dive 223 angelfish
were gone -- but the fish-detail figure could not have said which had been
removed. It is high-frequency energy, and on a raw frame that energy is mostly
grain. A denoiser that erases scales and noise scores the same on it as one
that erases only the noise.

This module separates the two by subtracting the noise floor. Open water has
no texture, so its power spectrum *is* the noise spectrum; the fish patch's
spectrum minus the water patch's is the texture's own power, independent of
how noisy the frame was. Measure that before and after a denoiser, over the
band where scales live, and the ratio is how much texture survived:

    retention = sum_band [P_fish_after - P_water_after]
              / sum_band [P_fish_before - P_water_before]

A denoiser that removes only noise scores near 1.0, because the subtraction
cancels the noise on both sides. One that smooths the fish scores well below
it. `test_perfect_denoiser_retains_everything` is the case that matters: the
old figure would have reported a large *drop* there, because what it was
mostly measuring had been removed.
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
    "texture_retention",
]

#: Spatial periods, in output pixels, over which scale texture is measured. At
#: the ranges the rig works, scale pitch on a labelled fish runs from a few
#: pixels to a couple of dozen. Below 3 px is the demosaic's own limit and is
#: nearly all grain; above 24 px is body shading rather than scales.
SCALE_PERIOD_MIN_PX = 3.0
SCALE_PERIOD_MAX_PX = 24.0

#: Periods where scale texture has no power and the spectrum is noise alone.
#: Used to level-match the noise floor to the fish patch itself.
NOISE_TAIL_MIN_PX = 2.0
NOISE_TAIL_MAX_PX = 2.7


def radial_power_spectrum(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum. Returns (frequency in cycles/px, power).

    The mean is removed and a Hann window applied before the FFT, so a bright
    patch does not read as low-frequency power and the patch edges do not leak
    across the spectrum. Power is averaged in annular bins of radial frequency,
    which makes the result independent of texture orientation -- scales run
    whichever way the fish is facing.
    """
    arr = np.asarray(patch, dtype=np.float64)
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(arr * win))) ** 2
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
    return centres, power


def _band_signal(fish: np.ndarray, water: np.ndarray, level: float | None = None) -> float:
    """Noise-floor-subtracted texture power in the scale band.

    The difference is summed across the band *before* being clipped at zero.
    Clipping per bin looked harmless and was not: each bin's difference
    carries sampling noise around the true value, and clipping the negative
    half of that noise leaves a positive bias that grows with the noise level.
    On a sigma-25 input it made a perfect denoiser read 0.66 retention, which
    is the metric being fooled by noise -- the one failure it exists to
    prevent. Summed first, the sampling noise averages out.

    The water spectrum is interpolated onto the fish patch's frequency axis,
    so the two patches need not be the same size.
    """
    freqs, p_fish = radial_power_spectrum(fish)
    floor = _noise_floor(freqs, p_fish, water, level)
    band = (freqs >= 1.0 / SCALE_PERIOD_MAX_PX) & (freqs <= 1.0 / SCALE_PERIOD_MIN_PX)
    return max(float((p_fish[band] - floor[band]).sum()), 0.0)


def _level_ratio(fish: np.ndarray, water: np.ndarray) -> float:
    """Fish-to-water noise level, read where the spectrum is noise alone."""
    freqs, p_fish = radial_power_spectrum(fish)
    f_water, p_water = radial_power_spectrum(water)
    p_water = np.interp(freqs, f_water, p_water)
    tail = (freqs >= 1.0 / NOISE_TAIL_MAX_PX) & (freqs <= 1.0 / NOISE_TAIL_MIN_PX)
    water_tail = float(p_water[tail].mean()) if tail.any() else 0.0
    return float(p_fish[tail].mean()) / water_tail if water_tail > 0 else 1.0


def _noise_floor(
    freqs: np.ndarray, p_fish: np.ndarray, water: np.ndarray, level: float | None
) -> np.ndarray:
    """The fish patch's noise spectrum: shape from water, level from the fish.

    Neither source alone is right. The water patch has the correct spectral
    *shape* -- demosaicing correlates neighbouring pixels, so the noise is not
    white and its roll-off has to come from real noise, not an assumption. But
    its *level* is wrong: shot noise scales with brightness, and a fish is not
    the same brightness as open water, so subtracting water's floor directly
    over- or under-corrects. That showed up as a retention of 1.16 on dive 1,
    texture apparently created by denoising.

    So the water spectrum is rescaled to match the fish patch's own power at
    periods of 2 to 2.7 px, where scales have no energy and the spectrum is
    noise alone. Shape from one, level from the other.

    ``level`` is that ratio, and it is measured on the *before* patches and
    reused for the *after* ones. The mismatch is a property of the scene --
    how much brighter the fish is than the water -- not of the denoiser, and
    measuring it after denoising is unstable: a heavy blur empties both tails
    and the ratio becomes 0/0. That broke the ordering of retention against
    blur strength before this was pinned.
    """
    f_water, p_water = radial_power_spectrum(water)
    p_water = np.interp(freqs, f_water, p_water)
    if level is None:
        tail = (freqs >= 1.0 / NOISE_TAIL_MAX_PX) & (freqs <= 1.0 / NOISE_TAIL_MIN_PX)
        water_tail = float(p_water[tail].mean()) if tail.any() else 0.0
        level = float(p_fish[tail].mean()) / water_tail if water_tail > 0 else 1.0
    return p_water * level


def texture_retention(
    fish_before: np.ndarray,
    water_before: np.ndarray,
    fish_after: np.ndarray,
    water_after: np.ndarray,
) -> float:
    """Fraction of scale-band texture power that survived a denoiser.

    Returns NaN when the input carried no measurable texture in the band --
    0/0 is not a retention figure and must not be averaged into a table.
    """
    level = _level_ratio(fish_before, water_before)
    before = _band_signal(fish_before, water_before, level)
    if before <= 0:
        return float("nan")
    # Denominator guard: with no measurable texture the spectrum is all noise,
    # and sampling variation between two noise patches would otherwise leave a
    # small positive "signal" that turns into a meaningless retention figure.
    freqs, p_fish = radial_power_spectrum(fish_before)
    floor = _noise_floor(freqs, p_fish, water_before, level)
    band = (freqs >= 1.0 / SCALE_PERIOD_MAX_PX) & (freqs <= 1.0 / SCALE_PERIOD_MIN_PX)
    if before < 0.15 * float(floor[band].sum()):
        return float("nan")
    return _band_signal(fish_after, water_after, level) / before


def _grain(patch: np.ndarray) -> float:
    """High-frequency residual after a 3x3 median -- the grain measure every
    other sweep in MEASUREMENTS.md uses, so the figures line up. A plain
    standard deviation was tried first and read a /1.2 where this reads /5:
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
        texture_retained=texture_retention(fish_before, water_before, fish_after, water_after),
    )
