"""Does a denoiser keep the scale texture on the fish?

Chris intends to identify individual fish from their scale patterns, so scale-
level texture is a future input rather than cosmetic detail. The Noise2Void
run made that concrete: it cut open-water grain 5.2x and fish "detail" 4.6x,
and the reticulated scales on the dive 223 angelfish were gone. The existing
fish-detail figure could not have said which -- it is high-frequency energy,
and on a raw frame that energy is mostly grain. It cannot tell scales from
noise, so a denoiser that erases both scores the same as one that erases only
the noise.

This metric separates them by subtracting the noise floor. Open water has no
texture, so its power spectrum *is* the noise spectrum; the fish patch's
spectrum minus the water patch's is the texture's own power. Do that before
and after denoising, over the band where scales live, and the ratio is how
much of the texture survived. A denoiser that removes only noise scores near
1.0. One that smooths the fish scores well below it.
"""

import numpy as np
import pytest

from enhancement_eval.texture import (
    SCALE_PERIOD_MAX_PX,
    SCALE_PERIOD_MIN_PX,
    TextureReport,
    grain_reduction,
    radial_power_spectrum,
    texture_retention,
)


def _rng(seed=20260903):
    return np.random.default_rng(seed)


def _scales(h=128, w=128, period=7.0, amplitude=12.0):
    """A quasi-periodic reticulated pattern -- two crossed sinusoids, the
    simplest thing with the spectral signature of fish scales."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return amplitude * (np.sin(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period))


def _scene(noise_sigma=8.0, seed=1):
    """Fish patch = texture + noise; water patch = the same noise, no texture."""
    r = _rng(seed)
    fish_clean = 120 + _scales()
    water_clean = np.full((128, 128), 120.0)
    fish = fish_clean + r.normal(0, noise_sigma, fish_clean.shape)
    water = water_clean + r.normal(0, noise_sigma, water_clean.shape)
    return fish_clean, fish, water


# ---------------------------------------------------------------------------
# The spectrum
# ---------------------------------------------------------------------------

def test_radial_spectrum_of_white_noise_is_flat():
    freqs, power = radial_power_spectrum(_rng().normal(0, 5, (256, 256)))
    mid = power[(freqs > 0.1) & (freqs < 0.45)]
    assert mid.std() / mid.mean() < 0.35, "white noise should be roughly flat across frequency"


def test_radial_spectrum_peaks_at_the_texture_period():
    freqs, power = radial_power_spectrum(_scales(period=8.0))
    peak_freq = freqs[np.argmax(power[freqs > 0.02]) + np.sum(freqs <= 0.02)]
    # The crossed pattern's fundamental sits at sqrt(2)/period in radial frequency.
    assert abs(peak_freq - np.sqrt(2) / 8.0) < 0.03, f"peak at {peak_freq:.3f}"


def test_radial_spectrum_ignores_the_mean():
    a = radial_power_spectrum(_scales())[1]
    b = radial_power_spectrum(_scales() + 500.0)[1]
    assert np.allclose(a[1:], b[1:], rtol=1e-6)


def test_scale_band_is_where_scales_actually_live():
    """Pinned so the band cannot drift to somewhere convenient. At the ranges
    the rig works, scale pitch on a labelled fish runs a few pixels to a couple
    of dozen."""
    assert 2 <= SCALE_PERIOD_MIN_PX <= 4
    assert 16 <= SCALE_PERIOD_MAX_PX <= 32


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_identity_denoiser_retains_everything():
    _, fish, water = _scene()
    r = texture_retention(fish, water, fish, water)
    assert 0.9 < r < 1.1, f"identity should retain ~1.0, got {r:.3f}"


def test_perfect_denoiser_retains_everything():
    """Returning the clean texture with the noise gone scores ~1.0. This is
    the case the old fish-detail figure got wrong: it would have read a
    large *drop* here, because the noise it was mostly measuring is gone."""
    fish_clean, fish, water = _scene()
    r = texture_retention(fish, water, fish_clean, np.full_like(water, 120.0))
    assert 0.85 < r < 1.15, f"perfect denoiser should retain ~1.0, got {r:.3f}"


def test_smoothing_denoiser_loses_the_texture():
    from scipy.ndimage import gaussian_filter

    _, fish, water = _scene()
    blur = lambda a: gaussian_filter(a, 3.0)
    r = texture_retention(fish, water, blur(fish), blur(water))
    assert r < 0.3, f"a sigma-3 blur should erase 7px scales, got {r:.3f}"


def test_mild_smoothing_loses_some_texture_not_all():
    from scipy.ndimage import gaussian_filter

    _, fish, water = _scene()
    blur = lambda a: gaussian_filter(a, 0.8)
    r = texture_retention(fish, water, blur(fish), blur(water))
    assert 0.3 < r < 0.95


def test_retention_is_ordered_by_how_much_smoothing_was_applied():
    from scipy.ndimage import gaussian_filter

    _, fish, water = _scene()
    rs = [texture_retention(fish, water, gaussian_filter(fish, s), gaussian_filter(water, s))
          for s in (0.5, 1.0, 2.0, 4.0)]
    assert rs == sorted(rs, reverse=True), rs


def test_retention_is_not_fooled_by_noise_level():
    """A denoiser that removes noise but keeps texture must score ~1.0 no
    matter how noisy the input was -- otherwise the metric is measuring the
    noise, which is the failure it exists to fix."""
    fish_clean, fish, water = _scene(noise_sigma=25.0)
    r = texture_retention(fish, water, fish_clean, np.full_like(water, 120.0))
    assert 0.8 < r < 1.2, f"got {r:.3f} at sigma 25"


def test_retention_survives_a_brightness_mismatch_between_fish_and_water():
    """Shot noise scales with brightness, so a fish patch and an open-water
    patch do not share a noise level. A metric that subtracts the water floor
    directly reads texture being *created* by a denoiser (1.16 on dive 1). The
    floor's level must come from the fish patch itself."""
    r = _rng(9)
    fish_clean = 120 + _scales()
    fish = fish_clean + r.normal(0, 16.0, fish_clean.shape)           # bright, noisy
    water = np.full((128, 128), 40.0) + r.normal(0, 6.0, (128, 128))  # dark, quiet
    identity = texture_retention(fish, water, fish, water)
    perfect = texture_retention(fish, water, fish_clean, np.full_like(water, 40.0))
    assert 0.85 < identity < 1.15, f"identity read {identity:.3f}"
    assert 0.8 < perfect < 1.2, f"perfect denoiser read {perfect:.3f}"


def test_retention_on_a_textureless_fish_patch_is_reported_as_undefined():
    """No texture to retain means the ratio is 0/0. It must say so rather
    than return a number that will be averaged into a table."""
    _, _, water = _scene()
    r = texture_retention(water, water, water, water)
    assert np.isnan(r)


# ---------------------------------------------------------------------------
# Grain, for the same report
# ---------------------------------------------------------------------------

def test_grain_reduction_reads_the_noise_floor_drop():
    _, _, water = _scene(noise_sigma=8.0)
    quieter = np.full_like(water, 120.0) + _rng(5).normal(0, 2.0, water.shape)
    g = grain_reduction(water, quieter)
    assert 3.0 < g < 5.5, f"sigma 8 -> 2 should read about 4x, got {g:.2f}"


def test_grain_reduction_ignores_an_illumination_gradient():
    """Open water is not flat -- it darkens away from the strobe. That gradient
    is not noise and no denoiser removes it, so it must not sit in the
    denominator making every reduction look smaller than it is."""
    r = _rng(6)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    gradient = 100 + 0.5 * xx + 0.3 * yy
    before = gradient + r.normal(0, 8.0, gradient.shape)
    after = gradient + r.normal(0, 2.0, gradient.shape)
    g = grain_reduction(before, after)
    assert 3.0 < g < 5.5, f"gradient should not dilute the ratio, got {g:.2f}"


def test_report_carries_both_numbers():
    from enhancement_eval.texture import assess

    fish_clean, fish, water = _scene()
    rep = assess(fish, water, fish_clean, np.full_like(water, 120.0) + _rng(3).normal(0, 1, water.shape))
    assert isinstance(rep, TextureReport)
    assert rep.texture_retained > 0.85
    assert rep.grain_reduction > 4.0
    assert f"{rep.texture_retained:.2f}" in str(rep)
