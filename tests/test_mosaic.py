"""Denoising in the CFA domain, where the noise is still independent.

Measured on a real frame, the correlation between channels' high-frequency
residuals is +0.029 (R,G) before demosaicing and **+0.328** after. Interpolation
mixes each output pixel from overlapping neighbourhoods of photosites, so it
manufactures exactly the cross-channel correlation that cross-channel denoising
assumes is absent. Guiding red with green in RGB space therefore guides it
partly toward its own noise -- which is why that attempt removed signal and
noise in equal proportion.

At the mosaic every photosite is one independent measurement, so a guide built
there is legitimate. And the two green photosites in each 2x2 cell sample
nearly the same point, so **G1 - G2 is signal-free**: a direct noise
measurement rather than a robust-estimator approximation that collapses on
starved channels.
"""

import numpy as np
import pytest

from enhancement_eval.mosaic import (
    green_noise_sigma,
    split_cfa,
    merge_cfa,
    denoise_cfa,
)

# Olympus TG-6 pattern as rawpy reports it: indices into "RGBG".
PATTERN = np.array([[0, 1], [3, 2]])
DESC = "RGBG"


def _mosaic(r=400.0, g=900.0, b=1200.0, noise=0.0, shape=(64, 64), seed=0):
    """Build a synthetic CFA frame with known per-photosite levels."""
    rng = np.random.default_rng(seed)
    m = np.zeros((shape[0] * 2, shape[1] * 2))
    m[0::2, 0::2] = r
    m[0::2, 1::2] = g
    m[1::2, 0::2] = g
    m[1::2, 1::2] = b
    if noise:
        m += rng.normal(0, noise, m.shape)
    return np.clip(m, 0, None)


def test_split_recovers_the_four_photosite_planes():
    planes = split_cfa(_mosaic(), PATTERN, DESC)
    assert set(planes) == {"R", "G1", "G2", "B"}
    assert planes["R"].mean() == pytest.approx(400.0)
    assert planes["B"].mean() == pytest.approx(1200.0)
    assert planes["G1"].mean() == pytest.approx(900.0)


def test_split_and_merge_round_trip_exactly():
    """Any denoiser here writes planes back into a mosaic. If the round trip is
    not exact, the demosaic downstream sees a scrambled pattern and the frame
    is destroyed in a way no colour metric would obviously catch."""
    m = _mosaic(noise=20.0, seed=1)
    planes = split_cfa(m, PATTERN, DESC)
    np.testing.assert_array_equal(merge_cfa(planes, PATTERN, DESC, m.shape), m)


def test_green_difference_measures_noise_with_signal_cancelled():
    """G1 and G2 sample nearly the same point, so their difference has no
    scene content -- the estimate needs no robustness tricks and cannot
    collapse on a dark channel the way MAD does."""
    planes = split_cfa(_mosaic(noise=50.0, seed=2), PATTERN, DESC)
    sigma = green_noise_sigma(planes)
    assert sigma == pytest.approx(50.0, rel=0.15)


def test_green_noise_estimate_survives_a_starved_frame():
    """The case that defeated every RGB-domain estimator: a near-black channel.
    Here the estimate comes from green, which is never starved, and it is a
    difference of two measurements rather than a percentile of one."""
    planes = split_cfa(_mosaic(r=8.0, g=60.0, b=90.0, noise=6.0, seed=3), PATTERN, DESC)
    assert green_noise_sigma(planes) == pytest.approx(6.0, rel=0.3)


def test_denoise_reduces_noise_in_every_plane():
    noisy = _mosaic(noise=40.0, seed=4)
    out = denoise_cfa(noisy, PATTERN, DESC, strength=0.8)
    before = split_cfa(noisy, PATTERN, DESC)
    after = split_cfa(out, PATTERN, DESC)
    for key in ("R", "G1", "G2", "B"):
        assert after[key].std() < before[key].std() * 0.7, key


def test_denoise_preserves_mosaic_geometry():
    """Shape and CFA phase must survive, or the demosaic downstream is wrong."""
    noisy = _mosaic(noise=30.0, seed=5)
    out = denoise_cfa(noisy, PATTERN, DESC, strength=0.5)
    assert out.shape == noisy.shape
    # the red photosites must still be the darkest plane
    planes = split_cfa(out, PATTERN, DESC)
    assert planes["R"].mean() < planes["G1"].mean() < planes["B"].mean()


def test_denoise_keeps_an_edge_present_in_the_scene():
    """Structure appears in every plane at once, so a CFA guide should hold it."""
    m = _mosaic(noise=25.0, seed=6)
    m[:, m.shape[1] // 2:] *= 2.0                       # a scene edge
    out = denoise_cfa(m, PATTERN, DESC, strength=0.8)
    planes = split_cfa(out, PATTERN, DESC)
    half = planes["R"].shape[1] // 2
    assert planes["R"][:, half + 6:].mean() > planes["R"][:, :half - 6].mean() * 1.6


def test_an_unexpected_pattern_is_refused():
    """A CFA with the wrong number of greens is not a Bayer frame, and writing
    planes back under that assumption would scramble the mosaic."""
    with pytest.raises(ValueError, match="Bayer"):
        split_cfa(_mosaic(), np.array([[0, 0], [0, 0]]), "RGBG")
