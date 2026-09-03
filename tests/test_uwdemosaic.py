"""A demosaic that allocates spatial detail by measured per-channel SNR.

Every stock demosaic tested treats the three channels as equally trustworthy.
Underwater they are not: measured on a real frame, green sits at 927.5 DN with
an SNR of 8.3 per photosite while red sits at 110.9 DN with roughly a third of
that. Recovering red's high frequencies faithfully therefore means recovering
its noise faithfully, which is why AHD, DHT, AAHD, DCB, PPG, VNG and linear all
land on the same 1:1 noise-for-detail line -- they differ in *where* on that
line they sit, not in the exchange rate.

This interpolates the colour difference (R - G) instead of R, and smooths it
hard. The justification is physical rather than merely convenient: underwater
the colour difference is set by the water column, which the attenuation fit
showed varies smoothly with range. So red keeps its slow chromatic variation
and inherits its sharp structure from green, which has the photons to support
it.

Not a new algorithm class -- residual / colour-difference interpolation is well
established. What is specific is choosing the frequency split from measured
SNR, and having a physical reason to believe the difference channel is smooth.
"""

import numpy as np
import pytest

from enhancement_eval.uwdemosaic import demosaic_underwater

PATTERN = np.array([[0, 1], [3, 2]])
DESC = "RGBG"


def _mosaic(scene, levels=(0.15, 1.0, 1.3), noise=0.0, seed=0):
    """CFA sample of a scene, with per-channel gain and optional noise."""
    rng = np.random.default_rng(seed)
    h, w = scene.shape
    m = np.zeros((h, w))
    m[0::2, 0::2] = scene[0::2, 0::2] * levels[0]          # R
    m[0::2, 1::2] = scene[0::2, 1::2] * levels[1]          # G
    m[1::2, 0::2] = scene[1::2, 0::2] * levels[1]          # G
    m[1::2, 1::2] = scene[1::2, 1::2] * levels[2]          # B
    if noise:
        m = m + rng.normal(0, noise, m.shape)
    return np.clip(m, 0, None)


def _edge(n=64):
    s = np.full((n, n), 0.3)
    s[:, n // 2:] = 0.8
    return s


def test_output_has_three_channels_and_the_input_size():
    out = demosaic_underwater(_mosaic(_edge()), PATTERN, DESC)
    assert out.shape == (64, 64, 3)


def test_green_structure_is_preserved():
    out = demosaic_underwater(_mosaic(_edge()), PATTERN, DESC)
    g = out[:, :, 1]
    assert g[:, 40:].mean() > g[:, :24].mean() * 2.0


def test_red_inherits_sharpness_from_green():
    """The whole point. Red's own samples are too noisy to carry the edge, so
    it must come from green -- while red keeps its own overall level."""
    out = demosaic_underwater(_mosaic(_edge(), noise=0.05, seed=1), PATTERN, DESC)
    r = out[:, :, 0]
    assert r[:, 40:].mean() > r[:, :24].mean() * 1.8      # edge present in red
    assert r.mean() < out[:, :, 1].mean()                 # still the dim channel


def test_red_noise_is_suppressed_relative_to_a_naive_interpolation():
    """Against interpolating red from its own samples, which is what treating
    every channel as equally trustworthy amounts to.

    The fixture gives red 4x green's noise, because that is the situation this
    exists for: on a real frame green carries twice the photosites and about
    eight times the signal level. With noise equal across channels there is
    nothing to borrow and the method correctly does nothing -- an earlier
    version of this test asserted a gain in exactly that case and failed for
    the right reason.
    """
    rng = np.random.default_rng(2)
    n = 96
    m = np.zeros((n, n))
    m[0::2, 0::2] = 0.15 + rng.normal(0, 0.040, (n // 2, n // 2))   # red: noisy
    m[0::2, 1::2] = 1.00 + rng.normal(0, 0.010, (n // 2, n // 2))   # green: clean
    m[1::2, 0::2] = 1.00 + rng.normal(0, 0.010, (n // 2, n // 2))
    m[1::2, 1::2] = 1.30 + rng.normal(0, 0.012, (n // 2, n // 2))
    m = np.clip(m, 0, None)

    out = demosaic_underwater(m, PATTERN, DESC)
    # Red's own samples carry sigma ~0.040; after borrowing structure from
    # green the reconstructed red should sit far below that.
    naive_sigma = float(m[0::2, 0::2].std())
    assert float(out[:, :, 0].std()) < naive_sigma * 0.6


def test_chromatic_variation_survives_when_it_is_smooth():
    """A slow colour gradient -- which is what attenuation produces -- must not
    be flattened away along with the noise."""
    n = 96
    scene = np.full((n, n), 0.5)
    m = _mosaic(scene, noise=0.0)
    ramp = np.linspace(0.5, 2.0, n)[None, :].repeat(n, 0)
    m[0::2, 0::2] *= ramp[0::2, 0::2]                     # red rises across frame
    out = demosaic_underwater(m, PATTERN, DESC)
    left = out[:, :20, 0].mean() / out[:, :20, 1].mean()
    right = out[:, 76:, 0].mean() / out[:, 76:, 1].mean()
    assert right > left * 1.8


def test_stronger_smoothing_suppresses_more_chroma_noise():
    m = _mosaic(np.full((96, 96), 0.5), noise=0.05, seed=3)
    soft = demosaic_underwater(m, PATTERN, DESC, chroma_radius=12)
    sharp = demosaic_underwater(m, PATTERN, DESC, chroma_radius=2)
    chroma = lambda a: float((a - a.mean(axis=2, keepdims=True)).std())
    assert chroma(soft) < chroma(sharp)


def test_a_non_bayer_pattern_is_refused():
    with pytest.raises(ValueError, match="Bayer"):
        demosaic_underwater(_mosaic(_edge()), np.array([[0, 0], [0, 0]]), "RGBG")
