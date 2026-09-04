"""BM3D on the luminance of the finished decode, with a measured noise PSD.

Why this, after Noise2Void: every N2V variant erased the dive 223 scale
lattice completely (retention 0.00 for random crops, fish crops, and a
shallow net alike). The lattice's 4.5 px period in the output is 2.25 px in
the half-resolution photosite planes N2V works on -- at Nyquist, where a
blind-spot network cannot tell a periodic pattern from pixel noise. The fix
has to work at full resolution.

BM3D is the classical answer for exactly this texture: it groups similar
patches and filters them jointly, and a periodic scale lattice is as
self-similar as image content gets. It is non-learned, deterministic, and can
only remove. And it takes a noise *power spectral density* rather than a
single sigma, which matters here -- demosaicing colours the noise, and the
open-water spectrum we already measure is that PSD.

It sits at the JPEG stage, after demosaic, on the L channel with a and b left
alone -- so it satisfies constraint #2 and goes through `probe_geometry`
like any other enhancer, which the mosaic-domain work could not.
"""

import numpy as np
import pytest

pytest.importorskip("bm3d")
if not hasattr(np, "trapz"):          # bm3d 4.0 on NumPy 2; see bm3d_arm._import_bm3d
    np.trapz = np.trapezoid

from enhancement_eval.bm3d_arm import (  # noqa: E402
    BM3DConfig,
    bm3d_enhancer,
    denoise_luminance,
    noise_psd_from_water,
)
from enhancement_eval.texture import texture_retention  # noqa: E402


def _rng(seed=20260903):
    return np.random.default_rng(seed)


def _scales(h=128, w=128, period=7.0, amplitude=12.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return amplitude * np.sin(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period)


def _scene(sigma=8.0, seed=1):
    r = _rng(seed)
    clean = 120 + _scales()
    noisy = clean + r.normal(0, sigma, clean.shape)
    water = 120 + r.normal(0, sigma, clean.shape)
    return clean, noisy, water


# ---------------------------------------------------------------------------
# The PSD
# ---------------------------------------------------------------------------

def test_psd_of_white_noise_is_flat_at_the_variance():
    psd = noise_psd_from_water(_rng().normal(0, 6.0, (256, 256)), size=64)
    assert psd.shape == (64, 64)
    assert 0.6 * 36 < np.median(psd) < 1.5 * 36


def test_psd_of_coloured_noise_is_not_flat():
    """The reason a PSD rather than a sigma: after demosaic the noise rolls
    off at high frequency, and a white-noise assumption over-smooths there."""
    from scipy.ndimage import gaussian_filter

    psd = noise_psd_from_water(gaussian_filter(_rng().normal(0, 6.0, (256, 256)), 1.0), size=64)
    centre = psd[28:36, 28:36].mean()
    corner = psd[:4, :4].mean()
    assert centre > 5 * corner


# ---------------------------------------------------------------------------
# Denoising a luminance patch
# ---------------------------------------------------------------------------

def test_denoise_reduces_open_water_noise():
    clean, noisy, water = _scene()
    out = denoise_luminance(noisy, water)
    assert (out - clean).std() < 0.5 * (noisy - clean).std()


def test_denoise_keeps_the_scale_lattice():
    """The test the whole arm exists for. N2V read 0.00 here on the real
    frame; a method worth keeping must hold most of the peak."""
    clean, noisy, water = _scene()
    out = denoise_luminance(noisy, water)
    r = texture_retention(noisy, out)
    assert r > 0.6, f"scale retention {r:.3f}"


def test_denoise_does_not_mutate_input():
    _, noisy, water = _scene()
    before = noisy.copy()
    denoise_luminance(noisy, water)
    assert np.array_equal(noisy, before)


# ---------------------------------------------------------------------------
# As an enhancer in the harness
# ---------------------------------------------------------------------------

FISH = np.s_[80:208, 80:208]


def _rgb_frame(h=256, w=256):
    """Colour gradients, open water in the top-left fifth, and a scale
    lattice confined to a 'fish' region -- the same in every channel, so it
    is luminance texture, which is what real scales are.

    Two earlier versions of this fixture each taught something. One put the
    lattice in blue alone, making it chroma texture that the chroma filter
    then correctly removed. The other put the lattice everywhere, including
    the top-left region the PSD is measured from -- so the noise model
    contained the scales and BM3D erased them frame-wide (retention 0.000).
    Real open water has no scales, but that is a genuine operational risk:
    a PSD region that contains texture removes that texture everywhere.
    """
    r = _rng(5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lattice = np.zeros((h, w))
    lattice[FISH] = _scales(128, 128)
    base = np.stack([60 + 40 * xx / w + lattice, 90 + 30 * yy / h + lattice, 120 + lattice], axis=-1)
    return np.clip(base + r.normal(0, 8, base.shape), 0, 255).astype(np.uint8)


def test_enhancer_preserves_shape_and_dtype():
    out = bm3d_enhancer(BM3DConfig())(_rgb_frame())
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8


def test_enhancer_keeps_luminance_scales_on_a_colour_frame():
    """The end-to-end version of the claim: through rgb2lab, a PSD measured
    from the frame's own open water, and back to uint8."""
    from skimage.color import rgb2lab

    img = _rgb_frame()
    out = bm3d_enhancer(BM3DConfig())(img)
    l_in = rgb2lab(img / 255.0)[..., 0][FISH] * 2.55
    l_out = rgb2lab(out / 255.0)[..., 0][FISH] * 2.55
    r = texture_retention(l_in, l_out)
    assert r > 0.6, f"scale retention through the enhancer {r:.3f}"


def test_enhancer_leaves_colour_alone_by_default():
    """With chroma filtering off, a and b pass through, so hue -- which the
    species and slate labels depend on -- is untouched."""
    from skimage.color import rgb2lab

    img = _rgb_frame()
    out = bm3d_enhancer(BM3DConfig())(img)
    a_in, a_out = rgb2lab(img / 255.0)[..., 1:], rgb2lab(out / 255.0)[..., 1:]
    assert np.abs(a_in - a_out).mean() < 1.5


def _chroma_grain(rgb):
    from scipy.ndimage import median_filter
    from skimage.color import rgb2lab

    ab = rgb2lab(rgb / 255.0)[..., 1:]
    return float(np.mean([np.std(ab[..., i] - median_filter(ab[..., i], 3)) for i in range(2)]))


def test_chroma_filtering_removes_colour_speckle():
    """Scales are luminance texture, so a and b carry no scale information
    and can be filtered hard. What remained after L-only filtering on dive
    223 was a faint coloured mottle in the water and rock."""
    img = _rgb_frame()
    off = bm3d_enhancer(BM3DConfig(chroma_strength=0.0))(img)
    on = bm3d_enhancer(BM3DConfig(chroma_strength=1.0))(img)
    assert _chroma_grain(on) < 0.5 * _chroma_grain(off)


def test_chroma_filtering_keeps_the_mean_colour():
    """Speckle goes, hue stays: the local mean of a and b must not move, or
    species and slate colours would drift."""
    from scipy.ndimage import uniform_filter
    from skimage.color import rgb2lab

    img = _rgb_frame()
    out = bm3d_enhancer(BM3DConfig(chroma_strength=1.0))(img)
    ab_in, ab_out = rgb2lab(img / 255.0)[..., 1:], rgb2lab(out / 255.0)[..., 1:]
    for i in range(2):
        assert np.abs(uniform_filter(ab_in[..., i], 15) - uniform_filter(ab_out[..., i], 15)).mean() < 1.0


def test_chroma_filtering_does_not_touch_luminance_scales():
    from skimage.color import rgb2lab

    img = _rgb_frame()
    l_in = rgb2lab(img / 255.0)[..., 0][FISH] * 2.55
    l_off = rgb2lab(bm3d_enhancer(BM3DConfig(chroma_strength=0.0))(img) / 255.0)[..., 0][FISH] * 2.55
    l_on = rgb2lab(bm3d_enhancer(BM3DConfig(chroma_strength=1.0))(img) / 255.0)[..., 0][FISH] * 2.55
    r_off, r_on = texture_retention(l_in, l_off), texture_retention(l_in, l_on)
    assert abs(r_on - r_off) < 0.1, f"chroma filtering changed L retention {r_off:.3f} -> {r_on:.3f}"


def test_chroma_filtering_does_not_move_pixels():
    from enhancement_eval.contract import MAX_DISPLACEMENT_PX, probe_geometry

    result = probe_geometry(bm3d_enhancer(BM3DConfig(chroma_strength=1.0)), name="bm3d+chroma")
    assert result.displacement_px < MAX_DISPLACEMENT_PX, result


def test_enhancer_does_not_move_pixels():
    from enhancement_eval.contract import MAX_DISPLACEMENT_PX, probe_geometry

    result = probe_geometry(bm3d_enhancer(BM3DConfig()), name="bm3d")
    assert result.displacement_px < MAX_DISPLACEMENT_PX, result


def test_enhancer_is_deterministic_to_one_lsb():
    """The bm3d package is multithreaded and its float reductions land in
    different orders run to run: measured, 6-9 pixels in 76,800 differ, by
    exactly 1 after rounding to uint8. That is rounding, not a random
    component -- but a test that demanded bit-equality would fail, and one
    that allowed more than 1 LSB would hide a real nondeterminism."""
    img = _rgb_frame()
    f = bm3d_enhancer(BM3DConfig())
    a, b = f(img).astype(int), f(img).astype(int)
    assert np.abs(a - b).max() <= 1
    assert (a != b).mean() < 0.001


def test_psd_array_matches_scalar_sigma_on_white_noise():
    """Pins the unit convention handed to the bm3d package. If the factor is
    wrong the filter is silently too weak or too strong everywhere, and no
    other test would notice -- both directions still 'denoise'."""
    import bm3d as _bm3d
    from enhancement_eval.bm3d_arm import _psd_for_image

    r = _rng(77)
    sigma = 0.08
    z = np.clip(0.5 + r.normal(0, sigma, (96, 96)), 0, 1)
    ref = _bm3d.bm3d(z, sigma_psd=sigma, profile="np")
    psd = noise_psd_from_water(r.normal(0, sigma, (256, 256)), size=64)
    ours = _bm3d.bm3d(z, sigma_psd=_psd_for_image(psd, z.shape), profile="np")
    assert np.abs(ours - ref).mean() < 0.01, f"mean abs diff {np.abs(ours - ref).mean():.4f}"
