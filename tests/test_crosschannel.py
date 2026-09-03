"""Cross-channel denoising: filter the starved channel, guided by the clean one.

Two ideas, both exploiting the same asymmetry. Red is photon-starved, so it is
the noisiest channel (water-region high-frequency energy 9.50 vs green's 6.88
on dive 370). But luminance is 0.21R + 0.72G + 0.07B, so the structure a
labeler reads -- scale texture, fin edges -- lives mostly in *green*. Red noise
therefore shows up as chroma speckle: visually loud, structurally useless.

  * **red-only** filters red hard and leaves green and blue untouched.
  * **guided** uses green as a guide to filter red and blue. Edges are shared
    across channels because they are edges of the same objects, so the guide
    supplies structure that genuinely exists in the same exposure.

The second is worth distinguishing from a generative denoiser. A GAN invents
detail that is not in the data; a guided filter transfers detail from another
measurement of the same scene at the same instant. If the guide has no edge
there, none is created -- which is the property the tests below pin.
"""

import numpy as np
import pytest

from enhancement_eval.crosschannel import guided_filter, denoise_channels


# --------------------------------------------------------------------------
# the guided filter itself
# --------------------------------------------------------------------------


def test_guided_filter_smooths_noise_in_a_flat_region():
    rng = np.random.default_rng(0)
    guide = np.full((64, 64), 0.5)
    noisy = np.clip(guide + rng.normal(0, 0.05, guide.shape), 0, 1)
    out = guided_filter(guide, noisy, radius=4, eps=1e-3)
    assert out.std() < noisy.std() / 2


def test_guided_filter_keeps_an_edge_the_guide_has():
    guide = np.zeros((64, 64)); guide[:, 32:] = 1.0
    noisy = guide.copy()
    out = guided_filter(guide, noisy, radius=4, eps=1e-4)
    assert out[:, :28].mean() < 0.15
    assert out[:, 36:].mean() > 0.85


def test_guided_filter_does_not_invent_an_edge_the_guide_lacks():
    """The property that separates this from a generative denoiser: structure
    is transferred from another measurement of the same scene, never created.
    A flat guide can only produce a flat result."""
    rng = np.random.default_rng(1)
    guide = np.full((64, 64), 0.5)
    speckled = np.clip(guide + rng.normal(0, 0.08, guide.shape), 0, 1)
    out = guided_filter(guide, speckled, radius=6, eps=1e-3)
    assert out.std() < 0.02


def test_guided_filter_is_shape_preserving():
    g = np.random.default_rng(2).random((48, 72))
    assert guided_filter(g, g, radius=3, eps=1e-3).shape == (48, 72)


def test_a_larger_eps_smooths_more():
    rng = np.random.default_rng(3)
    guide = np.clip(np.linspace(0, 1, 64)[None, :].repeat(64, 0), 0, 1)
    noisy = np.clip(guide + rng.normal(0, 0.05, guide.shape), 0, 1)
    soft = guided_filter(guide, noisy, radius=4, eps=1e-1)
    sharp = guided_filter(guide, noisy, radius=4, eps=1e-5)
    assert abs(soft - guide).mean() > abs(sharp - guide).mean()


# --------------------------------------------------------------------------
# applying it to a frame
# --------------------------------------------------------------------------


def test_red_only_leaves_green_and_blue_bit_identical():
    """The cheap version of the idea: if only red is touched, no luminance
    structure can be lost, because green carries 72% of luminance."""
    rng = np.random.default_rng(4)
    img = np.clip(rng.random((64, 64, 3)), 0, 1)
    out = denoise_channels(img, mode="red-only", strength=0.5)
    np.testing.assert_array_equal(out[:, :, 1], img[:, :, 1])
    np.testing.assert_array_equal(out[:, :, 2], img[:, :, 2])
    assert not np.array_equal(out[:, :, 0], img[:, :, 0])


def test_red_only_reduces_red_noise():
    rng = np.random.default_rng(5)
    img = np.zeros((96, 96, 3))
    img[:, :, 0] = np.clip(0.3 + rng.normal(0, 0.06, (96, 96)), 0, 1)
    img[:, :, 1] = 0.5
    img[:, :, 2] = 0.6
    out = denoise_channels(img, mode="red-only", strength=0.5)
    assert out[:, :, 0].std() < img[:, :, 0].std() / 2


def test_guided_mode_leaves_the_guide_channel_untouched():
    """Green is the guide and the cleanest channel; filtering it would throw
    away the very structure being borrowed."""
    rng = np.random.default_rng(6)
    img = np.clip(rng.random((64, 64, 3)), 0, 1)
    out = denoise_channels(img, mode="guided", strength=0.5)
    np.testing.assert_array_equal(out[:, :, 1], img[:, :, 1])


def test_guided_mode_preserves_structure_shared_with_green():
    """A fish edge appears in every channel. Guided filtering should keep it in
    red while removing red's independent noise."""
    rng = np.random.default_rng(7)
    img = np.zeros((96, 96, 3))
    edge = np.zeros((96, 96)); edge[:, 48:] = 1.0
    img[:, :, 1] = edge * 0.6 + 0.2                      # clean green
    img[:, :, 2] = edge * 0.5 + 0.25
    img[:, :, 0] = np.clip(edge * 0.4 + 0.15 + rng.normal(0, 0.07, (96, 96)), 0, 1)
    out = denoise_channels(img, mode="guided", strength=0.5)
    left, right = out[:, :40, 0], out[:, 56:, 0]
    assert right.mean() - left.mean() > 0.25            # edge survived
    assert right.std() < img[:, 56:, 0].std() / 2       # noise gone


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        denoise_channels(np.zeros((8, 8, 3)), mode="sideways", strength=0.5)


def test_denoise_moves_no_pixels():
    from enhancement_eval.contract import probe_geometry

    for mode in ("red-only", "guided"):
        def enhance(a, mode=mode):
            f = a.astype(np.float64) / 255.0
            return (np.clip(denoise_channels(f, mode=mode, strength=0.5), 0, 1) * 255).astype(a.dtype)

        assert probe_geometry(enhance, name=mode).displacement_px < 0.5


def test_noise_estimate_survives_a_starved_channel():
    """The bug this caught: `estimate_noise` quantised to uint8 for its median
    filter. On a linear underwater frame the red channel occupies roughly the
    bottom 5% of the range, so x255 leaves ~13 levels and the MAD collapses to
    exactly zero -- making eps zero and the whole filter a passthrough. Red-only
    denoising silently did nothing at all.
    """
    from enhancement_eval.crosschannel import estimate_noise

    rng = np.random.default_rng(11)
    starved = np.clip(0.02 + rng.normal(0, 0.004, (128, 128)), 0, 1)
    sigma = estimate_noise(starved)
    assert sigma > 0.0005, "noise estimate collapsed on a dark channel"
    assert sigma == pytest.approx(0.004, rel=0.6)


def test_red_only_actually_filters_a_starved_red_channel():
    """End of the same bug, at the level a caller sees."""
    rng = np.random.default_rng(12)
    img = np.zeros((128, 128, 3))
    img[:, :, 0] = np.clip(0.02 + rng.normal(0, 0.004, (128, 128)), 0, 1)  # starved red
    img[:, :, 1] = np.clip(0.35 + rng.normal(0, 0.01, (128, 128)), 0, 1)
    img[:, :, 2] = np.clip(0.45 + rng.normal(0, 0.01, (128, 128)), 0, 1)
    out = denoise_channels(img, mode="red-only", strength=0.9)
    assert out[:, :, 0].std() < img[:, :, 0].std() * 0.7


def test_noise_estimate_survives_a_quantised_channel():
    """The second half of the same bug, and the harder half.

    Linear red in open water averages 0.0129 -- so close to the sensor floor
    that it is coarsely quantised. More than half of all pixels then equal
    their own 3x3 median *exactly*, so the median absolute deviation is 0 no
    matter how carefully the median is computed. MAD is simply the wrong
    estimator for a starved channel, and every threshold derived from it
    becomes zero.
    """
    from enhancement_eval.crosschannel import estimate_noise

    rng = np.random.default_rng(13)
    # values on a coarse lattice, as a near-black 16-bit channel really is
    quantised = np.round(
        np.clip(0.013 + rng.normal(0, 0.0015, (128, 128)), 0, 1) * 65535
    ) / 65535.0
    assert estimate_noise(quantised) > 0.0002
