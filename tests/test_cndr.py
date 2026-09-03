"""CNDR: color-noise decoupling and reconstruction (Yan et al. 2026).

Journal of Ocean Engineering and Marine Energy 12:1151-1163,
doi:10.1007/s40722-026-00480-7. No code was released, so this is a
reimplementation from the equations; these tests are what stand in for a
reference implementation to check it against.

Why this method is worth implementing here: its stated problem is the one this
project measured independently. Paper eq 3 -- enhancement applies a per-channel
gain, so `y_hat = k_c*x + k_c*n`, and the noise gets the same gain as the
signal, worst in red because red needs the most gain. That is the CLAHE
water-noise amplification (36.7x) that made us turn CLAHE off in favour of an
L* stretch. The paper's DAC also avoids red overcompensation by compensating
from green and blue rather than from red, which is a principled alternative to
the `red_boost` knob we tuned by eye.

The tests that actually matter for shipping are the invariance ones:
`test_swt_is_shift_invariant` and the geometry probe. Constraint #1 on this
project is strictly pixel-wise -- no warp, resize, or crop -- and an undecimated
transform is the reason CNDR can satisfy it where a 256x256 GAN cannot.
"""

import numpy as np
import pytest

from enhancement_eval.cndr import (
    CNDRConfig,
    HAAR_LEVEL_GAIN,
    adaptive_gamma_correction,
    cndr_enhance,
    dual_stage_attenuation_compensation,
    feature_preserving_contrast,
    iswt2_haar,
    lowpass_synthesis,
    ll_max_for_levels,
    reference_guided_contrast,
    stretch_to_range,
    swt2_haar,
    wavelet_noise_sigma,
)


def _rng():
    return np.random.default_rng(20260903)


def _image(h=64, w=48, c=3):
    """A smooth gradient plus texture plus noise -- something with content at
    every frequency, so a transform bug cannot hide in a flat field."""
    r = _rng()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 40 + 60 * (xx / w) + 40 * (yy / h)
    tex = 18 * np.sin(xx / 2.5) * np.cos(yy / 3.5)
    img = np.stack([base + tex + 25, base + tex, base * 0.8 + tex], axis=-1)
    img += r.normal(0, 3, size=img.shape)
    if c == 1:
        img = img[..., :1]
    return np.clip(img, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

def test_swt_reconstructs_exactly():
    img = _image()[..., 0]
    ll, details = swt2_haar(img, levels=3)
    back = iswt2_haar(ll, details)
    assert np.allclose(back, img, atol=1e-3), f"max err {np.abs(back - img).max()}"


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_swt_reconstructs_exactly_at_every_level_count(levels):
    img = _image()[..., 0]
    ll, details = swt2_haar(img, levels=levels)
    assert len(details) == levels
    assert np.allclose(iswt2_haar(ll, details), img, atol=1e-3)


def test_every_subband_keeps_the_input_shape():
    """This is what 'stationary'/undecimated means, and it is why the method
    can be pixel-wise: no subsampling, so no coefficient ever stands for a
    different location than the pixel under it. Paper sec 3.1 chooses SWT over
    DWT for exactly this reason."""
    img = _image()[..., 0]
    ll, details = swt2_haar(img, levels=3)
    assert ll.shape == img.shape
    for level in details:
        for band in level:
            assert band.shape == img.shape


def test_swt_is_shift_invariant():
    """The property constraint #1 depends on.

    A decimated DWT is not shift invariant: translate the input by one pixel and
    the coefficients change character, so any coefficient-domain edit lands
    differently depending on where the content happens to sit. Undecimated
    coefficients simply translate with the image, so an edit is pixel-wise.
    """
    img = _image()[..., 0]
    shifted = np.roll(img, (1, 1), axis=(0, 1))
    ll_a, det_a = swt2_haar(img, levels=3)
    ll_b, det_b = swt2_haar(shifted, levels=3)
    assert np.allclose(np.roll(ll_a, (1, 1), axis=(0, 1)), ll_b, atol=1e-3)
    for la, lb in zip(det_a, det_b):
        for a, b in zip(la, lb):
            assert np.allclose(np.roll(a, (1, 1), axis=(0, 1)), b, atol=1e-3)


def test_low_frequency_gain_matches_the_papers_2040_constant():
    """Paper sec 3.3 states Cmax = 2040 for a 3-level SWT. That is 255 * 2**3,
    which pins the convention: unnormalized Haar, each level summing rather
    than averaging, so the LL range doubles per level. Getting this wrong would
    silently rescale every DAC compensation term."""
    assert HAAR_LEVEL_GAIN == 2
    assert ll_max_for_levels(3, peak=255.0) == 2040.0
    flat = np.full((16, 16), 255.0, dtype=np.float32)
    ll, _ = swt2_haar(flat, levels=3)
    assert np.allclose(ll, 2040.0)


def test_detail_bands_of_a_flat_field_are_zero():
    ll, details = swt2_haar(np.full((16, 16), 120.0, dtype=np.float32), levels=3)
    for level in details:
        for band in level:
            assert np.allclose(band, 0.0, atol=1e-6)


def test_lowpass_synthesis_matches_a_full_inverse_with_zero_details():
    """The optimization the full-size path depends on. ISWT is linear, so
    reconstructing with a modified LL and untouched details equals the original
    image plus the lowpass synthesis of the LL delta -- which means stage 1
    never has to hold nine detail subbands for a 12-megapixel frame."""
    img = _image()[..., 0]
    ll, details = swt2_haar(img, levels=3)
    delta = ll * 0.1 + 3.0
    zeros = [tuple(np.zeros_like(b) for b in level) for level in details]
    assert np.allclose(lowpass_synthesis(delta, levels=3), iswt2_haar(delta, zeros), atol=1e-3)


def test_modifying_only_ll_equals_image_plus_synthesis_of_the_delta():
    img = _image()[..., 0]
    ll, details = swt2_haar(img, levels=3)
    new_ll = ll * 1.05 + 2.0
    direct = iswt2_haar(new_ll, details)
    cheap = img + lowpass_synthesis(new_ll - ll, levels=3)
    assert np.allclose(direct, cheap, atol=1e-3)


# ---------------------------------------------------------------------------
# DAC -- dual-stage attenuation compensation (paper eqs 6-10)
# ---------------------------------------------------------------------------

def test_dac_lifts_the_starved_channels_of_a_blue_green_cast():
    """The ordinary underwater case: red is gone, blue and green survive."""
    r = _rng()
    ll = np.stack([
        np.full((32, 32), 0.08) + r.normal(0, 0.01, (32, 32)),
        np.full((32, 32), 0.45) + r.normal(0, 0.01, (32, 32)),
        np.full((32, 32), 0.60) + r.normal(0, 0.01, (32, 32)),
    ], axis=-1).astype(np.float32)
    out = dual_stage_attenuation_compensation(ll)
    spread_before = np.ptp(ll.reshape(-1, 3).mean(0))
    spread_after = np.ptp(out.reshape(-1, 3).mean(0))
    assert spread_after < spread_before, "DAC should reduce the channel imbalance"
    assert out[..., 0].mean() > ll[..., 0].mean(), "red must come up"


def test_dac_does_not_overcompensate_red_the_way_a_plain_gain_does():
    """Paper contribution 2, and the reason this is worth testing against our
    own `red_boost`. A naive per-channel gain equalizing the means multiplies
    red by mean_max/mean_red -- here about 7x, which multiplies red *noise* by
    7x too (paper eq 3). DAC instead adds a term built from green and blue, so
    the red noise is not amplified by that factor.

    The test is on noise, not on brightness, because noise amplification is the
    failure we actually measured.
    """
    r = _rng()
    red = np.full((64, 64), 0.08) + r.normal(0, 0.02, (64, 64))
    ll = np.stack([red, np.full((64, 64), 0.45), np.full((64, 64), 0.60)], axis=-1).astype(np.float32)

    naive_gain = ll[..., 2].mean() / ll[..., 0].mean()
    naive_red_noise = (ll[..., 0] * naive_gain).std()
    dac_red_noise = dual_stage_attenuation_compensation(ll)[..., 0].std()

    assert naive_gain > 5, "test premise: a plain gain would be large here"
    assert dac_red_noise < naive_red_noise, (
        f"DAC amplified red noise to {dac_red_noise:.4f}; a plain gain gives {naive_red_noise:.4f}"
    )


def test_dac_is_not_neutral_on_an_already_balanced_image():
    """A caveat that decides where DAC can go in our pipeline, so it is pinned
    here rather than left to be discovered on real frames.

    Eq 6-7 add `(1 - mean_c) * Cr` to green and blue unconditionally. On a
    balanced mid-grey input that adds a large constant to both -- DAC is a
    correction that assumes an uncorrected input, not an idempotent one. Our
    decode already applies a white balance before this point, so DAC must
    *replace* that white balance rather than follow it; stacking the two would
    double-correct. `test_cndr_reduces_the_channel_imbalance_of_a_cast_image`
    covers the case it is actually for.
    """
    ll = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = dual_stage_attenuation_compensation(ll)
    assert np.ptp(out.reshape(-1, 3).mean(0)) > 0.02, (
        "if this ever becomes neutral the caveat above is stale -- recheck the ordering"
    )
    assert out[..., 1].mean() > ll[..., 1].mean()
    assert out[..., 2].mean() > ll[..., 2].mean()


def test_dac_is_pixel_wise_in_the_sense_that_matters():
    """No spatial mixing: every output pixel depends only on the same pixel and
    on per-channel scalars. Shifting the input shifts the output exactly."""
    r = _rng()
    ll = r.uniform(0.05, 0.7, (24, 24, 3)).astype(np.float32)
    a = dual_stage_attenuation_compensation(ll)
    b = dual_stage_attenuation_compensation(np.roll(ll, 3, axis=1))
    assert np.allclose(np.roll(a, 3, axis=1), b, atol=1e-5)


# ---------------------------------------------------------------------------
# Stretch (eq 11)
# ---------------------------------------------------------------------------

def test_stretch_maps_to_the_requested_range():
    r = _rng()
    x = r.uniform(0.1, 0.6, (20, 20)).astype(np.float32)
    out = stretch_to_range(x, 0.0, 2040.0)
    assert np.isclose(out.min(), 0.0, atol=1e-3)
    assert np.isclose(out.max(), 2040.0, atol=1e-3)


def test_stretch_of_a_constant_channel_does_not_divide_by_zero():
    out = stretch_to_range(np.full((8, 8), 0.3, dtype=np.float32), 0.0, 2040.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# RCR (eqs 12-13)
# ---------------------------------------------------------------------------

def test_rcr_is_algebraically_a_scalar_gain():
    """Worth stating in a test because the paper dresses it up.

    Eq 12 is L_en = U (mu * Sigma) V^T with mu a scalar from eq 13. Since
    U (mu*Sigma) V^T = mu * (U Sigma V^T) = mu * L, the whole SVD subsection
    collapses to multiplying the L channel by a scalar. We implement what it
    actually is -- which also means no 3000x4000 SVD per frame -- and this test
    is the evidence that the shortcut is exact, not an approximation.
    """
    r = _rng()
    ll = r.uniform(10, 200, (40, 40)).astype(np.float32)
    ref = ll * 1.4
    out = reference_guided_contrast(ll, ref)

    u, s, vt = np.linalg.svd(ll.astype(np.float64), full_matrices=False)
    _, s_ref, _ = np.linalg.svd(ref.astype(np.float64), full_matrices=False)
    mu = (s_ref.max() + s.max()) / (2 * s.max())
    literal = (u * (mu * s)) @ vt

    assert np.allclose(out, literal, rtol=1e-4, atol=1e-3)
    assert np.allclose(out, mu * ll, rtol=1e-4, atol=1e-3)


def test_rcr_gain_is_one_when_the_reference_is_the_input():
    r = _rng()
    ll = r.uniform(10, 200, (30, 30)).astype(np.float32)
    assert np.allclose(reference_guided_contrast(ll, ll), ll, rtol=1e-5)


def test_rcr_brightens_when_the_reference_has_more_range():
    r = _rng()
    ll = r.uniform(10, 100, (30, 30)).astype(np.float32)
    assert reference_guided_contrast(ll, ll * 2.0).mean() > ll.mean()


# ---------------------------------------------------------------------------
# FCR (eqs 14-16)
# ---------------------------------------------------------------------------

def test_fcr_suppresses_small_coefficients_more_than_large_ones():
    """Paper eq 14: alpha = log(|H|+1)/log(max|H|+1), so alpha is near 0 for
    noise-scale coefficients and near 1 for real edges. This is the noise
    suppression, and it is a different mechanism from the cross-channel RGB
    denoising that failed here -- that one failed because demosaicing
    correlates the channels, which has no bearing on within-channel subband
    shrinkage.
    """
    detail = np.array([[0.05, 0.1], [40.0, 80.0]], dtype=np.float32)
    ref = detail.copy()
    out = feature_preserving_contrast([(detail, detail, detail)], [(ref, ref, ref)])[0][0]
    small_gain = out[0, 0] / detail[0, 0]
    large_gain = out[1, 1] / detail[1, 1]
    assert small_gain < large_gain, f"small {small_gain:.3f} should gain less than large {large_gain:.3f}"


def test_fcr_never_drops_a_coefficient_below_the_original():
    """Eq 16 adds the original back: H_en = beta*(alpha*H) + H. So features
    cannot be filtered away, which is the paper's answer to the blur that
    denoising preprocessing causes (its Fig 2d-e)."""
    r = _rng()
    detail = r.normal(0, 10, (16, 16)).astype(np.float32)
    out = feature_preserving_contrast([(detail, detail, detail)], [(detail, detail, detail)])[0][0]
    assert np.all(np.abs(out) >= np.abs(detail) - 1e-4)


def test_fcr_handles_an_all_zero_subband():
    zero = np.zeros((8, 8), dtype=np.float32)
    out = feature_preserving_contrast([(zero, zero, zero)], [(zero, zero, zero)])[0][0]
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Adaptive gamma correction (Huang et al. 2013), the reference image
# ---------------------------------------------------------------------------

def test_agcwd_brightens_a_dark_image():
    r = _rng()
    dark = np.clip(r.normal(40, 10, (64, 64)), 0, 255).astype(np.float32)
    assert adaptive_gamma_correction(dark).mean() > dark.mean()


def test_agcwd_preserves_range_and_is_monotonic():
    """It is a per-intensity lookup, so it cannot reorder pixel intensities --
    which is what keeps it from inventing structure."""
    r = _rng()
    img = np.clip(r.normal(120, 40, (64, 64)), 0, 255).astype(np.float32)
    out = adaptive_gamma_correction(img)
    assert out.min() >= -1e-3 and out.max() <= 255 + 1e-3
    order_in = np.argsort(img.ravel())
    vals = out.ravel()[order_in]
    assert np.all(np.diff(vals) >= -1e-4), "a lookup must be non-decreasing in intensity"


def test_agcwd_of_a_flat_image_is_finite():
    assert np.all(np.isfinite(adaptive_gamma_correction(np.full((8, 8), 50.0, dtype=np.float32))))


# ---------------------------------------------------------------------------
# Noise metric (eq 17) -- the paper's own yardstick
# ---------------------------------------------------------------------------

def test_wavelet_noise_sigma_rises_with_added_noise():
    img = _image()[..., 0]
    r = _rng()
    clean = wavelet_noise_sigma(img)
    noisy = wavelet_noise_sigma(img + r.normal(0, 12, img.shape).astype(np.float32))
    assert noisy > clean * 1.5


def test_wavelet_noise_sigma_recovers_a_known_sigma():
    """MAD/0.6745 on the diagonal subband is the standard estimator, so on pure
    Gaussian noise it should land near the true sigma rather than merely
    ordering correctly."""
    r = _rng()
    sigma = 8.0
    est = wavelet_noise_sigma(r.normal(0, sigma, (256, 256)).astype(np.float32), levels=1)
    assert 0.5 * sigma < est < 2.0 * sigma, f"estimated {est:.2f} for true {sigma}"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_cndr_preserves_shape_and_dtype():
    img = _image().astype(np.uint8)
    out = cndr_enhance(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_cndr_handles_odd_sizes():
    """Undecimated means no power-of-two requirement; a 4014x3016 frame must
    not need padding, because padding would be a geometric change."""
    out = cndr_enhance(_image(h=37, w=53).astype(np.uint8))
    assert out.shape == (37, 53, 3)


def test_cndr_reduces_the_channel_imbalance_of_a_cast_image():
    r = _rng()
    h = w = 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 60 + 40 * (xx / w)
    img = np.stack([base * 0.15, base * 0.75, base], axis=-1)
    img = np.clip(img + r.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    out = cndr_enhance(img)
    before = np.ptp(img.reshape(-1, 3).mean(0))
    after = np.ptp(out.reshape(-1, 3).mean(0))
    assert after < before, f"cast not reduced: {before:.1f} -> {after:.1f}"


def test_cndr_stages_can_be_disabled_independently():
    """Each piece has to be separately testable against what we already ship --
    DAC against `red_boost`, FCR against the CLAHE noise amplification -- so a
    win can be attributed to a stage rather than to the bundle."""
    img = _image().astype(np.uint8)
    everything = cndr_enhance(img, CNDRConfig())
    no_dac = cndr_enhance(img, CNDRConfig(apply_dac=False))
    no_fcr = cndr_enhance(img, CNDRConfig(apply_fcr=False))
    no_rcr = cndr_enhance(img, CNDRConfig(apply_rcr=False))
    assert not np.array_equal(everything, no_dac)
    assert not np.array_equal(everything, no_fcr)
    assert not np.array_equal(everything, no_rcr)


def test_cndr_with_every_stage_off_is_near_identity():
    """The transform round trip must not itself change the image; otherwise any
    measured effect is confounded with reconstruction error."""
    img = _image().astype(np.uint8)
    out = cndr_enhance(img, CNDRConfig(apply_dac=False, apply_rcr=False, apply_fcr=False))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_cndr_does_not_move_pixels():
    """Constraint #1, enforced with the harness's own probe rather than by
    inspection. This is the check a 256x256 GAN cannot pass at native
    resolution."""
    from enhancement_eval.contract import MAX_DISPLACEMENT_PX, probe_geometry

    result = probe_geometry(lambda a: cndr_enhance(a), name="cndr")
    assert result.displacement_px < MAX_DISPLACEMENT_PX, result


def test_cndr_is_deterministic():
    img = _image().astype(np.uint8)
    assert np.array_equal(cndr_enhance(img), cndr_enhance(img))
