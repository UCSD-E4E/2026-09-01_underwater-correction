"""Is the sensor noise the kind Noise2Void can actually remove?

Noise2Void trains a blind-spot network: it hides a pixel and predicts it from
its neighbourhood, and the prediction converges on the clean signal *only*
because the noise carries no information about itself across pixels. Two
assumptions hold that up:

  1. Noise is spatially independent given the signal. If neighbouring pixels
     share noise, the network reads that shared component as signal, predicts
     it, and preserves it. The result is a no-op that still trains to a
     convincing-looking loss curve.
  2. Noise is zero-mean per pixel. A fixed-pattern (per-photosite) offset is
     identical in every frame, so a blind-spot network cannot see it as noise
     at all -- and cannot remove it.

This module measures both before any model is built, which is the same
discipline the rest of the project runs on: a training loss is not evidence,
and neither is a denoised image that looks smoother.

The measurement leans on the G1-G2 trick already used for `green_noise_sigma`:
the two green photosites sample nearly the same scene point, so their
difference cancels the signal and leaves noise. Its spatial autocorrelation is
therefore the *noise* autocorrelation, with no assumption about how much of the
frame is flat.

One caveat this module has to respect: G1 and G2 sit at different positions, so
on high-frequency scene content the difference is not purely noise. Sampling
open water rather than the whole frame is what keeps scene structure out of the
estimate -- the same open-water convention as every other noise figure in
MEASUREMENTS.md.
"""

import numpy as np
import pytest

from enhancement_eval.noise_structure import (
    DEFAULT_LAGS,
    NOISE_INDEPENDENCE_MAX,
    NoiseStructure,
    assess_n2v_viability,
    fixed_pattern_fraction,
    noise_field,
    recommended_mask_width,
    spatial_autocorrelation,
)


def _rng(seed=20260903):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# The signal-free noise field
# ---------------------------------------------------------------------------

def test_noise_field_cancels_a_smooth_signal():
    """A gradient common to both greens must leave nothing behind."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    ramp = 100 + 0.5 * xx + 0.3 * yy
    planes = {"G1": ramp.copy(), "G2": ramp.copy()}
    assert np.allclose(noise_field(planes), 0.0, atol=1e-9)


def test_noise_field_recovers_the_per_photosite_sigma():
    """Var(G1-G2) = 2*sigma^2 for independent photosites, so the field is
    divided by sqrt(2) and reads back as sigma itself, not sigma*sqrt(2)."""
    r = _rng()
    sigma = 7.0
    planes = {"G1": r.normal(0, sigma, (256, 256)), "G2": r.normal(0, sigma, (256, 256))}
    assert 0.9 * sigma < noise_field(planes).std() < 1.1 * sigma


# ---------------------------------------------------------------------------
# Spatial autocorrelation -- assumption 1
# ---------------------------------------------------------------------------

def test_autocorrelation_of_white_noise_is_zero_away_from_the_origin():
    r = _rng()
    ac = spatial_autocorrelation(r.normal(0, 5, (512, 512)), DEFAULT_LAGS)
    assert np.isclose(ac[(0, 0)], 1.0, atol=1e-6)
    for lag, value in ac.items():
        if lag != (0, 0):
            assert abs(value) < 0.05, f"white noise correlated at {lag}: {value:.3f}"


def test_autocorrelation_detects_smoothed_noise():
    """The failure mode this gate exists to catch. Noise that has been through
    any spatial filter -- a denoiser upstream, an interpolation, LibRaw's own
    preprocessing -- correlates with its neighbours, and N2V then preserves it."""
    from scipy.ndimage import gaussian_filter

    r = _rng()
    smoothed = gaussian_filter(r.normal(0, 5, (512, 512)), sigma=1.0)
    ac = spatial_autocorrelation(smoothed, DEFAULT_LAGS)
    assert ac[(0, 1)] > 0.5
    assert ac[(1, 0)] > 0.5


def test_autocorrelation_is_symmetric_for_isotropic_noise():
    r = _rng()
    ac = spatial_autocorrelation(r.normal(0, 3, (400, 400)), DEFAULT_LAGS)
    assert abs(ac[(0, 1)] - ac[(1, 0)]) < 0.05


def test_autocorrelation_of_a_constant_field_is_finite():
    ac = spatial_autocorrelation(np.full((32, 32), 4.0), DEFAULT_LAGS)
    assert all(np.isfinite(v) for v in ac.values())


# ---------------------------------------------------------------------------
# Fixed-pattern noise -- assumption 2
# ---------------------------------------------------------------------------

def test_fixed_pattern_fraction_is_near_zero_for_independent_frames():
    """Bias correction matters here: the mean of N independent fields still has
    variance sigma^2/N, so an uncorrected ratio would report 1/N of fixed
    pattern where there is none."""
    r = _rng()
    fields = [r.normal(0, 5, (128, 128)) for _ in range(8)]
    assert abs(fixed_pattern_fraction(fields)) < 0.08


def test_fixed_pattern_fraction_is_near_one_when_the_pattern_dominates():
    r = _rng()
    pattern = r.normal(0, 5, (128, 128))
    fields = [pattern + r.normal(0, 0.3, (128, 128)) for _ in range(8)]
    assert fixed_pattern_fraction(fields) > 0.9


def test_fixed_pattern_fraction_lands_mid_range_for_a_mixture():
    r = _rng()
    pattern = r.normal(0, 4, (192, 192))
    fields = [pattern + r.normal(0, 4, (192, 192)) for _ in range(10)]
    frac = fixed_pattern_fraction(fields)
    assert 0.35 < frac < 0.65, f"expected about half, got {frac:.3f}"


def test_fixed_pattern_fraction_needs_at_least_two_fields():
    with pytest.raises(ValueError):
        fixed_pattern_fraction([np.zeros((8, 8))])


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def test_viability_passes_for_clean_independent_noise():
    r = _rng()
    result = assess_n2v_viability([r.normal(0, 5, (256, 256)) for _ in range(6)])
    assert isinstance(result, NoiseStructure)
    assert result.viable, result.verdict
    assert result.max_offdiagonal < NOISE_INDEPENDENCE_MAX


def test_viability_fails_on_spatially_correlated_noise():
    """A verdict that says 'do not train' has to be reachable, or the gate is
    decoration."""
    from scipy.ndimage import gaussian_filter

    r = _rng()
    fields = [gaussian_filter(r.normal(0, 5, (256, 256)), 1.0) for _ in range(6)]
    result = assess_n2v_viability(fields)
    assert not result.viable
    assert "correlat" in result.verdict.lower()


def test_viability_fails_on_dominant_fixed_pattern():
    r = _rng()
    pattern = r.normal(0, 5, (256, 256))
    fields = [pattern + r.normal(0, 0.3, (256, 256)) for _ in range(6)]
    result = assess_n2v_viability(fields)
    assert not result.viable
    assert "fixed" in result.verdict.lower()


def test_verdict_names_the_numbers_it_judged_on():
    """So a run's output can be pasted into MEASUREMENTS.md without rerunning it."""
    r = _rng()
    result = assess_n2v_viability([r.normal(0, 5, (128, 128)) for _ in range(4)])
    assert f"{result.max_offdiagonal:.3f}" in result.verdict
    assert f"{result.fixed_pattern_fraction:.3f}" in result.verdict


# ---------------------------------------------------------------------------
# Anisotropy -- which shape the blind spot has to be
# ---------------------------------------------------------------------------

def test_isotropic_noise_needs_only_a_point_blind_spot():
    r = _rng()
    result = assess_n2v_viability([r.normal(0, 5, (256, 256)) for _ in range(5)])
    assert result.horizontal_anisotropy < 2.0
    assert recommended_mask_width(result) == 1


def test_horizontally_correlated_noise_asks_for_a_wider_mask():
    """The measured case. Our sensor's noise correlates along the readout
    direction only -- r(0,1) up to 0.162 on bright pool frames, decaying
    geometrically, against ~0.01 vertically. A single-pixel blind spot leaves
    the correlated horizontal neighbour visible for the network to copy from,
    which is what StructN2V's line mask exists to prevent.
    """
    r = _rng()
    fields = []
    for _ in range(5):
        n = r.normal(0, 5, (256, 256))
        # First-order horizontal smoothing: an analog readout bandwidth limit,
        # correlated along rows and independent down columns.
        for k in range(1, n.shape[1]):
            n[:, k] += 0.45 * n[:, k - 1]
        fields.append(n)
    result = assess_n2v_viability(fields)
    assert result.horizontal_anisotropy > 3.0
    assert recommended_mask_width(result) > 1
    assert recommended_mask_width(result) % 2 == 1, "a centred mask needs odd width"


def test_mask_width_covers_the_correlation_length():
    """The width has to reach past where the correlation has decayed, or the
    network can still read a correlated neighbour just outside the mask."""
    r = _rng()
    fields = []
    for _ in range(4):
        n = r.normal(0, 5, (256, 256))
        for k in range(1, n.shape[1]):
            n[:, k] += 0.7 * n[:, k - 1]      # longer correlation length
        fields.append(n)
    slow = recommended_mask_width(assess_n2v_viability(fields))
    fields2 = []
    for _ in range(4):
        n = r.normal(0, 5, (256, 256))
        for k in range(1, n.shape[1]):
            n[:, k] += 0.3 * n[:, k - 1]      # shorter
        fields2.append(n)
    fast = recommended_mask_width(assess_n2v_viability(fields2))
    assert slow >= fast


def test_verdict_mentions_the_mask_when_noise_is_anisotropic():
    r = _rng()
    fields = []
    for _ in range(4):
        n = r.normal(0, 5, (256, 256))
        for k in range(1, n.shape[1]):
            n[:, k] += 0.5 * n[:, k - 1]
        fields.append(n)
    assert "mask" in assess_n2v_viability(fields).verdict.lower()
