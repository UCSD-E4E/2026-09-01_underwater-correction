"""Inverting the underwater image formation model with measured coefficients.

    I_c = J_c * rho_c * exp(-beta_c * z)  +  B_c * (1 - exp(-beta_c * z))

Both unknowns are measured here rather than assumed: beta from the slate fit,
z from `LaserDepth.range_m`. That is what separates this from gray-world and
white-patch, which guessed at scene statistics and produced magenta casts on
field frames -- a measured beta cannot over-correct, because it is not inferred
from the picture it is correcting.

The honest limitation is range. Sea-thru proper needs a per-pixel depth map;
this corpus has range at exactly one pixel, so z is uniform across the frame.
That is correct for the fish the laser is on and increasingly wrong for
background at another distance.
"""

import math

import numpy as np
import pytest

from enhancement_eval.seathru import estimate_veil, remove_water


def _observed(j, beta, z, veil, shape=(64, 64)):
    """Synthesize I_c under the model, for a flat target of radiance j."""
    img = np.zeros((*shape, 3))
    for c in range(3):
        t = math.exp(-beta[c] * z)
        img[:, :, c] = j[c] * t + veil[c] * (1 - t)
    return img


def test_recovers_a_known_radiance():
    """The round trip: synthesize an observation, invert it, get the target back."""
    j, beta, z, veil = (0.60, 0.45, 0.30), (0.40, 0.19, 0.21), 2.0, (0.05, 0.12, 0.18)
    out = remove_water(_observed(j, beta, z, veil), beta, z, veil=veil, normalize=False)
    for c in range(3):
        assert out[:, :, c].mean() == pytest.approx(j[c], abs=0.01)


def test_correction_grows_with_range():
    """At 1 m the water column has done little; at 5 m it dominates. A
    correction that did not scale with distance would be a fixed white balance
    wearing a physics costume."""
    beta, veil = (0.40, 0.19, 0.21), (0.0, 0.0, 0.0)
    flat = np.full((32, 32, 3), 0.3)
    near = remove_water(flat, beta, 1.0, veil=veil, normalize=False)
    far = remove_water(flat, beta, 5.0, veil=veil, normalize=False)
    assert far[:, :, 0].mean() > near[:, :, 0].mean()


def test_red_is_corrected_most_when_it_attenuates_most():
    beta = (0.40, 0.19, 0.21)
    flat = np.full((32, 32, 3), 0.3)
    out = remove_water(flat, beta, 3.0, veil=(0.0, 0.0, 0.0), normalize=False)
    gains = [out[:, :, c].mean() / 0.3 for c in range(3)]
    assert gains[0] > gains[2] > gains[1]        # matches beta_r > beta_b > beta_g


def test_zero_attenuation_is_a_no_op():
    """Clear water, or a target at zero range, must leave the frame alone."""
    flat = np.full((16, 16, 3), 0.42)
    out = remove_water(flat, (0.0, 0.0, 0.0), 3.0, veil=(0.0, 0.0, 0.0), normalize=False)
    np.testing.assert_allclose(out, flat, atol=1e-9)


def test_veil_is_estimated_from_the_dark_tail():
    """Where reflectance is near zero the observation is essentially
    backscatter -- the standard dark-channel argument."""
    img = np.full((64, 64, 3), 0.5)
    img[:8, :8] = (0.04, 0.09, 0.14)             # a dark corner
    veil = estimate_veil(img, percentile=1.0)
    assert veil[0] == pytest.approx(0.04, abs=0.02)
    assert veil[2] == pytest.approx(0.14, abs=0.02)


def test_negative_radiance_is_clipped_not_wrapped():
    """Subtracting a veil larger than the signal must floor at zero; letting it
    go negative and renormalising would invert the image."""
    img = np.full((16, 16, 3), 0.05)
    out = remove_water(img, (0.4, 0.2, 0.2), 2.0, veil=(0.5, 0.5, 0.5), normalize=False)
    assert out.min() >= 0.0


def test_a_negative_range_is_rejected():
    with pytest.raises(ValueError, match="range"):
        remove_water(np.full((8, 8, 3), 0.3), (0.4, 0.2, 0.2), -1.0)


def test_correction_moves_no_pixels():
    """Constraint #1 against the physics path too: it is per-pixel arithmetic,
    so the geometry probe must clear it."""
    from enhancement_eval.contract import probe_geometry

    def enhance(a):
        f = a.astype(np.float64) / 255.0
        out = remove_water(f, (0.385, 0.189, 0.206), 1.5)
        return (np.clip(out, 0, 1) * 255).astype(a.dtype)

    assert probe_geometry(enhance, name="seathru").displacement_px < 0.5
