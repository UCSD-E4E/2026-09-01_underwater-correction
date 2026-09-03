"""Fitting attenuation coefficients against the dive slate.

The annulus-around-the-dot method recovered the right *sign* on every field
dive and the right null in the pool, but its magnitudes varied 3.7x with
r2 = 0.18-0.34. The reason was structural: every frame is a different scene, so
reflectance varies frame to frame and does not average out at n = 65-135.

The dive slate removes that term. It is a known, constant-reflectance target,
so across frames of one dive the only thing changing with range is the water:

    log(I_c) = log(J_c * rho_c) - beta_c * z + (backscatter)

With rho_c fixed, the slope of log(I_c) on range estimates **beta_c itself**,
not merely a difference between channels. That is a strictly stronger result
than the annulus method could give.

The pool dives are the negative control. Clear water has almost no attenuation
over 2 m, so a correct method must return approximately zero there -- and 7 of
the 8 dives that currently carry slate frames with a metric range are Pool
Calibration, which makes the control free.
"""

import math

import pytest

from enhancement_eval.attenuation import (
    AttenuationFit,
    fit_loglinear,
    fit_attenuation,
    sample_slate_patch,
)


# --------------------------------------------------------------------------
# the regression primitive
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_slope():
    z = [1.0 + 0.5 * i for i in range(10)]      # >= MIN_SAMPLES
    y = [2.0 - 0.4 * v for v in z]
    fit = fit_loglinear(z, y)
    assert fit.slope == pytest.approx(-0.4)
    assert fit.r2 == pytest.approx(1.0)


def test_fit_reports_a_standard_error_and_interval():
    import random

    rng = random.Random(0)
    z = [1 + 0.05 * i for i in range(60)]
    y = [1.0 - 0.3 * v + rng.gauss(0, 0.05) for v in z]
    fit = fit_loglinear(z, y)
    assert fit.ci_low < -0.3 < fit.ci_high
    assert fit.stderr > 0


def test_a_fit_with_no_range_spread_is_refused():
    """Attenuation is a slope against distance. With every frame at the same
    range there is no lever, and any slope reported would be pure noise."""
    fit = fit_loglinear([2.0] * 20, [0.5] * 20)
    assert fit is None or math.isnan(fit.slope)


def test_too_few_points_is_refused():
    """Below MIN_SAMPLES a slope is not worth reporting, however clean it looks."""
    assert fit_loglinear([1.0, 2.0], [0.1, 0.2]) is None
    assert fit_loglinear([1.0 + i for i in range(6)], [0.4 * i for i in range(6)]) is None


def test_fit_flags_itself_unreliable_when_the_interval_spans_zero():
    """The pool case. A method that cannot say 'no attenuation here' would
    happily invent a correction for clear water."""
    import random

    rng = random.Random(1)
    z = [1 + 0.05 * i for i in range(60)]
    y = [0.4 + rng.gauss(0, 0.3) for _ in z]          # no dependence on range
    fit = fit_loglinear(z, y)
    assert fit.spans_zero


# --------------------------------------------------------------------------
# sampling the slate
# --------------------------------------------------------------------------


def test_sample_takes_the_bright_tail_inside_the_quad():
    """The slate is white paper carrying black markings. Its mean is a
    paper/ink mixture that depends on how much artwork is in view; the bright
    tail is the paper, and paper is the constant-reflectance surface the whole
    method rests on."""
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint16)
    img[:, :] = 400                                   # background outside
    img[30:70, 30:70] = 8000                          # slate paper
    img[40:50, 40:50] = 500                           # ink on the slate
    quad = np.array([[30, 30], [70, 30], [70, 70], [30, 70]])
    r, g, b = sample_slate_patch(img, quad, percentile=80.0)
    assert r > 7000 and g > 7000 and b > 7000


def test_sample_ignores_pixels_outside_the_quad():
    import numpy as np

    img = np.full((100, 100, 3), 60000, dtype=np.uint16)
    img[30:70, 30:70] = 5000
    quad = np.array([[30, 30], [70, 30], [70, 70], [30, 70]])
    r, g, b = sample_slate_patch(img, quad, percentile=90.0)
    assert max(r, g, b) < 10000


def test_sample_refuses_a_saturated_slate():
    """A blown-out slate carries no attenuation information: its pixels are at
    the container ceiling regardless of how much water they were seen through."""
    import numpy as np

    img = np.full((100, 100, 3), 65535, dtype=np.uint16)
    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    assert sample_slate_patch(img, quad) is None


def test_sample_refuses_a_quad_that_is_too_small():
    import numpy as np

    img = np.full((100, 100, 3), 5000, dtype=np.uint16)
    quad = np.array([[10, 10], [12, 10], [12, 12], [10, 12]])
    assert sample_slate_patch(img, quad) is None


# --------------------------------------------------------------------------
# the per-dive fit
# --------------------------------------------------------------------------


def _samples(beta, n=40, j=9000.0):
    """Synthetic slate observations obeying I_c = J_c * exp(-beta_c z)."""
    return [
        {"range_m": 1.0 + 3.0 * i / (n - 1),
         "R": j * math.exp(-beta[0] * (1.0 + 3.0 * i / (n - 1))),
         "G": j * math.exp(-beta[1] * (1.0 + 3.0 * i / (n - 1))),
         "B": j * math.exp(-beta[2] * (1.0 + 3.0 * i / (n - 1)))}
        for i in range(n)
    ]


def test_fit_recovers_absolute_coefficients_from_a_constant_target():
    """The whole point of using the slate: with reflectance held fixed, the
    slope gives beta_c itself rather than a difference between channels."""
    fit = fit_attenuation(_samples((0.60, 0.25, 0.18)))
    assert fit.beta_r == pytest.approx(0.60, abs=0.02)
    assert fit.beta_g == pytest.approx(0.25, abs=0.02)
    assert fit.beta_b == pytest.approx(0.18, abs=0.02)


def test_fit_recovers_the_expected_underwater_ordering():
    fit = fit_attenuation(_samples((0.60, 0.25, 0.18)))
    assert fit.beta_r > fit.beta_g > fit.beta_b
    assert fit.ordering_ok


def test_fit_reports_near_zero_on_clear_water():
    """The pool negative control, which is the only validation available until
    field slate frames exist."""
    fit = fit_attenuation(_samples((0.02, 0.015, 0.01)))
    assert abs(fit.beta_r) < 0.08
    assert fit.beta_r_fit.spans_zero or abs(fit.beta_r) < 0.05


def test_fit_carries_the_differentials_too():
    """Comparable with the annulus result, which could only ever measure these."""
    fit = fit_attenuation(_samples((0.60, 0.25, 0.18)))
    assert fit.d_rg == pytest.approx(0.35, abs=0.03)


def test_fit_of_too_few_samples_is_none():
    assert fit_attenuation(_samples((0.5, 0.2, 0.1), n=3)) is None


def test_fit_records_its_own_range_lever():
    """A slope over 1.0-1.2 m and one over 1.0-5.5 m are not equally
    trustworthy, and the report must be able to say which it has."""
    fit = fit_attenuation(_samples((0.60, 0.25, 0.18)))
    assert fit.range_span == pytest.approx(3.0, abs=0.1)


# --------------------------------------------------------------------------
# slate_rectangle geometry
# --------------------------------------------------------------------------


def test_two_point_rectangle_is_expanded_to_four_corners():
    """`DiveSlateLabel.slate_rectangle` is stored as two opposite corners, not
    a polygon:

        [[1708.4, 1079.6], [2298.0, 1534.5]]

    Passing that to `cv2.fillPoly` fills a degenerate two-point polygon -- a
    one-pixel-wide diagonal line across the slate. On a real frame that line is
    ~700 px long, so it clears a 400-pixel minimum and looks like a valid
    sample while containing almost none of the paper.
    """
    from enhancement_eval.attenuation import as_polygon

    poly = as_polygon([[10.0, 20.0], [50.0, 80.0]])
    assert len(poly) == 4
    xs = sorted(p[0] for p in poly)
    ys = sorted(p[1] for p in poly)
    assert xs[0] == 10.0 and xs[-1] == 50.0
    assert ys[0] == 20.0 and ys[-1] == 80.0


def test_a_four_point_polygon_is_passed_through():
    from enhancement_eval.attenuation import as_polygon

    quad = [[0.0, 0.0], [10.0, 1.0], [11.0, 9.0], [1.0, 8.0]]
    assert as_polygon(quad) == [tuple(p) for p in quad]


def test_corners_in_either_order_give_the_same_box():
    from enhancement_eval.attenuation import as_polygon

    assert set(as_polygon([[50.0, 80.0], [10.0, 20.0]])) == set(
        as_polygon([[10.0, 20.0], [50.0, 80.0]])
    )


def test_a_degenerate_rectangle_is_rejected():
    from enhancement_eval.attenuation import as_polygon

    assert as_polygon([[10.0, 20.0], [10.0, 20.0]]) is None
    assert as_polygon([[10.0, 20.0]]) is None
    assert as_polygon(None) is None


def test_sampling_a_two_point_rectangle_covers_the_slate_interior():
    """The regression: the sample must come from the paper, not a diagonal
    streak across it."""
    import numpy as np

    from enhancement_eval.attenuation import sample_slate_patch

    img = np.full((200, 200, 3), 300, dtype=np.uint16)
    img[50:150, 50:150] = 9000                       # the slate
    got = sample_slate_patch(img, [[50, 50], [150, 150]], percentile=50.0)
    assert got is not None
    assert min(got) > 8000, "sampled outside the slate"
