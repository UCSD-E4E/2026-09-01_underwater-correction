"""Decode variants -- Deliverable 2's one-line experiments, made measurable.

The baseline arm must reproduce `fishsense_core` exactly. That is not a nicety:
every number this harness reports is a *difference* against the baseline, so a
baseline that merely resembles production turns every reported delta into the
sum of the effect being studied and an uncontrolled reimplementation error. The
parity tests against the real `.ORF` fixture are the load-bearing ones here;
the unit tests below only pin the pieces.
"""

import numpy as np
import pytest

from enhancement_eval.decode import (
    DecodeConfig,
    WhiteBalance,
    gray_world_gains,
    white_patch_gains,
    normalize_gains,
)


# --------------------------------------------------------------------------
# white-balance gain estimators
# --------------------------------------------------------------------------


def test_gray_world_equalizes_channel_means():
    """The gray-world assumption: the scene averages to neutral. Underwater it
    is wrong in an interesting way -- the water column really is blue-green --
    which is exactly why it needs measuring rather than assuming."""
    img = np.zeros((10, 10, 3), dtype=np.uint16)
    img[:, :, 0] = 400   # B
    img[:, :, 1] = 200   # G
    img[:, :, 2] = 100   # R
    r, g, b = gray_world_gains(img)
    # Gains are relative to green, which is the reference channel because it
    # has twice the photosites and so the least noise.
    assert g == pytest.approx(1.0)
    assert r == pytest.approx(2.0)
    assert b == pytest.approx(0.5)


def test_white_patch_uses_the_bright_tail_not_the_single_brightest_pixel():
    """One hot pixel or one specular glint must not set the whole frame's
    white point. A percentile is the standard defence."""
    img = np.full((100, 100, 3), 100, dtype=np.uint16)
    img[0, 0] = (65535, 65535, 65535)  # a single blown pixel
    r, g, b = white_patch_gains(img, percentile=99.0)
    assert r == pytest.approx(1.0, abs=0.05)
    assert b == pytest.approx(1.0, abs=0.05)


def test_white_patch_recovers_a_known_cast():
    img = np.zeros((50, 50, 3), dtype=np.uint16)
    img[:, :, 0] = 800   # B
    img[:, :, 1] = 400   # G
    img[:, :, 2] = 200   # R
    r, g, b = white_patch_gains(img, percentile=90.0)
    assert r == pytest.approx(2.0, rel=0.05)
    assert b == pytest.approx(0.5, rel=0.05)


def test_gain_estimators_ignore_a_fully_black_channel_rather_than_dividing_by_zero():
    """A channel that is entirely zero is a broken decode, not a licence to
    return inf and blow the frame out."""
    img = np.zeros((10, 10, 3), dtype=np.uint16)
    img[:, :, 1] = 500
    gains = gray_world_gains(img)
    assert all(np.isfinite(v) for v in gains)


def test_normalize_gains_pins_green_to_one():
    """rawpy multiplies raw channel values, so an overall scale change is an
    exposure change, not a white-balance change. Pinning green keeps the two
    concerns separate -- otherwise a 'white balance' experiment silently
    changes brightness too and the result cannot be attributed."""
    assert normalize_gains((4.0, 2.0, 1.0)) == pytest.approx((2.0, 1.0, 0.5))


def test_normalize_gains_of_a_zero_green_falls_back_to_identity():
    assert normalize_gains((1.0, 0.0, 1.0)) == (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# DecodeConfig
# --------------------------------------------------------------------------


def test_default_config_is_the_production_chain():
    """The baseline arm. If this drifts from `fishsense_core`, every delta in
    every report is measuring the drift as well as the experiment."""
    cfg = DecodeConfig()
    assert cfg.white_balance is WhiteBalance.CAMERA
    assert cfg.auto_gamma_target == 20
    assert cfg.clahe_enabled is True
    assert cfg.clahe_clip_limit is None   # skimage default, 0.01
    assert cfg.clahe_kernel_size is None  # skimage default, 1/8 of each axis


def test_config_is_hashable_so_it_can_key_a_result_cache():
    """Decoding a 15 MB .ORF is the expensive step; caching by (checksum,
    config) is what makes the report re-runnable without re-decoding."""
    assert hash(DecodeConfig()) == hash(DecodeConfig())
    assert hash(DecodeConfig()) != hash(DecodeConfig(auto_gamma_target=40))


def test_config_has_a_stable_short_label_for_reports_and_filenames():
    assert DecodeConfig().label == "baseline"
    assert "grayworld" in DecodeConfig(white_balance=WhiteBalance.GRAY_WORLD).label
    assert "clip0.005" in DecodeConfig(clahe_clip_limit=0.005).label


def test_config_rejects_a_nonsensical_gamma_target():
    """`mean` is a 0-255 V-channel mean and the target is compared in log
    space, so a target of 0 makes `math.log` raise deep inside the decode."""
    with pytest.raises(ValueError, match="auto_gamma_target"):
        DecodeConfig(auto_gamma_target=0)


def test_config_rejects_a_clip_limit_outside_the_unit_interval():
    with pytest.raises(ValueError, match="clip_limit"):
        DecodeConfig(clahe_clip_limit=1.5)


# --------------------------------------------------------------------------
# CLAHE mode
# --------------------------------------------------------------------------


def test_default_clahe_mode_is_the_production_per_channel_behaviour():
    """`equalize_adapthist` has no RGB branch: it treats an HxWx3 array as a
    3-D volume and defaults `kernel_size` to `(H//8, W//8, max(3//8,1))` -- a
    tile depth of 1, which equalizes each colour channel independently. That is
    what production does today, so it stays the default; the parity test
    depends on it."""
    assert DecodeConfig().clahe_mode == "per_channel"


def test_luminance_mode_is_selectable():
    assert DecodeConfig(clahe_mode="luminance").clahe_mode == "luminance"


def test_an_unknown_clahe_mode_is_rejected():
    with pytest.raises(ValueError, match="clahe_mode"):
        DecodeConfig(clahe_mode="sideways")


def test_clahe_mode_appears_in_the_label():
    assert "luma" in DecodeConfig(clahe_mode="luminance").label


def test_clahe_mode_participates_in_the_cache_key():
    assert hash(DecodeConfig()) != hash(DecodeConfig(clahe_mode="luminance"))


def test_a_lower_clip_limit_reduces_flat_region_noise_amplification():
    """The mechanism behind "CLAHE breaks contrast in the ocean water".

    Open water is near-uniform, so its local histogram is narrow and CLAHE
    stretches that narrow range toward full scale -- what gets amplified is
    sensor noise. On this flat blue-green patch skimage's default clip of 0.01
    raises luminance SD ~62x; 0.003 cuts that to ~32x. (The absolute factors
    move with frame size, because the default kernel is `shape // 8` -- the
    ratio between the two clips is the stable part, and it is what is asserted.)

    The clip limit is the knob that governs this. It is *not* a colour-space
    problem, which is worth pinning down because "apply CLAHE to luminance
    only" is the reflexive fix and does not help here (see below).
    """
    import numpy as np

    from enhancement_eval.stages import apply_clahe

    rng = np.random.default_rng(0)
    flat = np.zeros((192, 192, 3))
    flat[:, :, 0], flat[:, :, 1], flat[:, :, 2] = 0.06, 0.34, 0.42
    flat = np.clip(flat + rng.normal(0, 0.004, flat.shape), 0, 1)
    base = flat.mean(axis=2).std()

    default = apply_clahe(flat, DecodeConfig()).mean(axis=2).std() / base
    clipped = apply_clahe(
        flat, DecodeConfig(clahe_clip_limit=0.003)
    ).mean(axis=2).std() / base

    assert default > 20
    assert clipped < 0.6 * default


def test_a_lower_clip_limit_also_improves_object_separation():
    """Counterintuitive, and the reason to prefer clipping over disabling.

    A permissive clip stretches the water's amplified noise toward full range,
    where it competes with the object. Clipping harder keeps the background
    compressed, so a real object-vs-water difference stands out *more*, not
    less -- while turning CLAHE off entirely gives up the enhancement
    altogether.
    """
    import numpy as np

    from enhancement_eval.stages import apply_clahe

    rng = np.random.default_rng(0)
    img = np.zeros((256, 256, 3))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = 0.06, 0.34, 0.42
    img[90:170, 90:170] = (0.11, 0.40, 0.46)
    img = np.clip(img + rng.normal(0, 0.004, img.shape), 0, 1)
    water, obj = np.s_[0:60, 0:60], np.s_[100:160, 100:160]

    def separation(a):
        return abs(a[obj].mean() - a[water].mean())

    assert separation(apply_clahe(img, DecodeConfig(clahe_clip_limit=0.003))) > separation(
        apply_clahe(img, DecodeConfig())
    )
    assert separation(apply_clahe(img, DecodeConfig(clahe_enabled=False))) < separation(
        apply_clahe(img, DecodeConfig())
    )


def test_luminance_mode_is_not_automatically_hue_preserving():
    """Recorded because it is the obvious fix and it does not work.

    Equalizing CIELAB L* while holding a*/b* fixed does **not** hold
    chromaticity fixed -- LAB to RGB is non-linear, so changing L* alone moves
    the RGB ratios, and out-of-gamut results clip on the way back. Measured on
    a flat water patch, luminance-only mode raises chromaticity spread ~10x
    while per-channel leaves it unchanged.
    """
    import numpy as np

    from enhancement_eval.stages import apply_clahe

    rng = np.random.default_rng(0)
    flat = np.zeros((192, 192, 3))
    flat[:, :, 0], flat[:, :, 1], flat[:, :, 2] = 0.06, 0.34, 0.42
    flat = np.clip(flat + rng.normal(0, 0.004, flat.shape), 0, 1)

    def chromaticity_spread(a):
        total = a.sum(axis=2, keepdims=True)
        return float((a / np.maximum(total, 1e-6)).std(axis=(0, 1)).mean())

    ref = chromaticity_spread(flat)
    per_channel = chromaticity_spread(apply_clahe(flat, DecodeConfig())) / ref
    luminance = chromaticity_spread(
        apply_clahe(flat, DecodeConfig(clahe_mode="luminance"))
    ) / ref
    assert luminance > per_channel


def test_apply_clahe_is_a_no_op_when_disabled():
    import numpy as np

    from enhancement_eval.stages import apply_clahe

    img = np.linspace(0, 1, 3 * 16 * 16).reshape(16, 16, 3)
    np.testing.assert_array_equal(apply_clahe(img, DecodeConfig(clahe_enabled=False)), img)


# --------------------------------------------------------------------------
# contrast stretching
# --------------------------------------------------------------------------


def test_stretch_is_off_by_default_so_the_baseline_stays_production():
    assert DecodeConfig().stretch_mode == "off"


def test_stretch_modes_are_validated():
    with pytest.raises(ValueError, match="stretch_mode"):
        DecodeConfig(stretch_mode="sideways")


def test_stretch_percentiles_must_be_ordered_and_in_range():
    with pytest.raises(ValueError, match="stretch"):
        DecodeConfig(stretch_mode="per_channel", stretch_low=99.0, stretch_high=1.0)
    with pytest.raises(ValueError, match="stretch"):
        DecodeConfig(stretch_mode="per_channel", stretch_low=-1.0)


def test_stretch_appears_in_the_label():
    assert "stretch" in DecodeConfig(stretch_mode="per_channel").label


def test_per_channel_stretch_expands_a_compressed_range_to_full_scale():
    """The underwater problem in one line: the scene occupies a narrow slice of
    the container's range, so everything reads flat. A global stretch maps that
    slice back onto [0, 1]."""
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    rng = np.random.default_rng(0)
    img = np.zeros((128, 128, 3))
    img[:, :, 0] = rng.uniform(0.40, 0.46, (128, 128))   # R: compressed and dark
    img[:, :, 1] = rng.uniform(0.55, 0.70, (128, 128))
    img[:, :, 2] = rng.uniform(0.60, 0.78, (128, 128))
    out = apply_stretch(img, DecodeConfig(stretch_mode="per_channel"))
    for c in range(3):
        assert out[:, :, c].min() < 0.05
        assert out[:, :, c].max() > 0.95


def test_per_channel_stretch_removes_a_colour_cast():
    """Each channel independently mapped to full range is, in effect, a white
    balance with a black point -- which matters underwater because backscatter
    is roughly additive, so a gain-only correction cannot remove it."""
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    rng = np.random.default_rng(1)
    img = np.zeros((128, 128, 3))
    img[:, :, 0] = rng.uniform(0.05, 0.12, (128, 128))   # R starved
    img[:, :, 1] = rng.uniform(0.30, 0.60, (128, 128))
    img[:, :, 2] = rng.uniform(0.40, 0.75, (128, 128))
    before = abs(img[:, :, 0].mean() - img[:, :, 2].mean())
    after_img = apply_stretch(img, DecodeConfig(stretch_mode="per_channel"))
    after = abs(after_img[:, :, 0].mean() - after_img[:, :, 2].mean())
    assert after < before


def test_luminance_stretch_expands_range_without_removing_the_cast():
    """The conservative option: more contrast, same colours. Worth having
    separately, because per-channel stretching is also a white balance and
    those are two different decisions to defend."""
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    rng = np.random.default_rng(2)
    img = np.zeros((128, 128, 3))
    img[:, :, 0] = rng.uniform(0.05, 0.12, (128, 128))
    img[:, :, 1] = rng.uniform(0.30, 0.45, (128, 128))
    img[:, :, 2] = rng.uniform(0.40, 0.55, (128, 128))
    out = apply_stretch(img, DecodeConfig(stretch_mode="luminance"))
    assert out.mean(axis=2).std() > img.mean(axis=2).std()
    # the cast survives: red still the weakest channel by a similar margin
    assert out[:, :, 0].mean() < out[:, :, 2].mean()


def test_stretch_amplifies_flat_region_noise_less_than_clahe_but_not_zero():
    """The honest version of "a global stretch is gentler".

    A global stretch applies ONE affine map, so a flat region is not handed its
    own local histogram the way CLAHE does. But the map's gain is set by the
    frame's own range, and an underwater frame is genuinely low-contrast -- so a
    large gain is exactly what makes it look better, and that gain lands on the
    noise too. Measured here: CLAHE ~22x, stretch ~11x. Half, not none.

    Worth pinning, because "stretch instead of CLAHE" reads like a free win and
    is not one: it buys uniform amplification instead of amplification
    concentrated in the flattest, emptiest parts of the frame.
    """
    import numpy as np

    from enhancement_eval.stages import apply_clahe, apply_stretch

    rng = np.random.default_rng(3)
    img = np.zeros((192, 192, 3))
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = 0.06, 0.34, 0.42
    img[80:150, 80:150] = (0.12, 0.42, 0.50)          # an object worth seeing
    img = np.clip(img + rng.normal(0, 0.004, img.shape), 0, 1)
    water = np.s_[0:60, 0:60]
    base = img.mean(axis=2)[water].std()

    clahe_noise = apply_clahe(img, DecodeConfig()).mean(axis=2)[water].std() / base
    stretch_noise = apply_stretch(
        img, DecodeConfig(stretch_mode="per_channel")
    ).mean(axis=2)[water].std() / base
    assert stretch_noise < clahe_noise
    assert stretch_noise > 2, "a stretch on a low-contrast frame is not noise-free"


def test_stretch_is_robust_to_a_few_extreme_pixels():
    """Percentiles, not min/max: one hot pixel or one specular glint must not
    set the whole frame's mapping.

    The scene here occupies 0.40-0.50 with two outliers at the rails. A min/max
    stretch would key on the outliers and leave the scene compressed in the
    middle; a percentile stretch keys on the scene and expands it.
    """
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    rng = np.random.default_rng(4)
    img = rng.uniform(0.40, 0.50, (128, 128, 3))
    img[0, 0] = 0.0
    img[0, 1] = 1.0
    out = apply_stretch(img, DecodeConfig(stretch_mode="per_channel"))
    scene = out[1:]                      # excluding the row holding the outliers
    assert scene.max() - scene.min() > 0.9


def test_a_flat_plane_is_left_alone_rather_than_stretched_into_noise():
    """When there is genuinely no range, expanding it would promote sensor
    noise to the entire signal."""
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    flat = np.full((64, 64, 3), 0.5)
    np.testing.assert_allclose(
        apply_stretch(flat, DecodeConfig(stretch_mode="per_channel")), flat
    )


def test_stretch_off_is_a_no_op():
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    img = np.linspace(0.2, 0.6, 3 * 16 * 16).reshape(16, 16, 3)
    np.testing.assert_array_equal(apply_stretch(img, DecodeConfig()), img)


# --------------------------------------------------------------------------
# per-channel stretch percentiles  (option 1: raise red's black point)
# --------------------------------------------------------------------------


def test_stretch_percentiles_accept_a_per_channel_triple():
    """Underwater the three channels are in completely different states -- red
    is a narrow noise-dominated band, blue is broad -- so one pair of
    percentiles for all three is the wrong shape of knob."""
    cfg = DecodeConfig(stretch_mode="per_channel", stretch_low=(90.0, 1.0, 1.0))
    assert cfg.stretch_low == (90.0, 1.0, 1.0)


def test_per_channel_percentiles_are_validated_channelwise():
    with pytest.raises(ValueError, match="stretch"):
        DecodeConfig(stretch_mode="per_channel", stretch_low=(90.0, 1.0),
                     stretch_high=99.0)
    with pytest.raises(ValueError, match="stretch"):
        DecodeConfig(stretch_mode="per_channel", stretch_low=(99.0, 1.0, 1.0),
                     stretch_high=(98.0, 99.0, 99.0))


def test_a_high_red_black_point_suppresses_the_red_noise_floor():
    """Option 1. The confetti comes from stretching the BOTTOM of the red
    channel -- which underwater is almost entirely noise -- up to full scale.
    Anchoring red's black point above that floor throws the noise away and
    keeps only genuinely red things.
    """
    import numpy as np

    from enhancement_eval.stages import apply_stretch

    rng = np.random.default_rng(5)
    img = np.zeros((160, 160, 3))
    img[:, :, 0] = rng.uniform(0.04, 0.09, (160, 160))   # red: noise floor only
    img[:, :, 1] = rng.uniform(0.30, 0.55, (160, 160))
    img[:, :, 2] = rng.uniform(0.40, 0.70, (160, 160))
    img[78:83, 78:83, 0] = 0.55                          # a laser dot, well above the floor

    flat = np.s_[0:50, 0:50]
    even = apply_stretch(img, DecodeConfig(stretch_mode="per_channel"))
    raised = apply_stretch(
        img, DecodeConfig(stretch_mode="per_channel",
                          stretch_low=(99.0, 1.0, 1.0), stretch_high=(99.99, 99.0, 99.0))
    )
    # background red noise collapses...
    assert raised[:, :, 0][flat].std() < even[:, :, 0][flat].std() / 3
    # ...while the dot still reaches the top of the range
    assert raised[78:83, 78:83, 0].mean() > 0.9


# --------------------------------------------------------------------------
# selective red-excess boost  (option 2)
# --------------------------------------------------------------------------


def test_red_boost_is_off_by_default():
    assert DecodeConfig().red_boost == 0.0


def test_red_boost_is_validated():
    with pytest.raises(ValueError, match="red_boost"):
        DecodeConfig(red_boost=-0.5)


def test_red_boost_appears_in_the_label():
    assert "redboost" in DecodeConfig(red_boost=0.3).label


def test_red_boost_lifts_a_coherent_dot_and_not_pixel_speckle():
    """Option 2. The laser is a coherent blob; the speckle that per-channel
    stretching amplifies is pixel-scale. Smoothing the red-excess map before
    thresholding separates them, so the boost lands on the dot alone.

    This SYNTHESISES emphasis rather than restoring signal -- legitimate as a
    target indicator on species / head-tail frames, circular if it were ever
    used on the laser-labeling task itself.
    """
    import numpy as np

    from enhancement_eval.stages import apply_red_boost

    rng = np.random.default_rng(6)
    img = np.zeros((160, 160, 3))
    img[:, :, 0] = 0.06
    img[:, :, 1], img[:, :, 2] = 0.36, 0.44
    img += rng.normal(0, 0.02, img.shape)                # pixel-scale speckle
    img = np.clip(img, 0, 1)
    img[78:84, 78:84, 0] = 0.55                          # the coherent dot
    before = img.copy()

    out = apply_red_boost(img, DecodeConfig(red_boost=0.4))
    dot_gain = out[78:84, 78:84, 0].mean() - before[78:84, 78:84, 0].mean()
    bg_gain = out[0:50, 0:50, 0].mean() - before[0:50, 0:50, 0].mean()
    assert dot_gain > 0.05
    assert bg_gain < dot_gain / 10


def test_red_boost_is_a_no_op_at_zero():
    import numpy as np

    from enhancement_eval.stages import apply_red_boost

    img = np.linspace(0.1, 0.7, 3 * 32 * 32).reshape(32, 32, 3)
    np.testing.assert_array_equal(apply_red_boost(img, DecodeConfig()), img)


def test_red_boost_does_not_move_pixels():
    """It reads a neighbourhood, so the geometry probe has to clear it."""
    import numpy as np

    from enhancement_eval.contract import probe_geometry
    from enhancement_eval.stages import apply_red_boost

    cfg = DecodeConfig(red_boost=0.4)
    probe_geometry(
        lambda a: (apply_red_boost(a.astype(np.float64) / 255.0, cfg) * 255).astype(a.dtype),
        name="redboost",
    )


# --------------------------------------------------------------------------
# adaptive red boost: threshold referenced to the response's own noise
# --------------------------------------------------------------------------


def test_red_boost_threshold_can_be_expressed_in_noise_sigmas():
    """An absolute threshold cannot work across these scenes.

    Measured on the showcase frames, the laser dot sits at the 99.999th
    percentile of the red channel on a reef and at the **58.8th** in a pool,
    where a white shirt and skin are redder than the dot. Any fixed level or
    percentile either misses the dot in one environment or fires on noise in
    the other. Referencing the threshold to the response's own robust sigma
    makes it scene-adaptive.
    """
    cfg = DecodeConfig(red_boost=0.4, red_boost_sigmas=6.0)
    assert cfg.red_boost_sigmas == 6.0


def test_red_boost_sigmas_are_validated():
    with pytest.raises(ValueError, match="red_boost_sigmas"):
        DecodeConfig(red_boost=0.4, red_boost_sigmas=-1.0)


def test_adaptive_threshold_rejects_pure_noise():
    """The failure the absolute threshold had: a difference-of-Gaussians on a
    noisy red channel has local maxima everywhere, and a fixed cut turns them
    into hundreds of false laser dots -- false positives shaped exactly like
    the target."""
    import numpy as np

    from enhancement_eval.stages import apply_red_boost

    rng = np.random.default_rng(11)
    noise_only = np.zeros((256, 256, 3))
    noise_only[:, :, 0] = 0.06
    noise_only[:, :, 1], noise_only[:, :, 2] = 0.36, 0.44
    noise_only = np.clip(noise_only + rng.normal(0, 0.02, noise_only.shape), 0, 1)

    out = apply_red_boost(noise_only, DecodeConfig(red_boost=0.4, red_boost_sigmas=6.0))
    lifted = (out[:, :, 0] - noise_only[:, :, 0]) > 0.05
    assert lifted.mean() < 0.001, "adaptive threshold still lit up the noise"


def test_adaptive_threshold_still_finds_a_real_dot_in_both_regimes():
    """The dot must survive whether it is the reddest thing in the frame (reef)
    or well down the distribution behind brighter red content (pool)."""
    import numpy as np

    from enhancement_eval.stages import apply_red_boost

    rng = np.random.default_rng(12)
    for bright_red_content in (False, True):
        img = np.zeros((256, 256, 3))
        img[:, :, 0] = 0.06
        img[:, :, 1], img[:, :, 2] = 0.36, 0.44
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
        if bright_red_content:
            # a pool: a large neutral-bright region, redder than the dot but
            # with no red *excess* because it is bright in every channel
            img[10:120, 10:120] = (0.80, 0.80, 0.82)
        img[150:160, 150:160] = (0.50, 0.30, 0.38)   # the dot: real red excess

        out = apply_red_boost(img, DecodeConfig(red_boost=0.4, red_boost_sigmas=6.0))
        gain = out[150:160, 150:160, 0].mean() - img[150:160, 150:160, 0].mean()
        assert gain > 0.05, f"dot not boosted (bright_red_content={bright_red_content})"
