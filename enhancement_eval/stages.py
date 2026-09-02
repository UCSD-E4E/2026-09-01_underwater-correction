"""The two places an enhancement can be injected, and what each one reaches.

This is the module that makes the harness answer the question that was actually
asked, so the asymmetry is worth stating plainly.

**`rectified` -- the JPEG stage.** Constraint #2's placement. The frame here is
uint8 BGR, post-auto-gamma, post-CLAHE, post-`cv2.undistort`: the exact bytes a
labeler sees and the exact bytes `FishSegmentation` decodes. Consumers reached:
human labelers, and the segmentation / head-tail pair.

**`linear` -- the laser detector's stage.** uint16 linear BGR in *sensor*
coordinates, no gamma, no CLAHE. Consumers reached: `LaserDetector` and
`classify_laser_color`.

The two do not overlap, and that is the finding that reshapes Deliverable 1.
`predict_laser_image` does not read the JPEG -- it downloads the `.ORF` and
builds a `LinearRawImage`. So an enhancer placed at the JPEG stage has
*structurally zero* effect on the laser detector: not a small effect, not one
needing a large sample to detect, but exactly zero. The `rectified` arm's laser
numbers are therefore a control, and the harness asserts they come out
unchanged rather than reporting them as a result.

Three further facts about the detector's input, all from
`fishsense_core._laser_detector`, bound how much the `linear` arm can even see:

* Its first three channels are `chromaticity_norm(rgb) = rgb / sum(rgb)`, which
  is **per-pixel scale-invariant**. Any enhancement that only moves luminance
  is invisible to them. What moves them is a change in channel *ratios* --
  which is to say, white balance.
* Two of its six channels are Bayer-excess, computed from the undemosaiced
  mosaic. An `ndarray -> ndarray` RGB enhancer cannot produce them, so the
  `linear` arm passes the *original* Bayer-excess alongside the enhanced BGR.
  That is an honest inconsistency and it makes the arm **under-measure**
  disruption: a third of the detector's input stays clean no matter what the
  enhancer does. Read a null result there with that in mind.
* `LinearRawImage` uses `use_camera_wb=True`, and the checkpoint was trained on
  that decode. A white-balance change at this stage is a systematic covariate
  shift on the model's primary input, not a mild perturbation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from enhancement_eval.decode import DecodeConfig, Gains, WhiteBalance

__all__ = [
    "apply_clahe",
    "apply_stretch",
    "apply_red_boost",
    "decode_rectified_stage",
    "decode_linear_stage",
    "rectify",
    "resolve_white_balance",
]


def _postprocess_kwargs(config: DecodeConfig, camera_gains: Gains | None) -> dict:
    """rawpy keywords for one decode arm.

    Everything except white balance is pinned to production. `user_flip=0` in
    particular is not optional: EXIF rotation would change the frame's shape
    and invalidate every label coordinate.
    """
    import rawpy

    kwargs = dict(
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        user_flip=0,
    )
    if config.white_balance is WhiteBalance.CAMERA:
        kwargs["use_camera_wb"] = True
    elif config.white_balance is WhiteBalance.RAWPY_AUTO:
        kwargs["use_auto_wb"] = True
    else:
        # Absolute raw multipliers. Derived as the camera's own multipliers
        # times a correction measured off a camera-WB decode, so the only thing
        # that changes between arms is the white point -- the colour matrix,
        # the demosaic and the black levels all stay identical.
        assert camera_gains is not None
        kwargs["user_wb"] = list(camera_gains)
    return kwargs


def _decode_rgb16(source: bytes, config: DecodeConfig, camera_gains, *, half=False):
    """Raw bytes -> linear uint16 RGB, in sensor coordinates."""
    import rawpy

    from fishsense_core.image.image import open_image_source

    kwargs = _postprocess_kwargs(config, camera_gains)
    if half:
        kwargs["half_size"] = True
    with open_image_source(source) as handle:
        with rawpy.imread(handle) as raw:
            return raw.postprocess(**kwargs)


def resolve_white_balance(
    source: bytes,
    config: DecodeConfig,
    *,
    slate_quad: np.ndarray | None = None,
) -> Gains | None:
    """Absolute rawpy `user_wb` multipliers for this arm, or None for CAMERA.

    Estimated as *camera multipliers x correction*, where the correction comes
    from a camera-WB decode of this same frame. Composing rather than replacing
    matters: it holds the colour matrix, demosaic and black levels fixed, so
    the difference between two arms is the white point and nothing else.

    The estimation pass runs at half resolution by default. The gains are a
    global statistic, so the quarter-cost pass moves them negligibly, and the
    full decode then runs exactly once.
    """
    from enhancement_eval.decode import (
        gray_world_gains,
        slate_patch_gains,
        white_patch_gains,
    )

    if config.white_balance in (WhiteBalance.CAMERA, WhiteBalance.RAWPY_AUTO):
        return None

    import rawpy

    from fishsense_core.image.image import open_image_source

    with open_image_source(source) as handle:
        with rawpy.imread(handle) as raw:
            camera_wb = list(float(v) for v in raw.camera_whitebalance)

    probe_rgb = _decode_rgb16(
        source,
        DecodeConfig(),  # camera WB, so the correction is measured off production
        None,
        half=config.wb_estimate_half_size,
    )
    probe_bgr = probe_rgb[:, :, ::-1]

    if config.white_balance is WhiteBalance.GRAY_WORLD:
        correction = gray_world_gains(probe_bgr)
    elif config.white_balance is WhiteBalance.WHITE_PATCH:
        correction = white_patch_gains(probe_bgr, percentile=config.wb_percentile)
    elif config.white_balance is WhiteBalance.SLATE:
        if slate_quad is None:
            raise ValueError(
                "WhiteBalance.SLATE needs the image's slate quad "
                "(DiveSlateLabel.slate_rectangle); none was supplied"
            )
        quad = np.asarray(slate_quad, dtype=float)
        if config.wb_estimate_half_size:
            quad = quad / 2.0
        correction = slate_patch_gains(
            probe_bgr, quad, percentile=config.wb_percentile
        )
    else:  # pragma: no cover - the enum is closed
        raise ValueError(f"unhandled white balance {config.white_balance}")

    # camera_whitebalance is [R, G, B, G2]; the correction is (R, G, B).
    red, green, blue = correction
    g1 = camera_wb[1] * green
    g2 = (camera_wb[3] if len(camera_wb) > 3 and camera_wb[3] else camera_wb[1]) * green
    return (camera_wb[0] * red, g1, camera_wb[2] * blue, g2)


def decode_linear_stage(
    source: bytes,
    config: DecodeConfig,
    *,
    slate_quad: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Raw bytes -> (uint16 BGR linear, Bayer-excess), in sensor coordinates.

    Mirrors `fishsense_core.image.linear_raw_image.LinearRawImage`. Only the
    white-balance knob reaches here: gamma and CLAHE live in the JPEG chain
    only, which is why a CLAHE sweep cannot move the laser detector's numbers.

    The Bayer-excess is read from the mosaic and is therefore *independent of
    every decode knob* -- it is computed before white balance is applied at
    all. Returned unchanged so the caller can hand the detector a complete
    6-channel input.
    """
    import cv2

    from fishsense_core.image.linear_raw_image import LinearRawImage

    camera_gains = resolve_white_balance(source, config, slate_quad=slate_quad)
    rgb = _decode_rgb16(source, config, camera_gains)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Bayer-excess is a property of the mosaic, so it is reused from
    # fishsense-core's own implementation rather than reimplemented -- there is
    # no decode knob that could change it, and a second copy would drift.
    bayer = LinearRawImage(source).bayer_excess
    return bgr, bayer


def decode_rectified_stage(source: bytes, config: DecodeConfig, *, slate_quad=None):
    """Raw bytes -> uint8 BGR, the JPEG chain before `cv2.undistort`.

    Mirrors `fishsense_core.image.raw_image.RawImage` step for step, with the
    three knobs injected. Kept in the same order and the same dtypes as the
    original: `img_as_ubyte` round trips are lossy, so reordering them or
    skipping one changes the output even when the maths looks equivalent.
    """
    import cv2
    from skimage.exposure import adjust_gamma
    from skimage.util import img_as_float, img_as_ubyte

    camera_gains = resolve_white_balance(source, config, slate_quad=slate_quad)
    img = img_as_float(_decode_rgb16(source, config, camera_gains))

    # V of HSV is max(channels), so feeding this RGB array to a BGR->HSV
    # conversion is harmless -- and it is what production does. Reproduced
    # rather than corrected: the point of this arm is to be production.
    hsv = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV)
    _, _, val = cv2.split(hsv)

    mean = float(np.mean(val))
    # Production's expression, including the `* 255` that makes the "target
    # mean brightness of 20" a good deal darker than it sounds. Preserved
    # exactly; `auto_gamma_target` moves the 20, not the formula.
    gamma = 1 / (math.log(config.auto_gamma_target * 255) / math.log(mean))
    img = adjust_gamma(img, gamma=gamma)

    img = apply_stretch(img, config)
    img = apply_clahe(img, config)
    img = apply_red_boost(img, config)

    return img_as_ubyte(img[:, :, ::-1])


def rectify(image: np.ndarray, camera_matrix, distortion) -> np.ndarray:
    """`cv2.undistort` with the camera's intrinsics -- the last step before the
    JPEG is encoded, and the step that puts pixels in label space.

    Size-preserving, which is why laser pixels, head/tail pixels and the JPEG
    are all one coordinate system.
    """
    import cv2

    return cv2.undistort(
        image,
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    )


def apply_clahe(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Local contrast enhancement, in the mode `config` selects.

    `img` is float RGB in [0, 1] -- the same array production hands
    `equalize_adapthist`.

    The two modes differ in what they are allowed to change. `per_channel` is
    production: skimage stretches each channel's histogram to full range
    independently, which underwater hands the largest gain to red (least
    signal, most noise) and turns flat open water into chroma speckle.
    `luminance` equalizes CIELAB L* only, so local contrast still lifts the
    fish outline while hue is left alone.
    """
    from skimage.exposure import equalize_adapthist

    if not config.clahe_enabled:
        return img

    kwargs = {}
    if config.clahe_clip_limit is not None:
        kwargs["clip_limit"] = config.clahe_clip_limit
    if config.clahe_kernel_size is not None:
        kwargs["kernel_size"] = config.clahe_kernel_size

    if config.clahe_mode == "per_channel":
        return equalize_adapthist(img, **kwargs)

    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(img)
    # L* is 0-100; equalize_adapthist wants a unit-range float, and rescaling
    # back by the same constant keeps the round trip exact for an untouched
    # image.
    lab[:, :, 0] = equalize_adapthist(lab[:, :, 0] / 100.0, **kwargs) * 100.0
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def apply_stretch(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Global percentile contrast stretch. `img` is float RGB in [0, 1].

    One affine map per channel (or one over L*), derived from percentiles of
    the frame itself. That globality is the entire point: CLAHE gives every
    tile its own histogram, so a near-uniform tile of open water gets its noise
    stretched to full scale, while a global map leaves that region flat
    relative to the rest of the scene and spends the range on real content.

    The black point is not incidental. Backscatter is an additive veiling
    light, so removing a floor is the physically-motivated half of this
    operation -- and it is the half a gain-only white balance (gray-world,
    white-patch) structurally cannot do.
    """
    if config.stretch_mode == "off":
        return img

    def _percentiles(value, channel):
        return value[channel] if isinstance(value, tuple) else value

    def _map(plane: np.ndarray, channel: int = 0) -> np.ndarray:
        lo = float(np.percentile(plane, _percentiles(config.stretch_low, channel)))
        hi = float(np.percentile(plane, _percentiles(config.stretch_high, channel)))
        if hi - lo < 1e-6:
            # A genuinely flat plane has no range to expand; stretching it
            # would turn its noise into the entire signal.
            return plane
        return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)

    if config.stretch_mode == "per_channel":
        out = np.empty_like(img)
        for channel in range(img.shape[2]):
            out[:, :, channel] = _map(img[:, :, channel], channel)
        return out

    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(img)
    lab[:, :, 0] = _map(lab[:, :, 0] / 100.0) * 100.0
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def apply_red_boost(img: np.ndarray, config: DecodeConfig) -> np.ndarray:
    """Lift red where the *spatially coherent* red excess is high.

    The laser dot is a coherent blob roughly ten pixels across; the red speckle
    that a per-channel stretch amplifies is pixel-scale. Blurring the red-excess
    map before thresholding is what tells them apart -- the blob survives the
    blur, the speckle averages away -- so the gain lands on the dot and not on
    the noise.

    Unlike everything else in this module this **synthesises** emphasis rather
    than recovering signal that attenuation removed. That is a real distinction:
    it is a labeling aid, defensible as a target indicator on species and
    head/tail frames (where the laser marks which fish is being measured), and
    circular if it were ever pointed at the laser-labeling task itself.

    Reads a neighbourhood but moves no pixel; `probe_geometry` covers it.
    """
    if not config.red_boost:
        return img

    from scipy.ndimage import gaussian_filter

    red = img[:, :, 0]
    excess = red - np.maximum(img[:, :, 1], img[:, :, 2])
    # Difference of Gaussians on the red-excess map. The dot's *absolute*
    # excess is negative in these frames (the scene is cyan), so what is
    # thresholded has to be its excess relative to the water around it.
    coherent = gaussian_filter(excess, sigma=config.red_boost_sigma) - gaussian_filter(
        excess, sigma=config.red_boost_background_sigma
    )

    if config.red_boost_sigmas:
        # Robust sigma of the blob response, so the cut scales with whatever
        # noise this particular frame's red channel carries. MAD rather than
        # std because the dot itself (and any other real blob) is exactly the
        # kind of outlier a std would absorb, inflating the threshold until
        # nothing passes.
        centre = float(np.median(coherent))
        mad = float(np.median(np.abs(coherent - centre)))
        sigma = 1.4826 * mad
        threshold = centre + config.red_boost_sigmas * sigma
        span = max(config.red_boost_sigmas * sigma, 1e-6)
        weight = np.clip((coherent - threshold) / span, 0.0, 1.0)
        out = img.copy()
        out[:, :, 0] = np.clip(red + config.red_boost * weight, 0.0, 1.0)
        return out

    threshold = config.red_boost_threshold
    # Soft ramp rather than a hard cut, so the dot does not acquire a stamped
    # edge that a labeler could mistake for a real boundary. The span is the
    # excess range over which the ramp reaches full, and it is set from the
    # quantity's real scale -- see `red_boost_span`.
    weight = np.clip(
        (coherent - threshold) / max(1e-6, config.red_boost_span), 0.0, 1.0
    )

    out = img.copy()
    out[:, :, 0] = np.clip(red + config.red_boost * weight, 0.0, 1.0)
    return out
