"""The enhancement contract: what an enhancer is allowed to do to an array.

Hard constraint #1 of the project is that a shipped enhancer is strictly
pixel-wise — no warp, resize, crop, or geometric change of any kind, because
every measurement is a pixel *coordinate* and the coordinate chain assumes the
JPEG shares geometry with the rectified frame. That constraint is worth more
than a code review: it is mechanically checkable, so the harness checks it on
every single call rather than trusting the author of an enhancer.

Note what is deliberately *not* forbidden: value locality. CLAHE reads a
neighbourhood, and CLAHE is a candidate. "Pixel-wise" here means no pixel
moves, not that no pixel may consult its neighbours.
"""

import numpy as np
import pytest

from enhancement_eval.contract import IDENTITY, GeometryViolation, guard


def _u8(h=8, w=6):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_guard_passes_a_value_only_enhancer():
    guarded = guard(lambda a: (a // 2).astype(a.dtype), name="halve")
    src = _u8()
    out = guarded(src)
    assert out.shape == src.shape
    assert out.dtype == src.dtype


def test_guard_allows_a_spatial_filter_that_preserves_geometry():
    """A neighbourhood operator is fine; a resampling one is not."""
    import cv2

    guarded = guard(lambda a: cv2.GaussianBlur(a, (3, 3), 0), name="blur")
    assert guarded(_u8()).shape == (8, 6, 3)


def test_guard_rejects_a_resize():
    import cv2

    guarded = guard(lambda a: cv2.resize(a, (3, 4)), name="resize")
    with pytest.raises(GeometryViolation, match="shape"):
        guarded(_u8())


def test_guard_rejects_a_crop():
    guarded = guard(lambda a: a[1:, 1:], name="crop")
    with pytest.raises(GeometryViolation, match="shape"):
        guarded(_u8())


def test_guard_rejects_a_transpose_on_a_non_square_frame():
    """A transpose keeps the element count but moves every labelled
    coordinate. On a non-square frame the shape check is enough; the square
    case is the locality probe's job, below."""
    guarded = guard(lambda a: np.swapaxes(a, 0, 1), name="transpose")
    with pytest.raises(GeometryViolation, match="shape"):
        guarded(_u8(8, 6))


def test_guard_rejects_a_dtype_change():
    """A float return silently changes what `cv2.imencode` and the detector's
    `chromaticity_norm` scale factor do -- 255 vs 65535."""
    guarded = guard(lambda a: a.astype(np.float32), name="tofloat")
    with pytest.raises(GeometryViolation, match="dtype"):
        guarded(_u8())


def test_guard_rejects_a_channel_drop():
    guarded = guard(lambda a: a[:, :, :1], name="gray")
    with pytest.raises(GeometryViolation, match="shape"):
        guarded(_u8())


def test_guard_names_the_offending_enhancer():
    guarded = guard(lambda a: a[1:], name="my-enhancer")
    with pytest.raises(GeometryViolation, match="my-enhancer"):
        guarded(_u8())


def test_guard_rejects_an_in_place_enhancer_without_corrupting_the_input():
    """An in-place enhancer would corrupt the baseline arm, which is scored
    from the same decoded array. That bug reads as the enhancement *helping*,
    which is the most expensive way for this harness to be wrong -- so it is
    refused loudly rather than tolerated, and the input survives intact.
    """
    def in_place(a):
        a[:] = 0
        return a

    src = _u8()
    original = src.copy()
    guarded = guard(in_place, name="inplace")
    with pytest.raises(GeometryViolation, match="in place"):
        guarded(src)
    np.testing.assert_array_equal(src, original)


def test_identity_enhancer_is_a_no_op():
    from enhancement_eval.contract import IDENTITY

    src = _u8()
    np.testing.assert_array_equal(guard(IDENTITY, name="identity")(src), src)


# --------------------------------------------------------------------------
# The locality probe: what the per-call guard structurally cannot catch.
# --------------------------------------------------------------------------
#
# On a square frame a transpose preserves shape and dtype, so `guard` cannot
# see it. Neither can it see a one-pixel roll, which is precisely the failure
# mode that costs 0.75% length error. The probe catches both generically, by
# perturbing one region of the input and checking the output only changes near
# that region. It runs once per enhancer at registration, not per frame.


def test_probe_accepts_a_pointwise_enhancer():
    from enhancement_eval.contract import probe_geometry

    probe_geometry(lambda a: (a // 2).astype(a.dtype), name="halve")


def test_probe_accepts_a_bounded_neighbourhood_filter():
    """CLAHE and friends are legitimate: influence stays local."""
    import cv2
    from enhancement_eval.contract import probe_geometry

    # A 31px kernel is far wider than anything in the pipeline and still
    # measures 0.02 px of displacement: symmetric influence keeps the centroid.
    assert probe_geometry(
        lambda a: cv2.GaussianBlur(a, (31, 31), 0), name="blur"
    ).displacement_px < 0.1


def test_probe_catches_a_one_pixel_roll():
    from enhancement_eval.contract import probe_geometry

    with pytest.raises(GeometryViolation, match="moved"):
        probe_geometry(lambda a: np.roll(a, 1, axis=1), name="roll1px")


def test_probe_catches_a_square_transpose():
    from enhancement_eval.contract import probe_geometry

    with pytest.raises(GeometryViolation, match="moved"):
        probe_geometry(
            lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1)), name="transpose"
        )


def test_probe_catches_a_flip():
    from enhancement_eval.contract import probe_geometry

    with pytest.raises(GeometryViolation, match="moved"):
        probe_geometry(lambda a: a[:, ::-1].copy(), name="hflip")


def test_probe_does_not_refuse_a_resample_round_trip():
    """Downsample-and-back preserves shape and dtype and does not displace: the
    histogram-preserving probe measures it at 0.000 px, because a symmetric
    resample keeps the response centroid exactly where the input changed.

    It is a loss of *detail*, not a move, and the probe deliberately does not
    police detail -- any real cost shows up in the scored consumer's numbers.
    """
    import cv2
    from enhancement_eval.contract import probe_geometry

    def wobble(a):
        h, w = a.shape[:2]
        small = cv2.resize(a, (int(w * 0.7), int(h * 0.7)))
        return cv2.resize(small, (w, h))

    assert probe_geometry(wobble, name="resize-roundtrip").displacement_px < 0.5


def test_probe_accepts_tile_adaptive_clahe():
    """A regression on the probe's own resolution.

    CLAHE derives a mapping per tile, so on a small probe frame the
    perturbation spans several tiles and the asymmetric response shifts the
    centroid ~0.8 px -- refusing production's own decode. Measured across sizes
    it reads 0.82 / 0.76 / 0.72 / 0.000 at 64 / 128 / 256 / 384 while a
    one-pixel roll reads 1.00 throughout, which is how we know the artefact
    belongs to the probe and not to CLAHE.
    """
    import cv2
    from enhancement_eval.contract import probe_geometry

    def clahe(a):
        return cv2.merge([cv2.createCLAHE(2.0, (8, 8)).apply(c) for c in cv2.split(a)])

    assert probe_geometry(clahe, name="clahe").displacement_px < 0.5


def test_probe_runs_on_uint16_too():
    """The linear-raw stage hands enhancers uint16; the probe must exercise
    the same dtype the enhancer will really see."""
    from enhancement_eval.contract import probe_geometry

    seen = []

    def spy(a):
        seen.append(a.dtype)
        return a.copy()

    probe_geometry(spy, name="spy", dtype=np.uint16)
    assert seen and all(d == np.uint16 for d in seen)


def test_probe_reports_the_measured_displacement_for_the_record():
    """The number goes in the report next to the arm's name, so a reader can
    see what geometric assurance the numbers rest on."""
    from enhancement_eval.contract import probe_geometry

    result = probe_geometry(IDENTITY, name="identity")
    assert result.name == "identity"
    assert result.displacement_px < 0.1
    assert not result.suspicious


def test_guard_returns_a_writeable_array_even_for_a_pass_through_enhancer():
    """A regression with a body count: the read-only view that catches in-place
    enhancers is also what IDENTITY returns unchanged, and OpenCV ignores
    numpy's WRITEABLE flag and writes into the buffer regardless. Handing that
    view to `LaserDetector.predict` dumps core -- so every *baseline* arm
    crashed the sweep until this was fixed."""
    out = guard(IDENTITY, name="identity")(_u8())
    assert out.flags.writeable


def test_guard_return_does_not_alias_the_protected_input():
    """Aliasing would let a later consumer write through into the array the
    other arm is still scoring from."""
    src = _u8()
    out = guard(IDENTITY, name="identity")(src)
    out[0, 0, 0] = 200
    assert src[0, 0, 0] != 200


def test_guard_does_not_copy_when_the_enhancer_already_allocated():
    """The copy is a correctness fix, not a blanket one: an enhancer that
    already returns fresh memory should not pay for a second full-frame copy
    on every image of a several-hundred-frame sweep."""
    made = np.zeros((8, 6, 3), dtype=np.uint8)
    out = guard(lambda a: made, name="fresh")(_u8())
    assert out is made


def test_importing_the_package_initializes_torch_before_rawpy():
    """A regression test for a core dump, not a style rule.

    `torch` must initialize its threading runtime before `rawpy` loads
    LibRaw's, or CPU inference segfaults with no traceback -- which on a
    several-hundred-frame sweep means an overnight run dies with no partial
    results and nothing to debug. Importing the package must be enough to get
    the order right; nothing should have to remember the rule.
    """
    import sys

    import enhancement_eval  # noqa: F401

    if "rawpy" in sys.modules:
        # If rawpy is loaded at all, torch must have got there first.
        assert "torch" in sys.modules


def test_probe_does_not_flag_a_global_tone_map_as_a_warp():
    """A regression on the probe itself.

    The probe perturbs one patch and asks where the output changed. That
    assumes influence is local -- true for a filter, false for a *global*
    operator. A percentile stretch recomputes its mapping from the whole
    frame's histogram, so adding a bright patch changes every pixel's output
    and the response centroid lands at the image centre, reading as ~19 px of
    displacement when nothing has moved.

    The global component has to be separated from the local one, or the probe
    rejects exactly the enhancement this project settled on.
    """
    import numpy as np
    from enhancement_eval.contract import probe_geometry

    def global_stretch(a):
        f = a.astype(np.float64)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99)
        return np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float64).__mul__(
            255
        ).astype(a.dtype)

    assert probe_geometry(global_stretch, name="global-stretch").displacement_px < 0.5


def test_probe_still_catches_a_roll_underneath_a_global_tone_map():
    """The dangerous combination: a warp hidden inside an operator that also
    changes the whole frame. Removing the global component must not remove the
    probe's ability to see the local displacement."""
    import numpy as np
    from enhancement_eval.contract import probe_geometry

    def stretch_and_roll(a):
        f = a.astype(np.float64)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99)
        out = np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1) * 255
        return np.roll(out.astype(a.dtype), 1, axis=1)

    with pytest.raises(GeometryViolation, match="moved"):
        probe_geometry(stretch_and_roll, name="stretch+roll")
