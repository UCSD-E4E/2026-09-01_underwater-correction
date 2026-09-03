"""CNDR: color-noise decoupling and reconstruction.

Reimplementation of Yan, Wang, Claramunt & Yang (2026), "Underwater image
enhancement via color-noise decoupling and reconstruction with application to
tidal stream turbine", Journal of Ocean Engineering and Marine Energy
12:1151-1163, doi:10.1007/s40722-026-00480-7. The authors released no code
("data available on request"), so this is built from the equations; equation
numbers below refer to the paper.

----------------------------------------------------------------------------
Why this method, out of the underwater-enhancement literature
----------------------------------------------------------------------------
Its stated problem is the one this project measured independently. Paper eq 3:
enhancement applies a per-channel gain, so ``y_hat = k_c*x + k_c*n`` -- the
noise receives exactly the gain the signal does, and worst in red, because red
needs the largest gain. That is the CLAHE open-water noise amplification we
measured at 36.7x, which is why the shipped recommendation turns CLAHE off in
favour of an L* stretch. The paper reaches the same place from theory: RCR and
FCR both operate on the CIELAB L channel specifically to avoid touching colour.

Two pieces are directly comparable to knobs we already have:

* **DAC** (sec 3.2) compensates the starved channels using green and blue as
  the reference rather than applying a large red gain, explicitly to avoid red
  overcompensation. That is a principled alternative to ``red_boost``, which we
  tuned by eye.
* **FCR** (sec 3.5) reweights wavelet coefficients so that edges are boosted
  more than noise. Note it is not a denoiser: its per-coefficient gain is
  ``1 + beta*alpha >= 1``, so noise grows too, just more slowly than detail.
  See :func:`feature_preserving_contrast`.

And it satisfies constraint #1 by construction: the transform is undecimated,
so nothing is ever resampled and every coefficient stays over the pixel it
describes. That is the property a 256x256 GAN cannot offer at 4014x3016.

----------------------------------------------------------------------------
What it measured, on our frames
----------------------------------------------------------------------------
Negative overall, against the shipped decode (L* stretch, CLAHE off), on six
reef and three pool frames -- see MEASUREMENTS.md. The complete published
method scores *below* what we ship on fish-vs-water separation (1.485 against
1.507). DAC is the weakest stage rather than the strongest: it costs a large
amount of fish detail (fish grain 6.05 against 10.61) and pushes red hard on
clear pool water (red mean 114 against our 69), which is the red
overcompensation it was designed to avoid, appearing out of distribution.

Kept in the tree because the transform and the metric are reusable, the
ablations are the evidence for the negative, and eq 17 gives a noise figure
that is directly comparable with the rest of MEASUREMENTS.md.

----------------------------------------------------------------------------
Two places where we implement what the paper means rather than what it says
----------------------------------------------------------------------------
**RCR is a scalar gain.** Eq 12 reads ``L_en = U (mu * Sigma) V^T`` with mu a
scalar from eq 13. But ``U (mu*Sigma) V^T = mu * (U Sigma V^T) = mu * L``, so
the SVD machinery collapses to multiplying the L channel by a scalar. We
implement the scalar -- which also avoids a full SVD of a 12-megapixel matrix
per frame -- and ``test_rcr_is_algebraically_a_scalar_gain`` is the evidence
that this is exact rather than an approximation. Only the largest singular
value is actually needed (eq 13), and that comes from power iteration.

**The normalization convention is pinned by a constant.** Sec 3.3 states
Cmax = 2040 for a 3-level SWT. 2040 = 255 * 2**3, so the LL range doubles per
level, which identifies orthonormal Haar filters (1/sqrt(2)) applied on both
axes. Guessing wrong here would silently rescale every compensation term in
eqs 6-10, so ``test_low_frequency_gain_matches_the_papers_2040_constant``
checks it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CNDRConfig",
    "HAAR_LEVEL_GAIN",
    "adaptive_gamma_correction",
    "cndr_enhance",
    "dual_stage_attenuation_compensation",
    "feature_preserving_contrast",
    "iswt2_haar",
    "ll_max_for_levels",
    "lowpass_synthesis",
    "reference_guided_contrast",
    "stretch_to_range",
    "swt2_haar",
    "wavelet_noise_sigma",
]

#: Factor by which the 2D LL range grows per decomposition level. Orthonormal
#: Haar contributes 1/sqrt(2) per axis, so 2 per level in 2D -- which is what
#: makes the paper's Cmax = 255 * 2**3 = 2040 come out right.
HAAR_LEVEL_GAIN = 2

_SQRT2 = np.sqrt(2.0, dtype=np.float64)
DEFAULT_LEVELS = 3


def ll_max_for_levels(levels: int, peak: float = 255.0) -> float:
    """Upper bound of the LL band for an image whose peak value is ``peak``.

    Paper sec 3.3 gives 2040 for a 3-level SWT of 8-bit data; this is that
    number, derived rather than hard-coded, so the code also works for a
    different level count or bit depth.
    """
    return float(peak) * float(HAAR_LEVEL_GAIN) ** int(levels)


# ---------------------------------------------------------------------------
# Stationary (undecimated) Haar wavelet transform -- paper sec 3.1, eq 5
# ---------------------------------------------------------------------------
# The "a trous" algorithm: instead of downsampling between levels, the filter
# is dilated. Every subband keeps the input's shape, coefficients stay aligned
# with their pixels, and the transform commutes with translation. Sec 3.1 picks
# SWT over DWT for the detail loss that downsampling causes; here it also
# carries the pixel-wise guarantee.


def _analysis(a: np.ndarray, axis: int, dilation: int) -> tuple[np.ndarray, np.ndarray]:
    """One level of Haar analysis along ``axis``, orthonormal, no decimation."""
    shifted = np.roll(a, -dilation, axis=axis)
    return (a + shifted) / _SQRT2, (a - shifted) / _SQRT2


def _synthesis(lo: np.ndarray, hi: np.ndarray | None, axis: int, dilation: int) -> np.ndarray:
    """Inverse of :func:`_analysis`.

    An undecimated level is redundant: both ``(lo+hi)/sqrt2`` and a shifted
    ``(lo-hi)/sqrt2`` reconstruct the signal. Averaging the two is the standard
    inverse and is what makes reconstruction exact.
    """
    if hi is None:
        first = lo / _SQRT2
        second = np.roll(lo, dilation, axis=axis) / _SQRT2
    else:
        first = (lo + hi) / _SQRT2
        second = np.roll(lo - hi, dilation, axis=axis) / _SQRT2
    return (first + second) / 2.0


def swt2_haar(
    image: np.ndarray, levels: int = DEFAULT_LEVELS
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Decompose a 2D image into ``levels`` of undecimated Haar subbands.

    Returns the deepest LL band -- the paper's "color components" -- and one
    ``(HL, LH, HH)`` triple per level, the "detail and noise components".
    Every array has the input's shape.
    """
    approx = np.asarray(image, dtype=np.float32)
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for level in range(levels):
        dilation = 1 << level
        lo_x, hi_x = _analysis(approx, axis=1, dilation=dilation)
        ll, lh = _analysis(lo_x, axis=0, dilation=dilation)
        hl, hh = _analysis(hi_x, axis=0, dilation=dilation)
        details.append((hl, lh, hh))
        approx = ll
    return approx, details


def iswt2_haar(
    ll: np.ndarray, details: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Invert :func:`swt2_haar`. Exact when the coefficients are unmodified."""
    approx = np.asarray(ll, dtype=np.float32)
    for level in range(len(details) - 1, -1, -1):
        dilation = 1 << level
        hl, lh, hh = details[level]
        lo_x = _synthesis(approx, lh, axis=0, dilation=dilation)
        hi_x = _synthesis(hl, hh, axis=0, dilation=dilation)
        approx = _synthesis(lo_x, hi_x, axis=1, dilation=dilation)
    return approx


def lowpass_synthesis(ll: np.ndarray, levels: int = DEFAULT_LEVELS) -> np.ndarray:
    """Reconstruct from an LL band with all detail bands zero.

    The point is memory, and at production frame sizes it is not a small point.
    ISWT is linear, so reconstructing with a modified LL and untouched details
    equals ``image + lowpass_synthesis(delta_LL)``. That lets the colour stage
    skip materializing nine detail subbands of a 12-megapixel frame -- roughly
    435 MB of float32 that never has to exist.
    """
    approx = np.asarray(ll, dtype=np.float32)
    for level in range(levels - 1, -1, -1):
        dilation = 1 << level
        lo_x = _synthesis(approx, None, axis=0, dilation=dilation)
        approx = _synthesis(lo_x, None, axis=1, dilation=dilation)
    return approx


# ---------------------------------------------------------------------------
# DAC -- dual-stage attenuation compensation, paper sec 3.2, eqs 6-10
# ---------------------------------------------------------------------------


def dual_stage_attenuation_compensation(ll_rgb: np.ndarray) -> np.ndarray:
    """Correct colour cast on the low-frequency components, in [0, 1].

    Stage 1 (eqs 6-7) handles the case the simple models miss: in turbid or
    low-light water *every* channel is attenuated, not just red, so green and
    blue are lifted using red as the reference.

    Stage 2 (eqs 8-10) then corrects the cast itself. The channels are ranked
    by mean, and the two weaker ones are compensated toward the strongest --
    but the reference is ``C' = (G_hat + B_hat)/2``, the mean of the surviving
    channels, *not* a gain applied to red. That is the whole point of the
    method for our purposes: a gain of `k` on a starved red channel multiplies
    its noise by `k` (eq 3), while an additive term built from clean channels
    does not.
    """
    out = np.asarray(ll_rgb, dtype=np.float32).copy()
    red = out[..., 0]

    # Stage 1: lift green and blue by their own deficit, referenced to red.
    for idx in (1, 2):
        out[..., idx] = out[..., idx] + (1.0 - float(out[..., idx].mean())) * red

    # Stage 2: compensate the two weaker channels toward the strongest, using
    # the mean of the compensated green and blue as the reference.
    reference = (out[..., 1] + out[..., 2]) / 2.0
    means = [float(out[..., c].mean()) for c in range(3)]
    order = np.argsort(means)[::-1]  # large, medium, small
    largest_mean = means[order[0]]
    for idx in order[1:]:
        out[..., idx] = out[..., idx] + (largest_mean - means[idx]) * reference
    return out


def stretch_to_range(channel: np.ndarray, low: float, high: float) -> np.ndarray:
    """Rescale to ``[low, high]`` -- paper eq 11.

    Needed because DAC works on normalized components while the detail bands
    were left untouched; the colour components have to return to the range the
    inverse transform expects or the two would recombine at different scales.
    """
    arr = np.asarray(channel, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, (low + high) / 2.0)
    return (arr - lo) / (hi - lo) * (high - low) + low


# ---------------------------------------------------------------------------
# Adaptive gamma correction (Huang et al. 2013) -- the reference image
# ---------------------------------------------------------------------------


def adaptive_gamma_correction(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Adaptive gamma correction with weighting distribution, on [0, 255].

    Huang, Cheng & Chiu (2013), cited by the paper in sec 3.4 as the way the
    reference image ``I_gamma`` is produced. The weighting distribution
    flattens the histogram's influence -- ``pdf_w = pdf_max * (norm_pdf)**a`` --
    so that a few dominant intensities (open water, in our frames) cannot
    monopolize the mapping the way plain histogram equalization lets them.

    A colour input is corrected on its value channel and the chromaticity is
    carried through by ratio, so this does not itself shift colour.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        value = arr.max(axis=-1)
        mapped = adaptive_gamma_correction(value, alpha=alpha)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(value > 1e-6, mapped / np.maximum(value, 1e-6), 1.0)
        return np.clip(arr * scale[..., None], 0.0, 255.0).astype(np.float32)

    hist, _ = np.histogram(np.clip(arr, 0, 255), bins=256, range=(0.0, 255.0))
    total = hist.sum()
    if total == 0:
        return arr.copy()
    pdf = hist.astype(np.float64) / total
    pdf_min, pdf_max = float(pdf.min()), float(pdf.max())
    if pdf_max - pdf_min < 1e-12:
        return arr.copy()

    weighted = pdf_max * ((pdf - pdf_min) / (pdf_max - pdf_min)) ** alpha
    weight_sum = weighted.sum()
    if weight_sum <= 0:
        return arr.copy()
    cdf_w = np.cumsum(weighted) / weight_sum
    gamma = np.clip(1.0 - cdf_w, 1e-3, None)

    levels = np.arange(256, dtype=np.float64)
    base = levels / 255.0
    lut = np.zeros(256, dtype=np.float64)
    # Skip level 0: 0**gamma is 1 where gamma has been clamped to 0, which
    # would map black to white.
    lut[1:] = 255.0 * np.power(base[1:], gamma[1:])
    # A lookup must be non-decreasing in intensity, or it would reorder pixels
    # and so invent structure. The formula is monotonic in principle; this is a
    # guard against the endpoints where gamma is clamped.
    lut = np.maximum.accumulate(lut)
    return np.interp(np.clip(arr, 0, 255), levels, lut).astype(np.float32)


# ---------------------------------------------------------------------------
# RCR -- reference-guided contrast reconstruction, paper sec 3.4, eqs 12-13
# ---------------------------------------------------------------------------


def _largest_singular_value(matrix: np.ndarray, iterations: int = 200) -> float:
    """Spectral norm by power iteration on ``A^T A``.

    Eq 13 needs only the largest singular value, and a full SVD of a
    4014x3016 frame per image is not worth paying for. The start vector is
    fixed rather than random so the result is deterministic.
    """
    a = np.asarray(matrix, dtype=np.float64)
    v = np.ones(a.shape[1], dtype=np.float64) / np.sqrt(a.shape[1])
    sigma = 0.0
    for _ in range(iterations):
        w = a.T @ (a @ v)
        norm = np.linalg.norm(w)
        if norm < 1e-30:
            return 0.0
        v = w / norm
        new_sigma = np.sqrt(norm)
        if abs(new_sigma - sigma) <= 1e-10 * max(new_sigma, 1.0):
            sigma = new_sigma
            break
        sigma = new_sigma
    return float(np.linalg.norm(a @ v))


def reference_guided_contrast(ll: np.ndarray, ll_reference: np.ndarray) -> np.ndarray:
    """Global contrast on the low-frequency L component -- eqs 12-13.

    ``mu = (max(Sigma_ref) + max(Sigma_L)) / (2 * max(Sigma_L))``, then eq 12.
    As the module docstring explains, eq 12 is a scalar multiply once the SVD
    is cancelled, so that is what runs here.
    """
    arr = np.asarray(ll, dtype=np.float32)
    sigma = _largest_singular_value(arr)
    if sigma <= 0:
        return arr.copy()
    sigma_ref = _largest_singular_value(np.asarray(ll_reference, dtype=np.float32))
    mu = (sigma_ref + sigma) / (2.0 * sigma)
    return (arr * mu).astype(np.float32)


# ---------------------------------------------------------------------------
# FCR -- feature-preserving contrast reconstruction, paper sec 3.5, eqs 14-16
# ---------------------------------------------------------------------------


def feature_preserving_contrast(
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    details_reference: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Local contrast with noise suppression on the detail bands.

    Eq 14 builds a per-coefficient weight ``alpha = log(|H|+1)/log(max|H|+1)``,
    near 0 for noise-scale coefficients and near 1 for real edges. Eq 15 scales
    by ``beta``, from the reference image's bands. Eq 16 is the part that
    distinguishes this from denoising:

        ``H_en = beta * (alpha * H) + H``

    The original coefficients are added back, so a feature cannot be filtered
    away -- the paper's answer to the blurring that denoising preprocessing
    causes (its Fig 2d-e).

    **This is not a denoiser, despite how the paper frames it.** The per-
    coefficient gain is ``1 + beta*alpha``, which is >= 1 everywhere, so every
    coefficient grows and none shrinks. The suppression in eq 14 is purely
    *relative*: noise-scale coefficients get a smaller alpha and so grow less
    than edges do. Absolute noise still rises. Measured on six reef frames,
    open-water grain went up 1.53x against the shipped decode while fish detail
    went up more -- an improvement in separation bought with more noise, which
    is the same trade CLAHE already offers and that we already declined.

    Stated explicitly because the paper's own framing ("suppresses noise",
    sec 3.5) invites the opposite reading, and because our measured failure was
    amplification rather than the presence of grain.
    """
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for level, level_ref in zip(details, details_reference):
        bands: list[np.ndarray] = []
        for band, band_ref in zip(level, level_ref):
            arr = np.asarray(band, dtype=np.float32)
            peak = float(np.abs(arr).max())
            if peak <= 0:
                bands.append(arr.copy())
                continue
            alpha = np.log(np.abs(arr) + 1.0) / np.log(peak + 1.0)
            peak_ref = float(np.abs(np.asarray(band_ref, dtype=np.float32)).max())
            beta = peak_ref / peak
            bands.append((beta * (alpha * arr) + arr).astype(np.float32))
        out.append(tuple(bands))  # type: ignore[arg-type]
    return out


# ---------------------------------------------------------------------------
# The paper's own noise metric, eq 17
# ---------------------------------------------------------------------------


def wavelet_noise_sigma(image: np.ndarray, levels: int = DEFAULT_LEVELS) -> float:
    """Mean MAD noise estimate over the diagonal subbands -- eq 17.

    ``sigma_bar = (1/n) * sum_i median|HH_i| / 0.6745``. Because the transform
    is orthonormal, this lands on the true sigma for Gaussian noise rather than
    merely ranking correctly, which makes it directly comparable with the
    open-water noise figures already in MEASUREMENTS.md.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    _, details = swt2_haar(arr, levels=levels)
    estimates = [float(np.median(np.abs(hh))) / 0.6745 for _, _, hh in details]
    return float(np.mean(estimates))


# ---------------------------------------------------------------------------
# The whole method
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CNDRConfig:
    """Stage toggles.

    Separately switchable because a bundled win is not attributable. DAC has to
    be comparable against the existing ``red_boost`` and FCR against the CLAHE
    noise amplification independently, or we learn only that "CNDR helps"
    without learning which part earned it.
    """

    levels: int = DEFAULT_LEVELS
    apply_dac: bool = True
    apply_rcr: bool = True
    apply_fcr: bool = True
    gamma_alpha: float = 0.5


def _colour_stage(rgb: np.ndarray, levels: int) -> np.ndarray:
    """DAC on the LL bands, reconstructed via the linear delta shortcut."""
    ll_max = ll_max_for_levels(levels)
    ll_stack = np.stack(
        [swt2_haar(rgb[..., c], levels=levels)[0] for c in range(3)], axis=-1
    )
    compensated = dual_stage_attenuation_compensation(ll_stack / ll_max)
    stretched = np.stack(
        [stretch_to_range(compensated[..., c], 0.0, ll_max) for c in range(3)], axis=-1
    )
    delta = stretched - ll_stack
    corrected = np.stack(
        [rgb[..., c] + lowpass_synthesis(delta[..., c], levels=levels) for c in range(3)],
        axis=-1,
    )
    return np.clip(corrected, 0.0, 255.0)


def _contrast_stage(rgb: np.ndarray, config: CNDRConfig) -> np.ndarray:
    """RCR and FCR on the CIELAB L channel of the colour-corrected image.

    Working on L is the paper's choice (sec 3), and it is also what we
    independently landed on for the shipped stretch: contrast changes that
    touch the channels separately move colour, and colour is what the slate and
    species labelling depend on.
    """
    from skimage.color import lab2rgb, rgb2lab

    reference = adaptive_gamma_correction(rgb, alpha=config.gamma_alpha)

    lab = rgb2lab(np.clip(rgb, 0, 255) / 255.0)
    lab_ref = rgb2lab(np.clip(reference, 0, 255) / 255.0)
    # skimage's L* is [0, 100]; carry it at [0, 255] so the wavelet stage and
    # the noise metric share one scale with everything else in this module.
    lightness = (lab[..., 0] * 2.55).astype(np.float32)
    lightness_ref = (lab_ref[..., 0] * 2.55).astype(np.float32)

    ll, details = swt2_haar(lightness, levels=config.levels)
    ll_ref, details_ref = swt2_haar(lightness_ref, levels=config.levels)

    if config.apply_rcr:
        ll = reference_guided_contrast(ll, ll_ref)
    if config.apply_fcr:
        details = feature_preserving_contrast(details, details_ref)

    enhanced = np.clip(iswt2_haar(ll, details) / 2.55, 0.0, 100.0)
    lab[..., 0] = enhanced
    return np.clip(lab2rgb(lab) * 255.0, 0.0, 255.0)


def cndr_enhance(image: np.ndarray, config: CNDRConfig | None = None) -> np.ndarray:
    """Run CNDR on an 8-bit RGB image, returning the same shape and dtype.

    Strictly pixel-wise: no resize, crop, or warp anywhere in the pipeline, and
    the undecimated transform means no resampling either. Verified by
    ``test_cndr_does_not_move_pixels`` through the harness's own geometry probe
    rather than by inspection.
    """
    config = config or CNDRConfig()
    original_dtype = np.asarray(image).dtype
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")

    if config.apply_dac:
        rgb = _colour_stage(rgb, config.levels)
    if config.apply_rcr or config.apply_fcr:
        rgb = _contrast_stage(rgb, config)

    if original_dtype == np.uint8:
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return rgb.astype(np.float32)
