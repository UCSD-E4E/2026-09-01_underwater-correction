"""BM3D on the luminance of the finished decode, with a measured noise PSD.

Why this, after Noise2Void. Every N2V variant erased the dive 223 scale
lattice completely -- retention 0.00 for random crops, fish crops, and a
shallow net alike -- and the reason is geometric rather than a training
choice. The lattice's 4.5 px period in the output is 2.25 px in the
half-resolution photosite planes N2V works on: at the plane's Nyquist limit,
where a blind-spot network cannot tell a periodic pattern from pixel noise.
Measured on the raw planes, the lattice's strongest excess sits at 2.57 px in
G1 and 1.59 px (aliased) in R. Anything that hopes to keep fine scales has to
work at full resolution.

BM3D (Dabov et al. 2007) is the classical answer for exactly this texture. It
groups similar patches across the image and filters them jointly in a 3D
transform, so a periodic lattice -- as self-similar as image content gets --
is reinforced rather than averaged away. It is non-learned, deterministic,
and can only remove.

Two choices particular to this pipeline:

**It takes a noise PSD, not a sigma.** Demosaicing colours the noise: it rolls
off at high frequency. A white-noise sigma would over-smooth there, which is
precisely where the scales live. The open-water patch's spectrum, which every
other sweep here already measures, *is* the PSD.

**It sits at the JPEG stage, on L only.** After demosaic, in CIELAB, with a and
b passed through untouched -- so hue, which the species and slate labels
depend on, does not move, and the arm satisfies constraint #2 and goes
through `probe_geometry` like any other enhancer. The mosaic-domain work
could not offer either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BM3DConfig", "bm3d_enhancer", "denoise_luminance", "noise_psd_from_water"]


def _import_bm3d():
    """The bm3d package (4.0.x) still calls ``np.trapz``, which NumPy 2
    removed in favour of ``np.trapezoid``. Same function, new name; the alias
    is restored before import so the package loads on this numpy."""
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    import bm3d

    return bm3d


@dataclass(frozen=True)
class BM3DConfig:
    """``psd_size`` is the tile used to estimate the PSD from open water;
    ``strength`` scales the PSD handed to BM3D (1.0 = as measured)."""

    psd_size: int = 64
    strength: float = 1.0
    profile: str = "np"


def noise_psd_from_water(water: np.ndarray, size: int = 64) -> np.ndarray:
    """Noise power spectral density from a textureless patch.

    Welch-style: the patch is cut into ``size``-square tiles, each mean-removed
    and Hann-windowed, and their |FFT|^2 averaged. Normalized by the window's
    energy so white noise of variance sigma^2 reads sigma^2 in every bin --
    the same convention as `texture.radial_power_spectrum`. Returned
    fft-shifted, DC at the centre.
    """
    arr = np.asarray(water, dtype=np.float64)
    h, w = arr.shape
    size = int(min(size, h, w))
    win = np.outer(np.hanning(size), np.hanning(size))
    norm = float((win ** 2).sum())
    acc = np.zeros((size, size))
    n = 0
    for y in range(0, h - size + 1, size // 2):
        for x in range(0, w - size + 1, size // 2):
            tile = arr[y:y + size, x:x + size]
            tile = tile - tile.mean()
            acc += np.abs(np.fft.fft2(tile * win)) ** 2 / norm
            n += 1
    return np.fft.fftshift(acc / max(n, 1))


def _psd_for_image(psd: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a small PSD onto an image-sized frequency grid, in the
    convention the `bm3d` package expects.

    The package wants a PSD the size of the image, in unnormalized-FFT units:
    white noise of variance sigma^2 is ``sigma^2 * M * N`` in every bin. Ours
    is per-bin variance, so the factor is M*N. Pinned by
    `test_psd_array_matches_scalar_sigma_on_white_noise`.
    """
    from scipy.ndimage import zoom

    h, w = shape
    resampled = zoom(psd, (h / psd.shape[0], w / psd.shape[1]), order=1)
    resampled = np.maximum(resampled, 1e-12)
    return np.fft.ifftshift(resampled) * float(h * w)


def denoise_luminance(
    lum: np.ndarray, water: np.ndarray, config: BM3DConfig | None = None
) -> np.ndarray:
    """BM3D on a [0, 255] luminance array, noise model taken from ``water``."""
    bm3d = _import_bm3d()

    config = config or BM3DConfig()
    arr = np.asarray(lum, dtype=np.float64) / 255.0
    psd = noise_psd_from_water(np.asarray(water, dtype=np.float64) / 255.0, config.psd_size)
    psd_img = _psd_for_image(psd * config.strength, arr.shape)
    out = bm3d.bm3d(arr, sigma_psd=psd_img, profile=config.profile)
    return np.clip(np.asarray(out, dtype=np.float64) * 255.0, 0.0, 255.0)


def bm3d_enhancer(config: BM3DConfig | None = None):
    """An `Enhancer`: uint8 RGB in, uint8 RGB out, L filtered, a and b kept.

    The open-water patch is the top-left fifth of the frame, the same
    convention as every noise figure in MEASUREMENTS.md.
    """
    from skimage.color import lab2rgb, rgb2lab

    config = config or BM3DConfig()

    def enhance(image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image)
        lab = rgb2lab(rgb.astype(np.float64) / 255.0)
        lum = lab[..., 0] * 2.55
        h, w = lum.shape
        water = lum[: max(h // 5, config.psd_size), : max(w // 5, config.psd_size)]
        lab[..., 0] = denoise_luminance(lum, water, config) / 2.55
        out = np.clip(lab2rgb(lab) * 255.0, 0, 255)
        return np.rint(out).astype(np.uint8)

    return enhance
