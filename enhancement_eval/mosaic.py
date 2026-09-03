"""Denoising in the CFA domain, before demosaicing correlates the channels.

Cross-channel denoising in RGB space assumes one channel can act as an
independent, cleaner reference for another. Demosaicing destroys that
assumption: each output pixel's R, G and B are interpolated from overlapping
neighbourhoods of photosites, so their noise becomes shared. Measured on a real
reef frame, the correlation between channels' high-frequency residuals is

    before demosaic   corr(R,G) = +0.029   corr(R,B) = +0.009   corr(G,B) = +0.101
    after  demosaic   corr(R,G) = +0.328   corr(R,B) = +0.145   corr(G,B) = +0.308

an order of magnitude higher. A guide whose noise is a third shared with the
channel it guides cannot separate that channel's signal from its noise, which
is why the RGB-domain attempt removed signal and noise in equal proportion.

At the mosaic each photosite is one independent measurement, so a guide built
there is legitimate. Two further things fall out for free:

* **G1 - G2 is signal-free.** The two green photosites in a 2x2 cell sample
  nearly the same scene point, so their difference has no scene content and
  measures noise directly. On the same frame that gives sigma = 112.2 DN
  against a mean green level of 927.5 DN -- an SNR of 8.3 per photosite. No
  robust-estimator approximation is needed, and unlike MAD it cannot collapse
  on a starved or coarsely-quantised channel.
* **Green is genuinely the better measurement.** It has twice the photosites
  and, on that frame, 8.4x the signal level of red (927.5 DN against 110.9).
  Guiding red with green is borrowing from a real advantage, not an assumed one.

The denoised mosaic is written back and handed to the normal `rawpy`
postprocess, so white balance, colour matrix and demosaic are untouched.
"""

from __future__ import annotations

import numpy as np

__all__ = ["split_cfa", "merge_cfa", "green_noise_sigma", "denoise_cfa", "decode_denoised"]


def _offsets(pattern: np.ndarray, desc: str) -> dict[str, tuple[int, int]]:
    """Map each photosite role to its (row, col) offset inside the 2x2 cell."""
    found: dict[str, list[tuple[int, int]]] = {}
    for di in range(2):
        for dj in range(2):
            found.setdefault(desc[int(pattern[di, dj])], []).append((di, dj))
    if len(found.get("G", [])) != 2 or len(found.get("R", [])) != 1 or len(found.get("B", [])) != 1:
        raise ValueError(
            f"not a Bayer pattern (R={len(found.get('R', []))} "
            f"G={len(found.get('G', []))} B={len(found.get('B', []))}); "
            "writing planes back under this assumption would scramble the mosaic"
        )
    g1, g2 = found["G"]
    return {"R": found["R"][0], "G1": g1, "G2": g2, "B": found["B"][0]}


def split_cfa(mosaic: np.ndarray, pattern: np.ndarray, desc: str) -> dict[str, np.ndarray]:
    """Split a CFA frame into its four half-resolution photosite planes."""
    return {
        key: mosaic[di::2, dj::2].astype(np.float64, copy=True)
        for key, (di, dj) in _offsets(pattern, desc).items()
    }


def merge_cfa(
    planes: dict[str, np.ndarray], pattern: np.ndarray, desc: str, shape
) -> np.ndarray:
    """Reassemble photosite planes into a CFA frame. Inverse of `split_cfa`."""
    out = np.zeros(shape, dtype=np.float64)
    for key, (di, dj) in _offsets(pattern, desc).items():
        out[di::2, dj::2] = planes[key]
    return out


def green_noise_sigma(planes: dict[str, np.ndarray]) -> float:
    """Per-photosite noise, measured from the two greens.

    G1 and G2 sample nearly the same scene point, so their difference cancels
    the signal and leaves noise. Var(G1-G2) = 2*sigma^2 for independent
    photosites, hence the sqrt(2).

    This is a *measurement*, not an estimate: it needs no assumption about what
    fraction of the frame is flat, and it cannot collapse to zero on a dark or
    quantised channel the way a median-absolute-deviation does.
    """
    diff = planes["G1"] - planes["G2"]
    return float(np.std(diff) / np.sqrt(2.0))


def denoise_cfa(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    desc: str,
    *,
    strength: float = 0.6,
    radius: int = 3,
) -> np.ndarray:
    """Guided denoise of every photosite plane, using the greens as the guide.

    The guide is (G1+G2)/2 -- the highest-SNR measurement available, with twice
    the photosites of red or blue and, underwater, several times their signal
    level. Because this happens before interpolation, its noise really is
    independent of red's and blue's, which is the condition the RGB-domain
    version could not satisfy.

    `strength` scales the guided filter's epsilon against the *measured* noise,
    so it means "smooth structure weaker than k noise-sigmas" and transfers
    across frames whose noise differs.
    """
    from enhancement_eval.crosschannel import guided_filter

    planes = split_cfa(mosaic, pattern, desc)
    sigma = green_noise_sigma(planes)
    k = 0.5 + 3.5 * float(np.clip(strength, 0.0, 1.0))
    eps = (k * sigma) ** 2

    guide = 0.5 * (planes["G1"] + planes["G2"])
    out = {key: guided_filter(guide, plane, radius=radius, eps=eps)
           for key, plane in planes.items()}
    return merge_cfa(out, pattern, desc, mosaic.shape)


def decode_denoised(source, *, strength: float = 0.6, radius: int = 3) -> np.ndarray:
    """Decode a raw with CFA denoising applied before demosaicing.

    The denoised mosaic is written back into rawpy's buffer, so `postprocess`
    performs the identical demosaic, white balance and colour matrix it would
    otherwise -- the only change is that it starts from a cleaner mosaic.
    """
    import rawpy

    from fishsense_core.image.image import open_image_source

    with open_image_source(source) as handle:
        with rawpy.imread(handle) as raw:
            visible = raw.raw_image_visible
            cleaned = denoise_cfa(
                visible.astype(np.float64), np.asarray(raw.raw_pattern),
                raw.color_desc.decode(), strength=strength, radius=radius,
            )
            info = np.iinfo(visible.dtype)
            visible[:] = np.clip(np.rint(cleaned), info.min, info.max).astype(visible.dtype)
            return raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
                output_bps=16, user_flip=0,
            )
