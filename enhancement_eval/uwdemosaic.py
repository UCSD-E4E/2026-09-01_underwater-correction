"""A demosaic that allocates spatial detail by measured per-channel SNR.

Every stock demosaic tested on this corpus -- AHD, DHT, AAHD, DCB, PPG, VNG,
linear, and AHD/DHT with LibRaw's FBDD -- lands on the same 1:1 exchange of
noise for detail. Sharper ones recover 16% more fish detail and 18% more noise;
smoother ones give up both. They differ in *where* they sit on that line, not in
the rate.

They share an assumption that is false underwater: that the three channels are
equally worth interpolating. Measured on a real frame, green sits at 927.5 DN
with an SNR of 8.3 per photosite; red sits at 110.9 DN with roughly a third of
that. Recovering red's high frequencies faithfully means recovering its noise
faithfully -- and then paying a denoiser to take it back out.

So interpolate the **colour difference** instead:

    R_full = G_full + smooth(R - G)

Red keeps its own slow chromatic variation and inherits its sharp structure
from green, which has the photons to justify it. The justification for smoothing
that difference hard is physical, not merely convenient: underwater the colour
difference is set by the water column, and the attenuation fit showed that
varies smoothly with range (`attenuation.py`).

This is **not a new algorithm class**. Residual and colour-difference
interpolation are long established, and green-guided chroma reconstruction is
standard in camera ISPs. What is specific here is choosing the frequency split
from the measured per-channel SNR of *this* sensor in *this* water, and having a
physical reason to believe the difference channel is smooth rather than just
hoping it is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["demosaic_underwater"]

#: Default half-width, in pixels, of the smoothing applied to the colour
#: difference. Larger trusts green more and red's own samples less. 6 is chosen
#: from the measured SNR ratio (green ~8.3, red ~3): red carries roughly a third
#: of green's usable bandwidth, so its detail is borrowed over a neighbourhood
#: several pixels wide rather than reconstructed per-pixel.
DEFAULT_CHROMA_RADIUS = 6


def _offsets(pattern: np.ndarray, desc: str) -> dict[str, tuple[int, int]]:
    found: dict[str, list[tuple[int, int]]] = {}
    for di in range(2):
        for dj in range(2):
            found.setdefault(desc[int(pattern[di, dj])], []).append((di, dj))
    if len(found.get("G", [])) != 2 or len(found.get("R", [])) != 1 or len(found.get("B", [])) != 1:
        raise ValueError(
            f"not a Bayer pattern (R={len(found.get('R', []))} "
            f"G={len(found.get('G', []))} B={len(found.get('B', []))})"
        )
    return {"R": found["R"][0], "G1": found["G"][0], "G2": found["G"][1], "B": found["B"][0]}


def _normalized_blur(values: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Blur samples that exist only where `mask` is 1, without dragging the
    zeros in between into the average.

    Dividing the blurred values by the blurred mask is what makes this a
    weighted mean over the samples actually present -- a plain blur of a sparse
    array would scale every result by the sampling density instead.
    """
    import cv2

    k = 2 * radius + 1
    box = lambda a: cv2.boxFilter(  # noqa: E731
        a, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT
    )
    weight = box(mask)
    return box(values) / np.maximum(weight, 1e-6)


def demosaic_underwater(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    desc: str,
    *,
    chroma_radius: int = DEFAULT_CHROMA_RADIUS,
    green_radius: int = 1,
) -> np.ndarray:
    """Demosaic a CFA frame, borrowing red and blue detail from green.

    Args:
        mosaic: 2-D CFA frame.
        pattern: rawpy's ``raw_pattern``.
        desc: rawpy's ``color_desc``, e.g. ``"RGBG"``.
        chroma_radius: smoothing half-width for the colour difference. This is
            the knob that trades chroma noise against chromatic resolution.
        green_radius: smoothing for filling green's missing quincunx samples.
            Kept small -- green is the channel whose detail is worth keeping.

    Returns:
        ``(H, W, 3)`` float RGB, same spatial size as the input. No pixel moves.
    """
    frame = mosaic.astype(np.float64, copy=False)
    height, width = frame.shape
    offsets = _offsets(pattern, desc)

    # Green: known on a quincunx (half the pixels). Fill the gaps with a small
    # normalized blur -- deliberately local, because this is the channel whose
    # high frequencies are worth having.
    green = np.zeros((height, width))
    green_mask = np.zeros((height, width))
    for key in ("G1", "G2"):
        di, dj = offsets[key]
        green[di::2, dj::2] = frame[di::2, dj::2]
        green_mask[di::2, dj::2] = 1.0
    green_full = _normalized_blur(green, green_mask, green_radius)
    # Keep the measured green samples exactly; only the gaps are interpolated.
    green_full = np.where(green_mask > 0, green, green_full)

    out = np.empty((height, width, 3), dtype=np.float64)
    out[:, :, 1] = green_full

    for index, key in ((0, "R"), (2, "B")):
        di, dj = offsets[key]
        mask = np.zeros((height, width))
        mask[di::2, dj::2] = 1.0
        # The colour difference, sampled only where this channel was measured.
        difference = np.zeros((height, width))
        difference[di::2, dj::2] = frame[di::2, dj::2] - green_full[di::2, dj::2]
        out[:, :, index] = green_full + _normalized_blur(difference, mask, chroma_radius)

    return np.clip(out, 0.0, None)
