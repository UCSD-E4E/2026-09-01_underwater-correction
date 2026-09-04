"""Noise2Void on the Bayer mosaic, trained on FishSense frames.

Krull, Buchholz & Jug (2019), "Noise2Void -- Learning Denoising from Single
Noisy Images", with the structured-noise mask of Broaddus et al. (2020).

This is the one learned method that does not inherit the out-of-distribution
problem that killed the slate detector on pool dives. It trains on our own
frames with no clean targets and no external dataset, so pool dives are
in-distribution because they are in the training set. And it is a denoiser
rather than a generator: it predicts a pixel from its neighbourhood and
structurally cannot invent content, which is what keeps it on the right side
of "do not put a generative model in the pipeline and declare victory because
the images look better".

Three design choices come from measurements in this project rather than from
the papers.

**It runs on the mosaic, not on RGB.** Per Chris: "by the time you have an
image, the noise has already contaminated the other colors." Measured here as
cross-channel noise correlation rising 0.029 -> 0.328 through demosaicing.
N2V's independence assumption has to be used where it is actually true, which
is at the photosites.

**The blind spot hides one plane at a time.** The four planes are distinct
photosites whose noise is very nearly independent (0.029 raw), so masking only
the target plane leaves the other three visible at that same site for the
network to predict from. That is the Bayer-aware cross-channel denoising Chris
asked about, in the domain where it is valid. Masking all four planes together
would be simpler and would throw that away.

**The mask is a horizontal line, not a point.** `noise_structure` measured this
sensor's noise as directional -- horizontal lag-1 correlation 7.5x the
vertical, decaying geometrically along the row, which is the readout chain's
bandwidth limit rather than the scene. A point mask leaves the correlated
horizontal neighbour visible for the network to copy the shared component
from. `recommended_mask_width` returned 5 plane pixels for our frames.

What this module deliberately does *not* claim: that a falling training loss
means anything. N2V's loss is computed against noisy targets and falls happily
while a network learns to reproduce noise. The evidence is
`test_training_reduces_error_against_the_clean_signal`, which measures error
against a known clean signal, and beyond that the usual harness run on real
frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "N2VConfig",
    "BlindSpotUNet",
    "apply_blind_spots",
    "denoise_planes",
    "make_blind_spots",
    "n2v_loss",
    "train",
]

PLANE_ORDER = ("R", "G1", "G2", "B")


@dataclass(frozen=True)
class N2VConfig:
    """Training knobs. Defaults are sized for a 6 GB card."""

    steps: int = 3000
    patch: int = 64
    batch: int = 32
    base: int = 32
    #: U-Net depth; see `BlindSpotUNet`. Fewer levels, smaller receptive
    #: field, less ability to average texture away.
    levels: int = 3
    lr: float = 3e-4
    mask_rate: float = 0.02
    #: Blind-spot width along the correlated axis. Take this from
    #: `noise_structure.recommended_mask_width` rather than guessing; 1 is a
    #: plain N2V point mask and is only right for isotropic noise.
    mask_width: int = 5
    #: Whether a site may be predicted from the other three planes.
    #:
    #: True exploits the near-independence of the four photosites' noise, which
    #: is the Bayer-aware cross-channel denoising this module was built for.
    #: But the planes are not co-located -- R, G1, G2 and B sit at the four
    #: corners of a Bayer quad -- and stacking them as channels at one array
    #: index asserts that they are. Borrowing across them therefore borrows
    #: from half a photosite away, and that showed up as a measured 0.21 px
    #: displacement of the finished decode. Set False to forbid the borrowing.
    cross_plane: bool = True
    seed: int = 0
    device: str | None = None


def _device(config: N2VConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Blind spots
# ---------------------------------------------------------------------------


def make_blind_spots(
    shape, rate: float, generator: torch.Generator | None = None, *, cross_plane: bool = True
) -> torch.Tensor:
    """Select pixels to hide, at most one plane per site.

    Sites are drawn first and a plane is then chosen for each, so the other
    three planes stay visible at that site. That is what lets the network learn
    the cross-plane relationship instead of only the spatial one.

    ``rate`` is the fraction of entries masked across the whole tensor, the
    conventional N2V meaning. Because each site masks only one of ``c`` planes,
    the site probability has to be ``rate * c`` for that to come out right --
    without the factor the effective rate would be a quarter of what was asked
    for, and the model would train on a quarter of the intended signal.

    With ``cross_plane=False`` every plane at a selected site is masked
    together, so the network cannot borrow across planes at all. That costs the
    cross-channel information but removes a mis-registration: the four planes
    sit at *different* positions in the Bayer quad, and stacking them as
    channels at one array index tells the network they are co-located. See
    `N2VConfig.cross_plane`.
    """
    b, c, h, w = shape
    device = generator.device if generator is not None else None
    site_rate = min(rate * c, 1.0) if cross_plane else rate
    site = torch.rand(b, 1, h, w, generator=generator, device=device) < site_rate
    if not cross_plane:
        return site.expand(b, c, h, w).clone()
    which = torch.randint(0, c, (b, 1, h, w), generator=generator, device=device)
    planes = torch.arange(c, device=which.device).view(1, c, 1, 1)
    return site & (which == planes)


def apply_blind_spots(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_width: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace each masked pixel -- and its horizontal band -- with a neighbour.

    Uniform pixel selection: the hidden value is overwritten with another value
    from the same plane, drawn from a small window. Replacing rather than
    zeroing keeps the input's statistics intact, so the network cannot identify
    masked positions by their value and learn to special-case them.

    ``mask_width`` blanks a horizontal segment centred on each masked pixel.
    The loss is still taken only at the centre; the extra width exists purely
    to deny the network the correlated horizontal neighbours.
    """
    out = x.clone()
    if not mask.any():
        return out
    half = mask_width // 2
    b, c, h, w = x.shape
    idx = mask.nonzero(as_tuple=False)

    # Donor values: a random shift, vertical so it can never land inside the
    # horizontal band being blanked.
    #
    # Reflected at the edges rather than clamped. Clamping lets a pixel in the
    # top rows donate to itself -- row 0 with shift 3 upward clamps back to row
    # 0 -- which leaves a "masked" pixel holding its original value. The
    # network then sees the answer it is being asked to predict, and those
    # sites teach it the identity.
    shift = torch.randint(2, 6, (idx.shape[0],), generator=generator, device=x.device)
    sign = torch.randint(0, 2, (idx.shape[0],), generator=generator, device=x.device) * 2 - 1
    donor_y = idx[:, 2] + shift * sign
    reflected = torch.where((donor_y < 0) | (donor_y > h - 1), idx[:, 2] - shift * sign, donor_y)
    donor_y = reflected.clamp(0, h - 1)

    for offset in range(-half, half + 1):
        xs = (idx[:, 3] + offset).clamp(0, w - 1)
        out[idx[:, 0], idx[:, 1], idx[:, 2], xs] = x[idx[:, 0], idx[:, 1], donor_y, xs]
    return out


def n2v_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error at the hidden pixels only.

    Only the hidden pixels supply gradient. If unmasked pixels counted, the
    network would be trained toward the identity -- it would learn to copy its
    input, noise included, which is the trivial minimum of that objective.
    """
    count = mask.sum()
    if count == 0:
        return prediction.sum() * 0.0
    return ((prediction - target) ** 2)[mask].sum() / count


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
    )


_LEGACY_KEYS = {
    "enc1": "encoders.0", "enc2": "encoders.1", "enc3": "encoders.2",
    "up2": "ups.0", "dec2": "decoders.0", "up1": "ups.1", "dec1": "decoders.1",
}


def _upgrade_legacy_keys(state_dict):
    """Map a three-level checkpoint's original layer names onto the current ones."""
    if not any(k.split(".")[0] in _LEGACY_KEYS for k in state_dict):
        return state_dict
    out = {}
    for key, value in state_dict.items():
        head, _, rest = key.partition(".")
        out[f"{_LEGACY_KEYS.get(head, head)}.{rest}" if rest else _LEGACY_KEYS.get(head, head)] = value
    return out


class BlindSpotUNet(nn.Module):
    """A small U-Net, 4 planes in and 4 out, with a configurable depth.

    Deliberately small: the receptive field only has to cover the neighbourhood
    that predicts a pixel, and a larger model on this much data would start
    memorizing scene content rather than learning noise statistics. It also has
    to fit alongside 12-megapixel inference on a 6 GB card.

    ``levels`` is the lever on texture. A blind-spot network suppresses
    whatever it cannot predict from context, and the wider its context, the
    more it can average across. Three levels see roughly 50 px of plane -- the
    whole body of a labelled fish -- and can replace scale texture with the
    body's mean shade, which is what the first model did (fish detail /4.6).
    One level is a local filter of about 5 px; two sit between. Measured in
    `test_fewer_levels_means_a_smaller_receptive_field`.

    Odd input sizes are padded internally and cropped back, so the output is
    exactly the input's shape -- production planes are 2007x1508, and padding
    that leaked to the caller would be a geometric change.
    """

    def __init__(self, channels: int = 4, base: int = 32, levels: int = 3):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        widths = [base * (2 ** i) for i in range(levels)]
        self.encoders = nn.ModuleList(
            [_block(channels, widths[0])]
            + [_block(widths[i - 1], widths[i]) for i in range(1, levels)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2) for i in range(levels - 1, 0, -1)]
        )
        self.decoders = nn.ModuleList(
            [_block(widths[i - 1] * 2, widths[i - 1]) for i in range(levels - 1, 0, -1)]
        )
        self.out = nn.Conv2d(widths[0], channels, 1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts checkpoints saved before `levels` existed.

        The first trained models named their layers enc1/enc2/enc3/up2/dec2/
        up1/dec1. Renaming the modules for a configurable depth silently broke
        loading them, which cost a re-measurement; a saved model should not
        stop loading because the class that made it was refactored.
        """
        return super().load_state_dict(_upgrade_legacy_keys(state_dict), strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        multiple = 2 ** (self.levels - 1)
        ph, pw = (-h) % multiple, (-w) % multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.decoders):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.out(x)[:, :, :h, :w]


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


def train(planes: np.ndarray, config: N2VConfig | None = None, *, log=None) -> BlindSpotUNet:
    """Train on an array of shape ``(n_frames, 4, H, W)``.

    Patches are drawn uniformly at random across frames and positions. Note the
    reported loss is against *noisy* targets and so has a floor at the noise
    variance -- it falling is not evidence of anything, which is why the real
    check lives in the tests and in the harness run.
    """
    config = config or N2VConfig()
    device = _device(config)
    torch.manual_seed(config.seed)
    # Sampling and masking happen on the CPU, where the dataset lives. A real
    # training set is tens of gigabytes of photosite planes and does not fit
    # beside the model on a 6 GB card; only the batch crosses to the GPU.
    generator = torch.Generator().manual_seed(config.seed)

    data = torch.as_tensor(np.asarray(planes, dtype=np.float32))
    n, c, h, w = data.shape
    if h < config.patch or w < config.patch:
        raise ValueError(f"patch {config.patch} does not fit in {h}x{w} planes")

    model = BlindSpotUNet(channels=c, base=config.base, levels=config.levels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()

    for step in range(config.steps):
        # CPU indices: the dataset is on the CPU, and only the batch moves.
        fi = torch.randint(0, n, (config.batch,), generator=generator)
        ys = torch.randint(0, h - config.patch + 1, (config.batch,), generator=generator)
        xs = torch.randint(0, w - config.patch + 1, (config.batch,), generator=generator)
        batch = torch.stack([
            data[fi[i], :, ys[i]:ys[i] + config.patch, xs[i]:xs[i] + config.patch]
            for i in range(config.batch)
        ])

        mask = make_blind_spots(batch.shape, config.mask_rate, generator=generator,
                                cross_plane=config.cross_plane)
        masked = apply_blind_spots(batch, mask, config.mask_width, generator=generator)
        batch, masked, mask = batch.to(device), masked.to(device), mask.to(device)
        loss = n2v_loss(model(masked), batch, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log is not None and (step % 200 == 0 or step == config.steps - 1):
            log(step, float(loss.item()))
    model.eval()
    return model


@torch.no_grad()
def denoise_planes(
    model: BlindSpotUNet, planes: np.ndarray, *, tile: int = 512, overlap: int = 32
) -> np.ndarray:
    """Run the model over full-size planes, returning the same shape.

    Tiled with an overlap that is cropped away, because a 12-megapixel plane
    stack does not fit in 6 GB and because a tile's edge pixels have a
    truncated receptive field. Only each tile's interior is kept, so no seam
    appears in the output -- the reason `probe_geometry` is run on the full
    decode path rather than on a single tile.
    """
    arr = np.asarray(planes, dtype=np.float32)
    single = arr.ndim == 3
    if single:
        arr = arr[None]
    device = next(model.parameters()).device
    out = np.empty_like(arr)

    for i in range(arr.shape[0]):
        x = torch.as_tensor(arr[i][None], device=device)
        _, _, h, w = x.shape
        acc = torch.zeros_like(x)
        step = max(tile - 2 * overlap, 1)
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                ya, xa = max(0, y1 - tile), max(0, x1 - tile)
                patch = model(x[:, :, ya:y1, xa:x1])
                # Keep only the interior, except at the true frame edges where
                # there is no neighbouring tile to supply a better estimate.
                ty0 = 0 if ya == 0 else overlap
                tx0 = 0 if xa == 0 else overlap
                ty1 = patch.shape[2] if y1 == h else patch.shape[2] - overlap
                tx1 = patch.shape[3] if x1 == w else patch.shape[3] - overlap
                acc[:, :, ya + ty0:ya + ty1, xa + tx0:xa + tx1] = patch[:, :, ty0:ty1, tx0:tx1]
        out[i] = acc[0].cpu().numpy()
    return out[0] if single else out
