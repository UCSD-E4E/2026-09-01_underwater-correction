"""Noise2Void on the Bayer mosaic, trained on our own frames.

The one learned method that does not inherit the out-of-distribution problem
that killed the slate detector: it trains on FishSense frames themselves, with
no clean targets and no external dataset, so pool dives are in-distribution
because they are in the training set. It is a denoiser, not a generator -- it
predicts a pixel from its neighbourhood and structurally cannot invent content.

Three design choices here come from measurements rather than from the paper:

**It runs on the mosaic, not on RGB.** Per Chris: "by the time you have an
image, the noise has already contaminated the other colors." Measured in this
project as cross-channel noise correlation going 0.029 -> 0.328 through
demosaicing. N2V's independence assumption has to be tested and used where it
is true, which is at the photosites.

**The blind spot masks one channel at a time.** The four planes are separate
photosites with independent noise (0.029 correlation raw), so hiding only the
target plane's pixel leaves the other three visible at that same site for the
network to predict from. That is the Bayer-aware cross-channel denoising Chris
asked about, in the domain where it is valid -- and it is why the mask is
per-channel rather than shared across all four.

**The mask is a 5-wide horizontal line, not a point.** `noise_structure`
measured the sensor's noise as directional: horizontal lag-1 correlation 7.5x
the vertical. A point mask leaves the correlated horizontal neighbour visible
to copy from. See `test_struct_mask_covers_the_correlated_axis`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from enhancement_eval.n2v import (  # noqa: E402
    N2VConfig,
    BlindSpotUNet,
    apply_blind_spots,
    denoise_planes,
    make_blind_spots,
    n2v_loss,
    train,
)


def _rng(seed=20260903):
    return np.random.default_rng(seed)


def _batch(b=2, c=4, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, c, h, w, generator=g)


# ---------------------------------------------------------------------------
# Blind spots
# ---------------------------------------------------------------------------

def test_blind_spots_select_roughly_the_requested_rate():
    mask = make_blind_spots((4, 4, 64, 64), rate=0.02, generator=torch.Generator().manual_seed(1))
    frac = mask.float().mean().item()
    assert 0.01 < frac < 0.035, f"selected {frac:.4f} of pixels"


def test_blind_spots_hit_one_channel_per_location():
    """The point of the per-channel mask: at a selected site the other three
    planes stay visible, so the network can use them to predict this one."""
    mask = make_blind_spots((2, 4, 48, 48), rate=0.05, generator=torch.Generator().manual_seed(2))
    per_site = mask.sum(dim=1)
    assert per_site.max().item() <= 1, "a site must never mask more than one plane"
    assert per_site.sum().item() > 0


def test_blind_spots_are_reproducible_from_a_seed():
    a = make_blind_spots((2, 4, 32, 32), rate=0.05, generator=torch.Generator().manual_seed(7))
    b = make_blind_spots((2, 4, 32, 32), rate=0.05, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_applying_blind_spots_changes_the_masked_centres():
    x = _batch()
    mask = make_blind_spots(x.shape, rate=0.1, generator=torch.Generator().manual_seed(3))
    out = apply_blind_spots(x, mask, mask_width=1, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(out[mask], x[mask]) or mask.sum() == 0


def test_applying_blind_spots_leaves_unmasked_pixels_untouched():
    """Only the masked band may change. If anything else moves, the loss is no
    longer measuring what it claims to."""
    x = _batch()
    mask = make_blind_spots(x.shape, rate=0.02, generator=torch.Generator().manual_seed(4))
    out = apply_blind_spots(x, mask, mask_width=1, generator=torch.Generator().manual_seed(4))
    changed = (out != x)
    assert torch.equal(changed & mask, mask), "every masked centre must change"
    assert changed.sum() == mask.sum(), "a width-1 mask must change exactly the centres"


def test_struct_mask_covers_the_correlated_axis():
    """StructN2V: with a 5-wide horizontal mask the two neighbours on each side
    are blanked too, so the network cannot read the horizontally correlated
    noise the sensor measurement found."""
    x = _batch(b=1, c=4, h=32, w=32)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 1, 16, 16] = True
    out = apply_blind_spots(x, mask, mask_width=5, generator=torch.Generator().manual_seed(5))
    changed = (out != x)[0, 1, 16]
    assert changed[14:19].all(), "the full 5-wide band must be replaced"
    assert not changed[:14].any() and not changed[19:].any(), "nothing outside the band"
    assert not (out != x)[0, 1, 15].any(), "the mask is horizontal, not a block"
    assert not (out != x)[0, 0].any(), "other planes stay visible at this site"


def test_struct_mask_stays_inside_the_patch_at_an_edge():
    x = _batch(b=1, c=4, h=16, w=16)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, 0, 8, 0] = True
    out = apply_blind_spots(x, mask, mask_width=5, generator=torch.Generator().manual_seed(6))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_ignores_unmasked_pixels():
    """The defining property of N2V's objective: only the hidden pixels supply
    gradient. If unmasked pixels counted, the network would be trained toward
    the identity and would learn to pass noise straight through."""
    pred, target = _batch(seed=1), _batch(seed=2)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[0, 0, 5, 5] = True
    loss = n2v_loss(pred, target, mask)
    expected = (pred[0, 0, 5, 5] - target[0, 0, 5, 5]) ** 2
    assert torch.allclose(loss, expected, atol=1e-6)


def test_loss_of_a_perfect_prediction_is_zero():
    x = _batch()
    mask = make_blind_spots(x.shape, rate=0.05, generator=torch.Generator().manual_seed(8))
    assert n2v_loss(x, x, mask).item() == pytest.approx(0.0, abs=1e-9)


def test_loss_with_no_mask_is_zero_not_nan():
    x = _batch()
    mask = torch.zeros_like(x, dtype=torch.bool)
    assert torch.isfinite(n2v_loss(x, x * 2, mask))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def test_unet_preserves_spatial_shape():
    net = BlindSpotUNet(channels=4, base=8)
    out = net(_batch(b=2, c=4, h=64, w=64))
    assert out.shape == (2, 4, 64, 64)


def test_unet_handles_sizes_that_are_not_multiples_of_the_downsampling():
    """Production planes are 2007x1508 -- neither dimension is a multiple of 8,
    and padding the input would be a geometric change."""
    net = BlindSpotUNet(channels=4, base=8)
    assert net(_batch(b=1, c=4, h=53, w=37)).shape == (1, 4, 53, 37)


# ---------------------------------------------------------------------------
# The claim that matters: does it actually remove noise?
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_training_reduces_error_against_the_clean_signal():
    """The only test that checks the thing we care about.

    A falling training loss proves nothing -- N2V's loss is measured against
    noisy targets, so it falls even when the network learns to reproduce noise.
    Here the clean signal is known, so error against it can be measured
    directly. If a trained model is not closer to clean than its noisy input
    is, the method does not work on this data no matter how the loss curve
    looks.
    """
    r = _rng()
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    clean = np.stack([
        0.5 + 0.3 * np.sin(xx / 9 + c) * np.cos(yy / 11) for c in range(4)
    ])
    noisy = clean + r.normal(0, 0.12, clean.shape).astype(np.float32)

    cfg = N2VConfig(steps=400, patch=64, batch=8, base=16, mask_width=1, seed=11)
    model = train(noisy[None], cfg)
    out = denoise_planes(model, noisy)

    before = float(np.mean((noisy - clean) ** 2))
    after = float(np.mean((out - clean) ** 2))
    assert after < before * 0.8, f"MSE to clean {before:.5f} -> {after:.5f}, not enough"


@pytest.mark.integration
def test_training_is_reproducible_from_a_seed():
    r = _rng()
    data = r.normal(0.5, 0.1, (1, 4, 96, 96)).astype(np.float32)
    cfg = N2VConfig(steps=30, patch=32, batch=4, base=8, seed=99)
    a = denoise_planes(train(data, cfg), data[0])
    b = denoise_planes(train(data, cfg), data[0])
    assert np.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------

def test_denoise_preserves_plane_shape():
    net = BlindSpotUNet(channels=4, base=8)
    planes = _rng().normal(0.5, 0.05, (4, 40, 52)).astype(np.float32)
    assert denoise_planes(net, planes).shape == planes.shape


def test_denoise_does_not_mutate_its_input():
    net = BlindSpotUNet(channels=4, base=8)
    planes = _rng().normal(0.5, 0.05, (4, 32, 32)).astype(np.float32)
    before = planes.copy()
    denoise_planes(net, planes)
    assert np.array_equal(planes, before)


def test_cross_plane_off_masks_every_plane_at_a_site():
    """The alternative to cross-plane borrowing, and the reason it exists.

    The four planes sit at different corners of the Bayer quad, so predicting
    one from the others at "the same" array index borrows from half a photosite
    away. Measured, that displaced the finished decode by 0.21 px -- inside the
    0.5 px hard limit but above the 0.1 px warning. Masking every plane at a
    site removes the borrowing entirely.
    """
    mask = make_blind_spots((2, 4, 48, 48), rate=0.05,
                            generator=torch.Generator().manual_seed(21), cross_plane=False)
    per_site = mask.sum(dim=1)
    assert set(per_site.unique().tolist()) <= {0, 4}, "a site is either fully masked or not at all"
    assert mask.any()


def test_cross_plane_on_and_off_hit_similar_pixel_counts():
    """So a comparison between the two is about the borrowing, not about one
    model seeing four times the supervision of the other."""
    kw = dict(rate=0.02, generator=torch.Generator().manual_seed(22))
    on = make_blind_spots((4, 4, 64, 64), **kw).float().mean().item()
    off = make_blind_spots((4, 4, 64, 64), rate=0.02,
                           generator=torch.Generator().manual_seed(22),
                           cross_plane=False).float().mean().item()
    assert 0.5 < off / on < 2.0, f"rates differ too much: {on:.4f} vs {off:.4f}"


# ---------------------------------------------------------------------------
# Receptive field -- the structural lever on texture
# ---------------------------------------------------------------------------

def _footprint(net, size=96):
    """Pixels of the output that a single input pixel can influence."""
    net.eval()
    with torch.no_grad():
        x = torch.zeros(1, 4, size, size)
        a = net(x)
        x[0, 0, size // 2, size // 2] = 1.0
        b = net(x)
    changed = (a - b).abs().sum(dim=(0, 1)) > 1e-7
    ys, xs = torch.nonzero(changed, as_tuple=True)
    return int(ys.max() - ys.min() + 1) if len(ys) else 0


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_unet_levels_preserve_shape(levels):
    net = BlindSpotUNet(channels=4, base=8, levels=levels)
    assert net(_batch(b=1, c=4, h=53, w=37)).shape == (1, 4, 53, 37)


def test_fewer_levels_means_a_smaller_receptive_field():
    """A blind-spot network suppresses whatever it cannot predict from its
    context, and the larger its context, the more it can average across. A
    net whose receptive field is a few scale periods can still see that scales
    are periodic; one that sees the whole fish body can replace them with the
    body's mean shade. Measured, not assumed: perturb one pixel, read the
    output footprint."""
    rf = {L: _footprint(BlindSpotUNet(channels=4, base=8, levels=L)) for L in (1, 2, 3)}
    assert rf[1] < rf[2] < rf[3], rf
    assert rf[1] <= 12, f"a one-level net should be a local filter, got {rf[1]} px"


def test_default_levels_is_three():
    assert BlindSpotUNet(channels=4, base=8).levels == 3


def test_checkpoints_from_before_the_depth_refactor_still_load():
    """The first trained models were saved with layer names enc1..dec1.
    Refactoring the class for a configurable depth must not orphan them --
    it did once, and cost a re-measurement."""
    legacy_names = {"encoders.0": "enc1", "encoders.1": "enc2", "encoders.2": "enc3",
                    "ups.0": "up2", "decoders.0": "dec2", "ups.1": "up1", "decoders.1": "dec1"}
    torch.manual_seed(3)
    original = BlindSpotUNet(channels=4, base=8, levels=3)
    legacy = {}
    for k, v in original.state_dict().items():
        head = ".".join(k.split(".")[:2]) if not k.startswith("out") else "out"
        rest = k[len(head) + 1:] if head != "out" else k[len("out") + 1:]
        legacy[f"{legacy_names.get(head, head)}.{rest}"] = v
    assert any(k.startswith("enc1.") for k in legacy)
    restored = BlindSpotUNet(channels=4, base=8, levels=3)
    restored.load_state_dict(legacy)
    x = _batch(b=1, c=4, h=32, w=32)
    original.eval(); restored.eval()
    with torch.no_grad():
        assert torch.allclose(original(x), restored(x))
