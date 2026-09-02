"""The configurations this work converged on, named and pinned.

Nothing here is deployed. The decisive evidence for a labeler-facing change is
the labeling trial (~127 annotations per arm for a 20% effect on species work,
about ninety minutes of labeling); these are the arms it would run.

How they were arrived at, briefly, because the negative results are the useful
part:

* **CLAHE is the problem, not the tuning.** `skimage.exposure.equalize_adapthist`
  has no RGB branch: it treats an HxWx3 frame as a 3-D volume and defaults
  `kernel_size` to `(H//8, W//8, max(3//8, 1))`. That trailing 1 is a
  channel-axis tile depth of one, so each colour channel is equalised
  independently. Near-uniform open water has a narrow local histogram, so what
  reaches full scale is its noise -- and red, the most attenuated channel,
  receives the largest gain.
* **Anything that gives red an independent gain from the bottom of its range
  fails on field frames**: gray-world, white-patch, per-channel stretch, and
  production's accidental per-channel CLAHE. The recommendation therefore does
  not touch white balance at all.
* **A global stretch is the right shape of tool.** Backscatter is an additive
  veiling floor, so subtracting a black point and rescaling matches the
  degradation; and one affine map over the frame cannot manufacture contrast
  out of a flat region the way a per-tile histogram does.
* **The dot threshold has to be scene-adaptive.** The laser dot sits at the
  99.999th percentile of the red channel on a reef and the 58.8th in a pool,
  where a white rash guard and skin are redder. No fixed level or percentile
  spans that; a threshold in sigmas of the response's own noise does.
"""

from __future__ import annotations

from enhancement_eval.decode import DecodeConfig

__all__ = [
    "PRODUCTION",
    "RECOMMENDED",
    "RECOMMENDED_WITH_DOT",
    "GENTLE_DENOISE",
    "describe",
]

#: The current decode, unchanged. The control arm for every comparison.
PRODUCTION = DecodeConfig()

#: The scene recommendation: recover contrast, touch nothing else.
#:
#: A global 1-99 percentile stretch of CIELAB lightness, with CLAHE off. Colour
#: is deliberately untouched -- the cyan cast remains, and fixing it needs the
#: range-based physics rather than another global gain.
RECOMMENDED = DecodeConfig(stretch_mode="luminance", clahe_enabled=False)

#: The same, plus a lift on red where the local red *excess* exceeds 6 sigma of
#: its own noise. Improves laser-dot signal-to-noise in both environments
#: (pool 2.68 -> 4.16, reef 1.28 -> 1.74), which no fixed threshold managed.
#:
#: Separate from `RECOMMENDED` because it is a labeling aid rather than a
#: restoration, and deserves its own decision: it helps a labeler see which
#: fish the measurement is keyed to, and it would be circular if it were ever
#: pointed at the laser-labeling task itself.
RECOMMENDED_WITH_DOT = DecodeConfig(
    stretch_mode="luminance", clahe_enabled=False,
    red_boost=0.45, red_boost_sigmas=6.0,
)

#: Opt-in grain reduction. Offered, not recommended: it removes 56% of the
#: open-water noise for 35% of the fish-detail measure, and the fish's scale
#: texture is only ~1.6x the grain in the same band, so the loss is real and
#: visible. Whether that trade is worth making is a judgement about what
#: labelers need to see, not something these numbers settle.
GENTLE_DENOISE = DecodeConfig(
    stretch_mode="luminance", clahe_enabled=False, denoise="tv-post",
)


def describe() -> str:
    """One-screen summary of what each arm changes, and what it deliberately does not."""
    return f"""\
Recommended configurations (none deployed)

  production           {PRODUCTION.label}
      camera white balance, auto-gamma to mean V 20, CLAHE at clip 0.01.

  recommended          {RECOMMENDED.label}
      + global 1-99 percentile stretch of CIELAB L*
      - CLAHE off
      Does NOT change white balance. Every variant that gave red its own
      independent gain failed on field frames, production's per-channel CLAHE
      included. The cyan cast is left alone on purpose; removing it needs the
      range-based physics, not another global gain.

  recommended+dot      {RECOMMENDED_WITH_DOT.label}
      + red lifted where local red excess exceeds 6 sigma of its own noise
      Laser-dot SNR: pool 2.68 -> 4.16, reef 1.28 -> 1.74. A labeling aid, so
      it is a separate decision from the scene treatment.

  gentle-denoise       {GENTLE_DENOISE.label}
      + total-variation denoise after the stretch
      Opt-in. 56% of the open-water grain for 35% of the fish-detail measure.
      Fish scale texture is only ~1.6x the grain in the same frequency band,
      so this trade cannot be avoided by a better filter.

What settles it: a randomized within-labeler trial. Label Studio already
records `lead_time` on every annotation, so no new instrumentation is needed --
about 127 annotations per arm for a 20% effect on species work, roughly ninety
minutes of labeling. Six frames looked at by eye is a screen, not a result.
"""
