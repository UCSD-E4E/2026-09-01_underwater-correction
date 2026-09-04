# Measured results

Numbers quoted in the module docstrings, collected in one place so they can be
re-derived or challenged. Every one comes from the real corpus (a restored
2026-09-02 backup) or from real `.ORF` frames on the NAS.

## Labeler effort — the oracle for "does this help labelers?"

Label Studio stamps every annotation with `lead_time`, `was_cancelled` and
`completed_by`, at 100% coverage. `python -m enhancement_eval effort --task X`.

| task | usable | median | residual SD | labeler var | dive var | remaining |
|---|---|---|---|---|---|---|
| head/tail | 35,902 | 15.4 s | 0.752 | 12.6% | 7.8% | 13,920 frames ≈ 60 h |
| species | 2,035 | 21.2 s | 0.632 | 11.5% | 14.1% | 29,529 frames ≈ 174 h |
| slate | 317 | 67.0 s | 0.370 | 22.1% | 18.5% | — |
| laser (out of scope) | 46,632 | 12.9 s | 0.546 | 40.3% | 9.5% | — |

* **Per-image difficulty is barely a real construct.** Two labelers agree about
  which image was slow at r = +0.065, 95% CI [-0.006, +0.136], over 755
  doubly-labelled laser images. Do not model labeling difficulty from pixels.
  It does not rule out a uniform speedup, which shifts the mean without needing
  image-level structure.
* **No repeat labeling exists after the laser stage**: 0 images with two
  species rows, 11 with two head/tail rows, 0 with two slate rows. Inter-labeler
  agreement has to be designed into a trial, not mined from the corpus.
* **Skips are a labeler habit, not an image property.** Per-labeler skip rates
  run 0-44.6%; of 2,517 head/tail images annotated twice, 2,490 were skipped by
  one labeler and not the other. Detecting a 20% relative reduction needs ~4,446
  annotations per arm via skips against 179 via time.

Trial size, randomized within-labeler, 80% power at two-sided 0.05:

| effect | head/tail | species | slate |
|---|---|---|---|
| 20% faster | 179 / arm | 127 / arm | 44 / arm |

## Decode candidates

Synthetic flat-water patch, water-region luminance SD relative to input:

| variant | water noise | hue drift | object separation |
|---|---|---|---|
| baseline (CLAHE clip 0.01) | 36.7x | 1.0x | 0.259 |
| luminance-only CLAHE | 34.5x | 10.3x | 0.511 |
| CLAHE clip 0.003 | 15.3x | 1.0x | 0.397 |
| no CLAHE | 1.0x | 1.0x | 0.050 |

Laser-dot signal-to-noise on six real frames, split by environment:

| variant | pool SNR | reef SNR |
|---|---|---|
| baseline | 2.68 | 1.28 |
| L* stretch, no CLAHE | 3.62 | 1.46 |
| per-channel stretch | 3.23 | 1.23 |
| red floor at p90 | 1.39 | 2.58 |
| fixed-threshold dot boost | 3.33 | 2.19 |
| **L\* stretch + 6-sigma adaptive dot** | **4.16** | **1.74** |

Where the laser dot sits in its frame's own red distribution -- the reason no
fixed percentile works:

| dive | site | dot percentile |
|---|---|---|
| 223 | reef | 99.999% |
| 370 | reef | 99.954% |
| 145 | reef | 98.157% |
| 61 | pool | 89.7% |
| 66 | pool | **58.8%** |
| 76 | pool | 78.7% |

## Geometry probe calibration

Measured displacement under a local perturbation:

| enhancer | displacement |
|---|---|
| identity, halve, 5px & 31px Gaussian, CLAHE | <= 0.074 px |
| resample round trip (0.5x / 0.33x / 0.7x) | 0.07 / 0.16 / 0.34 px |
| one-pixel roll | 1.01 px |
| square transpose | 7.07 px |
| horizontal flip | 20.99 px |

Nothing legitimate lands between 0.1 and 1.0, hence the 0.5 px hard limit.

## Cost

Full-frame decode on a 3016x4014 raw: `rawpy.postprocess` 2.2 s, auto-gamma
0.9 s, **`equalize_adapthist` 10.2 s**, conversion 0.6 s. CLAHE is 73% of it,
which is why caching the shared postprocess across arms would save only 16% and
why `--jobs` (process parallelism) is the optimization that matters.

## Burst stacking — closed, negative

EXIF timestamps have 1-second resolution, so "same second" only bounds the gap
from above. 13,087 frames sit within 1 s of the previous.

Registration is possible but not the problem. ORB + RANSAC over three dives:

| dive | inliers | median displacement |
|---|---|---|
| 223 | 1197 | 19.7 px |
| 370 | 1342 / 1787 / 1816 | 37–75 px |
| 145 | 4–9 | 270 / 1860 px (failed) |

Where it registers, noise averages down exactly as predicted — dive 370's
open-water noise fell to 0.46x on four frames against the 1/sqrt(4) = 0.50
ideal. Earlier attempts (ECC affine, phase correlation) failed only because
they are global-intensity methods on a low-contrast repetitive scene.

It still does not help, for two reasons visible in the stacked frame:

* **The fish smears.** The homography is fit on the whole scene, which is
  dominated by static reef; a swimming fish moves independently and blurs.
* **The laser dot multiplies.** The dot is not a scene feature — it is light
  projected from the rig. Moving the rig moves the dot *across* the scene, so
  aligning the scene necessarily de-aligns the dot. Four frames give four dots
  in a line: fabricated features indistinguishable from the real one, the same
  failure class as the difference-of-Gaussians dot boost.

So stacking cleans the part of the frame nobody labels and corrupts both parts
they do. Making it work needs non-rigid alignment on the fish plus dot masking,
for a bounded payoff (sqrt(N) on texture that is only 1.6x above the grain).

## Range-based attenuation — real, not yet invertible

Regressing log channel ratio on `LaserDepth.range_m`, sampled on a linear
decode in an annulus around the laser dot (the one region whose range is
known), per dive:

| dive | site | n | range | d log(R/G)/dz | r² |
|---|---|---|---|---|---|
| 341 | field | 135 | 0.46–2.75 m | −0.352 ± 0.043 | 0.34 |
| 349 | field | 115 | 1.05–2.98 m | −0.220 ± 0.044 | 0.18 |
| 465 | field | 65 | 1.05–5.45 m | −0.095 ± 0.026 | 0.18 |
| 60 | **pool** | 103 | 0.63–2.16 m | +0.286 ± 0.108 | 0.07 |

**What holds.** All three field dives are significantly negative (t = 8.2, 5.0,
3.7): red attenuates faster than green, recovered from nothing but the laser's
metric range. The pool dive shows no coherent signal, which is correct — clear
water has almost no differential attenuation over 2 m, so its regression picks
up scene confounds instead. Where significant, log(B/G) has the opposite sign
(341: +0.149, 465: +0.056), giving the textbook ordering beta_r > beta_g > beta_b.

**What does not.** Magnitudes vary 3.7x across field dives, and binned medians
show dive 465 nearly flat (−0.027/m by medians against −0.095/m from OLS, so
that slope is leverage-driven). Dive 341 is not monotonic. Backscatter
saturation does not explain it: 465 is flat at short range too.

**Why.** Every frame is a different scene, so reflectance varies frame to frame
and does not average out at n = 65–135. That is exactly what the r² of 0.18–0.34
is reporting.

**The fix, and why it is blocked.** Fitting on the dive slate would hold
reflectance constant and remove the dominant noise term. 220 slate frames carry
a metric range — but 7 of the 8 dives contributing them are Pool Calibration,
where there is no attenuation to measure. Only dive 341 has field slate frames
(30), too few to fit. Closing this needs field slate frames at varied range,
i.e. new collection, not new analysis.

## Attenuation from the dive slate — validated

Fitting on the slate instead of an annulus around the laser dot removes the
reflectance term: the slate is one printed sheet, so across frames of a dive
the only thing changing with range is the water. With reflectance fixed the
slope of log intensity on range gives **beta itself**, per channel, not just a
difference between channels.

Seven pool dives, 218 usable slate samples, r² = 0.86–0.96:

| channel | measured (pool median) | range across 7 dives | Pope & Fry 1997 |
|---|---|---|---|
| beta_r | **+0.263** | +0.218 … +0.311 | 0.24 – 0.34 /m |
| beta_g | **+0.040** | −0.026 … +0.129 | 0.057 /m |
| beta_b | **+0.001** | −0.044 … +0.055 | 0.009 /m |

All three land on the published absorption spectrum of pure water, with the
correct beta_r > beta_g > beta_b ordering in 7 of 8 fits. **This is the only
result in the project validated against an external standard rather than
against itself.**

The single field dive with enough slate frames gives
**beta = (0.385, 0.189, 0.206)** — higher than pool in every channel. Pure
absorption dominates in clear water; turbidity adds scattering, which is far
less wavelength-selective, so green and blue rise most. That is also why its
ordering marginally inverts.

**Two corrections to earlier readings.** The first slate run reported pool
coefficients from −0.28 to +0.60 and looked like a method failure; it was
sampling a one-pixel diagonal streak, because `slate_rectangle` stores two
opposite corners and `cv2.fillPoly` accepts that silently. And "clear water
should give beta ≈ 0" was wrong physics — pure water is itself a strong red
absorber, so the correct expectation for a pool *is* the pure-water spectrum.
The geometry fix took usable samples from 131 to 218 and r² from 0.05–0.82 to
0.86–0.96.

## Inverting the model — modest at current ranges

`seathru.remove_water` inverts the formation model with the fitted beta and the
laser's range. On dive 341 frames (measured beta, measured z) and on a borrowed
estimate for the Queen Angelfish frame, the result is **real but subtle**,
because the correction scales with range and these frames are close-up:

| range | implied gain R/G/B | red relative to green |
|---|---|---|
| 0.57 m | 1.25 / 1.11 / 1.13 | 1.12× |
| 1.54 m | 1.81 / 1.34 / 1.37 | 1.35× |
| 5 m (extrapolated) | — | ~2.7× |

At 1–1.5 m most of the visible improvement comes from the tone treatment, not
the physics. The physics earns its place at longer ranges, and as a *principled*
colour correction: a measured beta cannot over-correct into magenta the way
gray-world and white-patch did, because it is not inferred from the picture it
is correcting.

**A correction to a figure shown earlier.** A first render of the Sea-thru
comparison looked dramatically better than it should have. That was a bug: the
demo tone-mapped straight from linear and skipped the auto-gamma step, so
neither panel was the pipeline its label claimed. Rebuilt faithfully, the
physics contribution is the modest one tabulated above.

**Ceiling.** `remove_water` treats range as uniform across the frame, because
the corpus has range at exactly one pixel. That is right for the fish the laser
is on and increasingly wrong for background. Scaling a monocular depth estimate
by the laser's metric range is the step that lifts it, and is not implemented.

## Cross-channel denoising — tried, does not beat isotropic

Two ideas exploiting the same asymmetry: red is the noisiest channel
(water-region high-frequency energy 9.50 against green's 6.88 on dive 370),
while luminance is 0.21R + 0.72G + 0.07B, so structure lives mostly in green.

* **red-only** — filter red hard, leave green and blue bit-identical.
* **guided** — use green as a guide to filter red and blue (He et al.).

Two reef frames, relative to no denoise:

| variant | luma water | luma fish | chroma water | chroma fish | luma ratio |
|---|---|---|---|---|---|
| red-only 0.6 | 0.94x | 0.94x | 0.99x | 1.00x | 1.01 |
| red-only 0.9 | 0.92x | 0.93x | 0.99x | 1.00x | 1.04 |
| guided 0.6 | 0.71x | 0.72x | 0.95x | 0.81x | 1.04 |
| guided 0.9 | 0.70x | 0.70x | 0.95x | 0.82x | 0.99 |
| *tv-pre (isotropic)* | *0.22x* | *0.53x* | — | — | **1.66** |

Both remove noise and fish detail in near-equal proportion (ratio ~1.0), where
plain total variation manages 1.66.

**Why guided filtering fails here specifically.** It assumes a *clean* guide.
Green is only 28% less noisy than red (6.88 vs 9.50), not the 5-10x that would
make it a reliable structure reference, so it transfers its own noise into the
channels it is supposed to be cleaning.

**Why red-only cannot help much.** Red is 21% of luminance, and linear red in
open water averages 0.0129 -- it barely carries signal to begin with, so
cleaning it changes little that a labeler sees.

### Two bugs found on the way, both about starved channels

1. `estimate_noise` quantised to uint8 for its median filter. Linear red
   occupies roughly the bottom 5% of the range, so x255 leaves ~13 levels and
   the estimate collapsed to exactly zero -- making eps zero and the whole
   filter a silent no-op. Red-only denoising appeared to "do nothing" for two
   measurement rounds because of this, not because the idea was wrong.
2. Even computed in float, **MAD is the wrong estimator for a coarsely
   quantised channel**: over half the pixels equal their own 3x3 median
   exactly, so the median absolute deviation is 0 regardless. A 75th-percentile
   fallback fixes it.

Anyone deriving a threshold from a dark channel in this corpus will hit both.

## Bayer-domain denoising — the RGB attempts were in the wrong domain

Cross-channel denoising assumes one channel is an independent, cleaner
reference for another. Demosaicing destroys that: every output pixel's R, G and
B are interpolated from overlapping photosite neighbourhoods, so their noise
becomes shared. Measured on a reef frame, correlation of high-frequency
residuals between channels:

| pair | before demosaic | after demosaic |
|---|---|---|
| R, G | +0.029 | **+0.328** |
| R, B | +0.009 | +0.145 |
| G, B | +0.101 | +0.308 |

An order of magnitude. A guide whose noise is a third shared with the channel
it guides cannot separate that channel's signal from its noise -- which is
exactly why the RGB-domain guided filter scored a ratio of 0.99, removing
signal and noise in equal measure.

**G1 - G2 is a signal-free noise measurement.** The two green photosites in a
2x2 cell sample nearly the same point, so their difference has no scene
content: sigma = 112.2 DN against a mean green level of 927.5 DN, an SNR of 8.3
per photosite (red sits at 110.9 DN, 12% of green's level). This is a
measurement, not an estimate — it needs no assumption about how much of the
frame is flat and cannot collapse on a starved or quantised channel, which is
what defeated two earlier RGB-domain estimators.

CFA-domain denoising, guided by (G1+G2)/2, two reef frames:

| variant | luma water | luma fish | chroma water | luma ratio |
|---|---|---|---|---|
| cfa 0.05 | 0.31x | 0.44x | 0.63x | 1.22 |
| **cfa 0.12** | **0.25x** | **0.38x** | **0.58x** | **1.21** |
| cfa 0.2 | 0.20x | 0.33x | 0.55x | 1.19 |
| cfa 0.4 | 0.14x | 0.23x | 0.53x | 1.12 |
| *RGB guided 0.9* | *0.70x* | *0.70x* | *0.95x* | *0.99* |
| *isotropic tv-pre* | *0.22x* | *0.53x* | — | *1.66* |

**Verdict: the diagnosis was right, the ranking did not change.** Working
before the demosaic is the correct domain and it removes chroma noise far
better than anything in RGB space (0.58x against 0.95x). On the luma
texture-vs-noise ratio it still trails isotropic total variation, though that
metric counts the fish's own grain as detail and so under-credits any denoiser.
Visually `cfa 0.12` is the best result produced: scale texture intact, grain
substantially down, no waxy look. Strengths above ~0.2 smooth structure weaker
than 2 sigma, and the texture itself is only about 1.2 sigma, so they erase it.

The underlying limit is unchanged: fish texture is ~1.6x the grain in the same
band, and no filter in any domain separates them cleanly.

## Demosaic choice — every algorithm sits on the same 1:1 line

Production uses rawpy's default (AHD). Holding everything downstream fixed and
varying only the interpolation, on two reef frames:

| demosaic | luma water | luma fish | chroma water | s/frame |
|---|---|---|---|---|
| AHD (production) | 1.00x | 1.00x | 1.00x | 1.6 |
| **DHT** | 1.18x | **1.16x** | 0.99x | 2.3 |
| AAHD | 1.14x | 1.12x | 0.98x | 7.8 |
| DCB | 0.99x | 1.00x | 0.94x | 6.0 |
| PPG | 0.95x | 0.95x | 1.01x | 1.4 |
| VNG | 1.00x | 1.00x | 0.97x | 3.3 |
| linear | 0.75x | 0.70x | 1.02x | 1.5 |
| AHD + FBDD light | 0.84x | 0.84x | 0.95x | 3.3 |
| AHD + FBDD full | 0.86x | 0.86x | 0.93x | 3.8 |

Sharper algorithms recover more detail and proportionally more noise; smoother
ones give up both. Every ratio is near 1.0. Demosaic choice moves *where* you
sit on the noise-detail line, not the exchange rate — which is the same wall
every denoiser hit, for the same reason: the texture and the grain occupy the
same band.

LMMSE and AMaZE, the two algorithms most often recommended for noisy raws, are
not compiled into this LibRaw build (they need the GPL demosaic pack).

**DHT is a legitimate option** if labelers prefer sharper and grainier: +16%
fish detail for +18% noise, one parameter, 2.3 s/frame.

## An SNR-aware demosaic — tried, worse on every axis

Stock demosaics treat all three channels as equally worth interpolating.
Underwater they are not: green sits at 927.5 DN with SNR 8.3 per photosite,
red at 110.9 DN with roughly a third of that. So interpolate the colour
difference and let red inherit sharp structure from green:

    R_full = G_full + smooth(R - G)

with a physical justification for smoothing hard — the colour difference is set
by the water column, which the attenuation fit showed varies smoothly with
range. Prediction stated in advance: chroma noise down hard, luma fish detail
held near 1.00x.

| variant | luma water | luma fish | chroma water | chroma fish |
|---|---|---|---|---|
| AHD | 1.00x | 1.00x | 1.00x | 1.00x |
| uw r=3 | 1.56x | 1.52x | 1.03x | 1.22x |
| uw r=6 | 1.52x | 1.48x | 1.04x | 1.35x |
| uw r=12 | 1.49x | 1.43x | 1.05x | 1.48x |

**Wrong on every axis named.** Chroma noise unchanged in water and *worse* on
the fish, luma noise up ~50%.

At least part of the cause is an implementation flaw worth recording, because
it will bite anyone writing a demosaic: preserving the measured green samples
exactly while interpolating the gaps leaves a quincunx where half the pixels
carry raw noise and half carry 4-neighbour averages. On a perfectly flat
synthetic scene that alone produces **2.21x** the high-frequency energy of a
uniformly interpolated green — a checkerboard of noise *variance* that any
high-frequency metric reads as detail or noise.

Removing the checkerboard by interpolating green uniformly also removes green's
real detail, which is the channel worth keeping. Resolving that tension is what
directional interpolation in AHD and DHT exists to do, and matching them is a
serious engineering project rather than a parameter choice.

**Recommendation: do not write a bespoke demosaic.** The available gains lie
along a 1:1 line and are small; use DHT if sharper is wanted. The concept
(allocating bandwidth by per-channel SNR) is sound and is what ISPs already do
— the implementation quality is the barrier, not the idea.

## CNDR — reimplemented from the paper, negative

Yan, Wang, Claramunt & Yang (2026), "Underwater image enhancement via
color-noise decoupling and reconstruction with application to tidal stream
turbine", *J. Ocean Eng. Mar. Energy* 12:1151–1163,
doi:10.1007/s40722-026-00480-7. No code released, so `enhancement_eval/cndr.py`
is built from the equations; 35 tests, geometry probe 0.000 px.

Worth trying because its stated problem is the one measured here. Its eq 3 —
enhancement applies a per-channel gain, so `y_hat = k_c·x + k_c·n`, worst in
red because red needs the most gain — is the CLAHE noise amplification that
made us turn CLAHE off. It is also non-learned, so there is no training
distribution to be out of on pool dives.

Nine frames, six reef and three pool. Noise in open water, detail on the fish
between head and tail keypoints, as everywhere else in this file.

| variant (reef, 6 frames)     | water grain | σ̄ eq 17 | fish grain | separation |
|------------------------------|------------:|--------:|-----------:|-----------:|
| production (auto-gamma+CLAHE) |       7.035 |  14.479 |     10.929 |      1.560 |
| **recommended (ships)**       |   **7.118** |**12.417**| **10.608**|  **1.507** |
| CNDR full, replacing stretch  |       6.762 |  12.509 |      9.952 |      1.485 |
| CNDR DAC only, replacing WB   |       4.293 |   8.600 |      6.048 |      1.418 |
| CNDR contrast only, on top    |      11.060 |  18.982 |     17.020 |      1.573 |

**The complete published method is worse than what we ship** — separation 1.485
against 1.507, and it wins on only 4 of 6 frames with a mean delta of −0.022.

**DAC is the weakest stage, not the strongest.** This was the piece predicted to
beat the hand-tuned `red_boost`, on the reasoning that compensating from green
and blue (eqs 8–10) avoids multiplying red noise by a large gain. It does avoid
that — water grain drops to 4.293 — but it takes 43% of the fish detail with it
(6.048 against 10.608) and lands at the worst separation of any arm. On pool
frames it drives red to a mean of 114 against our 69: red overcompensation,
which is exactly what the stage exists to prevent, appearing out of
distribution. Pool water is nearly clear, so the cast DAC assumes is not there
to correct.

**FCR is not a denoiser**, despite sec 3.5's framing. Its per-coefficient gain
is `1 + β·α`, which is ≥ 1 everywhere, so no coefficient ever shrinks. Eq 14
suppresses noise only *relative* to features. Measured, open-water grain rose
**×1.53**. The contrast stage does buy the best separation in the table
(1.573, winning 5 of 6 frames) — but at ×1.50 the water noise, which is the
same separation-for-noise trade CLAHE offers and that we already declined.

**RCR is a scalar multiply.** Eq 12 is `U(μΣ)Vᵀ`, and since μ is a scalar from
eq 13, that is `μ·L` — a whole subsection of SVD machinery that cancels. It is
implemented as the scalar (proved exact in `test_rcr_is_algebraically_a_scalar_gain`),
which also avoids an SVD of a 12-megapixel matrix per frame. Its apparent
separation gain (+0.051 with water noise ×0.93) comes from a global brightening
interacting with clipping and 8-bit quantization, not from any structural
improvement, and should not be read as a win.

**Recommendation: do not ship CNDR.** Keep the module: the undecimated Haar
transform is reusable, eq 17 gives a noise figure directly comparable with the
rest of this file, and the per-stage ablations are the evidence for this
negative. The one idea worth carrying forward is FCR's *relative* weighting —
it is the only arm that improved separation on 6 of 6 reef frames — but it
needs a formulation whose gain can go below 1, which eq 16 forbids by
construction.

Cost: 22 s/frame for the full method against 6 s for the shipped decode.

## Noise2Void trained on our own corpus — works, with a real cost

`enhancement_eval/n2v.py`, 18 tests. Trained on 144 crops from 72 dives, 8000
steps in 191 s on the RTX 3060. Evaluation frames held out by image id. Both
arms share one finishing path, so the only difference is whether the mosaic was
denoised. Gated first by `noise_structure` — see below.

The one learned method that does not inherit the out-of-distribution problem
that killed the slate detector: no clean targets and no external dataset, so
pool dives are in-distribution because they are in the training set. And it is
a denoiser rather than a generator — it predicts a pixel from its neighbourhood
and structurally cannot invent content.

| reef, 6 held-out frames | shipped | + N2V | change |
|---|---:|---:|---|
| open-water grain | 7.12 | 1.40 | **÷5.2** |
| fish detail | 10.61 | 2.29 | **÷4.6** |
| separation | 1.507 | 1.714 | +0.21 (see below) |
| displacement | — | 0.083 px mean, 0.205 px worst | |

**The noise reduction is real and large.** On degraded frames it is not subtle:
dive 156 goes from near-pure grain to readable rock structure. Open water shows
no blotching and no tile seams. The laser dot survives and reads *more* clearly
against the cleaner background.

**The separation figure should not be quoted.** It rises 1.507 → 1.714 only
because both of its terms collapsed together — water grain ÷5.2 and fish detail
÷4.6. On this result the ratio is close to meaningless and the absolute numbers
are what to read. This is the confound that made the earlier whole-frame
"detail" metric useless, reappearing in a ratio instead of a raw number.

**The cost is fine texture.** The reticulated scale pattern on the dive 223
angelfish is gone. Head/tail outlining looks easier — the body outline, fins,
snout and eye are all cleaner. Species identification may be harder. No metric
here can adjudicate that; the labeling trial can, which is what it is for.

**Displacement is real but small.** 0.083 px mean, 0.205 px worst, verified
against controls: a symmetric Gaussian blur reads 0.003 px and a known 1 px
roll reads 1.000 px, so phase correlation is good to about 0.01 px here. Inside
the 0.5 px hard limit, above the 0.1 px warning on 2 of 9 frames.

Two hypotheses about the cause of that shift, both wrong, both instructive:

| configuration | mean shift | worst | grain |
|---|---:|---:|---:|
| cross-plane + 5-wide mask (as measured) | **0.083** | 0.205 | 5.2× |
| point mask instead | 0.190 | 0.269 | 5.6× |
| no cross-plane borrowing | 0.891 | 1.622 | 6.9× |

The blind spot masks one plane at a time so the network can predict it from the
other three, which is valid because the four photosites' noise is nearly
independent (0.029 raw). The suspicion was that borrowing across planes borrows
from half a photosite away, since R/G1/G2/B sit at different corners of the
Bayer quad. Removing the borrowing made displacement 4–8× *worse* and broke the
0.5 px limit on every reef frame — cross-plane information anchors the
prediction rather than displacing it. Replacing the measured 5-wide horizontal
mask with a point mask also made it worse. Both measurement-driven design
choices were right, and the residual shift looks intrinsic to heavy denoising
rather than attributable to either knob.

**Status: diagnostic, not shippable as-is.** It operates on the mosaic, so it
sits before the JPEG stage — the same status as the earlier CFA work under
constraint #2. **Recommendation: put it in the labeling trial as a third arm**
rather than shipping or discarding it. The detail loss is a genuine risk to
species ID and a genuine gain for head/tail, and only labelers can settle which
dominates.

## Noise2Void viability gate — the assumptions hold, and are directional

`enhancement_eval/noise_structure.py`, 18 tests. Run before training, because
N2V fails *silently* when its assumptions break: the loss curve looks healthy
and the output looks smoother, since the independent part of the noise still
goes. Measured from G1−G2, which is signal-free, sampled in open water.

| | measured | limit |
|---|---:|---:|
| max off-diagonal autocorrelation | 0.044 | 0.10 |
| fixed-pattern fraction | 0.002 | 0.30 |
| σ | 134.71 DN | |

Both assumptions hold. The control behaves as designed: whole-frame
autocorrelation reads 0.109 against 0.044 in open water, confirming that scene
detail leaks into G1−G2 and that the open-water crop is doing real work.

**The noise is directional**, which the gate turned into a design decision
rather than a footnote. Horizontal lag-1 correlation runs 7.5× the vertical —
up to 0.162 on bright pool frames against ~0.01 down columns. It is not
row-readout noise: that would stay flat along the row, and this decays
geometrically (0.162, 0.088, 0.050, 0.031, 0.016, 0.008), which is the analog
readout chain's horizontal bandwidth limit. So the blind spot is a 5-wide
horizontal line, from `recommended_mask_width`, not the paper's default point.
