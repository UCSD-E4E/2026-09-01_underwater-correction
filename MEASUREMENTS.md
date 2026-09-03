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
