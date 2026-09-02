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
