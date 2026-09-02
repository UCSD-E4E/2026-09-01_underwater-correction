# Underwater enhancement — offline evaluation harness

**Deliverable 1.** A read-only harness that scores an enhancement configuration
against the human-label corpus, so "does enhancement help?" is answered with
pipeline metrics rather than perceptual ones.

Nothing here writes to fishsense-api, Label Studio, Garage, or Postgres. The
only step that touches a database opens a read-only transaction and issues one
SELECT.

```
python -m enhancement_eval manifest --out manifest.csv --per-dive 40
python -m enhancement_eval evaluate --manifest manifest.csv \
    --experiment clahe-jpeg --raw-dir ./raw_cache \
    --slate-pdf-dir ./slate_pdfs --out results.csv
python -m enhancement_eval report --results results.csv
```

Run with the monorepo's venv, which already has rawpy, skimage, torch,
pymupdf and fishsense-core:
`~/Repos/school/e4e/fishsense/fishsense-lite/.venv/bin/python`.

---

## Primary focus: labeling after the laser stage

Scope: **species, head/tail and slate labeling**. Laser labeling and the
detectors are both out of scope; the laser's role here is to supply depth
structure, not to be improved.

"Easier for labelers" cannot be read off pixels, and "it looks better" is
exactly the evidence this project refuses. But the corpus carries a
**behavioural** record: Label Studio stamps every annotation with `lead_time`,
`was_cancelled` and `completed_by`, at 100% coverage.
`python -m enhancement_eval effort --task <species|headtail|slate>` reproduces
everything below.

The consequence that matters most: **a real A/B trial needs no new
instrumentation.** The outcome variable already records itself.

### Where the labeling time goes

| stage | usable annotations | median | total so far | **remaining** |
|---|---|---|---|---|
| head/tail | 35,902 | 15.4 s | 234.7 h | 13,920 frames ≈ **60 h** |
| species | 2,035 | 21.2 s | 16.1 h | 29,529 frames ≈ **174 h** |
| slate | 317 | 67.0 s | 7.1 h | — |

Head/tail is 91% of the post-laser labeling done *so far*; species is 74% of
what is *left*, because stage 2 now fires per laser-valid image and has barely
been populated. Both scale with every new dive.

**~233 hours of post-laser labeling still ahead.** A 20% speedup is ~47 hours.

### These tasks are far less labeler-dominated than laser

| task | residual SD | labeler variance | dive variance |
|---|---|---|---|
| head/tail | 0.752 | 12.6% | 7.8% |
| species | 0.632 | 11.5% | **14.1%** |
| slate | 0.370 | 22.1% | **18.5%** |
| *(laser, out of scope)* | *0.546* | *40.3%* | *9.5%* |

For laser, labeler identity swamps everything at 40.3% with a 6x spread between
the fastest and slowest labeler. After the laser stage it drops to 11-22%, and
for species and slate **the dive explains as much or more than the labeler**.
Dive-level variation is where water conditions live, which is the one thing
enhancement acts on. That is the most encouraging signal in this analysis --
and it is a lead, not a result: dive also confounds site, species mix, batch
and labeler assignment.

### Two things the corpus cannot tell you

**Per-image difficulty is barely a real construct.** On the laser corpus (the
only stage with enough repeat labeling to measure it), two labelers agree about
which image was slow at **r = +0.065, 95% CI [-0.006, +0.136]** over 755
doubly-labelled images. Only ~6% of labeler-adjusted variance is a stable
property of the frame. So: do not build a model predicting labeling difficulty
from pixels, and do not accept a correlation between image statistics and
labeling time as evidence. It does **not** rule out a real speedup -- a uniform
gain shifts the mean without needing image-level structure.

**There is no repeat labeling after the laser stage at all**: 0 images with two
completed species rows, 11 with two head/tail rows, 0 with two slate rows. So
inter-labeler *agreement* -- the outcome that connects to measurement quality,
since 1 px of head/tail error is 0.75% length error -- cannot be measured
observationally. It has to be designed into the trial.

**Skips are a labeler habit, not an image property.** Per-labeler skip rates run
0% to 44.6% (median 1.5%; 7 of 32 high-volume labelers never skip), and of 2,517
head/tail images annotated twice, 2,490 were skipped by one labeler and not the
other -- only 3 by everyone. Detecting a 20% relative reduction would need
~4,446 annotations per arm via skips against 179 via time. Use time.

### The trial

Randomized **within-labeler**, original vs enhanced, with **deliberate
replication** (each frame labelled by 2-3 labelers) so both speed and agreement
are measurable. Randomization balances image difficulty across arms, so the
weak per-image reliability costs power but biases nothing.

Annotations per arm, 80% power, two-sided 0.05:

| effect | head/tail | species | slate |
|---|---|---|---|
| 10% faster | 801 | 566 | 194 |
| 15% faster | 337 | 238 | 82 |
| **20% faster** | **179** | **127** | **44** |
| 30% faster | 70 | 50 | 17 |

A 20% species trial is ~90 minutes of labeling against ~174 hours of species
work remaining. **The trial is cheaper than any offline proxy for labeler
benefit**, so it is also the right way to *choose* the enhancement: run two or
three candidates from the decode experiments and let it decide.

---

## The laser is an input, not a consumer

`LaserDepth.range_m` is a metric range at a known pixel in every laser-labelled
frame — the physical anchor a Sea-thru-style enhancement fits against. That is
the depth structure worth having, and it makes the laser an **input** to this
work.

Laser *detection* must not become a downstream consumer of enhancement. That
would couple a working, calibrated path to an experimental one. The guarantee
is free as long as enhancement stays at the JPEG stage, because
`predict_laser_image` decodes the `.ORF` into a `LinearRawImage` and never
reads the JPEG — but "free unless somebody moves it" is not a guarantee, so it
is enforced:

- `assert_no_laser_dependency` refuses a linear-stage arm in any shipping
  candidate, and `tests/test_no_laser_dependency.py` asserts it over the real
  experiment table.
- Every experiment declares which consumers it can reach, so a reader can tell
  without tracing the pipeline that a CLAHE sweep cannot move laser detection.
- The `insulation-control` experiment measures it: a JPEG-chain-only change
  produced a **bit-identical** detector result — same pixel, same confidence to
  the last digit (`0.057967644184827805`), paired delta `0.00`.

Linear-stage arms still exist as **diagnostics** — the only way to measure what
the placement constraint costs is to measure the thing it forbids — and the CLI
prints a `DIAGNOSTIC ONLY — NOT A SHIPPING CANDIDATE` banner before running one.

Three facts about that detector, for the record, since they explain why the
constraint is cheap: its first three channels are `rgb / sum(rgb)`, which is
per-pixel scale-invariant (luminance-only enhancement is invisible to it); two
of six channels come from the undemosaiced Bayer mosaic, which no RGB enhancer
can produce; and it was trained on `use_camera_wb=True`, so a white-balance
change upstream is a covariate shift, not a perturbation.

---

## The consumers being scored

Both read the rectified frame, so both sit downstream of the JPEG-stage
placement:

| Consumer | Oracle | Reads |
|---|---|---|
| `FishSegmentation` + `FishHeadTailDetector`, laser-gated | human `HeadTailLabel` | the stage-5.1 **JPEG** out of Garage |
| `fishsense_core.slate.estimate_plane` | human `DiveSlateLabel.reference_points` | the **rectified array**, in-process |

That difference is why the JPEG round-trip is applied **per consumer, not per
stage**. `predict_slate_image` rectifies in memory and never touches a JPEG, so
round-tripping it would charge that stage a quantization loss it does not
actually suffer.

Human `reference_points` and `slate_rectangle` are both stored in *photo-frame*
pixels — the sync activity shifts the LS composite's PDF-panel width off the x
coordinates before persisting — so they are directly comparable to
`BoardEstimate.image_points` with no further transform.

### Slate points are paired positionally, never by nearest neighbour

Stage 13 consumes reference points by position, and the estimator's
characteristic failure is resolving the board's 4-fold corner ambiguity wrong —
which produces the same *set* of points in the wrong order. Nearest-neighbour
matching would score that catastrophe as a perfect fit.

### The question that retired the slate stage

It shipped 2026-08-02 behind `ECC >= 0.80` and was shut down the next day: pool
dives produced high-ECC (0.93–0.97) *false* fits. The gate used a confidence
score as a correctness proxy without anyone having measured whether it was one.

The harness measures it directly — `Spearman rho(ECC, median point error)`,
overall and per site, plus the error distribution among the fits the gate
*accepted*. The precedent is `LaserDepth.residual_m`, which looked like a
quality signal, measured rho = −0.026 against real error, and is now recorded
and deliberately never gated on. If ECC behaves the same way, no threshold on
it could ever have worked — a more useful conclusion than "the threshold was
wrong", because it rules out re-tuning as a fix.

---

## Three reporting rules, each a response to how that gate failed

1. **Per-image before per-corpus.** Every section leads with a distribution and
   a win/loss count; medians carry a bootstrap CI, and the report says in words
   when that interval spans zero.
2. **Split by dive and by pool-vs-field.** `Dive` has no site column, so the
   split is inferred — dives 65/71/77/80/83 are seeded from the postmortem, the
   rest are name/path guesses, and the report prints how many of each so nobody
   reads a regex as a measurement.
3. **Coverage never mixes with accuracy.** A non-detection is `None`, never a
   large error. An arm that abstains on the hardest frames and is marginally
   better on the rest would otherwise report a *better* median while being
   strictly worse. For slate, acceptance is reported separately from
   correctness for the same reason.

Stratification is by baseline RMS contrast quartile, because the claim worth
testing is that enhancement helps degraded inputs and **degrades** good ones.

No UIQM, PSNR or SSIM appears anywhere. That is deliberate.

---

## Constraint #1 is enforced mechanically, not by review

- **`guard`** wraps every call: shape, dtype, no in-place write to the shared
  input. Catches resize, crop, channel drop, non-square transpose.
- **`probe_geometry`** runs at Arm *construction*, so an arm that moves pixels
  cannot exist. It perturbs an off-centre patch and measures where the output
  responds — catching what shape checks structurally cannot.

Thresholds calibrated by measurement, not guesswork:

| Enhancer | Measured displacement |
|---|---|
| identity, halve, 5px & 31px Gaussian, CLAHE | ≤ 0.074 px |
| resample round trip (0.5× / 0.33× / 0.7×) | 0.07 / 0.16 / 0.34 px — flagged suspicious |
| **one-pixel roll** | **1.01 px** |
| square transpose | 7.07 px |
| horizontal flip | 20.99 px |

Nothing legitimate lands between 0.1 and 1.0, so the hard limit is 0.5 px. A
symmetric neighbourhood filter preserves the response centroid however wide it
is — which is why CLAHE passes and a 1 px roll does not, and 1 px of laser-dot
error is 0.75% length error.

---

## Two bugs the tests caught, both of which would have ruined a sweep

1. **Read-only arrays segfault the detector.** `guard` hands the enhancer a
   read-only view to catch in-place writes — but `IDENTITY` returns that same
   view, so every *baseline* arm passed a read-only frame downstream. OpenCV
   ignores numpy's `WRITEABLE` flag and writes through anyway; the process
   dumps core with no traceback. `guard` now returns a writeable, non-aliasing
   array.
2. **torch must import before rawpy.** Otherwise CPU inference segfaults inside
   `torch.nn.Linear.forward`, reliably, with no Python frames. Bisected: with
   `import torch` at process start both orderings succeed; without it both
   crash. `enhancement_eval/__init__.py` does the eager import.

Both would have appeared as a silent overnight core dump with no partial
results.

---

## Experiments

| Name | Kind | Reaches | What it answers |
|---|---|---|---|
| `wb-jpeg` | shipping candidate | fish, slate, labelers | `use_camera_wb=True` is a topside daylight preset applied through metres of water. |
| `clahe-jpeg` | shipping candidate | fish, slate, labelers | Clip-limit / kernel / auto-gamma sweep, incl. `noclahe`. The slate estimator's first step segments a *bright quad*, which local histogram equalization is well placed to disrupt — and nobody has measured whether CLAHE helps at all. |
| `wb-linear` | **diagnostic** | laser detection | What the placement constraint costs, by doing what it forbids. Never a proposal. |
| `insulation-control` | **diagnostic** | laser detection | Proves non-dependency. Every delta must be exactly zero. |

There is deliberately no `clahe-linear`: every arm would be bit-identical to
the baseline.

---

## What a run needs

1. **`FISHSENSE_DSN`** — a restored backup is preferable to prod; the interior
   host in `settings.toml` is a compose service name, unreachable from a dev box.
2. **`.ORF` originals from the NAS** (`--raw-dir`, `{checksum}.ORF`). Not
   substitutable with Garage: white balance cannot be redone from a frame
   already through camera WB, auto-gamma, CLAHE and 8-bit quantization, and
   Garage's `raw/` prefix is scratch deleted post-process.
   `manifest --checksums-out` writes the fetch list.
3. **Slate templates** (`--slate-pdf-dir`, `{slate_id}.pdf`) — required for the
   slate arm only. Same Garage `slate_pdf/` / NAS `DiveSlate.path` source.

Both directories are content-addressed, so files fetched by any means work and
`evaluate` needs no credentials of its own.

**Cost.** A full `.ORF` decode is ~15 s; detector inference peaks near 6 GB RSS.
Budget roughly *n_images × n_arms × 15 s*. The three phases are separate so
phase 3 re-runs for free.

## Tests

```
python -m pytest tests/ -q                        # everything
python -m pytest tests/ -q -m "not integration"   # fast only
```

The load-bearing one is `test_decode_parity.py`: the baseline arm is asserted
**byte-for-byte** equal to `RawImage` and `LinearRawImage` on a real `.ORF`.
Every reported number is a difference against that baseline, so a baseline that
merely resembles production would make every delta the sum of the effect being
studied and an uncontrolled reimplementation error.

The slate arm is validated against a scene with known ground truth: a template
warped by a known homography, recovered at ECC 0.9982 with a median point error
of 0.83 px.
