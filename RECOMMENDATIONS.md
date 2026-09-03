# Recommendations

Ordered by how well the evidence supports them. Nothing here is deployed.

## 1. Replace CLAHE with a global L* stretch — ready for trial

`DecodeConfig(stretch_mode="luminance", clahe_enabled=False)`, i.e.
`recommended` in `enhancement_eval.recommended`.

**Why.** `skimage.exposure.equalize_adapthist` has no RGB branch: it treats an
HxWx3 frame as a 3-D volume with a channel-axis tile depth of one, so each
colour channel is equalised independently. Near-uniform open water has a narrow
local histogram, so what reaches full scale is its noise — and red, the most
attenuated channel, receives the largest gain. This is per-channel behaviour by
accident, not by design.

**Evidence.** Water-region noise amplification drops 36.7x → 15.3x at clip
0.003 and to 1.0x with CLAHE off, while object-vs-water separation *improves*
(0.259 → 0.397): a permissive clip stretches background noise up until it
competes with the fish. Visual screen across six frames spanning pool and reef.

**Not yet evidence.** No labeler has been timed on it. Six frames judged by eye
is a screen, not a result.

**Confidence: moderate.** The mechanism is understood and the failure mode of
the alternatives is understood. The benefit to labelers is untested.

## 2. Run the trial — built, waiting

254 frames, 127 per arm, 67 dives with both arms present, drawn from the
species backlog so the labeling is work that has to happen anyway. ~90 minutes
of labeling detects a 20% effect. See `TRIAL.md`.

This is the only thing that can convert recommendation 1 from plausible to
established, and it is cheap enough that it is also the right way to *choose*
between candidates rather than something to run after deciding.

## 3. Optional: lift the laser dot — `recommended-dot`

Adds a red boost where the local red *excess* exceeds 6σ of its own noise.
Laser-dot signal-to-noise improves in both environments (pool 2.68 → 4.16, reef
1.28 → 1.74), which no fixed threshold managed: the dot sits at the 99.999th red
percentile on a reef and the **58.8th** in a pool.

A labeling aid rather than a restoration, so it deserves its own decision. It
helps a labeler see which fish the measurement is keyed to. It would be
circular if pointed at the laser-labeling task itself.

**Confidence: moderate**, same caveat as 1.

## 4. Collect field slate frames at varied range — highest long-term value

The attenuation fit reproduces the published pure-water absorption spectrum
across seven independent dives (r² 0.86–0.96). It is the only result here
validated against an external standard. What it lacks is field data: seven of
the eight dives with slate frames at metric range are Pool Calibration, and
only one field dive has enough.

**Ask:** shoot the dive slate at roughly 1, 2, 3 and 4 m on field dives.
30–50 frames per dive is plenty — the fit is driven by range *lever*, not frame
count, and the pool dives reached r² > 0.9 on ~2 m of spread with 20–33 frames.
Avoid direct glare; 40% of existing frames were rejected as saturated.

**Confidence: high** that the method works; the gap is data, not method.

## 5. Do not

* **Denoise by default.** Fish scale texture measures only 1.6x the grain in
  the same frequency band, so any filter strong enough to remove the grain takes
  most of the texture. `gentle-denoise` exists if a labeler wants it; it should
  not be the default.
* **Use a GAN for denoising.** The objection is the loss, not learned methods:
  an adversarial loss optimises for output a discriminator cannot distinguish
  from real, which means inventing plausible detail. Head/tail keypoints are
  clicked on edges (1 px = 0.75% length error) and species ID turns on fin and
  pattern detail. Invented and recovered detail look identical in the output, so
  the failure is unfalsifiable by inspection. If a learned denoiser is wanted,
  a self-supervised regression (Noise2Void / Noise2Self) blurs where uncertain
  instead of inventing, and needs no clean targets — none exist here.
* **Stack burst frames.** Registration works and noise averages down as
  predicted, but the fish smears and the laser dot multiplies into N false dots,
  because the dot is projected from the rig rather than fixed to the scene.
* **Touch white balance.** Every variant that gave red its own independent gain
  from the bottom of its range failed on field frames: gray-world, white-patch,
  per-channel stretch, and production's accidental per-channel CLAHE.

## 5b. Do not rebuild LibRaw for a better demosaic

Asked directly, the answer is no. DHT is the largest difference available among
stock algorithms (+16% fish detail, +18% noise) and is barely visible on one
reef frame and invisible on another. LMMSE and AMaZE — the two usually
recommended for noisy raws, and the only reason to rebuild — carry published
advantages of fractions of a dB over AHD. You would be rebuilding a toolchain
to buy something smaller than the difference you already cannot see.

The costs are not trivial either: those algorithms live in LibRaw's GPL2/GPL3
demosaic packs, which is a licensing question for `fishsense-core` rather than
a build flag; the data-worker would need a custom pinned rawpy wheel; and every
existing JPEG was rendered with stock AHD, so re-rendering mixes vintages inside
Label Studio projects and confounds the labeling-time comparison.

**Revisit only if** the trial shows labelers are measurably sensitive to the
sharpness/noise axis. The cheap way to learn that is to add DHT as a third trial
arm: 254 more frames of rendering, zero build work.

## 6. Open, in priority order

1. **Per-pixel depth.** `seathru.remove_water` assumes uniform range because
   the corpus has range at one pixel. Scaling a monocular depth estimate by the
   laser's metric range is the step that lifts it, and is what the original
   brief proposed.
2. **The detector metrics were never run.** The harness scores segmentation and
   slate detection against human labels; those numbers do not exist because
   detectors were deprioritised.
3. **Dive-level quality vs labeling time.** The correlation job died with an
   earlier crash and was not re-run.
