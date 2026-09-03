# The labeling trial

What settles whether the recommended decode helps labelers. Everything below is
local files; loading the trial into Label Studio is a deliberate human step.

    python -m enhancement_eval trial \
        --raw-root ~/mnt/fishsense_data/REEF/data \
        --out trial --per-dive 4

## Design

**Between-frames, randomized, blocked by dive.** Each frame is rendered in
exactly one arm, so a labeler never sees the same frame twice and there is no
learning or memory confound.

* **Frames come from the species backlog** — 27,139 images that still need
  labeling. The trial is therefore free: the work happens regardless, and the
  only change is which decode a frame was rendered through. Species is also the
  largest remaining burden (~174 h) and the task most plausibly limited by
  image quality, since it is recognition rather than localization.
* **Blocked by dive**, four frames per dive, two per arm. Dive explains 14% of
  species labeling-time variance. One frame per dive would put every dive in a
  single arm, which disables the blocking *and* makes the stratified question
  ("did it help more on the murkier dives?") unanswerable. At `--per-dive 4`,
  66 of 67 dives carry both arms.
* **No between-labeler comparison.** Labeler identity is 11–22% of variance
  after the laser stage, with a 6× spread between the fastest and slowest.
  Randomizing frames means every labeler contributes to both arms, so that
  variance cancels rather than confounding.
* **Both arms write the same filename.** The arm lives only in
  `allocation.csv`, so nothing downstream — and no labeler glancing at a URL —
  can read the condition off the file.

## Size

127 frames per arm detects a 20% reduction in labeling time at 80% power,
two-sided 0.05, given the measured residual SD of 0.632 for species work. At
the observed median of 21.2 s that is roughly ninety minutes of labeling in
total. For a 15% effect the requirement is 238 per arm; for 10%, 566.

## Output

    trial/
      allocation.csv    image_id, dive_id, dive_name, checksum, arm, decode, jpeg
      jpeg/<checksum>.JPG

## Reading the result

Label Studio stamps `lead_time` on every annotation, so no new instrumentation
is needed. Afterwards, extract with the same query `enhancement_eval effort`
uses, join to `allocation.csv` on `image_id`, and:

1. Drop cancelled annotations and anything over 300 s (abandoned tabs — 1.2% of
   the corpus, with a maximum of ten days).
2. Take `log(lead_time)`; labeling time is multiplicative and right-skewed, and
   differences in log space are the ratios worth quoting.
3. Subtract each labeler's own mean, then compare arms. Report the median
   difference with a bootstrap interval, the win/loss counts, and the same
   split by dive and by pool-vs-field that every other report here carries.

`enhancement_eval.metrics` has the pieces: `paired_delta`, `bootstrap_median_ci`,
`quantiles`, `stratify_by_quartile`.

## What would make the result invalid

* **Mixing JPEG vintages inside one project.** If existing frames were
  regenerated through a different decode while the trial ran, arm and vintage
  would be confounded. Load the trial into a fresh project, or into one whose
  other frames are untouched.
* **Telling labelers which arm is which.** The allocation file should not be
  visible to them.
* **Reading the result per image.** Per-image difficulty is barely a real
  construct (r = +0.065 between labelers on the same frame), so this trial can
  say the mean moved; it cannot say which images got easier.
