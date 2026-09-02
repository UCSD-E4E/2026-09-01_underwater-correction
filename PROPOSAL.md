# Proposed production change — not applied

The recommended decode, as a concrete change. Written down rather than made,
because the decisive evidence is a labeling trial that has not run, and
production is the only deployed instance.

## What would change

Today every labeler-facing JPEG is produced by

```python
RectifiedImage(RawImage(raw_bytes), intrinsics)
```

and `RawImage._get_data()` — inside the **fishsense-core wheel**, not this
repo — does:

```
rawpy.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
                  output_bps=16, user_flip=0)
  -> auto-gamma from the frame's own mean V
  -> equalize_adapthist()            # <- replaced
```

The proposal replaces that last step with a global percentile stretch of
CIELAB L*, leaving white balance and the auto-gamma untouched:

```
  -> auto-gamma from the frame's own mean V
  -> stretch L* from its 1st to its 99th percentile
```

## Two ways to land it, and the tradeoff

**A. Change fishsense-core.** One edit, but it is a blunt instrument: five call
sites decode through `RawImage`, so it would also change
`predict_slate_image` (retired, but still registered) and every preprocess
stage at once. It also needs a release and a version bump, and fishsense-core
is a separate repository.

**B. Decode locally in the four preprocess activities.** More code, but
surgical: it touches only the JPEG-producing stages
(`preprocess_laser_image`, `preprocess_species_image`,
`preprocess_headtail_image`, `preprocess_slate_image`) and leaves the detector
paths alone. It also keeps the change inside this monorepo, where it is tested.

B is the safer shape for a first trial. A is the right home once the change is
settled, because the decode belongs in the library.

## What does not change, deliberately

* **White balance.** Every variant that gave red its own independent gain
  failed on field frames — gray-world, white-patch, per-channel stretch, and
  production's own CLAHE, which is per-channel by accident.
* **The laser path.** `predict_laser_image` decodes the `.ORF` into a
  `LinearRawImage` and never reads the JPEG, so laser detection takes no
  dependency on any of this. Measured bit-identical.
* **Geometry.** All four arms probe at 0.000 px displacement.
* **The cyan cast.** Left alone; removing it needs the range-based physics
  anchored on `LaserDepth.range_m`, not another global gain.

## Before it ships

1. Run the trial: `--experiment recommended`, randomized within-labeler,
   ~127 annotations per arm for a 20% effect on species work.
2. Re-run the parity test. `tests/test_decode_parity.py` asserts the baseline
   arm is byte-identical to `RawImage`; if the library changes, that test's
   meaning changes with it and the baseline must be re-pinned.
3. Decide whether JPEGs are regenerated for the existing corpus or only for
   new dives. Mixed vintages inside one Label Studio project would confound
   any later comparison of labeling times.
