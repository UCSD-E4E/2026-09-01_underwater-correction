"""Reproduce every frame set the measurements used, from the database.

Selection is seeded and deterministic, so the same restore yields the same
frames. CSVs (with label geometry) go to data/ and are gitignored; the image
ids alone go to data/frame_sets.json and are committed, so a future session
can check it reproduced the same sets.

  heldout.csv    9 frames scored in every comparison: dives 223 and 370 (the
                 demosaic review frames) + 4 more reef + 3 pool.
  n2v_random.csv 75 frames for the random-crop N2V model (70 reef + 5 pool).
  n2v_fish.csv   120 head/tail frames for the fish-centred N2V models.
  bigfish.csv    held-out candidates with head-tail >= 400 px, for the
                 scale-peak scan.
"""
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import enhancement_eval  # noqa: F401
from enhancement_eval.manifest import MANIFEST_FIELDS, fetch_manifest, sample_per_dive
from scripts._common import DATA, POOL

dsn = os.environ.get("FISHSENSE_DSN") or sys.exit("set FISHSENSE_DSN")
DATA.mkdir(exist_ok=True)
allrows = fetch_manifest(dsn)
# dive_id stays an int throughout: sample_per_dive sorts dives, and a string
# sort ("1", "10", "100", ...) reorders them, which silently changed every
# seeded selection below the first time this ran. Comparisons use str() at the
# point of use; the CSV writer renders ints as text anyway.
for r in allrows:
    r["image_id"] = str(r["image_id"])


def dive(r):
    return str(r["dive_id"])


def write(name, rows):
    with open(DATA / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  {name:<16} {len(rows):>4} frames, {len({dive(r) for r in rows})} dives")


ht = [r for r in allrows if r.get("head_x") is not None]
reef1 = sample_per_dive(ht, per_dive=1)[:400]                    # manifest --per-dive 1 --require-headtail --limit 400
pool = sample_per_dive([r for r in allrows if dive(r) in POOL], per_dive=2)

keep = [r for r in reef1 if dive(r) in ("223", "370")]
others = [r for r in reef1 if dive(r) not in ("223", "370")]
keep += others[::max(1, len(others) // 4)][:4]
keep += pool[::max(1, len(pool) // 3)][:3]
heldout = keep
held_ids = {r["image_id"] for r in heldout}

all1 = sample_per_dive(allrows, per_dive=1)[:900]                 # manifest --per-dive 1 --limit 900
p = [r for r in all1 if dive(r) in POOL]; o = [r for r in all1 if dive(r) not in POOL]
random.Random(3).shuffle(o); sel = o[:70] + p; random.Random(4).shuffle(sel)
n2v_random = [r for r in sel if r["image_id"] not in held_ids][:75]

n2v_fish = [r for r in reef1 if r["image_id"] not in held_ids and r.get("head_x") is not None][:120]

used = held_ids | {r["image_id"] for r in n2v_random} | {r["image_id"] for r in n2v_fish}
ht6 = sample_per_dive(ht, per_dive=6)                            # manifest --require-headtail --per-dive 6
big = []
for r in ht6:
    if r["image_id"] in used: continue
    d = math.hypot(float(r["head_x"]) - float(r["tail_x"]), float(r["head_y"]) - float(r["tail_y"]))
    if d >= 400: big.append((d, r))
big.sort(key=lambda t: -t[0]); bigfish = [r for _, r in big]

for name, rows in (("heldout.csv", heldout), ("n2v_random.csv", n2v_random),
                   ("n2v_fish.csv", n2v_fish), ("bigfish.csv", bigfish)):
    write(name, rows)
json.dump({n: [r["image_id"] for r in rows] for n, rows in
           (("heldout", heldout), ("n2v_random", n2v_random), ("n2v_fish", n2v_fish), ("bigfish", bigfish))},
          open(DATA / "frame_sets.json", "w"), indent=0)
print("held-out:", [(dive(r), r["image_id"]) for r in heldout])
