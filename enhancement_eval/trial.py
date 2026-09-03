"""Building the labeling trial: selection, allocation, and the analysis plan.

The design is **between-frames, randomized, blocked by dive**. Each frame is
rendered in exactly one arm, so a labeler never sees the same frame twice and
there is no learning or memory confound. Frame difficulty becomes a noise
source, which randomization balances -- and which the power calculation already
accounts for, because the residual SD it uses was measured across frames.

Why blocked by dive: dive explains 8-14% of labeling-time variance depending on
the task, and an unblocked draw can hand one arm the murkier dives. The effect
would then be unattributable, which is the same failure the slate detector's
acceptance gate had in a different costume -- a number that looked fine because
nothing was stratified.

Why between-labeler comparison is avoided entirely: labeler identity explains
40.3% of variance on laser work and 11-22% after it, with a 6x spread between
the fastest and slowest labeler. Randomizing frames means every labeler
contributes to both arms, so that variance cancels instead of confounding.

Draw the frames from the **backlog** -- images that still need labeling anyway.
The trial is then free: the work has to happen regardless, and the only change
is which decode a given frame was rendered through.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

__all__ = ["assign_arms", "summarize_allocation", "BACKLOG_SQL"]

#: Frames that still need species labeling, which is the largest remaining
#: burden (29,529 frames, ~174 h at the observed median of 21.2 s). Restricted
#: to images whose dive is HIGH priority and which already carry a valid laser
#: label, because that is the cohort stage 2 actually draws from.
BACKLOG_SQL = """
SELECT i.id                       AS image_id,
       i.dive_id                  AS dive_id,
       COALESCE(d.name, '')       AS dive_name,
       COALESCE(d.path, '')       AS dive_path,
       i.path                     AS image_path,
       i.checksum                 AS checksum,
       ci.camera_matrix           AS camera_matrix,
       ci.distortion_coefficients AS distortion_coefficients
FROM image i
JOIN dive d ON d.id = i.dive_id
LEFT JOIN camera c ON c.id = COALESCE(i.camera_id, d.camera_id)
LEFT JOIN cameraintrinsics ci ON ci.camera_id = c.id
WHERE i.is_canonical
  AND ci.camera_matrix IS NOT NULL
  -- EXISTS rather than a join: 461 prod images carry two valid laser labels,
  -- and joining would emit the image twice. A GROUP BY cannot dedupe it either,
  -- because the intrinsics columns are `json`, which has no equality operator.
  AND EXISTS (
        SELECT 1 FROM laserlabel l
        WHERE l.image_id = i.id AND l.completed
          AND NOT COALESCE(l.superseded, false)
          AND l.x IS NOT NULL AND l.y IS NOT NULL
  )
  AND NOT EXISTS (
        SELECT 1 FROM specieslabel s
        WHERE s.image_id = i.id AND NOT COALESCE(s.superseded, false)
  )
ORDER BY i.dive_id, i.id
"""


def assign_arms(
    frames: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    seed: int,
    dive_key: str = "dive_id",
) -> list[dict[str, Any]]:
    """Allocate frames to arms, balanced within each dive and reproducible.

    Within a dive the arms are dealt from a shuffled repeating sequence, so
    each dive's own frames split as evenly as its count allows. The leftover
    from a dive that does not divide evenly is carried into the next dive
    rather than being dropped or always favouring the first arm -- otherwise a
    cohort with a long tail of one-frame dives silently loads arm A.

    Deterministic for a given seed: an allocation that cannot be regenerated
    cannot be audited after the fact.
    """
    if len(arms) < 2:
        raise ValueError(f"a trial needs at least two arms, got {list(arms)}")
    if not frames:
        return []

    rng = random.Random(seed)
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        grouped[frame[dive_key]].append(frame)

    out: list[dict[str, Any]] = []
    # `carry` is the position in the arm cycle, preserved across dives so the
    # remainders of many small dives distribute instead of piling onto one arm.
    carry = rng.randrange(len(arms))
    for dive in sorted(grouped, key=str):
        dive_frames = list(grouped[dive])
        rng.shuffle(dive_frames)
        for frame in dive_frames:
            out.append(dict(frame, arm=arms[carry % len(arms)]))
            carry += 1
    out.sort(key=lambda r: (str(r[dive_key]), r.get("image_id", 0)))
    return out


def summarize_allocation(rows: Sequence[Mapping[str, Any]]) -> str:
    """Human-readable check that the allocation came out balanced."""
    per_arm: dict[str, int] = defaultdict(int)
    per_dive: dict[Any, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        per_arm[row["arm"]] += 1
        per_dive[row["dive_id"]][row["arm"]] += 1

    arms = sorted(per_arm)
    lines = [f"{len(rows)} frames across {len(per_dive)} dives", ""]
    lines.append("  per arm:")
    for arm in arms:
        lines.append(f"    {arm:22s} {per_arm[arm]:5d}")
    # Report how many dives blocking can actually act on. A dive contributing
    # one frame lands in a single arm by necessity, so counting it as
    # "balanced" would flatter the allocation and hide that the per-dive
    # stratified comparison is unavailable for it.
    blockable = sum(1 for c in per_dive.values() if sum(c.values()) >= len(arms))
    worst = max(
        (max(c.values()) - min(c.values()) for c in per_dive.values() if len(c) > 1),
        default=0,
    )
    lines.append("")
    lines.append(
        f"  {blockable} of {len(per_dive)} dives carry at least one frame per arm "
        f"(largest imbalance among them: {worst})"
    )
    if blockable < len(per_dive) / 2:
        lines.append(
            "    NOTE: most dives appear in only one arm, so within-dive"
            " blocking is inactive and no per-dive comparison is possible."
            " Raise --per-dive."
        )
    lines.append("    " + "dive".ljust(30) + "".join(a.rjust(18) for a in arms))
    for dive in sorted(per_dive, key=str)[:12]:
        counts = per_dive[dive]
        lines.append(
            "    " + str(dive).ljust(30)
            + "".join(str(counts.get(a, 0)).rjust(18) for a in arms)
        )
    if len(per_dive) > 12:
        lines.append(f"    ... and {len(per_dive) - 12} more dives")
    return "\n".join(lines)
