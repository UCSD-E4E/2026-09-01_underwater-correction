"""The oracle set: what the human-label corpus offers as ground truth.

Read-only, always. Production is the only deployed instance and there is no
staging tier, so every statement here is a SELECT and
`test_the_manifest_query_is_pure_select` is a tripwire against the edit that
changes that. Nothing in this project writes to fishsense-api, Label Studio,
Garage, or Postgres.

One row per canonical image, carrying everything a scored comparison needs:

* the human `LaserLabel` positions, aggregated rather than joined -- 461 prod
  images carry two valid labels, and joining would multiply those rows out and
  silently weight them double;
* the human `HeadTailLabel`, deduped to the most recently updated qualifying
  row, because an image can hold one row per LS project (the legacy shared
  projects plus its per-dive one);
* `LaserDepth.range_m` -- the Euclidean slant distance, *not* `depth_m`, which
  is the Z component. Carried as a covariate here (attenuation is a function of
  range, so it is the natural axis to stratify enhancement effects along) and
  as the physical anchor Deliverable 3 would fit against;
* the camera intrinsics, needed both to rectify and to run the laser detector
  with `rectify_output=True` so its prediction lands in label space;
* the slate quad, where the image has one, for the slate-fitted white balance.

Credentials come from the environment, never from a config file, so no secret
is read into a transcript::

    export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'

Note the DSN is happiest pointed at a **restored backup** rather than prod:
this module is the only thing that touches a database, it is pure SELECT, and
a local restore removes the network path entirely. The interior host in
`deploy/incus/.../settings.toml` is the compose service name and is not
reachable from a dev box in any case.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

__all__ = ["MANIFEST_SQL", "MANIFEST_FIELDS", "parse_laser_points", "sample_per_dive"]

MANIFEST_FIELDS = [
    "image_id",
    "dive_id",
    "dive_name",
    "dive_path",
    "image_path",
    "checksum",
    "laser_points",
    "n_lasers",
    "head_x",
    "head_y",
    "tail_x",
    "tail_y",
    "range_m",
    "depth_m",
    "residual_m",
    "camera_matrix",
    "distortion_coefficients",
    "slate_rectangle",
    # Slate oracle + the template needed to re-run the estimator.
    "human_slate_points",
    "dive_slate_id",
    "slate_name",
    "slate_dpi",
    "slate_template_points",
    "slate_path",
]

# Every label subquery is aggregated or DISTINCT ON before it reaches the main
# SELECT, so the result is exactly one row per image no matter how many label
# rows an image carries. That is not tidiness: an image with two valid laser
# labels appearing twice would be scored twice and counted as two independent
# samples, which is the kind of error that survives review because the output
# still looks like a plausible number of images.
MANIFEST_SQL = """
WITH las AS (
    SELECT l.image_id,
           string_agg(l.x || ',' || l.y, ';' ORDER BY l.id) AS laser_points,
           count(*) AS n_lasers
    FROM laserlabel l
    WHERE l.completed
      AND NOT COALESCE(l.superseded, false)
      AND l.x IS NOT NULL AND l.y IS NOT NULL
    GROUP BY l.image_id
),
ht AS (
    SELECT DISTINCT ON (h.image_id)
           h.image_id, h.head_x, h.head_y, h.tail_x, h.tail_y
    FROM headtaillabel h
    WHERE h.completed
      AND NOT COALESCE(h.superseded, false)
      AND h.head_x IS NOT NULL AND h.head_y IS NOT NULL
      AND h.tail_x IS NOT NULL AND h.tail_y IS NOT NULL
    ORDER BY h.image_id, h.updated_at DESC NULLS LAST, h.id DESC
),
slate AS (
    -- Both geometry columns come from the same label row, so a frame cannot
    -- get its rectangle from one labeler's pass and its reference points from
    -- another's. `slate_rectangle` is used to fit a white balance off the
    -- slate's printed white; `reference_points` is the slate-detection oracle.
    -- Both are stored in photo-frame pixels: the sync activity shifts the LS
    -- composite's PDF-panel width off the x coords before persisting.
    SELECT DISTINCT ON (s.image_id)
           s.image_id, s.slate_rectangle, s.reference_points
    FROM diveslatelabel s
    WHERE s.completed
      AND NOT COALESCE(s.superseded, false)
      AND (s.slate_rectangle IS NOT NULL OR s.reference_points IS NOT NULL)
    ORDER BY s.image_id, s.updated_at DESC NULLS LAST, s.id DESC
)
SELECT i.id                        AS image_id,
       d.id                        AS dive_id,
       COALESCE(d.name, '')        AS dive_name,
       COALESCE(d.path, '')        AS dive_path,
       i.path                      AS image_path,
       i.checksum                  AS checksum,
       las.laser_points            AS laser_points,
       las.n_lasers                AS n_lasers,
       ht.head_x, ht.head_y, ht.tail_x, ht.tail_y,
       ld.range_m, ld.depth_m, ld.residual_m,
       ci.camera_matrix            AS camera_matrix,
       ci.distortion_coefficients  AS distortion_coefficients,
       slate.slate_rectangle       AS slate_rectangle,
       slate.reference_points      AS human_slate_points,
       ds.id                       AS dive_slate_id,
       ds.name                     AS slate_name,
       ds.dpi                      AS slate_dpi,
       ds.reference_points         AS slate_template_points,
       ds.path                     AS slate_path
FROM image i
JOIN dive d              ON d.id = i.dive_id
LEFT JOIN las            ON las.image_id = i.id
LEFT JOIN ht             ON ht.image_id = i.id
LEFT JOIN slate          ON slate.image_id = i.id
LEFT JOIN laserdepth ld  ON ld.image_id = i.id
-- The dive's slate template. Resolved through the dive rather than the label
-- because `Dive.dive_slate_id` is what stages 9/12/13 read, and a frame can
-- carry a slate label before its dive's slate type has been identified.
LEFT JOIN diveslate ds   ON ds.id = d.dive_slate_id
LEFT JOIN camera c       ON c.id = COALESCE(i.camera_id, d.camera_id)
LEFT JOIN cameraintrinsics ci ON ci.camera_id = c.id
-- An image earns a place if it can oracle *something*: a laser dot, a
-- head/tail pair, or a slate. Requiring a laser label (as an inner join did)
-- would have silently excluded the slate-only dives, which are exactly the
-- population the retired slate detector failed on.
WHERE i.is_canonical
  AND (las.image_id IS NOT NULL
       OR ht.image_id IS NOT NULL
       OR slate.image_id IS NOT NULL)
ORDER BY d.id, i.id
"""


def parse_laser_points(raw: str | None) -> list[tuple[float, float]]:
    """Unpack the aggregated `x,y;x,y` laser column.

    An image may carry more than one valid laser label. They are all kept: the
    scorer compares a prediction against the *nearest* human dot rather than
    guessing which one is "the" dot, because there is no basis in the data for
    that guess and picking wrong would report a real hit as a large miss.
    """
    points: list[tuple[float, float]] = []
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, _, y = chunk.partition(",")
        points.append((float(x), float(y)))
    return points


def sample_per_dive(
    rows: Sequence[dict[str, Any]], *, per_dive: int
) -> list[dict[str, Any]]:
    """Cap each dive's contribution, sampling evenly across it.

    Two decisions, both borrowed from `validate_headtail_predictions.py`:

    Capping per dive comes before any global limit, because the failure modes
    seen so far are dive-shaped -- pool clutter, rigid model props, a rig whose
    white balance was set once topside -- so an uncapped sample is dominated by
    whichever dive happens to be most heavily labelled, and that dive's
    idiosyncrasies then read as corpus-wide findings.

    Sampling evenly rather than taking the first N, because a dive's opening
    frames are setup shots, slate frames and a diver still settling. Head-of-
    list sampling measures a different population than the dive contains.

    Deterministic: a report that changes when re-run is not evidence.
    """
    if per_dive <= 0:
        return list(rows)

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dive_id"]].append(row)

    sampled: list[dict[str, Any]] = []
    for dive_id in sorted(grouped):
        dive_rows = grouped[dive_id]
        if len(dive_rows) <= per_dive:
            sampled.extend(dive_rows)
            continue
        step = len(dive_rows) / per_dive
        # De-duplicated by index: an integer-step sampler repeats rows when the
        # dive is only slightly longer than the cap, which inflates n without
        # adding information.
        seen: set[int] = set()
        for i in range(per_dive):
            index = min(len(dive_rows) - 1, int(i * step))
            if index not in seen:
                seen.add(index)
                sampled.append(dive_rows[index])
    return sampled


def fetch_manifest(dsn: str) -> list[dict[str, Any]]:
    """Run the oracle query. Pure SELECT; opens and closes one connection.

    Kept separate from `sample_per_dive` so the sampling is unit-testable
    without a database, and so the only function in this project that talks to
    Postgres is this one.
    """
    import psycopg

    with psycopg.connect(dsn) as conn:
        # An explicitly read-only transaction, so the database itself refuses a
        # write rather than relying on this module's discipline. Production is
        # the only deployed instance; belt and braces is proportionate.
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(MANIFEST_SQL)
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
