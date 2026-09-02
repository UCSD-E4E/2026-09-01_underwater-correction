"""Pool vs field -- where the out-of-distribution failure will show up.

`Dive` has no site column. It carries a name, a path, a datetime, a priority, a
camera, a slate and a calibration link, and nothing about where the water was.
So the split has to be inferred, and the harness has to say so: a report that
silently presents an inferred split as a measured one is a smaller version of
the mistake the slate detector made, where a gate that looked fine on the
aggregate was wrong on every pool dive.

Three tiers, in precedence order:

1. **Operator override.** Someone who knows the dive can say so at the command
   line. The seed list below cannot be complete, and baking a stale opinion
   into the source would be worse than admitting the gap.
2. **Seeded ids.** Dives 65, 71, 77, 80 and 83 are named in CLAUDE.md as the
   pool dives that produced high-ECC false slate fits. That is documented
   evidence about the exact population that broke the last acceptance gate, so
   it outranks any guess -- including a dive *name* that contradicts it.
3. **Name and path keywords.** A guess, marked as one.

Anything unmatched is `UNKNOWN`, never defaulted into `FIELD`. Defaulting
would fill the field bucket with precisely the dives whose provenance is least
clear, which is the population the pool/field comparison exists to isolate.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Mapping

__all__ = ["Site", "SiteResult", "KNOWN_POOL_DIVE_IDS", "classify_site"]


class Site(enum.Enum):
    POOL = "pool"
    FIELD = "field"
    UNKNOWN = "unknown"


#: The pool dives named in CLAUDE.md's slate-detector postmortem -- the ones
#: that produced high-ECC (0.93-0.97) *false* fits and passed the ECC >= 0.80
#: acceptance gate. Documented evidence, not a heuristic, and the specific
#: population any new acceptance criterion has to survive.
KNOWN_POOL_DIVE_IDS = frozenset({65, 71, 77, 80, 83})

# Word-bounded so 'pool' does not fire on 'Liverpool'. Substring matching on a
# free-text operator field is how a split quietly becomes wrong.
_POOL_PATTERN = re.compile(r"\b(pool|tank|test\s*tank)\b", re.IGNORECASE)
_FIELD_PATTERN = re.compile(
    r"\b(reef|ocean|kelp|jolla|bay|sea|wreck|transect|dock|cove)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class SiteResult:
    """A site classification and how much to trust it."""

    site: Site
    #: True only for an override or a seeded id. The report prints the split of
    #: confident vs inferred so a reader knows how much of the pool/field
    #: comparison rests on documentation and how much on a regex.
    confident: bool
    reason: str


def classify_site(
    dive_id: int,
    *,
    name: str | None,
    path: str | None,
    overrides: Mapping[int, Site] | None = None,
) -> SiteResult:
    """Classify one dive. See the module docstring for the precedence order."""
    if overrides and dive_id in overrides:
        return SiteResult(overrides[dive_id], True, "operator override")

    if dive_id in KNOWN_POOL_DIVE_IDS:
        return SiteResult(Site.POOL, True, "CLAUDE.md slate-detector postmortem")

    haystack = " ".join(part for part in (name, path) if part)
    if not haystack.strip():
        return SiteResult(Site.UNKNOWN, False, "no name or path")

    pool_hit = _POOL_PATTERN.search(haystack)
    field_hit = _FIELD_PATTERN.search(haystack)

    if pool_hit and not field_hit:
        return SiteResult(Site.POOL, False, f"name/path matched {pool_hit.group(0)!r}")
    if field_hit and not pool_hit:
        return SiteResult(
            Site.FIELD, False, f"name/path matched {field_hit.group(0)!r}"
        )
    if pool_hit and field_hit:
        # 'pool' and 'reef' in one string is a naming accident, not a site.
        # Refusing to choose is better than choosing arbitrarily.
        return SiteResult(Site.UNKNOWN, False, "name/path matched both pool and field")
    return SiteResult(Site.UNKNOWN, False, "no site keyword in name or path")
