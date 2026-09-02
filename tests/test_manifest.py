"""The oracle set, and the sampling that decides what it represents.

The corpus is a free oracle -- 83k completed human labels and 1109 computed
LaserDepth rows -- but only if the sample drawn from it represents the corpus.
Two failure modes are specific enough to test for:

* **Head-of-list sampling.** A dive's opening frames are setup shots, slate
  frames and a diver still settling. Taking the first N per dive measures a
  different population than the dive contains.
* **One dive standing in for the corpus.** The failure modes seen so far are
  dive-shaped -- pool clutter, rigid model props, a rig whose white balance
  was set once topside -- so an unbounded sample is dominated by whichever
  dive happens to be most heavily labelled.

Both are the same lessons `validate_headtail_predictions.py` records; the code
is reimplemented here rather than imported because that file is a standalone
tool, but the reasoning is its, not new.
"""

import pytest

from enhancement_eval.manifest import (
    MANIFEST_FIELDS,
    MANIFEST_SQL,
    parse_laser_points,
    sample_per_dive,
)


# --------------------------------------------------------------------------
# read-only discipline
# --------------------------------------------------------------------------


def test_the_manifest_query_is_pure_select():
    """Operational rule: everything here is READ-ONLY against prod, which is
    also the only deployed instance. This is a tripwire, not a proof -- but it
    catches the edit that adds a write to a query nobody re-reads."""
    lowered = MANIFEST_SQL.lower()
    for forbidden in (
        "insert", "update ", "delete", "drop", "truncate", "alter",
        "create", "grant", "commit",
    ):
        assert forbidden not in lowered, f"manifest SQL contains {forbidden!r}"
    assert lowered.strip().startswith("with") or lowered.strip().startswith("select")


def test_the_manifest_query_only_takes_canonical_images():
    """The same physical frames appear under several dive rows (prod dives 64
    and 66 are both `082929_FishModels_FSL07`), so a non-canonical join would
    score the same file twice and report the duplication as sample size."""
    assert "is_canonical" in MANIFEST_SQL


def test_the_manifest_query_excludes_superseded_labels():
    """A superseded laser label is one the RANSAC validator dead-lettered as
    an outlier. Scoring against it would measure agreement with a label the
    pipeline itself rejected."""
    assert MANIFEST_SQL.lower().count("superseded") >= 2


# --------------------------------------------------------------------------
# laser point parsing
# --------------------------------------------------------------------------


def test_parse_laser_points_handles_the_multi_label_case():
    """461 prod images carry two valid laser labels. They are aggregated
    rather than joined -- a join would multiply the image row out and silently
    weight those images double."""
    assert parse_laser_points("10.5,20.5;30,40") == [(10.5, 20.5), (30.0, 40.0)]


def test_parse_laser_points_of_empty_is_empty():
    assert parse_laser_points("") == []
    assert parse_laser_points(None) == []


def test_parse_laser_points_tolerates_trailing_separators():
    assert parse_laser_points("1,2;") == [(1.0, 2.0)]


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def _rows(dive_id: int, n: int):
    return [{"dive_id": dive_id, "image_id": i} for i in range(n)]


def test_sampling_caps_each_dive():
    rows = _rows(1, 100) + _rows(2, 100)
    sampled = sample_per_dive(rows, per_dive=10)
    assert len(sampled) == 20
    assert sum(1 for r in sampled if r["dive_id"] == 1) == 10


def test_sampling_spreads_across_the_dive_rather_than_taking_the_head():
    """A dive's opening frames are not representative of it."""
    sampled = sample_per_dive(_rows(1, 100), per_dive=5)
    ids = [r["image_id"] for r in sampled]
    assert ids != list(range(5))
    assert max(ids) > 50


def test_sampling_keeps_short_dives_whole():
    sampled = sample_per_dive(_rows(1, 3), per_dive=10)
    assert len(sampled) == 3


def test_sampling_is_deterministic():
    """A report that changes when re-run is not evidence."""
    rows = _rows(1, 57)
    assert sample_per_dive(rows, per_dive=9) == sample_per_dive(rows, per_dive=9)


def test_sampling_never_duplicates_an_image():
    """An integer-step sampler repeats rows on short dives if it is written
    carelessly, which inflates n without adding information."""
    sampled = sample_per_dive(_rows(1, 11), per_dive=10)
    ids = [r["image_id"] for r in sampled]
    assert len(ids) == len(set(ids))


def test_sampling_with_no_cap_returns_everything():
    rows = _rows(1, 40) + _rows(2, 7)
    assert len(sample_per_dive(rows, per_dive=0)) == 47


def test_sampling_preserves_dive_ordering_for_a_stable_report():
    rows = _rows(2, 5) + _rows(1, 5)
    sampled = sample_per_dive(rows, per_dive=2)
    assert [r["dive_id"] for r in sampled] == [1, 1, 2, 2]


# --------------------------------------------------------------------------
# results round-trip
# --------------------------------------------------------------------------


def test_dive_ids_survive_the_csv_round_trip_as_integers():
    """CSV has no types, and a dive_id read back as 65.0 does not compare equal
    to the integer 65 in `KNOWN_POOL_DIVE_IDS` -- which would silently demote
    every documented pool dive to 'inferred' and quietly weaken the one split
    that is supposed to catch out-of-distribution failure.
    """
    from enhancement_eval.cli import _numeric

    row = _numeric({"dive_id": "65", "image_id": "12", "laser_err_px": "1.5",
                    "arm": "baseline"})
    assert row["dive_id"] == 65 and isinstance(row["dive_id"], int)
    assert row["image_id"] == 12
    assert row["laser_err_px"] == 1.5
    assert row["arm"] == "baseline"


def test_a_documented_pool_dive_still_classifies_after_a_round_trip():
    from enhancement_eval.cli import _numeric
    from enhancement_eval.sites import Site, classify_site

    row = _numeric({"dive_id": "65", "dive_name": "reef dive", "dive_path": "/x"})
    result = classify_site(row["dive_id"], name=row["dive_name"], path=row["dive_path"])
    assert result.site is Site.POOL and result.confident


# --------------------------------------------------------------------------
# slate oracle
# --------------------------------------------------------------------------


def test_the_manifest_carries_the_slate_oracle_and_its_template():
    """Slate detection needs both sides: the human reference points to score
    against, and the DiveSlate template (dpi + its own reference points) to
    re-run the estimator at all."""
    for field in ("human_slate_points", "slate_name", "slate_dpi",
                  "slate_template_points"):
        assert field in MANIFEST_FIELDS


def test_the_slate_template_is_resolved_through_the_dive():
    """`Dive.dive_slate_id` is what stages 9/12/13 read. A frame can carry a
    slate label before its dive's slate type has been identified, so the
    template has to come from the dive, not from the label."""
    assert "d.dive_slate_id" in MANIFEST_SQL


def test_an_image_qualifies_on_any_oracle_not_only_a_laser_label():
    """Requiring a laser label would exclude the slate-only dives -- which are
    exactly the population the retired slate detector failed on, and so the
    population any replacement most needs to be measured against."""
    assert "LEFT JOIN las" in MANIFEST_SQL
    lowered = MANIFEST_SQL.lower()
    assert "slate.image_id is not null" in lowered
    assert "ht.image_id is not null" in lowered


def test_slate_geometry_comes_from_a_single_label_row():
    """Both columns are selected in one DISTINCT ON, so a frame cannot take its
    rectangle from one labeler's pass and its reference points from another's."""
    assert "s.slate_rectangle, s.reference_points" in MANIFEST_SQL


def test_booleans_survive_the_csv_round_trip():
    """`bool("False")` is True. Left unhandled, every coverage and slate-gate
    figure reads 100%, McNemar sees no discordant pairs, and the report
    cheerfully says "no coverage change" whatever actually happened -- the
    quietest possible way for this harness to be wrong.
    """
    from enhancement_eval.cli import _numeric

    row = _numeric({"laser_detected": "False", "slate_gate_passed": "True"})
    assert row["laser_detected"] is False
    assert row["slate_gate_passed"] is True


def test_boolean_parsing_survives_a_second_round_trip():
    """`report` may be re-run on a CSV it wrote; real bools must pass through."""
    from enhancement_eval.cli import _numeric

    row = _numeric({"laser_detected": False, "slate_gate_passed": True})
    assert row["laser_detected"] is False
    assert row["slate_gate_passed"] is True


def test_a_coverage_comparison_over_round_tripped_booleans_sees_the_flip():
    """The end-to-end consequence: a real lost detection must show up as a
    discordant pair rather than vanishing."""
    from enhancement_eval.cli import _numeric
    from enhancement_eval.metrics import coverage_flips

    baseline = [_numeric({"laser_detected": v})["laser_detected"] for v in ("True", "True")]
    enhanced = [_numeric({"laser_detected": v})["laser_detected"] for v in ("True", "False")]
    flips = coverage_flips(baseline, enhanced)
    assert flips.lost == 1 and flips.gained == 0
