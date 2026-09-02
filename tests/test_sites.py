"""Pool vs field, which is where the out-of-distribution failure will show up.

There is no site column on `Dive` -- `dive.py` has name, path, datetime,
priority, camera, slate and a calibration link, and nothing about where the
water was. So the split has to be inferred, and the harness has to be honest
that it is inferring it. The alternative -- quietly reporting one corpus-wide
number -- is precisely how the slate detector's ECC gate shipped: it looked
fine on the aggregate and was wrong on every pool dive.

The five dives named in CLAUDE.md as pool dives (65, 71, 77, 80, 83, the ones
that produced high-ECC false fits) are seeded as ground truth. Everything else
is a guess from the name and path, and says so.
"""

import pytest

from enhancement_eval.sites import KNOWN_POOL_DIVE_IDS, Site, classify_site


def test_the_known_pool_dives_from_the_slate_detector_postmortem_are_seeded():
    """These five are not a heuristic -- they are the documented population
    that broke the last acceptance gate that shipped."""
    assert KNOWN_POOL_DIVE_IDS == frozenset({65, 71, 77, 80, 83})
    for dive_id in KNOWN_POOL_DIVE_IDS:
        result = classify_site(dive_id, name="whatever", path="/x/y")
        assert result.site is Site.POOL
        assert result.confident


def test_a_seeded_id_beats_a_contradicting_name():
    """The seed list is evidence; a dive name is a typist's opinion."""
    result = classify_site(65, name="2024-08-21 reef dive 3", path="/data/reef")
    assert result.site is Site.POOL
    assert result.confident


def test_pool_is_inferred_from_the_name():
    assert classify_site(999, name="Pool Test 4", path="/d").site is Site.POOL
    assert classify_site(999, name="fishmodels pool", path="/d").site is Site.POOL


def test_pool_is_inferred_from_the_path():
    assert classify_site(999, name=None, path="/nas/2024/POOL/dive1").site is Site.POOL


def test_field_is_inferred_from_common_site_words():
    for name in ("2024-08-21 reef dive 3", "La Jolla Kelp 2", "Ocean transect"):
        assert classify_site(999, name=name, path="/d").site is Site.FIELD


def test_an_inferred_classification_is_not_marked_confident():
    """The whole point of carrying `confident` is that the report can say how
    much of the pool/field split is documented and how much is guessed."""
    result = classify_site(999, name="Pool Test 4", path="/d")
    assert result.site is Site.POOL
    assert not result.confident


def test_an_unrecognizable_dive_is_unknown_rather_than_defaulted_to_field():
    """Defaulting the unknown into `field` would inflate the field bucket with
    exactly the dives most likely to be unusual, and the pool/field comparison
    is the one that is supposed to catch that."""
    result = classify_site(999, name="FSL07 083012", path="/nas/raw/083012_FSL07")
    assert result.site is Site.UNKNOWN
    assert not result.confident


def test_a_null_name_and_path_do_not_raise():
    """`Dive.name` is nullable and prod has NULLs."""
    assert classify_site(999, name=None, path=None).site is Site.UNKNOWN


def test_matching_is_case_insensitive_and_word_bounded():
    """'pool' must not fire on 'Liverpool' or 'whirlpool'; substring matching
    on a free-text operator field is how a split quietly becomes wrong."""
    assert classify_site(999, name="Liverpool docks", path="/d").site is not Site.POOL
    assert classify_site(999, name="POOL", path="/d").site is Site.POOL


def test_an_operator_override_wins_over_everything():
    """An operator who knows the dive must be able to say so without editing
    the source -- the seed list cannot be complete and pretending otherwise
    would bake a stale opinion into the harness."""
    result = classify_site(
        999, name="reef dive", path="/d", overrides={999: Site.POOL}
    )
    assert result.site is Site.POOL
    assert result.confident
