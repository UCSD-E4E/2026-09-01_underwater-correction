"""Trial construction: frame selection and randomized assignment.

The design is between-frames, randomized within labeler: each frame is
rendered in exactly ONE arm, so a labeler never sees the same frame twice and
there is no learning or memory confound. Frame difficulty then becomes a noise
source, which randomization balances -- and which the power calculation already
accounts for, since the residual SD it uses was measured across frames.

Two properties matter enough to test. Assignment must be **blocked by dive**,
because dive explains 8-14% of labeling-time variance and an unblocked draw can
easily hand one arm the murkier dives. And it must be **reproducible**, because
a trial whose allocation cannot be regenerated cannot be audited afterwards.
"""

import pytest

from enhancement_eval.trial import assign_arms, summarize_allocation


def _frames(n, dives=4):
    return [{"image_id": i, "dive_id": i % dives, "checksum": f"c{i}"} for i in range(n)]


def test_assignment_is_balanced_overall():
    out = assign_arms(_frames(200), arms=("control", "treatment"), seed=1)
    counts = {}
    for row in out:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    assert abs(counts["control"] - counts["treatment"]) <= 1


def test_assignment_is_balanced_within_every_dive():
    """Dive explains 8-14% of labeling-time variance. An unblocked draw can
    hand one arm the murkier dives and the effect is then unattributable."""
    out = assign_arms(_frames(200, dives=5), arms=("control", "treatment"), seed=2)
    per_dive = {}
    for row in out:
        per_dive.setdefault(row["dive_id"], {}).setdefault(row["arm"], 0)
        per_dive[row["dive_id"]][row["arm"]] += 1
    for dive, counts in per_dive.items():
        assert abs(counts.get("control", 0) - counts.get("treatment", 0)) <= 1, dive


def test_assignment_is_reproducible():
    a = assign_arms(_frames(60), arms=("control", "treatment"), seed=7)
    b = assign_arms(_frames(60), arms=("control", "treatment"), seed=7)
    assert [r["arm"] for r in a] == [r["arm"] for r in b]


def test_a_different_seed_gives_a_different_allocation():
    a = assign_arms(_frames(60), arms=("control", "treatment"), seed=1)
    b = assign_arms(_frames(60), arms=("control", "treatment"), seed=2)
    assert [r["arm"] for r in a] != [r["arm"] for r in b]


def test_every_frame_appears_exactly_once():
    """A frame in both arms would be labelled twice, which is a different
    design with a memory confound, and would double-count in the analysis."""
    out = assign_arms(_frames(101), arms=("control", "treatment"), seed=3)
    ids = [r["image_id"] for r in out]
    assert len(ids) == len(set(ids)) == 101


def test_supports_more_than_two_arms():
    out = assign_arms(_frames(300, dives=3), arms=("a", "b", "c"), seed=4)
    counts = {}
    for row in out:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_a_dive_with_fewer_frames_than_arms_still_allocates():
    """Real cohorts have long tails: dives contributing one or two frames must
    not be dropped, and must not all land in the same arm across dives."""
    frames = [{"image_id": i, "dive_id": i, "checksum": f"c{i}"} for i in range(9)]
    out = assign_arms(frames, arms=("control", "treatment"), seed=5)
    assert len(out) == 9
    counts = {}
    for row in out:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    assert abs(counts["control"] - counts["treatment"]) <= 1


def test_empty_input_is_empty_output():
    assert assign_arms([], arms=("a", "b"), seed=1) == []


def test_at_least_two_arms_required():
    with pytest.raises(ValueError, match="two arms"):
        assign_arms(_frames(10), arms=("only",), seed=1)


def test_summary_reports_per_arm_and_per_dive_counts():
    out = assign_arms(_frames(120, dives=4), arms=("control", "treatment"), seed=6)
    text = summarize_allocation(out)
    assert "control" in text and "treatment" in text
    assert "dive" in text.lower()
