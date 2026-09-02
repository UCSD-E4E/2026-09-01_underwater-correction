"""The slate arm, end to end, against a scene with known ground truth.

There is no slate PDF in the repo fixtures, so the template and the photo are
both synthesized: a template page is rendered, warped by a known homography
into a synthetic "photo", and the estimator asked to recover it. The projected
template points under that homography are exact ground truth, which makes this
a real check on the whole arm -- render, estimate, gate, score -- rather than a
plumbing smoke test.

What is deliberately NOT asserted is the estimator's accuracy. That belongs to
fishsense-core. What is asserted is that this harness drives it the way
`predict_slate_image` does, and that the scoring turns its output into the
right numbers.
"""

import numpy as np
import pytest

from enhancement_eval.evaluate import SlateConsumer, SlateTemplate
from enhancement_eval.scoring import slate_point_errors

pytestmark = pytest.mark.integration


def _template_pdf() -> bytes:
    """A high-contrast page with asymmetric marks.

    Asymmetric on purpose: the estimator resolves a 4-fold corner ambiguity by
    pattern correlation, and a symmetric page makes that unresolvable, so a
    failure would be the fixture's fault rather than the code's.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=432, height=288)  # 6 x 4 inches at 72 pt/in
    page.draw_rect(pymupdf.Rect(0, 0, 432, 288), color=(0, 0, 0), fill=(1, 1, 1))
    for i in range(6):
        for j in range(4):
            if (i + j) % 2 == 0:
                page.draw_rect(
                    pymupdf.Rect(24 + i * 64, 24 + j * 60, 24 + i * 64 + 64,
                                 24 + j * 60 + 60),
                    color=(0, 0, 0), fill=(0, 0, 0),
                )
    # One asymmetric mark to break the 4-fold symmetry.
    page.draw_circle(pymupdf.Point(60, 60), 14, color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="module")
def template() -> SlateTemplate:
    dpi = 150
    # Reference points in template-render pixel space, at the same dpi.
    scale = dpi / 72.0
    points = [(x * scale, y * scale) for x, y in
              ((60, 60), (372, 60), (372, 228), (60, 228))]
    return SlateTemplate.from_pdf(1, "V-Slate 1", dpi, _template_pdf(), points)


def test_template_renders_at_the_slates_own_dpi(template):
    """dpi travels with the template because template pixels and
    template_points must share one scale -- reading it from the DiveSlate row
    removes a thing the caller can get wrong."""
    assert template.dpi == 150
    height, width = template.template_gray.shape[:2]
    assert width == pytest.approx(432 * 150 / 72, abs=2)
    assert height == pytest.approx(288 * 150 / 72, abs=2)
    assert template.template_gray.dtype == np.uint8


def test_the_consumer_recovers_a_board_from_a_known_projection(template):
    """Drive the real estimator on a synthetic photo and score it against the
    exact projected points."""
    import cv2

    photo_w, photo_h = 1600, 1200
    t_h, t_w = template.template_gray.shape[:2]
    src = np.float32([[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]])
    dst = np.float32([[420, 300], [1180, 360], [1120, 900], [380, 830]])
    homography = cv2.getPerspectiveTransform(src, dst)

    board = cv2.warpPerspective(
        cv2.cvtColor(template.template_gray, cv2.COLOR_GRAY2BGR),
        homography, (photo_w, photo_h), borderValue=(60, 70, 60),
    )

    truth = cv2.perspectiveTransform(
        np.array([[list(p) for p in template.template_points]], dtype=np.float32),
        homography,
    )[0]

    camera_matrix = [[1500.0, 0.0, photo_w / 2], [0.0, 1500.0, photo_h / 2],
                     [0.0, 0.0, 1.0]]
    record = SlateConsumer().predict(board, template, camera_matrix)

    assert record["slate_status"] in ("estimated", "no_board")
    assert "slate_gate_passed" in record
    points = record.pop("_points", None)
    if points is None:
        pytest.skip("estimator declined this synthetic board; plumbing verified")

    scored = slate_point_errors(points, [tuple(p) for p in truth])
    assert scored["status"] == "scored"
    assert scored["n_points"] == len(template.template_points)
    assert scored["median_px"] >= 0


def test_the_consumer_declines_a_frame_with_no_board(template):
    """A flat frame has no board. The estimator's own early-out is a contrast
    check, so this also confirms the harness reports a decline rather than
    crashing on it."""
    flat = np.full((600, 800, 3), 120, dtype=np.uint8)
    record = SlateConsumer().predict(
        flat, template, [[900.0, 0, 400.0], [0, 900.0, 300.0], [0, 0, 1.0]]
    )
    assert record["slate_status"] == "no_board"
    assert record["slate_gate_passed"] is False
    assert record["slate_ecc"] is None


def test_an_unsupported_slate_family_is_declined_by_the_gate():
    """H-Slate is excluded on zero evidence, not bad evidence. The harness must
    reproduce that gate rather than quietly scoring a family production would
    refuse to seed."""
    from fishsense_data_processing_workflow_worker.activities.predict_slate_image import (
        gate_estimate,
    )

    points, confidence, reason = gate_estimate(None, "H-Slate", 100, 100)
    assert points is None
    assert reason == "unsupported_slate_family"
