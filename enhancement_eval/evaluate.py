"""Run both arms over one image and record what each pipeline consumer did.

An "arm" is a decode configuration plus a post-decode enhancer, evaluated at
one of the two injection stages. Every arm is scored against the same human
labels on the same frame, so the comparison is paired image-for-image and
`metrics.paired_delta` can do its job.

Faithfulness notes, each of which changes a number if you get it wrong:

* **The JPEG round-trip is per *consumer*, not per stage.** The head/tail path
  reads the quality-95 JPEG out of Garage, so it is scored on decoded JPEG
  bytes -- otherwise an enhancement gets credit for detail that quantization
  removes before any consumer sees it, and subtle enhancements are exactly the
  ones that get eaten. The slate estimator does **not**: `predict_slate_image`
  rectifies in-process (`RectifiedImage(RawImage(raw_bytes), intrinsics)`) and
  never touches a JPEG, so round-tripping it would measure a loss that stage
  does not actually suffer.
* **The laser arm hands the detector the ORIGINAL Bayer-excess.** It is
  computed from the undemosaiced mosaic, so no RGB enhancer can produce a
  matching one. This makes the linear arm *under-measure* disruption: two of
  the detector's six channels stay clean whatever the enhancer does.
* **`classify_laser_color` is sampled in SENSOR coordinates.** The detector
  reports rectified pixels; `LinearRawImage` is in sensor coordinates, and the
  gap reaches ~48 px at the top-left of the laser region. Sampling one at the
  other's coordinates reads whatever happens to be there and returns a
  confident colour for it.
* **Quality is measured on the baseline frame only**, so both arms are
  stratified by the same number and a quartile means the same thing on each
  side.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from enhancement_eval.contract import (
    IDENTITY,
    Enhancer,
    ProbeResult,
    guard,
    probe_geometry,
)
from enhancement_eval.decode import DecodeConfig
from enhancement_eval.scoring import (
    image_quality,
    nearest_error,
    slate_point_errors,
)

__all__ = [
    "Stage",
    "Arm",
    "LaserConsumer",
    "HeadTailConsumer",
    "SlateConsumer",
    "SlateTemplate",
    "evaluate_image",
]

#: Production's JPEG quality, from `Image.to_jpeg_bytes`. The head/tail arm
#: round-trips through this so it scores the bytes a consumer really sees.
JPEG_QUALITY = 95


class Stage(enum.Enum):
    """Where the enhancer is injected. See `stages.py` for what each reaches."""

    #: uint8 BGR after undistort -- the JPEG a labeler sees and the
    #: segmenter decodes. Constraint #2's placement.
    RECTIFIED = "rectified"
    #: uint16 linear BGR in sensor coordinates -- the laser detector's input,
    #: and `classify_laser_color`'s. Upstream of the JPEG, so an enhancer here
    #: violates constraint #2 by construction; the arm exists to measure what
    #: that constraint is buying.
    LINEAR = "linear"


@dataclass(frozen=True)
class Arm:
    """One configuration to score. `label` keys every reported number.

    Constructing an Arm runs the geometry probe on its enhancer, so an arm that
    moves pixels cannot exist -- the sweep fails at setup rather than producing
    several hundred rows of quietly-invalid coordinates. `guard` catches a
    shape or dtype change on every frame, but a one-pixel roll and a
    square-frame transpose preserve both, and those are the failures that cost
    0.75% length error per pixel. The probe is the only thing that sees them,
    and running it at construction means no caller can forget to.
    """

    label: str
    stage: Stage
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    enhancer: Enhancer = IDENTITY
    #: Filled in by the probe at construction; belongs in the report so a
    #: reader can see the geometric assurance behind this arm's numbers.
    probe: ProbeResult | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Probed at the dtype this arm will really be handed: uint16 for the
        # linear stage, uint8 for the JPEG stage. An enhancer that branches on
        # dtype -- and one that scales by 255 vs 65535 must -- would otherwise
        # be validated on a path it never takes.
        dtype = np.uint16 if self.stage is Stage.LINEAR else np.uint8
        object.__setattr__(
            self,
            "probe",
            probe_geometry(self.enhancer, name=self.label, dtype=dtype),
        )

    @property
    def is_baseline(self) -> bool:
        return self.enhancer is IDENTITY and self.decode == DecodeConfig()


class LaserConsumer:
    """`LaserDetector` + `classify_laser_color`, as the pipeline calls them.

    Loaded once and reused: the checkpoint load is expensive and every image in
    a sweep pays for it otherwise.
    """

    def __init__(self, checkpoint_path: str | None = None):
        from fishsense_core.laser import LaserDetector

        self._detector = (
            LaserDetector.from_checkpoint(checkpoint_path)
            if checkpoint_path
            else LaserDetector.from_pretrained()
        )

    def predict(
        self,
        bgr16: np.ndarray,
        bayer_excess: np.ndarray | None,
        camera_matrix,
        distortion,
        wavelength: str | None = None,
    ) -> tuple[tuple[float, float] | None, float]:
        """Rectified (x, y) and confidence, or (None, confidence).

        `rectify_output=True` puts the answer in the same space as
        `LaserLabel.x/y`, which is what makes the comparison legitimate at all.
        Every other keyword is left at its default, because the defaults ARE
        the production recipe and changing one silently changes what is being
        measured.
        """
        prediction = self._detector.predict(
            bgr16,
            bayer_excess=bayer_excess,
            wavelength=wavelength,
            rectify_output=True,
            camera_matrix=np.asarray(camera_matrix, dtype=float),
            distortion=np.asarray(distortion, dtype=float),
        )
        confidence = float(getattr(prediction, "confidence", float("nan")))
        if prediction.x is None or prediction.y is None:
            return None, confidence
        return (float(prediction.x), float(prediction.y)), confidence

    @staticmethod
    def sample_color(
        bgr16: np.ndarray,
        rectified_point: tuple[float, float],
        camera_matrix,
        distortion,
    ) -> tuple[str | None, float | None]:
        """The tripwire. Sampled in sensor coordinates -- see the module note."""
        from fishsense_data_processing_workflow_worker.laser_color import (
            classify_laser_color,
            rectified_to_sensor_point,
        )

        sensor_x, sensor_y = rectified_to_sensor_point(
            rectified_point[0], rectified_point[1], camera_matrix, distortion
        )
        return classify_laser_color(bgr16, sensor_x, sensor_y)


class HeadTailConsumer:
    """`FishSegmentation` + `FishHeadTailDetector`, gated on the laser.

    The gate -- run the head/tail detector on the instance the laser landed on,
    rather than on the largest instance -- is `validate_headtail_predictions.py`'s,
    and it is reused rather than reinvented because its whole value is that the
    segmenter's failure mode on cluttered pool dives is confident false
    positives on divers' legs.
    """

    def __init__(self) -> None:
        from fishsense_core.fish import FishHeadTailDetector, FishSegmentation

        self._segmentation = FishSegmentation()
        self._segmentation.load_model()
        self._detector = FishHeadTailDetector()

    def predict(
        self, bgr8: np.ndarray, laser_points: Sequence[tuple[float, float]]
    ) -> dict[str, Any]:
        """Head/tail for the laser's fish, or an abstention with a reason.

        Abstentions are kept distinct from errors: abstaining is a *good*
        outcome relative to seeding a wrong keypoint, so folding them into an
        error rate would penalise the safe behaviour.
        """
        labels = self._segmentation.inference(bgr8)
        instances = [int(v) for v in np.unique(labels) if v]
        if not instances:
            return {"status": "no_detections", "n_instances": 0}

        height, width = labels.shape
        hit = 0
        for lx, ly in laser_points:
            xi, yi = int(round(lx)), int(round(ly))
            if 0 <= xi < width and 0 <= yi < height and labels[yi, xi]:
                hit = int(labels[yi, xi])
                break
        if not hit:
            return {"status": "laser_off_all_fish", "n_instances": len(instances)}

        mask = ((labels == hit).astype(np.uint8)) * 255
        # The detector is a PyO3 native call with no documented exception
        # hierarchy, and one unfittable mask must not abort a several-hundred-
        # image sweep.
        try:
            head, tail = self._detector.find_head_tail_img(mask)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "status": "headtail_failed",
                "n_instances": len(instances),
                "error": str(exc),
            }
        return {
            "status": "predicted",
            "n_instances": len(instances),
            "head": (float(head[0]), float(head[1])),
            "tail": (float(tail[0]), float(tail[1])),
            "instance_area_px": int(np.count_nonzero(labels == hit)),
        }


@dataclass(frozen=True)
class SlateTemplate:
    """A rendered `DiveSlate` template: what the estimator correlates against.

    `template_gray` is page 0 of the slate PDF rendered at the slate's own
    `dpi`, so template pixels and `template_points` share one scale -- which is
    the reason `dpi` travels with the template rather than being chosen here.
    """

    slate_id: int
    name: str
    dpi: int
    template_gray: np.ndarray
    template_points: tuple[tuple[float, float], ...]

    @classmethod
    def from_pdf(
        cls, slate_id: int, name: str, dpi: int, pdf_bytes: bytes, points
    ) -> "SlateTemplate":
        """Render page 0 to grayscale at `dpi`. Mirrors
        `predict_slate_image._render_template_gray` exactly."""
        import pymupdf

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            page = document.load_page(0)
            pixmap = page.get_pixmap(dpi=int(dpi), colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width
        )
        return cls(
            slate_id=slate_id,
            name=name,
            dpi=int(dpi),
            template_gray=gray,
            template_points=tuple((float(x), float(y)) for x, y in points),
        )


class SlateConsumer:
    """`fishsense_core.slate.estimate_plane`, as the retired stage called it.

    Reproduces `predict_slate_image` rather than improving on it: same
    classical path (`board_mask=None` by default, the supported fallback that
    costs ~13 points of coverage), same `gate_estimate` gates, same ECC floor.
    The question is what enhancement does to *that* stage, so any change here
    would be measuring something else.

    The board mask is off by default because it is a second learned model with
    its own distribution problems, and mixing it in would confound the one
    variable under test.
    """

    def __init__(self, min_confidence: float | None = None, use_board_mask: bool = False):
        from fishsense_data_processing_workflow_worker.activities.predict_slate_image import (
            DEFAULT_MIN_CONFIDENCE,
        )

        self.min_confidence = (
            DEFAULT_MIN_CONFIDENCE if min_confidence is None else min_confidence
        )
        self._masker = None
        if use_board_mask:
            from fishsense_core.slate import BoardMasker

            self._masker = BoardMasker.from_pretrained()

    def predict(
        self, bgr8: np.ndarray, template: SlateTemplate, camera_matrix
    ) -> dict[str, Any]:
        """Estimate the board plane on one rectified frame.

        Returns the raw estimate diagnostics *and* the gate's verdict, kept
        apart on purpose. The whole reason this stage was retired is that the
        ECC gate accepted confident wrong answers, so a harness that reported
        only gated output could not see the failure it is looking for.
        """
        from fishsense_core.slate import estimate_plane
        from fishsense_data_processing_workflow_worker.activities.predict_slate_image import (
            gate_estimate,
        )

        height, width = bgr8.shape[:2]
        board_mask = None
        if self._masker is not None:
            try:
                board_mask = self._masker.predict(bgr8)
            except Exception:  # pylint: disable=broad-except
                board_mask = None

        try:
            estimate = estimate_plane(
                bgr8,
                template.template_gray,
                [[x, y] for x, y in template.template_points],
                float(template.dpi),
                np.asarray(camera_matrix, dtype=float),
                board_mask=board_mask,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # A classical CV pipeline over a few hundred frames will meet one
            # it cannot handle; that must not abort the sweep, but it must be
            # counted rather than silently dropped.
            return {"slate_status": "estimator_failed", "slate_error": str(exc)}

        points, confidence, reason = gate_estimate(
            estimate, template.name, int(width), int(height), self.min_confidence
        )
        record: dict[str, Any] = {
            "slate_status": "estimated" if estimate is not None else "no_board",
            "slate_ecc": float(estimate.ecc_score) if estimate is not None else None,
            "slate_reproj_rms": (
                float(estimate.reprojection_rms) if estimate is not None else None
            ),
            "slate_area_px": (
                float(estimate.board_area_px) if estimate is not None else None
            ),
            # Did the production gate accept it? Recorded separately from
            # whether it was *right*, which is the entire point.
            "slate_gate_passed": points is not None,
            "slate_gate_reason": reason,
        }
        # Score the ungated estimate too. A fit the gate rejected may still
        # have been correct, and an enhancement that turns correct-but-rejected
        # into correct-and-accepted is a real gain the gated view cannot show.
        record["_points"] = (
            [(float(x), float(y)) for x, y in estimate.image_points]
            if estimate is not None
            else None
        )
        return record


def _jpeg_round_trip(bgr8: np.ndarray) -> np.ndarray:
    """Encode and decode at production's quality, so the scored bytes are the
    bytes a consumer actually receives."""
    import cv2

    ok, encoded = cv2.imencode(".jpg", bgr8, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _score_headtail(pred: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Position and length error against the human keypoints.

    Both head/tail assignments are scored and the better kept, so a systematic
    orientation swap surfaces as a low `orientation_ok` rate rather than
    silently doubling the position error.
    """
    import math

    out = {"ht_status": pred["status"], "ht_n_instances": pred.get("n_instances")}
    if pred["status"] != "predicted":
        return out
    if row.get("head_x") is None:
        return out | {"ht_status": "no_human_label"}

    hx, hy = float(row["head_x"]), float(row["head_y"])
    tx, ty = float(row["tail_x"]), float(row["tail_y"])
    (phx, phy), (ptx, pty) = pred["head"], pred["tail"]

    def dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    direct = dist(phx, phy, hx, hy) + dist(ptx, pty, tx, ty)
    swapped = dist(phx, phy, tx, ty) + dist(ptx, pty, hx, hy)
    orientation_ok = direct <= swapped
    if not orientation_ok:
        phx, phy, ptx, pty = ptx, pty, phx, phy

    human_len = dist(hx, hy, tx, ty)
    pred_len = dist(phx, phy, ptx, pty)
    return out | {
        "ht_orientation_ok": int(orientation_ok),
        "ht_err_head_px": dist(phx, phy, hx, hy),
        "ht_err_tail_px": dist(ptx, pty, tx, ty),
        "ht_human_len_px": human_len,
        "ht_pred_len_px": pred_len,
        # Normalized because a 40 px miss on a frame-filling snook and on a
        # distant fish are not the same mistake.
        "ht_err_head_pct": (100 * dist(phx, phy, hx, hy) / human_len) if human_len else None,
        # Signed: this is the number that reaches `Measurement`.
        "ht_len_err_pct": (100 * (pred_len - human_len) / human_len) if human_len else None,
    }


def evaluate_image(
    row: dict[str, Any],
    raw_bytes: bytes,
    arms: Sequence[Arm],
    *,
    laser: LaserConsumer | None = None,
    headtail: HeadTailConsumer | None = None,
    slate: SlateConsumer | None = None,
    slate_template: SlateTemplate | None = None,
    wavelength: str | None = None,
) -> list[dict[str, Any]]:
    """Score every arm on one image. Returns one record per arm.

    Both arms decode the same raw bytes, so the only difference between records
    is the arm's own configuration -- which is what makes the delta
    attributable.
    """
    from enhancement_eval.manifest import parse_laser_points
    from enhancement_eval.stages import decode_linear_stage, decode_rectified_stage, rectify

    human_lasers = parse_laser_points(row.get("laser_points"))
    camera_matrix = row.get("camera_matrix")
    distortion = row.get("distortion_coefficients")
    slate_quad = row.get("slate_rectangle")

    base = {
        "image_id": row.get("image_id"),
        "dive_id": row.get("dive_id"),
        "dive_name": row.get("dive_name"),
        "checksum": row.get("checksum"),
        "range_m": row.get("range_m"),
        "n_human_lasers": len(human_lasers),
    }

    records: list[dict[str, Any]] = []
    # Quality is computed once, on the baseline decode of whichever stage is in
    # play, and copied onto every arm's record -- so a stratification bucket
    # means the same population on each side of the comparison.
    quality_cache: dict[Stage, dict[str, float]] = {}

    for arm in arms:
        record = dict(base) | {"arm": arm.label, "stage": arm.stage.value,
                               "decode": arm.decode.label}
        enhance = guard(arm.enhancer, name=arm.label)

        if arm.stage is Stage.LINEAR:
            bgr16, bayer = decode_linear_stage(
                raw_bytes, arm.decode, slate_quad=slate_quad
            )
            if Stage.LINEAR not in quality_cache and arm.is_baseline:
                quality_cache[Stage.LINEAR] = image_quality(bgr16)
            enhanced = enhance(bgr16)

            if laser is not None and camera_matrix is not None:
                point, confidence = laser.predict(
                    enhanced, bayer, camera_matrix, distortion, wavelength
                )
                record["laser_confidence"] = confidence
                record["laser_detected"] = point is not None
                record["laser_err_px"] = nearest_error(point, human_lasers)
                if point is not None:
                    record["laser_x"], record["laser_y"] = point
                # The tripwire is sampled at the HUMAN dot, not the predicted
                # one, so a colour flip is attributable to the enhancement
                # rather than to the detector having moved.
                if human_lasers:
                    colour, margin = laser.sample_color(
                        enhanced, human_lasers[0], camera_matrix, distortion
                    )
                    record["laser_color"] = colour
                    record["laser_color_margin"] = margin
            record.update(quality_cache.get(Stage.LINEAR, {}))

        else:
            bgr8 = decode_rectified_stage(
                raw_bytes, arm.decode, slate_quad=slate_quad
            )
            if camera_matrix is not None:
                bgr8 = rectify(bgr8, camera_matrix, distortion)
            if Stage.RECTIFIED not in quality_cache and arm.is_baseline:
                quality_cache[Stage.RECTIFIED] = image_quality(bgr8)
            enhanced = enhance(bgr8)

            # Slate first, on the raw rectified array: `predict_slate_image`
            # rectifies in-process and never reads a JPEG.
            if slate is not None and slate_template is not None and camera_matrix:
                slate_record = slate.predict(enhanced, slate_template, camera_matrix)
                predicted_points = slate_record.pop("_points", None)
                record.update(slate_record)
                human_points = row.get("human_slate_points") or []
                errors = slate_point_errors(predicted_points, human_points)
                record.update(
                    {f"slate_{k}": v for k, v in errors.items() if k != "status"}
                )
                record["slate_score_status"] = errors["status"]

            # Head/tail second, on JPEG-decoded bytes: that path reads the
            # stage-5.1 JPEG out of Garage.
            if headtail is not None:
                record.update(
                    _score_headtail(
                        headtail.predict(_jpeg_round_trip(enhanced), human_lasers), row
                    )
                )
            record.update(quality_cache.get(Stage.RECTIFIED, {}))

        records.append(record)

    return records
