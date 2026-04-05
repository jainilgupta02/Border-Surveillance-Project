"""
Test Suite — Object Detection Module
======================================

All tests mock the YOLO model so no GPU, no .pt file, and no network
connection are required.  The test suite runs on any CI machine.

Run all tests:
    pytest tests/test_detector.py -v

Run only fast tests (skip integration):
    pytest tests/test_detector.py -v -m "not integration"

Run with coverage:
    pytest tests/test_detector.py --cov=detector --cov-report=html

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import time
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest

from detector import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_THREAT,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    FRAME_SIZE,
    BorderDetector,
    Detection,
    FrameResult,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_frame(h: int = 640, w: int = 640, dtype=np.uint8) -> np.ndarray:
    """Return a blank BGR frame."""
    if dtype == np.float32:
        return np.zeros((h, w, 3), dtype=np.float32)
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_yolo_box(
    class_id: int = 0,
    confidence: float = 0.85,
    x1: float = 100, y1: float = 100,
    x2: float = 300, y2: float = 400,
) -> MagicMock:
    """Build a mock YOLO box object matching the Ultralytics API."""
    box = MagicMock()
    box.cls  = MagicMock()
    box.cls.__getitem__ = MagicMock(return_value=class_id)
    box.conf = MagicMock()
    box.conf.__getitem__ = MagicMock(return_value=confidence)
    box.xyxy = MagicMock()
    box.xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(tolist=MagicMock(return_value=[x1, y1, x2, y2]))
    )
    return box


def _make_yolo_result(boxes: list) -> MagicMock:
    """Build a mock YOLO result object containing the given boxes."""
    result = MagicMock()
    result.boxes = boxes
    return result


def _make_detector(filter_classes=None) -> BorderDetector:
    """
    Build a BorderDetector with the YOLO model fully mocked.
    Returns both the detector and the mock model for further configuration.
    """
    with patch("detector.os.path.exists", return_value=True), \
         patch("detector.YOLO") as mock_yolo_cls:
        mock_yolo_cls.return_value = MagicMock()
        detector = BorderDetector(
            model_path="models/border_yolo.pt",
            filter_classes=filter_classes,
        )
    return detector


def _frame_item(
    frame_id: int = 1,
    frame: np.ndarray = None,
    motion_score: float = None,
) -> dict:
    """Build a frame_item dict as returned by extract_frames()."""
    return {
        "frame_id":       frame_id,
        "frame":          frame if frame is not None else _make_frame(),
        "flow_magnitude": None,
        "motion_score":   motion_score,
    }


# ===========================================================================
# 1. Detection dataclass
# ===========================================================================

class TestDetection:
    """Tests for the Detection dataclass and to_dict()."""

    def _make(self, class_id=0, conf=0.9, threat="medium") -> Detection:
        return Detection(
            class_id    = class_id,
            class_name  = CLASS_NAMES[class_id],
            confidence  = conf,
            bbox        = [10.0, 20.0, 100.0, 200.0],
            center_x    = 0.43,
            center_y    = 0.68,
            width_norm  = 0.14,
            height_norm = 0.28,
            area_norm   = 0.039,
            threat_level = threat,
        )

    def test_to_dict_has_all_required_keys(self):
        det = self._make()
        d = det.to_dict()
        for key in ("class_id", "class_name", "confidence", "bbox",
                    "center_x", "center_y", "width_norm", "height_norm",
                    "area_norm", "threat_level"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_confidence_rounded(self):
        det = self._make(conf=0.123456789)
        assert det.to_dict()["confidence"] == 0.1235

    def test_to_dict_bbox_rounded(self):
        det = self._make()
        det.bbox = [10.123456, 20.999, 100.1, 200.0]
        bbox = det.to_dict()["bbox"]
        assert all(isinstance(v, float) for v in bbox)
        assert bbox[0] == 10.12

    def test_threat_level_stored_correctly(self):
        for level in ("low", "medium", "high", "critical"):
            det = self._make(threat=level)
            assert det.to_dict()["threat_level"] == level


# ===========================================================================
# 2. FrameResult dataclass
# ===========================================================================

class TestFrameResult:
    """Tests for FrameResult dataclass and to_dict()."""

    def _make(self, n_detections: int = 2) -> FrameResult:
        dets = [
            Detection(
                class_id=i, class_name=CLASS_NAMES[i],
                confidence=0.8, bbox=[0, 0, 100, 100],
                threat_level=CLASS_THREAT[i],
            )
            for i in range(n_detections)
        ]
        return FrameResult(
            frame_id        = 5,
            timestamp       = time.time(),
            detections      = dets,
            detection_count = n_detections,
            has_critical    = False,
            has_high        = False,
            motion_score    = 3.5,
            inference_ms    = 42.1,
        )

    def test_to_dict_has_all_keys(self):
        fr = self._make()
        d  = fr.to_dict()
        for key in ("frame_id", "timestamp", "detection_count",
                    "has_critical", "has_high", "motion_score",
                    "inference_ms", "detections"):
            assert key in d

    def test_detections_serialised_as_list_of_dicts(self):
        fr = self._make(n_detections=3)
        d  = fr.to_dict()
        assert isinstance(d["detections"], list)
        assert len(d["detections"]) == 3
        assert isinstance(d["detections"][0], dict)

    def test_none_motion_score_stays_none(self):
        fr = self._make()
        fr.motion_score = None
        assert fr.to_dict()["motion_score"] is None

    def test_motion_score_rounded(self):
        fr = self._make()
        fr.motion_score = 3.123456
        assert fr.to_dict()["motion_score"] == 3.1235

    def test_has_critical_and_has_high_are_independent(self):
        fr = self._make()
        fr.has_critical = True
        fr.has_high     = False
        d = fr.to_dict()
        assert d["has_critical"] is True
        assert d["has_high"]     is False


# ===========================================================================
# 3. BorderDetector construction
# ===========================================================================

class TestBorderDetectorInit:
    """Tests for BorderDetector initialisation and model loading."""

    def test_loads_model_when_file_exists(self):
        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            det = BorderDetector(model_path="models/border_yolo.pt")
            mock_yolo.assert_called_once_with("models/border_yolo.pt")

    def test_falls_back_to_generic_model_when_custom_missing(self):
        with patch("detector.os.path.exists", return_value=False), \
             patch("detector.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            det = BorderDetector(model_path="models/border_yolo.pt")
            # Should have loaded the fallback, not the custom path
            called_path = mock_yolo.call_args[0][0]
            assert called_path != "models/border_yolo.pt"

    def test_raises_runtime_error_when_yolo_crashes(self):
        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO", side_effect=Exception("corrupt weights")):
            with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
                BorderDetector()

    def test_default_confidence_and_iou_set(self):
        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO", return_value=MagicMock()):
            det = BorderDetector()
            assert det.confidence == DEFAULT_CONFIDENCE
            assert det.iou        == DEFAULT_IOU

    def test_filter_classes_stored(self):
        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO", return_value=MagicMock()):
            det = BorderDetector(filter_classes={3, 6})
            assert det.filter_classes == {3, 6}


# ===========================================================================
# 4. BorderDetector._build_detection
# ===========================================================================

class TestBuildDetection:
    """Unit tests for the private _build_detection helper."""

    def setup_method(self):
        self.detector = _make_detector()

    def test_class_name_resolved_correctly(self):
        box = _make_yolo_box(class_id=3)
        det = self.detector._build_detection(box, 640, 640)
        assert det.class_name == "military_vehicle"

    def test_threat_level_critical_for_military_vehicle(self):
        box = _make_yolo_box(class_id=3)
        det = self.detector._build_detection(box, 640, 640)
        assert det.threat_level == "critical"

    def test_threat_level_low_for_vehicle(self):
        box = _make_yolo_box(class_id=1)
        det = self.detector._build_detection(box, 640, 640)
        assert det.threat_level == "low"

    def test_normalised_coordinates_in_unit_range(self):
        box = _make_yolo_box(x1=0, y1=0, x2=640, y2=640)
        det = self.detector._build_detection(box, 640, 640)
        assert 0.0 <= det.center_x <= 1.0
        assert 0.0 <= det.center_y <= 1.0
        assert 0.0 <= det.width_norm <= 1.0
        assert 0.0 <= det.height_norm <= 1.0
        assert 0.0 <= det.area_norm <= 1.0

    def test_area_norm_equals_width_times_height(self):
        box = _make_yolo_box(x1=0, y1=0, x2=320, y2=320)
        det = self.detector._build_detection(box, 640, 640)
        assert abs(det.area_norm - det.width_norm * det.height_norm) < 1e-6

    def test_unknown_class_id_uses_fallback_name(self):
        box = _make_yolo_box(class_id=99)
        det = self.detector._build_detection(box, 640, 640)
        assert det.class_name == "class_99"

    def test_bbox_stored_as_list_of_four_floats(self):
        box = _make_yolo_box(x1=50, y1=60, x2=200, y2=300)
        det = self.detector._build_detection(box, 640, 640)
        assert len(det.bbox) == 4
        # YOLO returns floats; mock returns ints — both are valid numeric types
        assert all(isinstance(v, (int, float)) for v in det.bbox)


# ===========================================================================
# 5. BorderDetector.detect()
# ===========================================================================

class TestDetect:
    """Tests for the main detect() method."""

    def setup_method(self):
        self.detector = _make_detector()

    def _run_detect(self, boxes: list, motion_score=None) -> FrameResult:
        """Helper: run detect() with given mock boxes."""
        mock_result = _make_yolo_result(boxes)
        self.detector._model.return_value = [mock_result]

        item = _frame_item(frame_id=7, motion_score=motion_score)
        return self.detector.detect(item)

    def test_returns_frame_result(self):
        result = self._run_detect([])
        assert isinstance(result, FrameResult)

    def test_frame_id_preserved(self):
        result = self._run_detect([])
        assert result.frame_id == 7

    def test_empty_detections_when_no_boxes(self):
        result = self._run_detect([])
        assert result.detection_count == 0
        assert result.detections == []

    def test_detection_count_matches_boxes(self):
        boxes = [_make_yolo_box(class_id=0), _make_yolo_box(class_id=1)]
        result = self._run_detect(boxes)
        assert result.detection_count == 2

    def test_has_critical_true_for_military_vehicle(self):
        result = self._run_detect([_make_yolo_box(class_id=3)])  # military_vehicle
        assert result.has_critical is True
        assert result.has_high     is False

    def test_has_critical_true_for_suspicious_object(self):
        result = self._run_detect([_make_yolo_box(class_id=6)])
        assert result.has_critical is True

    def test_has_high_true_for_crowd(self):
        result = self._run_detect([_make_yolo_box(class_id=2)])  # crowd
        assert result.has_high     is True
        assert result.has_critical is False

    def test_has_high_true_for_aircraft(self):
        result = self._run_detect([_make_yolo_box(class_id=4)])  # aircraft
        assert result.has_high     is True
        assert result.has_critical is False

    def test_has_critical_and_has_high_both_set_when_mixed(self):
        boxes = [_make_yolo_box(class_id=3), _make_yolo_box(class_id=2)]
        result = self._run_detect(boxes)
        assert result.has_critical is True
        assert result.has_high     is True

    def test_neither_flag_for_low_threat(self):
        result = self._run_detect([_make_yolo_box(class_id=1)])  # vehicle → low
        assert result.has_critical is False
        assert result.has_high     is False

    def test_motion_score_passed_through(self):
        result = self._run_detect([], motion_score=4.2)
        assert result.motion_score == pytest.approx(4.2)

    def test_none_motion_score_stays_none(self):
        result = self._run_detect([], motion_score=None)
        assert result.motion_score is None

    def test_inference_ms_is_positive(self):
        result = self._run_detect([])
        assert result.inference_ms >= 0.0

    def test_timestamp_is_recent(self):
        before = time.time()
        result = self._run_detect([])
        after  = time.time()
        assert before <= result.timestamp <= after

    def test_float32_frame_accepted_without_error(self):
        """detect() must handle float32 normalised frames from preprocessing."""
        mock_result = _make_yolo_result([])
        self.detector._model.return_value = [mock_result]

        float_frame = _make_frame(dtype=np.float32)
        item = _frame_item(frame=float_frame)
        result = self.detector.detect(item)
        assert isinstance(result, FrameResult)

    def test_filter_classes_excludes_unwanted(self):
        """Only class_ids in filter_classes should appear in detections."""
        detector = _make_detector(filter_classes={3})   # only military_vehicle
        boxes = [
            _make_yolo_box(class_id=0),  # person  → filtered out
            _make_yolo_box(class_id=3),  # military_vehicle → kept
        ]
        mock_result = _make_yolo_result(boxes)
        detector._model.return_value = [mock_result]

        result = detector.detect(_frame_item())
        assert result.detection_count == 1
        assert result.detections[0].class_id == 3

    def test_filter_classes_none_keeps_all(self):
        """When filter_classes is None every class passes through."""
        detector = _make_detector(filter_classes=None)
        boxes = [_make_yolo_box(class_id=i) for i in range(7)]
        mock_result = _make_yolo_result(boxes)
        detector._model.return_value = [mock_result]

        result = detector.detect(_frame_item())
        assert result.detection_count == 7

    def test_none_boxes_in_result_handled_gracefully(self):
        """If YOLO returns a result with boxes=None, no crash should occur."""
        mock_result = MagicMock()
        mock_result.boxes = None
        self.detector._model.return_value = [mock_result]
        result = self.detector.detect(_frame_item())
        assert result.detection_count == 0


# ===========================================================================
# 6. BorderDetector.annotate_frame()
# ===========================================================================

class TestAnnotateFrame:
    """Tests for the annotate_frame() method."""

    def setup_method(self):
        self.detector = _make_detector()

    def _make_result(self, detections=None) -> FrameResult:
        dets = detections or []
        return FrameResult(
            frame_id        = 1,
            timestamp       = time.time(),
            detections      = dets,
            detection_count = len(dets),
            has_critical    = any(d.threat_level == "critical" for d in dets),
            has_high        = any(d.threat_level == "high"     for d in dets),
            inference_ms    = 20.0,
        )

    def test_returns_uint8_array(self):
        frame  = _make_frame()
        result = self._make_result()
        out    = self.detector.annotate_frame(frame, result)
        assert out.dtype == np.uint8

    def test_does_not_modify_original_frame(self):
        frame  = _make_frame()
        frame[:] = 128
        original = frame.copy()
        result   = self._make_result()
        self.detector.annotate_frame(frame, result)
        np.testing.assert_array_equal(frame, original)

    def test_float32_frame_converted_to_uint8(self):
        frame  = _make_frame(dtype=np.float32)
        result = self._make_result()
        out    = self.detector.annotate_frame(frame, result)
        assert out.dtype == np.uint8

    def test_output_same_spatial_dimensions_as_input(self):
        frame  = _make_frame(h=480, w=640)
        result = self._make_result()
        out    = self.detector.annotate_frame(frame, result)
        assert out.shape[:2] == (480, 640)

    def test_annotates_with_detections_without_crash(self):
        frame = _make_frame()
        det = Detection(
            class_id=0, class_name="person", confidence=0.9,
            bbox=[50.0, 50.0, 200.0, 300.0],
            threat_level="medium",
        )
        result = self._make_result(detections=[det])
        out = self.detector.annotate_frame(frame, result)
        assert out is not None

    def test_motion_score_included_in_overlay_when_present(self):
        """Smoke test — just verify no crash when motion_score is set."""
        frame = _make_frame()
        result = self._make_result()
        result.motion_score = 5.5
        out = self.detector.annotate_frame(frame, result)
        assert out is not None


# ===========================================================================
# 7. BorderDetector.get_stats()
# ===========================================================================

class TestGetStats:
    """Tests for the get_stats() summary method."""

    def setup_method(self):
        self.detector = _make_detector()

    def _build_results(self) -> List[FrameResult]:
        """Build two deterministic FrameResults for testing."""
        det_person = Detection(
            class_id=0, class_name="person", confidence=0.9,
            bbox=[0, 0, 100, 100], threat_level="medium",
        )
        det_military = Detection(
            class_id=3, class_name="military_vehicle", confidence=0.85,
            bbox=[200, 200, 400, 400], threat_level="critical",
        )
        r1 = FrameResult(
            frame_id=1, timestamp=time.time(),
            detections=[det_person], detection_count=1,
            has_critical=False, has_high=False, inference_ms=30.0,
        )
        r2 = FrameResult(
            frame_id=2, timestamp=time.time(),
            detections=[det_person, det_military], detection_count=2,
            has_critical=True, has_high=False, inference_ms=40.0,
        )
        return [r1, r2]

    def test_returns_empty_dict_for_empty_input(self):
        assert self.detector.get_stats([]) == {}

    def test_total_frames_correct(self):
        results = self._build_results()
        stats   = self.detector.get_stats(results)
        assert stats["total_frames"] == 2

    def test_total_detections_correct(self):
        stats = self.detector.get_stats(self._build_results())
        assert stats["total_detections"] == 3   # 1 + 2

    def test_detections_per_frame(self):
        stats = self.detector.get_stats(self._build_results())
        assert stats["detections_per_frame"] == 1.5

    def test_class_counts(self):
        stats = self.detector.get_stats(self._build_results())
        assert stats["class_counts"]["person"]           == 2
        assert stats["class_counts"]["military_vehicle"] == 1

    def test_critical_frames_count(self):
        stats = self.detector.get_stats(self._build_results())
        assert stats["critical_frames"] == 1

    def test_avg_confidence_in_range(self):
        stats = self.detector.get_stats(self._build_results())
        assert 0.0 < stats["avg_confidence"] <= 1.0

    def test_avg_inference_ms_positive(self):
        stats = self.detector.get_stats(self._build_results())
        assert stats["avg_inference_ms"] > 0

    def test_no_detections_confidence_is_zero(self):
        r = FrameResult(
            frame_id=1, timestamp=time.time(),
            detection_count=0, inference_ms=15.0,
        )
        stats = self.detector.get_stats([r])
        assert stats["avg_confidence"] == 0.0

    def test_stats_has_high_frames_key(self):
        stats = self.detector.get_stats(self._build_results())
        assert "high_frames" in stats


# ===========================================================================
# 8. Integration tests  (no real model — full pipeline mocked)
# ===========================================================================

@pytest.mark.integration
class TestIntegration:
    """
    End-to-end tests using real OpenCV video I/O but a mocked YOLO model.
    These verify that detector.py and preprocessing.py work together.
    """

    def _make_video(self, tmp_path, n_frames=15) -> str:
        path   = str(tmp_path / "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
        for i in range(n_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (i * 10 % 256, i * 7 % 256, i * 5 % 256)
            out.write(frame)
        out.release()
        return path

    def test_process_video_yields_frame_results(self, tmp_path):
        video = self._make_video(tmp_path)

        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.return_value = [_make_yolo_result([])]
            mock_yolo_cls.return_value = mock_model

            detector = BorderDetector()
            results  = list(detector.process_video(
                video,
                frame_skip    = 3,
                compute_flow  = True,
                save_annotated= False,
                show_progress = False,
            ))

        assert len(results) > 0
        assert all(isinstance(r, FrameResult) for r in results)

    def test_process_video_frame_ids_increasing(self, tmp_path):
        video = self._make_video(tmp_path, n_frames=20)

        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.return_value = [_make_yolo_result([])]
            mock_yolo_cls.return_value = mock_model

            detector = BorderDetector()
            results  = list(detector.process_video(
                video, frame_skip=2, compute_flow=False, show_progress=False
            ))

        ids = [r.frame_id for r in results]
        assert ids == sorted(ids), "Frame IDs must be in ascending order"

    def test_process_video_saves_annotated_frames(self, tmp_path):
        video      = self._make_video(tmp_path, n_frames=6)
        output_dir = str(tmp_path / "detections")

        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.return_value = [_make_yolo_result([])]
            mock_yolo_cls.return_value = mock_model

            detector = BorderDetector()
            list(detector.process_video(
                video,
                frame_skip    = 1,
                compute_flow  = False,
                save_annotated= True,
                output_dir    = output_dir,
                show_progress = False,
            ))

        import os
        saved = os.listdir(output_dir)
        assert len(saved) == 6, "One annotated frame per processed frame"

    def test_get_stats_after_process_video(self, tmp_path):
        video = self._make_video(tmp_path, n_frames=10)

        with patch("detector.os.path.exists", return_value=True), \
             patch("detector.YOLO") as mock_yolo_cls:
            box        = _make_yolo_box(class_id=0, confidence=0.8)
            mock_model = MagicMock()
            mock_model.return_value = [_make_yolo_result([box])]
            mock_yolo_cls.return_value = mock_model

            detector = BorderDetector()
            results  = list(detector.process_video(
                video, frame_skip=2, compute_flow=False, show_progress=False
            ))

        stats = detector.get_stats(results)
        assert stats["total_frames"]     > 0
        assert stats["total_detections"] > 0
        assert stats["avg_confidence"]   > 0


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
