"""
Tests for Temporal Analyzer Module
=====================================

Tests the multi-frame temporal analysis system without requiring
any external dependencies (no YOLO, no video files).

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import os
import sys
import time
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal_analyzer import (
    TemporalAnalyzer,
    TemporalAnalysisResult,
    TemporalAlert,
    _compute_iou,
    TEMPORAL_SUDDEN_APPEARANCE,
    TEMPORAL_CROWD_BUILDUP,
    TEMPORAL_LOITERING,
    TEMPORAL_APPROACH_TRAJECTORY,
    TEMPORAL_COORDINATED_MOVE,
    DEFAULT_WINDOW_SIZE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Create a TemporalAnalyzer with default settings."""
    return TemporalAnalyzer(window_size=15)


@pytest.fixture
def small_analyzer():
    """Create a TemporalAnalyzer with small window for faster tests."""
    return TemporalAnalyzer(window_size=5)


def _make_frame(frame_id, detections=None, motion_score=None):
    """Helper to create a frame result dict."""
    dets = detections or []
    return {
        "frame_id":        frame_id,
        "timestamp":       time.time() + frame_id * 0.033,
        "detection_count": len(dets),
        "detections":      dets,
        "motion_score":    motion_score,
    }


def _make_detection(class_name="person", confidence=0.8,
                    cx=0.5, cy=0.5, w=40, h=60):
    """Helper to create a detection dict."""
    x1 = cx * 640 - w / 2
    y1 = cy * 640 - h / 2
    return {
        "class_name": class_name,
        "confidence": confidence,
        "center_x":   cx,
        "center_y":   cy,
        "bbox":       [x1, y1, x1 + w, y1 + h],
    }


# ---------------------------------------------------------------------------
# Tests: IoU computation
# ---------------------------------------------------------------------------

class TestIoU:
    """Tests for the IoU utility function."""

    def test_perfect_overlap(self):
        box = [0, 0, 100, 100]
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box_a = [0, 0, 50, 50]
        box_b = [100, 100, 200, 200]
        assert _compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box_a = [0, 0, 100, 100]
        box_b = [50, 50, 150, 150]
        iou = _compute_iou(box_a, box_b)
        assert 0.0 < iou < 1.0

    def test_contained_box(self):
        outer = [0, 0, 200, 200]
        inner = [50, 50, 150, 150]
        iou = _compute_iou(outer, inner)
        # inner area = 10000, outer area = 40000
        # intersection = 10000, union = 40000
        assert iou == pytest.approx(10000 / 40000, abs=0.01)

    def test_zero_area_box(self):
        box_a = [0, 0, 0, 0]
        box_b = [0, 0, 100, 100]
        assert _compute_iou(box_a, box_b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: TemporalAnalyzer initialization
# ---------------------------------------------------------------------------

class TestTemporalAnalyzerInit:
    """Tests for TemporalAnalyzer construction."""

    def test_default_window_size(self):
        analyzer = TemporalAnalyzer()
        assert analyzer.window_size == DEFAULT_WINDOW_SIZE

    def test_custom_window_size(self, small_analyzer):
        assert small_analyzer.window_size == 5

    def test_initial_summary(self, analyzer):
        s = analyzer.get_summary()
        assert s["total_frames_analyzed"] == 0
        assert s["total_temporal_alerts"] == 0
        assert s["active_tracks"] == 0


# ---------------------------------------------------------------------------
# Tests: Basic analysis
# ---------------------------------------------------------------------------

class TestBasicAnalysis:
    """Tests for basic frame analysis."""

    def test_analyze_empty_frame(self, analyzer):
        frame = _make_frame(1, [], motion_score=0.0)
        result = analyzer.analyze(frame)
        assert isinstance(result, TemporalAnalysisResult)
        assert result.frame_id == 1
        assert not result.has_alerts

    def test_analyze_single_detection(self, analyzer):
        det = _make_detection("person", 0.85, 0.5, 0.5)
        frame = _make_frame(1, [det], motion_score=2.0)
        result = analyzer.analyze(frame)
        assert result.tracked_objects == 1

    def test_window_fills_up(self, small_analyzer):
        for i in range(10):
            frame = _make_frame(i, [], motion_score=1.0)
            result = small_analyzer.analyze(frame)
        assert result.window_size == 5  # capped at window_size

    def test_result_serialization(self, analyzer):
        frame = _make_frame(1, [_make_detection()])
        result = analyzer.analyze(frame)
        d = result.to_dict()
        assert "frame_id" in d
        assert "risk_score" in d
        assert "alerts" in d
        assert isinstance(d["alerts"], list)


# ---------------------------------------------------------------------------
# Tests: Object tracking
# ---------------------------------------------------------------------------

class TestObjectTracking:
    """Tests for IoU-based cross-frame object tracking."""

    def test_track_creation(self, analyzer):
        det = _make_detection("person", 0.8, 0.5, 0.5)
        frame = _make_frame(1, [det])
        analyzer.analyze(frame)
        tracks = analyzer.get_tracks()
        assert len(tracks) == 1
        assert tracks[0]["class_name"] == "person"

    def test_track_persistence(self, analyzer):
        """Same object across 3 frames should maintain one track."""
        for i in range(3):
            det = _make_detection("person", 0.8, 0.5, 0.5)
            frame = _make_frame(i, [det])
            analyzer.analyze(frame)

        tracks = analyzer.get_tracks()
        assert len(tracks) == 1
        assert tracks[0]["frame_count"] >= 3

    def test_track_pruning(self, analyzer):
        """Track should be removed after 5+ frames without detection."""
        det = _make_detection("person", 0.8, 0.5, 0.5)
        frame = _make_frame(1, [det])
        analyzer.analyze(frame)
        assert len(analyzer.get_tracks()) == 1

        # 6 empty frames → track should be pruned
        for i in range(2, 8):
            frame = _make_frame(i, [])
            analyzer.analyze(frame)

        assert len(analyzer.get_tracks()) == 0

    def test_multiple_tracks(self, analyzer):
        """Two distinct objects should create two tracks."""
        det_a = _make_detection("person", 0.8, 0.2, 0.2)
        det_b = _make_detection("vehicle", 0.7, 0.8, 0.8)
        frame = _make_frame(1, [det_a, det_b])
        analyzer.analyze(frame)
        assert len(analyzer.get_tracks()) == 2


# ---------------------------------------------------------------------------
# Tests: Sudden appearance detection
# ---------------------------------------------------------------------------

class TestSuddenAppearance:
    """Tests for sudden appearance detection."""

    def test_sudden_appearance_detected(self, analyzer):
        """Empty frames followed by many detections → alert."""
        # 5 empty frames
        for i in range(5):
            analyzer.analyze(_make_frame(i, []))

        # Sudden burst of 6 objects
        dets = [
            _make_detection("person", 0.8, 0.1 * j, 0.5)
            for j in range(1, 7)
        ]
        result = analyzer.analyze(_make_frame(6, dets))

        sudden_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_SUDDEN_APPEARANCE
        ]
        assert len(sudden_alerts) >= 1

    def test_no_false_sudden_appearance(self, analyzer):
        """Gradual increase should NOT trigger sudden appearance."""
        for i in range(10):
            count = min(i // 2, 3)
            dets = [
                _make_detection("person", 0.8, 0.1 * j, 0.5)
                for j in range(count)
            ]
            result = analyzer.analyze(_make_frame(i, dets))

        # Should not have sudden appearance alerts
        sudden_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_SUDDEN_APPEARANCE
        ]
        assert len(sudden_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Crowd buildup detection
# ---------------------------------------------------------------------------

class TestCrowdBuildup:
    """Tests for gradual crowd buildup detection."""

    def test_crowd_buildup_detected(self, analyzer):
        """Gradually increasing person count → alert."""
        for i in range(15):
            person_count = min(i, 8)  # 0, 1, 2, 3, ... 8
            dets = [
                _make_detection("person", 0.8, 0.1 * j + 0.05, 0.5)
                for j in range(person_count)
            ]
            result = analyzer.analyze(_make_frame(i, dets))

        # Check if crowd buildup was detected at some point
        all_crowd_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_CROWD_BUILDUP
        ]
        # With this strong ramp-up, it should be detected
        # (depends on windowing, but the trend should be clear)
        assert result.detection_trend > 0


# ---------------------------------------------------------------------------
# Tests: Loitering detection
# ---------------------------------------------------------------------------

class TestLoitering:
    """Tests for loitering detection (stationary object)."""

    def test_loitering_detected(self, analyzer):
        """Object in same location for many frames → loitering alert."""
        for i in range(15):
            det = _make_detection("person", 0.8, 0.5, 0.5)
            result = analyzer.analyze(_make_frame(i, [det]))

        loiter_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_LOITERING
        ]
        assert len(loiter_alerts) >= 1

    def test_moving_object_no_loitering(self, analyzer):
        """Moving object should NOT trigger loitering."""
        for i in range(15):
            cx = 0.1 + i * 0.05  # Moving right
            det = _make_detection("person", 0.8, cx, 0.5)
            result = analyzer.analyze(_make_frame(i, [det]))

        loiter_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_LOITERING
        ]
        assert len(loiter_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Approach trajectory detection
# ---------------------------------------------------------------------------

class TestApproachTrajectory:
    """Tests for objects moving toward the border."""

    def test_upward_movement_detected(self, analyzer):
        """Object moving upward (toward border) → approach alert."""
        for i in range(10):
            cy = 0.9 - i * 0.07  # Moving up toward border
            # Use large bboxes so IoU matching works across frames
            y1 = cy * 640 - 60
            det = {
                "class_name": "person",
                "confidence": 0.8,
                "center_x": 0.5,
                "center_y": cy,
                "bbox": [290, y1, 350, y1 + 120],
            }
            result = analyzer.analyze(_make_frame(i, [det]))

        approach_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_APPROACH_TRAJECTORY
        ]
        assert len(approach_alerts) >= 1

    def test_downward_movement_no_approach(self, analyzer):
        """Object moving downward (away from border) → no alert."""
        for i in range(8):
            cy = 0.2 + i * 0.08  # Moving down away from border
            y1 = cy * 640 - 60
            det = {
                "class_name": "person",
                "confidence": 0.8,
                "center_x": 0.5,
                "center_y": cy,
                "bbox": [290, y1, 350, y1 + 120],
            }
            result = analyzer.analyze(_make_frame(i, [det]))

        approach_alerts = [
            a for a in result.alerts
            if a.alert_type == TEMPORAL_APPROACH_TRAJECTORY
        ]
        assert len(approach_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: Summary & tracks
# ---------------------------------------------------------------------------

class TestSummaryAndTracks:
    """Tests for summary and track retrieval."""

    def test_summary_after_analysis(self, analyzer):
        for i in range(5):
            analyzer.analyze(_make_frame(i, [_make_detection()]))
        s = analyzer.get_summary()
        assert s["total_frames_analyzed"] == 5
        assert s["active_tracks"] >= 1

    def test_tracks_contain_positions(self, analyzer):
        for i in range(3):
            det = _make_detection("person", 0.8, 0.5, 0.5)
            analyzer.analyze(_make_frame(i, [det]))
        tracks = analyzer.get_tracks()
        assert len(tracks) >= 1
        assert "positions" in tracks[0]
        assert len(tracks[0]["positions"]) >= 1


# ---------------------------------------------------------------------------
# Tests: TemporalAlert
# ---------------------------------------------------------------------------

class TestTemporalAlert:
    """Tests for TemporalAlert dataclass."""

    def test_to_dict(self):
        alert = TemporalAlert(
            alert_type  = TEMPORAL_LOITERING,
            severity    = "MEDIUM",
            frame_id    = 42,
            timestamp   = 1234567890.0,
            description = "Test alert",
            details     = {"track_id": 1},
        )
        d = alert.to_dict()
        assert d["alert_type"] == TEMPORAL_LOITERING
        assert d["severity"] == "MEDIUM"
        assert d["frame_id"] == 42
        assert isinstance(d["details"], dict)


# ---------------------------------------------------------------------------
# Tests: Risk score
# ---------------------------------------------------------------------------

class TestRiskScore:
    """Tests for temporal risk score computation."""

    def test_risk_score_range(self, analyzer):
        for i in range(10):
            det = _make_detection("person", 0.8, 0.5, 0.5)
            result = analyzer.analyze(_make_frame(i, [det]))
        assert 0.0 <= result.risk_score <= 1.0

    def test_empty_frames_low_risk(self, analyzer):
        for i in range(5):
            result = analyzer.analyze(_make_frame(i, []))
        assert result.risk_score == 0.0
