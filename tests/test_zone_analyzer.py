"""
Tests for Zone Analyzer Module
=================================

Tests the zone-based intrusion detection system without requiring
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

from zone_analyzer import (
    ZoneAnalyzer,
    ZoneAnalysisResult,
    ZoneViolation,
    _point_in_polygon,
    ZONE_RESTRICTED,
    ZONE_BUFFER,
    ZONE_OBSERVATION,
    ZONE_SAFE,
    DEFAULT_ZONES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Create a default ZoneAnalyzer."""
    return ZoneAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Create a ZoneAnalyzer with custom zones."""
    zones = {
        "critical_area": {
            "polygon": [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
            "level": ZONE_RESTRICTED,
            "label": "Critical Area",
            "color": "#ff0000",
        },
        "watch_area": {
            "polygon": [(0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5)],
            "level": ZONE_BUFFER,
            "label": "Watch Area",
            "color": "#ff9900",
        },
    }
    return ZoneAnalyzer(zones=zones, enable_night_boost=False)


@pytest.fixture
def border_frame():
    """Frame with a person in the border zone (top area)."""
    return {
        "frame_id": 1,
        "timestamp": time.time(),
        "detection_count": 1,
        "detections": [
            {
                "class_name": "person",
                "confidence": 0.85,
                "center_x": 0.5,
                "center_y": 0.1,  # In border zone (y < 0.25)
                "bbox": [280, 40, 360, 120],
            },
        ],
    }


@pytest.fixture
def buffer_frame():
    """Frame with a vehicle in the buffer zone."""
    return {
        "frame_id": 2,
        "timestamp": time.time(),
        "detection_count": 1,
        "detections": [
            {
                "class_name": "vehicle",
                "confidence": 0.72,
                "center_x": 0.3,
                "center_y": 0.35,  # In buffer zone (0.25–0.45)
                "bbox": [150, 200, 250, 260],
            },
        ],
    }


@pytest.fixture
def empty_frame():
    """Frame with no detections."""
    return {
        "frame_id": 3,
        "timestamp": time.time(),
        "detection_count": 0,
        "detections": [],
    }


@pytest.fixture
def multi_detection_frame():
    """Frame with detections in multiple zones."""
    return {
        "frame_id": 4,
        "timestamp": time.time(),
        "detection_count": 4,
        "detections": [
            {"class_name": "person",           "confidence": 0.90,
             "center_x": 0.5, "center_y": 0.1,
             "bbox": [280, 40, 360, 120]},       # Border zone
            {"class_name": "military_vehicle",  "confidence": 0.75,
             "center_x": 0.3, "center_y": 0.15,
             "bbox": [150, 60, 250, 140]},       # Border zone
            {"class_name": "vehicle",           "confidence": 0.68,
             "center_x": 0.7, "center_y": 0.35,
             "bbox": [400, 200, 500, 260]},      # Buffer zone
            {"class_name": "ship",              "confidence": 0.82,
             "center_x": 0.5, "center_y": 0.7,
             "bbox": [280, 420, 360, 480]},      # Observation zone
        ],
    }


# ---------------------------------------------------------------------------
# Tests: Point-in-polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    """Tests for the ray-casting point-in-polygon algorithm."""

    def test_point_inside_rectangle(self):
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert _point_in_polygon(0.5, 0.5, polygon) is True

    def test_point_outside_rectangle(self):
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert _point_in_polygon(1.5, 0.5, polygon) is False

    def test_point_inside_triangle(self):
        polygon = [(0, 0), (1, 0), (0.5, 1)]
        assert _point_in_polygon(0.5, 0.3, polygon) is True

    def test_point_outside_triangle(self):
        polygon = [(0, 0), (1, 0), (0.5, 1)]
        assert _point_in_polygon(0.9, 0.9, polygon) is False

    def test_border_zone_default(self):
        """Point with y=0.1 should be inside the default border zone."""
        border_poly = DEFAULT_ZONES["border_zone"]["polygon"]
        assert _point_in_polygon(0.5, 0.1, border_poly) is True

    def test_buffer_zone_default(self):
        """Point with y=0.35 should be inside the default buffer zone."""
        buffer_poly = DEFAULT_ZONES["buffer_zone"]["polygon"]
        assert _point_in_polygon(0.5, 0.35, buffer_poly) is True

    def test_observation_zone_default(self):
        """Point with y=0.7 should be inside the default observation zone."""
        obs_poly = DEFAULT_ZONES["observation_zone"]["polygon"]
        assert _point_in_polygon(0.5, 0.7, obs_poly) is True


# ---------------------------------------------------------------------------
# Tests: ZoneAnalyzer initialization
# ---------------------------------------------------------------------------

class TestZoneAnalyzerInit:
    """Tests for ZoneAnalyzer construction."""

    def test_default_zones(self, analyzer):
        assert len(analyzer.get_zones()) == 3

    def test_custom_zones(self, custom_analyzer):
        assert len(custom_analyzer.get_zones()) == 2

    def test_zone_names(self, analyzer):
        names = [z["name"] for z in analyzer.get_zones()]
        assert "border_zone" in names
        assert "buffer_zone" in names
        assert "observation_zone" in names


# ---------------------------------------------------------------------------
# Tests: Zone analysis
# ---------------------------------------------------------------------------

class TestZoneAnalysis:
    """Tests for ZoneAnalyzer.analyze()."""

    def test_border_zone_violation(self, analyzer, border_frame):
        result = analyzer.analyze(border_frame)
        assert result.has_violations
        assert result.violation_count >= 1
        assert result.max_severity == ZONE_RESTRICTED
        assert result.has_critical

    def test_buffer_zone_violation(self, analyzer, buffer_frame):
        result = analyzer.analyze(buffer_frame)
        assert result.has_violations
        assert result.violation_count >= 1
        assert result.max_severity == ZONE_BUFFER

    def test_empty_frame_no_violations(self, analyzer, empty_frame):
        result = analyzer.analyze(empty_frame)
        assert not result.has_violations
        assert result.violation_count == 0
        assert result.max_severity == ZONE_SAFE
        assert result.risk_score == 0.0

    def test_multi_zone_detections(self, analyzer, multi_detection_frame):
        result = analyzer.analyze(multi_detection_frame)
        assert result.has_violations
        assert result.violation_count >= 3  # border + buffer + observation
        assert result.has_critical  # military_vehicle in border zone

    def test_risk_score_range(self, analyzer, border_frame):
        result = analyzer.analyze(border_frame)
        assert 0.0 <= result.risk_score <= 1.0

    def test_high_threat_class_boost(self, analyzer):
        """Person in border zone should be flagged as high threat."""
        frame = {
            "frame_id": 10,
            "timestamp": time.time(),
            "detections": [
                {"class_name": "person", "confidence": 0.85,
                 "center_x": 0.5, "center_y": 0.1,
                 "bbox": [280, 40, 360, 120]},
            ],
        }
        result = analyzer.analyze(frame)
        violations = [v for v in result.violations if v.is_high_threat]
        assert len(violations) >= 1

    def test_reasons_populated(self, analyzer, border_frame):
        result = analyzer.analyze(border_frame)
        assert len(result.reasons) >= 1
        assert "person" in result.reasons[0].lower()

    def test_serialization(self, analyzer, border_frame):
        result = analyzer.analyze(border_frame)
        d = result.to_dict()
        assert "frame_id" in d
        assert "violations" in d
        assert "risk_score" in d
        assert isinstance(d["violations"], list)


# ---------------------------------------------------------------------------
# Tests: ZoneViolation
# ---------------------------------------------------------------------------

class TestZoneViolation:
    """Tests for ZoneViolation dataclass."""

    def test_to_dict(self):
        v = ZoneViolation(
            frame_id      = 1,
            timestamp     = 1234567890.0,
            zone_name     = "border_zone",
            zone_level    = ZONE_RESTRICTED,
            zone_label    = "Border Zone",
            priority      = "CRITICAL",
            class_name    = "person",
            confidence    = 0.85,
            center_x      = 0.5,
            center_y      = 0.1,
            is_high_threat = True,
            reason        = "person detected in Border Zone [HIGH THREAT]",
        )
        d = v.to_dict()
        assert d["frame_id"] == 1
        assert d["priority"] == "CRITICAL"
        assert d["is_high_threat"] is True
        assert isinstance(d["confidence"], float)


# ---------------------------------------------------------------------------
# Tests: Summary
# ---------------------------------------------------------------------------

class TestZoneSummary:
    """Tests for ZoneAnalyzer.get_summary()."""

    def test_initial_summary(self, analyzer):
        s = analyzer.get_summary()
        assert s["total_frames_analyzed"] == 0
        assert s["total_violations"] == 0
        assert s["zones_defined"] == 3

    def test_summary_after_analysis(self, analyzer, border_frame, empty_frame):
        analyzer.analyze(border_frame)
        analyzer.analyze(empty_frame)
        s = analyzer.get_summary()
        assert s["total_frames_analyzed"] == 2
        assert s["total_violations"] >= 1


# ---------------------------------------------------------------------------
# Tests: Custom zones
# ---------------------------------------------------------------------------

class TestCustomZones:
    """Tests with custom zone definitions."""

    def test_detection_in_custom_critical_zone(self, custom_analyzer):
        frame = {
            "frame_id": 1,
            "timestamp": time.time(),
            "detections": [
                {"class_name": "person", "confidence": 0.80,
                 "center_x": 0.25, "center_y": 0.25,
                 "bbox": [100, 100, 200, 200]},
            ],
        }
        result = custom_analyzer.analyze(frame)
        assert result.has_violations
        assert result.has_critical

    def test_detection_in_custom_watch_zone(self, custom_analyzer):
        frame = {
            "frame_id": 2,
            "timestamp": time.time(),
            "detections": [
                {"class_name": "vehicle", "confidence": 0.70,
                 "center_x": 0.75, "center_y": 0.25,
                 "bbox": [400, 100, 500, 200]},
            ],
        }
        result = custom_analyzer.analyze(frame)
        assert result.has_violations
        assert result.max_severity == ZONE_BUFFER
