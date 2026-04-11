"""
Zone-Based Intrusion Detection Module
=======================================

Adds spatial intelligence to the Border Surveillance pipeline by
defining configurable surveillance zones and detecting when objects
enter restricted areas.  Works entirely with normalised coordinates
(0–1) produced by detector.py — no pixel arithmetic needed.

Architecture position:
    detector.py → [THIS MODULE] → pipeline.py → alert_manager.py

Design principles:
    1.  ADDITIVE — this module does not modify any existing code.
        It reads FrameResult dicts and produces ZoneAlert dicts.
    2.  Configurable — zones are defined as simple polygon vertex lists
        in normalised (0–1) coordinates.  Default zones model a typical
        border crossing scenario: border zone (top), buffer zone (middle),
        observation zone (bottom).
    3.  Lightweight — uses ray-casting point-in-polygon test from NumPy;
        no GPU, no heavy dependencies.
    4.  Composable — output ZoneAlert dicts plug directly into anomaly.py
        and pipeline.py for score boosting and dashboard display.

Zone types:
    RESTRICTED  — 🔴 immediate CRITICAL alert on any detection
    BUFFER      — 🟠 HIGH alert — proximity warning
    OBSERVATION — 🟡 MEDIUM alert — activity logging
    SAFE        — 🟢 no alert (default for unzoned areas)

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Zone severity levels (ordered by priority)
ZONE_RESTRICTED  = "RESTRICTED"     # 🔴 Immediate alert
ZONE_BUFFER      = "BUFFER"         # 🟠 Proximity warning
ZONE_OBSERVATION = "OBSERVATION"    # 🟡 Activity logging
ZONE_SAFE        = "SAFE"           # 🟢 Normal area

# Maps zone type → alert priority (aligns with alert_manager.py tiers)
ZONE_PRIORITY_MAP = {
    ZONE_RESTRICTED:  "CRITICAL",
    ZONE_BUFFER:      "HIGH",
    ZONE_OBSERVATION: "MEDIUM",
    ZONE_SAFE:        "LOW",
}

# Classes that are especially alarming in restricted zones
HIGH_THREAT_IN_ZONE = {"person", "military_vehicle", "suspicious_object", "crowd"}

# Default zone definitions — normalised (0–1) coordinates.
# These model a typical top-down border camera view:
#   - Top 25%   → RESTRICTED  (actual border line)
#   - 25–45%    → BUFFER      (approach area)
#   - 45–100%   → OBSERVATION (general surveillance)
DEFAULT_ZONES: Dict[str, dict] = {
    "border_zone": {
        "polygon": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.25), (0.0, 0.25)],
        "level": ZONE_RESTRICTED,
        "label": "Border Zone (Restricted)",
        "color": "#e63946",
    },
    "buffer_zone": {
        "polygon": [(0.0, 0.25), (1.0, 0.25), (1.0, 0.45), (0.0, 0.45)],
        "level": ZONE_BUFFER,
        "label": "Buffer Zone (Proximity)",
        "color": "#f4a261",
    },
    "observation_zone": {
        "polygon": [(0.0, 0.45), (1.0, 0.45), (1.0, 1.0), (0.0, 1.0)],
        "level": ZONE_OBSERVATION,
        "label": "Observation Zone (General)",
        "color": "#2ec4b6",
    },
}


# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting)
# ---------------------------------------------------------------------------

def _point_in_polygon(
    px: float,
    py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Ray-casting algorithm to determine if a point (px, py) is inside
    a closed polygon defined by a list of (x, y) vertices.

    Args:
        px, py:   Point coordinates (normalised 0–1).
        polygon:  List of (x, y) vertex tuples forming a closed polygon.

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# ZoneViolation dataclass
# ---------------------------------------------------------------------------

@dataclass
class ZoneViolation:
    """
    A single zone violation event for one detection in one frame.

    Attributes:
        frame_id:       Source frame number.
        timestamp:      Wall-clock time of detection.
        zone_name:      Name of the violated zone (e.g. "border_zone").
        zone_level:     RESTRICTED / BUFFER / OBSERVATION.
        zone_label:     Human-readable zone label.
        priority:       Mapped alert priority (CRITICAL/HIGH/MEDIUM/LOW).
        class_name:     Detected object class that caused the violation.
        confidence:     Detection confidence.
        center_x:       Normalised x-coordinate of the detection centre.
        center_y:       Normalised y-coordinate of the detection centre.
        is_high_threat: True if the class is especially alarming in this zone.
        reason:         Human-readable violation description.
    """
    frame_id:       int
    timestamp:      float
    zone_name:      str
    zone_level:     str
    zone_label:     str
    priority:       str
    class_name:     str
    confidence:     float
    center_x:       float
    center_y:       float
    is_high_threat: bool  = False
    reason:         str   = ""

    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON / dashboard consumption."""
        return {
            "frame_id":       self.frame_id,
            "timestamp":      round(self.timestamp, 3),
            "zone_name":      self.zone_name,
            "zone_level":     self.zone_level,
            "zone_label":     self.zone_label,
            "priority":       self.priority,
            "class_name":     self.class_name,
            "confidence":     round(self.confidence, 4),
            "center_x":       round(self.center_x, 4),
            "center_y":       round(self.center_y, 4),
            "is_high_threat": self.is_high_threat,
            "reason":         self.reason,
        }


# ---------------------------------------------------------------------------
# ZoneAnalysisResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ZoneAnalysisResult:
    """
    Complete zone analysis output for a single frame.

    Attributes:
        frame_id:        Source frame number.
        violations:      List of ZoneViolation objects for this frame.
        violation_count: Total number of zone violations.
        max_severity:    Highest zone level violated (RESTRICTED > BUFFER > OBSERVATION).
        zone_scores:     Per-zone detection counts: {zone_name: count}.
        risk_score:      Composite zone risk score (0.0–1.0).
        reasons:         Human-readable list of all zone violation reasons.
    """
    frame_id:        int
    violations:      List[ZoneViolation]        = field(default_factory=list)
    violation_count: int                         = 0
    max_severity:    str                         = ZONE_SAFE
    zone_scores:     Dict[str, int]              = field(default_factory=dict)
    risk_score:      float                       = 0.0
    reasons:         List[str]                   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id":        self.frame_id,
            "violation_count": self.violation_count,
            "max_severity":    self.max_severity,
            "zone_scores":     self.zone_scores,
            "risk_score":      round(self.risk_score, 4),
            "reasons":         self.reasons,
            "violations":      [v.to_dict() for v in self.violations],
        }

    @property
    def has_critical(self) -> bool:
        return self.max_severity == ZONE_RESTRICTED

    @property
    def has_violations(self) -> bool:
        return self.violation_count > 0


# ---------------------------------------------------------------------------
# ZoneAnalyzer — main class
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {
    ZONE_SAFE:        0,
    ZONE_OBSERVATION: 1,
    ZONE_BUFFER:      2,
    ZONE_RESTRICTED:  3,
}


class ZoneAnalyzer:
    """
    Spatial intelligence layer for border surveillance.

    Checks every detection from detector.py against configurable
    surveillance zones and produces ZoneAnalysisResult objects
    that pipeline.py feeds into anomaly scoring.

    This module is PURELY ADDITIVE — it reads existing detection
    output and produces new zone-aware metadata without modifying
    any upstream code.

    Args:
        zones:     Dict of zone definitions.  Each zone has:
                   - polygon: list of (x, y) normalised vertices
                   - level: RESTRICTED / BUFFER / OBSERVATION
                   - label: human-readable description
                   - color: hex colour for dashboard rendering
                   If None, uses DEFAULT_ZONES.
        enable_night_boost: When True, boost zone risk during night
                            hours (21:00–05:00 local time).

    Example:
        >>> analyzer = ZoneAnalyzer()
        >>> for frame_result in detector.process_video("feed.mp4"):
        ...     zone_result = analyzer.analyze(frame_result.to_dict())
        ...     if zone_result.has_critical:
        ...         print(f"INTRUSION at frame {zone_result.frame_id}!")
    """

    def __init__(
        self,
        zones: Optional[Dict[str, dict]] = None,
        enable_night_boost: bool = True,
    ) -> None:
        self.zones = zones or DEFAULT_ZONES
        self.enable_night_boost = enable_night_boost

        # Pre-parse polygon vertices for fast lookup
        self._parsed_zones: List[dict] = []
        for name, zdef in self.zones.items():
            self._parsed_zones.append({
                "name":    name,
                "polygon": zdef["polygon"],
                "level":   zdef.get("level",  ZONE_OBSERVATION),
                "label":   zdef.get("label",  name),
                "color":   zdef.get("color",  "#ffffff"),
            })

        # Statistics counters
        self._total_frames_analyzed: int = 0
        self._total_violations:      int = 0
        self._violations_by_zone: Dict[str, int] = {
            z["name"]: 0 for z in self._parsed_zones
        }

        logger.info(
            "ZoneAnalyzer ready | %d zones defined: %s",
            len(self._parsed_zones),
            ", ".join(z["name"] for z in self._parsed_zones),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame_result: dict) -> ZoneAnalysisResult:
        """
        Analyze one frame's detections against all defined zones.

        Args:
            frame_result: Dict from FrameResult.to_dict() — produced
                          by detector.py.

        Returns:
            ZoneAnalysisResult with all violations for this frame.
        """
        frame_id   = frame_result.get("frame_id",   0)
        detections = frame_result.get("detections", [])
        timestamp  = frame_result.get("timestamp",  time.time())

        self._total_frames_analyzed += 1

        violations: List[ZoneViolation] = []
        zone_counts: Dict[str, int] = {z["name"]: 0 for z in self._parsed_zones}

        for det in detections:
            cx = det.get("center_x", 0.5)
            cy = det.get("center_y", 0.5)
            class_name  = det.get("class_name", "unknown")
            confidence  = det.get("confidence",  0.0)

            # Check each zone
            for zdef in self._parsed_zones:
                if _point_in_polygon(cx, cy, zdef["polygon"]):
                    zone_counts[zdef["name"]] += 1

                    # Only generate violation for non-SAFE zones
                    if zdef["level"] != ZONE_SAFE:
                        is_high_threat = class_name in HIGH_THREAT_IN_ZONE
                        priority = ZONE_PRIORITY_MAP.get(
                            zdef["level"], "LOW"
                        )

                        # Boost priority if high-threat class in restricted zone
                        if is_high_threat and zdef["level"] == ZONE_RESTRICTED:
                            priority = "CRITICAL"

                        reason = self._build_violation_reason(
                            class_name, confidence, zdef["label"],
                            zdef["level"], is_high_threat,
                        )

                        violation = ZoneViolation(
                            frame_id       = frame_id,
                            timestamp      = timestamp,
                            zone_name      = zdef["name"],
                            zone_level     = zdef["level"],
                            zone_label     = zdef["label"],
                            priority       = priority,
                            class_name     = class_name,
                            confidence     = confidence,
                            center_x       = cx,
                            center_y       = cy,
                            is_high_threat = is_high_threat,
                            reason         = reason,
                        )
                        violations.append(violation)
                        self._total_violations += 1
                        self._violations_by_zone[zdef["name"]] += 1

                    break  # A detection belongs to at most one zone

        # Compute aggregate severity
        max_severity = ZONE_SAFE
        for v in violations:
            if _SEVERITY_RANK.get(v.zone_level, 0) > \
                    _SEVERITY_RANK.get(max_severity, 0):
                max_severity = v.zone_level

        # Composite risk score (0–1)
        risk_score = self._compute_risk_score(violations, detections)

        # Night boost
        if self.enable_night_boost and self._is_night_time():
            risk_score = min(1.0, risk_score * 1.3)

        reasons = [v.reason for v in violations]

        return ZoneAnalysisResult(
            frame_id        = frame_id,
            violations      = violations,
            violation_count = len(violations),
            max_severity    = max_severity,
            zone_scores     = zone_counts,
            risk_score      = risk_score,
            reasons         = reasons,
        )

    def get_zones(self) -> List[dict]:
        """Return zone definitions for dashboard rendering."""
        return [
            {
                "name":    z["name"],
                "polygon": z["polygon"],
                "level":   z["level"],
                "label":   z["label"],
                "color":   z["color"],
            }
            for z in self._parsed_zones
        ]

    def get_summary(self) -> dict:
        """Return aggregate zone statistics across all analyzed frames."""
        return {
            "total_frames_analyzed": self._total_frames_analyzed,
            "total_violations":      self._total_violations,
            "violations_by_zone":    dict(self._violations_by_zone),
            "zones_defined":         len(self._parsed_zones),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_violation_reason(
        class_name:     str,
        confidence:     float,
        zone_label:     str,
        zone_level:     str,
        is_high_threat: bool,
    ) -> str:
        """Build a human-readable reason string for this violation."""
        threat_tag = " [HIGH THREAT]" if is_high_threat else ""
        return (
            f"{class_name} ({confidence:.0%}) detected in "
            f"{zone_label}{threat_tag}"
        )

    @staticmethod
    def _compute_risk_score(
        violations: List[ZoneViolation],
        detections: list,
    ) -> float:
        """
        Compute a composite risk score from zone violations.

        Scoring:
            - Each RESTRICTED violation       → +0.30
            - Each BUFFER violation            → +0.15
            - Each OBSERVATION violation       → +0.05
            - High-threat class in RESTRICTED  → additional +0.15
            - Multiple violations in same zone → +0.10 per extra

        Returns:
            float in [0.0, 1.0] — higher = more dangerous.
        """
        if not violations:
            return 0.0

        score = 0.0
        zone_hit_count: Dict[str, int] = {}

        for v in violations:
            # Base score by zone level
            if v.zone_level == ZONE_RESTRICTED:
                score += 0.30
            elif v.zone_level == ZONE_BUFFER:
                score += 0.15
            elif v.zone_level == ZONE_OBSERVATION:
                score += 0.05

            # High-threat bonus
            if v.is_high_threat and v.zone_level == ZONE_RESTRICTED:
                score += 0.15

            # Track zone hit counts for crowding penalty
            zone_hit_count[v.zone_name] = \
                zone_hit_count.get(v.zone_name, 0) + 1

        # Crowding penalty — multiple detections in same zone
        for count in zone_hit_count.values():
            if count > 1:
                score += 0.10 * (count - 1)

        return min(1.0, score)

    @staticmethod
    def _is_night_time() -> bool:
        """Return True if current local time is between 21:00 and 05:00."""
        import datetime
        hour = datetime.datetime.now().hour
        return hour >= 21 or hour < 5


# ---------------------------------------------------------------------------
# Quick test (run directly: python src/zone_analyzer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Border Surveillance AI — Zone Analyzer Module")
    print("=" * 55)

    analyzer = ZoneAnalyzer()
    print(f"\nZones defined: {len(analyzer.get_zones())}")
    for z in analyzer.get_zones():
        print(f"  {z['name']:20s} → {z['level']:12s}  {z['label']}")

    # Simulate a frame with detections
    test_frame = {
        "frame_id": 42,
        "timestamp": time.time(),
        "detection_count": 3,
        "detections": [
            {"class_name": "person",   "confidence": 0.85,
             "center_x": 0.5, "center_y": 0.1},   # In border zone
            {"class_name": "vehicle",  "confidence": 0.72,
             "center_x": 0.3, "center_y": 0.35},   # In buffer zone
            {"class_name": "ship",     "confidence": 0.90,
             "center_x": 0.7, "center_y": 0.8},    # In observation zone
        ],
    }

    result = analyzer.analyze(test_frame)

    print(f"\nFrame {result.frame_id} Zone Analysis:")
    print(f"  Violations:    {result.violation_count}")
    print(f"  Max severity:  {result.max_severity}")
    print(f"  Risk score:    {result.risk_score:.4f}")
    print(f"  Has critical:  {result.has_critical}")

    for v in result.violations:
        print(f"  → [{v.priority}] {v.reason}")

    print(f"\nSummary: {analyzer.get_summary()}")
    print("\nNext: python src/temporal_analyzer.py")
