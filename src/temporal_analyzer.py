"""
Temporal Video Analysis Module
================================

Adds multi-frame intelligence to the Border Surveillance pipeline by
analysing detection patterns ACROSS consecutive frames — not just
within individual frames.

This is what transforms the system from a frame-by-frame detector
into a true *video-aware* surveillance platform.

Architecture position:
    detector.py → [THIS MODULE] → pipeline.py → alert_manager.py
                                  (runs in parallel with anomaly.py)

Design principles:
    1.  ADDITIVE — this module does not modify any existing code.
    2.  Sliding window — maintains a rolling buffer of the last N
        frame results and analyses patterns within that window.
    3.  Lightweight tracking — uses IoU (Intersection over Union)
        matching to associate detections across frames, without
        requiring DeepSORT or any GPU-heavy tracker.
    4.  Five temporal detectors:
            a.  Sudden appearance — empty → many objects
            b.  Crowd buildup    — gradual increase in person count
            c.  Loitering        — same object in same location for too long
            d.  Approach trajectory — objects moving toward border (top)
            e.  Coordinated movement — multiple objects moving together

Temporal features (per window):
    - detection_trend:     slope of detection count over the window
    - motion_trend:        slope of motion score over the window
    - persistence_score:   how many objects stayed in the same zone
    - approach_count:      objects moving toward the border (decreasing y)
    - crowd_buildup_rate:  rate of person count increase

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import logging
import time
from collections import deque
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

# Default sliding window size (frames, not time)
DEFAULT_WINDOW_SIZE: int = 30

# Thresholds for temporal alerts
SUDDEN_APPEARANCE_THRESHOLD:  int   = 5     # ≥5 new detections from 0
CROWD_BUILDUP_RATE_THRESHOLD: float = 0.3   # persons per frame increase
LOITER_FRAMES_THRESHOLD:      int   = 10    # same location for 10+ frames
APPROACH_DISTANCE_THRESHOLD:  float = 0.05  # per-frame movement toward top
COORDINATED_MOVEMENT_MIN:     int   = 3     # ≥3 objects moving together
IOU_MATCH_THRESHOLD:          float = 0.3   # IoU for cross-frame matching

# Alert types
TEMPORAL_SUDDEN_APPEARANCE   = "sudden_appearance"
TEMPORAL_CROWD_BUILDUP       = "crowd_buildup"
TEMPORAL_LOITERING           = "loitering"
TEMPORAL_APPROACH_TRAJECTORY = "approach_trajectory"
TEMPORAL_COORDINATED_MOVE    = "coordinated_movement"


# ---------------------------------------------------------------------------
# IoU utility
# ---------------------------------------------------------------------------

def _compute_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:
    """
    Compute Intersection over Union for two bounding boxes.

    Args:
        box_a, box_b: [x1, y1, x2, y2] format (pixel or normalised).

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ---------------------------------------------------------------------------
# TemporalAlert dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalAlert:
    """
    A single temporal pattern alert.

    Attributes:
        alert_type:   One of the TEMPORAL_* constants.
        severity:     CRITICAL / HIGH / MEDIUM / LOW.
        frame_id:     Frame that triggered the alert.
        timestamp:    Wall-clock time.
        description:  Human-readable explanation.
        details:      Additional data (varies by alert type).
    """
    alert_type:   str
    severity:     str
    frame_id:     int
    timestamp:    float
    description:  str
    details:      Dict   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_type":  self.alert_type,
            "severity":    self.severity,
            "frame_id":    self.frame_id,
            "timestamp":   round(self.timestamp, 3),
            "description": self.description,
            "details":     self.details,
        }


# ---------------------------------------------------------------------------
# TemporalAnalysisResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalAnalysisResult:
    """
    Complete temporal analysis output for the current frame + window.

    Attributes:
        frame_id:           Current frame number.
        window_size:        Number of frames in the analysis window.
        alerts:             List of temporal pattern alerts detected.
        risk_score:         Composite temporal risk score (0.0–1.0).
        detection_trend:    Slope of detection count over the window
                            (positive = increasing threat activity).
        motion_trend:       Slope of motion score over the window.
        tracked_objects:    Number of objects being tracked across frames.
        approaching_count:  Objects moving toward border (top of frame).
        loitering_count:    Objects stationary in one zone for too long.
        reasons:            Human-readable list of temporal concerns.
    """
    frame_id:          int
    window_size:       int                     = 0
    alerts:            List[TemporalAlert]      = field(default_factory=list)
    risk_score:        float                   = 0.0
    detection_trend:   float                   = 0.0
    motion_trend:      float                   = 0.0
    tracked_objects:   int                     = 0
    approaching_count: int                     = 0
    loitering_count:   int                     = 0
    reasons:           List[str]               = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id":          self.frame_id,
            "window_size":       self.window_size,
            "risk_score":        round(self.risk_score, 4),
            "detection_trend":   round(self.detection_trend, 4),
            "motion_trend":      round(self.motion_trend, 4),
            "tracked_objects":   self.tracked_objects,
            "approaching_count": self.approaching_count,
            "loitering_count":   self.loitering_count,
            "alerts":            [a.to_dict() for a in self.alerts],
            "reasons":           self.reasons,
        }

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0


# ---------------------------------------------------------------------------
# Tracked object helper
# ---------------------------------------------------------------------------

@dataclass
class _TrackedObject:
    """Internal tracked object state across frames."""
    track_id:      int
    class_name:    str
    center_x:      float
    center_y:      float
    bbox:          List[float]
    first_frame:   int
    last_frame:    int
    frame_count:   int       = 1
    positions:     List[Tuple[float, float]] = field(default_factory=list)

    @property
    def y_displacement(self) -> float:
        """Net vertical displacement: negative = moving toward top (border)."""
        if len(self.positions) < 2:
            return 0.0
        return self.positions[-1][1] - self.positions[0][1]

    @property
    def is_approaching_border(self) -> bool:
        """True if object is moving upward (toward border zone)."""
        return self.y_displacement < -APPROACH_DISTANCE_THRESHOLD

    @property
    def is_loitering(self) -> bool:
        """True if object stayed in roughly the same spot for too long."""
        if self.frame_count < LOITER_FRAMES_THRESHOLD:
            return False
        if len(self.positions) < 2:
            return False
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        return x_range < 0.08 and y_range < 0.08


# ---------------------------------------------------------------------------
# TemporalAnalyzer — main class
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """
    Multi-frame temporal intelligence layer for border surveillance.

    Maintains a sliding window of recent frame results and analyses
    detection patterns across time.  Detects five types of suspicious
    temporal behaviour:

        1.  Sudden appearance — rapid increase in detections
        2.  Crowd buildup    — gradual person count increase
        3.  Loitering        — objects staying in one spot
        4.  Approach trajectory — objects moving toward border
        5.  Coordinated movement — multiple objects moving together

    This module is PURELY ADDITIVE — it reads existing FrameResult
    dicts and produces TemporalAnalysisResult objects without
    modifying any upstream code.

    Args:
        window_size: Number of frames to keep in the sliding window.

    Example:
        >>> analyzer = TemporalAnalyzer(window_size=30)
        >>> for frame_result in detector.process_video("feed.mp4"):
        ...     temporal = analyzer.analyze(frame_result.to_dict())
        ...     if temporal.has_alerts:
        ...         for alert in temporal.alerts:
        ...             print(f"[{alert.severity}] {alert.description}")
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.window_size = window_size

        # Sliding window of frame results
        self._frame_history: deque = deque(maxlen=window_size)

        # Lightweight object tracker
        self._tracked_objects: Dict[int, _TrackedObject] = {}
        self._next_track_id: int = 0

        # Statistics
        self._total_frames_analyzed: int = 0
        self._total_alerts_raised:   int = 0
        self._alert_type_counts: Dict[str, int] = {
            TEMPORAL_SUDDEN_APPEARANCE:   0,
            TEMPORAL_CROWD_BUILDUP:       0,
            TEMPORAL_LOITERING:           0,
            TEMPORAL_APPROACH_TRAJECTORY: 0,
            TEMPORAL_COORDINATED_MOVE:    0,
        }

        logger.info(
            "TemporalAnalyzer ready | window_size=%d | "
            "IoU_threshold=%.2f | loiter_frames=%d",
            window_size, IOU_MATCH_THRESHOLD, LOITER_FRAMES_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame_result: dict) -> TemporalAnalysisResult:
        """
        Analyze one frame in the context of the recent window.

        Args:
            frame_result: Dict from FrameResult.to_dict().

        Returns:
            TemporalAnalysisResult with temporal alerts and risk score.
        """
        frame_id   = frame_result.get("frame_id",   0)
        detections = frame_result.get("detections", [])
        timestamp  = frame_result.get("timestamp",  time.time())
        motion     = frame_result.get("motion_score")

        self._total_frames_analyzed += 1

        # Add to history
        self._frame_history.append({
            "frame_id":        frame_id,
            "detections":      detections,
            "detection_count": len(detections),
            "motion_score":    motion,
            "timestamp":       timestamp,
        })

        # Update object tracker
        self._update_tracks(frame_id, detections)

        # Run all temporal detectors
        alerts: List[TemporalAlert] = []

        if len(self._frame_history) >= 3:
            alerts += self._check_sudden_appearance(frame_id, timestamp)
            alerts += self._check_crowd_buildup(frame_id, timestamp)
            alerts += self._check_loitering(frame_id, timestamp)
            alerts += self._check_approach_trajectory(frame_id, timestamp)
            alerts += self._check_coordinated_movement(frame_id, timestamp)

        # Update counters
        for a in alerts:
            self._total_alerts_raised += 1
            if a.alert_type in self._alert_type_counts:
                self._alert_type_counts[a.alert_type] += 1

        # Compute trends
        detection_trend = self._compute_detection_trend()
        motion_trend    = self._compute_motion_trend()

        # Object tracking stats
        approaching = sum(
            1 for t in self._tracked_objects.values()
            if t.is_approaching_border
        )
        loitering = sum(
            1 for t in self._tracked_objects.values()
            if t.is_loitering
        )

        # Composite risk score
        risk_score = self._compute_risk_score(
            alerts, detection_trend, motion_trend,
            approaching, loitering,
        )

        reasons = [a.description for a in alerts]

        return TemporalAnalysisResult(
            frame_id          = frame_id,
            window_size       = len(self._frame_history),
            alerts            = alerts,
            risk_score        = risk_score,
            detection_trend   = detection_trend,
            motion_trend      = motion_trend,
            tracked_objects   = len(self._tracked_objects),
            approaching_count = approaching,
            loitering_count   = loitering,
            reasons           = reasons,
        )

    def get_tracks(self) -> List[dict]:
        """Return current tracked objects for dashboard rendering."""
        return [
            {
                "track_id":     t.track_id,
                "class_name":   t.class_name,
                "center_x":     round(t.center_x, 4),
                "center_y":     round(t.center_y, 4),
                "frame_count":  t.frame_count,
                "approaching":  t.is_approaching_border,
                "loitering":    t.is_loitering,
                "positions":    [(round(p[0], 4), round(p[1], 4))
                                for p in t.positions[-10:]],
            }
            for t in self._tracked_objects.values()
        ]

    def get_summary(self) -> dict:
        """Return aggregate temporal statistics."""
        return {
            "total_frames_analyzed":  self._total_frames_analyzed,
            "total_temporal_alerts":  self._total_alerts_raised,
            "alert_type_counts":      dict(self._alert_type_counts),
            "active_tracks":          len(self._tracked_objects),
            "window_size":            self.window_size,
        }

    # ------------------------------------------------------------------
    # Private: Object Tracking (IoU-based)
    # ------------------------------------------------------------------

    def _update_tracks(
        self,
        frame_id:   int,
        detections: list,
    ) -> None:
        """
        Update tracked objects with new detections using IoU matching.

        Simple greedy matching: for each detection, find the tracked
        object with the highest IoU; if above threshold, update that
        track.  Unmatched detections become new tracks.  Tracks not
        seen for 5+ frames are removed.
        """
        matched_tracks: set = set()
        new_detections: list = []

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cx   = det.get("center_x", 0.5)
            cy   = det.get("center_y", 0.5)

            best_iou   = 0.0
            best_track = None

            for tid, track in self._tracked_objects.items():
                if tid in matched_tracks:
                    continue
                iou = _compute_iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou   = iou
                    best_track = tid

            if best_iou >= IOU_MATCH_THRESHOLD and best_track is not None:
                # Update existing track
                track = self._tracked_objects[best_track]
                track.center_x   = cx
                track.center_y   = cy
                track.bbox       = bbox
                track.last_frame = frame_id
                track.frame_count += 1
                track.positions.append((cx, cy))
                # Limit positions history
                if len(track.positions) > 50:
                    track.positions = track.positions[-50:]
                matched_tracks.add(best_track)
            else:
                new_detections.append(det)

        # Create new tracks for unmatched detections
        for det in new_detections:
            tid = self._next_track_id
            self._next_track_id += 1
            cx  = det.get("center_x", 0.5)
            cy  = det.get("center_y", 0.5)
            self._tracked_objects[tid] = _TrackedObject(
                track_id    = tid,
                class_name  = det.get("class_name", "unknown"),
                center_x    = cx,
                center_y    = cy,
                bbox        = det.get("bbox", [0, 0, 0, 0]),
                first_frame = frame_id,
                last_frame  = frame_id,
                positions   = [(cx, cy)],
            )

        # Prune stale tracks (not seen for 5 frames)
        stale = [
            tid for tid, t in self._tracked_objects.items()
            if frame_id - t.last_frame > 5
        ]
        for tid in stale:
            del self._tracked_objects[tid]

    # ------------------------------------------------------------------
    # Private: Temporal Detectors
    # ------------------------------------------------------------------

    def _check_sudden_appearance(
        self, frame_id: int, timestamp: float,
    ) -> List[TemporalAlert]:
        """Detect sudden appearance of many objects from near-zero."""
        alerts = []
        if len(self._frame_history) < 3:
            return alerts

        counts = [h["detection_count"] for h in self._frame_history]

        # Check: was it quiet in the last few frames, then suddenly busy?
        recent_count  = counts[-1]
        prev_avg      = float(np.mean(counts[-5:-1])) if len(counts) > 4 else 0.0
        increase      = recent_count - prev_avg

        if prev_avg < 2 and increase >= SUDDEN_APPEARANCE_THRESHOLD:
            alerts.append(TemporalAlert(
                alert_type  = TEMPORAL_SUDDEN_APPEARANCE,
                severity    = "HIGH",
                frame_id    = frame_id,
                timestamp   = timestamp,
                description = (
                    f"Sudden appearance: {int(increase)} new objects detected "
                    f"(was {prev_avg:.0f}, now {recent_count})"
                ),
                details     = {
                    "previous_avg": round(prev_avg, 1),
                    "current_count": recent_count,
                    "increase": round(increase, 1),
                },
            ))

        return alerts

    def _check_crowd_buildup(
        self, frame_id: int, timestamp: float,
    ) -> List[TemporalAlert]:
        """Detect gradual increase in person count over the window."""
        alerts = []
        if len(self._frame_history) < 8:
            return alerts

        # Count persons across the window
        person_counts = []
        for h in self._frame_history:
            pc = sum(
                1 for d in h["detections"]
                if d.get("class_name") == "person"
            )
            person_counts.append(pc)

        if len(person_counts) < 5:
            return alerts

        # Compute person count trend (slope)
        x     = np.arange(len(person_counts), dtype=np.float64)
        y     = np.array(person_counts, dtype=np.float64)
        if np.std(x) == 0:
            return alerts
        slope = float(np.polyfit(x, y, 1)[0])

        if slope >= CROWD_BUILDUP_RATE_THRESHOLD and person_counts[-1] >= 3:
            alerts.append(TemporalAlert(
                alert_type  = TEMPORAL_CROWD_BUILDUP,
                severity    = "HIGH",
                frame_id    = frame_id,
                timestamp   = timestamp,
                description = (
                    f"Crowd buildup detected: person count rising "
                    f"at {slope:.2f}/frame (now {person_counts[-1]} persons)"
                ),
                details     = {
                    "buildup_rate": round(slope, 4),
                    "current_persons": person_counts[-1],
                    "window_start_persons": person_counts[0],
                },
            ))

        return alerts

    def _check_loitering(
        self, frame_id: int, timestamp: float,
    ) -> List[TemporalAlert]:
        """Detect objects that have stayed in the same location too long."""
        alerts = []

        for track in self._tracked_objects.values():
            if track.is_loitering:
                alerts.append(TemporalAlert(
                    alert_type  = TEMPORAL_LOITERING,
                    severity    = "MEDIUM",
                    frame_id    = frame_id,
                    timestamp   = timestamp,
                    description = (
                        f"Loitering detected: {track.class_name} "
                        f"(track #{track.track_id}) stationary for "
                        f"{track.frame_count} frames at "
                        f"({track.center_x:.2f}, {track.center_y:.2f})"
                    ),
                    details     = {
                        "track_id":    track.track_id,
                        "class_name":  track.class_name,
                        "frame_count": track.frame_count,
                        "center_x":    round(track.center_x, 4),
                        "center_y":    round(track.center_y, 4),
                    },
                ))

        return alerts

    def _check_approach_trajectory(
        self, frame_id: int, timestamp: float,
    ) -> List[TemporalAlert]:
        """Detect objects moving toward the border (upward in the frame)."""
        alerts = []

        approaching = [
            t for t in self._tracked_objects.values()
            if t.is_approaching_border and t.frame_count >= 3
        ]

        if approaching:
            class_names = [t.class_name for t in approaching]
            alerts.append(TemporalAlert(
                alert_type  = TEMPORAL_APPROACH_TRAJECTORY,
                severity    = "CRITICAL" if any(
                    c in {"person", "military_vehicle", "suspicious_object"}
                    for c in class_names
                ) else "HIGH",
                frame_id    = frame_id,
                timestamp   = timestamp,
                description = (
                    f"Border approach detected: {len(approaching)} object(s) "
                    f"moving toward border zone — "
                    f"{', '.join(set(class_names))}"
                ),
                details     = {
                    "approaching_count": len(approaching),
                    "classes": list(set(class_names)),
                    "tracks": [
                        {
                            "track_id":      t.track_id,
                            "class_name":    t.class_name,
                            "y_displacement": round(t.y_displacement, 4),
                        }
                        for t in approaching
                    ],
                },
            ))

        return alerts

    def _check_coordinated_movement(
        self, frame_id: int, timestamp: float,
    ) -> List[TemporalAlert]:
        """Detect multiple objects moving together in the same direction."""
        alerts = []

        # Need enough tracked objects
        active_tracks = [
            t for t in self._tracked_objects.values()
            if t.frame_count >= 3 and len(t.positions) >= 3
        ]

        if len(active_tracks) < COORDINATED_MOVEMENT_MIN:
            return alerts

        # Compute direction vectors for each track
        directions: List[Tuple[float, float]] = []
        for t in active_tracks:
            dx = t.positions[-1][0] - t.positions[-3][0]
            dy = t.positions[-1][1] - t.positions[-3][1]
            mag = (dx**2 + dy**2) ** 0.5
            if mag > 0.01:  # Moving objects only
                directions.append((dx / mag, dy / mag))
            else:
                directions.append((0.0, 0.0))

        # Count objects moving in similar directions (cosine similarity)
        moving_count = sum(
            1 for d in directions if (d[0]**2 + d[1]**2) > 0.01
        )

        if moving_count < COORDINATED_MOVEMENT_MIN:
            return alerts

        # Check if majority move in the same direction
        moving_dirs = [d for d in directions if (d[0]**2 + d[1]**2) > 0.01]
        if len(moving_dirs) < 2:
            return alerts

        # Compare using dot product (cosine similarity)
        ref = moving_dirs[0]
        same_direction = sum(
            1 for d in moving_dirs[1:]
            if ref[0] * d[0] + ref[1] * d[1] > 0.5  # cos > 0.5 → similar
        )

        if same_direction + 1 >= COORDINATED_MOVEMENT_MIN:
            alerts.append(TemporalAlert(
                alert_type  = TEMPORAL_COORDINATED_MOVE,
                severity    = "HIGH",
                frame_id    = frame_id,
                timestamp   = timestamp,
                description = (
                    f"Coordinated movement: {same_direction + 1} objects "
                    f"moving in the same direction"
                ),
                details     = {
                    "coordinated_count": same_direction + 1,
                    "total_moving": moving_count,
                },
            ))

        return alerts

    # ------------------------------------------------------------------
    # Private: Trend computation
    # ------------------------------------------------------------------

    def _compute_detection_trend(self) -> float:
        """Compute detection count trend (linear slope) over the window."""
        if len(self._frame_history) < 5:
            return 0.0
        counts = [h["detection_count"] for h in self._frame_history]
        x      = np.arange(len(counts), dtype=np.float64)
        y      = np.array(counts, dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    def _compute_motion_trend(self) -> float:
        """Compute motion score trend (linear slope) over the window."""
        if len(self._frame_history) < 5:
            return 0.0
        scores = [
            h["motion_score"] for h in self._frame_history
            if h["motion_score"] is not None
        ]
        if len(scores) < 5:
            return 0.0
        x = np.arange(len(scores), dtype=np.float64)
        y = np.array(scores, dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    # ------------------------------------------------------------------
    # Private: Risk scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_score(
        alerts:          List[TemporalAlert],
        detection_trend: float,
        motion_trend:    float,
        approaching:     int,
        loitering:       int,
    ) -> float:
        """
        Composite temporal risk score (0–1).

        Scoring:
            - Each CRITICAL alert  → +0.30
            - Each HIGH alert      → +0.20
            - Each MEDIUM alert    → +0.10
            - Positive detection trend → up to +0.10
            - Positive motion trend → up to +0.10
            - Each approaching object → +0.10
            - Each loitering object → +0.05
        """
        score = 0.0

        for a in alerts:
            if a.severity == "CRITICAL":
                score += 0.30
            elif a.severity == "HIGH":
                score += 0.20
            elif a.severity == "MEDIUM":
                score += 0.10

        # Trend contributions
        if detection_trend > 0:
            score += min(0.10, detection_trend * 0.05)
        if motion_trend > 0:
            score += min(0.10, motion_trend * 0.02)

        # Tracking contributions
        score += approaching * 0.10
        score += loitering * 0.05

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Quick test (run directly: python src/temporal_analyzer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Border Surveillance AI — Temporal Analyzer Module")
    print("=" * 55)

    analyzer = TemporalAnalyzer(window_size=15)

    # Simulate a sequence of frames with increasing person count
    print("\nSimulating 20-frame sequence with crowd buildup...")
    for i in range(1, 21):
        person_count = min(i // 3, 6)  # Gradual increase
        detections = []
        for p in range(person_count):
            detections.append({
                "class_name": "person",
                "confidence": 0.80,
                "center_x":   0.2 + p * 0.1,
                "center_y":   0.3 - i * 0.005,  # Moving upward (approaching)
                "bbox":       [100 + p * 50, 200 - i * 3, 140 + p * 50, 300 - i * 3],
            })

        frame_result = {
            "frame_id":        i * 3,
            "timestamp":       time.time() + i * 0.1,
            "detection_count": len(detections),
            "detections":      detections,
            "motion_score":    float(i * 0.5),
        }

        result = analyzer.analyze(frame_result)

        if result.has_alerts:
            for alert in result.alerts:
                print(f"  Frame {result.frame_id:3d} [{alert.severity:8s}] "
                      f"{alert.description}")

    print(f"\nSummary: {analyzer.get_summary()}")
    print(f"\nActive tracks: {len(analyzer.get_tracks())}")
    for t in analyzer.get_tracks()[:5]:
        print(f"  Track #{t['track_id']}: {t['class_name']} | "
              f"frames={t['frame_count']} | "
              f"approaching={t['approaching']} | "
              f"loitering={t['loitering']}")

    print("\nNext: python src/pipeline.py")
