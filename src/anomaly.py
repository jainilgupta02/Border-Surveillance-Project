"""
Anomaly Detection Module
=========================

Detects suspicious behavioural patterns in border surveillance footage by
analysing the *distribution* of detections across frames — not just whether
an object is present, but whether its position, size, motion, frequency, and
class combination are unusual for this video stream.

Architecture position:
    preprocessing.py → detector.py → [THIS MODULE] → alert_manager.py → dashboard

Design principles:
    1. Separation of concerns — this module knows nothing about video or YOLO.
       It only consumes FrameResult dicts and produces AnomalyResult dicts.
    2. Two-phase operation:
         Phase A (fit)   — observe N "baseline" frames, learn what normal looks like.
         Phase B (score) — score every incoming frame against that baseline.
    3. Persistence — the fitted model saves to disk so pipeline.py doesn't
       re-fit on every run.
    4. Explainability — every anomaly carries a human-readable reason so the
       dashboard and alert emails are meaningful.
    5. Graceful degradation — if fewer than MIN_SAMPLES frames are available
       before fit() is called, the module falls back to rule-based scoring so
       the pipeline never crashes.

Anomaly features (per frame — 10 dimensions):
    0  detection_count       how many objects in the frame
    1  class_diversity       number of distinct classes present
    2  avg_confidence        mean detection confidence
    3  max_confidence        highest single detection confidence
    4  critical_count        crowd + military_vehicle + suspicious_object count
    5  avg_center_x          spatial centre of all detections (normalised 0-1)
    6  avg_center_y          spatial centre of all detections (normalised 0-1)
    7  avg_width_norm        average normalised bounding box width
    8  motion_score          optical flow mean magnitude (0 if unavailable)
    9  suspicious_flag       1.0 if any CRITICAL_THREAT_CLASS present, else 0.0

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum frames needed before Isolation Forest can be fitted
MIN_SAMPLES: int = 30

# Isolation Forest contamination — expected fraction of anomalous frames.
# Set conservatively (0.08 ≈ 8%) to avoid over-alerting on short videos.
CONTAMINATION: float = 0.08

# Anomaly score thresholds (Isolation Forest decision_function output).
# More negative = more anomalous.  Range is roughly [-0.5, +0.2].
THRESHOLD_HIGH:     float = -0.08   # score below this → HIGH alert
THRESHOLD_CRITICAL: float = -0.18   # score below this → CRITICAL alert

# Default model persistence path
DEFAULT_MODEL_PATH: str = "models/anomaly_model.pkl"

# Threat class sets — drive both rule-based scoring and class boosts
HIGH_THREAT_CLASSES     = {"crowd", "aircraft"}
CRITICAL_THREAT_CLASSES = {"military_vehicle", "suspicious_object"}

# Feature vector dimension — must match extract_features() output length
FEATURE_DIM: int = 10


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(frame_result: dict) -> np.ndarray:
    """
    Convert a FrameResult dict into a fixed-length feature vector for the
    Isolation Forest.

    Args:
        frame_result: Dict from FrameResult.to_dict() — produced by detector.py.

    Returns:
        np.ndarray of shape (FEATURE_DIM,) = (10,), dtype float32.

    Note:
        All features are deliberately low-dimensional and interpretable.
        High-dimensional pixel features are NOT used — the Isolation Forest
        works on behavioural signals, not raw images.
    """
    detections      = frame_result.get("detections", [])
    motion_score    = frame_result.get("motion_score") or 0.0
    detection_count = len(detections)

    if detection_count == 0:
        # Empty frame — zeros except motion and neutral spatial centre
        return np.array([
            0.0,                    # 0: detection_count
            0.0,                    # 1: class_diversity
            0.0,                    # 2: avg_confidence
            0.0,                    # 3: max_confidence
            0.0,                    # 4: critical_count
            0.5,                    # 5: avg_center_x  (neutral centre)
            0.5,                    # 6: avg_center_y
            0.0,                    # 7: avg_width_norm
            float(motion_score),    # 8: motion_score
            0.0,                    # 9: suspicious_flag
        ], dtype=np.float32)

    class_names = [d.get("class_name", "") for d in detections]
    confidences = [d.get("confidence",  0.0) for d in detections]
    center_xs   = [d.get("center_x",    0.5) for d in detections]
    center_ys   = [d.get("center_y",    0.5) for d in detections]
    widths      = [d.get("width_norm",  0.0) for d in detections]

    critical_count  = sum(
        1 for c in class_names
        if c in HIGH_THREAT_CLASSES | CRITICAL_THREAT_CLASSES
    )
    suspicious_flag = 1.0 if any(
        c in CRITICAL_THREAT_CLASSES for c in class_names
    ) else 0.0

    return np.array([
        float(detection_count),
        float(len(set(class_names))),           # 1: class_diversity
        float(np.mean(confidences)),            # 2: avg_confidence
        float(np.max(confidences)),             # 3: max_confidence
        float(critical_count),                  # 4: critical_count
        float(np.mean(center_xs)),              # 5: avg_center_x
        float(np.mean(center_ys)),              # 6: avg_center_y
        float(np.mean(widths)),                 # 7: avg_width_norm
        float(motion_score),                    # 8: motion_score
        suspicious_flag,                        # 9: suspicious_flag
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# AnomalyResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Anomaly assessment for a single frame.

    Attributes:
        frame_id:        Matches FrameResult.frame_id.
        timestamp:       Wall-clock detection time (seconds since epoch).
        anomaly_score:   Raw Isolation Forest score.  More negative = worse.
                         Range roughly [-0.5, +0.2].
        anomaly_prob:    Normalised probability in [0, 1].  1 = most anomalous.
        alert_level:     "normal" | "high" | "critical"
        reasons:         Human-readable list explaining why this frame was flagged.
        detection_count: Number of objects detected in this frame.
        motion_score:    Optical flow score from preprocessing.py.
        features:        Raw 10-dim feature vector (for dashboard charting).
    """
    frame_id:        int
    timestamp:       float
    anomaly_score:   float
    anomaly_prob:    float
    alert_level:     str              # "normal" | "high" | "critical"
    reasons:         List[str]
    detection_count: int              = 0
    motion_score:    Optional[float]  = None
    features:        List[float]      = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to plain dict for alert_manager.py / Cosmos DB."""
        return {
            "frame_id":        self.frame_id,
            "timestamp":       round(self.timestamp, 3),
            "anomaly_score":   round(self.anomaly_score, 4),
            "anomaly_prob":    round(self.anomaly_prob,  4),
            "alert_level":     self.alert_level,
            "reasons":         self.reasons,
            "detection_count": self.detection_count,
            "motion_score":    round(self.motion_score, 4)
                               if self.motion_score is not None else None,
        }

    @property
    def is_alert(self) -> bool:
        """True if this frame warrants an alert (high or critical)."""
        return self.alert_level in ("high", "critical")


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Isolation Forest wrapper for frame-level behavioural anomaly detection.

    Usage (pipeline.py calls it like this):

        detector  = BorderDetector()
        anomaly   = AnomalyDetector()
        baseline  = []
        results   = []

        for frame_item in extract_frames(video):
            frame_result = detector.detect(frame_item)
            fr_dict      = frame_result.to_dict()

            if len(baseline) < MIN_SAMPLES:
                baseline.append(fr_dict)
                if anomaly.collect_baseline(fr_dict):
                    anomaly.fit()
                continue

            anomaly_result = anomaly.score(fr_dict)
            results.append(anomaly_result)
            if anomaly_result.is_alert:
                alert_manager.process(anomaly_result)

    Args:
        contamination: Expected fraction of anomalous frames (0.0–0.5).
        model_path:    Where to save / load the fitted model.
        random_state:  Fixed seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = CONTAMINATION,
        model_path:    str   = DEFAULT_MODEL_PATH,
        random_state:  int   = 42,
    ) -> None:
        self.contamination = contamination
        self.model_path    = model_path
        self.random_state  = random_state

        # BUG FIX: initialise _scaler to None so _ml_score never raises
        # AttributeError even if _load_model() fails or isn't called.
        self._model:          object           = None
        self._scaler:         object           = None
        self._is_fitted:      bool             = False
        self._baseline_data:  List[np.ndarray] = []

        # Try loading a previously saved model
        if os.path.exists(model_path):
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_baseline(self, frame_result: dict) -> bool:
        """
        Buffer one frame result for later fitting.

        Call this for each of the first N frames, then call fit() when
        this method returns True.

        Args:
            frame_result: Dict from FrameResult.to_dict().

        Returns:
            bool: True when MIN_SAMPLES frames have been collected.
        """
        features = extract_features(frame_result)
        self._baseline_data.append(features)
        return len(self._baseline_data) >= MIN_SAMPLES

    def fit(self, frame_results: Optional[List[dict]] = None) -> None:
        """
        Fit the Isolation Forest on baseline frame data.

        BUG FIX: frame_results are converted to features and REPLACE
        _baseline_data — they do NOT append to it.  This prevents
        double-counting when collect_baseline() has already been called.

        Args:
            frame_results: Optional list of FrameResult dicts.  When
                           provided, these replace any data already
                           buffered by collect_baseline().  When None,
                           uses data buffered by collect_baseline().

        Raises:
            ValueError: If fewer than MIN_SAMPLES frames are available
                        after merging both sources.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # BUG FIX: if frame_results supplied, use ONLY those (don't merge)
        if frame_results is not None:
            self._baseline_data = [extract_features(fr) for fr in frame_results]

        n = len(self._baseline_data)
        if n < MIN_SAMPLES:
            logger.warning(
                "Only %d baseline frames — minimum is %d.  "
                "Falling back to rule-based scoring.",
                n, MIN_SAMPLES,
            )
            self._is_fitted = False
            return

        X = np.vstack(self._baseline_data)

        self._scaler  = StandardScaler()
        X_scaled      = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            contamination = self.contamination,
            n_estimators  = 200,
            max_samples   = "auto",
            random_state  = self.random_state,
            n_jobs        = -1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True

        logger.info(
            "Isolation Forest fitted on %d baseline frames  |  "
            "contamination=%.2f  |  estimators=200",
            n, self.contamination,
        )
        self._save_model()

    def score(self, frame_result: dict) -> AnomalyResult:
        """
        Score a single frame for anomalous behaviour.

        Uses Isolation Forest if fitted, otherwise falls back to rule-based
        scoring using threat class flags so the pipeline never crashes.

        Args:
            frame_result: Dict from FrameResult.to_dict().

        Returns:
            AnomalyResult with alert_level and human-readable reasons.
        """
        features    = extract_features(frame_result)
        frame_id    = frame_result.get("frame_id",        0)
        timestamp   = frame_result.get("timestamp",       time.time())
        motion      = frame_result.get("motion_score")
        det_count   = frame_result.get("detection_count", 0)
        detections  = frame_result.get("detections",      [])

        if self._is_fitted:
            anomaly_score, anomaly_prob = self._ml_score(features)
        else:
            anomaly_score, anomaly_prob = self._rule_score(features, detections)

        # Hard boosts for critical classes — ML model cannot suppress these
        anomaly_score, boost_reasons = self._apply_class_boost(
            anomaly_score, detections, motion
        )

        # Re-compute probability after boost
        anomaly_prob = float(1.0 / (1.0 + np.exp(anomaly_score * 10)))
        anomaly_prob = max(0.0, min(1.0, anomaly_prob))

        # Determine alert level
        if anomaly_score < THRESHOLD_CRITICAL:
            alert_level = "critical"
        elif anomaly_score < THRESHOLD_HIGH:
            alert_level = "high"
        else:
            alert_level = "normal"

        reasons = self._build_reasons(
            alert_level, features, detections, motion, boost_reasons
        )

        return AnomalyResult(
            frame_id        = frame_id,
            timestamp       = timestamp,
            anomaly_score   = anomaly_score,
            anomaly_prob    = anomaly_prob,
            alert_level     = alert_level,
            reasons         = reasons,
            detection_count = det_count,
            motion_score    = motion,
            features        = features.tolist(),
        )

    def score_batch(self, frame_results: List[dict]) -> List[AnomalyResult]:
        """
        Score a list of frame results.  Convenience wrapper around score().

        Args:
            frame_results: List of FrameResult dicts.

        Returns:
            List of AnomalyResult objects in the same order.
        """
        return [self.score(fr) for fr in frame_results]

    def get_summary(self, results: List[AnomalyResult]) -> dict:
        """
        Aggregate anomaly statistics across a video session.

        Args:
            results: List of AnomalyResult objects.

        Returns:
            Summary dict for Cosmos DB / dashboard.
        """
        if not results:
            return {}

        scores   = [r.anomaly_score for r in results]
        probs    = [r.anomaly_prob  for r in results]
        normals  = sum(1 for r in results if r.alert_level == "normal")
        highs    = sum(1 for r in results if r.alert_level == "high")
        criticals = sum(1 for r in results if r.alert_level == "critical")

        return {
            "total_frames":      len(results),
            "normal_frames":     normals,
            "high_alert_frames": highs,
            "critical_frames":   criticals,
            "alert_rate":        round((highs + criticals) / len(results), 4),
            "avg_anomaly_score": round(float(np.mean(scores)), 4),
            "min_anomaly_score": round(float(np.min(scores)),  4),
            "avg_anomaly_prob":  round(float(np.mean(probs)),  4),
            "alert_frame_ids":   [r.frame_id for r in results if r.is_alert][:20],
            "model_fitted":      self._is_fitted,
        }

    # ------------------------------------------------------------------
    # Private: scoring
    # ------------------------------------------------------------------

    def _ml_score(self, features: np.ndarray) -> Tuple[float, float]:
        """Isolation Forest scoring path.  Only called when _is_fitted=True."""
        X        = self._scaler.transform(features.reshape(1, -1))
        raw      = float(self._model.decision_function(X)[0])
        prob     = float(1.0 / (1.0 + np.exp(raw * 10)))
        return raw, max(0.0, min(1.0, prob))

    def _rule_score(
        self,
        features:   np.ndarray,
        detections: list,
    ) -> Tuple[float, float]:
        """
        Fallback rule-based scoring when model is not yet fitted.
        Returns a synthetic score in the same range as Isolation Forest.
        """
        score       = 0.0
        class_names = [d.get("class_name", "") for d in detections]

        if any(c in CRITICAL_THREAT_CLASSES for c in class_names):
            score -= 0.20
        if any(c in HIGH_THREAT_CLASSES for c in class_names):
            score -= 0.10
        if features[0] > 15:    # unusually high detection_count
            score -= 0.08
        if features[8] > 15.0:  # unusually high motion_score
            score -= 0.06

        prob = float(1.0 / (1.0 + np.exp(score * 10)))
        return score, max(0.0, min(1.0, prob))

    def _apply_class_boost(
        self,
        score:      float,
        detections: list,
        motion:     Optional[float],
    ) -> Tuple[float, List[str]]:
        """
        Apply hard score boosts for critical class combinations.

        Guarantees that military_vehicle and suspicious_object always
        trigger alerts regardless of what the ML model scores them.
        """
        boost_reasons: List[str] = []
        class_names = [d.get("class_name", "") for d in detections]

        if "military_vehicle" in class_names:
            score = min(score, THRESHOLD_CRITICAL - 0.05)
            boost_reasons.append("military_vehicle detected")

        if "suspicious_object" in class_names:
            score = min(score, THRESHOLD_CRITICAL - 0.02)
            boost_reasons.append("suspicious_object detected")

        if "crowd" in class_names and motion and motion > 12.0:
            score = min(score, THRESHOLD_HIGH - 0.05)
            boost_reasons.append("crowd with high motion")

        # Combined threat — multiple high-risk classes in same frame
        threat_present = {
            c for c in class_names
            if c in HIGH_THREAT_CLASSES | CRITICAL_THREAT_CLASSES
        }
        if len(threat_present) >= 2:
            score -= 0.05
            boost_reasons.append(
                f"multiple threat classes: {', '.join(sorted(threat_present))}"
            )

        return score, boost_reasons

    @staticmethod
    def _build_reasons(
        alert_level:   str,
        features:      np.ndarray,
        detections:    list,
        motion:        Optional[float],
        boost_reasons: List[str],
    ) -> List[str]:
        """Compose human-readable alert reasons for this frame."""
        if alert_level == "normal":
            return []

        reasons     = list(boost_reasons)
        det_count   = int(features[0])
        class_names = [d.get("class_name", "") for d in detections]

        if det_count > 10:
            reasons.append(f"unusually high detection count ({det_count})")
        if motion and motion > 10.0:
            reasons.append(f"high motion activity (score={motion:.1f})")
        if "crowd" in class_names:
            reasons.append("crowd gathering detected")
        if "aircraft" in class_names:
            reasons.append("aircraft in surveillance zone")
        if not reasons:
            reasons.append(f"statistical anomaly (IF score={features[0]:.3f})")

        return reasons

    # ------------------------------------------------------------------
    # Private: model persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        """Persist fitted model + scaler to disk."""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        payload = {
            "model":   self._model,
            "scaler":  self._scaler,
            "version": "1.0",
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Anomaly model saved → %s", self.model_path)

    def _load_model(self) -> None:
        """Load a previously saved model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            self._model     = payload["model"]
            self._scaler    = payload["scaler"]
            self._is_fitted = True
            logger.info("Anomaly model loaded ← %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load anomaly model: %s", exc)
            self._is_fitted = False


# ---------------------------------------------------------------------------
# Quick test (run directly: python src/anomaly.py path/to/video.mp4)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from detector import BorderDetector
    from preprocessing import extract_frames

    print("Border Surveillance AI — Anomaly Detection Module")
    print("=" * 55)

    video = (
        sys.argv[1] if len(sys.argv) > 1
        else "data/test_videos/dota_aerial_test.mp4"
    )

    if not os.path.exists(video):
        print(f"Video not found: {video}")
        print("Usage: python src/anomaly.py path/to/video.mp4")
        sys.exit(1)

    yolo_detector  = BorderDetector()
    anomaly        = AnomalyDetector()
    scored_results: List[AnomalyResult] = []

    print(f"\nPhase 1: Collecting {MIN_SAMPLES} baseline frames...")

    for frame_item in extract_frames(video, frame_skip=3, compute_flow=True,
                                     show_progress=True):
        fr_dict = yolo_detector.detect(frame_item).to_dict()

        if not anomaly._is_fitted:
            ready = anomaly.collect_baseline(fr_dict)
            if ready and not anomaly._is_fitted:
                print(f"  Baseline collected. Fitting Isolation Forest...")
                anomaly.fit()
                print("  Model fitted ✅")
            continue

        result = anomaly.score(fr_dict)
        scored_results.append(result)

        if result.alert_level == "critical":
            print(f"  🚨 CRITICAL  frame={result.frame_id:5d}  "
                  f"score={result.anomaly_score:.3f}  reasons={result.reasons}")
        elif result.alert_level == "high":
            print(f"  ⚠️  HIGH     frame={result.frame_id:5d}  "
                  f"score={result.anomaly_score:.3f}  reasons={result.reasons}")

    print("\n" + "=" * 55)
    print("ANOMALY SUMMARY")
    print("=" * 55)
    summary = anomaly.get_summary(scored_results)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

    os.makedirs("data/detections", exist_ok=True)
    with open("data/detections/anomaly_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Summary → data/detections/anomaly_summary.json")
    print("Next step: python src/alert_manager.py")
