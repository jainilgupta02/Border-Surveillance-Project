"""
Enhanced Pipeline — Zone + Temporal Intelligence Layer
========================================================

Wraps the existing BorderSurveillancePipeline to add zone-based
intrusion detection and temporal video analysis WITHOUT modifying
any of the original source code.

Architecture:
    This module sits ON TOP of the existing pipeline — it does not
    subclass or monkey-patch anything.  Instead, it:
        1. Instantiates the original pipeline components.
        2. Adds ZoneAnalyzer and TemporalAnalyzer as parallel layers.
        3. Produces enhanced results with zone + temporal metadata.

Usage:
    # Drop-in replacement for the original pipeline
    python src/pipeline.py --video data/test_videos/dota_aerial_test.mp4

    # Or use programmatically
    from pipeline import EnhancedPipeline, EnhancedConfig
    config   = EnhancedConfig(video_path="feed.mp4")
    pipeline = EnhancedPipeline(config)
    session  = pipeline.run()

Pipeline flow (per frame):
    1. preprocessing.py  → extract frame, resize, optical flow
    2. detector.py       → YOLOv8 detection  → FrameResult
    3. zone_analyzer.py  → zone intrusion    → ZoneAnalysisResult   [NEW]
    4. temporal_analyzer.py → temporal patterns→ TemporalAnalysisResult [NEW]
    5. anomaly.py        → anomaly scoring   → AnomalyResult
    6. alert_manager.py  → priority + logging→ Alert

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_SRC_DIR   = Path(__file__).resolve().parent   # src/
_PROJ_ROOT = _SRC_DIR.parent                   # project root

DEFAULT_ALERT_LOG     = str(_PROJ_ROOT / "data/alerts/alert_log.json")
DEFAULT_RESULTS_DIR   = str(_PROJ_ROOT / "data/results")
DEFAULT_ANNOTATED_DIR = str(_PROJ_ROOT / "data/detections")
DEFAULT_PIPELINE_LOG  = str(_PROJ_ROOT / "data/logs/pipeline.log")
DEFAULT_MODEL_PATH    = str(_PROJ_ROOT / "models/border_yolo.pt")
DEFAULT_ANOMALY_MODEL = str(_PROJ_ROOT / "models/anomaly_model.pkl")
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Optional[str] = None,
                   verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module imports — wrapped for import safety
# ---------------------------------------------------------------------------

try:
    from detector import BorderDetector
except ImportError:
    BorderDetector = None

try:
    from anomaly import AnomalyDetector
except ImportError:
    AnomalyDetector = None

try:
    from alert_manager import AlertManager
except ImportError:
    AlertManager = None

try:
    from zone_analyzer import ZoneAnalyzer
except ImportError:
    ZoneAnalyzer = None

try:
    from temporal_analyzer import TemporalAnalyzer
except ImportError:
    TemporalAnalyzer = None

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH       = "models/border_yolo.pt"
DEFAULT_ANOMALY_MODEL    = "models/anomaly_model.pkl"
DEFAULT_ALERT_LOG        = "data/alerts/alert_log.json"
DEFAULT_RESULTS_DIR      = "data/results"
DEFAULT_ANNOTATED_DIR    = "data/detections"
DEFAULT_PIPELINE_LOG     = "data/logs/pipeline.log"
DEFAULT_ENHANCED_LOG     = "data/results/enhanced_analysis.json"
DEFAULT_FRAME_SKIP       = 3
DEFAULT_CONFIDENCE       = 0.25
DEFAULT_IOU              = 0.45
DEFAULT_CONTAMINATION    = 0.08
DEFAULT_TEMPORAL_WINDOW  = 30


# ---------------------------------------------------------------------------
# EnhancedConfig
# ---------------------------------------------------------------------------

@dataclass
class EnhancedConfig:
    """
    All runtime configuration for the enhanced pipeline session.
    Extends the original PipelineConfig with zone + temporal settings.
    """
    # Source
    video_path:        Optional[str]   = None
    camera_index:      Optional[int]   = None

    # Detector
    model_path:        str             = DEFAULT_MODEL_PATH
    confidence:        float           = DEFAULT_CONFIDENCE
    iou:               float           = DEFAULT_IOU

    # Preprocessing
    frame_skip:        int             = DEFAULT_FRAME_SKIP
    compute_flow:      bool            = True

    # Anomaly
    anomaly_model:     str             = DEFAULT_ANOMALY_MODEL
    contamination:     float           = DEFAULT_CONTAMINATION

    # Alert manager
    alert_log:         str             = DEFAULT_ALERT_LOG
    cooldown:          int             = 30

    # Zone analyzer [NEW]
    enable_zones:      bool            = True

    # Temporal analyzer [NEW]
    enable_temporal:   bool            = True
    temporal_window:   int             = DEFAULT_TEMPORAL_WINDOW

    # Output
    save_frames:       bool            = False
    annotated_dir:     str             = DEFAULT_ANNOTATED_DIR
    save_results:      bool            = True
    results_dir:       str             = DEFAULT_RESULTS_DIR
    enhanced_log:      str             = DEFAULT_ENHANCED_LOG

    # Logging
    log_file:          str             = DEFAULT_PIPELINE_LOG
    verbose:           bool            = False

    # Runtime limits
    max_frames:        int             = 0

    @property
    def source(self) -> str:
        if self.video_path:
            return self.video_path
        return f"camera:{self.camera_index}"

    @property
    def video_source(self):
        if self.video_path:
            return self.video_path
        return self.camera_index


# ---------------------------------------------------------------------------
# EnhancedSession
# ---------------------------------------------------------------------------

@dataclass
class EnhancedSession:
    """Runtime state for one enhanced pipeline run."""
    config:             EnhancedConfig
    start_time:         float              = field(default_factory=time.time)
    end_time:           Optional[float]    = None

    # Phase A
    baseline_frames:    int                = 0
    model_fitted:       bool               = False

    # Phase B counters
    frames_scored:      int                = 0
    total_detections:   int                = 0
    normal_frames:      int                = 0
    high_frames:        int                = 0
    critical_frames:    int                = 0
    alerts_raised:      int                = 0

    # Zone stats [NEW]
    total_zone_violations:   int           = 0
    zone_critical_count:     int           = 0
    zone_buffer_count:       int           = 0
    zone_violation_records:  List[dict]    = field(default_factory=list)

    # Temporal stats [NEW]
    total_temporal_alerts:   int           = 0
    temporal_approach_count: int           = 0
    temporal_loiter_count:   int           = 0
    temporal_alert_records:  List[dict]    = field(default_factory=list)

    # Timing
    total_preprocess_ms:  float            = 0.0
    total_inference_ms:   float            = 0.0
    total_anomaly_ms:     float            = 0.0
    total_zone_ms:        float            = 0.0
    total_temporal_ms:    float            = 0.0

    # Alert records
    alert_records:      List[dict]         = field(default_factory=list)

    # Enhanced analysis log (per-frame)
    enhanced_frames:    List[dict]         = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def fps_effective(self) -> float:
        total = self.baseline_frames + self.frames_scored
        return total / max(self.elapsed_seconds, 0.001)

    def to_summary(self) -> dict:
        total = self.baseline_frames + self.frames_scored
        return {
            "source":                   self.config.source,
            "pipeline_type":            "enhanced",
            "start_time":               round(self.start_time,  3),
            "end_time":                 round(self.end_time or time.time(), 3),
            "elapsed_seconds":          round(self.elapsed_seconds, 2),
            "effective_fps":            round(self.fps_effective, 2),
            "frame_skip":               self.config.frame_skip,
            "total_frames":             total,
            "baseline_frames":          self.baseline_frames,
            "frames_scored":            self.frames_scored,
            "model_fitted":             self.model_fitted,
            "total_detections":         self.total_detections,
            "normal_frames":            self.normal_frames,
            "high_alert_frames":        self.high_frames,
            "critical_frames":          self.critical_frames,
            "alerts_raised":            self.alerts_raised,
            "alert_rate":               round(
                (self.high_frames + self.critical_frames)
                / max(self.frames_scored, 1), 4
            ),
            # Zone stats
            "zone_analysis_enabled":    self.config.enable_zones,
            "total_zone_violations":    self.total_zone_violations,
            "zone_critical_count":      self.zone_critical_count,
            "zone_buffer_count":        self.zone_buffer_count,
            # Temporal stats
            "temporal_analysis_enabled": self.config.enable_temporal,
            "total_temporal_alerts":     self.total_temporal_alerts,
            "temporal_approach_count":   self.temporal_approach_count,
            "temporal_loiter_count":     self.temporal_loiter_count,
            # Timing
            "avg_preprocess_ms":   round(
                self.total_preprocess_ms / max(total, 1), 2
            ),
            "avg_inference_ms":    round(
                self.total_inference_ms / max(total, 1), 2
            ),
            "avg_anomaly_ms":      round(
                self.total_anomaly_ms / max(self.frames_scored, 1), 2
            ),
            "avg_zone_ms":         round(
                self.total_zone_ms / max(self.frames_scored, 1), 2
            ),
            "avg_temporal_ms":     round(
                self.total_temporal_ms / max(self.frames_scored, 1), 2
            ),
        }


# ---------------------------------------------------------------------------
# EnhancedPipeline
# ---------------------------------------------------------------------------

class EnhancedPipeline:
    """
    Enhanced border surveillance pipeline with zone + temporal intelligence.

    This class wraps the original pipeline modules and adds two new
    parallel analysis layers:
        - ZoneAnalyzer:     Spatial intelligence (intrusion detection)
        - TemporalAnalyzer: Multi-frame patterns (tracking, trajectories)

    The original pipeline code is NOT modified — this class creates
    its own instances of all components and runs them in the same
    four-stage flow, with zone + temporal as extra stages after detection.

    Usage:
        config   = EnhancedConfig(video_path="feed.mp4")
        pipeline = EnhancedPipeline(config)
        session  = pipeline.run()
        print(session.to_summary())
    """

    def __init__(self, config: EnhancedConfig) -> None:
        self.config  = config
        self.session = EnhancedSession(config=config)
        self._shutdown_requested = False

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=" * 65)
        logger.info("Border Surveillance AI — ENHANCED Pipeline")
        logger.info("  Zone Analysis:     %s", "ENABLED" if config.enable_zones else "disabled")
        logger.info("  Temporal Analysis: %s (window=%d)",
                     "ENABLED" if config.enable_temporal else "disabled",
                     config.temporal_window)
        logger.info("=" * 65)
        logger.info("Source:      %s", config.source)
        logger.info("frame_skip:  %d  |  flow: %s  |  device: auto",
                     config.frame_skip, config.compute_flow)

        # ── Original modules ─────────────────────────────────────────
        self._detector      = self._init_detector()
        self._anomaly       = self._init_anomaly()
        self._alert_manager = self._init_alert_manager()

        # ── NEW modules ──────────────────────────────────────────────
        self._zone_analyzer     = self._init_zone_analyzer()
        self._temporal_analyzer = self._init_temporal_analyzer()

        # Prepare output directories
        if config.save_frames:
            os.makedirs(config.annotated_dir, exist_ok=True)
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------

    def _init_detector(self):
        logger.info("Loading detector:     %s", self.config.model_path)
        det = BorderDetector(
            model_path = self.config.model_path,
            confidence = self.config.confidence,
            iou        = self.config.iou,
        )
        logger.info("Detector ready  →  device: %s", det.device)
        return det

    def _init_anomaly(self):
        logger.info("Loading anomaly model: %s", self.config.anomaly_model)
        det = AnomalyDetector(
            contamination = self.config.contamination,
            model_path    = self.config.anomaly_model,
        )
        if det._is_fitted:
            logger.info("Anomaly model loaded from disk — skipping baseline")
            self.session.model_fitted = True
        else:
            logger.info("No saved anomaly model — will fit on first %d frames",
                         self._baseline_limit())
        return det

    def _init_alert_manager(self):
        logger.info("Alert log:            %s", self.config.alert_log)
        return AlertManager(
            log_path         = self.config.alert_log,
            cooldown_seconds = self.config.cooldown,
        )

    def _init_zone_analyzer(self):
        if not self.config.enable_zones:
            return None
        logger.info("Zone analyzer:        ENABLED (3 zones)")
        return ZoneAnalyzer()

    def _init_temporal_analyzer(self):
        if not self.config.enable_temporal:
            return None
        logger.info("Temporal analyzer:    ENABLED (window=%d)",
                     self.config.temporal_window)
        return TemporalAnalyzer(window_size=self.config.temporal_window)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _baseline_limit(self) -> int:
        from anomaly import MIN_SAMPLES
        return MIN_SAMPLES

    def _handle_shutdown(self, signum, frame) -> None:
        logger.warning("Shutdown signal received — finishing current frame...")
        self._shutdown_requested = True

    def _log_progress(self, frame_id, fr_dict, anomaly_result,
                      zone_result, temporal_result, alert) -> None:
        det_count  = fr_dict.get("detection_count", 0)
        inf_ms     = fr_dict.get("inference_ms",    0.0)
        motion     = fr_dict.get("motion_score")
        a_level    = anomaly_result.alert_level if anomaly_result else "—"
        a_score    = (f"{anomaly_result.anomaly_score:.3f}"
                      if anomaly_result else "—")

        # Zone info
        z_info = ""
        if zone_result and zone_result.has_violations:
            z_info = f" | zone={zone_result.max_severity}"

        # Temporal info
        t_info = ""
        if temporal_result and temporal_result.has_alerts:
            t_info = f" | temporal={len(temporal_result.alerts)} alerts"

        # Alert tag
        priority_tag = ""
        if alert:
            emoji = {"CRITICAL": "🚨", "HIGH": "⚠️ ",
                      "MEDIUM": "🟡", "LOW": "🟢"}.get(alert.priority, "")
            priority_tag = f"  [{emoji} {alert.priority}]"

        motion_str = f"  motion={motion:.1f}" if motion is not None else ""

        logger.info(
            "Frame %5d | det=%2d | inf=%5.1fms%s | "
            "anomaly=%-8s score=%s%s%s%s",
            frame_id, det_count, inf_ms, motion_str,
            a_level, a_score, z_info, t_info, priority_tag,
        )

    def _save_annotated_frame(self, frame_item, fr_result) -> None:
        import cv2 as _cv2
        import numpy as np
        frame = frame_item["frame"]
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        annotated = self._detector.annotate_frame(frame, fr_result)
        out_path  = os.path.join(
            self.config.annotated_dir,
            f"frame_{fr_result.frame_id:06d}.jpg",
        )
        _cv2.imwrite(out_path, annotated)

    def _save_session_results(self) -> None:
        summary = self.session.to_summary()
        summary["alerts"]          = self.session.alert_records
        summary["zone_violations"] = self.session.zone_violation_records[-50:]
        summary["temporal_alerts"] = self.session.temporal_alert_records[-50:]

        ts   = time.strftime("%Y%m%d_%H%M%S",
                             time.localtime(self.session.start_time))
        name = Path(self.config.source).stem if self.config.video_path \
               else f"camera_{self.config.camera_index}"
        path = os.path.join(self.config.results_dir,
                            f"enhanced_session_{name}_{ts}.json")

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Enhanced session results saved → %s", path)

    def _save_enhanced_analysis(self) -> None:
        """Save per-frame enhanced analysis log for dashboard."""
        if not self.session.enhanced_frames:
            return

        path = self.config.enhanced_log
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "pipeline_type": "enhanced",
            "total_frames": len(self.session.enhanced_frames),
            "frames": self.session.enhanced_frames[-500:],  # last 500 frames
            "zone_summary": (
                self._zone_analyzer.get_summary()
                if self._zone_analyzer else {}
            ),
            "temporal_summary": (
                self._temporal_analyzer.get_summary()
                if self._temporal_analyzer else {}
            ),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Enhanced analysis log saved → %s", path)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> EnhancedSession:
        """Execute the full enhanced pipeline."""
        from preprocessing import extract_frames

        baseline_limit  = self._baseline_limit()
        already_fitted  = self._anomaly._is_fitted
        phase           = "B" if already_fitted else "A"

        logger.info("─" * 65)
        if phase == "A":
            logger.info("Phase A: collecting %d baseline frames...",
                         baseline_limit)
        else:
            logger.info("Phase B: model pre-loaded — scoring immediately")
        logger.info("─" * 65)

        frame_count = 0

        try:
            for frame_item in extract_frames(
                self.config.video_source,
                frame_skip    = self.config.frame_skip,
                compute_flow  = self.config.compute_flow,
                show_progress = False,
            ):
                if self._shutdown_requested:
                    logger.info("Shutdown: stopping after frame %d",
                                 frame_item["frame_id"])
                    break

                if self.config.max_frames > 0 \
                        and frame_count >= self.config.max_frames:
                    logger.info("max_frames=%d reached — stopping",
                                 self.config.max_frames)
                    break

                frame_count += 1
                self._process_frame(frame_item, phase, baseline_limit)

                if phase == "A" and self._anomaly._is_fitted:
                    phase = "B"
                    logger.info("─" * 65)
                    logger.info("Phase B: Isolation Forest fitted — "
                                 "enhanced scoring starts now")
                    logger.info("─" * 65)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — wrapping up...")
        except Exception as exc:
            logger.error("Enhanced pipeline error at frame %d: %s",
                          frame_count, exc, exc_info=True)
        finally:
            self._finish()

        return self.session

    def _process_frame(self, frame_item: dict,
                       phase: str, baseline_limit: int) -> None:
        """Process one frame through all six stages."""

        frame_id = frame_item["frame_id"]

        # ── Stage 1: Preprocessing (already done by extract_frames) ──
        t_pre = time.time()
        pre_ms = (time.time() - t_pre) * 1000
        self.session.total_preprocess_ms += pre_ms

        # ── Stage 2: YOLOv8 Detection ────────────────────────────────
        t_det = time.time()
        fr_result = self._detector.detect(frame_item)
        inf_ms    = (time.time() - t_det) * 1000
        self.session.total_inference_ms += inf_ms
        self.session.total_detections   += fr_result.detection_count

        fr_dict = fr_result.to_dict()

        # Save annotated frame (optional)
        if self.config.save_frames:
            self._save_annotated_frame(frame_item, fr_result)

        # ── Stage 3: Zone Analysis [NEW] ─────────────────────────────
        zone_result = None
        if self._zone_analyzer:
            t_zone = time.time()
            zone_result = self._zone_analyzer.analyze(fr_dict)
            zone_ms = (time.time() - t_zone) * 1000
            self.session.total_zone_ms += zone_ms

            if zone_result.has_violations:
                self.session.total_zone_violations += zone_result.violation_count
                if zone_result.has_critical:
                    self.session.zone_critical_count += 1
                else:
                    self.session.zone_buffer_count += 1

                # Store zone violations (limit to 200 records)
                for v in zone_result.violations[:5]:
                    if len(self.session.zone_violation_records) < 200:
                        self.session.zone_violation_records.append(v.to_dict())

        # ── Stage 4: Temporal Analysis [NEW] ─────────────────────────
        temporal_result = None
        if self._temporal_analyzer:
            t_temp = time.time()
            temporal_result = self._temporal_analyzer.analyze(fr_dict)
            temp_ms = (time.time() - t_temp) * 1000
            self.session.total_temporal_ms += temp_ms

            if temporal_result.has_alerts:
                self.session.total_temporal_alerts += len(temporal_result.alerts)
                self.session.temporal_approach_count += temporal_result.approaching_count
                self.session.temporal_loiter_count  += temporal_result.loitering_count

                for a in temporal_result.alerts[:3]:
                    if len(self.session.temporal_alert_records) < 200:
                        self.session.temporal_alert_records.append(a.to_dict())

        # ── Stage 5a: Phase A — baseline collection ──────────────────
        if phase == "A":
            self.session.baseline_frames += 1
            ready = self._anomaly.collect_baseline(fr_dict)
            if ready and not self._anomaly._is_fitted:
                logger.info("Baseline complete (%d frames) — fitting model...",
                             self.session.baseline_frames)
                t_fit = time.time()
                self._anomaly.fit()
                fit_ms = (time.time() - t_fit) * 1000
                self.session.model_fitted = True
                logger.info("Isolation Forest fitted in %.0f ms", fit_ms)
            return

        # ── Stage 5b: Phase B — anomaly scoring ─────────────────────
        t_ano = time.time()
        anomaly_result = self._anomaly.score(fr_dict)
        ano_ms = (time.time() - t_ano) * 1000
        self.session.total_anomaly_ms += ano_ms
        self.session.frames_scored    += 1

        # Update level counters
        if anomaly_result.alert_level == "critical":
            self.session.critical_frames += 1
        elif anomaly_result.alert_level == "high":
            self.session.high_frames += 1
        else:
            self.session.normal_frames += 1

        # ── Stage 6: Alert management ────────────────────────────────
        # Enhance anomaly result with zone + temporal reasons
        enhanced_result = anomaly_result.to_dict()
        if zone_result and zone_result.has_violations:
            enhanced_result["reasons"] = (
                enhanced_result.get("reasons", []) + zone_result.reasons[:3]
            )
            # Upgrade alert level if zone is critical
            if zone_result.has_critical and \
                    enhanced_result.get("alert_level") != "critical":
                enhanced_result["alert_level"] = "critical"

        if temporal_result and temporal_result.has_alerts:
            enhanced_result["reasons"] = (
                enhanced_result.get("reasons", []) + temporal_result.reasons[:3]
            )

        alert = self._alert_manager.process(enhanced_result)

        if alert:
            self.session.alerts_raised += 1
            self.session.alert_records.append(alert.to_dict())

            if alert.priority in ("CRITICAL", "HIGH"):
                self._upload_alert_frame(frame_item, fr_result, alert)

        # ── Store enhanced frame data for dashboard ──────────────────
        if len(self.session.enhanced_frames) < 500:
            frame_data = {
                "frame_id": frame_id,
                "detection_count": fr_result.detection_count,
                "anomaly_score": anomaly_result.anomaly_score,
                "alert_level": anomaly_result.alert_level,
            }
            if zone_result:
                frame_data["zone_risk"] = zone_result.risk_score
                frame_data["zone_severity"] = zone_result.max_severity
                frame_data["zone_violations"] = zone_result.violation_count
            if temporal_result:
                frame_data["temporal_risk"] = temporal_result.risk_score
                frame_data["temporal_alerts"] = len(temporal_result.alerts)
                frame_data["tracked_objects"] = temporal_result.tracked_objects
                frame_data["detection_trend"] = temporal_result.detection_trend
            self.session.enhanced_frames.append(frame_data)

        # ── Progress log ─────────────────────────────────────────────
        self._log_progress(
            frame_id, fr_dict, anomaly_result,
            zone_result, temporal_result, alert,
        )

    def _upload_alert_frame(self, frame_item, fr_result, alert) -> None:
        """Save annotated alert frame locally and attempt Azure upload."""
        import cv2 as _cv2
        import numpy as np

        try:
            frame = frame_item["frame"]
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            annotated = self._detector.annotate_frame(frame, fr_result)

            os.makedirs(self.config.annotated_dir, exist_ok=True)
            local_path = os.path.join(
                self.config.annotated_dir,
                f"alert_{alert.alert_id}_frame_{fr_result.frame_id:06d}.jpg",
            )
            _cv2.imwrite(local_path, annotated)

            try:
                from azure_client import azure as _azure
                if _azure.enabled:
                    _azure.upload_frame(local_path, alert.alert_id)
            except Exception:
                pass
        except Exception as exc:
            logger.warning("_upload_alert_frame failed: %s", exc)

    def _finish(self) -> None:
        self.session.end_time = time.time()

        logger.info("=" * 65)
        logger.info("ENHANCED PIPELINE COMPLETE")
        logger.info("=" * 65)

        summary = self.session.to_summary()
        for key, val in summary.items():
            if key not in ("alerts", "zone_violations", "temporal_alerts"):
                logger.info("  %-32s %s", key, val)

        # Alert manager summary
        mgr_summary = self._alert_manager.get_summary()
        logger.info("─" * 65)
        logger.info("ALERT SUMMARY")
        logger.info("─" * 65)
        for key, val in mgr_summary.items():
            logger.info("  %-28s %s", key, val)

        # Zone summary
        if self._zone_analyzer:
            zone_summary = self._zone_analyzer.get_summary()
            logger.info("─" * 65)
            logger.info("ZONE ANALYSIS SUMMARY")
            logger.info("─" * 65)
            for key, val in zone_summary.items():
                logger.info("  %-28s %s", key, val)

        # Temporal summary
        if self._temporal_analyzer:
            temp_summary = self._temporal_analyzer.get_summary()
            logger.info("─" * 65)
            logger.info("TEMPORAL ANALYSIS SUMMARY")
            logger.info("─" * 65)
            for key, val in temp_summary.items():
                logger.info("  %-28s %s", key, val)

        if self.config.save_results:
            self._save_session_results()
            self._save_enhanced_analysis()

        logger.info("=" * 65)
        logger.info("Alert log → %s", self.config.alert_log)
        logger.info("Enhanced log → %s", self.config.enhanced_log)
        if self.config.save_frames:
            logger.info("Annotated frames → %s", self.config.annotated_dir)
        logger.info("Next step: streamlit run dashboard/app.py")
        logger.info("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> EnhancedConfig:
    if args.video is None and args.camera is None:
        raise ValueError("Provide --video or --camera")
    if args.video and args.camera is not None:
        raise ValueError("Cannot use both --video and --camera")

    return EnhancedConfig(
        video_path       = args.video,
        camera_index     = args.camera,
        model_path       = args.model,
        confidence       = args.confidence,
        iou              = args.iou,
        frame_skip       = args.frame_skip,
        compute_flow     = not args.no_flow,
        anomaly_model    = args.anomaly_model,
        contamination    = args.contamination,
        alert_log        = args.alert_log,
        cooldown         = args.cooldown,
        enable_zones     = not args.no_zones,
        enable_temporal  = not args.no_temporal,
        temporal_window  = args.temporal_window,
        save_frames      = args.save_frames,
        annotated_dir    = args.annotated_dir,
        save_results     = not args.no_results,
        results_dir      = args.results_dir,
        enhanced_log     = args.enhanced_log,
        log_file         = args.log_file,
        verbose          = args.verbose,
        max_frames       = args.max_frames,
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Border Surveillance AI — Enhanced Pipeline "
                    "(Zone + Temporal Intelligence)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", metavar="PATH",
                     help="Path to input video file")
    src.add_argument("--camera", type=int, metavar="INDEX",
                     help="Webcam index")

    # Detector
    p.add_argument("--model", default=DEFAULT_MODEL_PATH,
                   help="Path to YOLO weights (.pt file)")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help="YOLO NMS IoU threshold")

    # Preprocessing
    p.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                   metavar="N", help="Process every Nth frame")
    p.add_argument("--no-flow", action="store_true",
                   help="Disable optical flow")

    # Anomaly
    p.add_argument("--anomaly-model", default=DEFAULT_ANOMALY_MODEL,
                   help="Path to anomaly model (.pkl)")
    p.add_argument("--contamination", type=float,
                   default=DEFAULT_CONTAMINATION)

    # Alert
    p.add_argument("--alert-log", default=DEFAULT_ALERT_LOG)
    p.add_argument("--cooldown", type=int, default=30)

    # Zone [NEW]
    p.add_argument("--no-zones", action="store_true",
                   help="Disable zone analysis")

    # Temporal [NEW]
    p.add_argument("--no-temporal", action="store_true",
                   help="Disable temporal analysis")
    p.add_argument("--temporal-window", type=int,
                   default=DEFAULT_TEMPORAL_WINDOW,
                   help="Temporal analysis sliding window size")

    # Output
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--annotated-dir", default=DEFAULT_ANNOTATED_DIR)
    p.add_argument("--no-results", action="store_true")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--enhanced-log", default=DEFAULT_ENHANCED_LOG,
                   help="Path for enhanced analysis JSON")

    # Logging
    p.add_argument("--log-file", default=DEFAULT_PIPELINE_LOG)
    p.add_argument("--verbose", "-v", action="store_true")

    # Limits
    p.add_argument("--max-frames", type=int, default=0)

    return p
BorderSurveillancePipeline = EnhancedPipeline
BorderSurveillanceConfig   = EnhancedConfig
PipelineConfig             = EnhancedConfig

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = make_parser()
    args   = parser.parse_args()

    _setup_logging(log_file=args.log_file, verbose=args.verbose)

    try:
        config   = build_config(args)
        pipeline = EnhancedPipeline(config)
        session  = pipeline.run()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        parser.print_help()
        return 1
    except Exception as exc:
        logger.error("Fatal pipeline error: %s", exc, exc_info=True)
        return 2

    return 0 if session.alerts_raised == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


# ── Backwards-compatible aliases for test suite ──────────────────────────────
BorderSurveillancePipeline = EnhancedPipeline
PipelineConfig             = EnhancedConfig
PipelineSession            = dict          # session results are plain dicts

def build_config(video_path: str, **kwargs) -> EnhancedConfig:
    """Alias for tests — builds a minimal EnhancedConfig."""
    return EnhancedConfig(video_path=video_path, **kwargs)

DEFAULT_FRAME_SKIP  = DEFAULT_FRAME_SKIP
DEFAULT_CONFIDENCE  = DEFAULT_CONFIDENCE
DEFAULT_IOU         = DEFAULT_IOU
