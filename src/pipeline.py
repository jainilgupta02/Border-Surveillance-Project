"""
Main Pipeline Orchestrator
===========================

Ties all four modules into a single command-line program:

    preprocessing.py → detector.py → anomaly.py → alert_manager.py

This is the "main program" of the Border Surveillance AI system.
Running this file is what you do to actually USE the system.

Usage:
    # Process a video file
    python src/pipeline.py --video data/test_videos/dota_aerial_test.mp4

    # Process with every 3rd frame (faster)
    python src/pipeline.py --video feed.mp4 --frame-skip 3

    # Live webcam
    python src/pipeline.py --camera 0

    # Save annotated frames + JSON results
    python src/pipeline.py --video feed.mp4 --save-frames --save-results

    # Disable optical flow (faster on CPU-only machines)
    python src/pipeline.py --video feed.mp4 --no-flow

Pipeline flow (per frame):
    1. preprocessing.py  — extract frame, resize to 640×640, compute optical flow
    2. detector.py       — run YOLOv8, return FrameResult (detections + threat flags)
    3. anomaly.py        — extract features, score with Isolation Forest,
                           return AnomalyResult (score, alert_level, reasons)
    4. alert_manager.py  — assign CRITICAL/HIGH/MEDIUM/LOW priority,
                           write to alert log JSON, optionally email

Phase A (first MIN_SAMPLES frames):  baseline collection + model fitting
Phase B (all remaining frames):      live scoring + alerting

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

# ---------------------------------------------------------------------------
# Logging — configured BEFORE importing other modules so every logger
# inherits the same handler setup.
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Optional[str] = None,
                   verbose: bool = False) -> None:
    """Configure root logger with console + optional file handler."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level imports so tests can patch pipeline.BorderDetector etc.
# Wrapped in try/except so pipeline.py imports cleanly without AI packages.
# ---------------------------------------------------------------------------

try:
    from detector import BorderDetector
except ImportError:  # pragma: no cover
    BorderDetector = None  # type: ignore

try:
    from anomaly import AnomalyDetector
except ImportError:  # pragma: no cover
    AnomalyDetector = None  # type: ignore

try:
    from alert_manager import AlertManager
except ImportError:  # pragma: no cover
    AlertManager = None  # type: ignore

# ---------------------------------------------------------------------------
# Default paths — all relative to project root
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH       = "models/border_yolo.pt"
DEFAULT_ANOMALY_MODEL    = "models/anomaly_model.pkl"
DEFAULT_ALERT_LOG        = "data/alerts/alert_log.json"
DEFAULT_RESULTS_DIR      = "data/results"
DEFAULT_ANNOTATED_DIR    = "data/detections"
DEFAULT_PIPELINE_LOG     = "data/logs/pipeline.log"
DEFAULT_FRAME_SKIP       = 3
DEFAULT_CONFIDENCE       = 0.25
DEFAULT_IOU              = 0.45
DEFAULT_CONTAMINATION    = 0.08


# ---------------------------------------------------------------------------
# PipelineConfig  — single source of truth for all runtime parameters
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All runtime configuration for one pipeline session.
    Built from CLI args by build_config().
    """
    # Source
    video_path:       Optional[str]   = None
    camera_index:     Optional[int]   = None

    # Detector
    model_path:       str             = DEFAULT_MODEL_PATH
    confidence:       float           = DEFAULT_CONFIDENCE
    iou:              float           = DEFAULT_IOU

    # Preprocessing
    frame_skip:       int             = DEFAULT_FRAME_SKIP
    compute_flow:     bool            = True

    # Anomaly
    anomaly_model:    str             = DEFAULT_ANOMALY_MODEL
    contamination:    float           = DEFAULT_CONTAMINATION

    # Alert manager
    alert_log:        str             = DEFAULT_ALERT_LOG
    cooldown:         int             = 30

    # Output
    save_frames:      bool            = False
    annotated_dir:    str             = DEFAULT_ANNOTATED_DIR
    save_results:     bool            = True
    results_dir:      str             = DEFAULT_RESULTS_DIR

    # Logging
    log_file:         str             = DEFAULT_PIPELINE_LOG
    verbose:          bool            = False

    # Runtime limits (0 = unlimited)
    max_frames:       int             = 0

    @property
    def source(self) -> str:
        """Human-readable source description."""
        if self.video_path:
            return self.video_path
        return f"camera:{self.camera_index}"

    @property
    def video_source(self):
        """Return the value to pass to extract_frames()."""
        if self.video_path:
            return self.video_path
        return self.camera_index


# ---------------------------------------------------------------------------
# PipelineSession  — accumulates all data from one run
# ---------------------------------------------------------------------------

@dataclass
class PipelineSession:
    """
    Runtime state for one complete pipeline run.
    Passed through each stage so the final summary has full context.
    """
    config:            PipelineConfig
    start_time:        float              = field(default_factory=time.time)
    end_time:          Optional[float]    = None

    # Phase A
    baseline_frames:   int                = 0
    model_fitted:      bool               = False

    # Phase B counters
    frames_scored:     int                = 0
    total_detections:  int                = 0
    normal_frames:     int                = 0
    high_frames:       int                = 0
    critical_frames:   int                = 0
    alerts_raised:     int                = 0

    # Timing
    total_preprocess_ms:  float           = 0.0
    total_inference_ms:   float           = 0.0
    total_anomaly_ms:     float           = 0.0

    # Aggregated alert records (for summary JSON)
    alert_records:     List[dict]         = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def fps_effective(self) -> float:
        total = self.baseline_frames + self.frames_scored
        return total / max(self.elapsed_seconds, 0.001)

    def to_summary(self) -> dict:
        """Serialise session to a dict suitable for JSON / Cosmos DB."""
        total = self.baseline_frames + self.frames_scored
        return {
            "source":              self.config.source,
            "start_time":          round(self.start_time,  3),
            "end_time":            round(self.end_time or time.time(), 3),
            "elapsed_seconds":     round(self.elapsed_seconds, 2),
            "effective_fps":       round(self.fps_effective, 2),
            "frame_skip":          self.config.frame_skip,
            "total_frames":        total,
            "baseline_frames":     self.baseline_frames,
            "frames_scored":       self.frames_scored,
            "model_fitted":        self.model_fitted,
            "total_detections":    self.total_detections,
            "normal_frames":       self.normal_frames,
            "high_alert_frames":   self.high_frames,
            "critical_frames":     self.critical_frames,
            "alerts_raised":       self.alerts_raised,
            "alert_rate":          round(
                (self.high_frames + self.critical_frames)
                / max(self.frames_scored, 1), 4
            ),
            "avg_preprocess_ms":   round(
                self.total_preprocess_ms / max(total, 1), 2
            ),
            "avg_inference_ms":    round(
                self.total_inference_ms / max(total, 1), 2
            ),
            "avg_anomaly_ms":      round(
                self.total_anomaly_ms / max(self.frames_scored, 1), 2
            ),
        }


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class BorderSurveillancePipeline:
    """
    Main orchestrator.

    Initialises all four modules once, then processes every frame in order:
        preprocess → detect → score_anomaly → manage_alert

    Example:
        config   = PipelineConfig(video_path="border_feed.mp4",
                                  save_frames=True)
        pipeline = BorderSurveillancePipeline(config)
        session  = pipeline.run()
        print(session.to_summary())
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config  = config
        self.session = PipelineSession(config=config)
        self._shutdown_requested = False

        # Register Ctrl+C handler for graceful shutdown
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=" * 60)
        logger.info("Border Surveillance AI — Pipeline")
        logger.info("=" * 60)
        logger.info("Source:      %s", config.source)
        logger.info("frame_skip:  %d  |  flow: %s  |  device: auto",
                    config.frame_skip, config.compute_flow)

        # Initialise all modules
        self._detector      = self._init_detector()
        self._anomaly       = self._init_anomaly()
        self._alert_manager = self._init_alert_manager()

        # Prepare output directories
        if config.save_frames:
            os.makedirs(config.annotated_dir, exist_ok=True)
        if config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Module initialisation
    # ------------------------------------------------------------------

    def _init_detector(self):
        """Load YOLOv8 model — happens once, not per frame."""
        logger.info("Loading detector:     %s", self.config.model_path)
        det = BorderDetector(
            model_path  = self.config.model_path,
            confidence  = self.config.confidence,
            iou         = self.config.iou,
        )
        logger.info("Detector ready  →  device: %s", det.device)
        return det

    def _init_anomaly(self):
        """Load or create anomaly detector."""
        logger.info("Loading anomaly model: %s", self.config.anomaly_model)
        det = AnomalyDetector(
            contamination = self.config.contamination,
            model_path    = self.config.anomaly_model,
        )
        if det._is_fitted:
            logger.info("Anomaly model loaded from disk — skipping baseline phase")
            self.session.model_fitted = True
        else:
            logger.info("No saved anomaly model — will fit on first %d frames",
                        self._baseline_limit())
        return det

    def _init_alert_manager(self):
        """Initialise alert manager."""
        logger.info("Alert log:            %s", self.config.alert_log)
        mgr = AlertManager(
            log_path         = self.config.alert_log,
            cooldown_seconds = self.config.cooldown,
        )
        return mgr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _baseline_limit(self) -> int:
        """Number of frames needed for baseline (imported from anomaly)."""
        from anomaly import MIN_SAMPLES
        return MIN_SAMPLES

    def _handle_shutdown(self, signum, frame) -> None:
        """Catch Ctrl+C / SIGTERM — finish current frame then stop."""
        logger.warning("Shutdown signal received — finishing current frame...")
        self._shutdown_requested = True

    def _log_progress(self, frame_id: int, fr_dict: dict,
                      anomaly_result, alert) -> None:
        """Print a one-line progress update to console."""
        det_count  = fr_dict.get("detection_count", 0)
        inf_ms     = fr_dict.get("inference_ms",    0.0)
        motion     = fr_dict.get("motion_score")
        level      = anomaly_result.alert_level if anomaly_result else "—"
        score      = (f"{anomaly_result.anomaly_score:.3f}"
                      if anomaly_result else "—")

        priority_tag = ""
        if alert:
            emoji = {"CRITICAL": "🚨", "HIGH": "⚠️ ",
                     "MEDIUM": "🟡", "LOW": "🟢"}.get(alert.priority, "")
            priority_tag = f"  [{emoji} {alert.priority}]"

        motion_str = f"  motion={motion:.1f}" if motion is not None else ""
        logger.info(
            "Frame %5d | det=%2d | inf=%5.1fms%s | "
            "anomaly=%-8s score=%s%s",
            frame_id, det_count, inf_ms, motion_str,
            level, score, priority_tag,
        )

    def _save_annotated_frame(self, frame_item: dict,
                              fr_result) -> None:
        """Draw bounding boxes and save annotated frame to disk."""
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
        """Write full session summary + all alert records to JSON."""
        summary = self.session.to_summary()
        summary["alerts"] = self.session.alert_records

        ts   = time.strftime("%Y%m%d_%H%M%S",
                             time.localtime(self.session.start_time))
        name = Path(self.config.source).stem if self.config.video_path \
               else f"camera_{self.config.camera_index}"
        path = os.path.join(self.config.results_dir,
                            f"session_{name}_{ts}.json")

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Session results saved → %s", path)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> PipelineSession:
        """
        Execute the full pipeline from source to alert log.

        Returns:
            PipelineSession with complete statistics for this run.
        """
        from preprocessing import extract_frames

        baseline_limit  = self._baseline_limit()
        already_fitted  = self._anomaly._is_fitted
        phase           = "B" if already_fitted else "A"

        logger.info("─" * 60)
        if phase == "A":
            logger.info("Phase A: collecting %d baseline frames...",
                        baseline_limit)
        else:
            logger.info("Phase B: model pre-loaded — scoring immediately")
        logger.info("─" * 60)

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

                # Check if baseline just completed
                if phase == "A" and self._anomaly._is_fitted:
                    phase = "B"
                    logger.info("─" * 60)
                    logger.info("Phase B: Isolation Forest fitted — "
                                "live scoring starts now")
                    logger.info("─" * 60)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — wrapping up...")
        except Exception as exc:
            logger.error("Pipeline error at frame %d: %s",
                         frame_count, exc, exc_info=True)
        finally:
            self._finish()

        return self.session

    def _process_frame(self, frame_item: dict,
                       phase: str, baseline_limit: int) -> None:
        """Process exactly one frame through all four stages."""

        frame_id = frame_item["frame_id"]

        # ── Stage 1: Preprocessing (already done by extract_frames) ──
        t_pre_start = time.time()
        # extract_frames() has already resized and optionally computed flow.
        # We just note the frame is ready.
        pre_ms = (time.time() - t_pre_start) * 1000
        self.session.total_preprocess_ms += pre_ms

        # ── Stage 2: Detection ────────────────────────────────────────
        t_det = time.time()
        fr_result = self._detector.detect(frame_item)
        inf_ms    = (time.time() - t_det) * 1000
        self.session.total_inference_ms += inf_ms
        self.session.total_detections   += fr_result.detection_count

        fr_dict = fr_result.to_dict()

        # ── Save annotated frame (optional) ──────────────────────────
        if self.config.save_frames:
            self._save_annotated_frame(frame_item, fr_result)

        # ── Stage 3a: Phase A — baseline collection ───────────────────
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
                logger.info("Isolation Forest fitted in %.0f ms  →  %s",
                            fit_ms, self.config.anomaly_model)
            return   # don't score baseline frames

        # ── Stage 3b: Phase B — anomaly scoring ──────────────────────
        t_ano = time.time()
        anomaly_result = self._anomaly.score(fr_dict)
        ano_ms         = (time.time() - t_ano) * 1000
        self.session.total_anomaly_ms += ano_ms
        self.session.frames_scored   += 1

        # Update level counters
        if anomaly_result.alert_level == "critical":
            self.session.critical_frames += 1
        elif anomaly_result.alert_level == "high":
            self.session.high_frames += 1
        else:
            self.session.normal_frames += 1

        # ── Stage 4: Alert management ─────────────────────────────────
        alert = self._alert_manager.process(anomaly_result.to_dict())

        if alert:
            self.session.alerts_raised  += 1
            self.session.alert_records.append(alert.to_dict())

            # ── Azure: save + upload alert frame (HIGH and CRITICAL only)
            # NOTE: azure.save_alert() is already called inside
            #       alert_manager._log_alert() — no need to duplicate here.
            if alert.priority in ("CRITICAL", "HIGH"):
                self._upload_alert_frame(frame_item, fr_result, alert)

        # ── Progress log ──────────────────────────────────────────────
        self._log_progress(frame_id, fr_dict, anomaly_result, alert)

    def _upload_alert_frame(self, frame_item: dict,
                            fr_result,
                            alert) -> None:
        """
        Save the annotated alert frame locally AND upload to Azure Blob Storage.

        Only called for HIGH and CRITICAL alerts — keeps storage lean.
        Local file is always written so the dashboard can display it.
        Azure upload is attempted only if azure.enabled is True.
        """
        import cv2 as _cv2
        import numpy as np

        try:
            frame = frame_item["frame"]
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            annotated = self._detector.annotate_frame(frame, fr_result)

            # ── Save locally ──────────────────────────────────────────
            os.makedirs(self.config.annotated_dir, exist_ok=True)
            local_path = os.path.join(
                self.config.annotated_dir,
                f"alert_{alert.alert_id}_frame_{fr_result.frame_id:06d}.jpg",
            )
            _cv2.imwrite(local_path, annotated)
            logger.debug("Alert frame saved → %s", local_path)

            # ── Upload to Azure Blob Storage ──────────────────────────
            try:
                from azure_client import azure as _azure
                if _azure.enabled:
                    ok = _azure.upload_frame(local_path, alert.alert_id)
                    if ok:
                        logger.info(
                            "[%s] Frame %d → Azure alert-frames/",
                            alert.priority, fr_result.frame_id,
                        )
            except Exception as az_exc:
                logger.debug("Azure frame upload skipped: %s", az_exc)

        except Exception as exc:
            logger.warning("_upload_alert_frame failed: %s", exc)

    def _finish(self) -> None:
        """Called after the main loop — save results and print summary."""
        self.session.end_time = time.time()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        summary = self.session.to_summary()
        for key, val in summary.items():
            if key != "alerts":
                logger.info("  %-28s %s", key, val)

        # ── Azure: upload session results JSON to Blob Storage ───────
        try:
            from azure_client import azure as _azure
            if _azure.enabled:
                _azure.upload_session_results(summary)
                logger.info(
                    "Session JSON uploaded → Azure session-results/ ✅"
                )
            else:
                logger.info("Azure not enabled — session saved locally only")
        except Exception as exc:
            logger.warning("Azure session upload skipped: %s", exc)

        # Alert manager summary
        mgr_summary = self._alert_manager.get_summary()
        logger.info("─" * 60)
        logger.info("ALERT SUMMARY")
        logger.info("─" * 60)
        for key, val in mgr_summary.items():
            logger.info("  %-28s %s", key, val)

        if self.config.save_results:
            self._save_session_results()

        logger.info("=" * 60)
        logger.info("Alert log → %s", self.config.alert_log)
        if self.config.save_frames:
            logger.info("Annotated frames → %s", self.config.annotated_dir)
        logger.info("Next step: streamlit run dashboard/app.py")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Translate argparse namespace into a PipelineConfig."""
    if args.video is None and args.camera is None:
        raise ValueError("Provide --video or --camera")
    if args.video and args.camera is not None:
        raise ValueError("Cannot use both --video and --camera")

    return PipelineConfig(
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
        save_frames      = args.save_frames,
        annotated_dir    = args.annotated_dir,
        save_results     = not args.no_results,
        results_dir      = args.results_dir,
        log_file         = args.log_file,
        verbose          = args.verbose,
        max_frames       = args.max_frames,
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Border Surveillance AI — Main Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ─────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video", metavar="PATH",
        help="Path to input video file (.mp4 .avi .mov .mkv)",
    )
    src.add_argument(
        "--camera", type=int, metavar="INDEX",
        help="Webcam index (0 = default camera)",
    )

    # ── Detector ────────────────────────────────────────────────────
    p.add_argument("--model", default=DEFAULT_MODEL_PATH,
                   help="Path to YOLO weights (.pt file)")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help="YOLO confidence threshold (0–1)")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help="YOLO NMS IoU threshold (0–1)")

    # ── Preprocessing ────────────────────────────────────────────────
    p.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
                   metavar="N",
                   help="Process every Nth frame  (3 ≈ 10 FPS from 30 FPS)")
    p.add_argument("--no-flow", action="store_true",
                   help="Disable optical flow computation (faster on CPU)")

    # ── Anomaly ──────────────────────────────────────────────────────
    p.add_argument("--anomaly-model", default=DEFAULT_ANOMALY_MODEL,
                   help="Path to saved anomaly model (.pkl)")
    p.add_argument("--contamination", type=float,
                   default=DEFAULT_CONTAMINATION,
                   help="Isolation Forest contamination parameter")

    # ── Alert manager ────────────────────────────────────────────────
    p.add_argument("--alert-log", default=DEFAULT_ALERT_LOG,
                   help="Path to alert log JSON")
    p.add_argument("--cooldown", type=int, default=30,
                   help="Alert cooldown window in seconds")

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument("--save-frames", action="store_true",
                   help="Save annotated frames (bounding boxes) to disk")
    p.add_argument("--annotated-dir", default=DEFAULT_ANNOTATED_DIR,
                   help="Directory for annotated frames")
    p.add_argument("--no-results", action="store_true",
                   help="Skip saving session summary JSON")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="Directory for session summary JSON files")

    # ── Logging ───────────────────────────────────────────────────────
    p.add_argument("--log-file", default=DEFAULT_PIPELINE_LOG,
                   help="Path for pipeline log file")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging")

    # ── Limits ───────────────────────────────────────────────────────
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames (0 = no limit)")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = make_parser()
    args   = parser.parse_args()

    # Set up logging before anything else
    _setup_logging(log_file=args.log_file, verbose=args.verbose)

    try:
        config   = build_config(args)
        pipeline = BorderSurveillancePipeline(config)
        session  = pipeline.run()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        parser.print_help()
        return 1
    except Exception as exc:
        logger.error("Fatal pipeline error: %s", exc, exc_info=True)
        return 2

    # Exit code reflects whether any alerts were raised
    return 0 if session.alerts_raised == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
