"""
Object Detection Module
========================

Runs YOLOv8 detection on preprocessed frames from the Border Surveillance
pipeline.  Sits directly after preprocessing.py and feeds structured results
into anomaly.py.

Pipeline position:
    Video Input → preprocessing.py → [THIS MODULE] → anomaly.py → Alerts

Key design decisions:
    - BorderDetector is a class so the model loads ONCE and is reused for
      every frame (loading per-frame would be ~10× slower).
    - Every detection is returned as a plain dict so anomaly.py and
      pipeline.py can consume it without importing this module's types.
    - Confidence threshold defaults to 0.25 — low enough to catch weak
      signals; anomaly.py re-filters on its own threshold.
    - has_critical only flags "critical" threat level, NOT "high" — so the
      alert manager can distinguish between the two correctly.

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

# YOLO imported at module level so tests can patch 'detector.YOLO'.
# Wrapped in try/except so the module is importable in CI environments
# where ultralytics is not installed (dataclass + annotation tests still run).
try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match data.yaml exactly
# ---------------------------------------------------------------------------

CLASS_NAMES: Dict[int, str] = {
    0: "person",
    1: "vehicle",
    2: "crowd",
    3: "military_vehicle",
    4: "aircraft",
    5: "ship",
    6: "suspicious_object",
}

# Threat levels — 4 tiers used by alert_manager.py
# critical → immediate action, high → alert, medium → log, low → ignore
CLASS_THREAT: Dict[int, str] = {
    0: "medium",    # person
    1: "low",       # vehicle
    2: "high",      # crowd
    3: "critical",  # military_vehicle
    4: "high",      # aircraft
    5: "medium",    # ship
    6: "critical",  # suspicious_object
}

# BGR colours for bounding box annotation
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0,   255,   0),    # person           → green
    1: (255, 165,   0),    # vehicle          → orange
    2: (0,     0, 255),    # crowd            → red
    3: (128,   0, 128),    # military_vehicle → purple
    4: (255, 255,   0),    # aircraft         → yellow
    5: (255,   0, 255),    # ship             → magenta
    6: (0,   165, 255),    # suspicious_object → deep orange
}

DEFAULT_MODEL_PATH  = "models/border_yolo.pt"
FALLBACK_MODEL_PATH = None          
DEFAULT_CONFIDENCE  = 0.30
DEFAULT_IOU         = 0.45
FRAME_SIZE          = 640                   # expected input resolution


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Single object detection result for one frame.

    Attributes:
        class_id:     Integer class index matching CLASS_NAMES (0–6).
        class_name:   Human-readable class label.
        confidence:   YOLO detection confidence in [0, 1].
        bbox:         [x1, y1, x2, y2] pixel coords in the 640×640 frame.
        center_x:     Horizontal bbox centre, normalised to [0, 1].
        center_y:     Vertical bbox centre, normalised to [0, 1].
        width_norm:   Normalised box width  (0–1).
        height_norm:  Normalised box height (0–1).
        area_norm:    Normalised box area   (0–1).  Used as anomaly feature.
        threat_level: Risk tag — "low" / "medium" / "high" / "critical".
    """
    class_id:     int
    class_name:   str
    confidence:   float
    bbox:         List[float]    # [x1, y1, x2, y2]
    center_x:     float = 0.0
    center_y:     float = 0.0
    width_norm:   float = 0.0
    height_norm:  float = 0.0
    area_norm:    float = 0.0
    threat_level: str   = "low"

    def to_dict(self) -> dict:
        """Serialise to plain dict — used by anomaly.py and Cosmos DB."""
        return {
            "class_id":    self.class_id,
            "class_name":  self.class_name,
            "confidence":  round(self.confidence, 4),
            "bbox":        [round(v, 2) for v in self.bbox],
            "center_x":    round(self.center_x,    4),
            "center_y":    round(self.center_y,    4),
            "width_norm":  round(self.width_norm,  4),
            "height_norm": round(self.height_norm, 4),
            "area_norm":   round(self.area_norm,   6),
            "threat_level": self.threat_level,
        }


# ---------------------------------------------------------------------------
# FrameResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """
    All detections for a single preprocessed frame.

    Attributes:
        frame_id:        1-indexed position in the source video.
        timestamp:       Wall-clock time of detection (seconds since epoch).
        detections:      List of Detection objects found in this frame.
        detection_count: Total number of objects detected.
        has_critical:    True only if a "critical" threat_level object found.
                         NOTE: "high" threat does NOT set this flag —
                         alert_manager.py handles high/critical separately.
        has_high:        True if any "high" threat_level object found.
        motion_score:    Optical flow mean magnitude from preprocessing.py
                         (None if compute_flow was not enabled).
        inference_ms:    YOLO inference time for this frame (milliseconds).
    """
    frame_id:        int
    timestamp:       float
    detections:      List[Detection] = field(default_factory=list)
    detection_count: int   = 0
    has_critical:    bool  = False
    has_high:        bool  = False
    motion_score:    Optional[float] = None
    inference_ms:    float = 0.0

    def to_dict(self) -> dict:
        """Serialise to plain dict for pipeline.py / Cosmos DB."""
        return {
            "frame_id":        self.frame_id,
            "timestamp":       round(self.timestamp, 3),
            "detection_count": self.detection_count,
            "has_critical":    self.has_critical,
            "has_high":        self.has_high,
            "motion_score":    round(self.motion_score, 4)
                               if self.motion_score is not None else None,
            "inference_ms":    round(self.inference_ms, 2),
            "detections":      [d.to_dict() for d in self.detections],
        }


# ---------------------------------------------------------------------------
# BorderDetector
# ---------------------------------------------------------------------------

class BorderDetector:
    """
    YOLOv8-based object detector for border surveillance.

    The model loads once at construction time and is reused across all
    calls — never re-instantiate per frame.

    Args:
        model_path:     Path to custom trained weights (.pt).
                        Falls back to yolov8n.pt if file not found.
        confidence:     Minimum detection confidence (0–1).
        iou:            NMS IoU threshold (0–1).
        device:         "cpu" or "cuda:0".  Auto-detected if None.
        filter_classes: Optional set of class IDs to keep.  None = keep all.
                        e.g. {3, 6} to only detect military vehicles and
                        suspicious objects.

    Example:
        >>> detector = BorderDetector()
        >>> from preprocessing import extract_frames
        >>> for item in extract_frames("border_feed.mp4", frame_skip=3,
        ...                            compute_flow=True):
        ...     result = detector.detect(item)
        ...     if result.has_critical:
        ...         print(f"CRITICAL at frame {result.frame_id}")
    """

    def __init__(
        self,
        model_path:     str = DEFAULT_MODEL_PATH,
        confidence:     float = DEFAULT_CONFIDENCE,
        iou:            float = DEFAULT_IOU,
        device:         Optional[str] = None,
        filter_classes: Optional[Set[int]] = None,
    ) -> None:
        self.model_path     = model_path
        self.confidence     = confidence
        self.iou            = iou
        self.device         = device or self._detect_device()
        self.filter_classes = filter_classes   # None → keep all 7 classes

        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Return 'cuda:0' if a GPU is available, otherwise 'cpu'."""
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
        
    def _load_model(self):
        """
        Load YOLO weights from disk.

        If the custom model is missing, falls back to yolov8n.pt (generic
        COCO weights) so the pipeline can still run for development/testing.
        Called once in __init__.
        """
        path = self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: '{path}'\n"
                f"Run: cp models/runs/border_surveillance_v9_finetune/weights/best.pt models/border_yolo.pt"
            )
        try:
            if YOLO is None:
                raise RuntimeError("ultralytics is not installed.")
            model = YOLO(path)
            logger.info(
                "Loaded model: %s  |  device: %s  |  conf: %.2f",
                path, self.device, self.confidence,
            )
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc
    
    def _build_detection(self, box, frame_w: int, frame_h: int) -> Detection:
        """
        Convert a single YOLO result box into a Detection object.

        Uses actual frame_w / frame_h for normalisation so the caller
        does not have to assume a square 640×640 frame.
        """
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx   = ((x1 + x2) / 2) / frame_w
        cy   = ((y1 + y2) / 2) / frame_h
        w    = (x2 - x1) / frame_w
        h    = (y2 - y1) / frame_h
        area = w * h

        return Detection(
            class_id    = class_id,
            class_name  = CLASS_NAMES.get(class_id, f"class_{class_id}"),
            confidence  = confidence,
            bbox        = [x1, y1, x2, y2],
            center_x    = cx,
            center_y    = cy,
            width_norm  = w,
            height_norm = h,
            area_norm   = area,
            threat_level = CLASS_THREAT.get(class_id, "low"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame_item: dict) -> FrameResult:
        """
        Run YOLO detection on a single frame dict from extract_frames().

        Args:
            frame_item: Dict yielded by preprocessing.extract_frames().
                        Required keys: "frame_id", "frame".
                        Optional keys: "motion_score".

        Returns:
            FrameResult with all detections for this frame.

        Example:
            >>> detector = BorderDetector()
            >>> from preprocessing import extract_frames
            >>> for item in extract_frames("feed.mp4"):
            ...     result = detector.detect(item)
            ...     if result.has_critical:
            ...         print(f"ALERT at frame {result.frame_id}")
        """
        frame_id     = frame_item["frame_id"]
        frame        = frame_item["frame"]
        motion_score = frame_item.get("motion_score")

        # Ensure uint8 for YOLO — preprocessing.py can return float32
        # if normalize=True was used.
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        frame_h, frame_w = frame.shape[:2]

        t_start = time.time()

        yolo_results = self._model(
            frame,
            conf    = self.confidence,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
        )

        inference_ms = (time.time() - t_start) * 1000

        # Parse detections
        detections: List[Detection] = []
        for res in yolo_results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                det = self._build_detection(box, frame_w, frame_h)

                # Apply class filter if configured
                if self.filter_classes and det.class_id not in self.filter_classes:
                    continue

                detections.append(det)

        # Separate has_critical and has_high — important for alert_manager
        has_critical = any(d.threat_level == "critical" for d in detections)
        has_high     = any(d.threat_level == "high"     for d in detections)

        frame_result = FrameResult(
            frame_id        = frame_id,
            timestamp       = time.time(),
            detections      = detections,
            detection_count = len(detections),
            has_critical    = has_critical,
            has_high        = has_high,
            motion_score    = motion_score,
            inference_ms    = inference_ms,
        )

        if detections:
            logger.debug(
                "Frame %d → %d detections [%s] | %.1f ms",
                frame_id,
                len(detections),
                ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in detections),
                inference_ms,
            )

        return frame_result

    def process_video(
        self,
        video_path:     str,
        frame_skip:     int  = 3,
        compute_flow:   bool = True,
        save_annotated: bool = False,
        output_dir:     str  = "data/detections",
        show_progress:  bool = True,
    ) -> Generator[FrameResult, None, None]:
        """
        End-to-end generator: video path → stream of FrameResult objects.

        Internally calls extract_frames() from preprocessing.py so the caller
        does not need to manage VideoCapture objects.

        Args:
            video_path:     Path to input video file.
            frame_skip:     Process every Nth frame (3 = ~10 FPS from 30 FPS).
            compute_flow:   Pass True to include motion_score in results.
            save_annotated: Save annotated frames (bounding boxes) to disk.
            output_dir:     Destination folder for annotated frames.
            show_progress:  Show a tqdm progress bar.

        Yields:
            FrameResult for each processed frame, in order.

        Example:
            >>> detector = BorderDetector()
            >>> results = list(detector.process_video("border_feed.mp4",
            ...                                        frame_skip=5))
            >>> print(detector.get_stats(results))
        """
        from preprocessing import extract_frames

        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)

        frame_count     = 0
        detection_total = 0

        for frame_item in extract_frames(
            video_path,
            frame_skip    = frame_skip,
            compute_flow  = compute_flow,
            show_progress = show_progress,
        ):
            result = self.detect(frame_item)

            if save_annotated:
                annotated = self.annotate_frame(frame_item["frame"], result)
                out_path  = os.path.join(
                    output_dir, f"frame_{result.frame_id:06d}.jpg"
                )
                cv2.imwrite(out_path, annotated)

            frame_count     += 1
            detection_total += result.detection_count
            yield result

        logger.info(
            "Processed %d frames | %d total detections | avg %.1f per frame",
            frame_count,
            detection_total,
            detection_total / max(frame_count, 1),
        )

    def annotate_frame(
        self,
        frame:  np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """
        Draw bounding boxes and info overlay on a copy of the frame.

        Handles both uint8 and float32 input frames gracefully.

        Args:
            frame:  640×640 frame (uint8 BGR or float32 normalised).
            result: FrameResult from detect().

        Returns:
            Annotated uint8 BGR frame copy.  Original is never modified.
        """
        # Convert to uint8 if normalised — cv2 drawing silently fails on float32
        if frame.dtype != np.uint8:
            annotated = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            annotated = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS.get(det.class_id, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label pill background + text
            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        # Top-left info overlay
        info = (
            f"Frame {result.frame_id} | "
            f"Objects: {result.detection_count} | "
            f"{result.inference_ms:.0f} ms"
        )
        if result.motion_score is not None:
            info += f" | Motion: {result.motion_score:.2f}"
        if result.has_critical:
            info += " | ⚠ CRITICAL"
        elif result.has_high:
            info += " | ! HIGH"

        cv2.putText(
            annotated, info,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        return annotated

    def get_stats(self, results: List[FrameResult]) -> dict:
        """
        Aggregate a list of FrameResults into summary statistics.

        Used by the dashboard and Cosmos DB reporting.

        Args:
            results: List of FrameResult objects (e.g. from process_video()).

        Returns:
            dict with keys:
                total_frames, total_detections, detections_per_frame,
                class_counts, critical_frames, high_frames,
                avg_confidence, avg_inference_ms.
        """
        if not results:
            return {}

        class_counts: Dict[str, int] = {}
        confidences:  List[float]    = []
        inf_times:    List[float]    = []
        critical = 0
        high     = 0

        for r in results:
            if r.has_critical:
                critical += 1
            if r.has_high:
                high += 1
            inf_times.append(r.inference_ms)
            for d in r.detections:
                class_counts[d.class_name] = (
                    class_counts.get(d.class_name, 0) + 1
                )
                confidences.append(d.confidence)

        total_det = sum(r.detection_count for r in results)

        return {
            "total_frames":         len(results),
            "total_detections":     total_det,
            "detections_per_frame": round(total_det / len(results), 2),
            "class_counts":         class_counts,
            "critical_frames":      critical,
            "high_frames":          high,
            "avg_confidence":       round(sum(confidences) / len(confidences), 4)
                                    if confidences else 0.0,
            "avg_inference_ms":     round(sum(inf_times) / len(inf_times), 2)
                                    if inf_times else 0.0,
        }


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python src/detector.py path/to/video.mp4)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Border Surveillance AI — Detector Module")
    print("=" * 55)

    video = sys.argv[1] if len(sys.argv) > 1 else "data/test_videos/sample.mp4"

    if not os.path.exists(video):
        print(f"Video not found: {video}")
        print("Usage: python src/detector.py path/to/video.mp4")
        sys.exit(1)

    detector = BorderDetector()
    results: List[FrameResult] = []

    print(f"Processing: {video}\n")

    for result in detector.process_video(
        video,
        frame_skip     = 5,
        compute_flow   = True,
        save_annotated = True,
        output_dir     = "data/detections",
        show_progress  = True,
    ):
        results.append(result)
        if result.has_critical:
            detected = ", ".join(
                f"{d.class_name}({d.confidence:.2f})"
                for d in result.detections
                if d.threat_level == "critical"
            )
            print(f"  [CRITICAL] Frame {result.frame_id}: {detected}")
        elif result.has_high:
            detected = ", ".join(
                f"{d.class_name}({d.confidence:.2f})"
                for d in result.detections
                if d.threat_level == "high"
            )
            print(f"  [HIGH]     Frame {result.frame_id}: {detected}")

    print("\n" + "=" * 55)
    print("DETECTION SUMMARY")
    print("=" * 55)
    stats = detector.get_stats(results)
    for key, val in stats.items():
        print(f"  {key:<25} {val}")
    print(f"\nAnnotated frames → data/detections/")
    print("Next: python src/anomaly.py")
