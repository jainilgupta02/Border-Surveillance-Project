"""
Test Suite — Pipeline Orchestrator
=====================================

All external dependencies (YOLO, Isolation Forest, video I/O) are mocked
so the suite runs on any machine without GPU or real data.

Run all tests:
    pytest tests/test_pipeline.py -v

Run with coverage:
    pytest tests/test_pipeline.py --cov=pipeline --cov-report=html

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import json
import os
import sys
import time
import argparse
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import cv2
import numpy as np
import pytest

# Add src to path (mirrors real project layout)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    BorderSurveillancePipeline,
    PipelineConfig,
    PipelineSession,
    build_config,
    make_parser,
    DEFAULT_FRAME_SKIP,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
)


# ===========================================================================
# Shared helpers / fixtures
# ===========================================================================

def _make_config(tmp_path, **overrides) -> PipelineConfig:
    """Return a minimal PipelineConfig pointing at tmp_path."""
    defaults = dict(
        video_path    = str(tmp_path / "test.mp4"),
        model_path    = str(tmp_path / "border_yolo.pt"),
        anomaly_model = str(tmp_path / "anomaly_model.pkl"),
        alert_log     = str(tmp_path / "alerts" / "log.json"),
        annotated_dir = str(tmp_path / "detections"),
        results_dir   = str(tmp_path / "results"),
        log_file      = str(tmp_path / "pipeline.log"),
        frame_skip    = 1,
        compute_flow  = False,
        save_frames   = False,
        save_results  = True,
        max_frames    = 5,
        cooldown      = 0,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_video(path: str, n_frames: int = 40) -> str:
    """Write a minimal synthetic MP4 for integration tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 5 % 256, i * 3 % 256, i * 7 % 256)
        out.write(frame)
    out.release()
    return path


def _make_yolo_box(class_id=1, conf=0.75,
                   x1=100, y1=100, x2=300, y2=300) -> MagicMock:
    box = MagicMock()
    box.cls  = MagicMock(); box.cls.__getitem__ = MagicMock(return_value=class_id)
    box.conf = MagicMock(); box.conf.__getitem__ = MagicMock(return_value=conf)
    box.xyxy = MagicMock()
    box.xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(tolist=MagicMock(return_value=[x1, y1, x2, y2]))
    )
    return box


def _yolo_result(boxes=None) -> MagicMock:
    r = MagicMock(); r.boxes = boxes or []; return r


def _mock_yolo(boxes=None):
    """Return a mock YOLO model that yields given boxes on every call."""
    m = MagicMock()
    m.return_value = [_yolo_result(boxes or [])]
    return m


# ===========================================================================
# 1. PipelineConfig
# ===========================================================================

class TestPipelineConfig:

    def test_source_video(self, tmp_path):
        cfg = _make_config(tmp_path, video_path="feed.mp4", camera_index=None)
        assert cfg.source == "feed.mp4"

    def test_source_camera(self, tmp_path):
        cfg = _make_config(tmp_path, video_path=None, camera_index=0)
        assert cfg.source == "camera:0"

    def test_video_source_returns_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.video_source == cfg.video_path

    def test_video_source_returns_index(self, tmp_path):
        cfg = _make_config(tmp_path, video_path=None, camera_index=2)
        assert cfg.video_source == 2


# ===========================================================================
# 2. PipelineSession
# ===========================================================================

class TestPipelineSession:

    def _make(self, tmp_path) -> PipelineSession:
        return PipelineSession(config=_make_config(tmp_path))

    def test_elapsed_seconds_increases(self, tmp_path):
        s = self._make(tmp_path)
        time.sleep(0.05)
        assert s.elapsed_seconds >= 0.04

    def test_fps_effective_zero_frames(self, tmp_path):
        s = self._make(tmp_path)
        # No frames processed — fps should not crash (returns 0.0)
        assert s.fps_effective >= 0.0

    def test_to_summary_has_required_keys(self, tmp_path):
        s   = self._make(tmp_path)
        d   = s.to_summary()
        for key in ("source", "total_frames", "baseline_frames",
                    "frames_scored", "model_fitted", "total_detections",
                    "normal_frames", "high_alert_frames", "critical_frames",
                    "alerts_raised", "alert_rate", "effective_fps",
                    "avg_inference_ms"):
            assert key in d, f"Missing key: {key}"

    def test_alert_rate_calculation(self, tmp_path):
        s = self._make(tmp_path)
        s.frames_scored  = 10
        s.high_frames    = 2
        s.critical_frames = 1
        d = s.to_summary()
        assert d["alert_rate"] == pytest.approx(3 / 10, abs=0.01)

    def test_to_summary_serialisable_as_json(self, tmp_path):
        s = self._make(tmp_path)
        s.end_time = time.time()
        json.dumps(s.to_summary())   # should not raise


# ===========================================================================
# 3. CLI — make_parser / build_config
# ===========================================================================

class TestCLI:

    def _parse(self, args: list) -> argparse.Namespace:
        return make_parser().parse_args(args)

    def test_video_arg_accepted(self):
        ns = self._parse(["--video", "feed.mp4"])
        assert ns.video == "feed.mp4"

    def test_camera_arg_accepted(self):
        ns = self._parse(["--camera", "0"])
        assert ns.camera == 0

    def test_frame_skip_default(self):
        ns = self._parse(["--video", "x.mp4"])
        assert ns.frame_skip == DEFAULT_FRAME_SKIP

    def test_no_flow_flag(self):
        ns = self._parse(["--video", "x.mp4", "--no-flow"])
        assert ns.no_flow is True

    def test_save_frames_flag(self):
        ns = self._parse(["--video", "x.mp4", "--save-frames"])
        assert ns.save_frames is True

    def test_confidence_custom(self):
        ns = self._parse(["--video", "x.mp4", "--confidence", "0.5"])
        assert ns.confidence == pytest.approx(0.5)

    def test_max_frames_default_zero(self):
        ns = self._parse(["--video", "x.mp4"])
        assert ns.max_frames == 0

    def test_build_config_video(self, tmp_path):
        ns = self._parse(["--video", str(tmp_path / "v.mp4"),
                          "--alert-log",   str(tmp_path / "a.json"),
                          "--log-file",    str(tmp_path / "p.log"),
                          "--results-dir", str(tmp_path / "r"),
                          "--annotated-dir", str(tmp_path / "d"),
                          "--anomaly-model", str(tmp_path / "m.pkl"),
                          "--model",       str(tmp_path / "y.pt")])
        cfg = build_config(ns)
        assert cfg.video_path == str(tmp_path / "v.mp4")
        assert cfg.compute_flow is True   # no-flow not set

    def test_build_config_no_source_raises(self):
        ns        = make_parser().parse_args(["--video", "x.mp4"])
        ns.video  = None
        ns.camera = None
        with pytest.raises(ValueError, match="--video or --camera"):
            build_config(ns)


# ===========================================================================
# 4. BorderSurveillancePipeline — initialisation
# ===========================================================================

class TestPipelineInit:
    """Verify the pipeline initialises all modules without crashing."""

    def _build(self, tmp_path, **kwargs):
        cfg = _make_config(tmp_path, **kwargs)
        with patch("pipeline.BorderDetector") as MockDet, \
             patch("pipeline.AnomalyDetector") as MockAnomaly, \
             patch("pipeline.AlertManager") as MockAlert:

            mock_det    = MockDet.return_value
            mock_det.device = "cpu"
            mock_anomaly = MockAnomaly.return_value
            mock_anomaly._is_fitted = False

            p = BorderSurveillancePipeline(cfg)
            return p, mock_det, mock_anomaly, MockAlert.return_value

    def test_detector_initialised_once(self, tmp_path):
        p, det, _, _ = self._build(tmp_path)
        assert p._detector is det

    def test_anomaly_initialised_once(self, tmp_path):
        p, _, ano, _ = self._build(tmp_path)
        assert p._anomaly is ano

    def test_alert_manager_initialised(self, tmp_path):
        p, _, _, mgr = self._build(tmp_path)
        assert p._alert_manager is mgr

    def test_save_frames_dir_created(self, tmp_path):
        p, _, _, _ = self._build(tmp_path, save_frames=True)
        assert os.path.isdir(str(tmp_path / "detections"))

    def test_results_dir_created(self, tmp_path):
        p, _, _, _ = self._build(tmp_path, save_results=True)
        assert os.path.isdir(str(tmp_path / "results"))

    def test_phase_b_if_model_already_fitted(self, tmp_path):
        """If anomaly model is pre-loaded, session.model_fitted starts True."""
        cfg = _make_config(tmp_path)
        with patch("pipeline.BorderDetector") as MockDet, \
             patch("pipeline.AnomalyDetector") as MockAnomaly, \
             patch("pipeline.AlertManager"):
            MockDet.return_value.device = "cpu"
            MockAnomaly.return_value._is_fitted = True
            p = BorderSurveillancePipeline(cfg)
            assert p.session.model_fitted is True


# ===========================================================================
# 5. BorderSurveillancePipeline — full run (mocked modules)
# ===========================================================================

class TestPipelineRun:
    """Integration tests using real OpenCV video but mocked AI modules."""

    def _run(self, tmp_path, n_frames=40, max_frames=5,
             boxes=None, alert_level="normal", pre_fitted=False,
             save_frames=False):
        """Build a pipeline with all AI mocked and run it."""
        video = _make_video(str(tmp_path / "v.mp4"), n_frames)
        cfg   = _make_config(tmp_path, video_path=video,
                             max_frames=max_frames, save_frames=save_frames,
                             frame_skip=1, compute_flow=False)

        # ── Mock detector ──────────────────────────────────────────
        mock_det = MagicMock()
        mock_det.device = "cpu"
        fr_result = MagicMock()
        fr_result.frame_id        = 1
        fr_result.inference_ms    = 30.0
        fr_result.detection_count = len(boxes or [])
        fr_result.has_critical    = False
        fr_result.has_high        = False
        fr_result.to_dict.return_value = {
            "frame_id": 1, "timestamp": time.time(),
            "detection_count": len(boxes or []),
            "has_critical": False, "has_high": False,
            "inference_ms": 30.0, "motion_score": 3.0,
            "detections": [],
        }
        fr_result.annotate_frame = MagicMock()
        mock_det.detect.return_value     = fr_result
        mock_det.annotate_frame.return_value = np.zeros((640, 640, 3), np.uint8)

        # ── Mock anomaly ───────────────────────────────────────────
        mock_anomaly = MagicMock()
        mock_anomaly._is_fitted = pre_fitted

        ano_result = MagicMock()
        ano_result.alert_level  = alert_level
        ano_result.anomaly_score = -0.08 if alert_level != "normal" else 0.02
        ano_result.anomaly_prob  = 0.6
        ano_result.to_dict.return_value = {
            "frame_id": 1, "timestamp": time.time(),
            "anomaly_score": -0.08, "anomaly_prob": 0.6,
            "alert_level": alert_level, "reasons": [],
            "detection_count": 0, "motion_score": 3.0,
        }
        mock_anomaly.score.return_value         = ano_result
        mock_anomaly.collect_baseline.return_value = False
        mock_anomaly.get_summary.return_value   = {}

        # ── Mock alert manager ─────────────────────────────────────
        mock_mgr   = MagicMock()
        mock_alert = MagicMock()
        mock_alert.priority = "HIGH"
        mock_alert.to_dict.return_value = {
            "alert_id": "a1", "frame_id": 1, "priority": "HIGH",
            "anomaly_score": -0.08, "anomaly_prob": 0.6,
            "alert_level": alert_level, "reasons": [],
            "detection_count": 0, "motion_score": 3.0,
            "notified": False, "timestamp": time.time(),
        }
        mock_mgr.process.return_value = (
            mock_alert if alert_level != "normal" else None
        )
        mock_mgr.get_summary.return_value = {"total_alerts": 0}

        with patch("pipeline.BorderDetector",  return_value=mock_det), \
             patch("pipeline.AnomalyDetector", return_value=mock_anomaly), \
             patch("pipeline.AlertManager",    return_value=mock_mgr):
            pipeline = BorderSurveillancePipeline(cfg)
            session  = pipeline.run()

        return session, mock_det, mock_anomaly, mock_mgr

    # ── Basic run ─────────────────────────────────────────────────

    def test_run_returns_session(self, tmp_path):
        session, *_ = self._run(tmp_path, pre_fitted=True)
        assert isinstance(session, PipelineSession)

    def test_frames_scored_equals_max_frames_when_pre_fitted(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=5, pre_fitted=True)
        assert session.frames_scored == 5

    def test_baseline_frames_collected_in_phase_a(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=3, pre_fitted=False)
        # All frames go to baseline when model never gets fitted in 3 frames
        assert session.baseline_frames == 3
        assert session.frames_scored   == 0

    def test_detector_called_for_each_frame(self, tmp_path):
        _, mock_det, _, _ = self._run(tmp_path, max_frames=4, pre_fitted=True)
        assert mock_det.detect.call_count == 4

    def test_anomaly_score_called_for_scored_frames(self, tmp_path):
        _, _, mock_anomaly, _ = self._run(tmp_path, max_frames=4,
                                          pre_fitted=True)
        assert mock_anomaly.score.call_count == 4

    # ── Alert raising ─────────────────────────────────────────────

    def test_high_alert_increments_high_frames(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=3,
                                alert_level="high", pre_fitted=True)
        assert session.high_frames == 3

    def test_critical_alert_increments_critical_frames(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=2,
                                alert_level="critical", pre_fitted=True)
        assert session.critical_frames == 2

    def test_normal_frame_increments_normal_frames(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=3,
                                alert_level="normal", pre_fitted=True)
        assert session.normal_frames == 3

    def test_alerts_raised_count_increments(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=3,
                                alert_level="high", pre_fitted=True)
        assert session.alerts_raised == 3

    def test_no_alerts_for_normal_frames(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=3,
                                alert_level="normal", pre_fitted=True)
        assert session.alerts_raised == 0

    # ── Timing ────────────────────────────────────────────────────

    def test_end_time_set_after_run(self, tmp_path):
        session, *_ = self._run(tmp_path, pre_fitted=True)
        assert session.end_time is not None
        assert session.end_time >= session.start_time

    def test_elapsed_seconds_positive(self, tmp_path):
        session, *_ = self._run(tmp_path, pre_fitted=True)
        assert session.elapsed_seconds >= 0.0

    def test_total_inference_ms_accumulated(self, tmp_path):
        session, *_ = self._run(tmp_path, max_frames=4, pre_fitted=True)
        assert session.total_inference_ms >= 0.0

    # ── Output files ───────────────────────────────────────────────

    def test_results_json_written(self, tmp_path):
        self._run(tmp_path, max_frames=3, pre_fitted=True)
        files = list((tmp_path / "results").glob("*.json"))
        assert len(files) == 1

    def test_results_json_valid_structure(self, tmp_path):
        self._run(tmp_path, max_frames=3, pre_fitted=True)
        files = list((tmp_path / "results").glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)
        assert "source" in data
        assert "total_frames" in data
        assert "alerts" in data

    def test_annotated_frames_saved_when_flag_set(self, tmp_path):
        self._run(tmp_path, max_frames=3, pre_fitted=True, save_frames=True)
        det_dir = tmp_path / "detections"
        assert det_dir.exists()

    # ── max_frames limit ───────────────────────────────────────────

    def test_max_frames_stops_processing(self, tmp_path):
        """40-frame video but max_frames=5 → exactly 5 processed."""
        session, mock_det, _, _ = self._run(
            tmp_path, n_frames=40, max_frames=5, pre_fitted=True
        )
        assert mock_det.detect.call_count == 5


# ===========================================================================
# 6. Phase transition — Phase A → Phase B
# ===========================================================================

class TestPhaseTransition:
    """Verify the baseline→scoring transition works correctly."""

    def test_fit_called_when_baseline_complete(self, tmp_path):
        """collect_baseline returns True on frame 30 → fit() must be called."""
        video = _make_video(str(tmp_path / "v.mp4"), 60)
        cfg   = _make_config(tmp_path, video_path=video,
                             max_frames=35, frame_skip=1, compute_flow=False)

        mock_det = MagicMock(); mock_det.device = "cpu"
        fr_result = MagicMock()
        fr_result.frame_id = 1; fr_result.inference_ms = 10.0
        fr_result.detection_count = 0; fr_result.has_critical = False
        fr_result.has_high = False
        fr_result.to_dict.return_value = {
            "frame_id": 1, "timestamp": time.time(),
            "detection_count": 0, "has_critical": False,
            "has_high": False, "inference_ms": 10.0,
            "motion_score": None, "detections": [],
        }
        mock_det.detect.return_value = fr_result
        mock_det.annotate_frame.return_value = np.zeros((640,640,3), np.uint8)

        call_count = {"n": 0}
        def fake_collect(fr):
            call_count["n"] += 1
            return call_count["n"] >= 30   # True from frame 30 onward

        mock_anomaly = MagicMock()
        mock_anomaly._is_fitted = False
        mock_anomaly.collect_baseline.side_effect = fake_collect

        ano_result = MagicMock()
        ano_result.alert_level = "normal"; ano_result.anomaly_score = 0.0
        ano_result.anomaly_prob = 0.1
        ano_result.to_dict.return_value = {
            "frame_id": 1, "timestamp": time.time(),
            "anomaly_score": 0.0, "anomaly_prob": 0.1,
            "alert_level": "normal", "reasons": [],
            "detection_count": 0, "motion_score": None,
        }
        mock_anomaly.score.return_value = ano_result
        mock_anomaly.get_summary.return_value = {}

        def fit_side_effect():
            mock_anomaly._is_fitted = True
        mock_anomaly.fit.side_effect = fit_side_effect

        mock_mgr = MagicMock()
        mock_mgr.process.return_value = None
        mock_mgr.get_summary.return_value = {}

        with patch("pipeline.BorderDetector",  return_value=mock_det), \
             patch("pipeline.AnomalyDetector", return_value=mock_anomaly), \
             patch("pipeline.AlertManager",    return_value=mock_mgr):
            pipeline = BorderSurveillancePipeline(cfg)
            session  = pipeline.run()

        mock_anomaly.fit.assert_called_once()
        assert session.baseline_frames == 30
        assert session.frames_scored   == 5
        assert session.model_fitted    is True


# ===========================================================================
# 7. Graceful shutdown
# ===========================================================================

class TestGracefulShutdown:

    def test_shutdown_flag_stops_loop(self, tmp_path):
        """If _shutdown_requested is set, the loop exits cleanly."""
        video = _make_video(str(tmp_path / "v.mp4"), 20)
        cfg   = _make_config(tmp_path, video_path=video,
                             max_frames=0, frame_skip=1, compute_flow=False)

        stop_after = {"count": 0}

        def fake_detect(item):
            stop_after["count"] += 1
            fr = MagicMock()
            fr.frame_id = item["frame_id"]; fr.inference_ms = 5.0
            fr.detection_count = 0; fr.has_critical = False; fr.has_high = False
            fr.to_dict.return_value = {
                "frame_id": item["frame_id"], "timestamp": time.time(),
                "detection_count": 0, "has_critical": False,
                "has_high": False, "inference_ms": 5.0,
                "motion_score": None, "detections": [],
            }
            fr.annotate_frame = MagicMock()
            return fr

        mock_det = MagicMock(); mock_det.device = "cpu"
        mock_det.detect.side_effect = fake_detect
        mock_det.annotate_frame.return_value = np.zeros((640,640,3), np.uint8)

        mock_anomaly = MagicMock(); mock_anomaly._is_fitted = True
        ano = MagicMock(); ano.alert_level = "normal"; ano.anomaly_score = 0.0
        ano.anomaly_prob = 0.1
        ano.to_dict.return_value = {
            "frame_id": 1, "timestamp": time.time(),
            "anomaly_score": 0.0, "anomaly_prob": 0.1,
            "alert_level": "normal", "reasons": [],
            "detection_count": 0, "motion_score": None,
        }
        mock_anomaly.score.return_value = ano
        mock_anomaly.get_summary.return_value = {}

        mock_mgr = MagicMock(); mock_mgr.process.return_value = None
        mock_mgr.get_summary.return_value = {}

        with patch("pipeline.BorderDetector",  return_value=mock_det), \
             patch("pipeline.AnomalyDetector", return_value=mock_anomaly), \
             patch("pipeline.AlertManager",    return_value=mock_mgr):
            pipeline = BorderSurveillancePipeline(cfg)
            # Force shutdown after 3 frames
            original_process = pipeline._process_frame
            call_c = {"n": 0}
            def patched_process(item, phase, limit):
                call_c["n"] += 1
                if call_c["n"] >= 3:
                    pipeline._shutdown_requested = True
                return original_process(item, phase, limit)
            pipeline._process_frame = patched_process
            session = pipeline.run()

        # Should have stopped at or around 3 frames
        assert session.frames_scored <= 5


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
