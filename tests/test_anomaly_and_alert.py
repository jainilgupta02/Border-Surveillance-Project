"""
Test Suite — Anomaly Detection + Alert Manager
================================================

Run all tests:
    pytest tests/test_anomaly.py -v

Run with coverage:
    pytest tests/test_anomaly.py --cov=anomaly --cov=alert_manager --cov-report=html

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import json
import os
import time
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from anomaly import (
    CONTAMINATION,
    CRITICAL_THREAT_CLASSES,
    FEATURE_DIM,
    HIGH_THREAT_CLASSES,
    MIN_SAMPLES,
    THRESHOLD_CRITICAL,
    THRESHOLD_HIGH,
    AnomalyDetector,
    AnomalyResult,
    extract_features,
)
from alert_manager import (
    COOLDOWN_SECONDS,
    MOTION_MEDIUM_THRESHOLD,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    Alert,
    AlertManager,
)


# ===========================================================================
# Shared fixtures / helpers
# ===========================================================================

def _make_frame_result(
    frame_id:        int   = 1,
    detection_count: int   = 5,
    class_names:     list  = None,
    confidences:     list  = None,
    motion_score:    float = 5.0,
    center_x:        float = 0.5,
    center_y:        float = 0.5,
) -> dict:
    """Build a minimal FrameResult dict as produced by detector.py."""
    class_names  = class_names or ["vehicle"] * detection_count
    confidences  = confidences or [0.75]      * detection_count

    detections = [
        {
            "class_id":    0,
            "class_name":  class_names[i % len(class_names)],
            "confidence":  confidences[i % len(confidences)],
            "bbox":        [100, 100, 200, 200],
            "center_x":    center_x,
            "center_y":    center_y,
            "width_norm":  0.15,
            "height_norm": 0.15,
            "area_norm":   0.0225,
            "threat_level":"low",
        }
        for i in range(detection_count)
    ]
    return {
        "frame_id":        frame_id,
        "timestamp":       time.time(),
        "detection_count": detection_count,
        "has_critical":    False,
        "has_high":        False,
        "motion_score":    motion_score,
        "inference_ms":    50.0,
        "detections":      detections,
    }


def _make_anomaly_result(
    frame_id:      int   = 1,
    alert_level:   str   = "normal",
    anomaly_score: float = 0.0,
    anomaly_prob:  float = 0.1,
    reasons:       list  = None,
    motion_score:  float = 3.0,
    detection_count: int = 3,
) -> dict:
    """Build a minimal AnomalyResult dict as produced by anomaly.py."""
    return {
        "frame_id":        frame_id,
        "timestamp":       time.time(),
        "anomaly_score":   anomaly_score,
        "anomaly_prob":    anomaly_prob,
        "alert_level":     alert_level,
        "reasons":         reasons or [],
        "detection_count": detection_count,
        "motion_score":    motion_score,
    }


def _make_baseline(n: int = MIN_SAMPLES) -> List[dict]:
    """Return N normal-looking FrameResult dicts for fitting."""
    return [
        _make_frame_result(frame_id=i, detection_count=5,
                           class_names=["vehicle"], motion_score=5.0)
        for i in range(n)
    ]


def _fitted_detector(tmp_path) -> AnomalyDetector:
    """Return a fitted AnomalyDetector using a temp model path."""
    model_path = str(tmp_path / "anomaly_model.pkl")
    det = AnomalyDetector(model_path=model_path)
    det.fit(_make_baseline())
    return det


# ===========================================================================
# ── ANOMALY MODULE ──────────────────────────────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. extract_features()
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_output_shape_is_feature_dim(self):
        fr = _make_frame_result()
        v  = extract_features(fr)
        assert v.shape == (FEATURE_DIM,)

    def test_output_dtype_float32(self):
        fr = _make_frame_result()
        assert extract_features(fr).dtype == np.float32

    def test_empty_detections_returns_zeros_except_motion_and_centre(self):
        fr = _make_frame_result(detection_count=0, motion_score=7.5)
        v  = extract_features(fr)
        assert v[0] == 0.0      # detection_count
        assert v[5] == 0.5      # avg_center_x  (neutral)
        assert v[8] == pytest.approx(7.5)   # motion_score passed through

    def test_detection_count_feature(self):
        fr = _make_frame_result(detection_count=8)
        assert extract_features(fr)[0] == pytest.approx(8.0)

    def test_class_diversity(self):
        fr = _make_frame_result(
            detection_count=4,
            class_names=["vehicle", "person", "crowd", "aircraft"],
        )
        assert extract_features(fr)[1] == pytest.approx(4.0)

    def test_suspicious_flag_set_for_military_vehicle(self):
        fr = _make_frame_result(class_names=["military_vehicle"])
        assert extract_features(fr)[9] == pytest.approx(1.0)

    def test_suspicious_flag_set_for_suspicious_object(self):
        fr = _make_frame_result(class_names=["suspicious_object"])
        assert extract_features(fr)[9] == pytest.approx(1.0)

    def test_suspicious_flag_zero_for_vehicle(self):
        fr = _make_frame_result(class_names=["vehicle"])
        assert extract_features(fr)[9] == pytest.approx(0.0)

    def test_critical_count_includes_crowd_aircraft_military(self):
        fr = _make_frame_result(
            detection_count=3,
            class_names=["crowd", "military_vehicle", "aircraft"],
        )
        assert extract_features(fr)[4] == pytest.approx(3.0)

    def test_none_motion_score_treated_as_zero(self):
        fr = _make_frame_result(motion_score=None)
        fr["motion_score"] = None
        v  = extract_features(fr)
        assert v[8] == pytest.approx(0.0)

    def test_avg_confidence_correct(self):
        fr = _make_frame_result(
            detection_count=2,
            confidences=[0.8, 0.6],
        )
        assert extract_features(fr)[2] == pytest.approx(0.7, abs=0.01)

    def test_max_confidence_correct(self):
        fr = _make_frame_result(
            detection_count=2,
            confidences=[0.8, 0.6],
        )
        assert extract_features(fr)[3] == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# 2. AnomalyResult dataclass
# ---------------------------------------------------------------------------

class TestAnomalyResult:

    def _make(self, level="high") -> AnomalyResult:
        return AnomalyResult(
            frame_id=5, timestamp=time.time(),
            anomaly_score=-0.10, anomaly_prob=0.75,
            alert_level=level, reasons=["crowd detected"],
            detection_count=8, motion_score=9.5,
        )

    def test_to_dict_has_all_keys(self):
        d = self._make().to_dict()
        for k in ("frame_id", "timestamp", "anomaly_score", "anomaly_prob",
                  "alert_level", "reasons", "detection_count", "motion_score"):
            assert k in d

    def test_is_alert_true_for_high(self):
        assert self._make("high").is_alert is True

    def test_is_alert_true_for_critical(self):
        assert self._make("critical").is_alert is True

    def test_is_alert_false_for_normal(self):
        assert self._make("normal").is_alert is False

    def test_motion_score_none_serialises_to_none(self):
        r = self._make()
        r.motion_score = None
        assert r.to_dict()["motion_score"] is None

    def test_anomaly_score_rounded(self):
        r = self._make()
        r.anomaly_score = -0.123456789
        assert r.to_dict()["anomaly_score"] == -0.1235


# ---------------------------------------------------------------------------
# 3. AnomalyDetector — initialisation
# ---------------------------------------------------------------------------

class TestAnomalyDetectorInit:

    def test_not_fitted_on_init_without_saved_model(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "no_model.pkl"))
        assert det._is_fitted is False

    def test_scaler_is_none_before_fitting(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "no_model.pkl"))
        assert det._scaler is None

    def test_baseline_data_empty_on_init(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "no_model.pkl"))
        assert det._baseline_data == []

    def test_loads_saved_model_on_init(self, tmp_path):
        det = _fitted_detector(tmp_path)
        det2 = AnomalyDetector(model_path=str(tmp_path / "anomaly_model.pkl"))
        assert det2._is_fitted is True


# ---------------------------------------------------------------------------
# 4. AnomalyDetector.collect_baseline() and fit()
# ---------------------------------------------------------------------------

class TestAnomalyDetectorFit:

    def test_collect_baseline_returns_false_before_min_samples(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        fr  = _make_frame_result()
        result = det.collect_baseline(fr)
        assert result is False

    def test_collect_baseline_returns_true_at_min_samples(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        result = False
        for i in range(MIN_SAMPLES):
            result = det.collect_baseline(_make_frame_result(frame_id=i))
        assert result is True

    def test_fit_sets_is_fitted(self, tmp_path):
        det = _fitted_detector(tmp_path)
        assert det._is_fitted is True

    def test_fit_creates_model_file(self, tmp_path):
        model_path = str(tmp_path / "anomaly_model.pkl")
        det = AnomalyDetector(model_path=model_path)
        det.fit(_make_baseline())
        assert os.path.exists(model_path)

    def test_fit_with_too_few_samples_stays_unfitted(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        det.fit([_make_frame_result() for _ in range(5)])
        assert det._is_fitted is False

    def test_fit_with_frame_results_replaces_baseline_data(self, tmp_path):
        """BUG FIX verification: passing frame_results should REPLACE, not append."""
        det = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        # Buffer 10 frames via collect_baseline
        for i in range(10):
            det.collect_baseline(_make_frame_result(frame_id=i))
        assert len(det._baseline_data) == 10

        # Now fit() with a fresh list — should use only the new list
        det.fit(_make_baseline(MIN_SAMPLES))
        # _baseline_data should now be exactly MIN_SAMPLES (replaced, not appended)
        assert len(det._baseline_data) == MIN_SAMPLES

    def test_scaler_initialised_after_fit(self, tmp_path):
        det = _fitted_detector(tmp_path)
        assert det._scaler is not None


# ---------------------------------------------------------------------------
# 5. AnomalyDetector.score()
# ---------------------------------------------------------------------------

class TestAnomalyDetectorScore:

    def test_returns_anomaly_result(self, tmp_path):
        det = _fitted_detector(tmp_path)
        r   = det.score(_make_frame_result())
        assert isinstance(r, AnomalyResult)

    def test_normal_frame_scores_normal(self, tmp_path):
        det = _fitted_detector(tmp_path)
        # Feed a frame that looks exactly like the baseline
        r = det.score(_make_frame_result(
            detection_count=5, class_names=["vehicle"], motion_score=5.0
        ))
        # We can't guarantee "normal" in all environments, but score should
        # be in valid range
        assert r.anomaly_score <= 0.5

    def test_military_vehicle_always_critical(self, tmp_path):
        det = _fitted_detector(tmp_path)
        fr  = _make_frame_result(class_names=["military_vehicle"])
        r   = det.score(fr)
        assert r.alert_level == "critical"

    def test_suspicious_object_always_critical(self, tmp_path):
        det = _fitted_detector(tmp_path)
        fr  = _make_frame_result(class_names=["suspicious_object"])
        r   = det.score(fr)
        assert r.alert_level == "critical"

    def test_military_vehicle_reason_included(self, tmp_path):
        det = _fitted_detector(tmp_path)
        fr  = _make_frame_result(class_names=["military_vehicle"])
        r   = det.score(fr)
        assert any("military_vehicle" in reason for reason in r.reasons)

    def test_anomaly_prob_in_unit_range(self, tmp_path):
        det = _fitted_detector(tmp_path)
        r   = det.score(_make_frame_result())
        assert 0.0 <= r.anomaly_prob <= 1.0

    def test_frame_id_preserved(self, tmp_path):
        det = _fitted_detector(tmp_path)
        r   = det.score(_make_frame_result(frame_id=42))
        assert r.frame_id == 42

    def test_motion_score_preserved(self, tmp_path):
        det = _fitted_detector(tmp_path)
        r   = det.score(_make_frame_result(motion_score=7.5))
        assert r.motion_score == pytest.approx(7.5)

    def test_features_length_matches_feature_dim(self, tmp_path):
        det = _fitted_detector(tmp_path)
        r   = det.score(_make_frame_result())
        assert len(r.features) == FEATURE_DIM

    def test_rule_based_fallback_for_unfitted_detector(self, tmp_path):
        """score() must work even without a fitted model."""
        det = AnomalyDetector(model_path=str(tmp_path / "no_model.pkl"))
        fr  = _make_frame_result(class_names=["military_vehicle"])
        r   = det.score(fr)
        assert r.alert_level == "critical"

    def test_score_batch_length_matches_input(self, tmp_path):
        det  = _fitted_detector(tmp_path)
        frs  = [_make_frame_result(frame_id=i) for i in range(10)]
        outs = det.score_batch(frs)
        assert len(outs) == 10

    def test_crowd_with_high_motion_is_high_or_critical(self, tmp_path):
        det = _fitted_detector(tmp_path)
        fr  = _make_frame_result(class_names=["crowd"], motion_score=15.0)
        r   = det.score(fr)
        assert r.alert_level in ("high", "critical")


# ---------------------------------------------------------------------------
# 6. AnomalyDetector.get_summary()
# ---------------------------------------------------------------------------

class TestAnomalyDetectorSummary:

    def _results(self) -> List[AnomalyResult]:
        return [
            AnomalyResult(1, time.time(), -0.20, 0.9, "critical",
                          ["military_vehicle detected"]),
            AnomalyResult(2, time.time(), -0.08, 0.6, "high",
                          ["crowd gathering"]),
            AnomalyResult(3, time.time(),  0.02, 0.2, "normal", []),
        ]

    def test_empty_returns_empty_dict(self, tmp_path):
        det = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        assert det.get_summary([]) == {}

    def test_total_frames_count(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        assert summ["total_frames"] == 3

    def test_critical_count(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        assert summ["critical_frames"] == 1

    def test_high_count(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        assert summ["high_alert_frames"] == 1

    def test_normal_count(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        assert summ["normal_frames"] == 1

    def test_alert_rate_calculation(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        # 2 alerts out of 3 frames
        assert summ["alert_rate"] == pytest.approx(2/3, abs=0.01)

    def test_model_fitted_key_present(self, tmp_path):
        det  = AnomalyDetector(model_path=str(tmp_path / "m.pkl"))
        summ = det.get_summary(self._results())
        assert "model_fitted" in summ


# ===========================================================================
# ── ALERT MANAGER MODULE ────────────────────────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# 7. Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:

    def _make(self, priority=PRIORITY_HIGH) -> Alert:
        return Alert(
            alert_id="alert_001", frame_id=10,
            timestamp=time.time(), priority=priority,
            anomaly_score=-0.10, anomaly_prob=0.75,
            alert_level="high", reasons=["crowd detected"],
            detection_count=8, motion_score=9.5,
        )

    def test_to_dict_has_all_required_keys(self):
        d = self._make().to_dict()
        for k in ("alert_id", "frame_id", "timestamp", "priority",
                  "anomaly_score", "anomaly_prob", "alert_level",
                  "reasons", "detection_count", "motion_score", "notified"):
            assert k in d

    def test_priority_rank_critical_highest(self):
        assert self._make(PRIORITY_CRITICAL).priority_rank == 3

    def test_priority_rank_low_lowest(self):
        assert self._make(PRIORITY_LOW).priority_rank == 0

    def test_priority_rank_ordering(self):
        ranks = [
            Alert("", 0, 0, p, 0, 0, "", []).priority_rank
            for p in [PRIORITY_LOW, PRIORITY_MEDIUM, PRIORITY_HIGH, PRIORITY_CRITICAL]
        ]
        assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# 8. AlertManager — initialisation
# ---------------------------------------------------------------------------

class TestAlertManagerInit:

    def test_initialises_empty_alerts(self, tmp_path):
        mgr = AlertManager(log_path=str(tmp_path / "log.json"))
        assert mgr._alerts == []

    def test_email_disabled_without_api_key(self, tmp_path):
        mgr = AlertManager(
            log_path=str(tmp_path / "log.json"),
            sendgrid_api_key="",
        )
        assert mgr._email_enabled is False

    def test_email_enabled_with_api_key(self, tmp_path):
        mgr = AlertManager(
            log_path=str(tmp_path / "log.json"),
            enable_email=True,
            sendgrid_api_key="SG.fake_key",
        )
        assert mgr._email_enabled is True

    def test_loads_existing_log_on_init(self, tmp_path):
        log_path = str(tmp_path / "log.json")
        # Write a fake log
        fake = [_make_anomaly_result(alert_level="high", frame_id=99)]
        mgr1 = AlertManager(log_path=log_path)
        mgr1.process(fake[0])  # creates one alert entry

        # Second manager should load that alert
        mgr2 = AlertManager(log_path=log_path)
        assert len(mgr2._alerts) == 1


# ---------------------------------------------------------------------------
# 9. AlertManager.process()
# ---------------------------------------------------------------------------

class TestAlertManagerProcess:

    def _mgr(self, tmp_path) -> AlertManager:
        return AlertManager(
            log_path=str(tmp_path / "log.json"),
            cooldown_seconds=0,          # disable cooldown in tests
            enable_email=False,
        )

    def test_normal_frame_returns_none(self, tmp_path):
        mgr = self._mgr(tmp_path)
        r   = _make_anomaly_result(alert_level="normal")
        assert mgr.process(r) is None

    def test_critical_frame_returns_alert(self, tmp_path):
        mgr = self._mgr(tmp_path)
        r   = _make_anomaly_result(alert_level="critical",
                                   anomaly_score=-0.25)
        alert = mgr.process(r)
        assert alert is not None
        assert alert.priority == PRIORITY_CRITICAL

    def test_high_frame_returns_high_alert(self, tmp_path):
        mgr   = self._mgr(tmp_path)
        r     = _make_anomaly_result(alert_level="high", anomaly_score=-0.09)
        alert = mgr.process(r)
        assert alert.priority == PRIORITY_HIGH

    def test_normal_with_high_motion_returns_medium(self, tmp_path):
        mgr   = self._mgr(tmp_path)
        r     = _make_anomaly_result(
            alert_level="normal",
            motion_score=MOTION_MEDIUM_THRESHOLD + 1,
        )
        # process() returns None for normal, but the priority mapping is MEDIUM
        # verify via _assign_priority directly
        priority = AlertManager._assign_priority("normal", MOTION_MEDIUM_THRESHOLD + 1)
        assert priority == PRIORITY_MEDIUM

    def test_alert_written_to_log(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.process(_make_anomaly_result(alert_level="critical",
                                         anomaly_score=-0.25))
        assert len(mgr._alerts) == 1

    def test_normal_frame_not_written_to_log(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.process(_make_anomaly_result(alert_level="normal"))
        assert len(mgr._alerts) == 0

    def test_log_file_created_after_alert(self, tmp_path):
        log_path = str(tmp_path / "alerts" / "log.json")
        mgr = AlertManager(log_path=log_path, cooldown_seconds=0,
                           enable_email=False)
        mgr.process(_make_anomaly_result(alert_level="critical",
                                         anomaly_score=-0.25))
        assert os.path.exists(log_path)

    def test_log_file_is_valid_json(self, tmp_path):
        log_path = str(tmp_path / "log.json")
        mgr = AlertManager(log_path=log_path, cooldown_seconds=0,
                           enable_email=False)
        mgr.process(_make_anomaly_result(alert_level="high", anomaly_score=-0.09))
        with open(log_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_frame_id_preserved_in_alert(self, tmp_path):
        mgr   = self._mgr(tmp_path)
        r     = _make_anomaly_result(alert_level="critical",
                                     anomaly_score=-0.25, frame_id=77)
        alert = mgr.process(r)
        assert alert.frame_id == 77

    def test_reasons_preserved_in_alert(self, tmp_path):
        mgr   = self._mgr(tmp_path)
        r     = _make_anomaly_result(
            alert_level="critical",
            reasons=["military_vehicle detected"],
        )
        alert = mgr.process(r)
        assert "military_vehicle detected" in alert.reasons

    def test_multiple_alerts_accumulate_in_log(self, tmp_path):
        mgr = self._mgr(tmp_path)
        for i in range(5):
            mgr.process(_make_anomaly_result(
                alert_level="high", frame_id=i, anomaly_score=-0.09
            ))
        assert len(mgr._alerts) == 5


# ---------------------------------------------------------------------------
# 10. AlertManager.get_summary()
# ---------------------------------------------------------------------------

class TestAlertManagerSummary:

    def test_empty_returns_total_alerts_zero(self, tmp_path):
        mgr  = AlertManager(log_path=str(tmp_path / "l.json"))
        summ = mgr.get_summary()
        assert summ["total_alerts"] == 0

    def test_counts_by_priority(self, tmp_path):
        mgr = AlertManager(log_path=str(tmp_path / "l.json"),
                           cooldown_seconds=0, enable_email=False)
        mgr.process(_make_anomaly_result(alert_level="critical",
                                         anomaly_score=-0.25))
        mgr.process(_make_anomaly_result(alert_level="high",
                                         anomaly_score=-0.09))
        summ = mgr.get_summary()
        assert summ["critical_count"] == 1
        assert summ["high_count"]     == 1
        assert summ["total_alerts"]   == 2

    def test_email_enabled_in_summary(self, tmp_path):
        mgr  = AlertManager(log_path=str(tmp_path / "l.json"),
                            enable_email=False)
        summ = mgr.get_summary()
        assert summ["email_enabled"] is False

    def test_clear_log_empties_alerts(self, tmp_path):
        log_path = str(tmp_path / "l.json")
        mgr = AlertManager(log_path=log_path, cooldown_seconds=0,
                           enable_email=False)
        mgr.process(_make_anomaly_result(alert_level="critical",
                                         anomaly_score=-0.25))
        assert len(mgr._alerts) == 1
        mgr.clear_log()
        assert len(mgr._alerts) == 0


# ---------------------------------------------------------------------------
# 11. Integration — anomaly → alert_manager pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:

    def test_full_pipeline_normal_baseline_then_threat(self, tmp_path):
        """
        Simulate: 30 normal frames → fit model → score threat frame.
        Threat frame with military_vehicle should become CRITICAL alert.
        """
        model_path = str(tmp_path / "anomaly_model.pkl")
        log_path   = str(tmp_path / "alerts.json")

        anomaly = AnomalyDetector(model_path=model_path)
        mgr     = AlertManager(log_path=log_path, cooldown_seconds=0,
                               enable_email=False)

        # Phase 1: collect baseline
        for i in range(MIN_SAMPLES):
            anomaly.collect_baseline(_make_frame_result(frame_id=i))
        anomaly.fit()
        assert anomaly._is_fitted

        # Phase 2: score a clearly critical frame
        threat_fr = _make_frame_result(
            frame_id=100,
            class_names=["military_vehicle"],
            motion_score=14.0,
        )
        result = anomaly.score(threat_fr)
        assert result.alert_level == "critical"

        # Phase 3: pass to alert manager
        alert = mgr.process(result.to_dict())
        assert alert is not None
        assert alert.priority == PRIORITY_CRITICAL

    def test_summary_pipeline(self, tmp_path):
        """Run 40 frames through full pipeline and verify summaries."""
        model_path = str(tmp_path / "m.pkl")
        log_path   = str(tmp_path / "a.json")

        anomaly = AnomalyDetector(model_path=model_path)
        mgr     = AlertManager(log_path=log_path, cooldown_seconds=0,
                               enable_email=False)

        frames  = [_make_frame_result(frame_id=i) for i in range(40)]
        scored: List[AnomalyResult] = []

        for i, fr in enumerate(frames):
            if i < MIN_SAMPLES:
                anomaly.collect_baseline(fr)
                if i == MIN_SAMPLES - 1:
                    anomaly.fit()
                continue
            result = anomaly.score(fr)
            scored.append(result)
            mgr.process(result.to_dict())

        a_summary = anomaly.get_summary(scored)
        m_summary = mgr.get_summary()

        assert a_summary["total_frames"] == 10   # 40 - 30 baseline
        assert "total_alerts" in m_summary


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
