"""
Full Pipeline Pilot
====================

Runs preprocessing → detector → anomaly → alert_manager on your real
video and prints a detailed report of every stage.

This is NOT a unit test.  It's a manual integration check you run
once to confirm everything is wired correctly before building pipeline.py.

Usage (from project root, venv active):
    python pilot.py data/test_videos/dota_aerial_test.mp4

What it checks and reports:
    Stage 1  — preprocessing:   frame extraction + optical flow
    Stage 2  — detector:        YOLO inference, detection counts
    Stage 3  — anomaly:         baseline collection, model fitting, scoring
    Stage 4  — alert_manager:   alert creation, JSON log writing
    Stage 5  — end-to-end:      full data flow from video to alert log
    Stage 6  — output files:    verifies all expected files were created

At the end it prints a single PASS / FAIL for each stage so you know
exactly what to fix if anything is broken.

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Suppress INFO logs during pilot — we print our own messages
logging.basicConfig(level=logging.WARNING)

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_MARK = f"{GREEN}✅ PASS{RESET}"
FAIL_MARK = f"{RED}❌ FAIL{RESET}"
WARN_MARK = f"{YELLOW}⚠  WARN{RESET}"
INFO_MARK = f"{CYAN}ℹ {RESET}"

SEP  = "─" * 60
SEP2 = "═" * 60


def header(title: str):
    print(f"\n{SEP2}")
    print(f"  {BOLD}{title}{RESET}")
    print(SEP2)


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str):   print(f"  {PASS_MARK}  {msg}")
def fail(msg: str): print(f"  {FAIL_MARK}  {msg}")
def warn(msg: str): print(f"  {WARN_MARK}  {msg}")
def info(msg: str): print(f"  {INFO_MARK}  {msg}")


# ── result accumulator ──────────────────────────────────────────────────────
stage_results: dict = {}


def record(stage: str, passed: bool, note: str = ""):
    stage_results[stage] = {"passed": passed, "note": note}


# ===========================================================================
# Stage 1 — Preprocessing
# ===========================================================================

def stage1_preprocessing(video_path: str) -> list:
    """
    Verify preprocessing.py can:
      - Open the video and report correct metadata
      - Extract frames at the expected shape and dtype
      - Compute optical flow without crashing
    Returns a list of frame_items for use in Stage 2.
    """
    section("STAGE 1 — preprocessing.py")
    frame_items = []

    try:
        from preprocessing import extract_frames, get_video_info

        # ── metadata ──────────────────────────────────────────────────
        info_data = get_video_info(video_path)
        ok(f"Video opened:  {info_data['total_frames']} frames  "
           f"@ {info_data['fps']:.1f} FPS")
        ok(f"Resolution:    {info_data['width']}×{info_data['height']}")
        ok(f"Duration:      {info_data['duration']:.1f} s")

        # ── frame extraction ──────────────────────────────────────────
        extracted = 0
        for item in extract_frames(
            video_path,
            frame_skip   = 5,
            compute_flow = False,
        ):
            frame_items.append(item)
            extracted += 1
            if extracted >= 5:
                break

        assert extracted > 0, "No frames extracted"
        first = frame_items[0]
        assert first["frame"].shape == (640, 640, 3), \
            f"Wrong shape: {first['frame'].shape}"
        import numpy as np
        assert first["frame"].dtype == np.uint8, \
            f"Wrong dtype: {first['frame'].dtype}"

        ok(f"Frame extraction:  {extracted} frames sampled")
        ok(f"Frame shape:       {first['frame'].shape}  "
           f"dtype={first['frame'].dtype}")
        ok(f"frame_id starts:   {first['frame_id']}")

        # ── optical flow ──────────────────────────────────────────────
        flow_scores = []
        for item in extract_frames(
            video_path,
            frame_skip   = 5,
            compute_flow = True,
        ):
            if item["motion_score"] is not None:
                flow_scores.append(item["motion_score"])
            if len(flow_scores) >= 5:
                break

        if flow_scores:
            ok(f"Optical flow:      {len(flow_scores)} scores  "
               f"mean={sum(flow_scores)/len(flow_scores):.2f}  "
               f"max={max(flow_scores):.2f}")
        else:
            warn("Optical flow returned no scores (video may be too short)")

        record("preprocessing", True)
        return frame_items

    except Exception as exc:
        fail(f"preprocessing failed: {exc}")
        record("preprocessing", False, str(exc))
        return []


# ===========================================================================
# Stage 2 — Detector
# ===========================================================================

def stage2_detector(video_path: str) -> list:
    """
    Verify detector.py can:
      - Load border_yolo.pt (or fallback) without error
      - Run inference and return FrameResult objects
      - Correctly populate has_critical / has_high flags
      - Annotate and save frames
    Returns a list of FrameResult dicts for Stage 3.
    """
    section("STAGE 2 — detector.py")
    frame_results = []

    try:
        from detector import BorderDetector
        from preprocessing import extract_frames

        t0       = time.time()
        detector = BorderDetector()
        load_ms  = (time.time() - t0) * 1000
        ok(f"Model loaded:      {load_ms:.0f} ms  device={detector.device}")

        # ── single frame inference ─────────────────────────────────────
        first_item = next(iter(extract_frames(
            video_path, frame_skip=1, compute_flow=False
        )))
        result     = detector.detect(first_item)
        ok(f"Single inference:  {result.inference_ms:.1f} ms  "
           f"detections={result.detection_count}")

        # ── process 40 frames (collect for anomaly stage) ─────────────
        count = 0
        for item in extract_frames(
            video_path,
            frame_skip   = 5,
            compute_flow = True,
        ):
            fr = detector.detect(item)
            frame_results.append(fr.to_dict())
            count += 1
            if count >= 40:
                break

        total_det  = sum(r["detection_count"] for r in frame_results)
        class_hits = {}
        for r in frame_results:
            for d in r["detections"]:
                cn = d["class_name"]
                class_hits[cn] = class_hits.get(cn, 0) + 1

        ok(f"Processed frames:  {count}")
        ok(f"Total detections:  {total_det}  "
           f"(avg {total_det/max(count,1):.1f}/frame)")
        ok(f"Classes seen:      {class_hits}")

        critical_fr = sum(1 for r in frame_results if r["has_critical"])
        high_fr     = sum(1 for r in frame_results if r["has_high"])
        ok(f"Critical frames:   {critical_fr}")
        ok(f"High frames:       {high_fr}")

        # ── annotate one frame ────────────────────────────────────────
        os.makedirs("data/pilot", exist_ok=True)
        annotated = detector.annotate_frame(
            first_item["frame"], result
        )
        import cv2
        cv2.imwrite("data/pilot/annotated_sample.jpg", annotated)
        ok("Annotated frame:   saved → data/pilot/annotated_sample.jpg")

        record("detector", True)
        return frame_results

    except Exception as exc:
        fail(f"detector failed: {exc}")
        record("detector", False, str(exc))
        return []


# ===========================================================================
# Stage 3 — Anomaly Detection
# ===========================================================================

def stage3_anomaly(frame_results: list) -> list:
    """
    Verify anomaly.py can:
      - Collect baseline and fit Isolation Forest on real detections
      - Score frames and return AnomalyResult objects
      - Apply class boosts (military_vehicle → always critical)
      - Save model to models/anomaly_model.pkl
    Returns a list of AnomalyResult dicts for Stage 4.
    """
    section("STAGE 3 — anomaly.py")
    anomaly_results = []

    if not frame_results:
        fail("No frame_results from Stage 2 — skipping")
        record("anomaly", False, "no input data")
        return []

    try:
        from anomaly import AnomalyDetector, MIN_SAMPLES

        detector = AnomalyDetector(model_path="models/anomaly_model_pilot.pkl")

        # ── Phase 1: collect baseline ──────────────────────────────────
        baseline_count = 0
        fitted         = False

        for fr in frame_results:
            if baseline_count < MIN_SAMPLES:
                ready = detector.collect_baseline(fr)
                baseline_count += 1
                if ready and not fitted:
                    info(f"Baseline collected ({MIN_SAMPLES} frames) — fitting...")
                    t0 = time.time()
                    detector.fit()
                    fit_ms = (time.time() - t0) * 1000
                    ok(f"Isolation Forest:  fitted in {fit_ms:.0f} ms")
                    ok(f"Model saved:       models/anomaly_model_pilot.pkl")
                    fitted = True
                continue

            # ── Phase 2: score ─────────────────────────────────────────
            result = detector.score(fr)
            anomaly_results.append(result)

        # ── If we didn't get MIN_SAMPLES, try rule-based fallback ──────
        if not fitted:
            warn(f"Only {baseline_count} frames — less than MIN_SAMPLES={MIN_SAMPLES}")
            warn("Scoring with rule-based fallback (no Isolation Forest)")
            for fr in frame_results:
                anomaly_results.append(detector.score(fr))

        if not anomaly_results:
            warn("No frames were scored — all used for baseline")
            record("anomaly", True, "only baseline collected")
            return []

        # ── report ─────────────────────────────────────────────────────
        normals   = sum(1 for r in anomaly_results if r.alert_level == "normal")
        highs     = sum(1 for r in anomaly_results if r.alert_level == "high")
        criticals = sum(1 for r in anomaly_results if r.alert_level == "critical")
        scores    = [r.anomaly_score for r in anomaly_results]

        ok(f"Frames scored:     {len(anomaly_results)}")
        ok(f"Normal:            {normals}")
        ok(f"High alerts:       {highs}")
        ok(f"Critical alerts:   {criticals}")
        ok(f"Score range:       min={min(scores):.3f}  "
           f"max={max(scores):.3f}  "
           f"mean={sum(scores)/len(scores):.3f}")
        ok(f"Model is fitted:   {detector._is_fitted}")

        # ── verify class boost ─────────────────────────────────────────
        # Inject a synthetic military_vehicle frame and confirm it's critical
        synthetic = {
            "frame_id": 9999, "timestamp": time.time(),
            "detection_count": 1, "motion_score": 5.0,
            "has_critical": True, "has_high": False,
            "inference_ms": 50.0,
            "detections": [{
                "class_id": 3, "class_name": "military_vehicle",
                "confidence": 0.85, "bbox": [100, 100, 300, 300],
                "center_x": 0.3, "center_y": 0.3,
                "width_norm": 0.31, "height_norm": 0.31,
                "area_norm": 0.096, "threat_level": "critical",
            }],
        }
        syn_result = detector.score(synthetic)
        if syn_result.alert_level == "critical":
            ok("Class boost check: military_vehicle → CRITICAL ✓")
        else:
            fail(f"Class boost FAILED — got {syn_result.alert_level}")
            record("anomaly", False, "class boost not working")
            return anomaly_results

        # Save summary JSON
        os.makedirs("data/pilot", exist_ok=True)
        summary = detector.get_summary(anomaly_results)
        with open("data/pilot/anomaly_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        ok("Summary saved:     data/pilot/anomaly_summary.json")

        record("anomaly", True)
        return [r.to_dict() for r in anomaly_results]

    except Exception as exc:
        fail(f"anomaly failed: {exc}")
        record("anomaly", False, str(exc))
        return []


# ===========================================================================
# Stage 4 — Alert Manager
# ===========================================================================

def stage4_alert_manager(anomaly_results: list) -> bool:
    """
    Verify alert_manager.py can:
      - Process AnomalyResult dicts and assign correct priorities
      - Write a valid JSON alert log
      - Return summary statistics
    """
    section("STAGE 4 — alert_manager.py")

    if not anomaly_results:
        warn("No anomaly results — injecting synthetic data for alert test")
        anomaly_results = [
            {
                "frame_id": 1, "timestamp": time.time(),
                "anomaly_score": -0.25, "anomaly_prob": 0.92,
                "alert_level": "critical",
                "reasons": ["military_vehicle detected"],
                "detection_count": 3, "motion_score": 9.0,
            },
            {
                "frame_id": 2, "timestamp": time.time(),
                "anomaly_score": -0.08, "anomaly_prob": 0.60,
                "alert_level": "high",
                "reasons": ["crowd gathering detected"],
                "detection_count": 8, "motion_score": 11.5,
            },
            {
                "frame_id": 3, "timestamp": time.time(),
                "anomaly_score": 0.03, "anomaly_prob": 0.18,
                "alert_level": "normal",
                "reasons": [],
                "detection_count": 4, "motion_score": 4.0,
            },
        ]

    try:
        from alert_manager import AlertManager, PRIORITY_CRITICAL, PRIORITY_HIGH

        log_path = "data/pilot/alert_log.json"
        mgr      = AlertManager(
            log_path         = log_path,
            cooldown_seconds = 0,         # disable cooldown for pilot
            enable_email     = False,
        )
        mgr.clear_log()   # start fresh

        alert_count    = 0
        critical_count = 0
        high_count     = 0

        for r in anomaly_results:
            alert = mgr.process(r)
            if alert:
                alert_count += 1
                if alert.priority == PRIORITY_CRITICAL:
                    critical_count += 1
                    info(f"CRITICAL alert — frame {alert.frame_id}: "
                         f"{alert.reasons}")
                elif alert.priority == PRIORITY_HIGH:
                    high_count += 1
                    info(f"HIGH alert    — frame {alert.frame_id}: "
                         f"{alert.reasons}")

        ok(f"Alerts generated:  {alert_count}")
        ok(f"Critical:          {critical_count}")
        ok(f"High:              {high_count}")

        # ── verify JSON log is valid ───────────────────────────────────
        with open(log_path) as f:
            log_data = json.load(f)

        assert isinstance(log_data, list), "Log is not a list"
        ok(f"JSON log valid:    {len(log_data)} entries "
           f"→ {log_path}")

        if log_data:
            entry = log_data[0]
            required_keys = {
                "alert_id", "frame_id", "priority", "anomaly_score",
                "reasons", "notified",
            }
            missing = required_keys - set(entry.keys())
            if missing:
                fail(f"Log entry missing keys: {missing}")
                record("alert_manager", False, f"missing keys {missing}")
                return False
            ok(f"Log entry structure correct (all required keys present)")

        # ── summary ───────────────────────────────────────────────────
        summary = mgr.get_summary()
        ok(f"Summary total:     {summary['total_alerts']} alerts")
        ok(f"By priority:       {summary.get('by_priority', {})}")

        record("alert_manager", True)
        return True

    except Exception as exc:
        fail(f"alert_manager failed: {exc}")
        record("alert_manager", False, str(exc))
        return False


# ===========================================================================
# Stage 5 — Output File Verification
# ===========================================================================

def stage5_outputs():
    """
    Confirm all expected output files were created during the pilot.
    """
    section("STAGE 5 — Output File Verification")

    expected_files = {
        "data/pilot/annotated_sample.jpg": "Annotated video frame",
        "data/pilot/anomaly_summary.json": "Anomaly summary JSON",
        "data/pilot/alert_log.json":       "Alert log JSON",
        "models/anomaly_model_pilot.pkl":  "Fitted anomaly model",
    }

    all_present = True
    for path, description in expected_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            ok(f"{description:<30} → {path}  ({size:,} bytes)")
        else:
            fail(f"{description:<30} → MISSING: {path}")
            all_present = False

    record("output_files", all_present)


# ===========================================================================
# Final Summary
# ===========================================================================

def print_final_summary():
    header("PILOT SUMMARY")

    all_passed = True
    for stage, result in stage_results.items():
        passed = result["passed"]
        note   = result.get("note", "")
        mark   = PASS_MARK if passed else FAIL_MARK
        line   = f"  {mark}  {stage}"
        if note:
            line += f"  ({note})"
        print(line)
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"  {GREEN}{BOLD}🎉 All stages passed — stack is healthy!{RESET}")
        print()
        print("  Your pipeline data flow is verified:")
        print("  Video → preprocessing → detector → anomaly → alert_manager")
        print()
        print(f"  {BOLD}Next step:{RESET} build pipeline.py")
        print("  This will tie all 4 modules into one command:")
        print("  python pipeline.py --video feed.mp4")
    else:
        failed = [s for s, r in stage_results.items() if not r["passed"]]
        print(f"  {RED}{BOLD}⚠  {len(failed)} stage(s) failed: "
              f"{', '.join(failed)}{RESET}")
        print()
        print("  Common fixes:")
        print("  - preprocessing fail → pip install opencv-python")
        print("  - detector fail      → check models/border_yolo.pt exists")
        print("  - anomaly fail       → pip install scikit-learn")
        print("  - alert_manager fail → check data/ directory is writable")

    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    header("Border Surveillance AI — Full Pipeline Pilot")

    if len(sys.argv) < 2:
        print(f"\n  {RED}Usage: python pilot.py path/to/video.mp4{RESET}")
        print(f"\n  Example:")
        print("    python pilot.py data/test_videos/dota_aerial_test.mp4")
        sys.exit(1)

    video = sys.argv[1]
    if not os.path.exists(video):
        print(f"\n  {RED}Video not found: {video}{RESET}")
        sys.exit(1)

    print(f"\n  Video: {video}")
    print(f"  All output files → data/pilot/\n")

    # Run each stage — later stages use output from earlier ones
    frame_items     = stage1_preprocessing(video)
    frame_results   = stage2_detector(video)
    anomaly_results = stage3_anomaly(frame_results)
    stage4_alert_manager(anomaly_results)
    stage5_outputs()

    print_final_summary()
