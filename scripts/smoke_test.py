"""
Smoke Test — Preprocessing + Detector + border_yolo.pt
========================================================

Run this script to verify your full stack is working before
moving on to anomaly.py.

Usage (from project root):
    python smoke_test.py                        # uses synthetic video
    python smoke_test.py path/to/your/video.mp4 # uses real video

What this checks:
    ✅ Step 1 — preprocessing.py can open a video and extract frames
    ✅ Step 2 — preprocessing.py optical flow works
    ✅ Step 3 — border_yolo.pt loads without errors
    ✅ Step 4 — detector.py runs inference on a real frame
    ✅ Step 5 — end-to-end: video → frames → detections → annotated output
    ✅ Step 6 — model is using your custom classes (not COCO 80 classes)

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import os
import sys
import time
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
SEP   = "─" * 55


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str):
    print(f"  {PASS}  {msg}")


def fail(msg: str):
    print(f"  {FAIL}  {msg}")


def warn(msg: str):
    print(f"  {WARN}  {msg}")


def make_synthetic_video(path: str, n_frames: int = 30) -> str:
    """
    Create a short synthetic test video so you can run the smoke test
    even without a real surveillance video.

    Draws moving shapes that give the detector something to work with.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))

    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Scrolling background gradient (triggers optical flow)
        frame[:, :] = (i * 3 % 50, i * 2 % 40, 20)

        # Moving rectangle — simulates an object crossing frame
        x = int((i / n_frames) * 580)
        cv2.rectangle(frame, (x, 180), (x + 60, 300), (200, 200, 200), -1)

        # Second moving object
        y = int((i / n_frames) * 380)
        cv2.rectangle(frame, (300, y), (360, y + 40), (100, 180, 100), -1)

        out.write(frame)

    out.release()
    return path


# ---------------------------------------------------------------------------
# Step 1 — preprocessing: frame extraction
# ---------------------------------------------------------------------------

def test_preprocessing_frames(video_path: str) -> bool:
    section("STEP 1 — preprocessing.py: frame extraction")
    try:
        from preprocessing import extract_frames, get_video_info

        info = get_video_info(video_path)
        ok(f"Video opened: {info['total_frames']} frames @ {info['fps']:.1f} FPS")
        ok(f"Resolution: {info['width']}×{info['height']}")

        frames = []
        for item in extract_frames(video_path, frame_skip=5, compute_flow=False):
            frames.append(item)
            if len(frames) >= 3:
                break

        if not frames:
            fail("extract_frames() yielded nothing")
            return False

        first = frames[0]
        ok(f"extract_frames() works — got {len(frames)} frames (sampled)")
        ok(f"Frame shape: {first['frame'].shape}  dtype: {first['frame'].dtype}")
        ok(f"frame_id starts at: {first['frame_id']}")

        assert first["frame"].shape == (640, 640, 3), "Wrong frame shape"
        assert first["frame"].dtype == np.uint8,      "Wrong dtype"

        ok("Frame shape and dtype correct (640×640, uint8)")
        return True

    except Exception as exc:
        fail(f"preprocessing.py failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 2 — preprocessing: optical flow
# ---------------------------------------------------------------------------

def test_optical_flow(video_path: str) -> bool:
    section("STEP 2 — preprocessing.py: optical flow")
    try:
        from preprocessing import extract_frames

        scores = []
        for item in extract_frames(video_path, frame_skip=2, compute_flow=True):
            if item["motion_score"] is not None:
                scores.append(item["motion_score"])
            if len(scores) >= 5:
                break

        if not scores:
            warn("No motion scores produced (video may be too short)")
            return True   # not a hard failure

        ok(f"Optical flow working — got {len(scores)} scores")
        ok(f"Motion scores: min={min(scores):.3f}  max={max(scores):.3f}  "
           f"mean={sum(scores)/len(scores):.3f}")

        assert all(s >= 0.0 for s in scores), "Motion scores must be non-negative"
        ok("All motion scores non-negative ✓")
        return True

    except Exception as exc:
        fail(f"Optical flow failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 3 — model loading
# ---------------------------------------------------------------------------

def test_model_loading() -> bool:
    section("STEP 3 — border_yolo.pt: model loading")
    try:
        from detector import BorderDetector, CLASS_NAMES

        model_path = "models/border_yolo.pt"

        if not os.path.exists(model_path):
            warn(f"Custom model not found at {model_path}")
            warn("Detector will fall back to generic yolov8n.pt")
            warn("Train your model and copy best.pt to models/border_yolo.pt")
        else:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            ok(f"Found {model_path}  ({size_mb:.1f} MB)")

        t0 = time.time()
        detector = BorderDetector(model_path=model_path)
        load_ms  = (time.time() - t0) * 1000

        ok(f"Model loaded in {load_ms:.0f} ms")
        ok(f"Device: {detector.device}")
        ok(f"Confidence threshold: {detector.confidence}")
        ok(f"IoU threshold: {detector.iou}")

        return detector

    except Exception as exc:
        fail(f"Model loading failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 4 — single frame inference
# ---------------------------------------------------------------------------

def test_single_inference(detector, video_path: str) -> bool:
    section("STEP 4 — detector.py: single frame inference")
    try:
        from preprocessing import extract_frames

        # Grab one real frame
        frame_item = next(iter(extract_frames(video_path, frame_skip=1,
                                              compute_flow=False)))

        t0     = time.time()
        result = detector.detect(frame_item)
        inf_ms = (time.time() - t0) * 1000

        ok(f"detect() ran in {inf_ms:.1f} ms")
        ok(f"Frame ID: {result.frame_id}")
        ok(f"Detections: {result.detection_count}")
        ok(f"has_critical: {result.has_critical}  |  has_high: {result.has_high}")

        if result.detections:
            for det in result.detections:
                ok(f"  → {det.class_name}  conf={det.confidence:.2f}  "
                   f"threat={det.threat_level}  "
                   f"bbox=[{', '.join(f'{v:.0f}' for v in det.bbox)}]")
        else:
            warn("No objects detected in this frame (expected for synthetic video)")
            warn("On real surveillance video you should see detections")

        # Verify result structure
        d = result.to_dict()
        required = {"frame_id", "timestamp", "detection_count",
                    "has_critical", "has_high", "inference_ms", "detections"}
        missing  = required - set(d.keys())
        if missing:
            fail(f"to_dict() missing keys: {missing}")
            return False

        ok("to_dict() structure correct")
        return True

    except Exception as exc:
        fail(f"Single inference failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 5 — end-to-end pipeline
# ---------------------------------------------------------------------------

def test_end_to_end(detector, video_path: str) -> bool:
    section("STEP 5 — end-to-end: video → frames → detections → saved images")
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            results = list(detector.process_video(
                video_path,
                frame_skip     = 5,
                compute_flow   = True,
                save_annotated = True,
                output_dir     = out_dir,
                show_progress  = False,
            ))

            saved = os.listdir(out_dir)

            ok(f"Processed {len(results)} frames")
            ok(f"Saved {len(saved)} annotated images to temp dir")

            total_det = sum(r.detection_count for r in results)
            ok(f"Total detections across all frames: {total_det}")

            avg_ms = sum(r.inference_ms for r in results) / max(len(results), 1)
            ok(f"Average inference time: {avg_ms:.1f} ms/frame")

            if len(saved) != len(results):
                fail(f"Mismatch: {len(results)} results but {len(saved)} saved files")
                return False

            ok("Annotated frame count matches processed frame count ✓")

        stats = detector.get_stats(results)
        ok(f"get_stats() summary:")
        for key, val in stats.items():
            print(f"       {key:<25} {val}")

        return True

    except Exception as exc:
        fail(f"End-to-end pipeline failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Step 6 — class verification
# ---------------------------------------------------------------------------

def test_class_verification(detector) -> bool:
    section("STEP 6 — model class verification")
    try:
        from detector import CLASS_NAMES

        expected = {
            0: "person", 1: "vehicle", 2: "crowd",
            3: "military_vehicle", 4: "aircraft",
            5: "ship", 6: "suspicious_object",
        }

        ok("Expected classes in detector.py:")
        for cid, name in expected.items():
            print(f"       {cid}: {name}")

        # Try to read model's own class names if available
        try:
            model_names = detector._model.names   # ultralytics exposes this
            ok(f"\n  Actual classes in border_yolo.pt ({len(model_names)} total):")
            for cid, name in model_names.items():
                match = "✓" if expected.get(cid) == name else "⚠ MISMATCH"
                print(f"       {cid}: {name}  {match}")

            mismatches = [
                cid for cid, name in expected.items()
                if model_names.get(cid) != name
            ]
            if mismatches:
                warn(f"Class mismatches at IDs: {mismatches}")
                warn("Update CLASS_NAMES in detector.py to match your data.yaml")
                return False
            else:
                ok("All classes match data.yaml ✓")
        except AttributeError:
            warn("Could not read class names from model — skipping mismatch check")

        return True

    except Exception as exc:
        fail(f"Class verification failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  Border Surveillance AI — Full Stack Smoke Test")
    print("=" * 55)

    # Determine video source
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"\n{FAIL}  Video not found: {video_path}")
            sys.exit(1)
        print(f"\n  Using real video: {video_path}")
        synthetic = False
    else:
        # Create a synthetic video in temp dir
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = tmp.name
        tmp.close()
        make_synthetic_video(video_path, n_frames=40)
        print(f"\n  No video provided — using synthetic test video")
        print(f"  Tip: python smoke_test.py path/to/real_video.mp4")
        synthetic = True

    results = {}

    # Run all steps
    results["preprocessing_frames"] = test_preprocessing_frames(video_path)
    results["optical_flow"]          = test_optical_flow(video_path)

    detector = test_model_loading()
    results["model_loading"]         = detector is not None

    if detector:
        results["single_inference"]  = test_single_inference(detector, video_path)
        results["end_to_end"]        = test_end_to_end(detector, video_path)
        results["class_verification"]= test_class_verification(detector)
    else:
        for key in ("single_inference", "end_to_end", "class_verification"):
            results[key] = False

    # Cleanup synthetic video
    if synthetic and os.path.exists(video_path):
        os.unlink(video_path)

    # Final summary
    section("SUMMARY")
    all_passed = True
    for step, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {step}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 All checks passed! Stack is healthy.")
        print("     Ready to build: anomaly.py")
    else:
        print("  ⚠️  Some checks failed. Fix the issues above before continuing.")
        print("     Common fixes:")
        print("       - 'model not found' → copy best.pt to models/border_yolo.pt")
        print("       - 'import error'    → pip install ultralytics opencv-python")
        print("       - class mismatch    → update CLASS_NAMES in detector.py")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
