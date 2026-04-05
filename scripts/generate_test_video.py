"""
Synthetic Test Video Generator
================================

Creates a realistic test video with KNOWN anomalies at known frame numbers.
Use this to verify that the pipeline detects what it should detect.

The video has 3 acts:
    Act 1  (frames  1–60):  Normal traffic — vehicles moving slowly
    Act 2  (frames 61–90):  Anomaly — sudden crowd gathering + fast motion
    Act 3  (frames 91–120): Back to normal

When you run the pipeline on this video, you KNOW the anomaly detector
should flag frames 61–90.  If it does, the pipeline is working correctly.

Usage:
    python scripts/generate_test_video.py
    python src/pipeline.py --video data/test_videos/synthetic_test.mp4
    
Then check that critical/high alerts appear between frames ~60–90.

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import os
import math
import random
import cv2
import numpy as np

OUTPUT_PATH = "data/test_videos/synthetic_test.mp4"
FPS         = 10.0
WIDTH       = 640
HEIGHT      = 640

os.makedirs("data/test_videos", exist_ok=True)


def draw_vehicle(frame, x, y, w=60, h=30, color=(200, 200, 200)):
    """Draw a simple rectangle representing a vehicle."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 1)


def draw_person(frame, x, y, color=(100, 200, 100)):
    """Draw a simple stick figure representing a person."""
    # Body
    cv2.circle(frame, (x, y - 12), 5, color, -1)        # head
    cv2.line(frame,  (x, y - 7),  (x, y + 10), color, 2) # torso
    cv2.line(frame,  (x, y),      (x - 8, y + 15), color, 2)  # left leg
    cv2.line(frame,  (x, y),      (x + 8, y + 15), color, 2)  # right leg
    cv2.line(frame,  (x, y - 3),  (x - 10, y + 5), color, 2)  # left arm
    cv2.line(frame,  (x, y - 3),  (x + 10, y + 5), color, 2)  # right arm


def draw_road(frame):
    """Draw a simple road background."""
    frame[:] = (40, 60, 40)                               # dark green ground
    cv2.rectangle(frame, (0, 280), (640, 360), (80, 80, 80), -1)  # road
    # Lane markings
    for x in range(0, 640, 60):
        cv2.rectangle(frame, (x, 315), (x + 30, 325), (200, 200, 50), -1)


def add_noise(frame, intensity=5):
    """Add slight noise to make the video feel more realistic."""
    noise = np.random.randint(-intensity, intensity,
                              frame.shape, dtype=np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_info_overlay(frame, frame_num, act_name, n_people, n_vehicles):
    """Burn frame info into the video for easy verification."""
    cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(frame,
                f"Frame {frame_num:03d} | Act: {act_name} | "
                f"People: {n_people} | Vehicles: {n_vehicles}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Act generators
# ---------------------------------------------------------------------------

def act1_normal(writer, start_frame, n_frames):
    """
    Act 1: Normal traffic.
    2–3 vehicles moving slowly, 1–2 people walking.
    Low motion score, few detections — should be NORMAL.
    """
    print(f"  Writing Act 1 (Normal): frames {start_frame}–"
          f"{start_frame + n_frames - 1}")

    vehicle_x = [50, 200, 380]   # starting x positions
    person_x  = [300, 450]

    for i in range(n_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_road(frame)

        # Vehicles move slowly left to right
        for j, vx in enumerate(vehicle_x):
            vx_now = (vx + i * 3 + j * 80) % 700
            draw_vehicle(frame, vx_now, 290 + j * 10)

        # 1–2 people walk slowly
        for j, px in enumerate(person_x):
            px_now = (px + i * 1) % 640
            draw_person(frame, px_now, 350 + j * 20)

        frame = add_noise(frame)
        add_info_overlay(frame, start_frame + i, "NORMAL",
                         len(person_x), len(vehicle_x))
        writer.write(frame)


def act2_anomaly(writer, start_frame, n_frames):
    """
    Act 2: Anomaly — sudden crowd gathering with fast movement.
    8–12 people appearing quickly in centre, vehicles stopped, high motion.
    Should trigger HIGH or CRITICAL anomaly alerts.
    """
    print(f"  Writing Act 2 (ANOMALY): frames {start_frame}–"
          f"{start_frame + n_frames - 1}  ← expect alerts here")

    crowd_centre_x = 320
    crowd_centre_y = 380

    prev_frame = None

    for i in range(n_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_road(frame)

        # Stopped vehicles (anomaly: vehicles not moving)
        draw_vehicle(frame, 80,  295, color=(180, 180, 180))
        draw_vehicle(frame, 420, 295, color=(180, 180, 180))

        # Growing crowd — more people appear each frame
        n_crowd = min(6 + i, 15)
        random.seed(i * 42)
        for p in range(n_crowd):
            angle  = (p / max(n_crowd, 1)) * 2 * math.pi
            radius = 30 + (i % 5) * 5    # crowd pulsing = high motion
            px = int(crowd_centre_x + radius * math.cos(angle))
            py = int(crowd_centre_y + radius * math.sin(angle))
            px = max(10, min(630, px))
            py = max(50, min(620, py))
            draw_person(frame, px, py, color=(50, 50, 220))  # red-ish

        # Add strong noise (simulates camera shake / fast motion)
        frame = add_noise(frame, intensity=15)

        # Draw motion lines to further confuse the flow algorithm
        for _ in range(20):
            x1 = random.randint(200, 440)
            y1 = random.randint(300, 460)
            x2 = x1 + random.randint(-30, 30)
            y2 = y1 + random.randint(-30, 30)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 100, 100), 1)

        add_info_overlay(frame, start_frame + i, "ANOMALY ⚠",
                         n_crowd, 2)
        writer.write(frame)
        prev_frame = frame


def act3_normal(writer, start_frame, n_frames):
    """
    Act 3: Back to normal — crowd dispersed, traffic resumes.
    Should return to NORMAL anomaly scores.
    """
    print(f"  Writing Act 3 (Normal): frames {start_frame}–"
          f"{start_frame + n_frames - 1}")

    vehicle_x = [100, 320, 500]

    for i in range(n_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_road(frame)

        for j, vx in enumerate(vehicle_x):
            vx_now = (vx + i * 4 + j * 70) % 700
            draw_vehicle(frame, vx_now, 290 + j * 10)

        # Just 1 person remaining
        draw_person(frame, (200 + i * 2) % 640, 355)

        frame = add_noise(frame)
        add_info_overlay(frame, start_frame + i, "NORMAL",
                         1, len(vehicle_x))
        writer.write(frame)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic test video...")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Resolution: {WIDTH}×{HEIGHT}  FPS: {FPS}")
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        print(f"ERROR: Cannot open video writer for {OUTPUT_PATH}")
        return

    # Act 1: 60 frames of normal traffic (frames 1–60)
    act1_normal(writer, start_frame=1,  n_frames=60)

    # Act 2: 30 frames of anomaly (frames 61–90)  ← EXPECT ALERTS HERE
    act2_anomaly(writer, start_frame=61, n_frames=30)

    # Act 3: 30 frames of normal again (frames 91–120)
    act3_normal(writer, start_frame=91, n_frames=30)

    writer.release()

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print()
    print(f"✅ Video created: {OUTPUT_PATH}  ({size_mb:.1f} MB, 120 frames)")
    print()
    print("Now run the pipeline:")
    print(f"  python src/pipeline.py --video {OUTPUT_PATH} --frame-skip 1 --no-flow")
    print()
    print("Expected result:")
    print("  Frames  1–30  → collected as baseline (Phase A)")
    print("  Frames 31–60  → scored as NORMAL        (Phase B)")
    print("  Frames 61–90  → scored as HIGH/CRITICAL  ← alerts should fire here")
    print("  Frames 91–120 → scored as NORMAL again")
    print()
    print("Check: data/alerts/alert_log.json — should contain entries")
    print("       with frame_id between 61 and 90")


if __name__ == "__main__":
    main()
