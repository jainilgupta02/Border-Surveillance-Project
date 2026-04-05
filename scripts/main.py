import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from detector import BorderDetector

detector = BorderDetector()

video_path = "data/test_videos/dota_aerial_test.mp4"

for result in detector.process_video(video_path):

    if result.has_critical:
        print("🚨 CRITICAL ALERT:", result.frame_id)

    elif result.has_high:
        print("⚠️ HIGH ALERT:", result.frame_id)