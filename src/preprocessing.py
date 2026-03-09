import cv2
import os


def load_video(video_path):
    """
    Opens the video file and returns the capture object.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    return cap


def preprocess_frame(frame, size=(640, 640)):
    """
    Resize frame and prepare it for detection.
    """

    frame = cv2.resize(frame, size)

    return frame


def extract_frames(video_path, resize=(640, 640), frame_skip=1):
    """
    Generator that yields preprocessed frames from a video.
    frame_skip can reduce CPU/GPU load.
    """

    cap = load_video(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = preprocess_frame(frame, resize)

        yield frame_count, frame

    cap.release()


# save_frame optional
def save_frame(frame, output_dir, frame_id):
    """
    Save frame as an image file.
    Useful for dataset creation and debugging.
    """

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"frame_{frame_id}.jpg")

    cv2.imwrite(filename, frame)