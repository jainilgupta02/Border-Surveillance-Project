"""
Video Preprocessing Module
===========================

Core preprocessing pipeline for the Border Surveillance AI system.
Handles frame extraction, resizing, normalization, optical flow computation,
and frame persistence — feeding directly into the YOLOv8 detection layer.

Pipeline position:
    Video Input → [THIS MODULE] → YOLOv8 Detection → Anomaly Detection → Alerts

Architecture reference:
    Resize (640×640) | Normalize (0–1) | Optical Flow | Save / Azure Upload

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   March 2026
"""

import cv2
import os
import logging
from typing import Generator, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOLO_SIZE: Tuple[int, int] = (640, 640)          # Default input size for YOLOv8
SUPPORTED_FORMATS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


# ---------------------------------------------------------------------------
# 1. Video Loading
# ---------------------------------------------------------------------------

def load_video(video_path: Union[str, int]) -> cv2.VideoCapture:
    """
    Open a video file or camera stream and return the capture object.

    Args:
        video_path: Path to a video file (str) or a camera index (int).
                    Supported file formats: .mp4  .avi  .mov  .mkv
                    Use 0 for the default webcam, 1 for a second camera, etc.

    Returns:
        cv2.VideoCapture: An opened capture object ready for frame reading.

    Raises:
        ValueError: If the file does not exist, has an unsupported extension,
                    or cannot be opened by OpenCV.

    Example:
        >>> cap = load_video("border_feed.mp4")
        >>> ret, frame = cap.read()
        >>> cap.release()

        >>> cap = load_video(0)   # webcam
        >>> ret, frame = cap.read()
        >>> cap.release()
    """
    if isinstance(video_path, str):
        if not video_path.lower().endswith(SUPPORTED_FORMATS):
            raise ValueError(
                f"Unsupported video format. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    logger.info("Opened video source: %s", video_path)
    return cap


# ---------------------------------------------------------------------------
# 2. Frame-level Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    size: Tuple[int, int] = YOLO_SIZE,
    normalize: bool = False,
) -> np.ndarray:
    """
    Resize a raw video frame and optionally normalize pixel values to [0, 1].

    YOLOv8 accepts BGR uint8 input directly — pass ``normalize=False`` (the
    default) when feeding into the detector.  Pass ``normalize=True`` when
    feeding into the Isolation Forest / anomaly branch that expects float
    feature vectors.

    Args:
        frame:     Raw BGR frame from cv2 as a NumPy array of shape (H, W, 3).
        size:      Target (width, height).  Default (640, 640) matches YOLOv8n.
        normalize: When True, converts to float32 and scales pixels to [0, 1].

    Returns:
        np.ndarray: Resized frame, dtype uint8 or float32 depending on
                    ``normalize``.  Shape is always (height, width, 3).

    Raises:
        ValueError: If ``frame`` is None or ``size`` is invalid.

    Example:
        >>> frame = cv2.imread("sample.jpg")

        >>> # For YOLOv8 detection
        >>> yolo_input = preprocess_frame(frame)
        >>> print(yolo_input.shape, yolo_input.dtype)
        (640, 640, 3) uint8

        >>> # For anomaly feature extraction
        >>> normed = preprocess_frame(frame, normalize=True)
        >>> print(normed.min(), normed.max())
        0.0  1.0
    """
    if frame is None:
        raise ValueError("Input frame is None")

    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a tuple of (width, height)")

    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("size dimensions must be positive integers")

    resized = cv2.resize(frame, size)

    if normalize:
        resized = resized.astype(np.float32) / 255.0

    return resized


# ---------------------------------------------------------------------------
# 3. Optical Flow  (feeds Anomaly Detection layer)
# ---------------------------------------------------------------------------

def compute_optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute dense optical flow between two consecutive frames using
    Farneback's algorithm.

    Optical flow is the primary motion feature used by the Anomaly Detection
    layer (Isolation Forest + Random Forest ensemble) to flag unusual movement
    patterns such as rapid intrusions, crowd surges, or erratic trajectories.

    Args:
        prev_frame: Previous frame (BGR uint8 or float32, any size).
        curr_frame: Current frame (BGR uint8 or float32, same size as prev).

    Returns:
        Tuple of:
        - flow_magnitude (np.ndarray): Per-pixel motion magnitude map,
          shape (H, W), dtype float32.
        - mean_magnitude (float): Scalar mean motion intensity across the
          frame; a high value signals rapid global movement.

    Raises:
        ValueError: If either frame is None or the frames have mismatched
                    spatial dimensions.

    Example:
        >>> prev = cv2.imread("frame_001.jpg")
        >>> curr = cv2.imread("frame_002.jpg")
        >>> flow_map, motion_score = compute_optical_flow(prev, curr)
        >>> print(f"Motion intensity: {motion_score:.4f}")

    Note:
        Frames are internally converted to grayscale for the flow computation.
        The returned ``flow_magnitude`` can be thresholded (e.g. > 5.0) to
        create a binary motion mask for downstream spatial analysis.
    """
    if prev_frame is None or curr_frame is None:
        raise ValueError("Both prev_frame and curr_frame must be non-None arrays")

    def _to_gray(f: np.ndarray) -> np.ndarray:
        """Convert BGR or float32 frame to uint8 grayscale."""
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f

    prev_gray = _to_gray(prev_frame)
    curr_gray = _to_gray(curr_frame)

    if prev_gray.shape != curr_gray.shape:
        raise ValueError(
            f"Frame size mismatch: prev {prev_gray.shape} vs curr {curr_gray.shape}"
        )

    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Convert (dx, dy) vectors to magnitude
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = float(np.mean(magnitude))

    return magnitude, mean_magnitude


# ---------------------------------------------------------------------------
# 4. Frame Generator  (main pipeline entry point)
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Union[str, int],
    resize: Tuple[int, int] = YOLO_SIZE,
    frame_skip: int = 1,
    normalize: bool = False,
    compute_flow: bool = False,
    show_progress: bool = False,
) -> Generator[dict, None, None]:
    """
    Memory-efficient generator that yields preprocessed frames from a video.

    Each yielded item is a dictionary so that callers can destructure only what
    they need without breaking if new fields are added in future iterations.

    Args:
        video_path:    Path to a video file or camera index.
        resize:        Target (width, height) for each frame.
        frame_skip:    Yield every Nth frame (1 = all frames, 5 = every 5th).
                       Higher values reduce CPU/GPU load for long videos.
        normalize:     Pass True to get float32 frames in [0, 1].
                       Required by the anomaly feature branch.
        compute_flow:  When True, compute optical flow between successive
                       yielded frames and include it in the output dict.
        show_progress: Display a tqdm progress bar (requires tqdm package).

    Yields:
        dict with keys:
            - ``frame_id``        (int):        1-indexed position in the video.
            - ``frame``           (np.ndarray): Preprocessed frame.
            - ``flow_magnitude``  (np.ndarray | None): Per-pixel motion map
                                  (only present when compute_flow=True).
            - ``motion_score``    (float | None): Mean motion intensity
                                  (only present when compute_flow=True).

    Raises:
        ValueError: If parameters are invalid or the video cannot be opened.

    Example:
        >>> # Basic: feed frames into YOLOv8
        >>> for item in extract_frames("feed.mp4", frame_skip=2):
        ...     detections = yolo_model(item["frame"])

        >>> # With optical flow for anomaly detection
        >>> for item in extract_frames("feed.mp4", compute_flow=True):
        ...     motion = item["motion_score"]
        ...     if motion > 5.0:
        ...         anomaly_model.score(item["frame"])

    Note:
        The video capture is always released in the ``finally`` block, even if
        the caller breaks out of the loop early.
    """
    # --- Parameter validation -----------------------------------------------
    if not isinstance(resize, tuple) or len(resize) != 2:
        raise ValueError("resize must be a tuple of (width, height)")

    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("resize dimensions must be positive integers")

    if not isinstance(frame_skip, int) or frame_skip < 1:
        raise ValueError("frame_skip must be a positive integer >= 1")

    # --- Open video ----------------------------------------------------------
    cap = load_video(video_path)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video: %d frames @ %.2f FPS  |  Source resolution: %dx%d",
        total_frames, fps, width, height,
    )
    logger.info(
        "Config: resize=%s  frame_skip=%d  normalize=%s  optical_flow=%s",
        resize, frame_skip, normalize, compute_flow,
    )

    # --- Optional progress bar -----------------------------------------------
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Extracting frames", unit="fr")
        except ImportError:
            logger.warning("tqdm not installed — progress bar disabled.")

    # --- Extraction loop -----------------------------------------------------
    frame_count     = 0
    processed_count = 0
    prev_processed  = None   # kept for optical flow computation

    try:
        while True:
            ret, raw_frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if pbar:
                pbar.update(1)

            if frame_count % frame_skip != 0:
                continue

            # Preprocess
            processed = preprocess_frame(raw_frame, size=resize, normalize=normalize)
            processed_count += 1

            # Build output dict
            output: dict = {
                "frame_id":       frame_count,
                "frame":          processed,
                "flow_magnitude": None,
                "motion_score":   None,
            }

            # Optical flow (requires at least one previous processed frame)
            if compute_flow and prev_processed is not None:
                try:
                    flow_map, motion = compute_optical_flow(prev_processed, processed)
                    output["flow_magnitude"] = flow_map
                    output["motion_score"]   = motion
                except ValueError as exc:
                    logger.warning("Optical flow skipped for frame %d: %s", frame_count, exc)

            prev_processed = processed
            yield output

    finally:
        cap.release()

        if pbar:
            pbar.close()

        if frame_count > 0:
            logger.info(
                "Done: %d/%d frames yielded (%.1f%%).",
                processed_count, frame_count,
                processed_count / frame_count * 100,
            )


# ---------------------------------------------------------------------------
# 5. Frame Saving
# ---------------------------------------------------------------------------

def save_frame(
    frame: np.ndarray,
    output_dir: str,
    frame_id: int,
    prefix: str = "frame",
) -> str:
    """
    Persist a single frame to disk as a JPEG image.

    Useful for building labelled datasets, debug snapshots, and the alert
    thumbnail attachments sent via SendGrid.

    Args:
        frame:      Frame to save (uint8 or float32).
        output_dir: Destination directory (created automatically if absent).
        frame_id:   Non-negative integer used in the filename.
        prefix:     Filename prefix.  Final name: ``{prefix}_{frame_id:06d}.jpg``

    Returns:
        str: Absolute path to the saved file.

    Raises:
        ValueError: If ``frame`` is None or ``frame_id`` is invalid.
        IOError:    If OpenCV fails to write the file.

    Example:
        >>> frame = cv2.imread("raw.jpg")
        >>> path = save_frame(frame, "data/processed/frames", frame_id=42)
        >>> print(path)
        data/processed/frames/frame_000042.jpg

        >>> path = save_frame(frame, "alerts", 7, prefix="alert")
        >>> # → alerts/alert_000007.jpg
    """
    if frame is None:
        raise ValueError("Input frame is None")

    if not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")

    os.makedirs(output_dir, exist_ok=True)

    # Ensure uint8 for cv2.imwrite (convert if normalized float)
    save_frame_data = frame
    if frame.dtype != np.uint8:
        save_frame_data = (frame * 255).clip(0, 255).astype(np.uint8)

    filename = os.path.join(output_dir, f"{prefix}_{frame_id:06d}.jpg")

    success = cv2.imwrite(filename, save_frame_data)

    if not success:
        raise IOError(f"Failed to write frame to: {filename}")

    logger.debug("Saved frame %d → %s", frame_id, filename)
    return filename


# ---------------------------------------------------------------------------
# 6. Video Metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: Union[str, int]) -> dict:
    """
    Return metadata about a video without reading all frames.

    Args:
        video_path: Path to a video file or camera index.

    Returns:
        dict with keys:
            - ``fps``          (float): Frames per second.
            - ``total_frames`` (int):   Total frame count.
            - ``width``        (int):   Frame width in pixels.
            - ``height``       (int):   Frame height in pixels.
            - ``duration``     (float): Duration in seconds.

    Example:
        >>> info = get_video_info("sample.mp4")
        >>> print(f"Duration: {info['duration']:.1f}s  |  FPS: {info['fps']}")
    """
    cap = load_video(video_path)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    info = {
        "fps":          fps,
        "total_frames": total_frames,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration":     total_frames / fps if fps > 0 else 0.0,
    }

    cap.release()
    return info


# ---------------------------------------------------------------------------
# Quick sanity check (run directly: python preprocessing.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Border Surveillance AI — Preprocessing Module")
    print("=" * 55)
    print("\nFunctions available:")
    print("  load_video(path)                   → VideoCapture")
    print("  preprocess_frame(frame, ...)       → np.ndarray")
    print("  compute_optical_flow(prev, curr)   → (magnitude, score)")
    print("  extract_frames(path, ...)          → Generator[dict]")
    print("  save_frame(frame, dir, id, ...)    → str (saved path)")
    print("  get_video_info(path)               → dict")
    print("\nBasic usage:")
    print("""
    from preprocessing import extract_frames

    for item in extract_frames("border_feed.mp4", frame_skip=2, compute_flow=True):
        frame_id     = item["frame_id"]
        frame        = item["frame"]          # → YOLOv8
        motion_score = item["motion_score"]   # → Anomaly model
        print(f"Frame {frame_id}: motion={motion_score:.2f}")
    """)
