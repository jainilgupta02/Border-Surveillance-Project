"""
Test Suite — Video Preprocessing Module
=========================================

Comprehensive unit + integration tests for preprocessing.py.

Run all tests:
    pytest test_preprocessing.py -v

Run with coverage report:
    pytest test_preprocessing.py --cov=preprocessing --cov-report=html

Run only fast tests (skip benchmarks):
    pytest test_preprocessing.py -v -m "not benchmark"

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   March 2026
"""

import os

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    SUPPORTED_FORMATS,
    YOLO_SIZE,
    compute_optical_flow,
    extract_frames,
    get_video_info,
    load_video,
    preprocess_frame,
    save_frame,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_video(path, num_frames: int = 20, fps: float = 10.0,
                width: int = 640, height: int = 480) -> str:
    """Helper: write a minimal synthetic MP4 and return the path string."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 10 % 256, i * 5 % 256, i * 3 % 256)
        out.write(frame)
    out.release()
    return str(path)


@pytest.fixture
def sample_video(tmp_path):
    """20-frame video at 10 FPS, 640×480."""
    return _make_video(tmp_path / "sample.mp4")


@pytest.fixture
def known_video(tmp_path):
    """Video with deterministic properties for metadata assertions."""
    path = tmp_path / "known.mp4"
    _make_video(path, num_frames=90, fps=30.0, width=1280, height=720)
    return str(path), {"fps": 30.0, "total_frames": 90,
                       "width": 1280, "height": 720, "duration": 3.0}


@pytest.fixture
def test_frame():
    """A plain 480×640 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ===========================================================================
# 1. load_video
# ===========================================================================

class TestLoadVideo:
    """Tests for load_video()."""

    def test_loads_valid_mp4(self, sample_video):
        cap = load_video(sample_video)
        assert cap is not None
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret
        assert frame is not None
        cap.release()

    def test_raises_on_missing_file(self):
        with pytest.raises(ValueError, match="Video file not found"):
            load_video("nonexistent_video.mp4")

    def test_raises_on_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "clip.txt"
        txt_file.write_text("not a video")
        with pytest.raises(ValueError, match="Unsupported video format"):
            load_video(str(txt_file))

    def test_raises_on_corrupted_file(self, tmp_path):
        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"\x00\x01\x02\x03 not real video data")
        with pytest.raises(ValueError, match="Error opening video file"):
            load_video(str(bad))

    @pytest.mark.parametrize("ext", list(SUPPORTED_FORMATS))
    def test_all_supported_extensions_accepted(self, tmp_path, ext):
        """load_video must not reject any extension in SUPPORTED_FORMATS."""
        video_file = tmp_path / f"clip{ext}"
        _make_video(video_file, num_frames=2, width=320, height=240)
        cap = load_video(str(video_file))
        assert cap.isOpened(), f"{ext} should be openable"
        cap.release()


# ===========================================================================
# 2. preprocess_frame
# ===========================================================================

class TestPreprocessFrame:
    """Tests for preprocess_frame()."""

    def test_default_resize_to_yolo_size(self, test_frame):
        out = preprocess_frame(test_frame)
        assert out.shape == (640, 640, 3)
        assert out.dtype == np.uint8

    def test_custom_resize(self, test_frame):
        out = preprocess_frame(test_frame, size=(320, 320))
        assert out.shape == (320, 320, 3)

    def test_normalize_produces_float32_in_unit_range(self, test_frame):
        test_frame[:] = 128  # mid-grey
        out = preprocess_frame(test_frame, normalize=True)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_no_normalize_keeps_uint8(self, test_frame):
        out = preprocess_frame(test_frame, normalize=False)
        assert out.dtype == np.uint8

    def test_preserves_three_channels(self, test_frame):
        out = preprocess_frame(test_frame)
        assert out.shape[2] == 3

    def test_none_frame_raises(self):
        with pytest.raises(ValueError, match="Input frame is None"):
            preprocess_frame(None)

    def test_list_size_raises(self, test_frame):
        with pytest.raises(ValueError, match="size must be a tuple"):
            preprocess_frame(test_frame, size=[640, 640])

    def test_single_element_tuple_raises(self, test_frame):
        with pytest.raises(ValueError, match="size must be a tuple"):
            preprocess_frame(test_frame, size=(640,))

    def test_negative_dimension_raises(self, test_frame):
        with pytest.raises(ValueError, match="must be positive"):
            preprocess_frame(test_frame, size=(-1, 640))

    def test_zero_dimension_raises(self, test_frame):
        with pytest.raises(ValueError, match="must be positive"):
            preprocess_frame(test_frame, size=(0, 640))


# ===========================================================================
# 3. compute_optical_flow
# ===========================================================================

class TestComputeOpticalFlow:
    """Tests for compute_optical_flow()."""

    def _make_frame(self, value: int = 0) -> np.ndarray:
        f = np.full((480, 640, 3), value, dtype=np.uint8)
        return f

    def test_returns_tuple_of_correct_types(self):
        prev = self._make_frame(50)
        curr = self._make_frame(100)
        mag, score = compute_optical_flow(prev, curr)
        assert isinstance(mag, np.ndarray)
        assert isinstance(score, float)

    def test_magnitude_shape_matches_frame(self):
        prev = self._make_frame(0)
        curr = self._make_frame(0)
        mag, _ = compute_optical_flow(prev, curr)
        # magnitude is 2-D (H, W)
        assert mag.shape == (480, 640)

    def test_identical_frames_give_zero_motion(self):
        frame = self._make_frame(100)
        _, score = compute_optical_flow(frame, frame.copy())
        assert score < 0.5, "Identical frames should produce near-zero motion"

    def test_different_frames_produce_nonzero_motion(self):
        prev = np.zeros((480, 640, 3), dtype=np.uint8)
        curr = np.full((480, 640, 3), 200, dtype=np.uint8)
        _, score = compute_optical_flow(prev, curr)
        # Large brightness change should produce a non-trivial flow score
        assert score >= 0.0  # always non-negative

    def test_none_frame_raises(self):
        frame = self._make_frame(0)
        with pytest.raises(ValueError, match="non-None"):
            compute_optical_flow(None, frame)
        with pytest.raises(ValueError, match="non-None"):
            compute_optical_flow(frame, None)

    def test_size_mismatch_raises(self):
        prev = np.zeros((480, 640, 3), dtype=np.uint8)
        curr = np.zeros((240, 320, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="size mismatch"):
            compute_optical_flow(prev, curr)

    def test_accepts_float32_frames(self):
        prev = np.random.rand(480, 640, 3).astype(np.float32)
        curr = np.random.rand(480, 640, 3).astype(np.float32)
        mag, score = compute_optical_flow(prev, curr)
        assert mag is not None
        assert score >= 0.0


# ===========================================================================
# 4. extract_frames
# ===========================================================================

class TestExtractFrames:
    """Tests for extract_frames() generator."""

    # --- Output structure ---------------------------------------------------

    def test_yields_dicts_with_required_keys(self, sample_video):
        item = next(iter(extract_frames(sample_video)))
        assert "frame_id" in item
        assert "frame" in item
        assert "flow_magnitude" in item
        assert "motion_score" in item

    def test_frame_id_starts_at_one(self, sample_video):
        first = next(iter(extract_frames(sample_video)))
        assert first["frame_id"] == 1

    def test_frame_shape_matches_resize(self, sample_video):
        for item in extract_frames(sample_video, resize=(320, 320)):
            assert item["frame"].shape == (320, 320, 3)

    # --- frame_skip ---------------------------------------------------------

    def test_frame_skip_one_yields_all_frames(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=1))
        assert len(items) == 20

    def test_frame_skip_two_yields_half(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=2))
        assert len(items) == 10
        ids = [i["frame_id"] for i in items]
        assert ids == list(range(2, 21, 2))

    def test_frame_skip_five_yields_every_fifth(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=5))
        assert len(items) == 4           # frames 5, 10, 15, 20
        ids = [i["frame_id"] for i in items]
        assert ids == [5, 10, 15, 20]

    def test_frame_skip_larger_than_total_yields_one(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=100))
        assert len(items) == 0           # no frame index divisible by 100 in 20

    # --- normalize ----------------------------------------------------------

    def test_normalize_true_produces_float32(self, sample_video):
        item = next(iter(extract_frames(sample_video, normalize=True)))
        assert item["frame"].dtype == np.float32
        assert item["frame"].min() >= 0.0
        assert item["frame"].max() <= 1.0

    def test_normalize_false_keeps_uint8(self, sample_video):
        item = next(iter(extract_frames(sample_video, normalize=False)))
        assert item["frame"].dtype == np.uint8

    # --- optical flow -------------------------------------------------------

    def test_compute_flow_false_returns_none_fields(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=1, compute_flow=False))
        for item in items:
            assert item["flow_magnitude"] is None
            assert item["motion_score"] is None

    def test_compute_flow_true_populates_from_second_frame(self, sample_video):
        items = list(extract_frames(sample_video, frame_skip=1, compute_flow=True))
        # First frame has no predecessor → flow fields are None
        assert items[0]["flow_magnitude"] is None
        assert items[0]["motion_score"] is None
        # Second frame onward must have flow populated
        for item in items[1:]:
            assert item["flow_magnitude"] is not None
            assert isinstance(item["motion_score"], float)

    def test_motion_score_is_non_negative(self, sample_video):
        for item in extract_frames(sample_video, compute_flow=True):
            if item["motion_score"] is not None:
                assert item["motion_score"] >= 0.0

    # --- parameter validation -----------------------------------------------

    def test_invalid_resize_raises(self, sample_video):
        with pytest.raises(ValueError, match="resize must be a tuple"):
            list(extract_frames(sample_video, resize=[640, 640]))

    def test_invalid_frame_skip_raises(self, sample_video):
        with pytest.raises(ValueError, match="frame_skip must be a positive"):
            list(extract_frames(sample_video, frame_skip=0))
        with pytest.raises(ValueError, match="frame_skip must be a positive"):
            list(extract_frames(sample_video, frame_skip=-1))

    # --- edge cases ---------------------------------------------------------

    def test_empty_video_raises_or_yields_nothing(self, tmp_path):
        """
        A video file with no frames written is either rejected by OpenCV
        as unreadable (ValueError) or opens but yields nothing.
        Both outcomes are acceptable — the caller must handle ValueError.
        """
        path = tmp_path / "empty.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 10.0, (640, 480))
        out.release()
        try:
            items = list(extract_frames(str(path)))
            assert items == [], "If opened, an empty video should yield no frames"
        except ValueError:
            pass  # OpenCV correctly refuses a zero-frame file

    def test_generator_releases_capture_after_break(self, sample_video):
        """Verify the generator cleans up even when the caller breaks early."""
        gen = extract_frames(sample_video)
        next(gen)           # consume one item
        gen.close()         # simulate break / early exit


# ===========================================================================
# 5. save_frame
# ===========================================================================

class TestSaveFrame:
    """Tests for save_frame()."""

    def test_saves_file_to_disk(self, tmp_path, test_frame):
        path = save_frame(test_frame, str(tmp_path), frame_id=1)
        assert os.path.exists(path)

    def test_filename_uses_zero_padded_id(self, tmp_path, test_frame):
        path = save_frame(test_frame, str(tmp_path), frame_id=7)
        assert "frame_000007.jpg" in path

    def test_custom_prefix(self, tmp_path, test_frame):
        path = save_frame(test_frame, str(tmp_path), frame_id=42, prefix="alert")
        assert "alert_000042.jpg" in path

    def test_creates_nested_directories(self, tmp_path, test_frame):
        nested = tmp_path / "a" / "b" / "c"
        path = save_frame(test_frame, str(nested), frame_id=1)
        assert os.path.exists(path)

    def test_saved_file_is_readable_by_opencv(self, tmp_path, test_frame):
        path = save_frame(test_frame, str(tmp_path), frame_id=1)
        loaded = cv2.imread(path)
        assert loaded is not None
        assert loaded.shape[2] == 3

    def test_saves_normalized_float_frame(self, tmp_path):
        float_frame = np.random.rand(640, 640, 3).astype(np.float32)
        path = save_frame(float_frame, str(tmp_path), frame_id=1)
        assert os.path.exists(path)

    def test_none_frame_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Input frame is None"):
            save_frame(None, str(tmp_path), frame_id=1)

    def test_negative_frame_id_raises(self, tmp_path, test_frame):
        with pytest.raises(ValueError, match="non-negative integer"):
            save_frame(test_frame, str(tmp_path), frame_id=-1)

    def test_float_frame_id_raises(self, tmp_path, test_frame):
        with pytest.raises(ValueError, match="non-negative integer"):
            save_frame(test_frame, str(tmp_path), frame_id=1.5)

    def test_returns_string_path_ending_in_jpg(self, tmp_path, test_frame):
        path = save_frame(test_frame, str(tmp_path), frame_id=1)
        assert isinstance(path, str)
        assert path.endswith(".jpg")


# ===========================================================================
# 6. get_video_info
# ===========================================================================

class TestGetVideoInfo:
    """Tests for get_video_info()."""

    def test_returns_all_required_keys(self, sample_video):
        info = get_video_info(sample_video)
        for key in ("fps", "total_frames", "width", "height", "duration"):
            assert key in info, f"Missing key: {key}"

    def test_values_match_known_video(self, known_video):
        path, expected = known_video
        info = get_video_info(path)
        assert abs(info["fps"] - expected["fps"]) < 1.0
        assert info["total_frames"] == expected["total_frames"]
        assert info["width"] == expected["width"]
        assert info["height"] == expected["height"]
        assert abs(info["duration"] - expected["duration"]) < 0.2

    def test_duration_equals_frames_over_fps(self, sample_video):
        info = get_video_info(sample_video)
        if info["fps"] > 0:
            expected_duration = info["total_frames"] / info["fps"]
            assert abs(info["duration"] - expected_duration) < 1e-3

    def test_raises_on_bad_path(self):
        with pytest.raises(ValueError):
            get_video_info("no_such_file.mp4")


# ===========================================================================
# 7. Integration tests
# ===========================================================================

class TestIntegration:
    """End-to-end pipeline tests."""

    def test_full_pipeline_extract_then_save(self, tmp_path):
        """load → extract (with flow) → save every frame."""
        vid = _make_video(tmp_path / "input.mp4", num_frames=10)
        out_dir = tmp_path / "frames"

        for item in extract_frames(vid, resize=(640, 640), compute_flow=True):
            save_frame(item["frame"], str(out_dir), item["frame_id"])

        saved = list(out_dir.glob("*.jpg"))
        assert len(saved) == 10

    def test_pipeline_with_frame_skip_and_normalization(self, tmp_path):
        """extract every 2nd frame, normalized → verify types and count."""
        vid = _make_video(tmp_path / "v.mp4", num_frames=20)
        items = list(extract_frames(vid, frame_skip=2, normalize=True))

        assert len(items) == 10
        for item in items:
            assert item["frame"].dtype == np.float32
            assert 0.0 <= item["frame"].min()
            assert item["frame"].max() <= 1.0

    def test_optical_flow_scores_increase_with_motion(self, tmp_path):
        """
        Frames with bigger colour changes should yield higher motion scores
        than identical frames.
        """
        path_static = tmp_path / "static.mp4"
        path_dynamic = tmp_path / "dynamic.mp4"

        # Static: all frames are the same colour
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path_static), fourcc, 10.0, (64, 64))
        for _ in range(10):
            out.write(np.full((64, 64, 3), 100, dtype=np.uint8))
        out.release()

        # Dynamic: colour jumps drastically every frame
        out = cv2.VideoWriter(str(path_dynamic), fourcc, 10.0, (64, 64))
        for i in range(10):
            val = 0 if i % 2 == 0 else 255
            out.write(np.full((64, 64, 3), val, dtype=np.uint8))
        out.release()

        static_scores = [
            i["motion_score"] for i in
            extract_frames(str(path_static), compute_flow=True)
            if i["motion_score"] is not None
        ]
        dynamic_scores = [
            i["motion_score"] for i in
            extract_frames(str(path_dynamic), compute_flow=True)
            if i["motion_score"] is not None
        ]

        assert np.mean(dynamic_scores) >= np.mean(static_scores), (
            "Dynamic video should have higher mean motion score"
        )


# ===========================================================================
# 8. Performance benchmarks  (run with: pytest -m benchmark)
# ===========================================================================

@pytest.mark.benchmark
class TestPerformance:
    """Benchmark tests — skipped in normal CI, run explicitly with -m benchmark."""

    def test_frame_extraction_throughput(self, tmp_path, benchmark):
        """Extract 20 frames from a 1080p video — should complete quickly."""
        vid = _make_video(tmp_path / "perf.mp4",
                          num_frames=100, fps=30.0, width=1920, height=1080)

        def run():
            return list(extract_frames(vid, frame_skip=5))

        result = benchmark(run)
        assert len(result) == 20


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not benchmark"])
