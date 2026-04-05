"""
Dataset Preprocessing Pipeline — Border Surveillance AI
=========================================================

Converts xView, VisDrone, DOTA, and VEDAI raw datasets into a single
unified YOLO-format dataset at data/processed/.

Output structure:
    data/processed/
    ├── train/ images/ labels/
    ├── val/   images/ labels/
    ├── test/  images/ labels/
    └── data.yaml

Global class mapping (matches detector.py and data.yaml):
    0 → person
    1 → vehicle
    2 → crowd
    3 → military_vehicle
    4 → aircraft
    5 → ship
    6 → suspicious_object

Usage:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    source venv/bin/activate
    pip install tqdm opencv-python --break-system-packages
    python scripts/preprocess_all_datasets.py

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging — file + console
# ---------------------------------------------------------------------------

LOG_FILE = "data/preprocess.log"
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not installed — progress bars disabled.  "
                   "pip install tqdm")


def progress(iterable, desc="", total=None):
    """Wrap iterable with tqdm if available, else return as-is."""
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, total=total, unit="file", leave=False)
    return iterable


# ---------------------------------------------------------------------------
# Paths — edit BASE_DIR if your project lives elsewhere
# ---------------------------------------------------------------------------

BASE_DIR    = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
RAW_DIR     = BASE_DIR / "data" / "raw"
OUT_DIR     = BASE_DIR / "data" / "processed"

XVIEW_DIR   = RAW_DIR / "xview"
VISDRONE_DIR= RAW_DIR / "visdrone"
DOTA_DIR    = RAW_DIR / "dota"
VEDAI_DIR   = RAW_DIR / "vedai"

IMG_SIZE    = 640
SPLIT_RATIO = {"train": 0.80, "val": 0.10, "test": 0.10}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Global class system
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "person",            # 0
    "vehicle",           # 1
    "crowd",             # 2
    "military_vehicle",  # 3
    "aircraft",          # 4
    "ship",              # 5
    "suspicious_object", # 6
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# xView class map  (xView original ID → our ID)
# ---------------------------------------------------------------------------

XVIEW_CLASS_MAP: Dict[int, int] = {
    # person
    77: 0,
    # crowd
    79: 2,
    # vehicle (cars, trucks, buses, vans, utility vehicles …)
    18: 1, 19: 1, 20: 1, 21: 1, 23: 1, 24: 1,
    50: 1, 53: 1, 56: 1, 57: 1, 59: 1, 60: 1,
    61: 1, 64: 1, 65: 1, 66: 1, 72: 1, 73: 1,
    74: 1, 76: 1,
    # military vehicle
    17: 3, 26: 3, 28: 3, 62: 3, 63: 3, 71: 3,
    # aircraft
    11: 4, 12: 4, 15: 4, 47: 4, 49: 4,
    # ship
    32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5,
    40: 5, 41: 5, 42: 5, 44: 5,
    # suspicious object
    83: 6, 84: 6, 86: 6, 89: 6, 91: 6,
}

# ---------------------------------------------------------------------------
# VisDrone class map  (VisDrone class ID → our ID, -1 = ignore)
# ---------------------------------------------------------------------------

VISDRONE_CLASS_MAP: Dict[int, int] = {
    0:  -1,   # ignored region
    1:   0,   # pedestrian  → person
    2:   0,   # people      → person (small group)
    3:   1,   # bicycle     → vehicle
    4:   1,   # car         → vehicle
    5:   1,   # van         → vehicle
    6:   1,   # truck       → vehicle
    7:   1,   # tricycle    → vehicle
    8:   1,   # awning-tricycle → vehicle
    9:   1,   # bus         → vehicle
    10:  1,   # motor       → vehicle
    11: -1,   # others      → ignore
}

# ---------------------------------------------------------------------------
# DOTA class map  (lowercase DOTA name → our ID, -1 = ignore)
# ---------------------------------------------------------------------------

DOTA_CLASS_MAP: Dict[str, int] = {
    "plane":               4,   # aircraft
    "ship":                5,
    "storage-tank":        6,   # suspicious_object
    "baseball-diamond":   -1,
    "tennis-court":       -1,
    "basketball-court":   -1,
    "ground-track-field": -1,
    "harbor":              6,
    "bridge":             -1,
    "large-vehicle":       1,
    "small-vehicle":       1,
    "helicopter":          4,
    "roundabout":         -1,
    "soccer-ball-field":  -1,
    "swimming-pool":      -1,
    "container-crane":     3,   # military_vehicle
    "airport":             4,
    "helipad":             4,
}

# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    images_written: int = 0
    labels_written: int = 0
    annotations_ok: int = 0
    annotations_skipped: int = 0
    files_missing: int = 0
    files_corrupt: int = 0
    empty_labels: int = 0
    errors: List[str] = field(default_factory=list)

    def merge(self, other: "Stats") -> None:
        self.images_written    += other.images_written
        self.labels_written    += other.labels_written
        self.annotations_ok    += other.annotations_ok
        self.annotations_skipped += other.annotations_skipped
        self.files_missing     += other.files_missing
        self.files_corrupt     += other.files_corrupt
        self.empty_labels      += other.empty_labels
        self.errors.extend(other.errors)


GLOBAL_STATS = Stats()


# ===========================================================================
# Shared utilities
# ===========================================================================

def setup_output_dirs() -> None:
    """Create all output directories."""
    for split in ("train", "val", "test"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready: %s", OUT_DIR)


def split_files(files: List[Path]) -> Dict[str, List[Path]]:
    """Randomly split a list of files into train / val / test."""
    random.seed(RANDOM_SEED)
    random.shuffle(files)
    n     = len(files)
    t_end = int(n * SPLIT_RATIO["train"])
    v_end = t_end + int(n * SPLIT_RATIO["val"])
    return {
        "train": files[:t_end],
        "val":   files[t_end:v_end],
        "test":  files[v_end:],
    }


def safe_stem(dataset: str, original_stem: str) -> str:
    """
    Build a globally unique filename stem to avoid collisions across datasets.
    Format: {dataset}_{original_stem}
    """
    return f"{dataset}_{original_stem}"


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read and return image, or None on failure."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError("cv2.imread returned None")
        return img
    except Exception as exc:
        logger.debug("Cannot read image %s: %s", path, exc)
        return None


def resize_and_save(img: np.ndarray, dst: Path) -> bool:
    """Resize image to IMG_SIZE × IMG_SIZE and write to dst."""
    try:
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        return cv2.imwrite(str(dst), resized)
    except Exception as exc:
        logger.debug("Cannot write %s: %s", dst, exc)
        return False


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def validate_yolo_box(cx: float, cy: float,
                      w: float, h: float) -> bool:
    """Return True if all values are in (0, 1] and non-zero."""
    return (0.0 < cx <= 1.0 and 0.0 < cy <= 1.0
            and 0.0 < w  <= 1.0 and 0.0 < h  <= 1.0)


def write_label(dst: Path, lines: List[str]) -> None:
    """Write YOLO label lines to dst."""
    with open(dst, "w") as f:
        f.writelines(lines)


def write_data_yaml() -> None:
    """Write data.yaml for YOLOv8 training."""
    content = f"""# Border Surveillance — Unified YOLO Training Config
# Generated by preprocess_all_datasets.py

path: {OUT_DIR}
train: train/images
val:   val/images
test:  test/images

nc: {NUM_CLASSES}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"

    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    logger.info("data.yaml written → %s", yaml_path)


# ===========================================================================
# xView
# ===========================================================================

def process_xview() -> Stats:
    """
    xView is already in YOLO format.
    Re-map class IDs to our 7-class system and copy images.

    Expected structure:
        data/raw/xview/
        ├── images/   (.jpg or .tif)
        └── labels/   (.txt YOLO format with original xView class IDs)
    """
    stats  = Stats()
    src    = XVIEW_DIR
    logger.info("── xView ──────────────────────────────────────────")

    images_dir = src / "images"
    labels_dir = src / "labels"

    if not images_dir.exists():
        logger.warning("xView images dir not found: %s  — skipping", images_dir)
        return stats

    images = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.tif"))
    )
    logger.info("xView: found %d images", len(images))

    splits = split_files(images)

    for split_name, files in splits.items():
        for img_path in progress(files, desc=f"xView/{split_name}"):
            stem = safe_stem("xview", img_path.stem)

            # ── image ──────────────────────────────────────────────
            img = read_image(img_path)
            if img is None:
                stats.files_corrupt += 1
                continue

            dst_img = OUT_DIR / split_name / "images" / f"{stem}.jpg"
            if not resize_and_save(img, dst_img):
                stats.files_corrupt += 1
                continue
            stats.images_written += 1

            # ── label ──────────────────────────────────────────────
            label_src = labels_dir / (img_path.stem + ".txt")
            label_lines: List[str] = []

            if not label_src.exists():
                # Try same folder as image
                label_src = img_path.parent / (img_path.stem + ".txt")

            if label_src.exists():
                with open(label_src) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            stats.annotations_skipped += 1
                            continue
                        try:
                            orig_cls = int(float(parts[0]))
                            mapped   = XVIEW_CLASS_MAP.get(orig_cls, -1)
                            if mapped == -1:
                                stats.annotations_skipped += 1
                                continue
                            cx, cy, w, h = map(float, parts[1:5])
                            cx, cy, w, h = (clamp01(cx), clamp01(cy),
                                            clamp01(w),  clamp01(h))
                            if not validate_yolo_box(cx, cy, w, h):
                                stats.annotations_skipped += 1
                                continue
                            label_lines.append(
                                f"{mapped} {cx:.6f} {cy:.6f} "
                                f"{w:.6f} {h:.6f}\n"
                            )
                            stats.annotations_ok += 1
                        except ValueError:
                            stats.annotations_skipped += 1
            else:
                stats.files_missing += 1

            dst_lbl = OUT_DIR / split_name / "labels" / f"{stem}.txt"
            write_label(dst_lbl, label_lines)
            stats.labels_written += 1
            if not label_lines:
                stats.empty_labels += 1

    logger.info("xView done: %d images, %d labels, %d annotations",
                stats.images_written, stats.labels_written,
                stats.annotations_ok)
    return stats


# ===========================================================================
# VisDrone
# ===========================================================================

def _parse_visdrone_annotation(ann_path: Path,
                                img_w: int, img_h: int,
                                stats: Stats) -> List[str]:
    """
    Parse one VisDrone annotation file.

    VisDrone format per line:
        bbox_left, bbox_top, bbox_width, bbox_height,
        score, class_id, truncation, occlusion

    Returns list of YOLO label lines.
    """
    lines: List[str] = []
    if not ann_path.exists():
        return lines

    with open(ann_path) as f:
        for raw in f:
            parts = raw.strip().split(",")
            if len(parts) < 6:
                stats.annotations_skipped += 1
                continue
            try:
                x, y, bw, bh = (float(parts[0]), float(parts[1]),
                                 float(parts[2]), float(parts[3]))
                cls_id = int(parts[5])
            except ValueError:
                stats.annotations_skipped += 1
                continue

            mapped = VISDRONE_CLASS_MAP.get(cls_id, -1)
            if mapped == -1:
                stats.annotations_skipped += 1
                continue

            if bw <= 0 or bh <= 0:
                stats.annotations_skipped += 1
                continue

            cx = clamp01((x + bw / 2) / img_w)
            cy = clamp01((y + bh / 2) / img_h)
            w  = clamp01(bw / img_w)
            h  = clamp01(bh / img_h)

            if not validate_yolo_box(cx, cy, w, h):
                stats.annotations_skipped += 1
                continue

            lines.append(f"{mapped} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            stats.annotations_ok += 1

    return lines


def process_visdrone() -> Stats:
    """
    Process VisDrone dataset.

    Expected structure:
        data/raw/visdrone/
        ├── train/
        │   ├── images/
        │   └── annotations/
        └── val/
            ├── images/
            └── annotations/
    """
    stats = Stats()
    src   = VISDRONE_DIR
    logger.info("── VisDrone ────────────────────────────────────────")

    if not src.exists():
        logger.warning("VisDrone dir not found: %s — skipping", src)
        return stats

    # VisDrone provides train/val splits — we respect them + carve out test
    visdrone_splits = {
        "train": src / "train",
        "val":   src / "val",
    }

    for vd_split, vd_dir in visdrone_splits.items():
        img_dir = vd_dir / "images"
        ann_dir = vd_dir / "annotations"

        if not img_dir.exists():
            logger.warning("VisDrone %s images not found — skipping", vd_split)
            continue

        images = sorted(
            list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.png"))
        )
        logger.info("VisDrone %s: %d images", vd_split, len(images))

        # Keep val as val, train gets a small 10% carved into test
        if vd_split == "train":
            n      = len(images)
            t_end  = int(n * 0.90)
            batches = {
                "train": images[:t_end],
                "test":  images[t_end:],
            }
        else:
            batches = {"val": images}

        for out_split, files in batches.items():
            for img_path in progress(files, desc=f"VisDrone/{out_split}"):
                stem = safe_stem("visdrone", img_path.stem)

                img = read_image(img_path)
                if img is None:
                    stats.files_corrupt += 1
                    continue

                h_orig, w_orig = img.shape[:2]
                dst_img = OUT_DIR / out_split / "images" / f"{stem}.jpg"
                if not resize_and_save(img, dst_img):
                    stats.files_corrupt += 1
                    continue
                stats.images_written += 1

                ann_path  = ann_dir / (img_path.stem + ".txt")
                lbl_lines = _parse_visdrone_annotation(
                    ann_path, w_orig, h_orig, stats
                )

                dst_lbl = OUT_DIR / out_split / "labels" / f"{stem}.txt"
                write_label(dst_lbl, lbl_lines)
                stats.labels_written += 1
                if not lbl_lines:
                    stats.empty_labels += 1
                    if not ann_path.exists():
                        stats.files_missing += 1

    logger.info("VisDrone done: %d images, %d labels, %d annotations",
                stats.images_written, stats.labels_written,
                stats.annotations_ok)
    return stats


# ===========================================================================
# DOTA
# ===========================================================================

def _obb_to_hbb_yolo(parts: List[str],
                     img_w: int, img_h: int,
                     stats: Stats) -> Optional[str]:
    """
    Convert one DOTA OBB annotation line to YOLO HBB format.

    DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class_name [difficult]
    Returns YOLO string or None if the class should be ignored.
    """
    if len(parts) < 9:
        stats.annotations_skipped += 1
        return None

    try:
        coords     = list(map(float, parts[:8]))
        class_name = parts[8].lower().strip()
    except ValueError:
        stats.annotations_skipped += 1
        return None

    mapped = DOTA_CLASS_MAP.get(class_name, -1)
    if mapped == -1:
        stats.annotations_skipped += 1
        return None

    xs = coords[0::2]
    ys = coords[1::2]

    cx = clamp01(((min(xs) + max(xs)) / 2) / img_w)
    cy = clamp01(((min(ys) + max(ys)) / 2) / img_h)
    w  = clamp01((max(xs) - min(xs)) / img_w)
    h  = clamp01((max(ys) - min(ys)) / img_h)

    if not validate_yolo_box(cx, cy, w, h):
        stats.annotations_skipped += 1
        return None

    stats.annotations_ok += 1
    return f"{mapped} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def process_dota() -> Stats:
    """
    Process DOTA dataset.

    Handles two possible label formats:
      A) OBB format:  x1 y1 x2 y2 x3 y3 x4 y4 class_name
      B) YOLO format: class_id cx cy w h  (if already converted)

    Expected structure:
        data/raw/dota/
        ├── images/
        └── labels/
    """
    stats  = Stats()
    src    = DOTA_DIR
    logger.info("── DOTA ────────────────────────────────────────────")

    img_dir = src / "images"
    lbl_dir = src / "labels"

    if not img_dir.exists():
        logger.warning("DOTA images dir not found: %s — skipping", img_dir)
        return stats

    images = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpg"))
    )
    logger.info("DOTA: found %d images", len(images))

    splits = split_files(images)

    for split_name, files in splits.items():
        for img_path in progress(files, desc=f"DOTA/{split_name}"):
            stem = safe_stem("dota", img_path.stem)

            img = read_image(img_path)
            if img is None:
                stats.files_corrupt += 1
                continue

            h_orig, w_orig = img.shape[:2]
            dst_img = OUT_DIR / split_name / "images" / f"{stem}.jpg"
            if not resize_and_save(img, dst_img):
                stats.files_corrupt += 1
                continue
            stats.images_written += 1

            lbl_src = lbl_dir / (img_path.stem + ".txt")
            lbl_lines: List[str] = []

            if lbl_src.exists():
                with open(lbl_src) as f:
                    raw_lines = f.readlines()

                for line in raw_lines:
                    line = line.strip()
                    if not line or line.startswith("imagesource") \
                            or line.startswith("gsd"):
                        continue
                    parts = line.split()

                    # Detect format: OBB has 9+ tokens starting with coords
                    try:
                        float(parts[0])
                        is_coord_first = True
                    except ValueError:
                        is_coord_first = False

                    if is_coord_first and len(parts) >= 9:
                        # OBB format
                        result = _obb_to_hbb_yolo(
                            parts, w_orig, h_orig, stats
                        )
                        if result:
                            lbl_lines.append(result)
                    elif not is_coord_first and len(parts) >= 5:
                        # Already YOLO (class_name first, non-numeric)
                        class_name = parts[0].lower()
                        mapped     = DOTA_CLASS_MAP.get(class_name, -1)
                        if mapped == -1:
                            stats.annotations_skipped += 1
                            continue
                        try:
                            cx, cy, w, h = map(float, parts[1:5])
                            cx, cy, w, h = (clamp01(cx), clamp01(cy),
                                            clamp01(w),  clamp01(h))
                            if validate_yolo_box(cx, cy, w, h):
                                lbl_lines.append(
                                    f"{mapped} {cx:.6f} {cy:.6f} "
                                    f"{w:.6f} {h:.6f}\n"
                                )
                                stats.annotations_ok += 1
                            else:
                                stats.annotations_skipped += 1
                        except ValueError:
                            stats.annotations_skipped += 1
                    elif len(parts) >= 5:
                        # Plain YOLO (numeric class ID first)
                        try:
                            orig_cls    = int(float(parts[0]))
                            cx, cy, w, h = map(float, parts[1:5])
                            # DOTA YOLO labels keep original class IDs
                            # Try treating as already-mapped (0-6 range)
                            if 0 <= orig_cls < NUM_CLASSES:
                                mapped = orig_cls
                            else:
                                stats.annotations_skipped += 1
                                continue
                            cx, cy, w, h = (clamp01(cx), clamp01(cy),
                                            clamp01(w),  clamp01(h))
                            if validate_yolo_box(cx, cy, w, h):
                                lbl_lines.append(
                                    f"{mapped} {cx:.6f} {cy:.6f} "
                                    f"{w:.6f} {h:.6f}\n"
                                )
                                stats.annotations_ok += 1
                            else:
                                stats.annotations_skipped += 1
                        except ValueError:
                            stats.annotations_skipped += 1
            else:
                stats.files_missing += 1

            dst_lbl = OUT_DIR / split_name / "labels" / f"{stem}.txt"
            write_label(dst_lbl, lbl_lines)
            stats.labels_written += 1
            if not lbl_lines:
                stats.empty_labels += 1

    logger.info("DOTA done: %d images, %d labels, %d annotations",
                stats.images_written, stats.labels_written,
                stats.annotations_ok)
    return stats


# ===========================================================================
# VEDAI
# ===========================================================================

def _parse_vedai_annotation(ann_path: Path,
                             img_w: int, img_h: int,
                             stats: Stats) -> List[str]:
    """
    Parse a VEDAI annotation file.

    VEDAI Annotations512 format (one vehicle per line):
        vehicle_id  cx  cy  angle  w  h  is_occluded  is_cut  class_id  ...

    All coordinates are in pixels (absolute), not normalised.
    Classes: all map to vehicle (1) since VEDAI is vehicle-only.
    """
    lines: List[str] = []
    if not ann_path.exists():
        return lines

    with open(ann_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 6:
                stats.annotations_skipped += 1
                continue
            try:
                # cx, cy, angle, w, h  (pixels)
                cx_px = float(parts[1])
                cy_px = float(parts[2])
                w_px  = float(parts[4])
                h_px  = float(parts[5])
            except (ValueError, IndexError):
                stats.annotations_skipped += 1
                continue

            if w_px <= 0 or h_px <= 0:
                stats.annotations_skipped += 1
                continue

            cx = clamp01(cx_px / img_w)
            cy = clamp01(cy_px / img_h)
            w  = clamp01(w_px  / img_w)
            h  = clamp01(h_px  / img_h)

            if not validate_yolo_box(cx, cy, w, h):
                stats.annotations_skipped += 1
                continue

            # All VEDAI objects are vehicles (class 1)
            lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            stats.annotations_ok += 1

    return lines


def process_vedai() -> Stats:
    """
    Process VEDAI dataset.

    Expected structure:
        data/raw/vedai/
        ├── Vehicles512/      (images, .png)
        └── Annotations512/   (text files, one per image)

    Annotation filename format: {image_id}_ann.txt
    """
    stats  = Stats()
    src    = VEDAI_DIR
    logger.info("── VEDAI ────────────────────────────────────────────")

    img_dir = src / "Vehicles512"
    ann_dir = src / "Annotations512"

    if not img_dir.exists():
        logger.warning("VEDAI images dir not found: %s — skipping", img_dir)
        return stats

    images = sorted(img_dir.glob("*.png"))
    logger.info("VEDAI: found %d images", len(images))

    splits = split_files(images)

    for split_name, files in splits.items():
        for img_path in progress(files, desc=f"VEDAI/{split_name}"):
            stem = safe_stem("vedai", img_path.stem)

            img = read_image(img_path)
            if img is None:
                stats.files_corrupt += 1
                continue

            h_orig, w_orig = img.shape[:2]
            dst_img = OUT_DIR / split_name / "images" / f"{stem}.jpg"
            if not resize_and_save(img, dst_img):
                stats.files_corrupt += 1
                continue
            stats.images_written += 1

            # VEDAI annotation file: {stem}_ann.txt
            ann_path  = ann_dir / f"{img_path.stem}_ann.txt"
            lbl_lines = _parse_vedai_annotation(
                ann_path, w_orig, h_orig, stats
            )
            if not ann_path.exists():
                stats.files_missing += 1

            dst_lbl = OUT_DIR / split_name / "labels" / f"{stem}.txt"
            write_label(dst_lbl, lbl_lines)
            stats.labels_written += 1
            if not lbl_lines:
                stats.empty_labels += 1

    logger.info("VEDAI done: %d images, %d labels, %d annotations",
                stats.images_written, stats.labels_written,
                stats.annotations_ok)
    return stats


# ===========================================================================
# Post-processing verification
# ===========================================================================

def verify_dataset() -> None:
    """
    Walk the processed dataset and report:
      - images without a label file (and vice versa)
      - label files with invalid YOLO lines
    """
    logger.info("── Verification ─────────────────────────────────────")
    issues = 0

    for split in ("train", "val", "test"):
        img_dir = OUT_DIR / split / "images"
        lbl_dir = OUT_DIR / split / "labels"

        img_stems = {p.stem for p in img_dir.glob("*.jpg")}
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

        orphan_imgs  = img_stems - lbl_stems
        orphan_lbls  = lbl_stems - img_stems

        if orphan_imgs:
            logger.warning("%s: %d images without label", split, len(orphan_imgs))
            issues += len(orphan_imgs)
        if orphan_lbls:
            logger.warning("%s: %d labels without image", split, len(orphan_lbls))
            issues += len(orphan_lbls)

        # Validate label format
        bad_labels = 0
        for lbl_path in lbl_dir.glob("*.txt"):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls = int(parts[0])
                        vals = list(map(float, parts[1:5]))
                        if cls < 0 or cls >= NUM_CLASSES:
                            bad_labels += 1
                        if not all(0.0 <= v <= 1.0 for v in vals):
                            bad_labels += 1
                    except (ValueError, IndexError):
                        bad_labels += 1

        if bad_labels:
            logger.warning("%s: %d invalid label lines", split, bad_labels)
            issues += bad_labels

    if issues == 0:
        logger.info("Verification passed — no issues found ✓")
    else:
        logger.warning("Verification found %d issue(s) — see log above", issues)


# ===========================================================================
# Final summary
# ===========================================================================

def print_summary(stats: Stats) -> None:
    sep = "═" * 55
    logger.info(sep)
    logger.info("PREPROCESSING SUMMARY")
    logger.info(sep)
    logger.info("Images written      : %d", stats.images_written)
    logger.info("Label files written : %d", stats.labels_written)
    logger.info("Annotations kept    : %d", stats.annotations_ok)
    logger.info("Annotations skipped : %d", stats.annotations_skipped)
    logger.info("Empty label files   : %d", stats.empty_labels)
    logger.info("Missing label files : %d", stats.files_missing)
    logger.info("Corrupt images      : %d", stats.files_corrupt)
    if stats.errors:
        logger.warning("Errors              : %d (see log)", len(stats.errors))

    logger.info(sep)
    logger.info("Split breakdown:")
    for split in ("train", "val", "test"):
        n_img = len(list((OUT_DIR / split / "images").glob("*.jpg")))
        n_lbl = len(list((OUT_DIR / split / "labels").glob("*.txt")))
        logger.info("  %-6s  images=%d  labels=%d", split, n_img, n_lbl)
    logger.info(sep)
    logger.info("data.yaml → %s/data.yaml", OUT_DIR)
    logger.info("Full log  → %s", LOG_FILE)
    logger.info(sep)
    logger.info("Next step:")
    logger.info("  yolo train model=yolov8n.pt "
                'data="%s/data.yaml" epochs=50 imgsz=640', OUT_DIR)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    logger.info("Border Surveillance AI — Dataset Preprocessing Pipeline")
    logger.info("Processing: xView + VisDrone + DOTA + VEDAI")
    logger.info("Output:     %s", OUT_DIR)

    # Verify raw directories exist
    logger.info("Checking raw dataset locations...")
    for name, path in [
        ("xView",    XVIEW_DIR),
        ("VisDrone", VISDRONE_DIR),
        ("DOTA",     DOTA_DIR),
        ("VEDAI",    VEDAI_DIR),
    ]:
        status = "✓" if path.exists() else "✗ NOT FOUND"
        logger.info("  %-10s %s  %s", name, path, status)

    setup_output_dirs()

    # Process each dataset and accumulate stats
    for processor in [
        process_xview,
        process_visdrone,
        process_dota,
        process_vedai,
    ]:
        try:
            ds_stats = processor()
            GLOBAL_STATS.merge(ds_stats)
        except Exception as exc:
            logger.error("Dataset processor failed: %s", exc, exc_info=True)
            GLOBAL_STATS.errors.append(str(exc))

    write_data_yaml()
    verify_dataset()
    print_summary(GLOBAL_STATS)


if __name__ == "__main__":
    main()
