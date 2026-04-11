#!/usr/bin/env python3
"""
Border Surveillance — Smart Balanced Dataset Preprocessor v2
==============================================================

Processes xView, VisDrone, and DOTA raw datasets into a balanced
YOLO-format dataset optimised for CPU training with limited resources.

Key improvements over v1:
  1. Smart class-aware vehicle undersampling (~15% of vehicle-only images)
  2. Rare class image oversampling via horizontal flip augmentation
  3. Proper DOTA OBB → HBB conversion with class mapping
  4. Removes VEDAI (deleted per user decision)
  5. Stratified train/val/test split ensuring rare classes in all splits
  6. Comprehensive quality filtering & statistics report

Classes (7):
  0: person
  1: vehicle
  2: crowd
  3: military_vehicle
  4: aircraft
  5: ship
  6: suspicious_object

Usage:
  python scripts/preprocess_balanced_v2.py
"""

import os
import sys
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time
import json

# ============================================================
# CONFIGURATION
# ============================================================

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
STAGING_DIR = BASE_DIR / "data" / "_staging"

# Image settings
IMG_SIZE = 640
JPEG_QUALITY = 90  # Balance between quality and file size

# Balancing parameters — these control the class distribution
VEHICLE_ONLY_KEEP_RATIO = 0.12     # Keep 12% of vehicle-only images
CROWD_VEHICLE_KEEP_RATIO = 0.25    # Keep 25% of crowd+vehicle only images
RARE_CLASSES = {3, 4, 5, 6}        # military_vehicle, aircraft, ship, suspicious_object
PERSON_CLASS = 0
VEHICLE_CLASS = 1
CROWD_CLASS = 2

# Augmentation
AUGMENT_RARE_CLASSES = True         # Flip-augment images with rare classes
RARE_AUGMENT_THRESHOLD = 500       # Only augment if class has < 500 images

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Class system
NUM_CLASSES = 7
CLASS_NAMES = [
    "person",
    "vehicle",
    "crowd",
    "military_vehicle",
    "aircraft",
    "ship",
    "suspicious_object",
]

# ============================================================
# xView class mapping (xView original ID → our 7-class ID)
# xView labels are ALREADY remapped in the raw data!
# class_id in xView labels: 0=person, 1=vehicle, 2=crowd,
# 3=military_vehicle, 4=aircraft, 5=ship, 6=suspicious_object
# So we just pass through (already YOLO format, already remapped)
# ============================================================

# VisDrone class mapping
# VisDrone annotations: bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, truncation, occlusion
# VisDrone categories:
#   0: ignored regions    → skip
#   1: pedestrian         → 0 (person)
#   2: people             → 0 (person)
#   3: bicycle            → skip (not in our classes)
#   4: car                → 1 (vehicle)
#   5: van                → 1 (vehicle)
#   6: truck              → 1 (vehicle)
#   7: tricycle           → 1 (vehicle)
#   8: awning-tricycle    → 1 (vehicle)
#   9: bus                → 1 (vehicle)
#   10: motor             → 1 (vehicle)
#   11: others            → skip

VISDRONE_CLASS_MAP = {
    1: 0,   # pedestrian → person
    2: 0,   # people → person
    4: 1,   # car → vehicle
    5: 1,   # van → vehicle
    6: 1,   # truck → vehicle
    7: 1,   # tricycle → vehicle
    8: 1,   # awning-tricycle → vehicle
    9: 1,   # bus → vehicle
    10: 1,  # motor → vehicle
}

# DOTA class mapping
# DOTA class name → our class ID
DOTA_CLASS_MAP = {
    "small-vehicle":        1,  # vehicle
    "large-vehicle":        1,  # vehicle
    "plane":                4,  # aircraft
    "helicopter":           4,  # aircraft
    "ship":                 5,  # ship
    "harbor":               5,  # ship (harbor infrastructure)
    "storage-tank":         6,  # suspicious_object
    "ground-track-field":   None,  # skip
    "baseball-diamond":     None,  # skip
    "tennis-court":         None,  # skip
    "basketball-court":     None,  # skip
    "soccer-ball-field":    None,  # skip
    "roundabout":           None,  # skip
    "swimming-pool":        None,  # skip
    "bridge":               None,  # skip
    "container-crane":      None,  # skip
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(msg, level="INFO"):
    """Simple logger with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_image(img, target_size=IMG_SIZE):
    """Resize image to target_size x target_size using letterboxing."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create letterboxed image (pad with gray)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    return canvas, scale, dx, dy, w, h


def adjust_labels_for_letterbox(labels, orig_w, orig_h, scale, dx, dy, target_size=IMG_SIZE):
    """
    Adjust YOLO normalized labels after letterbox resize.
    labels: list of (class_id, cx, cy, w, h) in original normalized coords
    Returns: list of (class_id, cx, cy, w, h) in letterboxed normalized coords
    """
    adjusted = []
    for cls_id, cx, cy, bw, bh in labels:
        # Convert to absolute in original image
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_w = bw * orig_w
        abs_h = bh * orig_h

        # Apply scale + offset
        new_cx = abs_cx * scale + dx
        new_cy = abs_cy * scale + dy
        new_w = abs_w * scale
        new_h = abs_h * scale

        # Normalize to target_size
        ncx = new_cx / target_size
        ncy = new_cy / target_size
        nw = new_w / target_size
        nh = new_h / target_size

        # Clamp
        ncx = max(0.001, min(0.999, ncx))
        ncy = max(0.001, min(0.999, ncy))
        nw = min(nw, 2 * min(ncx, 1 - ncx))
        nh = min(nh, 2 * min(ncy, 1 - ncy))

        # Filter out tiny boxes (< 3px equivalent)
        if nw * target_size >= 3 and nh * target_size >= 3:
            adjusted.append((int(cls_id), ncx, ncy, nw, nh))

    return adjusted


def write_yolo_label(filepath, labels):
    """Write labels in YOLO format to a file."""
    with open(filepath, "w") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def read_yolo_label(filepath):
    """Read YOLO format label file. Returns list of (class_id, cx, cy, w, h)."""
    labels = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Basic validation
                    if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        if cls_id >= 0 and cls_id < NUM_CLASSES:
                            labels.append((cls_id, cx, cy, w, h))
    except Exception as e:
        pass
    return labels


def horizontal_flip_labels(labels):
    """Flip labels horizontally for augmentation."""
    flipped = []
    for cls_id, cx, cy, w, h in labels:
        flipped.append((cls_id, 1.0 - cx, cy, w, h))
    return flipped


# ============================================================
# DATASET PROCESSORS
# ============================================================

def process_xview():
    """
    Process xView dataset.
    xView images are pre-cropped JPGs with YOLO-format labels already remapped.
    We just resize to 640x640 and copy labels.
    """
    log("Processing xView dataset...")
    xview_dir = RAW_DIR / "xview"

    if not xview_dir.exists():
        log("xView directory not found, skipping.", "WARN")
        return 0

    staging_img_dir = STAGING_DIR / "images"
    staging_lbl_dir = STAGING_DIR / "labels"
    ensure_dir(staging_img_dir)
    ensure_dir(staging_lbl_dir)

    # Find all images recursively
    img_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = []
    for ext in img_extensions:
        images.extend((xview_dir / "images").rglob(f"*{ext}"))

    count = 0
    skipped = 0

    for img_path in sorted(images):
        stem = img_path.stem

        # Find matching label
        label_path = None
        for lbl_dir in [xview_dir / "labels" / "train", xview_dir / "labels" / "val", xview_dir / "labels"]:
            candidate = lbl_dir / f"{stem}.txt"
            if candidate.exists():
                label_path = candidate
                break

        if label_path is None:
            skipped += 1
            continue

        # Read labels
        labels = read_yolo_label(label_path)
        if not labels:
            skipped += 1
            continue

        # Read and resize image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        resized, scale, dx, dy, orig_w, orig_h = resize_image(img)
        adjusted_labels = adjust_labels_for_letterbox(labels, orig_w, orig_h, scale, dx, dy)

        if not adjusted_labels:
            skipped += 1
            continue

        # Save
        out_name = f"xview_{stem}"
        cv2.imwrite(str(staging_img_dir / f"{out_name}.jpg"), resized,
                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        write_yolo_label(staging_lbl_dir / f"{out_name}.txt", adjusted_labels)
        count += 1

        if count % 500 == 0:
            log(f"  xView: {count} images processed...")

    log(f"  xView complete: {count} images processed, {skipped} skipped")
    return count


def process_visdrone():
    """
    Process VisDrone dataset.
    VisDrone annotations are CSV: bbox_left, bbox_top, bbox_width, bbox_height, score, category_id, truncation, occlusion
    """
    log("Processing VisDrone dataset...")
    visdrone_dir = RAW_DIR / "visdrone"

    if not visdrone_dir.exists():
        log("VisDrone directory not found, skipping.", "WARN")
        return 0

    staging_img_dir = STAGING_DIR / "images"
    staging_lbl_dir = STAGING_DIR / "labels"
    ensure_dir(staging_img_dir)
    ensure_dir(staging_lbl_dir)

    count = 0
    skipped = 0

    # Process both train and val splits
    for split in ["train", "val"]:
        split_dir = visdrone_dir / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"

        if not img_dir.exists() or not ann_dir.exists():
            continue

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

        for img_path in img_files:
            stem = img_path.stem
            ann_path = ann_dir / f"{stem}.txt"

            if not ann_path.exists():
                skipped += 1
                continue

            # Read and parse VisDrone annotation
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                h, w = img.shape[:2]
            except Exception:
                skipped += 1
                continue

            # Parse annotations into YOLO format
            labels = []
            try:
                with open(ann_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) < 6:
                            continue

                        bbox_left = int(parts[0])
                        bbox_top = int(parts[1])
                        bbox_width = int(parts[2])
                        bbox_height = int(parts[3])
                        # parts[4] = score
                        category_id = int(parts[5])

                        # Map VisDrone class to our class
                        if category_id not in VISDRONE_CLASS_MAP:
                            continue

                        our_class = VISDRONE_CLASS_MAP[category_id]

                        # Convert to YOLO normalized format
                        cx = (bbox_left + bbox_width / 2) / w
                        cy = (bbox_top + bbox_height / 2) / h
                        bw = bbox_width / w
                        bh = bbox_height / h

                        # Clamp
                        cx = max(0.001, min(0.999, cx))
                        cy = max(0.001, min(0.999, cy))
                        bw = max(0.001, min(0.999, bw))
                        bh = max(0.001, min(0.999, bh))

                        # Filter tiny boxes (< 3px in 640 equivalent)
                        if bw * w >= 3 and bh * h >= 3:
                            labels.append((our_class, cx, cy, bw, bh))
            except Exception:
                skipped += 1
                continue

            if not labels:
                skipped += 1
                continue

            # Resize with letterbox
            resized, scale, dx, dy, orig_w, orig_h = resize_image(img)
            adjusted_labels = adjust_labels_for_letterbox(labels, orig_w, orig_h, scale, dx, dy)

            if not adjusted_labels:
                skipped += 1
                continue

            # Save
            out_name = f"visdrone_{split}_{stem}"
            cv2.imwrite(str(staging_img_dir / f"{out_name}.jpg"), resized,
                         [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            write_yolo_label(staging_lbl_dir / f"{out_name}.txt", adjusted_labels)
            count += 1

            if count % 1000 == 0:
                log(f"  VisDrone: {count} images processed...")

    log(f"  VisDrone complete: {count} images processed, {skipped} skipped")
    return count


def process_dota():
    """
    Process DOTA dataset.
    DOTA labels are in OBB format: x1 y1 x2 y2 x3 y3 x4 y4 class_name [difficult]
    First 2 lines are headers (imagesource, gsd).
    We convert OBB to horizontal bounding box (HBB).
    """
    log("Processing DOTA dataset...")
    dota_dir = RAW_DIR / "dota"

    if not dota_dir.exists():
        log("DOTA directory not found, skipping.", "WARN")
        return 0

    staging_img_dir = STAGING_DIR / "images"
    staging_lbl_dir = STAGING_DIR / "labels"
    ensure_dir(staging_img_dir)
    ensure_dir(staging_lbl_dir)

    img_dir = dota_dir / "images"
    lbl_dir = dota_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        log("DOTA images or labels directory not found, skipping.", "WARN")
        return 0

    # Find all label files
    label_files = sorted(lbl_dir.glob("*.txt"))

    count = 0
    skipped = 0
    large_image_tiles = 0

    for lbl_path in label_files:
        stem = lbl_path.stem

        # Find matching image
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        # Parse DOTA OBB labels
        labels = []
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip header lines
                if line.startswith("imagesource") or line.startswith("gsd"):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                try:
                    # Parse 4 corner points
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[2]), float(parts[3])
                    x3, y3 = float(parts[4]), float(parts[5])
                    x4, y4 = float(parts[6]), float(parts[7])
                    class_name = parts[8]
                except (ValueError, IndexError):
                    continue

                # Map DOTA class to our class
                if class_name not in DOTA_CLASS_MAP:
                    continue
                our_class = DOTA_CLASS_MAP[class_name]
                if our_class is None:
                    continue

                # Convert OBB to HBB (axis-aligned bounding box)
                xs = [x1, x2, x3, x4]
                ys = [y1, y2, y3, y4]
                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                # Store as absolute pixel coords for now
                labels.append((our_class, xmin, ymin, xmax, ymax))

        except Exception:
            skipped += 1
            continue

        if not labels:
            skipped += 1
            continue

        # Read image
        try:
            # Check file size first to avoid loading huge files
            file_size_mb = img_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 150:  # Skip images > 150MB
                log(f"  Skipping oversized image: {img_path.name} ({file_size_mb:.0f}MB)", "WARN")
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
        except Exception:
            skipped += 1
            continue

        # For large DOTA images, do tiling
        if w > IMG_SIZE * 1.5 or h > IMG_SIZE * 1.5:
            tile_count = _tile_dota_image(img, labels, stem, w, h,
                                          staging_img_dir, staging_lbl_dir)
            count += tile_count
            large_image_tiles += tile_count
        else:
            # Small images: just resize with letterbox
            yolo_labels = []
            for cls_id, xmin, ymin, xmax, ymax in labels:
                cx = ((xmin + xmax) / 2) / w
                cy = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                if 0 < cx < 1 and 0 < cy < 1 and bw > 0.003 and bh > 0.003:
                    yolo_labels.append((cls_id, cx, cy, bw, bh))

            if not yolo_labels:
                skipped += 1
                continue

            resized, scale, dx, dy, orig_w, orig_h = resize_image(img)
            adjusted = adjust_labels_for_letterbox(yolo_labels, orig_w, orig_h, scale, dx, dy)

            if adjusted:
                out_name = f"dota_{stem}"
                cv2.imwrite(str(staging_img_dir / f"{out_name}.jpg"), resized,
                             [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                write_yolo_label(staging_lbl_dir / f"{out_name}.txt", adjusted)
                count += 1

        if count % 500 == 0 and count > 0:
            log(f"  DOTA: {count} images/tiles processed...")

    log(f"  DOTA complete: {count} images/tiles processed ({large_image_tiles} from tiling), {skipped} skipped")
    return count


def _tile_dota_image(img, labels, stem, w, h, staging_img_dir, staging_lbl_dir):
    """
    Tile a large DOTA image into 640x640 patches with stride 480 (25% overlap).
    labels: list of (class_id, xmin, ymin, xmax, ymax) in absolute pixels.
    Returns number of valid tiles generated.
    """
    tile_size = IMG_SIZE
    stride = 480
    tile_count = 0

    # Cap image dimensions to prevent memory issues
    max_dim = 6000
    if w > max_dim or h > max_dim:
        scale_down = min(max_dim / w, max_dim / h)
        new_w, new_h = int(w * scale_down), int(h * scale_down)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Adjust labels
        adjusted_labels = []
        for cls_id, xmin, ymin, xmax, ymax in labels:
            adjusted_labels.append((cls_id, xmin * scale_down, ymin * scale_down,
                                    xmax * scale_down, ymax * scale_down))
        labels = adjusted_labels
        w, h = new_w, new_h

    # Generate tile positions
    x_positions = list(range(0, max(1, w - tile_size + 1), stride))
    if not x_positions or x_positions[-1] + tile_size < w:
        x_positions.append(max(0, w - tile_size))

    y_positions = list(range(0, max(1, h - tile_size + 1), stride))
    if not y_positions or y_positions[-1] + tile_size < h:
        y_positions.append(max(0, h - tile_size))

    # Deduplicate positions
    x_positions = sorted(set(x_positions))
    y_positions = sorted(set(y_positions))

    for ty in y_positions:
        for tx in x_positions:
            # Extract tile
            tile = img[ty:ty + tile_size, tx:tx + tile_size]
            th, tw = tile.shape[:2]

            # Pad if tile is smaller than expected (edge case)
            if th < tile_size or tw < tile_size:
                padded = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            # Find annotations whose CENTER falls in this tile
            tile_labels = []
            for cls_id, xmin, ymin, xmax, ymax in labels:
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2

                if tx <= cx < tx + tile_size and ty <= cy < ty + tile_size:
                    # Convert to tile-local normalized coordinates
                    local_cx = (cx - tx) / tile_size
                    local_cy = (cy - ty) / tile_size
                    local_w = (xmax - xmin) / tile_size
                    local_h = (ymax - ymin) / tile_size

                    # Clip box to tile boundaries
                    local_cx = max(0.001, min(0.999, local_cx))
                    local_cy = max(0.001, min(0.999, local_cy))
                    local_w = min(local_w, 2 * min(local_cx, 1 - local_cx))
                    local_h = min(local_h, 2 * min(local_cy, 1 - local_cy))

                    # Skip tiny fragments (< 3px equivalent)
                    if local_w * tile_size >= 3 and local_h * tile_size >= 3:
                        tile_labels.append((int(cls_id), local_cx, local_cy, local_w, local_h))

            # Only save tiles with annotations
            if tile_labels:
                out_name = f"dota_{stem}_t{tx}_{ty}"
                cv2.imwrite(str(staging_img_dir / f"{out_name}.jpg"), tile,
                             [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                write_yolo_label(staging_lbl_dir / f"{out_name}.txt", tile_labels)
                tile_count += 1

    return tile_count


# ============================================================
# BALANCING ENGINE
# ============================================================

def classify_and_balance():
    """
    Scan all staged images, classify by class content, and apply balancing rules.
    Returns a list of (image_stem, category) pairs to keep.
    """
    log("Phase 2: Analysing class distribution in staging...")

    staging_lbl_dir = STAGING_DIR / "labels"
    label_files = sorted(staging_lbl_dir.glob("*.txt"))

    # Build index: stem → set of classes present
    image_index = {}
    class_image_counts = Counter()  # class_id → number of images containing it
    class_annotation_counts = Counter()  # class_id → total annotations
    total_annotations = 0

    for lbl_path in label_files:
        stem = lbl_path.stem
        labels = read_yolo_label(lbl_path)
        if not labels:
            continue

        classes_present = set(cls_id for cls_id, _, _, _, _ in labels)
        image_index[stem] = classes_present

        for cls_id in classes_present:
            class_image_counts[cls_id] += 1

        for cls_id, _, _, _, _ in labels:
            class_annotation_counts[cls_id] += 1
            total_annotations += 1

    log(f"  Total staged images: {len(image_index)}")
    log(f"  Total annotations: {total_annotations}")
    log(f"  Class distribution (images):")
    for cls_id in range(NUM_CLASSES):
        pct = class_image_counts[cls_id] / max(1, len(image_index)) * 100
        log(f"    {cls_id} ({CLASS_NAMES[cls_id]:20s}): {class_image_counts[cls_id]:6d} images ({pct:5.1f}%), "
            f"{class_annotation_counts[cls_id]:7d} annotations")

    # Classify images into categories
    categories = defaultdict(list)  # category → list of stems

    for stem, classes in image_index.items():
        has_rare = bool(classes & RARE_CLASSES)
        has_person = PERSON_CLASS in classes
        has_crowd = CROWD_CLASS in classes
        has_vehicle = VEHICLE_CLASS in classes

        if has_rare:
            categories["rare"].append(stem)
        elif has_person:
            categories["person"].append(stem)
        elif has_crowd and has_vehicle:
            categories["crowd_vehicle"].append(stem)
        elif has_crowd:
            categories["crowd_only"].append(stem)
        elif has_vehicle:
            categories["vehicle_only"].append(stem)
        else:
            categories["other"].append(stem)

    log(f"\n  Category breakdown:")
    for cat, stems in sorted(categories.items()):
        log(f"    {cat:20s}: {len(stems):6d} images")

    # Apply balancing rules
    selected = []

    # 1. Keep ALL rare class images
    selected.extend([(s, "rare") for s in categories.get("rare", [])])

    # 2. Keep ALL person images
    selected.extend([(s, "person") for s in categories.get("person", [])])

    # 3. Keep subset of crowd+vehicle
    crowd_vehicle = categories.get("crowd_vehicle", [])
    random.shuffle(crowd_vehicle)
    keep_n = int(len(crowd_vehicle) * CROWD_VEHICLE_KEEP_RATIO)
    selected.extend([(s, "crowd_vehicle") for s in crowd_vehicle[:keep_n]])

    # 4. Keep ALL crowd-only
    selected.extend([(s, "crowd_only") for s in categories.get("crowd_only", [])])

    # 5. Keep subset of vehicle-only (the big reduction)
    vehicle_only = categories.get("vehicle_only", [])
    random.shuffle(vehicle_only)
    keep_n = int(len(vehicle_only) * VEHICLE_ONLY_KEEP_RATIO)
    selected.extend([(s, "vehicle_only") for s in vehicle_only[:keep_n]])

    # 6. Keep other
    selected.extend([(s, "other") for s in categories.get("other", [])])

    log(f"\n  After balancing: {len(selected)} images selected (was {len(image_index)})")

    # Print projected class distribution
    projected_class_counts = Counter()
    projected_annotation_counts = Counter()
    for stem, _ in selected:
        classes = image_index[stem]
        for cls_id in classes:
            projected_class_counts[cls_id] += 1

        labels = read_yolo_label(STAGING_DIR / "labels" / f"{stem}.txt")
        for cls_id, _, _, _, _ in labels:
            projected_annotation_counts[cls_id] += 1

    total_proj = sum(projected_annotation_counts.values())
    log(f"\n  Projected class distribution after balancing:")
    for cls_id in range(NUM_CLASSES):
        pct = projected_annotation_counts[cls_id] / max(1, total_proj) * 100
        log(f"    {cls_id} ({CLASS_NAMES[cls_id]:20s}): {projected_class_counts[cls_id]:6d} images, "
            f"{projected_annotation_counts[cls_id]:7d} annotations ({pct:5.1f}%)")

    return selected, image_index


def augment_rare_class_images(selected, image_index):
    """
    Create horizontally flipped copies of images containing rare classes
    to boost their representation.
    """
    log("Phase 2b: Augmenting rare class images with horizontal flips...")

    staging_img_dir = STAGING_DIR / "images"
    staging_lbl_dir = STAGING_DIR / "labels"

    # Count how many images each rare class has
    rare_class_counts = Counter()
    for stem, cat in selected:
        if cat == "rare":
            for cls_id in image_index[stem]:
                if cls_id in RARE_CLASSES:
                    rare_class_counts[cls_id] += 1

    augmented_count = 0

    for stem, cat in list(selected):  # Use list() to avoid modification during iteration
        if cat != "rare":
            continue

        classes = image_index[stem]
        should_augment = any(
            cls_id in RARE_CLASSES and rare_class_counts[cls_id] < RARE_AUGMENT_THRESHOLD
            for cls_id in classes
        )

        if not should_augment:
            continue

        # Read image and labels
        img_path = staging_img_dir / f"{stem}.jpg"
        lbl_path = staging_lbl_dir / f"{stem}.txt"

        if not img_path.exists() or not lbl_path.exists():
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
        except Exception:
            continue

        labels = read_yolo_label(lbl_path)
        if not labels:
            continue

        # Create horizontal flip
        flipped_img = cv2.flip(img, 1)  # 1 = horizontal flip
        flipped_labels = horizontal_flip_labels(labels)

        # Save augmented version
        aug_stem = f"{stem}_hflip"
        cv2.imwrite(str(staging_img_dir / f"{aug_stem}.jpg"), flipped_img,
                     [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        write_yolo_label(staging_lbl_dir / f"{aug_stem}.txt", flipped_labels)

        # Add to selected and index
        selected.append((aug_stem, "rare_aug"))
        image_index[aug_stem] = classes.copy()
        augmented_count += 1

    log(f"  Created {augmented_count} augmented images for rare classes")
    return selected, image_index


# ============================================================
# STRATIFIED SPLITTING
# ============================================================

def stratified_split(selected, image_index):
    """
    Split selected images into train/val/test with stratification.
    Ensures rare classes appear proportionally in all splits.
    """
    log("Phase 3: Stratified train/val/test splitting...")

    # Group by category
    by_category = defaultdict(list)
    for stem, cat in selected:
        by_category[cat].append(stem)

    train_stems = []
    val_stems = []
    test_stems = []

    for cat, stems in by_category.items():
        random.shuffle(stems)
        n = len(stems)
        n_val = max(1, int(n * VAL_RATIO))
        n_test = max(1, int(n * TEST_RATIO))
        n_train = n - n_val - n_test

        if n < 3:
            # If very few images, put them all in train
            train_stems.extend(stems)
            continue

        train_stems.extend(stems[:n_train])
        val_stems.extend(stems[n_train:n_train + n_val])
        test_stems.extend(stems[n_train + n_val:])

    log(f"  Split sizes: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")
    return train_stems, val_stems, test_stems


# ============================================================
# FINAL OUTPUT
# ============================================================

def write_final_output(train_stems, val_stems, test_stems):
    """
    Copy selected images from staging to final output directories.
    """
    log("Phase 4: Writing final output...")

    staging_img_dir = STAGING_DIR / "images"
    staging_lbl_dir = STAGING_DIR / "labels"

    for split_name, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        out_img_dir = OUTPUT_DIR / split_name / "images"
        out_lbl_dir = OUTPUT_DIR / split_name / "labels"
        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        for stem in stems:
            src_img = staging_img_dir / f"{stem}.jpg"
            src_lbl = staging_lbl_dir / f"{stem}.txt"

            if src_img.exists() and src_lbl.exists():
                shutil.copy2(str(src_img), str(out_img_dir / f"{stem}.jpg"))
                shutil.copy2(str(src_lbl), str(out_lbl_dir / f"{stem}.txt"))

        log(f"  {split_name}: {len(stems)} images written")


def write_data_yaml():
    """Write the data.yaml configuration file for YOLOv8."""
    yaml_content = f"""# Border Surveillance Dataset v2 — Balanced
# Auto-generated by preprocess_balanced_v2.py

path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    log(f"  data.yaml written to {yaml_path}")


def print_final_statistics():
    """Print comprehensive statistics about the final dataset."""
    log("\n" + "=" * 70)
    log("FINAL DATASET STATISTICS")
    log("=" * 70)

    total_images = 0
    total_annotations = 0
    overall_class_counts = Counter()

    for split_name in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        img_dir = OUTPUT_DIR / split_name / "images"

        if not lbl_dir.exists():
            continue

        n_images = len(list(img_dir.glob("*.jpg")))
        split_class_counts = Counter()
        split_annotations = 0

        for lbl_path in lbl_dir.glob("*.txt"):
            labels = read_yolo_label(lbl_path)
            for cls_id, _, _, _, _ in labels:
                split_class_counts[cls_id] += 1
                overall_class_counts[cls_id] += 1
                split_annotations += 1

        total_images += n_images
        total_annotations += split_annotations

        log(f"\n  {split_name.upper()} split:")
        log(f"    Images: {n_images}")
        log(f"    Annotations: {split_annotations}")
        for cls_id in range(NUM_CLASSES):
            pct = split_class_counts[cls_id] / max(1, split_annotations) * 100
            log(f"      {cls_id} ({CLASS_NAMES[cls_id]:20s}): {split_class_counts[cls_id]:7d} ({pct:5.1f}%)")

    log(f"\n  OVERALL:")
    log(f"    Total images: {total_images}")
    log(f"    Total annotations: {total_annotations}")
    log(f"    Class distribution:")
    for cls_id in range(NUM_CLASSES):
        pct = overall_class_counts[cls_id] / max(1, total_annotations) * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        log(f"      {cls_id} ({CLASS_NAMES[cls_id]:20s}): {overall_class_counts[cls_id]:7d} ({pct:5.1f}%) {bar}")

    # Calculate imbalance ratio
    max_count = max(overall_class_counts.values()) if overall_class_counts else 1
    min_count = min(overall_class_counts.values()) if overall_class_counts else 1
    imbalance = max_count / max(1, min_count)
    log(f"\n    Imbalance ratio (max/min): {imbalance:.1f}x")

    prev_imbalance = 572062 / 351  # From previous analysis
    log(f"    Previous imbalance ratio: {prev_imbalance:.1f}x")
    log(f"    Improvement: {prev_imbalance / max(1, imbalance):.1f}x better")

    log("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    log("=" * 70)
    log("Border Surveillance — Smart Balanced Dataset Preprocessor v2")
    log("=" * 70)
    log(f"  Raw data dir:    {RAW_DIR}")
    log(f"  Output dir:      {OUTPUT_DIR}")
    log(f"  Staging dir:     {STAGING_DIR}")
    log(f"  Image size:      {IMG_SIZE}x{IMG_SIZE}")
    log(f"  Vehicle keep:    {VEHICLE_ONLY_KEEP_RATIO*100:.0f}%")
    log(f"  Random seed:     {SEED}")
    log("")

    # ---- Phase 0: Clean up ----
    log("Phase 0: Cleaning up old data...")

    if OUTPUT_DIR.exists():
        log(f"  Deleting existing processed data: {OUTPUT_DIR}")
        shutil.rmtree(str(OUTPUT_DIR))

    if STAGING_DIR.exists():
        log(f"  Deleting existing staging data: {STAGING_DIR}")
        shutil.rmtree(str(STAGING_DIR))

    ensure_dir(STAGING_DIR / "images")
    ensure_dir(STAGING_DIR / "labels")
    log("  Clean up complete.\n")

    # ---- Phase 1: Process datasets ----
    log("Phase 1: Processing raw datasets into staging...")
    t1 = time.time()

    xview_count = process_xview()
    visdrone_count = process_visdrone()
    dota_count = process_dota()

    total_staged = xview_count + visdrone_count + dota_count
    log(f"\n  Phase 1 complete in {time.time() - t1:.1f}s")
    log(f"  Total staged: {total_staged} images "
        f"(xView={xview_count}, VisDrone={visdrone_count}, DOTA={dota_count})\n")

    if total_staged == 0:
        log("ERROR: No images were staged! Check raw data paths.", "ERROR")
        sys.exit(1)

    # ---- Phase 2: Balance ----
    t2 = time.time()
    selected, image_index = classify_and_balance()

    if AUGMENT_RARE_CLASSES:
        selected, image_index = augment_rare_class_images(selected, image_index)

    log(f"\n  Phase 2 complete in {time.time() - t2:.1f}s")
    log(f"  Final selected: {len(selected)} images\n")

    # ---- Phase 3: Split ----
    t3 = time.time()
    train_stems, val_stems, test_stems = stratified_split(selected, image_index)
    log(f"  Phase 3 complete in {time.time() - t3:.1f}s\n")

    # ---- Phase 4: Write output ----
    t4 = time.time()
    ensure_dir(OUTPUT_DIR)
    write_final_output(train_stems, val_stems, test_stems)
    write_data_yaml()
    log(f"  Phase 4 complete in {time.time() - t4:.1f}s\n")

    # ---- Phase 5: Statistics ----
    print_final_statistics()

    # ---- Cleanup staging ----
    log("\nCleaning up staging directory...")
    shutil.rmtree(str(STAGING_DIR))
    log("Staging cleaned up.")

    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"ALL DONE in {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Train with: yolo detect train data={OUTPUT_DIR / 'data.yaml'} model=yolov8n.pt epochs=50 batch=2 imgsz=640")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
