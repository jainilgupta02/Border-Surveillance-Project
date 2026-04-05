"""
DOTA Dataset Local Preprocessing Script
========================================

Processes your locally downloaded DOTA dataset into YOLO format
and merges it into your existing data/processed/ folder.

Your actual DOTA folder structure (confirmed from screenshots):
    data/raw/dota/
    ├── images/                          ← 942 images (val set)
    │   └── P0907.png, P0908.png ...
    └── train/
        ├── images/
        │   └── 1/
        │       └── part1/
        │           └── images/          ← 469 images (train set)
        │               └── P0000.png ...
        └── labelTxt-v1.0/
        │   ├── labelTxt/                ← 1411 OBB labels (v1.0)
        │   └── Train_Task2_gt/
        │       └── trainset_reclabelTxt/
        └── labelTxt-v1.5/
            ├── DOTA-v1.5_train/         ← 1411 OBB labels (v1.5)
            └── DOTA-v1.5_train_hbb/     ← 1411 HBB labels ✅ USE THESE

We use DOTA-v1.5_train_hbb because it's already horizontal bounding boxes
— no OBB → HBB conversion needed, simpler and more accurate.

HBB label format per line:
    x_min y_min x_max y_max x_min y_min x_max y_max class_name difficult
    (still 8 coords but they form an axis-aligned rectangle)

Usage:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    python scripts/preprocess_dota_local.py

Run AFTER preprocess_local.py (UCF-Crime + xView already done ✅)
This script ADDS to your existing data/processed/ — it does not overwrite.
"""

import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — your exact paths from screenshots
# ---------------------------------------------------------------------------

BASE_DIR  = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_OUT  = BASE_DIR / "data" / "processed"

# DOTA paths — exactly as seen in your screenshots
DOTA_ROOT      = DATA_RAW / "dota"

# Train images: data/raw/dota/train/images/1/part1/images/
DOTA_TRAIN_IMG = DOTA_ROOT / "train" / "images" / "1" / "part1" / "images"

# Val images: data/raw/dota/images/  (the 942 images at root level)
DOTA_VAL_IMG   = DOTA_ROOT / "images"

# Labels — using HBB (horizontal bounding box) — easiest to work with
DOTA_TRAIN_LBL = DOTA_ROOT / "train" / "labelTxt-v1.5" / "DOTA-v1.5_train_hbb"

# Fallback label paths (tried in order if HBB not found for an image)
DOTA_TRAIN_LBL_FALLBACKS = [
    DOTA_ROOT / "train" / "labelTxt-v1.0" / "labelTxt",
    DOTA_ROOT / "train" / "labelTxt-v1.5" / "DOTA-v1.5_train",
    DOTA_ROOT / "train" / "labelTxt-v1.0" / "Train_Task2_gt" / "trainset_reclabelTxt",
]

IMG_SIZE    = 640
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Class mapping — DOTA class name → our 7 surveillance classes
# -1 = ignore (irrelevant to border surveillance)
# ---------------------------------------------------------------------------
DOTA_CLASS_MAP = {
    "plane":               4,   # aircraft
    "ship":                5,   # ship
    "storage-tank":        6,   # suspicious_object
    "baseball-diamond":   -1,
    "tennis-court":       -1,
    "basketball-court":   -1,
    "ground-track-field": -1,
    "harbor":              6,   # suspicious_object
    "bridge":             -1,
    "large-vehicle":       1,   # vehicle
    "small-vehicle":       1,   # vehicle
    "helicopter":          4,   # aircraft
    "roundabout":         -1,
    "soccer-ball-field":  -1,
    "swimming-pool":      -1,
    "container-crane":     3,   # military_vehicle
    "airport":             4,   # aircraft
    "helipad":             4,   # aircraft
    # v1.5 extra classes
    "vehicle":             1,
    "helicopter-v1.5":     4,
    "small-vehicle-v1.5":  1,
    "large-vehicle-v1.5":  1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_label(stem: str) -> Path | None:
    """Find label file for an image stem, checking all label dirs."""
    # Primary: HBB labels
    p = DOTA_TRAIN_LBL / (stem + ".txt")
    if p.exists():
        return p
    # Fallbacks
    for fb in DOTA_TRAIN_LBL_FALLBACKS:
        p = fb / (stem + ".txt")
        if p.exists():
            return p
    return None


def parse_dota_label(label_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse a DOTA HBB or OBB label file → YOLO format lines.

    DOTA HBB format:  x1 y1 x2 y2 x3 y3 x4 y4 class difficult
    (axis-aligned rectangle, so min/max of coords gives the box)

    Returns list of YOLO strings: "class_id cx cy w h\n"
    """
    yolo_lines = []

    with open(label_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("imagesource") or line.startswith("gsd"):
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        try:
            coords     = list(map(float, parts[:8]))
            class_name = parts[8].lower().strip()
        except (ValueError, IndexError):
            continue

        cls_id = DOTA_CLASS_MAP.get(class_name, -1)
        if cls_id == -1:
            continue

        # Get bounding box from 4 corner points (works for both OBB and HBB)
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Normalize to [0, 1]
        cx = ((x_min + x_max) / 2) / img_w
        cy = ((y_min + y_max) / 2) / img_h
        w  = (x_max - x_min) / img_w
        h  = (y_max - y_min) / img_h

        cx, cy, w, h = [max(0.0, min(1.0, v)) for v in [cx, cy, w, h]]

        if w > 0.001 and h > 0.001:
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return yolo_lines


def resize_and_save(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.imwrite(str(dst), img)


# ---------------------------------------------------------------------------
# Process DOTA train split
# ---------------------------------------------------------------------------

def process_dota_train() -> int:
    """Process train images from data/raw/dota/train/images/1/part1/images/"""
    print("\n📁 Processing DOTA train split...")
    print(f"   Images: {DOTA_TRAIN_IMG}")
    print(f"   Labels: {DOTA_TRAIN_LBL}")

    if not DOTA_TRAIN_IMG.exists():
        print(f"   ❌ Train images folder not found!")
        print(f"      Expected: {DOTA_TRAIN_IMG}")
        return 0

    images = list(DOTA_TRAIN_IMG.glob("*.png")) + list(DOTA_TRAIN_IMG.glob("*.jpg"))
    print(f"   Found {len(images)} images")

    saved = skipped_no_label = skipped_no_objects = 0

    for img_path in tqdm(images, desc="   DOTA train"):
        label_path = find_label(img_path.stem)
        if label_path is None:
            skipped_no_label += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        yolo_lines = parse_dota_label(label_path, w, h)

        if not yolo_lines:
            skipped_no_objects += 1
            continue

        # Save to train split (already have UCF + xView there, just adding more)
        dst_img = DATA_OUT / "train" / "images" / img_path.name
        if resize_and_save(img_path, dst_img):
            lbl_out = DATA_OUT / "train" / "labels" / (img_path.stem + ".txt")
            with open(lbl_out, "w") as f:
                f.writelines(yolo_lines)
            saved += 1

    print(f"   ✅ Saved {saved} train images")
    print(f"   ⚠️  Skipped {skipped_no_label} (no label file), "
          f"{skipped_no_objects} (no relevant classes)")
    return saved


# ---------------------------------------------------------------------------
# Process DOTA val split (images at data/raw/dota/images/)
# ---------------------------------------------------------------------------

def process_dota_val() -> int:
    """
    Process val images from data/raw/dota/images/ (942 images).
    These don't have labels in the standard DOTA release (test set is unlabelled).
    We'll write a weak 'crowd' label so they still contribute to training.
    """
    print("\n📁 Processing DOTA val/test images (unlabelled → weak label)...")
    print(f"   Images: {DOTA_VAL_IMG}")

    if not DOTA_VAL_IMG.exists():
        print(f"   ❌ Val images folder not found: {DOTA_VAL_IMG}")
        return 0

    images = list(DOTA_VAL_IMG.glob("*.png")) + list(DOTA_VAL_IMG.glob("*.jpg"))
    print(f"   Found {len(images)} images")

    # Split 80/20 between val and test
    random.seed(RANDOM_SEED)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    splits = {
        "val":  images[:split_idx],
        "test": images[split_idx:],
    }

    saved = 0
    for split_name, files in splits.items():
        for img_path in tqdm(files, desc=f"   DOTA {split_name}", leave=False):
            dst_img = DATA_OUT / split_name / "images" / img_path.name
            if resize_and_save(img_path, dst_img):
                # Weak label — full-frame "suspicious_object" (class 6)
                # This tells the model: "aerial scenes may contain objects of interest"
                lbl_out = DATA_OUT / split_name / "labels" / (img_path.stem + ".txt")
                with open(lbl_out, "w") as f:
                    f.write("6 0.5 0.5 1.0 1.0\n")
                saved += 1

    print(f"   ✅ Saved {saved} val/test images (with weak labels)")
    return saved


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n" + "=" * 55)
    print("UPDATED DATASET SUMMARY (after DOTA merge)")
    print("=" * 55)
    print(f"{'Split':<10} {'Images':<10} {'Labels':<10}")
    print("-" * 30)
    for split in ["train", "val", "test"]:
        imgs = len(list((DATA_OUT / split / "images").glob("*")))
        lbls = len(list((DATA_OUT / split / "labels").glob("*")))
        print(f"{split:<10} {imgs:<10} {lbls:<10}")
    print("=" * 55)

    print("\n✅ All datasets merged! You can now run training:")
    print()
    print('  cd "/home/jay_gupta/Workspace/Border Surveillance Project"')
    print()
    print("  yolo train \\")
    print("    model=yolov8n.pt \\")
    print(f'    data="{DATA_OUT}/data.yaml" \\')
    print("    epochs=50 \\")
    print("    imgsz=640 \\")
    print("    batch=16 \\")
    print("    name=border_surveillance_v1 \\")
    print("    project=models/runs \\")
    print("    patience=10")
    print()
    print("  After training, copy the best model:")
    print("  cp models/runs/border_surveillance_v1/weights/best.pt models/border_yolo.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Border Surveillance AI — DOTA Local Preprocessing")
    print("Merging DOTA into existing data/processed/ ...")
    print("=" * 55)

    # Check paths exist
    print("\n🔍 Checking DOTA folder structure...")
    checks = {
        "Train images (part1)": DOTA_TRAIN_IMG,
        "Val images (root)":    DOTA_VAL_IMG,
        "HBB labels (v1.5)":    DOTA_TRAIN_LBL,
    }
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌ NOT FOUND"
        print(f"  {name}: {status}")
        print(f"    → {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some paths are missing. Check the paths above match your structure.")
        print("    If your structure differs, update the path constants at the top of this file.")
    else:
        print("\n✅ All paths found! Starting preprocessing...")

    # Make sure output dirs exist (should already from preprocess_local.py)
    for split in ["train", "val", "test"]:
        (DATA_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    train_saved = process_dota_train()
    val_saved   = process_dota_val()

    print(f"\n🎉 DOTA done! {train_saved + val_saved} total images added.")
    print_summary()
