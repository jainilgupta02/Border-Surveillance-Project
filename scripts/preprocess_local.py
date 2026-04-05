"""
Dataset Preprocessing Script — Local (WSL)
============================================

Processes UCF-Crime and xView datasets into YOLO format.
Run this script ONCE before training.

Your dataset structure (confirmed from screenshots):
    data/raw/
    ├── ucf_crime/
    │   └── frames/          ← extracted frames (already done ✅)
    ├── xview/
    │   └── extracted/       ← extracted images + labels (already done ✅)
    └── dota/                ← empty here, processed on Colab separately

Output:
    data/processed/
    ├── train/images/   train/labels/
    ├── val/images/     val/labels/
    ├── test/images/    test/labels/
    └── data.yaml

Usage:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    python scripts/preprocess_local.py

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   March 2026
"""

import os
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — paths match your actual WSL structure
# ---------------------------------------------------------------------------

BASE_DIR    = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
DATA_RAW    = BASE_DIR / "data" / "raw"
DATA_OUT    = BASE_DIR / "data" / "processed"

UCF_FRAMES  = DATA_RAW / "ucf_crime" / "frames"
XVIEW_DIR   = DATA_RAW / "xview" / "extracted"

IMG_SIZE    = 640
SPLIT_RATIO = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42

# 7 surveillance classes (matches your project design)
CLASSES = [
    "person",           # 0
    "vehicle",          # 1
    "crowd",            # 2
    "military_vehicle", # 3
    "aircraft",         # 4
    "ship",             # 5
    "suspicious_object",# 6
]

# xView class ID → our class ID mapping
XVIEW_CLASS_MAP = {
    11: 4, 12: 4, 15: 4, 47: 4, 49: 4,   # aircraft
    17: 3, 26: 3, 28: 3, 62: 3, 63: 3, 71: 3,  # military_vehicle
    18: 1, 19: 1, 20: 1, 21: 1, 23: 1, 24: 1,  # vehicle
    50: 1, 53: 1, 56: 1, 57: 1, 59: 1, 60: 1,
    61: 1, 64: 1, 65: 1, 66: 1, 72: 1, 73: 1,
    74: 1, 76: 1,
    32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5,  # ship
    40: 5, 41: 5, 42: 5, 44: 5,
    77: 0,                                       # person
    79: 2,                                       # crowd
    83: 6, 84: 6, 86: 6, 89: 6, 91: 6,          # suspicious_object
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_output_dirs():
    """Create the YOLO directory structure."""
    for split in ["train", "val", "test"]:
        (DATA_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directories created: {DATA_OUT}")


def resize_and_save(src: Path, dst: Path) -> bool:
    """Resize image to IMG_SIZE × IMG_SIZE and save to dst."""
    img = cv2.imread(str(src))
    if img is None:
        return False
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.imwrite(str(dst), img)


def split_files(file_list: list) -> dict:
    """Randomly split a file list into train/val/test."""
    random.seed(RANDOM_SEED)
    random.shuffle(file_list)
    n         = len(file_list)
    train_end = int(n * SPLIT_RATIO["train"])
    val_end   = train_end + int(n * SPLIT_RATIO["val"])
    return {
        "train": file_list[:train_end],
        "val":   file_list[train_end:val_end],
        "test":  file_list[val_end:],
    }


# ---------------------------------------------------------------------------
# 1. UCF-Crime
# ---------------------------------------------------------------------------

def process_ucf_crime():
    """
    UCF-Crime has no bounding-box annotations.
    Strategy: treat every frame as a 'crowd / suspicious_object' scene.
    We write a full-image bounding box (class=2: crowd) as a weak label.
    This teaches the model the visual style of surveillance anomaly frames.
    """
    print("\n📁 Processing UCF-Crime...")

    if not UCF_FRAMES.exists():
        print(f"  ⚠️  Frames folder not found: {UCF_FRAMES}")
        print("  Make sure archive.zip is extracted into frames/")
        return 0

    images = (
        list(UCF_FRAMES.rglob("*.jpg"))
        + list(UCF_FRAMES.rglob("*.png"))
    )

    if not images:
        print("  ⚠️  No images found in frames/. Check extraction.")
        return 0

    print(f"  Found {len(images)} frames")
    splits = split_files(images)
    saved  = 0

    for split_name, files in splits.items():
        for img_path in tqdm(files, desc=f"  UCF {split_name}", leave=False):
            dst_img = DATA_OUT / split_name / "images" / img_path.name

            if resize_and_save(img_path, dst_img):
                # Full-frame weak label: class=2 (crowd), center, full W/H
                lbl = DATA_OUT / split_name / "labels" / (img_path.stem + ".txt")
                with open(lbl, "w") as f:
                    f.write("2 0.5 0.5 1.0 1.0\n")
                saved += 1

    print(f"  ✅ UCF-Crime: {saved} frames saved")
    return saved


# ---------------------------------------------------------------------------
# 2. xView
# ---------------------------------------------------------------------------

def process_xview():
    """
    xView satellite images — labels already converted to YOLO format
    by xview_geojson_to_yolo.py (class IDs already in our 0-6 schema).
    Read labels directly without remapping.
    """
    print("\n📁 Processing xView...")

    if not XVIEW_DIR.exists():
        print(f"  ⚠️  xView folder not found: {XVIEW_DIR}")
        return 0

    # Find images — xView uses .tif files
    images = (
        list(XVIEW_DIR.rglob("*.jpg"))
        + list(XVIEW_DIR.rglob("*.png"))
        + list(XVIEW_DIR.rglob("*.tif"))
    )

    if not images:
        print("  ⚠️  No images found in extracted/. Check extraction.")
        return 0

    print(f"  Found {len(images)} images")
    splits = split_files(images)
    saved  = 0

    for split_name, files in splits.items():
        for img_path in tqdm(files, desc=f"  xView {split_name}", leave=False):

            # Labels were written by xview_geojson_to_yolo.py into labels/ folder
            label_candidates = [
                img_path.parent / (img_path.stem + ".txt"),
                img_path.parent.parent / "labels" / (img_path.stem + ".txt"),
                XVIEW_DIR / "labels" / (img_path.stem + ".txt"),
            ]
            label_src = next((p for p in label_candidates if p.exists()), None)

            if label_src is None:
                continue  # no label for this image — skip

            # Labels are ALREADY in our 0-6 class schema — use directly
            new_lines = []
            with open(label_src) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id > 6:
                        continue  # safety check — skip out-of-range
                    new_lines.append(f"{cls_id} {' '.join(parts[1:5])}\n")

            if not new_lines:
                continue

            safe_name = img_path.stem + ".jpg"
            dst_img   = DATA_OUT / split_name / "images" / safe_name

            if resize_and_save(img_path, dst_img):
                lbl = DATA_OUT / split_name / "labels" / (img_path.stem + ".txt")
                with open(lbl, "w") as f:
                    f.writelines(new_lines)
                saved += 1

    print(f"  ✅ xView: {saved} images saved")
    return saved


# ---------------------------------------------------------------------------
# 3. data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml():
    """Write the YOLO training config file."""
    yaml_content = f"""# Border Surveillance — YOLO Training Config
# Generated by preprocess_local.py

path: {DATA_OUT}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASSES)}
names:
"""
    for i, cls in enumerate(CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    yaml_path = DATA_OUT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ data.yaml written → {yaml_path}")


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

def print_summary():
    """Count files in each split and print a summary table."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"{'Split':<10} {'Images':<10} {'Labels':<10}")
    print("-" * 30)
    for split in ["train", "val", "test"]:
        imgs = len(list((DATA_OUT / split / "images").glob("*")))
        lbls = len(list((DATA_OUT / split / "labels").glob("*")))
        print(f"{split:<10} {imgs:<10} {lbls:<10}")
    print("=" * 50)
    print(f"\nOutput → {DATA_OUT}")
    print("\nNext step:")
    print("  1. Run Colab script for DOTA, download dota_processed.zip")
    print("  2. Unzip and merge into data/processed/ (same folder structure)")
    print("  3. Then run YOLOv8 training:")
    print(f'     yolo train model=yolov8n.pt data="{DATA_OUT}/data.yaml" epochs=50 imgsz=640')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Border Surveillance AI — Dataset Preprocessing")
    print("Processing: UCF-Crime + xView (local WSL)")
    print("=" * 55)

    # Verify datasets exist before starting
    print("\n🔍 Checking dataset locations...")
    print(f"  UCF-Crime frames : {UCF_FRAMES} {'✅' if UCF_FRAMES.exists() else '❌ NOT FOUND'}")
    print(f"  xView extracted  : {XVIEW_DIR}  {'✅' if XVIEW_DIR.exists() else '❌ NOT FOUND'}")

    setup_output_dirs()
    ucf_count   = process_ucf_crime()
    xview_count = process_xview()
    write_data_yaml()
    print_summary()

    total = ucf_count + xview_count
    print(f"\n🎉 Done! {total} total images processed.")
    print("Now run the Colab script for DOTA, then merge and train.")
