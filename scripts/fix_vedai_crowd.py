"""
Fix Script — VEDAI + Crowd Labels
===================================
Fixes two root cause bugs:

BUG 1 — VEDAI wrong format parsing:
  process_vedai_only.py read columns 1,2,3,4 as x,y,w,h
  BUT column 3 is class_id, not a coordinate.
  Correct format: cx cy angle CLASS_ID occluded in_roi x1 y1 x2 y2 x3 y3 x4 y4
  Fix: delete processed vedai_* files, re-process from raw with correct parsing.
  VEDAI class 9 = military → maps to class 3 (military_vehicle)

BUG 2 — Crowd = 0 instances:
  VisDrone category 1 = "people" (group of people) = crowd
  Was either mapped to -1 (ignored) or to person (0) by mistake.
  Fix: re-scan VisDrone raw labels and remap category 1 → class 2 (crowd)
  Also recheck processed visdrone labels for any category-1 entries.

Run:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    python fix_vedai_crowd.py

After this runs, start fine-tuning from existing model (see bottom of file).
"""

import cv2
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR  = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
RAW_DIR   = BASE_DIR / "data/raw"
PROC_DIR  = BASE_DIR / "data/processed"
IMG_SIZE  = 640
IMG_SIZE_VEDAI = 512.0   # VEDAI images are 512×512

# ── VEDAI class mapping (column 3 in annotation) ─────────────────────────
VEDAI_CLASS_MAP = {
    1:  1,   # car           → vehicle
    2:  1,   # truck         → vehicle
    3:  1,   # pickup        → vehicle
    4:  1,   # tractor       → vehicle
    5:  1,   # camping car   → vehicle
    6:  1,   # bus           → vehicle
    7:  1,   # van           → vehicle
    8: -1,   # motorcycle    → ignore
    9:  3,   # military      → military_vehicle ← KEY FIX
    10: -1,  # bicycle       → ignore
    11: -1,  # others        → ignore
}

# ── VisDrone class mapping ────────────────────────────────────────────────
VISDRONE_CLASS_MAP = {
    0:  0,   # pedestrian     → person
    1:  2,   # people (group) → crowd  ← KEY FIX
    2: -1,   # bicycle        → ignore
    3:  1,   # car            → vehicle
    4:  1,   # van            → vehicle
    5:  1,   # truck          → vehicle
    6: -1,   # tricycle       → ignore
    7: -1,   # awning-tricycle→ ignore
    8:  1,   # bus            → vehicle
    9: -1,   # motor          → ignore
    10: -1,  # others         → ignore
}


def is_valid_box(cx, cy, w, h):
    return (0.001 < w < 0.99 and 0.001 < h < 0.99 and
            0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)


# ─────────────────────────────────────────────────────────────────────────
# PART 1 — Fix VEDAI
# ─────────────────────────────────────────────────────────────────────────

def fix_vedai():
    print("\n" + "="*55)
    print("PART 1: Re-processing VEDAI from raw")
    print("="*55)

    # Step 1: Delete all existing vedai_* processed files
    deleted = 0
    for split in ["train", "val", "test"]:
        for img in (PROC_DIR / split / "images").glob("vedai_*"):
            img.unlink()
            deleted += 1
        for lbl in (PROC_DIR / split / "labels").glob("vedai_*"):
            lbl.unlink()
            deleted += 1
    print(f"  Deleted {deleted} old VEDAI files from processed/")

    # Step 2: Find raw VEDAI folders
    vedai_raw = RAW_DIR / "vedai"
    img_dir   = vedai_raw / "Vehicles512"
    ann_dir   = vedai_raw / "Annotations512"

    if not img_dir.exists() or not ann_dir.exists():
        print(f"  ❌ VEDAI raw folders not found at {vedai_raw}")
        return 0

    images = list(img_dir.glob("*.png"))
    print(f"  Found {len(images)} raw VEDAI images")

    # Step 3: Split
    random.seed(42)
    random.shuffle(images)
    n = len(images)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)
    splits = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:],
    }

    saved = skipped = mil_count = 0

    for split_name, imgs in splits.items():
        for img_path in tqdm(imgs, desc=f"  VEDAI {split_name}"):
            # VEDAI annotation filename = just the 8-digit number (no _co/_ir)
            base_id  = img_path.stem.split("_")[0]
            ann_path = ann_dir / f"{base_id}.txt"

            if not ann_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            yolo_lines = []
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    # Need at least: cx cy angle class_id occluded in_roi x1 y1 x2 y2 x3 y3 x4 y4
                    if len(parts) < 14:
                        continue
                    try:
                        cx_px   = float(parts[0])
                        cy_px   = float(parts[1])
                        cls_raw = int(parts[3])          # column 3 = class_id
                        # OBB corner points (pixel coords)
                        xs = [float(parts[6]),  float(parts[8]),
                              float(parts[10]), float(parts[12])]
                        ys = [float(parts[7]),  float(parts[9]),
                              float(parts[11]), float(parts[13])]
                    except (ValueError, IndexError):
                        continue

                    mapped = VEDAI_CLASS_MAP.get(cls_raw, -1)
                    if mapped == -1:
                        continue

                    # Normalize to [0,1]
                    cx = cx_px / IMG_SIZE_VEDAI
                    cy = cy_px / IMG_SIZE_VEDAI
                    w  = (max(xs) - min(xs)) / IMG_SIZE_VEDAI
                    h  = (max(ys) - min(ys)) / IMG_SIZE_VEDAI
                    cx, cy, w, h = [max(0.0, min(1.0, v)) for v in [cx, cy, w, h]]

                    if is_valid_box(cx, cy, w, h):
                        yolo_lines.append(
                            f"{mapped} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        )
                        if mapped == 3:
                            mil_count += 1

            if not yolo_lines:
                skipped += 1
                continue

            # Save image (resize to 640)
            new_name = f"vedai_{img_path.stem}.jpg"
            out_img  = PROC_DIR / split_name / "images" / new_name
            out_lbl  = PROC_DIR / split_name / "labels" / new_name.replace(".jpg", ".txt")

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(out_img), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lbl, "w") as f:
                f.writelines(yolo_lines)
            saved += 1

    print(f"\n  ✅ VEDAI fixed: {saved} images saved, {skipped} skipped")
    print(f"  ✅ Military vehicle annotations: {mil_count} instances")
    return saved


# ─────────────────────────────────────────────────────────────────────────
# PART 2 — Fix Crowd from VisDrone
# ─────────────────────────────────────────────────────────────────────────

def fix_crowd_visdrone():
    print("\n" + "="*55)
    print("PART 2: Re-processing VisDrone crowd labels")
    print("="*55)

    visdrone_raw = RAW_DIR / "visdrone"
    if not visdrone_raw.exists():
        print(f"  ❌ VisDrone raw not found at {visdrone_raw}")
        return 0

    # Collect all VisDrone images from train + val splits
    all_images = []
    for split in ["train", "val"]:
        split_dir = visdrone_raw / split
        if not split_dir.exists():
            continue
        img_subdir = split_dir / "images"
        ann_subdir = split_dir / "annotations"
        if not img_subdir.exists():
            # Sometimes structure is flat
            img_subdir = split_dir
            ann_subdir = split_dir
        imgs = list(img_subdir.glob("*.jpg")) + list(img_subdir.glob("*.png"))
        for img in imgs:
            # Find annotation
            ann = ann_subdir / (img.stem + ".txt")
            if ann.exists():
                all_images.append((img, ann))

    if not all_images:
        print("  ❌ No VisDrone images with annotations found")
        print("  Checking alternative structure...")
        # Try flat structure
        for split in ["train", "val"]:
            split_dir = visdrone_raw / split
            if split_dir.exists():
                imgs = list(split_dir.glob("**/*.jpg"))
                anns = list(split_dir.glob("**/*.txt"))
                print(f"    {split}: {len(imgs)} images, {len(anns)} annotations")
        return 0

    print(f"  Found {len(all_images)} VisDrone images with annotations")

    # Delete existing visdrone processed files and reprocess
    deleted = 0
    for sp in ["train", "val", "test"]:
        for img in (PROC_DIR / sp / "images").glob("visdrone_*"):
            img.unlink(); deleted += 1
        for lbl in (PROC_DIR / sp / "labels").glob("visdrone_*"):
            lbl.unlink(); deleted += 1
    print(f"  Deleted {deleted} old VisDrone processed files")

    # Re-split
    random.seed(42)
    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)
    splits = {
        "train": all_images[:train_end],
        "val":   all_images[train_end:val_end],
        "test":  all_images[val_end:],
    }

    saved = skipped = crowd_count = 0

    for split_name, items in splits.items():
        for img_path, ann_path in tqdm(items, desc=f"  VisDrone {split_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            img_h, img_w = img.shape[:2]

            yolo_lines = []
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        x   = float(parts[0])
                        y   = float(parts[1])
                        w   = float(parts[2])
                        h   = float(parts[3])
                        cat = int(parts[5])
                    except (ValueError, IndexError):
                        continue

                    mapped = VISDRONE_CLASS_MAP.get(cat, -1)
                    if mapped == -1:
                        continue

                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    cx, cy, nw, nh = [max(0.0, min(1.0, v)) for v in [cx, cy, nw, nh]]

                    if is_valid_box(cx, cy, nw, nh):
                        yolo_lines.append(
                            f"{mapped} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                        )
                        if mapped == 2:
                            crowd_count += 1

            if not yolo_lines:
                skipped += 1
                continue

            new_name = f"visdrone_{img_path.stem}.jpg"
            out_img  = PROC_DIR / split_name / "images" / new_name
            out_lbl  = PROC_DIR / split_name / "labels" / new_name.replace(".jpg", ".txt")

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(out_img), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lbl, "w") as f:
                f.writelines(yolo_lines)
            saved += 1

    print(f"\n  ✅ VisDrone fixed: {saved} images saved, {skipped} skipped")
    print(f"  ✅ Crowd (class 2) annotations: {crowd_count} instances")
    return crowd_count


# ─────────────────────────────────────────────────────────────────────────
# PART 3 — Verify final counts
# ─────────────────────────────────────────────────────────────────────────

def verify_counts():
    print("\n" + "="*55)
    print("FINAL CLASS DISTRIBUTION (train split)")
    print("="*55)

    CLASS_NAMES = {
        0: "person", 1: "vehicle", 2: "crowd",
        3: "military_vehicle", 4: "aircraft",
        5: "ship", 6: "suspicious_object"
    }

    lbl_dir = PROC_DIR / "train" / "labels"
    counts  = {i: 0 for i in range(7)}

    for lbl in lbl_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        counts[int(parts[0])] += 1
                    except (ValueError, KeyError):
                        pass

    print(f"  {'Class':<25} {'Instances':>12}  {'Status'}")
    print(f"  {'-'*50}")
    for cls_id, name in CLASS_NAMES.items():
        count = counts[cls_id]
        if count == 0:
            status = "❌ STILL EMPTY"
        elif count < 100:
            status = "⚠️  Very low"
        elif count < 500:
            status = "⚠️  Low"
        else:
            status = "✅ Good"
        print(f"  {name:<25} {count:>12,}  {status}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Border Surveillance AI — VEDAI + Crowd Fix")
    print("="*55)

    vedai_saved  = fix_vedai()
    crowd_count  = fix_crowd_visdrone()

    # Clear YOLO cache so it rescans labels
    import subprocess
    subprocess.run(
        "find data/processed -name '*.cache' -delete",
        shell=True, cwd=str(BASE_DIR)
    )
    print("\n✅ Cache cleared")

    verify_counts()

    print("\n" + "="*55)
    print("NEXT STEP — Fine-tune from existing model:")
    print("="*55)
    print("""
nohup yolo detect train \\
  model=models/border_yolo.pt \\
  data="data/processed/data.yaml" \\
  epochs=5 \\
  imgsz=640 \\
  batch=2 \\
  device=cpu \\
  workers=0 \\
  optimizer=AdamW \\
  lr0=0.0001 \\
  lrf=0.01 \\
  cos_lr=True \\
  patience=3 \\
  cache=False \\
  rect=True \\
  amp=False \\
  name=border_surveillance_v9_finetune \\
  project=models/runs \\
  > ~/finetune.log 2>&1 &

echo "PID: $!"
tail -f ~/finetune.log
""")
