"""
Dataset Quality Checker
========================

Run this BEFORE any retraining to verify your processed dataset
has correct labels, no corruption, and good class distribution.

Usage:
    python scripts/check_dataset.py

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
PROCESSED    = BASE_DIR / "data" / "processed"

CLASS_NAMES = {
    0: "person",  1: "vehicle",  2: "crowd",
    3: "military_vehicle",       4: "aircraft",
    5: "ship",    6: "suspicious_object",
}

SPLITS = ["train", "val", "test"]

# ─────────────────────────────────────────────────────────────────────────

def check_split(split: str) -> dict:
    img_dir = PROCESSED / split / "images"
    lbl_dir = PROCESSED / split / "labels"

    if not img_dir.exists():
        print(f"  ⚠  {split}/images not found — skipping")
        return {}

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    labels = list(lbl_dir.glob("*.txt"))

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    orphan_imgs = img_stems - lbl_stems   # images with no label
    orphan_lbls = lbl_stems - img_stems   # labels with no image

    # ── Class distribution ────────────────────────────────────────────
    class_counts  = defaultdict(int)
    empty_labels  = 0
    bad_boxes     = 0
    total_boxes   = 0
    corrupt_imgs  = 0

    for lbl_path in labels:
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                bad_boxes += 1
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                bad_boxes += 1
                continue

            # Validate values
            if not (0 <= cls_id <= 6):
                bad_boxes += 1
                continue
            if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                bad_boxes += 1
                continue
            if w <= 0 or h <= 0:
                bad_boxes += 1
                continue

            class_counts[cls_id] += 1
            total_boxes += 1

    # ── Spot-check image readability ──────────────────────────────────
    sample = random.sample(images, min(50, len(images)))
    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt_imgs += 1
        elif img.shape[:2] != (640, 640):
            corrupt_imgs += 1

    return {
        "images":        len(images),
        "labels":        len(labels),
        "orphan_images": len(orphan_imgs),
        "orphan_labels": len(orphan_lbls),
        "empty_labels":  empty_labels,
        "bad_boxes":     bad_boxes,
        "total_boxes":   total_boxes,
        "corrupt_imgs":  corrupt_imgs,
        "class_counts":  dict(class_counts),
    }


def print_report(split: str, r: dict):
    if not r:
        return

    print(f"\n{'─'*50}")
    print(f"  {split.upper()}")
    print(f"{'─'*50}")
    print(f"  Images:           {r['images']:,}")
    print(f"  Labels:           {r['labels']:,}")
    print(f"  Total boxes:      {r['total_boxes']:,}")

    # Issues
    issues = []
    if r["orphan_images"]:
        issues.append(f"  ⚠  Images without label:  {r['orphan_images']}")
    if r["orphan_labels"]:
        issues.append(f"  ⚠  Labels without image:  {r['orphan_labels']}")
    if r["empty_labels"]:
        issues.append(f"  ⚠  Empty label files:     {r['empty_labels']}")
    if r["bad_boxes"]:
        issues.append(f"  ❌  Invalid bounding boxes: {r['bad_boxes']}")
    if r["corrupt_imgs"]:
        issues.append(f"  ❌  Corrupt/wrong-size images: {r['corrupt_imgs']}")

    if issues:
        print("\n  ISSUES:")
        for i in issues:
            print(i)
    else:
        print("  ✅ No issues found")

    print("\n  Class distribution:")
    total = r["total_boxes"]
    for cls_id in range(7):
        n    = r["class_counts"].get(cls_id, 0)
        pct  = n / total * 100 if total > 0 else 0
        bar  = "█" * int(pct / 2)
        warn = " ⚠ MISSING" if n == 0 else (
               " ⚠ VERY FEW" if n < 50 else "")
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:<20} "
              f"{n:6,}  {pct:5.1f}%  {bar}{warn}")


def check_class_balance(results: dict):
    """Check if any class dominates so much it will bias the model."""
    print(f"\n{'═'*50}")
    print("  CLASS BALANCE ACROSS ALL SPLITS")
    print(f"{'═'*50}")

    totals = defaultdict(int)
    for r in results.values():
        for cls_id, cnt in r.get("class_counts", {}).items():
            totals[cls_id] += cnt

    grand_total = sum(totals.values())
    if grand_total == 0:
        print("  No boxes found!")
        return

    print(f"  Total annotations: {grand_total:,}\n")

    max_count = max(totals.values()) if totals else 1
    for cls_id in range(7):
        n   = totals.get(cls_id, 0)
        pct = n / grand_total * 100

        if n == 0:
            status = "❌ NO DATA — model will NOT detect this class"
        elif pct > 70:
            status = "⚠  HEAVILY IMBALANCED — will over-detect this class"
        elif pct < 1:
            status = "⚠  TOO RARE — poor recall expected for this class"
        else:
            status = "✅"

        print(f"  {CLASS_NAMES[cls_id]:<22} {n:7,}  {pct:5.1f}%  {status}")

    # Give practical advice
    print()
    missing = [CLASS_NAMES[i] for i in range(7) if totals.get(i, 0) == 0]
    rare    = [CLASS_NAMES[i] for i in range(7)
               if 0 < totals.get(i, 0) < 100]

    if missing:
        print(f"  🚨 MISSING CLASSES: {', '.join(missing)}")
        print("     Model has zero training examples for these.")
        print("     It will never detect them — regardless of epochs.")
    if rare:
        print(f"  ⚠  RARE CLASSES: {', '.join(rare)}")
        print("     Very few examples — detection will be unreliable.")


def main():
    print("═" * 50)
    print("  Border Surveillance — Dataset Quality Check")
    print("═" * 50)
    print(f"  Dataset: {PROCESSED}")

    if not PROCESSED.exists():
        print(f"\n  ❌ Processed dataset not found: {PROCESSED}")
        print("  Run: python scripts/preprocess_all_datasets.py")
        return

    results = {}
    for split in SPLITS:
        print(f"\n  Checking {split}...", end=" ", flush=True)
        results[split] = check_split(split)
        print("done")

    for split, r in results.items():
        print_report(split, r)

    check_class_balance(results)

    print(f"\n{'═'*50}")
    print("  RECOMMENDATIONS")
    print(f"{'═'*50}")
    print("""
  IF missing classes:
    → Those classes cannot be detected, period.
    → Either accept the limitation or add more data for them.
    → Document this in your report (this is honest research).

  IF heavily imbalanced (one class > 70%):
    → Model will over-detect that class
    → Add class_weights to training or
    → Undersample the dominant class

  IF dataset looks good (all classes present, balanced):
    → Problem is epochs — need at least 30–50 epochs
    → On your CPU: impossible before submission
    → Solution: use existing model + lower conf threshold
  """)


if __name__ == "__main__":
    main()
