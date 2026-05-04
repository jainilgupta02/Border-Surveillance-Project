#!/usr/bin/env python3
"""
v2.1 Patch — Vehicle Annotation Cap + Further Vehicle-Only Reduction
=====================================================================
Runs on existing data/processed/ — no image re-processing needed.
Takes ~10-30 seconds.

Changes:
  1. Caps vehicle (class 1) annotations to max 10 per image
  2. Removes 50% of remaining vehicle-only images
  3. Prints before/after statistics
"""

import random
import os
import sys
import time
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

NUM_CLASSES = 7
CLASS_NAMES = [
    "person", "vehicle", "crowd", "military_vehicle",
    "aircraft", "ship", "suspicious_object"
]

# ---- Patch parameters ----
MAX_VEHICLE_PER_IMAGE = 10      # Max vehicle annotations kept per image
VEHICLE_ONLY_REMOVE_RATIO = 0.50  # Remove 50% of vehicle-only images


def read_labels(path):
    labels = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if 0 <= cls_id < NUM_CLASSES:
                        labels.append((cls_id, cx, cy, w, h))
    except Exception:
        pass
    return labels


def write_labels(path, labels):
    with open(path, 'w') as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def count_stats(splits):
    """Count images and annotations per class across splits."""
    total_images = 0
    total_annotations = Counter()

    for split in splits:
        lbl_dir = PROCESSED_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob("*.txt"):
            labels = read_labels(lbl_path)
            if labels:
                total_images += 1
                for cls_id, _, _, _, _ in labels:
                    total_annotations[cls_id] += 1

    return total_images, total_annotations


def print_stats(total_images, total_annotations, header=""):
    total_ann = sum(total_annotations.values())
    print(f"\n  {header}")
    print(f"  {'=' * 60}")
    print(f"  Total images:      {total_images}")
    print(f"  Total annotations: {total_ann}")
    print(f"  Class distribution:")
    for cls_id in range(NUM_CLASSES):
        count = total_annotations[cls_id]
        pct = count / max(1, total_ann) * 100
        bar_len = int(pct)
        bar = "█" * bar_len + "░" * (60 - bar_len)
        print(f"    {cls_id} ({CLASS_NAMES[cls_id]:20s}): {count:7d} ({pct:5.1f}%) {bar}")

    max_c = max(total_annotations.values()) if total_annotations else 1
    min_c = min(total_annotations.values()) if total_annotations else 1
    print(f"\n  Imbalance ratio: {max_c / max(1, min_c):.1f}x")


def main():
    start = time.time()
    splits = ["train", "val", "test"]

    print("=" * 70)
    print("v2.1 Patch — Vehicle Annotation Cap + Reduction")
    print("=" * 70)

    # ---- Before stats ----
    before_images, before_ann = count_stats(splits)
    print_stats(before_images, before_ann, "BEFORE PATCH")

    # ---- Phase 1: Cap vehicle annotations per image ----
    print(f"\n  Phase 1: Capping vehicle annotations to max {MAX_VEHICLE_PER_IMAGE} per image...")
    total_capped = 0
    total_vehicle_removed = 0

    for split in splits:
        lbl_dir = PROCESSED_DIR / split / "labels"
        if not lbl_dir.exists():
            continue

        capped = 0
        removed_ann = 0
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            labels = read_labels(lbl_path)
            if not labels:
                continue

            vehicle_labels = [l for l in labels if l[0] == 1]
            non_vehicle_labels = [l for l in labels if l[0] != 1]

            if len(vehicle_labels) > MAX_VEHICLE_PER_IMAGE:
                before_count = len(vehicle_labels)
                random.shuffle(vehicle_labels)
                vehicle_labels = vehicle_labels[:MAX_VEHICLE_PER_IMAGE]
                removed_ann += (before_count - MAX_VEHICLE_PER_IMAGE)
                write_labels(lbl_path, non_vehicle_labels + vehicle_labels)
                capped += 1

        print(f"    {split}: {capped} images capped, {removed_ann} vehicle annotations removed")
        total_capped += capped
        total_vehicle_removed += removed_ann

    print(f"    TOTAL: {total_capped} images capped, {total_vehicle_removed} vehicle annotations removed")

    # ---- Phase 2: Remove excess vehicle-only images ----
    print(f"\n  Phase 2: Removing {VEHICLE_ONLY_REMOVE_RATIO*100:.0f}% of vehicle-only images...")
    total_removed_images = 0

    for split in splits:
        lbl_dir = PROCESSED_DIR / split / "labels"
        img_dir = PROCESSED_DIR / split / "images"
        if not lbl_dir.exists():
            continue

        # Find vehicle-only images
        vehicle_only_files = []
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            labels = read_labels(lbl_path)
            if not labels:
                continue
            classes = set(l[0] for l in labels)
            if classes == {1}:  # Only vehicle annotations
                vehicle_only_files.append(lbl_path)

        # Shuffle and remove a fraction
        random.shuffle(vehicle_only_files)
        remove_count = int(len(vehicle_only_files) * VEHICLE_ONLY_REMOVE_RATIO)

        removed = 0
        for lbl_path in vehicle_only_files[:remove_count]:
            stem = lbl_path.stem
            img_path = img_dir / f"{stem}.jpg"
            lbl_path.unlink(missing_ok=True)
            if img_path.exists():
                img_path.unlink()
            removed += 1

        print(f"    {split}: removed {removed}/{len(vehicle_only_files)} vehicle-only images")
        total_removed_images += removed

    print(f"    TOTAL: {total_removed_images} vehicle-only images removed")

    # ---- After stats ----
    after_images, after_ann = count_stats(splits)
    print_stats(after_images, after_ann, "AFTER PATCH")

    # ---- Comparison ----
    before_total = sum(before_ann.values())
    after_total = sum(after_ann.values())
    print(f"\n  IMPROVEMENT SUMMARY:")
    print(f"  {'=' * 60}")
    print(f"  Images:      {before_images} → {after_images} ({after_images - before_images:+d})")
    print(f"  Annotations: {before_total} → {after_total} ({after_total - before_total:+d})")

    before_veh_pct = before_ann[1] / max(1, before_total) * 100
    after_veh_pct = after_ann[1] / max(1, after_total) * 100
    print(f"  Vehicle %:   {before_veh_pct:.1f}% → {after_veh_pct:.1f}%")

    for cls_id in range(NUM_CLASSES):
        before_pct = before_ann[cls_id] / max(1, before_total) * 100
        after_pct = after_ann[cls_id] / max(1, after_total) * 100
        change = after_pct - before_pct
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "=")
        print(f"    {CLASS_NAMES[cls_id]:20s}: {before_pct:5.1f}% → {after_pct:5.1f}% {arrow}")

    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
