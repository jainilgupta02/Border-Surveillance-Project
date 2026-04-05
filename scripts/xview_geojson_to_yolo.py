"""
xView GeoJSON → YOLO Label Converter
=====================================
Run this ONCE before running preprocess_local.py.

It reads:  data/raw/xview/extracted/labels/xView_train.geojson
It writes: data/raw/xview/extracted/labels/<image_name>.txt  (one per image)

Then preprocess_local.py will find the label files and process xView correctly.

Usage:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    python xview_geojson_to_yolo.py
"""

import json
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────
BASE_DIR   = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
GEOJSON    = BASE_DIR / "data/raw/xview/extracted/labels/xView_train.geojson"
IMAGES_DIR = BASE_DIR / "data/raw/xview/extracted/images"
LABELS_OUT = BASE_DIR / "data/raw/xview/extracted/labels"

# ── xView type_id → our 7-class schema ──────────────────
# (These are the ORIGINAL xView type_ids from the GeoJSON properties)
XVIEW_CLASS_MAP = {
    11: 4, 12: 4, 15: 4, 47: 4, 49: 4,              # aircraft
    17: 3, 26: 3, 28: 3, 62: 3, 63: 3, 71: 3,        # military_vehicle
    18: 1, 19: 1, 20: 1, 21: 1, 23: 1, 24: 1,        # vehicle
    50: 1, 53: 1, 56: 1, 57: 1, 59: 1, 60: 1,
    61: 1, 64: 1, 65: 1, 66: 1, 72: 1, 73: 1,
    74: 1, 76: 1,
    32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5,        # ship
    40: 5, 41: 5, 42: 5, 44: 5,
    77: 0,                                             # person
    79: 2,                                             # crowd
    83: 6, 84: 6, 86: 6, 89: 6, 91: 6,                # suspicious_object
}


def geojson_to_yolo():
    if not GEOJSON.exists():
        print(f"❌ GeoJSON not found: {GEOJSON}")
        return

    print(f"📂 Reading {GEOJSON.name}  ({GEOJSON.stat().st_size // 1024} KB)…")
    with open(GEOJSON) as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"   {len(features)} annotation features found")

    # ── Group annotations by image filename ─────────────
    # GeoJSON properties contain:
    #   "image_id"     → e.g. "1042.tif"
    #   "type_id"      → xView class integer
    #   "bounds_imcoords" → "xmin,ymin,xmax,ymax"  (pixel coords)
    per_image = defaultdict(list)

    skipped_class = 0
    skipped_malformed = 0

    for feat in features:
        props = feat.get("properties", {})
        img_id   = props.get("image_id", "")
        type_id  = props.get("type_id")
        bounds   = props.get("bounds_imcoords", "")

        if not img_id or type_id is None or not bounds:
            skipped_malformed += 1
            continue

        our_cls = XVIEW_CLASS_MAP.get(int(type_id), -1)
        if our_cls == -1:
            skipped_class += 1
            continue

        try:
            coords = list(map(float, bounds.split(",")))
            xmin, ymin, xmax, ymax = coords[:4]
        except (ValueError, TypeError):
            skipped_malformed += 1
            continue

        per_image[img_id].append((our_cls, xmin, ymin, xmax, ymax))

    print(f"   Images with usable annotations : {len(per_image)}")
    print(f"   Annotations skipped (irrelevant class): {skipped_class}")
    print(f"   Annotations skipped (malformed)       : {skipped_malformed}")

    # ── Write per-image YOLO .txt files ─────────────────
    written = 0
    skipped_no_image = 0

    for img_id, boxes in per_image.items():
        # Confirm the image actually exists so we don't write orphan labels
        img_path = IMAGES_DIR / img_id
        if not img_path.exists():
            # Also check train_images subfolder (some xView zips have it)
            img_path = IMAGES_DIR / "train_images" / img_id
            if not img_path.exists():
                skipped_no_image += 1
                continue

        # We need image dimensions to normalize coordinates.
        # xView images are typically 500–3000 px; read actual size.
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_no_image += 1
                continue
            img_h, img_w = img.shape[:2]
        except Exception:
            skipped_no_image += 1
            continue

        # Build YOLO lines: class cx cy w h  (normalized 0-1)
        lines = []
        for cls_id, xmin, ymin, xmax, ymax in boxes:
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in [cx, cy, bw, bh]]
            if bw > 0 and bh > 0:
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if not lines:
            continue

        stem = Path(img_id).stem
        out_path = LABELS_OUT / f"{stem}.txt"
        with open(out_path, "w") as f:
            f.writelines(lines)
        written += 1

    print(f"\n✅ Written {written} label .txt files → {LABELS_OUT}")
    if skipped_no_image:
        print(f"   (Skipped {skipped_no_image} images not found on disk)")

    print("\n🎉 Done! Now re-run:")
    print('   python scripts/preprocess_local.py')
    print("   xView should now save images correctly.")


if __name__ == "__main__":
    print("=" * 55)
    print("xView GeoJSON → YOLO Label Converter")
    print("=" * 55)
    geojson_to_yolo()
