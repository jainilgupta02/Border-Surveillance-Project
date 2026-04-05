"""
Fix for preprocess_local.py — xView double-remapping bug
=========================================================
The xview_geojson_to_yolo.py converter already wrote labels with our
class IDs (0-6). preprocess_local.py was then trying to remap those
through XVIEW_CLASS_MAP again (which expects original xView IDs like
11, 17, 18...), causing every annotation to be filtered as "irrelevant".

This patch replaces process_xview() with a version that reads the
already-converted labels directly, no remapping needed.

Run from your project root:
    cd "/home/jay_gupta/Workspace/Border Surveillance Project"
    python fix_xview_patch.py
"""

import re
from pathlib import Path

TARGET = Path("/home/jay_gupta/Workspace/Border Surveillance Project/scripts/preprocess_local.py")

# Also check root if scripts/ copy doesn't exist
if not TARGET.exists():
    TARGET = Path("/home/jay_gupta/Workspace/Border Surveillance Project/preprocess_local.py")

if not TARGET.exists():
    print("❌ Could not find preprocess_local.py")
    print("   Make sure you're in the project root and the file exists.")
    exit(1)

OLD = '''def process_xview():
    """
    xView provides satellite images + YOLO-format label files.
    We remap xView class IDs to our 7-class schema.
    Expected structure inside extracted/:
        images/  (or .tif/.jpg directly)
        labels/  (or same folder as images with .txt files)
    """
    print("\\n📁 Processing xView...")

    if not XVIEW_DIR.exists():
        print(f"  ⚠️  xView folder not found: {XVIEW_DIR}")
        return 0

    # Find images — xView can have .tif, .jpg, .png
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

            # xView labels can be alongside images or in a labels/ sibling folder
            label_candidates = [
                img_path.parent / (img_path.stem + ".txt"),
                img_path.parent.parent / "labels" / (img_path.stem + ".txt"),
                XVIEW_DIR / "labels" / (img_path.stem + ".txt"),
            ]
            label_src = next((p for p in label_candidates if p.exists()), None)

            if label_src is None:
                continue  # skip unlabelled images

            # Read and remap labels
            new_lines = []
            with open(label_src) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    orig_cls  = int(parts[0])
                    mapped_cls = XVIEW_CLASS_MAP.get(orig_cls, -1)
                    if mapped_cls == -1:
                        continue  # ignore irrelevant classes
                    new_lines.append(f"{mapped_cls} {' '.join(parts[1:5])}\\n")

            if not new_lines:
                continue  # nothing useful in this image\'s labels

            # Safe filename (avoid .tif → .jpg issues with duplicates)
            safe_name = img_path.stem + ".jpg"
            dst_img   = DATA_OUT / split_name / "images" / safe_name

            if resize_and_save(img_path, dst_img):
                lbl = DATA_OUT / split_name / "labels" / (img_path.stem + ".txt")
                with open(lbl, "w") as f:
                    f.writelines(new_lines)
                saved += 1

    print(f"  ✅ xView: {saved} images saved")
    return saved'''

NEW = '''def process_xview():
    """
    xView satellite images — labels already converted to YOLO format
    by xview_geojson_to_yolo.py (class IDs already in our 0-6 schema).
    Read labels directly without remapping.
    """
    print("\\n📁 Processing xView...")

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
                    new_lines.append(f"{cls_id} {' '.join(parts[1:5])}\\n")

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
    return saved'''


content = TARGET.read_text()

if "Labels are ALREADY in our 0-6 class schema" in content:
    print("✅ Patch already applied — nothing to do.")
    print("   Just re-run: python scripts/preprocess_local.py")
elif OLD.strip() not in content:
    print("⚠️  Could not find the exact function text to replace.")
    print("   The file may have been modified already.")
    print("   Try applying the fix manually — see the NEW function above.")
else:
    patched = content.replace(OLD.strip(), NEW.strip())
    TARGET.write_text(patched)
    print(f"✅ Patch applied to {TARGET}")
    print()
    print("Now re-run preprocessing:")
    print('   python scripts/preprocess_local.py')
    print()
    print("xView should now save ~90-128 images correctly.")
