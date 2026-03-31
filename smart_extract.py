import zipfile
import os
import sys

# ============================================================
# PATHS — ZIPs are already in the project folders
# ============================================================
PROJECT_DATA = "/home/jay_gupta/Workspace/Border Surveillance Project/data"
RAW          = f"{PROJECT_DATA}/raw"

# ZIPs are already here — confirmed from your screenshots
UCF_ZIP_PATH   = f"{RAW}/ucf_crime/archive.zip"
XVIEW_IMG_PATH = f"{RAW}/xview/train_images.zip"
XVIEW_LBL_PATH = f"{RAW}/xview/train_labels.zip"

# Output folders
UCF_OUT    = f"{RAW}/ucf_crime/frames"
XVIEW_OUT  = f"{RAW}/xview/extracted"

# ============================================================
# CONFIG
# ============================================================
UCF_NEEDED_CLASSES = [
    "Train/Fighting/",
    "Train/Robbery/",
    "Train/Assault/",
    "Train/NormalVideos/",
    "Test/Fighting/",
    "Test/Robbery/",
    "Test/NormalVideos/"
]
UCF_MAX_FRAMES  = 500
XVIEW_MAX_IMGS  = 200

# ============================================================
# STEP 0: Verify ZIPs exist
# ============================================================
print("\n" + "="*55)
print("STEP 0: Verifying ZIP files...")
print("="*55)

def check(path, name):
    if not os.path.exists(path):
        print(f"  ❌ NOT FOUND: {path}")
        return False
    mb = os.path.getsize(path) / (1024*1024)
    print(f"  ✅ Found: {name} ({mb:.1f} MB)")
    return True

ucf_ok  = check(UCF_ZIP_PATH,   "archive.zip")
ximg_ok = check(XVIEW_IMG_PATH, "train_images.zip")
xlbl_ok = check(XVIEW_LBL_PATH, "train_labels.zip")

if not ucf_ok or not xlbl_ok:
    print("\n⚠️  Fix missing ZIPs then re-run.")
    sys.exit(1)

# ============================================================
# STEP 1: Peek inside UCF ZIP
# ============================================================
print("\n" + "="*55)
print("STEP 1: Peeking inside archive.zip structure...")
print("="*55)
with zipfile.ZipFile(UCF_ZIP_PATH, 'r') as z:
    all_files = z.namelist()
    top = set(f.split('/')[0] for f in all_files if '/' in f)
    print(f"  Total files: {len(all_files)}")
    print(f"  Top-level folders: {top}")
    print(f"  Sample entries:")
    for f in all_files[:10]:
        print(f"    {f}")

# ============================================================
# STEP 2: Extract UCF-Crime (smart)
# ============================================================
print("\n" + "="*55)
print("STEP 2: Extracting UCF-Crime frames...")
print("="*55)
os.makedirs(UCF_OUT, exist_ok=True)

with zipfile.ZipFile(UCF_ZIP_PATH, 'r') as z:
    all_files = z.namelist()
    total = 0
    for cls in UCF_NEEDED_CLASSES:
        # Match regardless of wrapper folder
        class_files = [
            f for f in all_files
            if cls in f and f.endswith('.png')
        ]
        selected = class_files[:UCF_MAX_FRAMES]
        label = cls.strip('/')
        print(f"  {label:35s} → {len(selected)} frames")
        for file in selected:
            z.extract(file, UCF_OUT)
            total += 1

print(f"\n  ✅ UCF-Crime done: {total} total frames")
print(f"  Location: {UCF_OUT}")

# ============================================================
# STEP 3: Extract xView Labels (all)
# ============================================================
print("\n" + "="*55)
print("STEP 3: Extracting xView Labels...")
print("="*55)
lbl_out = f"{XVIEW_OUT}/labels"
os.makedirs(lbl_out, exist_ok=True)

with zipfile.ZipFile(XVIEW_LBL_PATH, 'r') as z:
    z.extractall(lbl_out)
    print(f"  ✅ Labels done: {len(z.namelist())} files → {lbl_out}")

# ============================================================
# STEP 4: Extract xView Images (first 200 only)
# ============================================================
print("\n" + "="*55)
print(f"STEP 4: Extracting xView Images (first {XVIEW_MAX_IMGS})...")
print("="*55)

if ximg_ok:
    img_out = f"{XVIEW_OUT}/images"
    os.makedirs(img_out, exist_ok=True)
    with zipfile.ZipFile(XVIEW_IMG_PATH, 'r') as z:
        all_imgs = [
            f for f in z.namelist()
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))
        ]
        selected = all_imgs[:XVIEW_MAX_IMGS]
        print(f"  Total images in ZIP : {len(all_imgs)}")
        print(f"  Extracting          : {len(selected)}")
        for i, file in enumerate(selected):
            z.extract(file, img_out)
            if (i+1) % 50 == 0:
                print(f"  ... {i+1}/{len(selected)} done")
    print(f"  ✅ xView images done → {img_out}")
else:
    print("  ⏳ train_images.zip not found — skipping")

# ============================================================
# STEP 5: Final summary
# ============================================================
print("\n" + "="*55)
print("FINAL FOLDER SUMMARY")
print("="*55)
for root, dirs, files in os.walk(RAW):
    depth = root.replace(RAW, '').count(os.sep)
    if depth > 3:
        continue
    indent = '  ' * depth
    folder = os.path.basename(root)
    print(f"{indent}📁 {folder}/  [{len(files)} files]")

print("\n✅ All done! Data is ready for EDA.\n")
