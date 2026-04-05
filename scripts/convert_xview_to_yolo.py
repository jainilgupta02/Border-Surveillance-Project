import json
import os
from tqdm import tqdm
from PIL import Image

# ================== PATHS ==================
ROOT = "/home/jay_gupta/Workspace/Border Surveillance Project/data/raw/xview"

IMAGES_DIR = os.path.join(ROOT, "images/train")
LABELS_DIR = os.path.join(ROOT, "labels/train")
GEOJSON_PATH = os.path.join(ROOT, "train_labels/xView_train.geojson")

os.makedirs(LABELS_DIR, exist_ok=True)

# ================== CLASS MAP ==================
CLASS_MAP = {
    11: 0,  # Small Car
    12: 1,  # Truck
    13: 2,  # Bus
    15: 3,  # Van
    18: 4,  # Pickup
    20: 5,  # Cargo Truck
    21: 6,  # Trailer
    24: 7   # Person
}

# ================== LOAD GEOJSON ==================
print("📂 Loading GeoJSON...")
with open(GEOJSON_PATH) as f:
    data = json.load(f)

features = data.get("features", [])
print(f"✅ Total annotations: {len(features)}")

# ================== STATS ==================
written = 0
skipped_class = 0
missing_images = 0
invalid_boxes = 0

print("\n🚀 Converting to YOLO format...\n")

# ================== PROCESS ==================
for feature in tqdm(features):

    props = feature.get("properties", {})

    image_id = props.get("image_id")
    bounds = props.get("bounds_imcoords")
    type_id = props.get("type_id")

    if image_id is None or bounds is None or type_id is None:
        continue

    # ---- CLASS FILTER ----
    class_id = CLASS_MAP.get(type_id)
    if class_id is None:
        skipped_class += 1
        continue

    # ---- EMPTY BOUNDS ----
    if bounds == "":
        continue

    # ---- PARSE BBOX ----
    try:
        coords = list(map(float, bounds.split(",")))
        if len(coords) != 4:
            invalid_boxes += 1
            continue
    except:
        invalid_boxes += 1
        continue

    x1, y1, x2, y2 = coords

    # ---- HANDLE IMAGE NAME (FIXED) ----
    image_name = str(image_id)

    if image_name.endswith(".tif"):
        image_file = image_name
    else:
        try:
            image_file = f"{int(image_name)}.tif"
        except:
            continue

    image_path = os.path.join(IMAGES_DIR, image_file)

    if not os.path.exists(image_path):
        missing_images += 1
        continue

    # ---- GET IMAGE SIZE ----
    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except:
        missing_images += 1
        continue

    # ---- VALIDATE BOX ----
    if x2 <= x1 or y2 <= y1:
        invalid_boxes += 1
        continue

    # ---- YOLO FORMAT ----
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    # Clamp values
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    # ---- LABEL FILE NAME (FIXED) ----
    image_base = image_file.replace(".tif", "")
    label_file = os.path.join(LABELS_DIR, f"{image_base}.txt")

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    written += 1

# ================== SUMMARY ==================
print("\n✅ DONE!")
print("====================================")
print(f"✔ Labels written      : {written}")
print(f"⚠ Skipped classes     : {skipped_class}")
print(f"❌ Missing images      : {missing_images}")
print(f"❌ Invalid boxes       : {invalid_boxes}")
print("====================================")