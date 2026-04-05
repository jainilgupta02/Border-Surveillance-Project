import cv2
import random
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
RAW_DIR = BASE_DIR / "data/raw/xview"
OUT_DIR = BASE_DIR / "data/processed"

IMG_SIZE = 640

# Split ratios (we will re-split globally)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# xView → YOUR 7 CLASSES
CLASS_MAP = {
    # person
    0: 0, 1: 0,

    # vehicle
    2: 1, 3: 1, 4: 1, 5: 1, 6: 1,

    # aircraft
    7: 4,

    # ship
    8: 5,
}

# ==========================================

def get_all_images(images_dir):
    images = []

    for split in ["train", "val"]:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue

        images.extend(
            list(split_dir.glob("*.[jJ][pP][gG]")) +
            list(split_dir.glob("*.[pP][nN][gG]")) +
            list(split_dir.glob("*.[tT][iI][fF]")) +
            list(split_dir.glob("*.[tT][iI][fF][fF]"))
        )

    return sorted(images)


def create_output_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_bbox(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    try:
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except:
        return None

    if cls not in CLASS_MAP:
        return None

    cls = CLASS_MAP[cls]
    return f"{cls} {x} {y} {w} {h}"


def main():
    print("\n── xView Processing ─────────────────────────────")

    images_dir = RAW_DIR / "images"
    labels_dir = RAW_DIR / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ images/labels folder not found in xview")
        return

    images = get_all_images(images_dir)

    if len(images) == 0:
        print("❌ No images found. Check structure/extensions.")
        return

    print(f"✅ Found {len(images)} images")

    # Shuffle and split
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    total_written = 0
    total_skipped = 0

    for split, imgs in splits.items():
        print(f"\n📦 Processing {split}: {len(imgs)} images")

        for img_path in tqdm(imgs):
            # detect original split (train/val)
            original_split = img_path.parent.name

            label_path = labels_dir / original_split / (img_path.stem + ".txt")

            if not label_path.exists():
                total_skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                total_skipped += 1
                continue

            # resize
            resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # unique name to avoid collision
            new_name = f"xview_{img_path.stem}.jpg"

            out_img = OUT_DIR / split / "images" / new_name
            out_lbl = OUT_DIR / split / "labels" / new_name.replace(".jpg", ".txt")

            # save image
            cv2.imwrite(str(out_img), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # process labels
            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    converted = convert_bbox(line)
                    if converted:
                        new_lines.append(converted)

            if len(new_lines) == 0:
                total_skipped += 1
                continue

            with open(out_lbl, "w") as f:
                f.write("\n".join(new_lines))

            total_written += 1

    print("\n==============================")
    print("✅ xView PROCESSING COMPLETE")
    print(f"✔ Images written : {total_written}")
    print(f"⚠ Skipped       : {total_skipped}")
    print("==============================\n")


if __name__ == "__main__":
    create_output_dirs()
    main()