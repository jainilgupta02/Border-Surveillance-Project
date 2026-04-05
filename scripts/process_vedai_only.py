import cv2
import random
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR = Path("/home/jay_gupta/Workspace/Border Surveillance Project")
RAW_DIR = BASE_DIR / "data/raw/vedai"
OUT_DIR = BASE_DIR / "data/processed"

IMG_SIZE = 640

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

VEHICLE_CLASS_ID = 1

# ==========================================

def create_output_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def find_dirs():
    images_dir = None
    labels_dir = None

    for p in RAW_DIR.iterdir():
        name = p.name.lower()

        if "vehic" in name:
            images_dir = p
        elif "annotation" in name:
            labels_dir = p

    if not images_dir or not labels_dir:
        print("❌ Folder detection failed")
        return None, None

    print(f"✅ Images dir: {images_dir}")
    print(f"✅ Labels dir: {labels_dir}")

    return images_dir, labels_dir


def get_label_file(img_path, labels_dir):
    # extract base id (remove _co / _ir)
    base_id = img_path.stem.split("_")[0]
    return labels_dir / f"{base_id}.txt"


def parse_annotation(line):
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except:
        return None

    return f"{VEHICLE_CLASS_ID} {x} {y} {w} {h}"


def main():
    print("\n── VEDAI Processing (FINAL FIX) ─────────────────")

    images_dir, labels_dir = find_dirs()
    if images_dir is None:
        return

    images = list(images_dir.glob("*.png"))
    print(f"✅ Found {len(images)} images")

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
            label_path = get_label_file(img_path, labels_dir)

            if not label_path.exists():
                total_skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                total_skipped += 1
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            new_name = f"vedai_{img_path.stem}.jpg"

            out_img = OUT_DIR / split / "images" / new_name
            out_lbl = OUT_DIR / split / "labels" / new_name.replace(".jpg", ".txt")

            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parsed = parse_annotation(line)
                    if parsed:
                        new_lines.append(parsed)

            if len(new_lines) == 0:
                total_skipped += 1
                continue

            with open(out_lbl, "w") as f:
                f.write("\n".join(new_lines))

            total_written += 1

    print("\n==============================")
    print("✅ VEDAI PROCESSING COMPLETE")
    print(f"✔ Images written : {total_written}")
    print(f"⚠ Skipped       : {total_skipped}")
    print("==============================\n")


if __name__ == "__main__":
    create_output_dirs()
    main()