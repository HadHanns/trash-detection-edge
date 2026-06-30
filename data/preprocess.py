#!/usr/bin/env python3
"""
preprocess.py
=============
Konversi dataset TACO dari format COCO JSON ke format YOLO.

Format YOLO per baris:
    class_id  x_center  y_center  width  height
    (semua nilai koordinat dinormalisasi 0–1)

Langkah:
1. Baca anotasi COCO JSON
2. Map TACO categories → super-classes (7 kelas untuk saluran air)
3. Konversi bounding box COCO [x, y, w, h] → YOLO normalized format
4. Split dataset: 70% train / 20% val / 10% test (stratified)
5. Generate struktur direktori YOLO
6. Generate data.yaml

Penggunaan:
    python data/preprocess.py \\
        --annotations data/raw/annotations.json \\
        --images-dir data/raw/images \\
        --output-dir data/yolo \\
        --config config/config.yaml
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Default class mapping (TACO → water channel super-classes) ───────────────
# Mapping lengkap 60 kategori TACO → 7 super-class untuk saluran air.
# Key: nama kategori TACO (lowercase, exact atau partial match)
# Value: super-class index (0–6)
#
# Super-classes:
#   0 = plastic_bottle   1 = plastic_bag    2 = carton
#   3 = cup              4 = styrofoam      5 = cigarette
#   6 = other_trash
#
# Catatan: key lebih panjang dicocokkan lebih dulu (longest-key-first)
# sehingga "styrofoam cup" lebih prioritas dari "cup".

DEFAULT_CLASS_MAPPING = {
    # ── plastic_bottle (class 0) ──────────────────────────────────────────────
    "clear plastic bottle":       0,
    "plastic bottle cap":         0,   # tutup botol → grouped dengan botol
    "plastic bottle":             0,
    "glass bottle":               0,   # botol kaca → dimasukkan ke bottle group
    "drink can":                  0,   # kaleng minuman → near-bottle context
    "aerosol":                    0,

    # ── plastic_bag / wrapper (class 1) ──────────────────────────────────────
    "other plastic wrapper":      1,
    "plastic film":               1,
    "plastic gloves":             1,
    "plastic bag":                1,
    "wrapper":                    1,
    "crisp packet":               1,
    "sugar packet":               1,
    "spread tub":                 1,
    "toilet tube":                1,   # packaging / wrapper

    # ── carton / paper (class 2) ─────────────────────────────────────────────
    "corrugated carton":          2,
    "drink carton":               2,
    "other carton":               2,
    "meal carton":                2,
    "pizza box":                  2,
    "egg carton":                 2,
    "cardboard":                  2,
    "carton":                     2,
    "normal paper":               2,
    "paper bag":                  2,
    "paper cup":                  2,   # paper cup → carton category
    "magazine paper":             2,
    "wrapping paper":             2,
    "paper":                      2,

    # ── cup / disposable cup (class 3) ────────────────────────────────────────
    "disposable plastic cup":     3,
    "plastic cup":                3,
    "cup":                        3,

    # ── styrofoam (class 4) ───────────────────────────────────────────────────
    "styrofoam cup":              4,   # harus sebelum "cup"
    "styrofoam piece":            4,
    "foam cup":                   4,
    "styrofoam":                  4,

    # ── cigarette (class 5) ───────────────────────────────────────────────────
    "cigarette":                  5,
    "blunt":                      5,
    "cigar":                      5,

    # ── other_trash (class 6) ─────────────────────────────────────────────────
    "unlabeled litter":           6,
    "other plastic":              6,
    "other litter":               6,
    "plastic straw":              6,
    "plastic lid":                6,
    "plastic utensil":            6,
    "broken glass":               6,
    "glass":                      6,
    "pop tab":                    6,
    "scrap metal":                6,
    "metal bottle cap":           6,
    "rope":                       6,
    "straw":                      6,
    "can":                        6,
    "tin can":                    6,
    "food waste":                 6,
    "battery":                    6,
    "shoe":                       6,
    "squeezable tube":            6,
    "six pack rings":             6,
    "single-use carrier bag":     6,
    "unlabelled":                 6,
}

CLASS_NAMES = [
    "plastic_bottle",   # 0
    "plastic_bag",      # 1
    "carton",           # 2
    "cup",              # 3
    "styrofoam",        # 4
    "cigarette",        # 5
    "other_trash",      # 6
]


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def map_category(cat_name: str, class_mapping: dict) -> int:
    """
    Map TACO category name -> super-class index.

    Strategi matching (case-insensitive):
    1. Coba exact match (cat_lower == key)
    2. Coba key sebagai substring dari category name (key in cat_lower)

    Kunci lebih panjang dicocokkan lebih dulu sehingga
    "styrofoam cup" menang atas "cup".

    Returns 6 (other_trash) jika tidak ada mapping.
    """
    cat_lower = cat_name.lower().strip()

    # Sort by key length descending: specific keys matched first
    for key in sorted(class_mapping.keys(), key=len, reverse=True):
        if key == cat_lower:          # 1. exact match
            return class_mapping[key]
        if key in cat_lower:          # 2. key is substring of category name
            return class_mapping[key]
    return 6  # unknown -> other_trash


def coco_bbox_to_yolo(bbox: list, img_width: int, img_height: int) -> tuple:
    """
    Konversi bounding box dari format COCO ke format YOLO.

    COCO format : [x_min, y_min, width, height] (absolute pixels)
    YOLO format : [x_center, y_center, width, height] (normalized 0–1)
    """
    x_min, y_min, w, h = bbox

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    w = min(w, img_width - x_min)
    h = min(h, img_height - y_min)

    if w <= 0 or h <= 0:
        return None  # invalid bbox

    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    # Ensure all values in [0, 1]
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    width_norm = np.clip(width_norm, 0.0, 1.0)
    height_norm = np.clip(height_norm, 0.0, 1.0)

    return (x_center, y_center, width_norm, height_norm)


def get_image_dimensions(image_info: dict, images_dir: Path) -> tuple[int, int]:
    """
    Dapatkan dimensi gambar.
    Prioritas: dari metadata COCO, lalu baca file langsung jika perlu.
    """
    width = image_info.get("width", 0)
    height = image_info.get("height", 0)

    if width > 0 and height > 0:
        return width, height

    # Fallback: read from file
    file_name = image_info.get("file_name", "")
    img_path = images_dir / file_name
    if img_path.exists():
        try:
            with Image.open(img_path) as img:
                return img.size  # (width, height)
        except Exception:
            pass

    return 0, 0


def stratified_split(
    image_ids: list,
    image_to_classes: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple[list, list, list]:
    """
    Stratified train/val/test split berdasarkan distribusi class.
    
    Args:
        image_ids: List of image IDs
        image_to_classes: Dict mapping image_id → set of class_ids
        train_ratio, val_ratio: Split ratios (test = remainder)
        seed: Random seed

    Returns:
        (train_ids, val_ids, test_ids)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group images by primary class (most frequent class per image)
    class_to_images = defaultdict(list)
    for img_id in image_ids:
        classes = image_to_classes.get(img_id, set())
        if classes:
            # Assign to most frequent class (simple stratification)
            primary_class = min(classes)  # deterministic
            class_to_images[primary_class].append(img_id)
        else:
            class_to_images[-1].append(img_id)  # no annotation group

    train_ids, val_ids, test_ids = [], [], []

    for cls, ids in class_to_images.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    # Shuffle final lists
    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def write_yolo_label(
    label_path: Path,
    annotations: list
) -> None:
    """Write YOLO format label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for ann in annotations:
            class_id, x_c, y_c, w, h = ann
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def generate_data_yaml(
    output_dir: Path,
    class_names: list,
    train_path: str,
    val_path: str,
    test_path: str
) -> Path:
    """Generate data.yaml untuk Ultralytics YOLO."""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"✅ data.yaml generated: {yaml_path}")
    return yaml_path


def print_split_stats(
    split_ids: dict,
    image_to_classes: dict,
    class_names: list
) -> None:
    """Print statistik distribusi per split."""
    logger.info("\n📊 Dataset Split Statistics:")
    logger.info(f"{'Split':<10} {'Images':>8} {'Annotations':>12}")
    logger.info("-" * 35)

    for split_name, ids in split_ids.items():
        total_anns = sum(len(image_to_classes.get(img_id, [])) for img_id in ids)
        logger.info(f"{split_name:<10} {len(ids):>8} {total_anns:>12}")

    logger.info("\n📊 Class Distribution per Split:")
    header = f"{'Class':<20}" + "".join(f"{'  '+s:>10}" for s in split_ids.keys())
    logger.info(header)
    logger.info("-" * (20 + 10 * len(split_ids)))

    for cls_id, cls_name in enumerate(class_names):
        row = f"{cls_name:<20}"
        for ids in split_ids.values():
            count = sum(
                1 for img_id in ids
                if cls_id in image_to_classes.get(img_id, set())
            )
            row += f"{count:>10}"
        logger.info(row)


# ── Main Processing ──────────────────────────────────────────────────────────

def process_dataset(
    annotations_file: Path,
    images_dir: Path,
    output_dir: Path,
    class_mapping: dict,
    class_names: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_bbox_area: float = 100.0  # minimum bbox area in pixels to keep
) -> Path:
    """
    Main processing function: COCO → YOLO format conversion.
    
    Returns path to generated data.yaml.
    """
    logger.info(f"📂 Loading annotations: {annotations_file}")

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build lookup tables
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    img_id_to_info = {img["id"]: img for img in coco_data["images"]}

    # Build image → annotations mapping
    img_to_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    logger.info(f"Total images in annotations: {len(img_id_to_info)}")
    logger.info(f"Total annotations: {len(coco_data['annotations'])}")

    # ── Process annotations per image ────────────────────────────────────────
    processed_images = {}   # image_id → list of YOLO annotation tuples
    image_to_classes = {}   # image_id → set of class_ids (for stratification)
    skipped_images = 0
    skipped_annotations = 0
    unmapped_categories = Counter()

    logger.info("🔄 Processing annotations...")
    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Processing images"):
        file_name = img_info.get("file_name", "")
        img_path = images_dir / file_name

        # Skip if image file doesn't exist
        if not img_path.exists():
            skipped_images += 1
            continue

        # Get image dimensions
        width, height = get_image_dimensions(img_info, images_dir)
        if width == 0 or height == 0:
            logger.warning(f"Could not get dimensions for image {img_id}: {file_name}")
            skipped_images += 1
            continue

        # Process each annotation for this image
        yolo_anns = []
        classes_in_image = set()

        for ann in img_to_anns.get(img_id, []):
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            class_id = map_category(cat_name, class_mapping)

            if class_id == -1:
                unmapped_categories[cat_name] += 1
                skipped_annotations += 1
                continue

            bbox = ann.get("bbox", [])
            if not bbox or len(bbox) != 4:
                skipped_annotations += 1
                continue

            # Check minimum area
            if bbox[2] * bbox[3] < min_bbox_area:
                # Still keep small objects (this is a small object detection task!)
                # but log them
                pass

            yolo_bbox = coco_bbox_to_yolo(bbox, width, height)
            if yolo_bbox is None:
                skipped_annotations += 1
                continue

            yolo_anns.append((class_id, *yolo_bbox))
            classes_in_image.add(class_id)

        # Only include images with at least one valid annotation
        if yolo_anns:
            processed_images[img_id] = {
                "file_name": file_name,
                "annotations": yolo_anns
            }
            image_to_classes[img_id] = classes_in_image

    logger.info(
        f"\n📊 Processing Results:\n"
        f"   ✅ Images processed  : {len(processed_images)}\n"
        f"   ⏭️  Images skipped    : {skipped_images}\n"
        f"   ⏭️  Annotations skipped: {skipped_annotations}"
    )

    if unmapped_categories:
        logger.warning(f"\n⚠️  Unmapped categories (skipped):")
        for cat, count in unmapped_categories.most_common(10):
            logger.warning(f"   {cat}: {count} annotations")

    # ── Stratified split ─────────────────────────────────────────────────────
    all_image_ids = list(processed_images.keys())
    train_ids, val_ids, test_ids = stratified_split(
        all_image_ids, image_to_classes, train_ratio, val_ratio, seed
    )

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    print_split_stats(split_ids, image_to_classes, class_names)

    # ── Write YOLO dataset structure ─────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        imgs_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        imgs_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Writing {split_name} split ({len(ids)} images)...")

        for img_id in tqdm(ids, desc=f"Writing {split_name}", unit="img"):
            img_data = processed_images[img_id]
            file_name = img_data["file_name"]
            src_path = images_dir / file_name

            # Copy image
            img_filename = Path(file_name).name
            dst_img_path = imgs_out / img_filename
            if not dst_img_path.exists():
                shutil.copy2(src_path, dst_img_path)

            # Write label file
            label_filename = Path(img_filename).stem + ".txt"
            label_path = labels_out / label_filename
            write_yolo_label(label_path, img_data["annotations"])

    # ── Generate data.yaml ───────────────────────────────────────────────────
    yaml_path = generate_data_yaml(
        output_dir=output_dir,
        class_names=class_names,
        train_path="images/train",
        val_path="images/val",
        test_path="images/test"
    )

    logger.info(f"\n✅ Preprocessing complete!")
    logger.info(f"   Output dir : {output_dir}")
    logger.info(f"   data.yaml  : {yaml_path}")
    logger.info(f"   Next step  : python src/trainer.py --data {yaml_path}")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Konversi dataset TACO dari format COCO ke YOLO"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/annotations.json"),
        help="Path ke file anotasi COCO JSON"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Direktori gambar TACO"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo"),
        help="Direktori output format YOLO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path ke config.yaml"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    args = parser.parse_args()

    # Use config if available
    class_mapping = DEFAULT_CLASS_MAPPING.copy()
    class_names   = CLASS_NAMES
    if Path(args.config).exists():
        cfg = load_config(args.config)
        if "dataset" in cfg and "class_mapping" in cfg["dataset"]:
            # PENTING: lowercase semua keys agar cocok dengan map_category()
            # yang membandingkan dalam lowercase.
            # Merge dengan DEFAULT: config boleh override, tapi DEFAULT tetap jadi base.
            config_map = {
                str(k).lower().strip(): int(v)
                for k, v in cfg["dataset"]["class_mapping"].items()
            }
            class_mapping.update(config_map)   # DEFAULT + override dari config
        if "dataset" in cfg and "class_names" in cfg["dataset"]:
            class_names = list(cfg["dataset"]["class_names"].values())


    Path("logs").mkdir(exist_ok=True)

    process_dataset(
        annotations_file=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        class_mapping=class_mapping,
        class_names=class_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
