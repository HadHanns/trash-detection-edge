#!/usr/bin/env python3
"""
preprocess_raw2.py
==================
Konversi dataset raw_2 (COCO JSON dengan 7 kategori khusus) ke format YOLO.

Kategori raw_2:
  id=1  bottle   → 0 (plastic_bottle)
  id=2  o_bottle → 0 (plastic_bottle)
  id=3  food_pack→ 2 (carton)
  id=4  bag      → 1 (plastic_bag)
  id=5  o_bag    → 1 (plastic_bag)
  id=6  o_pla    → 6 (other_trash)
  id=7  non_pla  → 6 (other_trash)

Output: data/yolo_2/ dengan struktur images/train, images/val, images/test
        dan file labels/ serta data.yaml

Penggunaan:
    python data/preprocess_raw2.py
    python data/preprocess_raw2.py --train-ratio 0.7 --val-ratio 0.2 --seed 42
"""

import os
import sys
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Tambahkan path root agar bisa import dari data/ ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocess_raw2.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Konfigurasi path ─────────────────────────────────────────────────────────
DEFAULT_ANNOTATIONS = ROOT / "data" / "raw_2" / "annotations_7cat.json"
DEFAULT_IMAGES_DIR  = ROOT / "data" / "raw_2" / "images"
DEFAULT_OUTPUT_DIR  = ROOT / "data" / "yolo_2"

# ── Mapping: category name (lowercase) → super-class index ───────────────────
# Sesuai kesepakatan:
#   bottle    → 0 (plastic_bottle)
#   o_bottle  → 0 (plastic_bottle)
#   food_pack → 2 (carton)
#   bag       → 1 (plastic_bag)
#   o_bag     → 1 (plastic_bag)
#   o_pla     → 6 (other_trash)
#   non_pla   → 6 (other_trash)

CAT_NAME_TO_CLASS = {
    "bottle":    0,
    "o_bottle":  0,
    "food_pack": 2,
    "bag":       1,
    "o_bag":     1,
    "o_pla":     6,
    "non_pla":   6,
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

# ── Helper Functions ──────────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> Optional[tuple]:
    """COCO [x_min, y_min, w, h] → YOLO normalized [x_c, y_c, w, h]."""
    x_min, y_min, w, h = bbox

    x_min = max(0.0, float(x_min))
    y_min = max(0.0, float(y_min))
    w     = min(float(w), img_w - x_min)
    h     = min(float(h), img_h - y_min)

    if w <= 0 or h <= 0:
        return None

    x_c = np.clip((x_min + w / 2) / img_w, 0.0, 1.0)
    y_c = np.clip((y_min + h / 2) / img_h, 0.0, 1.0)
    wn  = np.clip(w / img_w,  0.0, 1.0)
    hn  = np.clip(h / img_h, 0.0, 1.0)

    return (x_c, y_c, wn, hn)


def get_image_dimensions(img_info: dict, images_dir: Path) -> tuple[int, int]:
    """Dapatkan dimensi dari metadata atau baca file."""
    w = img_info.get("width", 0)
    h = img_info.get("height", 0)
    if w > 0 and h > 0:
        return int(w), int(h)

    file_name = img_info.get("file_name", "")
    img_path  = images_dir / file_name
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
    val_ratio:   float = 0.2,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """Stratified split berdasarkan kelas dominan tiap gambar."""
    random.seed(seed)
    np.random.seed(seed)

    class_to_images: dict[int, list] = defaultdict(list)
    for img_id in image_ids:
        classes = image_to_classes.get(img_id, set())
        if classes:
            primary = min(classes)
            class_to_images[primary].append(img_id)
        else:
            class_to_images[-1].append(img_id)

    train_ids, val_ids, test_ids = [], [], []
    for cls, ids in class_to_images.items():
        random.shuffle(ids)
        n       = len(ids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def write_label(label_path: Path, annotations: list) -> None:
    """Tulis file label YOLO."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls_id, x_c, y_c, wn, hn in annotations:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}\n")


def generate_data_yaml(output_dir: Path, class_names: list) -> Path:
    """Buat data.yaml untuk Ultralytics YOLO."""
    import yaml

    data = {
        "path":  str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"✅ data.yaml → {yaml_path}")
    return yaml_path


# ── Main Processing ───────────────────────────────────────────────────────────

def process(
    annotations_file: Path,
    images_dir:        Path,
    output_dir:        Path,
    train_ratio:       float = 0.7,
    val_ratio:         float = 0.2,
    seed:              int   = 42,
) -> Path:

    logger.info(f"📂 Annotations : {annotations_file}")
    logger.info(f"📂 Images dir  : {images_dir}")
    logger.info(f"📂 Output dir  : {output_dir}")

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookups
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    logger.info(f"Total images in JSON  : {len(img_id_to_info)}")
    logger.info(f"Total annotations     : {len(coco['annotations'])}")
    logger.info(f"Categories in JSON    : {list(cat_id_to_name.values())}")

    # Map category_id → class_index
    cat_id_to_cls: dict[int, int] = {}
    for cat_id, cat_name in cat_id_to_name.items():
        cls_idx = CAT_NAME_TO_CLASS.get(cat_name.lower().strip(), 6)  # default → other_trash
        cat_id_to_cls[cat_id] = cls_idx
        logger.info(f"  Category mapping: {cat_name!r} (id={cat_id}) → class {cls_idx} ({CLASS_NAMES[cls_idx]})")

    # Group annotations by image
    img_to_anns: dict = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    # ── Process per image ────────────────────────────────────────────────────
    processed: dict  = {}       # img_id → {"file_name": ..., "annotations": [...]}
    img_to_classes:  dict = {}  # img_id → set of class_ids
    skipped_imgs = 0
    skipped_anns = 0

    logger.info("🔄 Processing annotations...")
    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Images"):
        file_name = img_info.get("file_name", "")
        img_path  = images_dir / file_name

        if not img_path.exists():
            skipped_imgs += 1
            continue

        w, h = get_image_dimensions(img_info, images_dir)
        if w == 0 or h == 0:
            logger.warning(f"Dimensi tidak diketahui: {file_name}")
            skipped_imgs += 1
            continue

        yolo_anns    = []
        classes_seen = set()

        for ann in img_to_anns.get(img_id, []):
            cls_idx = cat_id_to_cls.get(ann["category_id"], 6)
            bbox    = ann.get("bbox", [])
            if not bbox or len(bbox) != 4:
                skipped_anns += 1
                continue

            yolo_bbox = coco_bbox_to_yolo(bbox, w, h)
            if yolo_bbox is None:
                skipped_anns += 1
                continue

            yolo_anns.append((cls_idx, *yolo_bbox))
            classes_seen.add(cls_idx)

        if yolo_anns:
            processed[img_id]       = {"file_name": file_name, "annotations": yolo_anns}
            img_to_classes[img_id]  = classes_seen

    logger.info(f"✅ Images processed  : {len(processed)}")
    logger.info(f"⏭️  Images skipped    : {skipped_imgs}")
    logger.info(f"⏭️  Annotations skipped: {skipped_anns}")

    # ── Split ────────────────────────────────────────────────────────────────
    all_ids = list(processed.keys())
    train_ids, val_ids, test_ids = stratified_split(
        all_ids, img_to_classes, train_ratio, val_ratio, seed
    )
    logger.info(f"📊 Split — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # ── Write YOLO structure ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split_name, ids in splits.items():
        imgs_out   = output_dir / "images"  / split_name
        labels_out = output_dir / "labels"  / split_name
        imgs_out.mkdir(parents=True,   exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Writing {split_name} ({len(ids)} images)...")
        for img_id in tqdm(ids, desc=f"  {split_name}", unit="img"):
            data      = processed[img_id]
            file_name = data["file_name"]
            src_path  = images_dir / file_name

            img_fn    = Path(file_name).name
            dst_img   = imgs_out / img_fn
            if not dst_img.exists():
                shutil.copy2(src_path, dst_img)

            lbl_fn  = Path(img_fn).stem + ".txt"
            write_label(labels_out / lbl_fn, data["annotations"])

    # ── data.yaml ────────────────────────────────────────────────────────────
    yaml_path = generate_data_yaml(output_dir, CLASS_NAMES)

    logger.info("\n✅ Preprocessing raw_2 selesai!")
    logger.info(f"   Output : {output_dir}")
    logger.info(f"   YAML   : {yaml_path}")
    logger.info(f"\n▶  Langkah selanjutnya: jalankan train_final.py")

    return yaml_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw_2 dataset (COCO 7-cat) → YOLO format"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-dir",  type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    process(
        annotations_file=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
