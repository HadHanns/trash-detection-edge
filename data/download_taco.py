#!/usr/bin/env python3
"""
download_taco.py
================
Script untuk mendownload dataset TACO (Trash Annotations in Context).

Dataset TACO: https://github.com/pedropro/TACO
- Anotasi dalam format COCO JSON
- Gambar dihosting di Flickr (external URLs)

Penggunaan:
    python data/download_taco.py [--output-dir ./data/raw] [--max-images 1000]

Notes:
    - Beberapa gambar mungkin tidak tersedia (404 dari Flickr) — script
      menangani ini secara graceful dan mencatat gambar yang gagal.
    - Jika download gagal, script mencoba mirror alternatif dari
      repository resmi TACO di GitHub.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/download_taco.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
TACO_RAW_DIR = Path("data/raw")
TACO_IMAGES_DIR = TACO_RAW_DIR / "images"
TACO_ANNOTATIONS_FILE = TACO_RAW_DIR / "annotations.json"
FAILED_DOWNLOADS_FILE = TACO_RAW_DIR / "failed_downloads.txt"

REQUEST_TIMEOUT = 30       # seconds per image request
MAX_RETRIES = 3
RETRY_DELAY = 2            # seconds between retries
MAX_WORKERS = 4            # concurrent download threads


def download_file(url: str, dest_path: Path, timeout: int = REQUEST_TIMEOUT) -> bool:
    """
    Download single file dari URL ke dest_path.
    Returns True jika berhasil, False jika gagal.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif response.status_code == 404:
                logger.debug(f"[404] Image not found: {url}")
                return False
            else:
                logger.warning(
                    f"[Attempt {attempt}] HTTP {response.status_code} for {url}"
                )
        except requests.exceptions.Timeout:
            logger.warning(f"[Attempt {attempt}] Timeout: {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Attempt {attempt}] Error: {e} — {url}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return False


def download_annotations(output_dir: Path) -> Path:
    """
    Download file anotasi TACO (COCO JSON format).
    Returns path ke file anotasi yang berhasil didownload.
    """
    ann_path = output_dir / "annotations.json"
    if ann_path.exists():
        logger.info(f"Annotations already exist: {ann_path}")
        return ann_path

    logger.info("Downloading TACO annotations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    success = download_file(TACO_ANNOTATIONS_URL, ann_path)

    if not success:
        raise RuntimeError(
            f"Failed to download annotations from {TACO_ANNOTATIONS_URL}. "
            "Please check your internet connection or download manually."
        )

    logger.info(f"✅ Annotations downloaded: {ann_path}")
    return ann_path


def parse_annotations(ann_path: Path) -> dict:
    """
    Parse COCO JSON annotation file.
    Returns dict berisi images, annotations, categories.
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        f"📊 Dataset stats: "
        f"{len(data['images'])} images, "
        f"{len(data['annotations'])} annotations, "
        f"{len(data['categories'])} categories"
    )
    return data


def download_images(
    images_info: list,
    images_dir: Path,
    max_images: int = None,
    max_workers: int = MAX_WORKERS
) -> tuple[list, list]:
    """
    Download gambar TACO secara concurrent.
    
    Args:
        images_info: List of image dicts dari COCO annotations
        images_dir: Direktori tujuan penyimpanan gambar
        max_images: Batas maksimum gambar yang didownload (None = semua)
        max_workers: Jumlah thread concurrent

    Returns:
        (successful_images, failed_images) - list of image dicts
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    if max_images:
        images_info = images_info[:max_images]
        logger.info(f"Limited to first {max_images} images")

    successful = []
    failed = []
    already_exists = []

    def _download_single(img_info: dict):
        """Worker function for downloading a single image."""
        file_name = img_info.get("file_name", "")
        flickr_url = img_info.get("flickr_url", "")
        coco_url = img_info.get("coco_url", flickr_url)

        # Use file_name as local path (preserve directory structure)
        local_path = images_dir / file_name
        if local_path.exists():
            return ("exists", img_info)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Try flickr_url first, then coco_url
        for url in [flickr_url, coco_url]:
            if url and download_file(url, local_path):
                return ("success", img_info)

        return ("failed", img_info)

    logger.info(f"🔽 Downloading {len(images_info)} images with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single, img): img
            for img in images_info
        }

        with tqdm(total=len(futures), desc="Downloading images", unit="img") as pbar:
            for future in as_completed(futures):
                status, img_info = future.result()
                if status == "success":
                    successful.append(img_info)
                elif status == "exists":
                    already_exists.append(img_info)
                    successful.append(img_info)  # count as usable
                else:
                    failed.append(img_info)
                pbar.update(1)
                pbar.set_postfix({
                    "✅": len(successful),
                    "❌": len(failed)
                })

    # Save failed list for reference
    if failed:
        FAILED_DOWNLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_DOWNLOADS_FILE, "w") as f:
            for img in failed:
                f.write(f"{img.get('id')} | {img.get('flickr_url', 'N/A')}\n")
        logger.warning(
            f"⚠️  {len(failed)} images failed. See {FAILED_DOWNLOADS_FILE}"
        )

    logger.info(
        f"\n📊 Download Summary:\n"
        f"   ✅ Successful : {len(successful)}\n"
        f"   🔄 Pre-existing: {len(already_exists)}\n"
        f"   ❌ Failed     : {len(failed)}\n"
        f"   📁 Saved to  : {images_dir}"
    )

    return successful, failed


def save_filtered_annotations(
    original_data: dict,
    successful_images: list,
    output_path: Path
) -> None:
    """
    Simpan anotasi COCO yang sudah difilter hanya untuk gambar yang berhasil didownload.
    Ini memastikan preprocessing tidak mencoba memuat gambar yang tidak ada.
    """
    # Build set of available image IDs
    available_ids = {img["id"] for img in successful_images}

    # Filter images
    filtered_images = [
        img for img in original_data["images"]
        if img["id"] in available_ids
    ]

    # Filter annotations
    filtered_annotations = [
        ann for ann in original_data["annotations"]
        if ann["image_id"] in available_ids
    ]

    filtered_data = {
        "info": original_data.get("info", {}),
        "licenses": original_data.get("licenses", []),
        "categories": original_data["categories"],
        "images": filtered_images,
        "annotations": filtered_annotations
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"✅ Filtered annotations saved:\n"
        f"   Images     : {len(filtered_images)}\n"
        f"   Annotations: {len(filtered_annotations)}\n"
        f"   Path       : {output_path}"
    )


def print_class_distribution(data: dict) -> None:
    """Print distribusi class dari dataset."""
    from collections import Counter

    cat_id_to_name = {
        cat["id"]: cat["name"]
        for cat in data["categories"]
    }

    class_counts = Counter(
        cat_id_to_name[ann["category_id"]]
        for ann in data["annotations"]
        if ann["category_id"] in cat_id_to_name
    )

    logger.info("\n📊 Class Distribution (Top 20):")
    for cls_name, count in class_counts.most_common(20):
        bar = "█" * min(count // 10, 50)
        logger.info(f"   {cls_name:<35} {count:>5} {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Download TACO dataset untuk deteksi sampah di saluran air"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TACO_RAW_DIR,
        help="Direktori output untuk dataset (default: data/raw)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Batas maksimum gambar (default: semua)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Jumlah thread concurrent download (default: 4)"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip download gambar, hanya download anotasi"
    )
    args = parser.parse_args()

    # Setup output dirs
    raw_dir = args.output_dir
    images_dir = raw_dir / "images"
    Path("logs").mkdir(exist_ok=True)

    # Step 1: Download annotations
    ann_path = download_annotations(raw_dir)

    # Step 2: Parse annotations
    data = parse_annotations(ann_path)

    # Step 3: Print class distribution
    print_class_distribution(data)

    if args.skip_images:
        logger.info("⏭️  Skipping image download (--skip-images flag)")
        return

    # Step 4: Download images
    successful, failed = download_images(
        images_info=data["images"],
        images_dir=images_dir,
        max_images=args.max_images,
        max_workers=args.max_workers
    )

    # Step 5: Save filtered annotations (only successfully downloaded images)
    if failed:
        filtered_ann_path = raw_dir / "annotations_filtered.json"
        save_filtered_annotations(data, successful, filtered_ann_path)
        logger.info(f"ℹ️  Use {filtered_ann_path} for preprocessing (filtered for available images)")
    else:
        logger.info(f"ℹ️  All images downloaded. Use {ann_path} for preprocessing.")

    logger.info("\n✅ TACO download complete!")
    logger.info(f"   Next step: python data/preprocess.py --annotations {raw_dir}/annotations.json")


if __name__ == "__main__":
    main()
