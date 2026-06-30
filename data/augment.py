#!/usr/bin/env python3
"""
augment.py
==========
Augmentasi khusus domain saluran air untuk meningkatkan generalisasi model.

Strategi:
1. **Alpha Blending** — overlay objek sampah dari TACO ke gambar latar air
2. **Domain-specific transforms** — simulasi kondisi saluran air:
   - Refleksi cahaya (brightness/contrast variation)
   - Blur (kualitas kamera tepi sungai)
   - Gaussian noise
   - Warna air (color shift ke arah biru-coklat)
3. **Mosaic augmentation** — 4-image mosaic untuk variasi konteks

Penggunaan:
    python data/augment.py \\
        --source-dir data/yolo \\
        --background-dir data/backgrounds \\
        --output-dir data/yolo_augmented \\
        --num-synthetic 200
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
WATER_CHANNEL_COLORS = [
    # (mean_hue, std) in HSV — typical water channel color ranges
    (105, 15),   # greenish water
    (30, 20),    # murky brown water
    (95, 10),    # clean blue water
    (45, 25),    # muddy water
]

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Image I/O ────────────────────────────────────────────────────────────────

def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Could not load image: {path}")
    return img


def load_yolo_labels(label_path: Path) -> list:
    """
    Load YOLO format labels.
    Returns list of [class_id, x_center, y_center, width, height].
    """
    if not label_path.exists():
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


def save_yolo_labels(labels: list, label_path: Path) -> None:
    """Save YOLO format labels."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


# ── Water Channel Background Generation ─────────────────────────────────────

def generate_synthetic_water_background(
    width: int = 640,
    height: int = 640,
    style: str = "random"
) -> np.ndarray:
    """
    Generate synthetic water channel background image.
    
    Styles: 'random', 'clean', 'murky', 'muddy', 'green'
    Returns BGR numpy array.
    """
    styles = {
        "clean":  {"color": (200, 160, 80),  "noise_std": 8},   # Blue water
        "murky":  {"color": (100, 130, 120), "noise_std": 15},  # Murky green
        "muddy":  {"color": (60,  100, 140), "noise_std": 20},  # Brown muddy
        "green":  {"color": (120, 160, 60),  "noise_std": 10},  # Green algae
    }
    if style == "random":
        style = random.choice(list(styles.keys()))

    cfg = styles[style]
    base_color = np.array(cfg["color"], dtype=np.float32)

    # Base water texture
    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:] = base_color

    # Add horizontal ripple effect (simulating water surface)
    ripple_freq = random.uniform(0.02, 0.08)
    ripple_amp = random.uniform(5, 20)
    for y in range(height):
        ripple = ripple_amp * np.sin(2 * np.pi * ripple_freq * y + random.uniform(0, np.pi))
        bg[y] = np.clip(bg[y] + ripple, 0, 255)

    # Add Gaussian noise
    noise = np.random.normal(0, cfg["noise_std"], bg.shape)
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    # Slight blur for soft water look
    bg = cv2.GaussianBlur(bg, (5, 5), 1.0)

    # Random vignette (darker edges - simulating camera)
    if random.random() < 0.5:
        bg = apply_vignette(bg)

    return bg  # BGR


def apply_vignette(img: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Apply vignette effect (darken edges)."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    mask = 1 - strength * (dist / max_dist)
    mask = np.clip(mask, 0, 1)
    result = (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    return result


# ── Domain Augmentation Transforms ───────────────────────────────────────────

class WaterChannelAugmentor:
    """
    Augmentor khusus domain saluran air untuk meningkatkan robustness model
    terhadap kondisi nyata saluran air perkotaan.
    """

    def __init__(
        self,
        alpha_range: tuple = (0.6, 1.0),
        noise_std: float = 10.0,
        blur_prob: float = 0.3,
        blur_kernel: int = 3,
        brightness_range: tuple = (0.6, 1.4),
        contrast_range: tuple = (0.7, 1.3),
        color_shift_prob: float = 0.4,
    ):
        self.alpha_range = alpha_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.color_shift_prob = color_shift_prob

    def apply_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Random brightness and contrast adjustment."""
        alpha = random.uniform(*self.contrast_range)   # contrast
        beta = random.uniform(-30, 30)                 # brightness
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def apply_water_color_shift(self, img: np.ndarray) -> np.ndarray:
        """
        Simulasi warna air — shift hue ke range biru-coklat-hijau
        yang umum pada saluran air perkotaan.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Hue shift ±15 degrees
        hue_shift = random.uniform(-15, 15)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        # Saturation adjustment
        sat_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (simulate low-quality edge camera)."""
        noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Random blur (motion blur or Gaussian)."""
        if random.random() < 0.5:
            # Gaussian blur
            k = random.choice([3, 5])
            return cv2.GaussianBlur(img, (k, k), 0)
        else:
            # Motion blur (horizontal or vertical)
            k = random.choice([5, 7, 9])
            kernel = np.zeros((k, k))
            if random.random() < 0.5:
                kernel[k // 2, :] = 1.0 / k  # horizontal
            else:
                kernel[:, k // 2] = 1.0 / k  # vertical
            return cv2.filter2D(img, -1, kernel)

    def apply_reflection(self, img: np.ndarray) -> np.ndarray:
        """Simulate water reflection (light flares on water surface)."""
        h, w = img.shape[:2]
        overlay = img.copy().astype(np.float32)
        n_reflections = random.randint(1, 3)
        for _ in range(n_reflections):
            x = random.randint(0, w - 1)
            y = random.randint(0, h // 2)  # reflections on upper half
            radius = random.randint(20, 80)
            intensity = random.uniform(30, 100)
            cv2.circle(overlay, (x, y), radius, (intensity, intensity, intensity), -1)
        alpha = random.uniform(0.2, 0.5)
        result = cv2.addWeighted(img.astype(np.float32), 1 - alpha, overlay, alpha, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def augment(self, img: np.ndarray, apply_all: bool = False) -> np.ndarray:
        """
        Apply full water channel augmentation pipeline.
        
        Args:
            img: BGR image
            apply_all: If True, apply all transforms. Otherwise random.
        """
        # Always apply brightness/contrast
        img = self.apply_brightness_contrast(img)

        # Optional transforms
        if apply_all or random.random() < self.blur_prob:
            img = self.apply_blur(img)
        if apply_all or random.random() < 0.5:
            img = self.apply_noise(img)
        if apply_all or random.random() < self.color_shift_prob:
            img = self.apply_water_color_shift(img)
        if apply_all or random.random() < 0.3:
            img = self.apply_reflection(img)

        return img


# ── Object Overlay (Alpha Blending) ─────────────────────────────────────────

def extract_object_from_bbox(
    img: np.ndarray,
    label: list,
    padding: float = 0.05
) -> tuple[Optional[np.ndarray], tuple]:
    """
    Crop objek dari gambar berdasarkan YOLO bbox label.
    
    Returns (cropped_image, original_bbox_pixels) or (None, None).
    """
    class_id, x_c, y_c, w, h = label
    H, W = img.shape[:2]

    # Convert normalized to pixel coordinates
    x1 = int((x_c - w / 2 - padding) * W)
    y1 = int((y_c - h / 2 - padding) * H)
    x2 = int((x_c + w / 2 + padding) * W)
    y2 = int((y_c + h / 2 + padding) * H)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def overlay_object_on_background(
    background: np.ndarray,
    obj_crop: np.ndarray,
    position: tuple,
    alpha: float = 0.85,
    scale: float = 1.0
) -> tuple[np.ndarray, tuple]:
    """
    Overlay objek sampah di atas background menggunakan alpha blending.
    
    Args:
        background: Background BGR image
        obj_crop: Object crop BGR image
        position: (x, y) top-left position on background
        alpha: Opacity (0=transparent, 1=opaque)
        scale: Scale factor for object size

    Returns:
        (composited_image, yolo_label_tuple) or (background, None)
    """
    bg = background.copy()
    bg_h, bg_w = bg.shape[:2]

    # Resize object
    obj_h, obj_w = obj_crop.shape[:2]
    new_w = int(obj_w * scale)
    new_h = int(obj_h * scale)
    if new_w < 10 or new_h < 10:
        return bg, None

    obj_resized = cv2.resize(obj_crop, (new_w, new_h))

    x, y = position
    x = max(0, min(x, bg_w - new_w))
    y = max(0, min(y, bg_h - new_h))

    x2, y2 = min(x + new_w, bg_w), min(y + new_h, bg_h)
    actual_w = x2 - x
    actual_h = y2 - y

    if actual_w <= 0 or actual_h <= 0:
        return bg, None

    # Alpha blend
    roi = bg[y:y2, x:x2].astype(np.float32)
    obj_patch = obj_resized[:actual_h, :actual_w].astype(np.float32)
    blended = roi * (1 - alpha) + obj_patch * alpha
    bg[y:y2, x:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    # YOLO label for placed object
    x_center = (x + actual_w / 2) / bg_w
    y_center = (y + actual_h / 2) / bg_h
    width_norm = actual_w / bg_w
    height_norm = actual_h / bg_h

    return bg, (x_center, y_center, width_norm, height_norm)


# ── Mosaic Augmentation ──────────────────────────────────────────────────────

def create_mosaic(
    images: list,        # list of np.ndarray (BGR)
    labels: list,        # list of list[list] (YOLO labels per image)
    output_size: int = 640
) -> tuple[np.ndarray, list]:
    """
    4-image mosaic augmentation.
    Combines 4 images into a single mosaic with adjusted labels.
    
    Args:
        images: List of 4 BGR images
        labels: List of 4 label lists
        output_size: Output image size (square)

    Returns:
        (mosaic_image, combined_labels)
    """
    assert len(images) == 4 and len(labels) == 4, "Need exactly 4 images for mosaic"

    s = output_size
    half = s // 2
    mosaic = np.zeros((s, s, 3), dtype=np.uint8)
    combined_labels = []

    positions = [(0, 0), (half, 0), (0, half), (half, half)]  # (x_offset, y_offset)

    for idx, (img, lbls, (x_off, y_off)) in enumerate(zip(images, labels, positions)):
        h, w = img.shape[:2]
        # Resize to half size
        img_resized = cv2.resize(img, (half, half))
        mosaic[y_off:y_off + half, x_off:x_off + half] = img_resized

        # Adjust labels
        for lbl in lbls:
            class_id, x_c, y_c, w_n, h_n = lbl
            # Scale to mosaic coordinates
            new_x_c = (x_c * half + x_off) / s
            new_y_c = (y_c * half + y_off) / s
            new_w = w_n * half / s
            new_h = h_n * half / s
            combined_labels.append([class_id, new_x_c, new_y_c, new_w, new_h])

    return mosaic, combined_labels


# ── Synthetic Image Generation ────────────────────────────────────────────────

def generate_synthetic_water_images(
    source_dir: Path,
    output_dir: Path,
    num_images: int = 200,
    background_dir: Optional[Path] = None,
    augmentor: Optional[WaterChannelAugmentor] = None,
    alpha_range: tuple = (0.6, 1.0),
    image_size: int = 640
) -> None:
    """
    Generate gambar sintetis: overlay objek sampah dari dataset TACO
    ke background saluran air.

    Args:
        source_dir: YOLO dataset dir dengan images/ dan labels/
        output_dir: Output dir untuk gambar sintetis
        num_images: Jumlah gambar sintetis yang dibuat
        background_dir: Dir gambar background saluran air nyata (opsional)
        augmentor: WaterChannelAugmentor instance
        alpha_range: Rentang alpha untuk blending
        image_size: Ukuran output image
    """
    if augmentor is None:
        augmentor = WaterChannelAugmentor()

    # Collect source images and labels
    train_img_dir = source_dir / "images" / "train"
    train_lbl_dir = source_dir / "labels" / "train"

    if not train_img_dir.exists():
        logger.error(f"Source train dir not found: {train_img_dir}")
        return

    source_images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    if not source_images:
        logger.warning(f"No images found in {train_img_dir}")
        return

    # Collect backgrounds
    backgrounds = []
    if background_dir and background_dir.exists():
        backgrounds = list(background_dir.rglob("*.jpg")) + list(background_dir.rglob("*.png"))
        logger.info(f"Found {len(backgrounds)} background images")

    # Output dirs
    out_img_dir = output_dir / "images" / "train"
    out_lbl_dir = output_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎨 Generating {num_images} synthetic water channel images...")

    generated = 0
    attempts = 0
    max_attempts = num_images * 5

    with tqdm(total=num_images, desc="Generating synthetic images") as pbar:
        while generated < num_images and attempts < max_attempts:
            attempts += 1

            # Select random source image
            src_path = random.choice(source_images)
            lbl_path = train_lbl_dir / (src_path.stem + ".txt")

            src_img = load_image(src_path)
            if src_img is None:
                continue

            src_labels = load_yolo_labels(lbl_path)
            if not src_labels:
                continue

            # Select random object to overlay
            label = random.choice(src_labels)
            obj_crop, _ = extract_object_from_bbox(src_img, label)
            if obj_crop is None or obj_crop.size == 0:
                continue

            # Generate or load background
            if backgrounds and random.random() < 0.6:
                bg_path = random.choice(backgrounds)
                bg = load_image(bg_path)
                if bg is None:
                    bg = generate_synthetic_water_background(image_size, image_size)
                else:
                    bg = cv2.resize(bg, (image_size, image_size))
            else:
                style = random.choice(["clean", "murky", "muddy", "green"])
                bg = generate_synthetic_water_background(image_size, image_size, style)

            # Apply augmentation to background
            bg = augmentor.augment(bg)

            # Random placement
            bg_h, bg_w = bg.shape[:2]
            obj_h, obj_w = obj_crop.shape[:2]
            scale = random.uniform(0.5, 1.5)
            new_w = max(20, int(obj_w * scale))
            new_h = max(20, int(obj_h * scale))

            x = random.randint(0, max(0, bg_w - new_w))
            y = random.randint(0, max(0, bg_h - new_h))
            alpha = random.uniform(*alpha_range)

            composite, yolo_bbox = overlay_object_on_background(bg, obj_crop, (x, y), alpha, scale)
            if yolo_bbox is None:
                continue

            new_labels = [[int(label[0]), *yolo_bbox]]

            # Save
            out_name = f"synthetic_{generated:05d}.jpg"
            cv2.imwrite(str(out_img_dir / out_name), composite)
            save_yolo_labels(new_labels, out_lbl_dir / f"synthetic_{generated:05d}.txt")

            generated += 1
            pbar.update(1)

    logger.info(
        f"\n✅ Synthetic generation complete:\n"
        f"   Generated: {generated} / {num_images}\n"
        f"   Output   : {out_img_dir}"
    )


# ── Standard Augmentation Pass ────────────────────────────────────────────────

def augment_existing_dataset(
    source_dir: Path,
    output_dir: Path,
    augmentor: WaterChannelAugmentor,
    multiplier: int = 2
) -> None:
    """
    Apply augmentations to existing training set and save augmented copies.
    Multiplies dataset size by `multiplier`.
    """
    train_img_dir = source_dir / "images" / "train"
    train_lbl_dir = source_dir / "labels" / "train"
    out_img_dir = output_dir / "images" / "train"
    out_lbl_dir = output_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    logger.info(f"Augmenting {len(images)} training images × {multiplier}...")

    count = 0
    for img_path in tqdm(images, desc="Augmenting"):
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        img = load_image(img_path)
        labels = load_yolo_labels(lbl_path)

        if img is None:
            continue

        for i in range(multiplier):
            aug_img = augmentor.augment(img)
            out_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(out_img_dir / out_name), aug_img)
            # Labels unchanged (no geometric transforms for simplicity)
            save_yolo_labels(labels, out_lbl_dir / f"{img_path.stem}_aug{i}.txt")
            count += 1

    logger.info(f"✅ Generated {count} augmented images → {out_img_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Augmentasi domain saluran air untuk dataset deteksi sampah"
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/yolo"))
    parser.add_argument("--background-dir", type=Path, default=Path("data/backgrounds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_augmented"))
    parser.add_argument("--num-synthetic", type=int, default=200)
    parser.add_argument("--augment-multiplier", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    augmentor = WaterChannelAugmentor(
        alpha_range=(0.6, 1.0),
        noise_std=10.0,
        blur_prob=0.3,
        color_shift_prob=0.4
    )

    # Step 1: Generate synthetic overlay images
    generate_synthetic_water_images(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_images=args.num_synthetic,
        background_dir=args.background_dir if args.background_dir.exists() else None,
        augmentor=augmentor,
        image_size=args.image_size
    )

    # Step 2: Augment existing training images
    augment_existing_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        augmentor=augmentor,
        multiplier=args.augment_multiplier
    )

    logger.info("\n✅ All augmentation complete!")
    logger.info(f"   Use {args.output_dir}/data.yaml for training (update data.yaml path)")


if __name__ == "__main__":
    main()
