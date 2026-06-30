#!/usr/bin/env python3
"""
inference_manual.py
===================
Implementasi manual slicing + NMS sebagai fallback jika SAHI tidak tersedia.

Mengimplementasikan pendekatan SAHI secara manual:
1. Potong gambar menjadi patches dengan ukuran 640×640 dan overlap 0.2
2. Jalankan YOLO inference per patch
3. Transform koordinat bbox dari patch space → original image space
4. Gabungkan semua prediksi dengan NMS

Penggunaan:
    from src.inference_manual import ManualSlicingPredictor
    
    predictor = ManualSlicingPredictor(
        model_path="weights/best.pt",
        slice_size=640,
        overlap_ratio=0.2
    )
    result = predictor.predict("image.jpg")
    
    # Atau via CLI:
    python src/inference_manual.py --model weights/best.pt --source image.jpg
"""

import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Union, Optional, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.ops as ops

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Class Names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "plastic_bottle",
    "plastic_bag",
    "carton",
    "cup",
    "styrofoam",
    "cigarette",
    "other_trash",
]

CLASS_COLORS = {
    0: (0, 128, 255),
    1: (0, 255, 128),
    2: (255, 128, 0),
    3: (255, 0, 128),
    4: (128, 0, 255),
    5: (0, 255, 255),
    6: (128, 128, 128),
}


# ── Slice Generation ──────────────────────────────────────────────────────────

def generate_slices(
    image_height: int,
    image_width: int,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.2
) -> List[Tuple[int, int, int, int]]:
    """
    Hasilkan daftar koordinat slice (x1, y1, x2, y2) dengan overlap.
    
    Algoritma:
    - Hitung stride = slice_size * (1 - overlap_ratio)
    - Iterasi posisi dengan stride tersebut
    - Pastikan slice terakhir mencakup tepi gambar

    Args:
        image_height, image_width: Dimensi gambar original
        slice_height, slice_width: Ukuran setiap patch
        overlap_ratio: Fraksi overlap antar patch (0.2 = 20%)

    Returns:
        List of (x1, y1, x2, y2) tuples (pixel coords)
    """
    stride_y = int(slice_height * (1 - overlap_ratio))
    stride_x = int(slice_width * (1 - overlap_ratio))

    slices = []

    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x1 = x
            y1 = y
            x2 = min(x + slice_width, image_width)
            y2 = min(y + slice_height, image_height)

            slices.append((x1, y1, x2, y2))

            if x2 == image_width:
                break
            x += stride_x

        if y2 == image_height:
            break
        y += stride_y

    return slices


def crop_slice(
    image: np.ndarray,
    slice_coords: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, float, float]:
    """
    Crop patch dari gambar dan resize ke target_size.
    
    Returns:
        (patch_image, scale_x, scale_y)
        scale adalah faktor untuk mengkonversi koordinat patch ke original
    """
    x1, y1, x2, y2 = slice_coords
    patch = image[y1:y2, x1:x2]

    patch_h, patch_w = patch.shape[:2]
    target_w, target_h = target_size

    if patch_w != target_w or patch_h != target_h:
        patch_resized = cv2.resize(patch, (target_w, target_h))
        scale_x = patch_w / target_w
        scale_y = patch_h / target_h
    else:
        patch_resized = patch
        scale_x = 1.0
        scale_y = 1.0

    return patch_resized, scale_x, scale_y


def transform_boxes_to_original(
    boxes: np.ndarray,           # [N, 4] xyxy in patch space
    slice_coords: Tuple[int, int, int, int],
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> np.ndarray:
    """
    Transform bbox koordinat dari patch space ke original image space.
    
    Formula:
        x_orig = x_patch * scale_x + x1_slice
        y_orig = y_patch * scale_y + y1_slice
    """
    if len(boxes) == 0:
        return boxes

    x1_slice, y1_slice, _, _ = slice_coords
    transformed = boxes.copy()

    # Scale from resized patch back to original patch size
    transformed[:, 0] = boxes[:, 0] * scale_x  # x1
    transformed[:, 1] = boxes[:, 1] * scale_y  # y1
    transformed[:, 2] = boxes[:, 2] * scale_x  # x2
    transformed[:, 3] = boxes[:, 3] * scale_y  # y2

    # Translate to original image coordinates
    transformed[:, 0] += x1_slice
    transformed[:, 1] += y1_slice
    transformed[:, 2] += x1_slice
    transformed[:, 3] += y1_slice

    return transformed


# ── Multi-class NMS ───────────────────────────────────────────────────────────

def multiclass_nms(
    boxes: np.ndarray,      # [N, 4] xyxy
    scores: np.ndarray,     # [N]
    class_ids: np.ndarray,  # [N]
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Suppression per class.
    
    Menggunakan torchvision.ops.nms untuk efisiensi.
    Adds class offset to boxes so NMS is applied per class independently.

    Args:
        boxes: [N, 4] xyxy bounding boxes
        scores: [N] confidence scores
        class_ids: [N] class indices
        iou_threshold: IoU threshold untuk NMS
        score_threshold: Minimum confidence threshold

    Returns:
        (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    # Filter by score threshold
    score_mask = scores >= score_threshold
    if not score_mask.any():
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    boxes = boxes[score_mask]
    scores = scores[score_mask]
    class_ids = class_ids[score_mask]

    # Convert to torch tensors for torchvision NMS
    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    class_ids_t = torch.from_numpy(class_ids.astype(np.float32))

    # Class offset trick: shift boxes by class_id * max_coord
    # This ensures NMS is applied independently per class
    max_coord = boxes_t.max() + 1
    offsets = class_ids_t * max_coord
    boxes_offset = boxes_t + offsets[:, None]

    # Apply NMS
    keep_idx = ops.nms(boxes_offset, scores_t, iou_threshold)
    keep = keep_idx.numpy()

    return boxes[keep], scores[keep], class_ids[keep]


# ── Manual Slicing Predictor ──────────────────────────────────────────────────

class ManualSlicingPredictor:
    """
    Implementasi manual SAHI-style slicing inference.
    
    Fallback jika library SAHI tidak tersedia.
    
    Proses:
    1. Generate patch positions (640×640, overlap 20%)
    2. Crop dan resize setiap patch
    3. Jalankan YOLO inference per patch
    4. Transform koordinat kembali ke original image space
    5. Apply Multi-class NMS untuk menggabungkan semua prediksi
    """

    def __init__(
        self,
        model_path: str,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        class_names: list = CLASS_NAMES,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        self.device = device
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        from ultralytics import YOLO

        logger.info(f"📦 Loading YOLO model: {self.model_path}")
        self._model = YOLO(self.model_path)
        logger.info(f"✅ Model loaded for manual slicing inference")

    def _run_yolo_on_patch(
        self,
        patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Jalankan YOLO inference pada satu patch gambar.
        
        Returns:
            (boxes_xyxy, scores, class_ids) — semua dalam patch coordinates
        """
        with torch.no_grad():
            results = self._model(
                patch,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=self.slice_size
            )

        result = results[0]
        boxes_data = result.boxes

        if boxes_data is None or len(boxes_data) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32)
            )

        boxes = boxes_data.xyxy.cpu().numpy()
        scores = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy().astype(np.int32)

        return boxes, scores, class_ids

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        verbose: bool = False
    ):
        """
        Jalankan full manual slicing inference pada satu gambar.
        
        Returns:
            DetectionResult dengan semua deteksi setelah NMS
        """
        # Import here to avoid circular import
        from src.inference_sahi import DetectionResult

        # Load image
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")

        H, W = img.shape[:2]
        t_start = time.perf_counter()

        # Generate slices
        slices = generate_slices(H, W, self.slice_size, self.slice_size, self.overlap_ratio)
        n_slices = len(slices)

        if verbose:
            logger.info(
                f"Image {W}×{H} → {n_slices} patches "
                f"({self.slice_size}×{self.slice_size}, "
                f"overlap={self.overlap_ratio})"
            )

        # Collect predictions from all patches
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for slice_coords in slices:
            # Crop patch
            patch, scale_x, scale_y = crop_slice(
                img, slice_coords, (self.slice_size, self.slice_size)
            )

            # Run YOLO on patch
            boxes, scores, class_ids = self._run_yolo_on_patch(patch)

            if len(boxes) == 0:
                continue

            # Transform to original image coordinates
            boxes_orig = transform_boxes_to_original(
                boxes, slice_coords, scale_x, scale_y
            )

            all_boxes.append(boxes_orig)
            all_scores.append(scores)
            all_class_ids.append(class_ids)

        # Merge all predictions
        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_scores = np.concatenate(all_scores, axis=0)
            merged_class_ids = np.concatenate(all_class_ids, axis=0)
        else:
            merged_boxes = np.empty((0, 4), dtype=np.float32)
            merged_scores = np.empty(0, dtype=np.float32)
            merged_class_ids = np.empty(0, dtype=np.int32)

        # Apply multi-class NMS
        final_boxes, final_scores, final_class_ids = multiclass_nms(
            merged_boxes,
            merged_scores,
            merged_class_ids,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence_threshold
        )

        # Clip boxes to image boundaries
        if len(final_boxes) > 0:
            final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, W)
            final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, H)

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        if verbose:
            logger.info(
                f"  Pre-NMS : {len(merged_boxes)} boxes from {n_slices} patches\n"
                f"  Post-NMS: {len(final_boxes)} final detections\n"
                f"  Time    : {inference_time_ms:.1f} ms"
            )

        return DetectionResult(
            boxes=final_boxes,
            scores=final_scores,
            class_ids=final_class_ids,
            class_names=self.class_names,
            image_shape=(H, W),
            inference_time_ms=inference_time_ms,
            method="manual_slicing"
        )

    def predict_full_image(self, image: Union[str, Path, np.ndarray]):
        """Baseline: predict without slicing."""
        from src.inference_sahi import DetectionResult

        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(str(image))

        t_start = time.perf_counter()
        boxes, scores, class_ids = self._run_yolo_on_patch(img)
        t_end = time.perf_counter()

        # Apply NMS
        boxes, scores, class_ids = multiclass_nms(
            boxes, scores, class_ids,
            self.iou_threshold, self.confidence_threshold
        )

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=self.class_names,
            image_shape=img.shape[:2],
            inference_time_ms=(t_end - t_start) * 1000,
            method="full_image_no_sahi"
        )

    def get_slice_info(self, image_height: int, image_width: int) -> dict:
        """Return info tentang slicing strategy untuk gambar tertentu."""
        slices = generate_slices(
            image_height, image_width,
            self.slice_size, self.slice_size,
            self.overlap_ratio
        )
        n_cols = math.ceil(
            (image_width - self.slice_size * self.overlap_ratio) /
            (self.slice_size * (1 - self.overlap_ratio))
        ) + 1
        n_rows = math.ceil(
            (image_height - self.slice_size * self.overlap_ratio) /
            (self.slice_size * (1 - self.overlap_ratio))
        ) + 1

        return {
            "image_size": f"{image_width}×{image_height}",
            "slice_size": f"{self.slice_size}×{self.slice_size}",
            "overlap_ratio": self.overlap_ratio,
            "n_slices": len(slices),
            "grid": f"~{n_cols}×{n_rows}",
            "slices": slices[:5],  # show first 5
        }


def main():
    parser = argparse.ArgumentParser(
        description="Manual slicing inference (SAHI fallback) untuk deteksi sampah"
    )
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/manual")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with full-image (no slicing)")
    args = parser.parse_args()

    # Setup CPU-only inference
    from src.cpu_inference import setup_cpu_inference
    setup_cpu_inference(num_threads=4)

    predictor = ManualSlicingPredictor(
        model_path=args.model,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device="cpu"
    )

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        img = cv2.imread(str(source))
        H, W = img.shape[:2]
        slice_info = predictor.get_slice_info(H, W)
        logger.info(f"\n📐 Slice info: {slice_info}")

        # Manual slicing
        result = predictor.predict(img, verbose=args.verbose)
        logger.info(f"\n{result}")

        # Comparison
        if args.compare:
            full_result = predictor.predict_full_image(img)
            logger.info(f"\n🆚 Full image (no slicing): {full_result}")
            logger.info(
                f"\n📈 SAHI gain: "
                f"{result.n_detections - full_result.n_detections:+d} detections"
            )

        # Visualization
        from src.utils.visualization import draw_detections
        vis = draw_detections(
            img, result.boxes, result.scores,
            result.class_ids, CLASS_COLORS, predictor.class_names
        )
        out_path = output / f"manual_{source.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        logger.info(f"✅ Saved: {out_path}")

    elif source.is_dir():
        images = list(source.rglob("*.jpg")) + list(source.rglob("*.png"))
        for img_path in images[:args.max_images if hasattr(args, "max_images") and args.max_images else None]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            result = predictor.predict(img, verbose=args.verbose)
            logger.info(f"  {img_path.name}: {result}")

    if args.benchmark and source.is_file():
        from src.cpu_inference import measure_inference
        img = cv2.imread(str(source))
        stats = measure_inference(
            lambda: predictor.predict(img),
            n_runs=10,
            warmup_runs=3
        )
        logger.info(
            f"Manual SAHI — Mean: {stats['mean_latency_ms']:.1f}ms "
            f"| FPS: {stats['mean_fps']:.2f}"
        )


if __name__ == "__main__":
    main()
