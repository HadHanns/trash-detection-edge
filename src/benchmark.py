#!/usr/bin/env python3
"""
benchmark.py
============
Script perbandingan komprehensif: YOLO-only vs YOLO+SAHI

Mengukur:
- Akurasi  : mAP@0.5, mAP@0.5:0.95 pada test/val set
- Kecepatan: FPS, latency (ms) rata-rata per gambar

Output:
- Tabel ringkasan di terminal
- results/benchmark_results.json
- results/benchmark_results.csv

Penggunaan:
    python src/benchmark.py
    python src/benchmark.py --split val --device cpu --runs 5
    python src/benchmark.py --split val --device 0    # GPU
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import cv2
import torch

# Tambahkan root ke sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from src.evaluate import (
    load_gt_for_split,
    mAPEvaluator,
    compute_map_range,
    CLASS_NAMES,
)

# ── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/benchmark.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5
SLICE_SIZE     = 640
OVERLAP_RATIO  = 0.2
N_CLASSES      = 7


# ── Speed Measurement ─────────────────────────────────────────────────────────

def measure_speed(
    fn,
    images: list,
    warmup: int = 3,
    repeats: int = 3,
    name: str = "",
) -> Dict[str, float]:
    """
    Ukur kecepatan inference.

    Args:
        fn       : callable fn(img_bgr) → _ (return diabaikan)
        images   : list numpy BGR arrays
        warmup   : jumlah gambar warm-up (tidak dihitung)
        repeats  : jumlah ulangan per gambar

    Returns:
        dict dengan mean_ms, std_ms, fps, total_ms
    """
    # Warmup
    logger.info(f"   Warming up ({warmup} images)...")
    for img in images[:warmup]:
        fn(img)

    latencies = []
    n = len(images)
    for i, img in enumerate(images):
        times_per_img = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(img)
            t1 = time.perf_counter()
            times_per_img.append((t1 - t0) * 1000)
        latency = np.mean(times_per_img)
        latencies.append(latency)
        if (i + 1) % 5 == 0 or (i + 1) == n:
            avg_so_far = np.mean(latencies)
            logger.info(f"   [{i+1}/{n}] avg so far: {avg_so_far:.0f} ms")

    mean_ms = float(np.mean(latencies))
    std_ms  = float(np.std(latencies))
    fps     = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms"  : round(mean_ms, 2),
        "std_latency_ms"   : round(std_ms, 2),
        "fps"              : round(fps, 2),
        "total_images"     : len(latencies),
    }


# ── YOLO-only Predictor ───────────────────────────────────────────────────────

class YOLOPredictor:
    """Standard YOLO full-image inference (no slicing)."""

    def __init__(self, model_path: str, device: str = "cpu"):
        from ultralytics import YOLO
        logger.info(f"📦 Loading YOLO: {model_path}")
        self.model  = YOLO(model_path)
        self.device = device
        logger.info("✅ YOLO loaded")

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (boxes_xyxy, scores, class_ids)."""
        with torch.no_grad():
            results = self.model(
                img,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=self.device,
                verbose=False,
                imgsz=SLICE_SIZE,
            )
        r = results[0].boxes
        if r is None or len(r) == 0:
            return (
                np.empty((0, 4), np.float32),
                np.empty(0, np.float32),
                np.empty(0, np.int32),
            )
        boxes    = r.xyxy.cpu().numpy()
        scores   = r.conf.cpu().numpy()
        cls_ids  = r.cls.cpu().numpy().astype(np.int32)
        # Clamp class IDs to valid range
        cls_ids  = np.clip(cls_ids, 0, N_CLASSES - 1)
        return boxes, scores, cls_ids

    def __call__(self, img):
        return self.predict(img)


# ── SAHI Predictor (manual slicing fallback) ──────────────────────────────────

class SAHIPredictor:
    """
    YOLO + slicing (manual SAHI-style).
    Menggunakan ManualSlicingPredictor yang sudah diimplementasikan.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        from src.inference_manual import ManualSlicingPredictor
        logger.info(f"📦 Loading SAHI predictor: {model_path}")
        self._pred = ManualSlicingPredictor(
            model_path=model_path,
            slice_size=SLICE_SIZE,
            overlap_ratio=OVERLAP_RATIO,
            confidence_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=device,
        )
        logger.info("✅ SAHI predictor loaded")

    def predict(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = self._pred.predict(img)
        return result.boxes, result.scores, result.class_ids

    def __call__(self, img):
        return self.predict(img)


# ── Evaluation Loop ───────────────────────────────────────────────────────────

def evaluate_predictor(
    predictor,
    images: Dict[str, np.ndarray],
    gt: Dict[str, Tuple[np.ndarray, np.ndarray]],
    name: str,
    warmup: int = 2,
    repeats: int = 3,
) -> dict:
    """
    Evaluasi lengkap satu predictor: mAP + speed.

    Args:
        predictor : callable(img) → (boxes, scores, class_ids)
        images    : stem → BGR numpy
        gt        : stem → (gt_boxes, gt_cls)
        name      : nama metode untuk logging

    Returns:
        dict hasil evaluasi
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Evaluating: {name}")
    logger.info(f"{'='*60}")

    image_list = list(images.values())

    # 1) Speed test
    logger.info(f"⏱️  Speed test ({len(image_list)} images × {repeats} runs)...")
    speed = measure_speed(predictor, image_list, warmup=warmup, repeats=repeats, name=name)
    logger.info(
        f"   → {speed['mean_latency_ms']:.1f} ± {speed['std_latency_ms']:.1f} ms  "
        f"| {speed['fps']:.2f} FPS"
    )

    # 2) Collect predictions for mAP
    logger.info(f"📊 Running inference on all {len(images)} images for mAP...")
    all_predictions: Dict[str, Tuple] = {}
    for i, (stem, img) in enumerate(images.items()):
        boxes, scores, cls_ids = predictor(img)
        all_predictions[stem] = (boxes, scores, cls_ids)
        n_det = len(boxes)
        if (i + 1) % 5 == 0 or (i + 1) == len(images):
            logger.info(f"   [{i+1}/{len(images)}] {stem}: {n_det} detections")

    # 3) mAP@0.5 dan mAP@0.5:0.95
    logger.info("📐 Computing mAP@[0.5:0.95]...")
    map_metrics = compute_map_range(
        gt, all_predictions,
        n_classes=N_CLASSES,
        iou_range=np.arange(0.5, 1.0, 0.05),
    )

    # Also compute per-class mAP@0.5
    ev50 = mAPEvaluator(n_classes=N_CLASSES, iou_threshold=0.5)
    for stem, (gt_boxes, gt_cls) in gt.items():
        if stem not in all_predictions:
            continue
        pred_boxes, pred_scores, pred_cls = all_predictions[stem]
        ev50.add(gt_boxes, gt_cls, pred_boxes, pred_scores, pred_cls)
    # compute per-class detail
    per_class = ev50.compute()

    # Show all classes, even if AP=0
    logger.info(f"   mAP@0.50       : {map_metrics['mAP50']:.4f}")
    logger.info(f"   mAP@0.50:0.95  : {map_metrics['mAP50_95']:.4f}")
    logger.info("   Per-class AP@0.5 (semua kelas):")
    for c, cls_name in enumerate(CLASS_NAMES):
        ap  = per_class["per_class_AP"].get(cls_name, 0.0)
        ngt = ev50._n_gt.get(c, 0)
        logger.info(f"     {cls_name:<20} : {ap:.4f}  (GT={ngt})")

    return {
        "method"      : name,
        "mAP50"       : round(map_metrics["mAP50"], 4),
        "mAP50_95"    : round(map_metrics["mAP50_95"], 4),
        "per_class_AP": per_class["per_class_AP"],
        "speed"       : speed,
        "per_iou_mAP" : map_metrics["per_iou"],
    }


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_comparison_table(results: list):
    """Cetak tabel perbandingan yang rapi di terminal."""
    sep = "─" * 70
    print(f"\n{'='*70}")
    print("  📊  HASIL PERBANDINGAN: YOLO-only vs YOLO+SAHI")
    print(f"{'='*70}")
    print(f"  {'Metrik':<25} {'YOLO-only':>15} {'YOLO+SAHI':>15}  {'Δ':>8}")
    print(sep)

    keys = [
        ("mAP@0.5",      "mAP50"),
        ("mAP@0.5:0.95", "mAP50_95"),
    ]
    r0, r1 = results[0], results[1]

    for label, key in keys:
        v0, v1 = r0[key], r1[key]
        delta  = v1 - v0
        sign   = "+" if delta >= 0 else ""
        print(f"  {label:<25} {v0:>15.4f} {v1:>15.4f}  {sign}{delta:.4f}")

    print(sep)
    speed_keys = [
        ("Latency (ms)",   "mean_latency_ms"),
        ("FPS",            "fps"),
    ]
    for label, key in speed_keys:
        v0 = r0["speed"][key]
        v1 = r1["speed"][key]
        delta = v1 - v0
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<25} {v0:>15.2f} {v1:>15.2f}  {sign}{delta:.2f}")

    print(f"{'='*70}\n")

    # Per-class comparison
    print("  📋  Per-class AP@0.5:")
    print(f"  {'Kelas':<22} {'YOLO':>10} {'SAHI':>10} {'Δ':>8}")
    print("  " + "─" * 52)
    for cls_name in CLASS_NAMES:
        v0 = r0["per_class_AP"].get(cls_name, 0.0)
        v1 = r1["per_class_AP"].get(cls_name, 0.0)
        delta = v1 - v0
        sign  = "+" if delta >= 0 else ""
        print(f"  {cls_name:<22} {v0:>10.4f} {v1:>10.4f} {sign}{delta:.4f}")
    print()


def save_results(results: list, output_dir: Path):
    """Simpan hasil ke JSON dan CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 JSON saved: {json_path}")

    # CSV (ringkasan)
    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = [
        "method", "mAP50", "mAP50_95",
        "mean_latency_ms", "std_latency_ms", "fps",
    ]
    rows = []
    for r in results:
        rows.append({
            "method"          : r["method"],
            "mAP50"           : r["mAP50"],
            "mAP50_95"        : r["mAP50_95"],
            "mean_latency_ms" : r["speed"]["mean_latency_ms"],
            "std_latency_ms"  : r["speed"]["std_latency_ms"],
            "fps"             : r["speed"]["fps"],
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"📊 CSV saved: {csv_path}")

    return json_path, csv_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO-only vs YOLO+SAHI untuk deteksi sampah"
    )
    parser.add_argument(
        "--model", default="weights/best.pt",
        help="Path ke model weights (.pt)"
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Dataset split untuk evaluasi"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device inference: 'cpu' atau '0' (GPU)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Jumlah gambar warmup untuk speed test"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Jumlah ulangan per gambar untuk speed test"
    )
    parser.add_argument(
        "--output", default="results",
        help="Direktori output hasil"
    )
    parser.add_argument(
        "--max-images", type=int, default=10,
        help="Batas jumlah gambar untuk speed test (default=10 agar cepat)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  🗑️  Trash Detection Benchmark: YOLO vs YOLO+SAHI")
    logger.info("=" * 60)
    logger.info(f"  Model  : {args.model}")
    logger.info(f"  Split  : {args.split}")
    logger.info(f"  Device : {args.device}")
    logger.info(f"  Warmup : {args.warmup} | Runs: {args.runs}")
    logger.info("=" * 60)

    # ── Paths ──────────────────────────────────────────────────────────────
    data_root  = ROOT / "data" / "yolo"
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split

    if not images_dir.exists():
        logger.error(f"Images dir not found: {images_dir}")
        sys.exit(1)

    # ── Load images & GT ───────────────────────────────────────────────────
    logger.info(f"\n📂 Loading {args.split} set from: {images_dir}")
    gt = load_gt_for_split(images_dir, labels_dir)
    logger.info(f"   GT loaded: {len(gt)} images")

    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in sorted(images_dir.glob("*.*")) if p.suffix.lower() in ext]
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    images: Dict[str, np.ndarray] = {}
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images[p.stem] = img

    logger.info(f"   Images loaded: {len(images)}")

    if not images:
        logger.error("No images found! Check path.")
        sys.exit(1)

    # ── Initialize Predictors ──────────────────────────────────────────────
    yolo_pred = YOLOPredictor(args.model, device=args.device)
    sahi_pred = SAHIPredictor(args.model, device=args.device)

    # ── Run Evaluations ────────────────────────────────────────────────────
    results = []

    result_yolo = evaluate_predictor(
        predictor=yolo_pred,
        images=images,
        gt=gt,
        name="YOLO-only",
        warmup=args.warmup,
        repeats=args.runs,
    )
    results.append(result_yolo)

    result_sahi = evaluate_predictor(
        predictor=sahi_pred,
        images=images,
        gt=gt,
        name="YOLO+SAHI",
        warmup=args.warmup,
        repeats=args.runs,
    )
    results.append(result_sahi)

    # ── Print & Save ───────────────────────────────────────────────────────
    print_comparison_table(results)
    json_path, csv_path = save_results(results, Path(args.output))

    logger.info(f"✅ Benchmark selesai!")
    logger.info(f"   Results: {json_path}")
    logger.info(f"   CSV    : {csv_path}")


if __name__ == "__main__":
    main()
