#!/usr/bin/env python3
"""
inference_sahi.py
=================
Inference menggunakan SAHI (Slicing Aided Hyper Inference) untuk deteksi
objek kecil (sampah) di saluran air perkotaan.

SAHI membagi gambar besar menjadi patches kecil (640×640), melakukan deteksi
pada setiap patch, lalu menggabungkan hasilnya dengan NMS. Pendekatan ini
secara signifikan meningkatkan deteksi objek kecil yang sering terlewat
oleh deteksi pada gambar penuh.

Pipeline:
    Gambar Input → Slicing (640×640, overlap 0.2) → YOLO per patch →
    Koordinat Patch → Gambar Asli → NMS → Hasil Final

Penggunaan:
    # Single image
    python src/inference_sahi.py \\
        --model weights/best.pt \\
        --source path/to/image.jpg \\
        --output results/

    # Video atau direktori
    python src/inference_sahi.py \\
        --model weights/best.pt \\
        --source data/yolo/images/test/ \\
        --output results/ \\
        --benchmark

Referensi:
    SAHI Paper: https://arxiv.org/abs/2202.06934
    SAHI Repo : https://github.com/obss/sahi
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/inference_sahi.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── SAHI Import with Fallback ────────────────────────────────────────────────
SAHI_AVAILABLE = False
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image_as_pil
    SAHI_AVAILABLE = True
    logger.info("✅ SAHI library available")
except ImportError:
    logger.warning(
        "⚠️  SAHI library not found. "
        "Using manual slicing fallback (inference_manual.py). "
        "Install SAHI: pip install sahi"
    )


# ── SAHI Configuration ────────────────────────────────────────────────────────
SAHI_CONFIG = {
    "slice_height": 640,
    "slice_width": 640,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,
    "postprocess_type": "NMS",
    "postprocess_match_threshold": 0.5,
    "confidence_threshold": 0.25,
}

CLASS_NAMES = [
    "plastic_bottle",
    "plastic_bag",
    "carton",
    "cup",
    "styrofoam",
    "cigarette",
    "other_trash",
]

# Class colors (BGR) for visualization
CLASS_COLORS = {
    0: (0, 128, 255),    # plastic_bottle — orange
    1: (0, 255, 128),    # plastic_bag    — green
    2: (255, 128, 0),    # carton         — blue
    3: (255, 0, 128),    # cup            — pink
    4: (128, 0, 255),    # styrofoam      — purple
    5: (0, 255, 255),    # cigarette      — yellow
    6: (128, 128, 128),  # other_trash    — gray
}


# ── Detection Result Dataclass ────────────────────────────────────────────────

class DetectionResult:
    """Standardized detection result across SAHI and manual modes."""

    def __init__(
        self,
        boxes: np.ndarray,    # [N, 4] in xyxy format (pixels)
        scores: np.ndarray,   # [N]
        class_ids: np.ndarray,  # [N]
        class_names: list,
        image_shape: tuple,
        inference_time_ms: float,
        method: str = "sahi"
    ):
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        self.class_names = class_names
        self.image_shape = image_shape
        self.inference_time_ms = inference_time_ms
        self.method = method
        self.n_detections = len(boxes)

    @property
    def fps(self) -> float:
        return 1000.0 / self.inference_time_ms if self.inference_time_ms > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "n_detections": self.n_detections,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "fps": round(self.fps, 2),
            "method": self.method,
            "detections": [
                {
                    "class_id": int(self.class_ids[i]),
                    "class_name": self.class_names[int(self.class_ids[i])]
                    if int(self.class_ids[i]) < len(self.class_names) else "unknown",
                    "confidence": round(float(self.scores[i]), 4),
                    "bbox_xyxy": [
                        float(self.boxes[i][0]),
                        float(self.boxes[i][1]),
                        float(self.boxes[i][2]),
                        float(self.boxes[i][3])
                    ]
                }
                for i in range(self.n_detections)
            ]
        }

    def __str__(self) -> str:
        return (
            f"DetectionResult: {self.n_detections} detections, "
            f"{self.inference_time_ms:.1f}ms ({self.fps:.1f} FPS), "
            f"method={self.method}"
        )


# ── SAHI Predictor ────────────────────────────────────────────────────────────

class SAHIPredictor:
    """
    YOLO + SAHI inference wrapper.
    
    Menggunakan library SAHI resmi untuk sliced prediction.
    Falls back ke ManualSlicingPredictor jika SAHI tidak tersedia.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        postprocess_type: str = "NMS",
        class_names: list = CLASS_NAMES,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.postprocess_type = postprocess_type
        self.class_names = class_names
        self.device = device
        self._model = None

        if not SAHI_AVAILABLE:
            raise ImportError(
                "SAHI not available. Use ManualSlicingPredictor or "
                "install SAHI: pip install sahi"
            )

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model melalui SAHI AutoDetectionModel."""
        logger.info(f"📦 Loading model via SAHI: {self.model_path}")

        # Try ultralytics type first (YOLOv8/v11)
        for model_type in ["ultralytics", "yolov8"]:
            try:
                self._model = AutoDetectionModel.from_pretrained(
                    model_type=model_type,
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device
                )
                logger.info(f"✅ SAHI model loaded (type: {model_type})")
                return
            except Exception as e:
                logger.warning(f"⚠️  SAHI model_type='{model_type}' failed: {e}")

        raise RuntimeError(f"Could not load model via SAHI: {self.model_path}")

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_visual: bool = False
    ) -> DetectionResult:
        """
        Jalankan SAHI sliced prediction pada satu gambar.
        
        Args:
            image: Path ke gambar atau numpy array (BGR)
            return_visual: Jika True, tambahkan hasil visual ke return

        Returns:
            DetectionResult dengan semua deteksi
        """
        # Handle numpy input — save to temp for SAHI
        temp_path = None
        if isinstance(image, np.ndarray):
            temp_path = "/tmp/_sahi_input.jpg"
            cv2.imwrite(temp_path, image)
            img_path = temp_path
        else:
            img_path = str(image)

        t_start = time.perf_counter()

        try:
            result = get_sliced_prediction(
                image=img_path,
                detection_model=self._model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                postprocess_type=self.postprocess_type,
                postprocess_match_threshold=self.iou_threshold,
                verbose=0
            )
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        # Parse SAHI result → standardized format
        boxes, scores, class_ids = self._parse_sahi_result(result)

        # Get image shape
        if isinstance(image, np.ndarray):
            img_shape = image.shape[:2]
        else:
            img = cv2.imread(img_path)
            img_shape = img.shape[:2] if img is not None else (0, 0)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=self.class_names,
            image_shape=img_shape,
            inference_time_ms=inference_time_ms,
            method="sahi"
        )

    def _parse_sahi_result(
        self,
        sahi_result
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse SAHI ObjectPredictionList → numpy arrays."""
        predictions = sahi_result.object_prediction_list

        if not predictions:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32)
            )

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            bbox = pred.bbox
            boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            scores.append(pred.score.value)
            class_ids.append(pred.category.id)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(class_ids, dtype=np.int32)
        )

    def predict_full_image(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> DetectionResult:
        """
        Prediksi tanpa slicing (baseline comparison).
        Untuk membandingkan dengan SAHI dalam evaluasi.
        """
        from ultralytics import YOLO

        if not hasattr(self, "_yolo_model"):
            self._yolo_model = YOLO(self.model_path)

        if isinstance(image, np.ndarray):
            img_input = image
        else:
            img_input = cv2.imread(str(image))

        t_start = time.perf_counter()
        with torch.no_grad():
            yolo_result = self._yolo_model(
                img_input,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        # Parse ultralytics result
        boxes_tensor = yolo_result.boxes
        if boxes_tensor is not None and len(boxes_tensor) > 0:
            boxes = boxes_tensor.xyxy.cpu().numpy()
            scores = boxes_tensor.conf.cpu().numpy()
            class_ids = boxes_tensor.cls.cpu().numpy().astype(np.int32)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
            class_ids = np.empty(0, dtype=np.int32)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=self.class_names,
            image_shape=img_input.shape[:2],
            inference_time_ms=inference_time_ms,
            method="full_image"
        )


# ── Batch & Directory Inference ───────────────────────────────────────────────

def run_inference_on_directory(
    predictor: SAHIPredictor,
    source_dir: Path,
    output_dir: Path,
    save_visualizations: bool = True,
    max_images: Optional[int] = None,
    compare_full_image: bool = False,
) -> list[dict]:
    """
    Run inference pada semua gambar dalam direktori.
    
    Args:
        predictor: SAHIPredictor instance
        source_dir: Direktori gambar input
        output_dir: Direktori untuk menyimpan hasil
        save_visualizations: Simpan gambar dengan bbox tergambar
        max_images: Batas jumlah gambar (None = semua)
        compare_full_image: Juga jalankan full-image (no SAHI) untuk comparison

    Returns:
        List of result dicts
    """
    from src.utils.visualization import draw_detections

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = [
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in img_extensions
    ]

    if max_images:
        image_paths = image_paths[:max_images]

    logger.info(f"🔍 Running SAHI inference on {len(image_paths)} images...")
    logger.info(f"   Model: {predictor.model_path}")
    logger.info(f"   Slice: {predictor.slice_size}×{predictor.slice_size}")
    logger.info(f"   Overlap: {predictor.overlap_ratio}")

    all_results = []
    total_time = 0.0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not load: {img_path}")
            continue

        # SAHI sliced prediction
        sahi_result = predictor.predict(img)
        total_time += sahi_result.inference_time_ms

        result_dict = {
            "image": img_path.name,
            "image_path": str(img_path),
            "sahi": sahi_result.to_dict()
        }

        # Optional: full image comparison
        if compare_full_image:
            full_result = predictor.predict_full_image(img)
            result_dict["full_image"] = full_result.to_dict()

        all_results.append(result_dict)

        # Save visualization
        if save_visualizations:
            vis_img = draw_detections(
                img,
                sahi_result.boxes,
                sahi_result.scores,
                sahi_result.class_ids,
                CLASS_COLORS,
                predictor.class_names
            )
            save_path = output_dir / f"sahi_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), vis_img)

        logger.info(
            f"   {img_path.name}: "
            f"{sahi_result.n_detections} detections, "
            f"{sahi_result.inference_time_ms:.1f}ms"
        )

    # Summary
    n = len(all_results)
    if n > 0:
        avg_time = total_time / n
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
        avg_dets = sum(r["sahi"]["n_detections"] for r in all_results) / n

        logger.info(
            f"\n📊 Inference Summary:\n"
            f"   Images processed : {n}\n"
            f"   Avg latency      : {avg_time:.1f} ms\n"
            f"   Avg FPS          : {avg_fps:.2f}\n"
            f"   Avg detections   : {avg_dets:.1f}\n"
            f"   Output saved to  : {output_dir}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="SAHI inference untuk deteksi sampah di saluran air"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path ke model weights (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Gambar atau direktori input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sahi",
        help="Direktori output"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run edge benchmark after inference"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run full-image (no SAHI) for comparison"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip saving visualization images"
    )
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    # Setup CPU-only inference
    from src.cpu_inference import setup_cpu_inference
    setup_cpu_inference(num_threads=4)

    # Check SAHI availability
    if not SAHI_AVAILABLE:
        logger.error(
            "SAHI not available. Please run:\n"
            "  pip install sahi\n"
            "Or use inference_manual.py for manual slicing fallback."
        )
        sys.exit(1)

    # Initialize predictor
    predictor = SAHIPredictor(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        device="cpu",
    )

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        # Single image
        img = cv2.imread(str(source))
        if img is None:
            logger.error(f"Could not load image: {source}")
            sys.exit(1)

        result = predictor.predict(img)
        logger.info(f"\n{result}")
        logger.info(f"\nDetections:\n{result.to_dict()}")

        if not args.no_visualize:
            from src.utils.visualization import draw_detections
            vis = draw_detections(
                img, result.boxes, result.scores, result.class_ids,
                CLASS_COLORS, predictor.class_names
            )
            out_path = output / f"sahi_{source.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
            logger.info(f"✅ Visualization saved: {out_path}")

    elif source.is_dir():
        # Directory
        results = run_inference_on_directory(
            predictor=predictor,
            source_dir=source,
            output_dir=output,
            save_visualizations=not args.no_visualize,
            max_images=args.max_images,
            compare_full_image=args.compare
        )

        # Save JSON results
        import json
        json_path = output / "inference_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Results saved: {json_path}")

    else:
        logger.error(f"Source not found: {source}")
        sys.exit(1)

    # Benchmark mode
    if args.benchmark:
        from src.cpu_inference import measure_inference
        logger.info("\n⏱️  Running CPU inference benchmark...")
        img_path_for_bench = source if source.is_file() else next(source.rglob("*.jpg"), None)
        if img_path_for_bench:
            img_bench = cv2.imread(str(img_path_for_bench))
            if img_bench is not None:
                stats = measure_inference(
                    lambda: predictor.predict(img_bench),
                    n_runs=10,
                    warmup_runs=3
                )
                logger.info(
                    f"YOLO+SAHI — Mean: {stats['mean_latency_ms']:.1f}ms "
                    f"| FPS: {stats['mean_fps']:.2f}"
                )


if __name__ == "__main__":
    main()
