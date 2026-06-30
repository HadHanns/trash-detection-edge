#!/usr/bin/env python3
"""
trainer.py
==========
Fine-tuning YOLOv11n (atau YOLOv8n sebagai fallback) pada dataset TACO
untuk deteksi sampah di saluran air perkotaan.

Pipeline:
1. Apply edge constraints (CPU-only, 2 threads) — konsisten dengan inference
2. Load pre-trained YOLOv11n.pt
3. Fine-tune pada dataset TACO (data.yaml)
4. Simpan model terbaik ke weights/best.pt
5. Export ke ONNX dan TorchScript untuk deployment edge

Penggunaan:
    python src/trainer.py \\
        --data data/yolo/data.yaml \\
        --epochs 100 \\
        --batch 8 \\
        --imgsz 640 \\
        --output weights/

Note:
    Training di CPU akan LAMBAT. Untuk penelitian, jalankan di:
    - Google Colab (GPU T4) atau Kaggle (GPU P100)
    - Mac M-series (MPS) dengan melepas edge constraints saat training
    - Setelah training, constraint diterapkan HANYA saat inference benchmark
"""

import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional

import yaml
import torch

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Model Loading with Fallback ───────────────────────────────────────────────

def load_yolo_model(model_name: str = "yolo11n.pt"):
    """
    Load YOLO model dengan fallback mechanism.
    
    Prioritas:
    1. YOLOv11n (yolo11n.pt) — via ultralytics >= 8.3
    2. YOLOv8n (yolov8n.pt) — fallback jika YOLOv11 tidak tersedia

    Returns:
        (model, model_name_used)
    """
    from ultralytics import YOLO

    candidates = [model_name, "yolov8n.pt"]
    if model_name not in candidates:
        candidates.insert(0, model_name)

    for candidate in candidates:
        try:
            logger.info(f"🔄 Attempting to load: {candidate}")
            model = YOLO(candidate)
            logger.info(f"✅ Model loaded: {candidate}")
            logger.info(f"   Architecture: {model.model.__class__.__name__}")
            return model, candidate
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {candidate}: {e}")

    raise RuntimeError(
        "Could not load any YOLO model. "
        "Install ultralytics: pip install ultralytics>=8.3.0"
    )


def verify_data_yaml(data_yaml_path: Path) -> dict:
    """
    Verify data.yaml exists and has required fields.
    Returns parsed config.
    """
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    required_fields = ["train", "val", "nc", "names"]
    for field in required_fields:
        if field not in data_cfg:
            raise ValueError(f"Missing required field '{field}' in {data_yaml_path}")

    logger.info(
        f"📊 Dataset config:\n"
        f"   Classes: {data_cfg['nc']}\n"
        f"   Names  : {data_cfg['names']}\n"
        f"   Train  : {data_cfg['train']}\n"
        f"   Val    : {data_cfg['val']}\n"
        f"   Test   : {data_cfg.get('test', 'N/A')}"
    )
    return data_cfg


def get_training_device() -> str:
    """
    Determine optimal training device.
    
    Priority: MPS (Mac M-series) > CUDA > CPU
    NOTE: Edge constraints (CPU only) are applied during INFERENCE benchmark,
          NOT during training (for practical training speed).
    """
    if torch.backends.mps.is_available():
        logger.info("🍎 Mac M-series GPU (MPS) detected — using for training")
        return "mps"
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 CUDA GPU detected: {gpu_name} — using for training")
        return "0"  # GPU index 0
    else:
        logger.info("💻 CPU mode (training will be slow)")
        return "cpu"


def train_model(
    data_yaml: Path,
    model_name: str = "yolo11n.pt",
    output_dir: Path = Path("weights"),
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.001,
    lrf: float = 0.01,
    patience: int = 15,
    workers: int = 2,
    device: Optional[str] = None,
    project: str = "runs/train",
    name: str = "taco_yolo11n",
    resume: bool = False,
    freeze_layers: int = 0,
    export_onnx: bool = True,
    export_torchscript: bool = False,
) -> dict:
    """
    Main training function.
    
    Args:
        data_yaml: Path ke data.yaml
        model_name: YOLO model file name
        output_dir: Direktori untuk menyimpan best.pt
        epochs: Jumlah epoch training
        imgsz: Input image size
        batch: Batch size
        lr0: Initial learning rate
        lrf: Final LR fraction
        patience: Early stopping patience
        workers: DataLoader workers
        device: Training device (auto-detect if None)
        project: Ultralytics run directory
        name: Run name
        resume: Resume dari checkpoint terakhir
        freeze_layers: Number of backbone layers to freeze (0 = fine-tune all)
        export_onnx: Export model ke ONNX setelah training
        export_torchscript: Export ke TorchScript

    Returns:
        Dict berisi path model dan metrics
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify dataset
    data_cfg = verify_data_yaml(data_yaml)

    # Load model
    model, actual_model_name = load_yolo_model(model_name)

    # Determine device
    if device is None:
        device = get_training_device()

    logger.info(
        f"\n🚀 Starting Training:\n"
        f"   Model   : {actual_model_name}\n"
        f"   Dataset : {data_yaml}\n"
        f"   Classes : {data_cfg['nc']}\n"
        f"   Epochs  : {epochs}\n"
        f"   Batch   : {batch}\n"
        f"   ImgSz   : {imgsz}\n"
        f"   Device  : {device}\n"
        f"   LR      : {lr0} → {lr0 * lrf:.6f}\n"
        f"   Patience: {patience}\n"
    )

    start_time = time.time()

    # Fine-tuning with Ultralytics YOLO API
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=project,
            name=name,
            resume=resume,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            optimizer="AdamW",
            cos_lr=True,
            amp=False,              # Disable AMP for CPU edge compatibility
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
            # Augmentation params
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.1,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            # Freeze backbone layers if specified
            freeze=freeze_layers if freeze_layers > 0 else None,
        )
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    elapsed = time.time() - start_time
    logger.info(f"⏱️  Training completed in {elapsed/3600:.2f} hours")

    # Find best model checkpoint
    run_dir = Path(project) / name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    if not best_pt.exists():
        best_pt = last_pt
        logger.warning("⚠️  best.pt not found, using last.pt")

    # Copy to output_dir
    output_best = output_dir / "best.pt"
    output_last = output_dir / "last.pt"
    shutil.copy2(best_pt, output_best)
    if last_pt.exists():
        shutil.copy2(last_pt, output_last)

    logger.info(f"✅ Best model saved to: {output_best}")

    # Extract metrics
    metrics = {}
    try:
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        elif hasattr(results, "box"):
            metrics = {
                "mAP50": results.box.map50,
                "mAP50-95": results.box.map,
                "precision": results.box.mp,
                "recall": results.box.mr,
            }
    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")

    logger.info(
        f"\n📊 Training Metrics:\n"
        + "\n".join(f"   {k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    )

    # ── Export Models ─────────────────────────────────────────────────────────
    export_results = {}

    if export_onnx:
        export_results["onnx"] = export_model(
            model_path=output_best,
            format="onnx",
            imgsz=imgsz,
            output_dir=output_dir
        )

    if export_torchscript:
        export_results["torchscript"] = export_model(
            model_path=output_best,
            format="torchscript",
            imgsz=imgsz,
            output_dir=output_dir
        )

    return {
        "best_model_path": str(output_best),
        "run_dir": str(run_dir),
        "metrics": metrics,
        "training_time_hours": elapsed / 3600,
        "model_used": actual_model_name,
        "exports": export_results,
    }


def export_model(
    model_path: Path,
    format: str = "onnx",
    imgsz: int = 640,
    output_dir: Path = Path("weights")
) -> Optional[str]:
    """
    Export model ke format tertentu untuk edge deployment.
    
    Supported formats: 'onnx', 'torchscript', 'tflite'
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(f"Model not found for export: {model_path}")
        return None

    try:
        logger.info(f"📦 Exporting model to {format.upper()}...")
        model = YOLO(str(model_path))
        export_path = model.export(
            format=format,
            imgsz=imgsz,
            simplify=True,   # simplify ONNX graph
            device="cpu",
            half=False,      # FP32 for CPU compatibility
        )
        logger.info(f"✅ Exported: {export_path}")

        # Copy to output_dir
        if export_path and Path(export_path).exists():
            dest = output_dir / Path(export_path).name
            shutil.copy2(export_path, dest)
            logger.info(f"📁 Copied to: {dest}")
            return str(dest)

    except Exception as e:
        logger.error(f"❌ Export to {format} failed: {e}")

    return None


def validate_model(
    model_path: Path,
    data_yaml: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.5,
    split: str = "test"
) -> dict:
    """
    Evaluate model pada test set dan return metrics.
    """
    from ultralytics import YOLO

    logger.info(f"\n🔍 Validating model on {split} split...")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Data : {data_yaml}")

    model = YOLO(str(model_path))

    # Apply edge constraints for fair evaluation
    torch.set_num_threads(2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        plots=True,
        verbose=True
    )

    results = {}
    try:
        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "class_mAP50": {
                name: float(ap)
                for name, ap in zip(metrics.names.values(), metrics.box.ap50)
            } if hasattr(metrics.box, "ap50") else {},
        }
        logger.info(
            f"\n📊 Validation Metrics ({split} split):\n"
            f"   mAP@0.5     : {results['mAP50']:.4f}\n"
            f"   mAP@0.5:0.95: {results['mAP50-95']:.4f}\n"
            f"   Precision   : {results['precision']:.4f}\n"
            f"   Recall      : {results['recall']:.4f}"
        )
    except Exception as e:
        logger.warning(f"Could not parse metrics: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv11n untuk deteksi sampah di saluran air"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/yolo/data.yaml"),
        help="Path ke data.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Model checkpoint (default: yolo11n.pt)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights"),
        help="Output directory for weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: 'cpu', 'mps', '0' (GPU). Auto-detect if not specified."
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of backbone layers to freeze (0 = full fine-tune)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation (no training)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip model export after training"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="taco_yolo11n",
        help="Run name for Ultralytics logging"
    )
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    if args.validate_only:
        # Just validate existing model
        model_path = args.output / "best.pt"
        if not model_path.exists():
            model_path = Path("weights/best.pt")
        validate_model(model_path, args.data, args.imgsz)
    else:
        # Full training
        results = train_model(
            data_yaml=args.data,
            model_name=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            patience=args.patience,
            workers=args.workers,
            device=args.device,
            name=args.name,
            resume=args.resume,
            freeze_layers=args.freeze,
            export_onnx=not args.no_export,
        )

        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Best model : {results['best_model_path']}")
        logger.info(f"   Training   : {results['training_time_hours']:.2f} hours")


if __name__ == "__main__":
    main()
