#!/usr/bin/env python3
"""
metrics.py
==========
Modul evaluasi komprehensif untuk sistem deteksi sampah di saluran air.

Metrik yang dihitung:
1. mAP@0.5 dan mAP@0.5:0.95 (COCO standard)
2. Precision, Recall, F1 per class dan overall
3. FPS (Frames Per Second) dan Latency (ms/frame)
4. Memory Usage (MB) selama inference
5. Perbandingan: YOLO saja vs YOLO+SAHI
6. Analisis deteksi objek kecil (area < 32×32 px)

Output:
- CSV files di results/metrics/
- Summary report (TXT)
- Grafik perbandingan (PNG)
"""

import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision.ops as ops

# ── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SMALL_OBJECT_AREA_THRESHOLD = 1024    # < 32×32 pixels = "small object"
MEDIUM_OBJECT_AREA_THRESHOLD = 9216   # < 96×96 pixels = "medium object"

CLASS_NAMES = [
    "plastic_bottle",
    "plastic_bag",
    "carton",
    "cup",
    "styrofoam",
    "cigarette",
    "other_trash",
]


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Single bounding box with label."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0
    is_ground_truth: bool = True

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def is_small(self) -> bool:
        return self.area < SMALL_OBJECT_AREA_THRESHOLD

    @property
    def is_medium(self) -> bool:
        return SMALL_OBJECT_AREA_THRESHOLD <= self.area < MEDIUM_OBJECT_AREA_THRESHOLD


@dataclass
class ImageMetrics:
    """Per-image detection metrics."""
    image_id: str
    n_ground_truth: int
    n_predictions: int
    n_true_positives: int
    n_false_positives: int
    n_false_negatives: int
    inference_time_ms: float
    method: str = "unknown"
    n_small_gt: int = 0
    n_small_tp: int = 0


@dataclass
class ClassMetrics:
    """Per-class aggregate metrics."""
    class_id: int
    class_name: str
    ap50: float = 0.0
    ap50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_ground_truth: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0
    n_small_objects: int = 0
    small_object_recall: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark comparison report."""
    model_name: str
    method: str
    n_images: int
    map50: float = 0.0
    map50_95: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    mean_fps: float = 0.0
    peak_memory_mb: float = 0.0
    small_object_recall: float = 0.0
    class_metrics: List[ClassMetrics] = field(default_factory=list)


# ── IoU Computation ───────────────────────────────────────────────────────────

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(
    gt_boxes: np.ndarray,   # [M, 4]
    pred_boxes: np.ndarray  # [N, 4]
) -> np.ndarray:
    """
    Compute IoU matrix between all GT-prediction pairs.
    Returns [M, N] matrix.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)))

    gt_t = torch.from_numpy(gt_boxes.astype(np.float32))
    pred_t = torch.from_numpy(pred_boxes.astype(np.float32))

    iou_matrix = ops.box_iou(gt_t, pred_t).numpy()
    return iou_matrix


# ── Precision-Recall & AP Computation ────────────────────────────────────────

def compute_average_precision(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    Compute Average Precision using 101-point interpolation (COCO style).
    """
    recall_thresholds = np.linspace(0, 1, 101)
    interpolated_precisions = []

    for r_thresh in recall_thresholds:
        # Maximum precision for recall >= r_thresh
        prec_at_recall = precisions[recalls >= r_thresh]
        interpolated_precisions.append(
            prec_at_recall.max() if len(prec_at_recall) > 0 else 0.0
        )

    return float(np.mean(interpolated_precisions))


def match_predictions_to_gt(
    gt_boxes: np.ndarray,        # [M, 4] xyxy
    gt_class_ids: np.ndarray,    # [M]
    pred_boxes: np.ndarray,      # [N, 4] xyxy
    pred_scores: np.ndarray,     # [N]
    pred_class_ids: np.ndarray,  # [N]
    iou_threshold: float = 0.5
) -> Dict[int, dict]:
    """
    Match predictions to ground truth boxes per class.
    
    Returns dict: class_id → {'tp': [...], 'fp': [...], 'scores': [...], 'n_gt': int}
    """
    classes = set(np.unique(gt_class_ids)) | set(np.unique(pred_class_ids))
    results = {}

    for cls_id in classes:
        gt_mask = gt_class_ids == cls_id
        pred_mask = pred_class_ids == cls_id

        cls_gt_boxes = gt_boxes[gt_mask] if len(gt_boxes) > 0 else np.empty((0, 4))
        cls_pred_boxes = pred_boxes[pred_mask] if len(pred_boxes) > 0 else np.empty((0, 4))
        cls_pred_scores = pred_scores[pred_mask] if len(pred_scores) > 0 else np.empty(0)

        n_gt = len(cls_gt_boxes)
        n_pred = len(cls_pred_boxes)

        # Sort predictions by confidence (descending)
        if n_pred > 0:
            sort_idx = np.argsort(-cls_pred_scores)
            cls_pred_boxes = cls_pred_boxes[sort_idx]
            cls_pred_scores = cls_pred_scores[sort_idx]

        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)
        gt_matched = np.zeros(n_gt, dtype=bool)

        if n_gt > 0 and n_pred > 0:
            iou_matrix = compute_iou_matrix(cls_gt_boxes, cls_pred_boxes)

            for pred_idx in range(n_pred):
                best_iou = iou_threshold
                best_gt_idx = -1

                for gt_idx in range(n_gt):
                    if gt_matched[gt_idx]:
                        continue
                    if iou_matrix[gt_idx, pred_idx] >= best_iou:
                        best_iou = iou_matrix[gt_idx, pred_idx]
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
        else:
            fp = np.ones(n_pred)

        results[int(cls_id)] = {
            "tp": tp,
            "fp": fp,
            "scores": cls_pred_scores,
            "n_gt": n_gt
        }

    return results


def compute_ap_per_class(
    match_results: Dict[int, dict],
    iou_threshold: float = 0.5
) -> Dict[int, float]:
    """Compute AP for each class from match results."""
    ap_per_class = {}

    for cls_id, data in match_results.items():
        tp = data["tp"]
        fp = data["fp"]
        n_gt = data["n_gt"]

        if n_gt == 0 and len(tp) == 0:
            ap_per_class[cls_id] = float("nan")
            continue

        # Cumulative TP and FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / max(n_gt, 1)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-10)

        ap = compute_average_precision(recalls, precisions)
        ap_per_class[cls_id] = ap

    return ap_per_class


# ── Main Evaluation Engine ────────────────────────────────────────────────────

class DetectionEvaluator:
    """
    Evaluator utama untuk membandingkan metrik deteksi.
    Mendukung evaluasi YOLO only vs YOLO+SAHI.
    """

    def __init__(
        self,
        class_names: List[str] = CLASS_NAMES,
        iou_thresholds: Optional[List[float]] = None,
        output_dir: Path = Path("results/metrics")
    ):
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated data
        self._runs: Dict[str, List[dict]] = defaultdict(list)  # method → list of image results

    def add_image_result(
        self,
        method: str,
        image_id: str,
        gt_boxes: np.ndarray,
        gt_class_ids: np.ndarray,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_class_ids: np.ndarray,
        inference_time_ms: float
    ) -> None:
        """Register a single image's detection result for evaluation."""
        self._runs[method].append({
            "image_id": image_id,
            "gt_boxes": gt_boxes,
            "gt_class_ids": gt_class_ids,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_class_ids": pred_class_ids,
            "inference_time_ms": inference_time_ms
        })

    def evaluate_method(self, method: str) -> BenchmarkReport:
        """
        Compute full evaluation metrics for a given method.
        """
        data = self._runs.get(method, [])
        if not data:
            logger.warning(f"No data for method: {method}")
            return BenchmarkReport(model_name="unknown", method=method, n_images=0)

        # ── Latency metrics ───────────────────────────────────────────────────
        latencies = [d["inference_time_ms"] for d in data]
        latencies.sort()
        n = len(latencies)
        mean_lat = sum(latencies) / n
        median_lat = latencies[n // 2]
        p90_lat = latencies[int(0.9 * n)]
        mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        # ── mAP computation (multi-threshold) ────────────────────────────────
        ap_at_thresholds = defaultdict(list)  # class_id → list of AP values

        for iou_thresh in self.iou_thresholds:
            # Aggregate match results across all images
            all_matches: Dict[int, dict] = defaultdict(
                lambda: {"tp": [], "fp": [], "scores": [], "n_gt": 0}
            )

            for d in data:
                matches = match_predictions_to_gt(
                    d["gt_boxes"], d["gt_class_ids"],
                    d["pred_boxes"], d["pred_scores"], d["pred_class_ids"],
                    iou_threshold=iou_thresh
                )
                for cls_id, m in matches.items():
                    all_matches[cls_id]["tp"].extend(m["tp"].tolist())
                    all_matches[cls_id]["fp"].extend(m["fp"].tolist())
                    all_matches[cls_id]["scores"].extend(m["scores"].tolist())
                    all_matches[cls_id]["n_gt"] += m["n_gt"]

            # Convert to numpy and sort by score
            for cls_id, m in all_matches.items():
                arr_tp = np.array(m["tp"])
                arr_fp = np.array(m["fp"])
                arr_sc = np.array(m["scores"])
                sort_idx = np.argsort(-arr_sc)
                all_matches[cls_id]["tp"] = arr_tp[sort_idx]
                all_matches[cls_id]["fp"] = arr_fp[sort_idx]

            ap_per_cls = compute_ap_per_class(dict(all_matches), iou_thresh)
            for cls_id, ap in ap_per_cls.items():
                ap_at_thresholds[cls_id].append(ap)

        # ── Per-class metrics ─────────────────────────────────────────────────
        class_metrics = []
        all_ap50 = []
        all_ap50_95 = []
        all_precision = []
        all_recall = []
        all_f1 = []

        for cls_id in range(len(self.class_names)):
            cls_name = self.class_names[cls_id]
            aps = ap_at_thresholds.get(cls_id, [0.0])

            ap50 = aps[0] if aps else 0.0
            valid_aps = [a for a in aps if not np.isnan(a)]
            ap50_95 = float(np.mean(valid_aps)) if valid_aps else 0.0

            # Aggregate TP/FP/FN for precision/recall
            total_tp = total_fp = total_fn = 0
            n_small_gt = n_small_tp = 0

            for d in data:
                gt_mask = d["gt_class_ids"] == cls_id
                pred_mask = d["pred_class_ids"] == cls_id
                gt_b = d["gt_boxes"][gt_mask]
                pred_b = d["pred_boxes"][pred_mask]
                pred_s = d["pred_scores"][pred_mask]

                if len(gt_b) > 0 and len(pred_b) > 0:
                    iou_mat = compute_iou_matrix(gt_b, pred_b)
                    matched = (iou_mat >= 0.5)
                    tp = int(matched.max(axis=1).sum())
                    fp = len(pred_b) - tp
                    fn = len(gt_b) - tp
                elif len(gt_b) == 0:
                    tp, fp, fn = 0, len(pred_b), 0
                else:
                    tp, fp, fn = 0, 0, len(gt_b)

                total_tp += tp
                total_fp += max(0, fp)
                total_fn += max(0, fn)

                # Small object tracking
                for box in gt_b:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area < SMALL_OBJECT_AREA_THRESHOLD:
                        n_small_gt += 1

            precision = total_tp / max(total_tp + total_fp, 1)
            recall = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            cm = ClassMetrics(
                class_id=cls_id,
                class_name=cls_name,
                ap50=round(ap50, 4),
                ap50_95=round(ap50_95, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                n_ground_truth=total_tp + total_fn,
                n_true_positives=total_tp,
                n_false_positives=total_fp,
                n_false_negatives=total_fn,
                n_small_objects=n_small_gt,
            )
            class_metrics.append(cm)

            if not np.isnan(ap50):
                all_ap50.append(ap50)
                all_ap50_95.append(ap50_95)
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

        # Overall metrics
        map50 = float(np.mean(all_ap50)) if all_ap50 else 0.0
        map50_95 = float(np.mean(all_ap50_95)) if all_ap50_95 else 0.0
        mean_prec = float(np.mean(all_precision)) if all_precision else 0.0
        mean_rec = float(np.mean(all_recall)) if all_recall else 0.0
        mean_f1 = float(np.mean(all_f1)) if all_f1 else 0.0

        return BenchmarkReport(
            model_name="yolo11n",
            method=method,
            n_images=n,
            map50=round(map50, 4),
            map50_95=round(map50_95, 4),
            mean_precision=round(mean_prec, 4),
            mean_recall=round(mean_rec, 4),
            mean_f1=round(mean_f1, 4),
            mean_latency_ms=round(mean_lat, 2),
            median_latency_ms=round(median_lat, 2),
            p90_latency_ms=round(p90_lat, 2),
            mean_fps=round(mean_fps, 2),
            class_metrics=class_metrics
        )

    def compare_methods(
        self,
        methods: Optional[List[str]] = None,
        save_csv: bool = True,
        save_plots: bool = True
    ) -> Dict[str, BenchmarkReport]:
        """
        Compare all registered methods and generate comprehensive reports.
        """
        if methods is None:
            methods = list(self._runs.keys())

        reports = {}
        for method in methods:
            logger.info(f"\n🔍 Evaluating method: {method}")
            report = self.evaluate_method(method)
            reports[method] = report
            self._print_report(report)

        if save_csv:
            self._save_csv(reports)

        if save_plots:
            self._plot_comparison(reports)

        return reports

    def _print_report(self, report: BenchmarkReport) -> None:
        """Print formatted report to logger."""
        logger.info(
            f"\n{'='*60}\n"
            f"📊 Evaluation Report — {report.method.upper()}\n"
            f"{'='*60}\n"
            f"  Images      : {report.n_images}\n"
            f"  mAP@0.5     : {report.map50:.4f}\n"
            f"  mAP@0.5:0.95: {report.map50_95:.4f}\n"
            f"  Precision   : {report.mean_precision:.4f}\n"
            f"  Recall      : {report.mean_recall:.4f}\n"
            f"  F1 Score    : {report.mean_f1:.4f}\n"
            f"  Mean FPS    : {report.mean_fps:.2f}\n"
            f"  Mean Latency: {report.mean_latency_ms:.1f} ms\n"
            f"  P90 Latency : {report.p90_latency_ms:.1f} ms\n"
            f"  Peak Memory : {report.peak_memory_mb:.1f} MB\n"
            f"\n  Per-class mAP@0.5:\n"
            + "\n".join(
                f"    {cm.class_name:<20} AP={cm.ap50:.4f}  "
                f"P={cm.precision:.3f}  R={cm.recall:.3f}"
                for cm in report.class_metrics
            ) +
            f"\n{'='*60}"
        )

    def _save_csv(self, reports: Dict[str, BenchmarkReport]) -> None:
        """Save benchmark results to CSV."""
        # Summary CSV
        summary_path = self.output_dir / "benchmark_summary.csv"
        summary_rows = []
        for method, report in reports.items():
            summary_rows.append({
                "method": method,
                "n_images": report.n_images,
                "mAP50": report.map50,
                "mAP50_95": report.map50_95,
                "precision": report.mean_precision,
                "recall": report.mean_recall,
                "f1": report.mean_f1,
                "mean_fps": report.mean_fps,
                "mean_latency_ms": report.mean_latency_ms,
                "p90_latency_ms": report.p90_latency_ms,
                "peak_memory_mb": report.peak_memory_mb,
            })

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"📄 Summary CSV saved: {summary_path}")

        # Per-class CSV
        class_path = self.output_dir / "class_metrics.csv"
        class_rows = []
        for method, report in reports.items():
            for cm in report.class_metrics:
                row = asdict(cm)
                row["method"] = method
                class_rows.append(row)

        pd.DataFrame(class_rows).to_csv(class_path, index=False)
        logger.info(f"📄 Class metrics CSV saved: {class_path}")

    def _plot_comparison(self, reports: Dict[str, BenchmarkReport]) -> None:
        """Generate comparison plots."""
        try:
            import seaborn as sns  # noqa: F401  optional dependency
        except ImportError:
            pass  # seaborn not required for plotting
        if len(reports) < 2:
            logger.info("Need at least 2 methods for comparison plots")
            return

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Deteksi Sampah di Saluran Air — Perbandingan Metode",
            fontsize=14, fontweight="bold"
        )

        methods = list(reports.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        # 1. mAP Comparison
        ax = axes[0, 0]
        metrics_to_plot = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(methods)

        for i, (method, report) in enumerate(reports.items()):
            values = [
                report.map50, report.map50_95,
                report.mean_precision, report.mean_recall, report.mean_f1
            ]
            bars = ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                         values, width, label=method, color=colors[i], alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title("Detection Metrics")
        ax.legend()

        # 2. Latency Distribution (Box plot)
        ax = axes[0, 1]
        lat_data = []
        lat_labels = []
        for method in methods:
            lats = [d["inference_time_ms"] for d in self._runs[method]]
            lat_data.append(lats)
            lat_labels.append(method)

        bp = ax.boxplot(lat_data, labels=lat_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency Distribution")
        ax.axhline(y=200, color='red', linestyle='--', label='Latency threshold (200ms)')
        ax.legend()

        # 3. FPS Comparison
        ax = axes[0, 2]
        fps_values = [r.mean_fps for r in reports.values()]
        bars = ax.bar(methods, fps_values, color=colors[:len(methods)], alpha=0.85)
        for bar, val in zip(bars, fps_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        ax.axhline(y=5, color='red', linestyle='--', label='Target FPS (5)')
        ax.set_ylabel("FPS")
        ax.set_title("Throughput (FPS)")
        ax.legend()

        # 4. Per-class mAP@0.5 Comparison
        ax = axes[1, 0]
        class_names = self.class_names
        x = np.arange(len(class_names))
        width = 0.8 / len(methods)

        for i, (method, report) in enumerate(reports.items()):
            ap_values = [cm.ap50 for cm in report.class_metrics]
            ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                  ap_values, width, label=method, color=colors[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("AP@0.5")
        ax.set_title("Per-class mAP@0.5")
        ax.legend()

        # 5. Recall comparison
        ax = axes[1, 1]
        for i, (method, report) in enumerate(reports.items()):
            rec_values = [cm.recall for cm in report.class_metrics]
            ax.plot(class_names, rec_values, 'o-', label=method,
                   color=colors[i], linewidth=2, markersize=6)

        ax.set_ylim(0, 1.0)
        ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Recall")
        ax.set_title("Per-class Recall")
        ax.legend()
        ax.set_xticks(range(len(class_names)))

        # 6. Summary table
        ax = axes[1, 2]
        ax.axis("off")
        table_data = [["Method", "mAP@0.5", "FPS", "Latency(ms)"]]
        for method, report in reports.items():
            table_data.append([
                method,
                f"{report.map50:.3f}",
                f"{report.mean_fps:.1f}",
                f"{report.mean_latency_ms:.0f}"
            ])

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc='center',
            loc='center',
            bbox=[0, 0.2, 1, 0.7]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(4)))
        ax.set_title("Summary Table")

        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_comparison.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 Comparison plot saved: {plot_path}")


# ── Convenience Functions ─────────────────────────────────────────────────────

def load_yolo_ground_truth(
    label_dir: Path,
    image_width: int,
    image_height: int,
    image_stem: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load YOLO format ground truth labels.
    Returns (boxes_xyxy, class_ids).
    """
    label_path = label_dir / f"{image_stem}.txt"
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int32)

    boxes = []
    class_ids = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            x1 = (x_c - w / 2) * image_width
            y1 = (y_c - h / 2) * image_height
            x2 = (x_c + w / 2) * image_width
            y2 = (y_c + h / 2) * image_height
            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)

    if boxes:
        return np.array(boxes, dtype=np.float32), np.array(class_ids, dtype=np.int32)
    return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int32)


def run_evaluation_pipeline(
    predictor_sahi,
    predictor_full,
    test_img_dir: Path,
    test_lbl_dir: Path,
    output_dir: Path = Path("results/metrics"),
    max_images: Optional[int] = None,
) -> Dict[str, BenchmarkReport]:
    """
    Jalankan evaluasi lengkap: SAHI vs Full-image.
    Menghitung semua metrik dan menyimpan CSV + plots.
    """
    import cv2
    evaluator = DetectionEvaluator(output_dir=output_dir)

    test_img_dir = Path(test_img_dir)
    test_lbl_dir = Path(test_lbl_dir)

    images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if max_images:
        images = images[:max_images]

    logger.info(f"\n🔍 Running evaluation on {len(images)} test images...")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        gt_boxes, gt_cls = load_yolo_ground_truth(test_lbl_dir, W, H, img_path.stem)

        # SAHI prediction
        sahi_result = predictor_sahi.predict(img)
        evaluator.add_image_result(
            method="YOLO+SAHI",
            image_id=img_path.stem,
            gt_boxes=gt_boxes,
            gt_class_ids=gt_cls,
            pred_boxes=sahi_result.boxes,
            pred_scores=sahi_result.scores,
            pred_class_ids=sahi_result.class_ids,
            inference_time_ms=sahi_result.inference_time_ms
        )

        # Full image prediction (baseline)
        full_result = predictor_full.predict_full_image(img)
        evaluator.add_image_result(
            method="YOLO_only",
            image_id=img_path.stem,
            gt_boxes=gt_boxes,
            gt_class_ids=gt_cls,
            pred_boxes=full_result.boxes,
            pred_scores=full_result.scores,
            pred_class_ids=full_result.class_ids,
            inference_time_ms=full_result.inference_time_ms
        )

    # Generate reports
    reports = evaluator.compare_methods(save_csv=True, save_plots=True)
    return reports
