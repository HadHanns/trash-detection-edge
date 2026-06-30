#!/usr/bin/env python3
"""
evaluate.py
===========
Utilitas komputasi mAP (mean Average Precision) untuk evaluasi deteksi objek.

Mendukung:
- Parse ground truth dari format YOLO (.txt label files)
- Compute per-class AP menggunakan PASCAL VOC 11-point interpolation
- Compute mAP@0.5 dan mAP@0.5:0.95
- Digunakan oleh benchmark.py untuk membandingkan YOLO vs YOLO+SAHI
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


CLASS_NAMES = [
    "plastic_bottle",
    "plastic_bag",
    "carton",
    "cup",
    "styrofoam",
    "cigarette",
    "other_trash",
]


# ── Ground Truth Parsing ──────────────────────────────────────────────────────

def parse_yolo_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse satu file label YOLO (normalized xywh center) → xyxy piksel.

    Format YOLO: class_id cx cy w h  (semua 0–1)

    Returns:
        boxes_xyxy : np.ndarray [N, 4]
        class_ids  : np.ndarray [N]
    """
    if not label_path.exists():
        return np.empty((0, 4), np.float32), np.empty(0, np.int32)

    boxes, cls_ids = [], []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        cls_ids.append(cid)

    if not boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.int32)

    return np.array(boxes, np.float32), np.array(cls_ids, np.int32)


def load_gt_for_split(
    images_dir: Path,
    labels_dir: Path,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Muat semua ground truth untuk satu split (val/test).

    Returns:
        dict  image_stem → (boxes_xyxy, class_ids)
    """
    import cv2

    gt = {}
    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, cls_ids = parse_yolo_label(label_path, w, h)
        gt[img_path.stem] = (boxes, cls_ids)

    return gt


# ── IoU ───────────────────────────────────────────────────────────────────────

def compute_iou_matrix(
    pred_boxes: np.ndarray,   # [M, 4] xyxy
    gt_boxes: np.ndarray,     # [N, 4] xyxy
) -> np.ndarray:              # [M, N]
    """Hitung IoU antara setiap pasangan pred × gt."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), np.float32)

    px1, py1, px2, py2 = (pred_boxes[:, i] for i in range(4))
    gx1, gy1, gx2, gy2 = (gt_boxes[:, i]  for i in range(4))

    inter_x1 = np.maximum(px1[:, None], gx1[None, :])
    inter_y1 = np.maximum(py1[:, None], gy1[None, :])
    inter_x2 = np.minimum(px2[:, None], gx2[None, :])
    inter_y2 = np.minimum(py2[:, None], gy2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    union   = area_p[:, None] + area_g[None, :] - inter

    return np.where(union > 0, inter / union, 0).astype(np.float32)


# ── Precision–Recall & AP ─────────────────────────────────────────────────────

def compute_ap_for_class(
    pred_scores: np.ndarray,    # [M] confidence tiap prediksi
    pred_matched: np.ndarray,   # [M] bool — apakah TP?
    n_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Hitung AP untuk satu kelas menggunakan PASCAL VOC 11-point interpolation.

    Returns:
        ap, precisions, recalls
    """
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    # Sort by descending confidence
    sort_idx = np.argsort(-pred_scores)
    matched  = pred_matched[sort_idx]

    tp_cum = np.cumsum(matched).astype(np.float32)
    fp_cum = np.cumsum(~matched).astype(np.float32)

    recalls    = tp_cum / (n_gt + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Prepend sentinel values
    recalls    = np.concatenate([[0.0], recalls,    [recalls[-1] if len(recalls) > 0 else 0.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Monotone-decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    ap /= 11.0

    return float(ap), precisions, recalls


# ── mAP Computation ───────────────────────────────────────────────────────────

class mAPEvaluator:
    """
    Akumulasi prediksi dan hitung mAP@IoU.

    Cara pakai:
        ev = mAPEvaluator(n_classes=7, iou_threshold=0.5)
        for image_stem, result in predictions.items():
            gt_boxes, gt_cls = gt[image_stem]
            ev.add(gt_boxes, gt_cls, result.boxes, result.scores, result.class_ids)
        metrics = ev.compute()
    """

    def __init__(self, n_classes: int = 7, iou_threshold: float = 0.5):
        self.n_classes    = n_classes
        self.iou_threshold = iou_threshold
        # Per class: list of (score, is_tp)
        self._per_class_preds: Dict[int, List[Tuple[float, bool]]] = {
            c: [] for c in range(n_classes)
        }
        self._n_gt: Dict[int, int] = {c: 0 for c in range(n_classes)}

    def add(
        self,
        gt_boxes:    np.ndarray,  # [N, 4]
        gt_cls:      np.ndarray,  # [N]
        pred_boxes:  np.ndarray,  # [M, 4]
        pred_scores: np.ndarray,  # [M]
        pred_cls:    np.ndarray,  # [M]
    ):
        """Tambahkan satu gambar ke evaluator."""
        # Count GT per class
        for c in gt_cls:
            self._n_gt[int(c)] += 1

        if len(pred_boxes) == 0:
            return

        if len(gt_boxes) == 0:
            # All preds are FP
            for i in range(len(pred_boxes)):
                self._per_class_preds[int(pred_cls[i])].append(
                    (float(pred_scores[i]), False)
                )
            return

        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)   # [M, N]
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # Sort predictions by score descending for greedy matching
        sort_idx = np.argsort(-pred_scores)
        matched_flag = np.zeros(len(pred_boxes), dtype=bool)

        for m in sort_idx:
            pc = int(pred_cls[m])
            # Find best GT of same class with IoU > threshold
            iou_row = iou_mat[m]  # [N]
            best_iou, best_g = -1.0, -1
            for g in range(len(gt_boxes)):
                if int(gt_cls[g]) == pc and not gt_matched[g] and iou_row[g] > best_iou:
                    best_iou, best_g = iou_row[g], g

            is_tp = (best_iou >= self.iou_threshold)
            if is_tp:
                gt_matched[best_g] = True
                matched_flag[m] = True

            self._per_class_preds[pc].append((float(pred_scores[m]), is_tp))

    def compute(self) -> Dict[str, float]:
        """
        Compute mAP@IoU.

        Returns dict dengan:
            map50, per_class_ap, ...
        """
        aps = []
        per_class = {}

        for c in range(self.n_classes):
            preds = self._per_class_preds[c]
            n_gt  = self._n_gt[c]

            if n_gt == 0 and not preds:
                continue

            if not preds:
                per_class[c] = 0.0
                aps.append(0.0)
                continue

            scores  = np.array([p[0] for p in preds], np.float32)
            matched = np.array([p[1] for p in preds], bool)

            ap, _, _ = compute_ap_for_class(scores, matched, n_gt)
            per_class[c] = ap
            aps.append(ap)

        return {
            "mAP": float(np.mean(aps)) if aps else 0.0,
            "per_class_AP": {
                CLASS_NAMES[c]: round(v, 4)
                for c, v in per_class.items()
                if c < len(CLASS_NAMES)
            },
            "n_classes_with_gt": len(aps),
        }


def compute_map_range(
    gt: Dict[str, Tuple[np.ndarray, np.ndarray]],
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int = 7,
    iou_range: np.ndarray = np.arange(0.5, 1.0, 0.05),
) -> Dict[str, float]:
    """
    Compute mAP@[0.5:0.05:0.95] (COCO-style).

    predictions: stem → (boxes, scores, class_ids)
    """
    aps_at_iou = []
    for iou_th in iou_range:
        ev = mAPEvaluator(n_classes=n_classes, iou_threshold=float(iou_th))
        for stem, (gt_boxes, gt_cls) in gt.items():
            if stem not in predictions:
                continue
            pred_boxes, pred_scores, pred_cls = predictions[stem]
            ev.add(gt_boxes, gt_cls, pred_boxes, pred_scores, pred_cls)
        metrics = ev.compute()
        aps_at_iou.append(metrics["mAP"])

    return {
        "mAP50"    : float(aps_at_iou[0]) if aps_at_iou else 0.0,
        "mAP50_95" : float(np.mean(aps_at_iou)) if aps_at_iou else 0.0,
        "per_iou"  : {
            f"{iou:.2f}": round(ap, 4)
            for iou, ap in zip(iou_range, aps_at_iou)
        }
    }
