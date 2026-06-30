#!/usr/bin/env python3
"""
nms_utils.py
============
Utilitas NMS (Non-Maximum Suppression) untuk penggabungan prediksi
dari multiple patches dalam manual slicing.

Mencakup:
- Standard NMS (per class)
- Soft-NMS (mengurangi score overlapping boxes, tidak menghapus)
- WBF — Weighted Boxes Fusion (alternatif NMS untuk overlapping detections)
"""

import numpy as np
import torch
import torchvision.ops as ops
from typing import Tuple, Optional


def standard_nms(
    boxes: np.ndarray,       # [N, 4] xyxy
    scores: np.ndarray,      # [N]
    class_ids: np.ndarray,   # [N]
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-class NMS menggunakan offset trick.
    
    Setiap class diberi offset besar sehingga NMS diaplikasikan
    secara independen per class.
    
    Returns: (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    # Score threshold filter
    mask = scores >= score_threshold
    if not mask.any():
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    boxes_f = boxes[mask].astype(np.float32)
    scores_f = scores[mask].astype(np.float32)
    class_ids_f = class_ids[mask].astype(np.float32)

    boxes_t = torch.from_numpy(boxes_f)
    scores_t = torch.from_numpy(scores_f)
    class_ids_t = torch.from_numpy(class_ids_f)

    # Class offset
    max_coord = boxes_t.max() + 1
    offsets = class_ids_t * max_coord
    boxes_offset = boxes_t + offsets[:, None]

    keep = ops.nms(boxes_offset, scores_t, iou_threshold).numpy()

    return boxes_f[keep], scores_f[keep], class_ids_f[keep].astype(np.int32)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    method: str = "gaussian"  # "gaussian" or "linear"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-NMS: mengurangi score boxes yang overlap tinggi
    daripada menghapusnya langsung.
    
    Berguna untuk deteksi objek yang saling berdekatan (sampah menumpuk).
    
    Methods:
    - 'gaussian': score *= exp(-IoU^2 / sigma)
    - 'linear': score *= (1 - IoU) jika IoU > iou_threshold
    
    Returns: (filtered_boxes, filtered_scores, filtered_class_ids)
    """
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    result_boxes = []
    result_scores = []
    result_class_ids = []

    # Process per class
    unique_classes = np.unique(class_ids)
    for cls_id in unique_classes:
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask].copy().astype(np.float32)
        cls_scores = scores[cls_mask].copy().astype(np.float32)
        n = len(cls_boxes)

        for i in range(n):
            # Find max score prediction
            max_idx = np.argmax(cls_scores)
            if cls_scores[max_idx] < score_threshold:
                break

            # Add to results
            result_boxes.append(cls_boxes[max_idx])
            result_scores.append(cls_scores[max_idx])
            result_class_ids.append(cls_id)

            # Suppress overlapping boxes
            best_box = cls_boxes[max_idx]
            cls_scores[max_idx] = 0.0  # Remove from consideration

            for j in range(n):
                if cls_scores[j] == 0:
                    continue
                iou = _compute_iou_pair(best_box, cls_boxes[j])
                if method == "gaussian":
                    cls_scores[j] *= np.exp(-(iou ** 2) / sigma)
                elif method == "linear":
                    if iou > iou_threshold:
                        cls_scores[j] *= (1 - iou)

    if not result_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    return (
        np.array(result_boxes, dtype=np.float32),
        np.array(result_scores, dtype=np.float32),
        np.array(result_class_ids, dtype=np.int32)
    )


def weighted_boxes_fusion(
    boxes_list: list,      # List of [N_i, 4] arrays
    scores_list: list,     # List of [N_i] arrays
    class_ids_list: list,  # List of [N_i] arrays
    iou_threshold: float = 0.55,
    score_threshold: float = 0.25,
    weights: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Boxes Fusion (WBF) — alternatif NMS untuk ensemble/slicing.
    
    WBF menggunakan weighted average dari overlapping boxes
    daripada hanya memilih box dengan score tertinggi.
    
    Reference: https://arxiv.org/abs/1910.13302
    
    Args:
        boxes_list: List of box arrays dari setiap model/patch
        scores_list: List of score arrays
        class_ids_list: List of class id arrays
        iou_threshold: IoU threshold untuk clustering
        score_threshold: Minimum score threshold
        weights: Weight per model (default: equal weights)

    Returns: (fused_boxes, fused_scores, fused_class_ids)
    """
    if not boxes_list:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    if weights is None:
        weights = [1.0] * len(boxes_list)

    # Normalize weights
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    # Flatten all predictions
    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_model_idxs = []

    for model_idx, (boxes, scores, cls_ids) in enumerate(
        zip(boxes_list, scores_list, class_ids_list)
    ):
        mask = scores >= score_threshold
        if not mask.any():
            continue
        all_boxes.extend(boxes[mask].tolist())
        all_scores.extend(scores[mask].tolist())
        all_class_ids.extend(cls_ids[mask].tolist())
        all_model_idxs.extend([model_idx] * int(mask.sum()))

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32)
        )

    all_boxes = np.array(all_boxes, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)
    all_class_ids = np.array(all_class_ids, dtype=np.int32)

    # Sort by score descending
    sort_idx = np.argsort(-all_scores)
    all_boxes = all_boxes[sort_idx]
    all_scores = all_scores[sort_idx]
    all_class_ids = all_class_ids[sort_idx]

    # Cluster overlapping boxes
    clusters: list = []   # each cluster: list of box indices
    cluster_class: list = []

    for i in range(len(all_boxes)):
        cls_id = all_class_ids[i]
        matched_cluster = -1

        for c_idx, (cluster, c_cls) in enumerate(zip(clusters, cluster_class)):
            if c_cls != cls_id:
                continue
            # Compute IoU with cluster representative (first box in cluster)
            rep_box = all_boxes[cluster[0]]
            iou = _compute_iou_pair(all_boxes[i], rep_box)
            if iou >= iou_threshold:
                matched_cluster = c_idx
                break

        if matched_cluster >= 0:
            clusters[matched_cluster].append(i)
        else:
            clusters.append([i])
            cluster_class.append(cls_id)

    # Fuse each cluster
    fused_boxes = []
    fused_scores = []
    fused_class_ids = []

    for cluster, c_cls in zip(clusters, cluster_class):
        cluster_boxes = all_boxes[cluster]
        cluster_scores = all_scores[cluster]

        # Weighted average
        score_weights = cluster_scores / cluster_scores.sum()
        fused_box = (cluster_boxes * score_weights[:, None]).sum(axis=0)
        fused_score = cluster_scores.mean()

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_class_ids.append(c_cls)

    return (
        np.array(fused_boxes, dtype=np.float32),
        np.array(fused_scores, dtype=np.float32),
        np.array(fused_class_ids, dtype=np.int32)
    )


def _compute_iou_pair(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
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


def filter_small_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    min_area: float = 25.0,   # pixels^2
    min_side: float = 3.0     # pixels
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out degenerate boxes yang terlalu kecil.
    Berguna untuk menghilangkan false positives dari patch boundaries.
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    valid = (areas >= min_area) & (widths >= min_side) & (heights >= min_side)
    return boxes[valid], scores[valid], class_ids[valid]
