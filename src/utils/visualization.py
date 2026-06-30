#!/usr/bin/env python3
"""
visualization.py
================
Utilitas visualisasi untuk menampilkan hasil deteksi dan analisis dataset.

Fungsi:
1. draw_detections() — gambar bounding box pada gambar
2. draw_comparison() — tampilkan SAHI vs Full-image side by side
3. plot_training_curves() — kurva loss dan mAP dari training
4. plot_class_distribution() — distribusi kelas dataset
5. visualize_slices() — visualisasi patch SAHI pada gambar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Default Color Palette ─────────────────────────────────────────────────────
DEFAULT_COLORS = {
    0: (0, 128, 255),    # plastic_bottle — orange-blue
    1: (50, 205, 50),    # plastic_bag    — green
    2: (255, 140, 0),    # carton         — orange
    3: (220, 20, 60),    # cup            — crimson
    4: (148, 0, 211),    # styrofoam      — purple
    5: (0, 206, 209),    # cigarette      — teal
    6: (169, 169, 169),  # other_trash    — gray
}

CLASS_NAMES = [
    "plastic_bottle", "plastic_bag", "carton", "cup",
    "styrofoam", "cigarette", "other_trash"
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2


# ── Core Drawing Functions ────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,       # [N, 4] xyxy
    scores: np.ndarray,      # [N]
    class_ids: np.ndarray,   # [N]
    colors: Optional[Dict[int, Tuple]] = None,
    class_names: Optional[List[str]] = None,
    show_confidence: bool = True,
    line_thickness: int = BOX_THICKNESS,
    alpha: float = 0.15,     # fill opacity
) -> np.ndarray:
    """
    Gambar bounding box deteksi pada gambar.
    
    Menggunakan filled rectangle dengan transparency untuk tampilan
    yang lebih informatif.

    Args:
        image: BGR numpy array
        boxes: [N, 4] bounding boxes dalam format xyxy
        scores: [N] confidence scores
        class_ids: [N] class indices
        colors: Dict class_id → (B, G, R) color
        class_names: List of class name strings
        show_confidence: Tampilkan confidence score di label
        line_thickness: Ketebalan garis bbox
        alpha: Opacity fill rectangle

    Returns:
        Annotated BGR image (copy)
    """
    if colors is None:
        colors = DEFAULT_COLORS
    if class_names is None:
        class_names = CLASS_NAMES

    vis = image.copy()
    overlay = vis.copy()
    n_detections = len(boxes)

    for i in range(n_detections):
        box = boxes[i]
        score = float(scores[i])
        cls_id = int(class_ids[i])

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = colors.get(cls_id, (200, 200, 200))

        # Filled rectangle overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Solid border
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)

        # Label
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        if show_confidence:
            label = f"{cls_name} {score:.2f}"
        else:
            label = cls_name

        # Label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        label_y1 = max(y1 - label_h - baseline - 4, 0)
        cv2.rectangle(
            vis,
            (x1, label_y1),
            (x1 + label_w + 4, label_y1 + label_h + baseline + 4),
            color, -1
        )

        # Label text
        text_color = (255, 255, 255)  # white text
        cv2.putText(
            vis, label,
            (x1 + 2, label_y1 + label_h + 2),
            FONT, FONT_SCALE, text_color, FONT_THICKNESS
        )

    # Blend overlay (fill transparency)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Stats overlay in top-left corner
    stats_text = f"Detections: {n_detections}"
    cv2.putText(vis, stats_text, (10, 25), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, stats_text, (10, 25), FONT, 0.7, (0, 0, 0), 1)

    return vis


def draw_ground_truth(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_class_ids: np.ndarray,
    colors: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw ground truth boxes with dashed style."""
    if colors is None:
        colors = DEFAULT_COLORS
    if class_names is None:
        class_names = CLASS_NAMES

    vis = image.copy()
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        cls_id = int(gt_class_ids[i])
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = colors.get(cls_id, (200, 200, 200))

        # Draw dashed rectangle for GT
        dash_length = 10
        for side in range(4):
            if side == 0:  # top
                pts = [(x1 + j, y1) for j in range(0, x2 - x1, dash_length * 2)]
                for pt in pts:
                    cv2.line(vis, pt, (min(pt[0] + dash_length, x2), y1), color, 2)
            elif side == 1:  # bottom
                pts = [(x1 + j, y2) for j in range(0, x2 - x1, dash_length * 2)]
                for pt in pts:
                    cv2.line(vis, pt, (min(pt[0] + dash_length, x2), y2), color, 2)
            elif side == 2:  # left
                pts = [(x1, y1 + j) for j in range(0, y2 - y1, dash_length * 2)]
                for pt in pts:
                    cv2.line(vis, pt, (x1, min(pt[1] + dash_length, y2)), color, 2)
            elif side == 3:  # right
                pts = [(x2, y1 + j) for j in range(0, y2 - y1, dash_length * 2)]
                for pt in pts:
                    cv2.line(vis, pt, (x2, min(pt[1] + dash_length, y2)), color, 2)

        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        cv2.putText(vis, f"GT:{cls_name}", (x1, y2 + 15), FONT, 0.4, color, 1)

    return vis


def draw_comparison(
    image: np.ndarray,
    result_a,        # DetectionResult
    result_b,        # DetectionResult
    label_a: str = "YOLO only",
    label_b: str = "YOLO+SAHI",
    colors: Optional[Dict] = None,
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Side-by-side comparison of two detection methods.
    
    Returns wide image with both results shown side by side.
    """
    vis_a = draw_detections(
        image, result_a.boxes, result_a.scores, result_a.class_ids,
        colors, class_names
    )
    vis_b = draw_detections(
        image, result_b.boxes, result_b.scores, result_b.class_ids,
        colors, class_names
    )

    # Add method labels
    H, W = image.shape[:2]
    header_h = 40

    def add_header(img, text, latency_ms, n_det):
        banner = np.zeros((header_h, W, 3), dtype=np.uint8)
        label = f"{text} | {n_det} dets | {latency_ms:.0f}ms"
        cv2.putText(banner, label, (10, 28), FONT, 0.65, (255, 255, 255), 2)
        return np.vstack([banner, img])

    vis_a = add_header(vis_a, label_a, result_a.inference_time_ms, result_a.n_detections)
    vis_b = add_header(vis_b, label_b, result_b.inference_time_ms, result_b.n_detections)

    # Add divider
    divider = np.zeros((vis_a.shape[0], 4, 3), dtype=np.uint8)
    divider[:] = (255, 255, 0)  # yellow divider

    comparison = np.hstack([vis_a, divider, vis_b])
    return comparison


def visualize_slices(
    image: np.ndarray,
    slices: List[Tuple[int, int, int, int]],
    alpha: float = 0.3,
    grid_color: Tuple = (0, 255, 255),
) -> np.ndarray:
    """
    Visualize SAHI slicing grid on image.
    Shows how the image is divided into patches.
    """
    vis = image.copy()
    overlay = vis.copy()

    for i, (x1, y1, x2, y2) in enumerate(slices):
        # Alternating colors for adjacent patches
        color = grid_color if i % 2 == 0 else (255, 165, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # Patch index
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(vis, str(i + 1), (cx - 10, cy + 10), FONT, 0.7, color, 2)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Info
    info = f"SAHI Slices: {len(slices)}"
    cv2.putText(vis, info, (10, 25), FONT, 0.8, (255, 255, 255), 2)

    return vis


# ── Matplotlib Plots ──────────────────────────────────────────────────────────

def plot_training_curves(
    results_csv_path: str,
    output_dir: Path = Path("results/visualizations"),
    title: str = "Training Progress"
) -> None:
    """
    Plot training loss and mAP curves from Ultralytics results.csv.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        logger.error(f"Could not load results CSV: {e}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.style.use("seaborn-v0_8-darkgrid")

    epochs = range(1, len(df) + 1)

    # Loss curves
    loss_cols = {
        "Box Loss": ("train/box_loss", "val/box_loss"),
        "Obj Loss": ("train/obj_loss", "val/obj_loss"),
        "Cls Loss": ("train/cls_loss", "val/cls_loss"),
    }

    for idx, (loss_name, (train_col, val_col)) in enumerate(loss_cols.items()):
        ax = axes[0, idx]
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], "b-", label="Train", linewidth=1.5)
        if val_col in df.columns:
            ax.plot(epochs, df[val_col], "r-", label="Val", linewidth=1.5)
        ax.set_title(loss_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    # mAP curves
    map_cols = [
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("metrics/precision(B)", "Precision"),
    ]

    for idx, (col, label) in enumerate(map_cols):
        ax = axes[1, idx]
        if col in df.columns:
            ax.plot(epochs, df[col], "g-", label=label, linewidth=2)
            # Mark best
            best_epoch = df[col].idxmax()
            ax.axvline(x=best_epoch + 1, color="red", linestyle="--",
                      label=f"Best: {df[col].max():.4f}")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Training curves saved: {save_path}")


def plot_class_distribution(
    label_dir: Path,
    class_names: List[str] = CLASS_NAMES,
    output_dir: Path = Path("results/visualizations"),
    split_name: str = "train"
) -> None:
    """
    Plot class distribution histogram for a dataset split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = Path(label_dir)
    class_counts = {i: 0 for i in range(len(class_names))}

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(
        class_names,
        [class_counts.get(i, 0) for i in range(len(class_names))],
        color=colors, edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, class_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
               str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title(f"Class Distribution — {split_name} split", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(class_names, rotation=30, ha="right")

    plt.tight_layout()
    save_path = output_dir / f"class_distribution_{split_name}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Class distribution plot saved: {save_path}")


def create_detection_legend(
    class_names: List[str] = CLASS_NAMES,
    colors: Optional[Dict] = None,
) -> np.ndarray:
    """
    Create a legend image for detection visualization.
    Returns BGR numpy array.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    n_classes = len(class_names)
    legend_h = max(200, n_classes * 35 + 20)
    legend_w = 250
    legend = np.ones((legend_h, legend_w, 3), dtype=np.uint8) * 40  # dark bg

    cv2.putText(legend, "Classes", (10, 25), FONT, 0.7, (255, 255, 255), 1)

    for i, cls_name in enumerate(class_names):
        y = 45 + i * 35
        color = colors.get(i, (200, 200, 200))
        cv2.rectangle(legend, (10, y - 15), (30, y + 5), color, -1)
        cv2.putText(legend, cls_name, (40, y), FONT, 0.5, (255, 255, 255), 1)

    return legend
