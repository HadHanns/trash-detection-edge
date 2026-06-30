#!/usr/bin/env python3
"""
test_pipeline.py
================
Unit tests untuk pipeline deteksi sampah di saluran air.

Mencakup:
- CPU inference setup (cpu_inference module)
- Manual slicing logic (generate_slices, transform_boxes)
- NMS utilities
- Preprocessing functions (coco_bbox_to_yolo, stratified_split)
- Visualization (draw_detections)
- Metrics computation (IoU, AP)

Jalankan:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --cov=src
"""

import sys
import os
import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Test: CPU Inference (simulasi edge computing sederhana)
# ─────────────────────────────────────────────────────────────────────────────

class TestCPUInference:
    """Tests for simple CPU inference utilities (cpu_inference module)."""

    def test_setup_cpu_inference_returns_config(self):
        """setup_cpu_inference should return a config dict with required keys."""
        from src.cpu_inference import setup_cpu_inference
        config = setup_cpu_inference(num_threads=2)
        assert "device" in config
        assert config["device"] == "cpu"
        assert "num_threads" in config
        assert "cuda_disabled" in config
        assert config["cuda_disabled"] is True

    def test_setup_cpu_inference_sets_threads(self):
        """setup_cpu_inference should apply the requested thread count."""
        from src.cpu_inference import setup_cpu_inference
        setup_cpu_inference(num_threads=2)
        assert torch.get_num_threads() == 2

    def test_setup_cpu_inference_no_threads(self):
        """setup_cpu_inference with num_threads=None should not crash."""
        from src.cpu_inference import setup_cpu_inference
        config = setup_cpu_inference(num_threads=None)
        assert config["device"] == "cpu"

    def test_setup_cpu_inference_disables_cuda(self):
        """setup_cpu_inference should set CUDA_VISIBLE_DEVICES to empty string."""
        from src.cpu_inference import setup_cpu_inference
        setup_cpu_inference()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_measure_inference_returns_stats(self):
        """measure_inference should return all required statistical keys."""
        from src.cpu_inference import measure_inference
        stats = measure_inference(lambda: None, n_runs=5, warmup_runs=1)

        required_keys = [
            "n_runs", "mean_latency_ms", "median_latency_ms",
            "min_latency_ms", "max_latency_ms", "p90_latency_ms",
            "mean_fps", "all_latencies_ms"
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_measure_inference_correct_n_runs(self):
        """measure_inference should record exactly n_runs latencies."""
        from src.cpu_inference import measure_inference
        stats = measure_inference(lambda: None, n_runs=8, warmup_runs=2)
        assert stats["n_runs"] == 8
        assert len(stats["all_latencies_ms"]) == 8

    def test_measure_inference_fps_positive(self):
        """FPS should be a positive number."""
        from src.cpu_inference import measure_inference
        stats = measure_inference(lambda: None, n_runs=5, warmup_runs=1)
        assert stats["mean_fps"] > 0

    def test_measure_inference_latency_ordering(self):
        """min <= median <= p90 <= max."""
        from src.cpu_inference import measure_inference
        import time
        stats = measure_inference(lambda: time.sleep(0.001), n_runs=10, warmup_runs=2)
        assert stats["min_latency_ms"] <= stats["median_latency_ms"]
        assert stats["median_latency_ms"] <= stats["p90_latency_ms"]
        assert stats["p90_latency_ms"] <= stats["max_latency_ms"]

    def test_timer_context_manager(self):
        """timer() context manager should measure elapsed time."""
        from src.cpu_inference import timer
        import time
        with timer() as t:
            time.sleep(0.02)  # 20ms
        assert t.elapsed_ms >= 15.0
        assert t.elapsed_ms < 200.0

    def test_compare_methods_returns_all_methods(self):
        """compare_methods should return stats for all provided methods."""
        from src.cpu_inference import compare_methods
        methods = {
            "method_a": lambda: None,
            "method_b": lambda: None,
        }
        results = compare_methods(methods, n_runs=3, warmup_runs=1)
        assert "method_a" in results
        assert "method_b" in results
        assert results["method_a"]["mean_fps"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Manual Slicing (generate_slices & transform_boxes)
# ─────────────────────────────────────────────────────────────────────────────

class TestManualSlicing:
    """Tests for slice generation and coordinate transforms."""

    def test_generate_slices_small_image(self):
        """Small image <= slice_size should return exactly 1 slice."""
        from src.inference_manual import generate_slices
        slices = generate_slices(480, 640, slice_height=640, slice_width=640)
        assert len(slices) == 1
        assert slices[0] == (0, 0, 640, 480)

    def test_generate_slices_large_image(self):
        """Large image should return multiple slices."""
        from src.inference_manual import generate_slices
        slices = generate_slices(1280, 1920, slice_height=640, slice_width=640, overlap_ratio=0.2)
        assert len(slices) > 1
        for (x1, y1, x2, y2) in slices:
            assert x1 >= 0 and y1 >= 0
            assert x2 <= 1920 and y2 <= 1280
            assert x2 > x1 and y2 > y1

    def test_generate_slices_overlap(self):
        """Slices should have correct overlap (stride = size * (1 - overlap))."""
        from src.inference_manual import generate_slices
        slices = generate_slices(640, 1280, 640, 640, overlap_ratio=0.2)
        assert len(slices) >= 2
        # stride = 640 * 0.8 = 512
        assert slices[1][0] == 512

    def test_generate_slices_covers_full_image(self):
        """All pixels should be covered by at least one slice."""
        from src.inference_manual import generate_slices
        H, W = 800, 1200
        slices = generate_slices(H, W, 640, 640, overlap_ratio=0.2)

        coverage = np.zeros((H, W), dtype=bool)
        for (x1, y1, x2, y2) in slices:
            coverage[y1:y2, x1:x2] = True
        assert coverage.all(), "Some pixels not covered by any slice!"

    def test_transform_boxes_to_original(self):
        """Box coordinates should transform correctly from patch to image."""
        from src.inference_manual import transform_boxes_to_original
        boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        slice_coords = (100, 200, 740, 840)
        transformed = transform_boxes_to_original(boxes, slice_coords, 1.0, 1.0)

        assert transformed[0, 0] == pytest.approx(110.0)  # 10 + 100
        assert transformed[0, 1] == pytest.approx(220.0)  # 20 + 200
        assert transformed[0, 2] == pytest.approx(200.0)  # 100 + 100
        assert transformed[0, 3] == pytest.approx(400.0)  # 200 + 200

    def test_transform_boxes_with_scale(self):
        """Box coordinates should account for resize scale factor."""
        from src.inference_manual import transform_boxes_to_original
        scale_x = scale_y = 400 / 640
        boxes = np.array([[64, 64, 320, 320]], dtype=np.float32)
        slice_coords = (0, 0, 400, 400)
        transformed = transform_boxes_to_original(boxes, slice_coords, scale_x, scale_y)

        expected_x1 = 64 * scale_x
        expected_y1 = 64 * scale_y
        assert transformed[0, 0] == pytest.approx(expected_x1, rel=0.01)
        assert transformed[0, 1] == pytest.approx(expected_y1, rel=0.01)

    def test_transform_empty_boxes(self):
        """Empty input should return empty output."""
        from src.inference_manual import transform_boxes_to_original
        boxes = np.empty((0, 4), dtype=np.float32)
        result = transform_boxes_to_original(boxes, (0, 0, 640, 640))
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: NMS Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestNMSUtils:
    """Tests for NMS functions."""

    def test_standard_nms_removes_duplicates(self):
        """NMS should remove highly overlapping boxes."""
        from src.utils.nms_utils import standard_nms

        boxes = np.array([
            [100, 100, 200, 200],
            [101, 101, 201, 201],  # near-duplicate
            [300, 300, 400, 400],  # separate box
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        class_ids = np.array([0, 0, 0], dtype=np.int32)

        filtered_boxes, filtered_scores, filtered_cls = standard_nms(
            boxes, scores, class_ids, iou_threshold=0.5
        )
        assert len(filtered_boxes) == 2

    def test_standard_nms_different_classes(self):
        """NMS should not suppress boxes of different classes."""
        from src.utils.nms_utils import standard_nms

        boxes = np.array([
            [100, 100, 200, 200],
            [100, 100, 200, 200],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)

        filtered_boxes, _, _ = standard_nms(boxes, scores, class_ids, iou_threshold=0.5)
        assert len(filtered_boxes) == 2

    def test_standard_nms_empty_input(self):
        """NMS with empty input should return empty arrays."""
        from src.utils.nms_utils import standard_nms

        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        class_ids = np.empty(0, dtype=np.int32)

        f_boxes, f_scores, f_cls = standard_nms(boxes, scores, class_ids)
        assert len(f_boxes) == 0
        assert len(f_scores) == 0
        assert len(f_cls) == 0

    def test_standard_nms_score_threshold(self):
        """Boxes below score threshold should be removed."""
        from src.utils.nms_utils import standard_nms

        boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
        scores = np.array([0.9, 0.1], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int32)

        f_boxes, f_scores, _ = standard_nms(boxes, scores, class_ids, score_threshold=0.25)
        assert len(f_boxes) == 1
        assert f_scores[0] == pytest.approx(0.9)

    def test_filter_small_boxes(self):
        """Degenerate tiny boxes should be filtered."""
        from src.utils.nms_utils import filter_small_boxes

        boxes = np.array([
            [0, 0, 100, 100],
            [0, 0, 2, 2],       # too small (area=4)
            [0, 0, 50, 50],
        ], dtype=np.float32)
        scores = np.ones(3, dtype=np.float32)
        class_ids = np.zeros(3, dtype=np.int32)

        f_boxes, f_scores, _ = filter_small_boxes(boxes, scores, class_ids, min_area=25.0)
        assert len(f_boxes) == 2

    def test_soft_nms_keeps_both_boxes(self):
        """Soft-NMS should retain boxes above score threshold."""
        from src.utils.nms_utils import soft_nms

        boxes = np.array([
            [100, 100, 200, 200],
            [110, 110, 210, 210],  # overlapping
            [400, 400, 500, 500],  # separate
        ], dtype=np.float32)
        scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0, 0], dtype=np.int32)

        f_boxes, f_scores, _ = soft_nms(
            boxes, scores, class_ids, sigma=0.5, score_threshold=0.25
        )
        assert len(f_boxes) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Test: Preprocessing Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for data preprocessing functions."""

    def test_coco_bbox_to_yolo_basic(self):
        """COCO to YOLO bbox conversion should be correct."""
        from data.preprocess import coco_bbox_to_yolo

        result = coco_bbox_to_yolo([25, 25, 50, 50], 100, 100)
        assert result is not None
        x_c, y_c, w, h = result
        assert x_c == pytest.approx(0.5)
        assert y_c == pytest.approx(0.5)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_coco_bbox_to_yolo_corner(self):
        """Box at top-left corner."""
        from data.preprocess import coco_bbox_to_yolo

        result = coco_bbox_to_yolo([0, 0, 100, 100], 640, 480)
        assert result is not None
        x_c, y_c, w, h = result
        assert x_c == pytest.approx(100 / (2 * 640))
        assert y_c == pytest.approx(100 / (2 * 480))
        assert w == pytest.approx(100 / 640)
        assert h == pytest.approx(100 / 480)

    def test_coco_bbox_to_yolo_clamps_values(self):
        """All output values should be in [0, 1]."""
        from data.preprocess import coco_bbox_to_yolo

        result = coco_bbox_to_yolo([600, 470, 100, 100], 640, 480)
        if result is not None:
            x_c, y_c, w, h = result
            assert 0 <= x_c <= 1
            assert 0 <= y_c <= 1
            assert 0 <= w <= 1
            assert 0 <= h <= 1

    def test_coco_bbox_to_yolo_invalid_zero_area(self):
        """Zero-area box should return None."""
        from data.preprocess import coco_bbox_to_yolo
        result = coco_bbox_to_yolo([50, 50, 0, 0], 640, 480)
        assert result is None

    def test_map_category(self):
        """Category mapping should correctly match actual TACO fine-grained category names."""
        from data.preprocess import map_category, DEFAULT_CLASS_MAPPING

        # Nama kategori TACO yang sebenarnya (case-insensitive)
        assert map_category("Clear plastic bottle", DEFAULT_CLASS_MAPPING) == 0    # plastic_bottle
        assert map_category("Plastic bottle cap", DEFAULT_CLASS_MAPPING) == 0      # plastic_bottle
        assert map_category("Plastic film", DEFAULT_CLASS_MAPPING) == 1            # plastic_bag
        assert map_category("Other plastic wrapper", DEFAULT_CLASS_MAPPING) == 1   # plastic_bag
        assert map_category("Drink carton", DEFAULT_CLASS_MAPPING) == 2            # carton
        assert map_category("Normal paper", DEFAULT_CLASS_MAPPING) == 2            # carton/paper
        assert map_category("Disposable plastic cup", DEFAULT_CLASS_MAPPING) == 3  # cup
        assert map_category("Styrofoam cup", DEFAULT_CLASS_MAPPING) == 4           # styrofoam
        assert map_category("Styrofoam piece", DEFAULT_CLASS_MAPPING) == 4         # styrofoam
        assert map_category("Cigarette", DEFAULT_CLASS_MAPPING) == 5               # cigarette
        assert map_category("Unlabeled litter", DEFAULT_CLASS_MAPPING) == 6        # other_trash
        assert map_category("Broken glass", DEFAULT_CLASS_MAPPING) == 6            # other_trash
        assert map_category("Plastic straw", DEFAULT_CLASS_MAPPING) == 6           # other_trash
        assert map_category("Pop tab", DEFAULT_CLASS_MAPPING) == 6                 # other_trash
        assert map_category("Unknown category xyz", DEFAULT_CLASS_MAPPING) == -1

    def test_stratified_split_ratios(self):
        """Stratified split should approximate given ratios."""
        from data.preprocess import stratified_split

        image_ids = list(range(100))
        image_to_classes = {i: {i % 7} for i in image_ids}

        train, val, test = stratified_split(
            image_ids, image_to_classes, train_ratio=0.7, val_ratio=0.2, seed=42
        )

        total = len(train) + len(val) + len(test)
        assert total == 100
        assert 65 <= len(train) <= 75
        assert 15 <= len(val) <= 25

    def test_stratified_split_no_overlap(self):
        """No image ID should appear in multiple splits."""
        from data.preprocess import stratified_split

        image_ids = list(range(200))
        image_to_classes = {i: {i % 7} for i in image_ids}
        train, val, test = stratified_split(image_ids, image_to_classes)

        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Metrics Computation
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for detection evaluation metrics."""

    def test_compute_iou_identical_boxes(self):
        """IoU of identical boxes should be 1.0."""
        from src.utils.metrics import compute_iou
        box = np.array([100, 100, 200, 200], dtype=np.float32)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        from src.utils.metrics import compute_iou
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([200, 200, 300, 300], dtype=np.float32)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_compute_iou_partial_overlap(self):
        """IoU of half-overlapping boxes should be 1/3."""
        from src.utils.metrics import compute_iou
        box1 = np.array([0, 0, 100, 100], dtype=np.float32)
        box2 = np.array([50, 0, 150, 100], dtype=np.float32)
        assert compute_iou(box1, box2) == pytest.approx(1 / 3, rel=0.01)

    def test_compute_iou_matrix_shape(self):
        """IoU matrix should have shape [M, N]."""
        from src.utils.metrics import compute_iou_matrix
        gt_boxes = np.random.rand(4, 4).astype(np.float32)
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        pred_boxes = np.random.rand(6, 4).astype(np.float32)
        pred_boxes[:, 2:] += pred_boxes[:, :2]

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        assert iou_mat.shape == (4, 6)

    def test_compute_average_precision_perfect(self):
        """Perfect precision-recall curve should give AP=1.0."""
        from src.utils.metrics import compute_average_precision
        recalls = np.array([0.0, 0.5, 1.0])
        precisions = np.array([1.0, 1.0, 1.0])
        ap = compute_average_precision(recalls, precisions)
        assert ap == pytest.approx(1.0)

    def test_compute_average_precision_zero(self):
        """Zero precision everywhere should give AP=0.0."""
        from src.utils.metrics import compute_average_precision
        recalls = np.array([0.0, 0.5, 1.0])
        precisions = np.array([0.0, 0.0, 0.0])
        ap = compute_average_precision(recalls, precisions)
        assert ap == pytest.approx(0.0)

    def test_match_predictions_perfect_match(self):
        """Perfect prediction should give TP=1, FP=0 for each GT."""
        from src.utils.metrics import match_predictions_to_gt

        gt_boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        gt_cls = np.array([0], dtype=np.int32)
        pred_boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        pred_scores = np.array([0.9], dtype=np.float32)
        pred_cls = np.array([0], dtype=np.int32)

        results = match_predictions_to_gt(
            gt_boxes, gt_cls, pred_boxes, pred_scores, pred_cls, iou_threshold=0.5
        )
        assert results[0]["tp"][0] == 1
        assert results[0]["fp"][0] == 0
        assert results[0]["n_gt"] == 1

    def test_match_predictions_no_overlap(self):
        """Non-overlapping prediction should give FP=1."""
        from src.utils.metrics import match_predictions_to_gt

        gt_boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        gt_cls = np.array([0], dtype=np.int32)
        pred_boxes = np.array([[500, 500, 600, 600]], dtype=np.float32)
        pred_scores = np.array([0.9], dtype=np.float32)
        pred_cls = np.array([0], dtype=np.int32)

        results = match_predictions_to_gt(
            gt_boxes, gt_cls, pred_boxes, pred_scores, pred_cls, iou_threshold=0.5
        )
        assert results[0]["fp"][0] == 1
        assert results[0]["tp"][0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Visualization
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    """Tests for visualization functions."""

    def test_draw_detections_returns_same_shape(self):
        """draw_detections should return image with same H x W."""
        from src.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = np.array([[50, 50, 200, 200], [300, 100, 500, 400]], dtype=np.float32)
        scores = np.array([0.9, 0.75], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int32)

        result = draw_detections(image, boxes, scores, class_ids)
        assert result.shape == image.shape

    def test_draw_detections_no_modify_original(self):
        """draw_detections should not modify the input image."""
        from src.utils.visualization import draw_detections

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        original = image.copy()
        boxes = np.array([[50, 50, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        _ = draw_detections(image, boxes, scores, class_ids)
        np.testing.assert_array_equal(image, original)

    def test_draw_detections_empty_boxes(self):
        """Empty detection list should return annotated image without boxes."""
        from src.utils.visualization import draw_detections

        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        class_ids = np.empty(0, dtype=np.int32)

        result = draw_detections(image, boxes, scores, class_ids)
        assert result.shape == image.shape

    def test_visualize_slices_returns_correct_shape(self):
        """Slice visualization should match input image dimensions."""
        from src.utils.visualization import visualize_slices
        from src.inference_manual import generate_slices

        image = np.zeros((640, 1280, 3), dtype=np.uint8)
        slices = generate_slices(640, 1280, 640, 640, overlap_ratio=0.2)
        result = visualize_slices(image, slices)
        assert result.shape == image.shape


# ─────────────────────────────────────────────────────────────────────────────
# Test: Integration (lightweight)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Lightweight integration tests (no model required)."""

    def test_slice_and_transform_pipeline(self):
        """Test full slice->transform->NMS pipeline with dummy data."""
        from src.inference_manual import generate_slices, transform_boxes_to_original
        from src.utils.nms_utils import standard_nms

        H, W = 800, 1200
        slices = generate_slices(H, W, 640, 640, overlap_ratio=0.2)
        assert len(slices) > 0

        all_boxes, all_scores, all_class_ids = [], [], []
        for slice_coords in slices[:3]:
            boxes = np.array([[50, 50, 150, 150], [200, 200, 350, 350]], dtype=np.float32)
            scores = np.array([0.9, 0.7], dtype=np.float32)
            cls_ids = np.array([0, 1], dtype=np.int32)

            boxes_orig = transform_boxes_to_original(boxes, slice_coords, 1.0, 1.0)
            all_boxes.append(boxes_orig)
            all_scores.append(scores)
            all_class_ids.append(cls_ids)

        merged_boxes = np.concatenate(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_class_ids = np.concatenate(all_class_ids)

        f_boxes, f_scores, f_cls = standard_nms(
            merged_boxes, merged_scores, merged_class_ids, iou_threshold=0.5
        )

        assert len(f_boxes) > 0
        assert (f_boxes[:, 0] >= 0).all()
        assert (f_boxes[:, 1] >= 0).all()

    def test_evaluator_basic_workflow(self):
        """DetectionEvaluator should handle a basic evaluation run."""
        from src.utils.metrics import DetectionEvaluator
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = DetectionEvaluator(
                class_names=["plastic_bottle", "plastic_bag"],
                output_dir=Path(tmp_dir)
            )

            gt_boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
            gt_cls = np.array([0, 1], dtype=np.int32)
            pred_boxes = np.array([[5, 5, 95, 95]], dtype=np.float32)
            pred_scores = np.array([0.9], dtype=np.float32)
            pred_cls = np.array([0], dtype=np.int32)

            evaluator.add_image_result(
                method="test_method",
                image_id="img_001",
                gt_boxes=gt_boxes,
                gt_class_ids=gt_cls,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_class_ids=pred_cls,
                inference_time_ms=100.0
            )

            report = evaluator.evaluate_method("test_method")
            assert report.n_images == 1
            assert report.mean_latency_ms == pytest.approx(100.0)
            assert 0 <= report.map50 <= 1.0

    def test_cpu_inference_setup_and_measure(self):
        """Integration: setup CPU inference then measure a dummy function."""
        from src.cpu_inference import setup_cpu_inference, measure_inference

        config = setup_cpu_inference(num_threads=2)
        assert config["device"] == "cpu"

        stats = measure_inference(lambda: None, n_runs=5, warmup_runs=1)
        assert stats["mean_latency_ms"] >= 0
        assert stats["mean_fps"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Configuration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
