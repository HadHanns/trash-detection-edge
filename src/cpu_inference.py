#!/usr/bin/env python3
"""
cpu_inference.py
================
Utilitas sederhana untuk simulasi inferensi berbasis CPU.

Pendekatan simulasi edge computing dalam penelitian ini:
- Inference dijalankan di CPU (torch.device('cpu')) tanpa GPU
- Thread opsional dibatasi untuk mensimulasikan resource terbatas
- Metrik utama: latency (ms) dan FPS

Tidak ada asumsi hardware spesifik (RPi, Jetson, dll).
Simulasi ini hanya bertujuan menunjukkan perbandingan performa
antara YOLO saja dan YOLO+SAHI pada kondisi CPU-only.

Penggunaan:
    from src.cpu_inference import setup_cpu_inference, measure_inference

    setup_cpu_inference(num_threads=4)

    stats = measure_inference(lambda: model(image), n_runs=20)
    print(f"Mean latency: {stats['mean_latency_ms']:.1f} ms")
    print(f"Mean FPS    : {stats['mean_fps']:.2f}")
"""

import os
import time
import logging
import contextlib
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_cpu_inference(num_threads: Optional[int] = None) -> dict:
    """
    Konfigurasi inference berbasis CPU.

    Args:
        num_threads: Jumlah CPU thread yang digunakan.
                     None = biarkan PyTorch memilih otomatis.
                     Contoh: 2 atau 4 untuk mensimulasikan resource terbatas.

    Returns:
        Dict berisi konfigurasi yang diterapkan.
    """
    # Force CPU — nonaktifkan GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Disable gradient computation untuk inference
    torch.set_grad_enabled(False)

    # Set thread jika diminta
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # Tidak bisa diubah setelah thread parallel dimulai

    config = {
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "cuda_disabled": True,
    }

    logger.info(
        f"[CPU Inference] device=cpu | threads={config['num_threads']} | GPU=disabled"
    )
    return config


# ── Latency Measurement ───────────────────────────────────────────────────────

@contextlib.contextmanager
def timer():
    """
    Context manager sederhana untuk mengukur waktu eksekusi.

    Penggunaan:
        with timer() as t:
            result = model(image)
        print(f"{t.elapsed_ms:.1f} ms")
    """
    class _Timer:
        elapsed_ms: float = 0.0

    t = _Timer()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (time.perf_counter() - t0) * 1000.0


def measure_inference(
    model_fn: Callable,
    n_runs: int = 20,
    warmup_runs: int = 3,
) -> dict:
    """
    Ukur latency dan FPS dari sebuah fungsi inference.

    Args:
        model_fn : Callable tanpa argumen yang menjalankan satu inference.
        n_runs   : Jumlah run yang diukur.
        warmup_runs: Jumlah run pemanasan (tidak dihitung).

    Returns:
        Dict berisi:
          - mean_latency_ms
          - median_latency_ms
          - min_latency_ms
          - max_latency_ms
          - p90_latency_ms
          - mean_fps
          - all_latencies_ms
    """
    # Warmup
    logger.info(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        model_fn()

    # Timed runs
    logger.info(f"Measuring ({n_runs} runs)...")
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    mean_lat = sum(latencies) / n
    median_lat = latencies_sorted[n // 2]
    p90_lat = latencies_sorted[int(0.9 * n)]
    mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    results = {
        "n_runs": n_runs,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "p90_latency_ms": round(p90_lat, 2),
        "mean_fps": round(mean_fps, 2),
        "all_latencies_ms": [round(x, 2) for x in latencies],
    }

    logger.info(
        f"\n[Benchmark Results]\n"
        f"  Runs        : {n_runs}\n"
        f"  Mean latency: {results['mean_latency_ms']:.1f} ms\n"
        f"  Median      : {results['median_latency_ms']:.1f} ms\n"
        f"  P90         : {results['p90_latency_ms']:.1f} ms\n"
        f"  Min / Max   : {results['min_latency_ms']:.1f} / {results['max_latency_ms']:.1f} ms\n"
        f"  Mean FPS    : {results['mean_fps']:.2f}\n"
    )

    return results


def compare_methods(
    methods: dict,
    n_runs: int = 20,
    warmup_runs: int = 3,
) -> dict:
    """
    Bandingkan latency dari beberapa metode inference.

    Args:
        methods: Dict {"nama_metode": callable_fn, ...}
        n_runs, warmup_runs: Lihat measure_inference()

    Returns:
        Dict {"nama_metode": benchmark_stats, ...}

    Contoh:
        results = compare_methods({
            "YOLO_only":  lambda: predictor.predict_full_image(img),
            "YOLO+SAHI":  lambda: predictor.predict(img),
        })
    """
    all_results = {}
    for name, fn in methods.items():
        logger.info(f"\n--- Measuring: {name} ---")
        stats = measure_inference(fn, n_runs=n_runs, warmup_runs=warmup_runs)
        all_results[name] = stats

    # Print comparison table
    logger.info("\n" + "=" * 55)
    logger.info(f"{'Method':<20} {'Mean(ms)':>10} {'FPS':>8} {'P90(ms)':>10}")
    logger.info("=" * 55)
    for name, stats in all_results.items():
        logger.info(
            f"{name:<20} {stats['mean_latency_ms']:>10.1f} "
            f"{stats['mean_fps']:>8.2f} {stats['p90_latency_ms']:>10.1f}"
        )
    logger.info("=" * 55)

    return all_results
