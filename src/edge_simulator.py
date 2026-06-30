"""
edge_simulator.py — DEPRECATED
================================
File ini dipertahankan hanya untuk kompatibilitas impor.

Gunakan modul baru yang lebih sederhana:
    from src.cpu_inference import setup_cpu_inference, measure_inference

Tidak ada lagi kelas EdgeSimulator berbasis spesifikasi hardware.
Pendekatan penelitian ini adalah CPU-only inference tanpa asumsi perangkat spesifik.
"""

# Re-export dari modul baru untuk kompatibilitas
from src.cpu_inference import (
    setup_cpu_inference,
    measure_inference,
    compare_methods,
    timer,
)

__all__ = [
    "setup_cpu_inference",
    "measure_inference",
    "compare_methods",
    "timer",
]
