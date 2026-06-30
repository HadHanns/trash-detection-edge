"""
convert_weights.py  —  Konversi best.pt (state_dict, nc=7) ke format YOLO standar
"""
import os, sys
from pathlib import Path
from copy import deepcopy

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import torch
from ultralytics.nn.tasks import DetectionModel

SRC  = Path("C:/yolo_out/full_run/weights/best.pt")
DEST = Path("weights/best.pt")

NAMES = {
    0: "plastic_bottle",
    1: "plastic_bag",
    2: "carton",
    3: "cup",
    4: "styrofoam",
    5: "cigarette",
    6: "other_trash",
}

def main():
    print(f"[INFO] Memuat checkpoint dari: {SRC}")
    ckpt = torch.load(str(SRC), map_location="cpu", weights_only=False)

    state_dict = ckpt.get("model", ckpt)
    nc    = ckpt.get("nc", 7)
    names = ckpt.get("names", NAMES)
    epoch = ckpt.get("epoch", 68)
    print(f"[INFO] nc={nc}, epoch={epoch}")

    # Buat DetectionModel dengan arsitektur yolo11n dan nc=7
    print("[INFO] Membangun model yolo11n dengan nc=7...")
    det_model = DetectionModel("yolo11n.yaml", nc=nc)
    det_model.names = names

    # Load bobot terlatih
    missing, unexpected = det_model.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Simpan dalam format standar Ultralytics (model object, bukan state_dict)
    m = deepcopy(det_model).half()
    m.nc    = nc
    m.names = names
    save_ckpt = {
        "epoch"      : epoch,
        "best_fitness": ckpt.get("fitness", 0.324),
        "model"      : m,
        "ema"        : None,
        "updates"    : 0,
        "optimizer"  : None,
        "train_args" : {"nc": nc, "imgsz": 640},
        "date"       : "2026-06-16",
        "version"    : "8.4.68",
    }

    DEST.parent.mkdir(exist_ok=True)
    torch.save(save_ckpt, str(DEST))
    sz = DEST.stat().st_size / 1024
    print(f"\n[SUCCESS] Disimpan ke: {DEST.resolve()} ({sz:.0f} KB)")

    # Verifikasi dengan YOLO loader
    print("\n[TEST] Memuat ulang dengan YOLO()...")
    try:
        from ultralytics import YOLO
        m = YOLO(str(DEST))
        print(f"[TEST ✅] nc={m.model.nc}, names={list(m.names.values())}")
    except Exception as e:
        print(f"[TEST ⚠] {e}")
        print("        (model tetap bisa dipakai via torch.load)")

if __name__ == "__main__":
    main()
