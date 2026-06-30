"""
train_debug.py  —  Debug script untuk menangkap error saving checkpoint
"""
import os, sys, shutil, traceback
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
# Paksa encoding UTF-8 agar tidak ada UnicodeError saat menulis log
os.environ["PYTHONIOENCODING"] = "utf-8"

from ultralytics import YOLO

DATA_YAML  = Path("data/yolo/data.yaml").resolve()
MODEL      = "yolo11n.pt"
EPOCHS     = 3
BATCH      = 2
IMGSZ      = 320        # lebih kecil lagi untuk mempercepat
DEVICE     = 0
# Gunakan path ABSOLUT langsung di root drive, hindari path panjang
OUT_ROOT   = Path("C:/yolo_out")
RUN_NAME   = "run1"

def main():
    print(f"[INFO] data.yaml : {DATA_YAML}")
    print(f"[INFO] out_root  : {OUT_ROOT}")

    if not DATA_YAML.exists():
        print("[ERROR] data.yaml tidak ditemukan!"); sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    weights_dir = OUT_ROOT / RUN_NAME / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)
    print("[INFO] Model loaded. Memulai training...")

    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(OUT_ROOT),   # path absolut pendek
            name=RUN_NAME,
            exist_ok=True,
            amp=False,
            workers=0,
            save=True,
            save_period=1,
            plots=False,
            verbose=True,
            patience=50,             # jangan early-stop
        )
        print(f"\n[INFO] results.save_dir = {getattr(results, 'save_dir', 'N/A')}")
    except Exception:
        print("\n[ERROR] Exception saat training:")
        traceback.print_exc()
        sys.exit(1)

    # Cari best.pt/last.pt
    found = list(OUT_ROOT.rglob("best.pt")) + list(OUT_ROOT.rglob("last.pt"))
    if not found:
        print("[ERROR] Tidak ada .pt ditemukan di", OUT_ROOT)
        sys.exit(1)

    best = found[0]
    dest = Path("weights/best.pt")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(best, dest)
    print(f"\n[SUCCESS] Bobot disalin ke: {dest.resolve()}")

if __name__ == "__main__":
    main()
