"""
train_direct.py
Script training minimal langsung menggunakan Ultralytics API.
Fix: gunakan exist_ok=True, tangkap error eksplisit, dan cek bobot dari path
     yang sebenarnya ditulis Ultralytics (bukan path yang kita kirim).
"""
import os
import sys
import shutil
import traceback
from pathlib import Path

# Bypass Polars CPU check
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

DATA_YAML  = Path("data/yolo/data.yaml")
MODEL      = "yolo11n.pt"
EPOCHS     = 5
BATCH      = 2
IMGSZ      = 416
DEVICE     = 0          # 0 = GPU pertama; ganti "cpu" jika tidak ada GPU
OUTPUT_DIR = Path("weights")


def main():
    print(f"[INFO] Data  : {DATA_YAML.resolve()}")
    print(f"[INFO] Model : {MODEL}")
    print(f"[INFO] Device: {DEVICE}")
    print()

    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml tidak ditemukan di: {DATA_YAML.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Load model
    print(f"[INFO] Loading {MODEL}...")
    model = YOLO(MODEL)
    print(f"[INFO] Model loaded OK")

    # --- Jalankan training ---
    print(f"[INFO] Mulai training ({EPOCHS} epoch, batch={BATCH}, imgsz={IMGSZ})...")
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            # Biarkan Ultralytics menentukan direktori output-nya sendiri
            # agar tidak terjadi konflik path
            exist_ok=True,
            amp=False,      # matikan AMP (kompatibilitas GTX 970)
            workers=0,      # 0 = hindari multiprocessing issue di Windows
            save=True,
            save_period=1,  # simpan checkpoint tiap epoch
            plots=False,    # matikan plot (tidak butuh matplotlib)
            verbose=True,
            patience=15,
        )
    except Exception as exc:
        print(f"\n[ERROR] Training gagal dengan exception:")
        traceback.print_exc()
        sys.exit(1)

    print("\n[INFO] Training selesai. Mencari file bobot...")

    # Cari best.pt dari trainer.save_dir (path yang Ultralytics benar-benar pakai)
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None

    # Fallback: cari di semua kemungkinan path
    candidates = []
    if save_dir:
        candidates.append(save_dir / "weights" / "best.pt")
        candidates.append(save_dir / "weights" / "last.pt")

    # Tambahkan fallback umum
    for p in Path(".").rglob("best.pt"):
        candidates.append(p)
    for p in Path(".").rglob("last.pt"):
        candidates.append(p)

    best_pt = None
    for c in candidates:
        if c.exists():
            best_pt = c
            print(f"[INFO] Bobot ditemukan: {best_pt.resolve()}")
            break

    if best_pt is None:
        print("[ERROR] Tidak ada file bobot yang ditemukan setelah training.")
        print("        Cek folder runs/ secara manual.")
        sys.exit(1)

    dest = OUTPUT_DIR / "best.pt"
    shutil.copy2(best_pt, dest)
    print(f"\n[SUCCESS] Bobot tersimpan di: {dest.resolve()}")
    print(f"[INFO] Siap digunakan untuk inference.")


if __name__ == "__main__":
    main()
