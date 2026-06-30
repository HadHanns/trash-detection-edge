"""
train_callback.py  —  Gunakan callback untuk manual save + diagnosa crash torch.save
"""
import os, sys, shutil, traceback
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from ultralytics import YOLO

DATA_YAML  = Path("data/yolo/data.yaml").resolve()
MODEL      = "yolo11n.pt"
EPOCHS     = 3
BATCH      = 2
IMGSZ      = 320
DEVICE     = 0
SAVE_DIR   = Path("C:/yolo_out/cb_run")
OUTPUT_DIR = Path("weights")

SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "weights").mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Callback: simpan state_dict setelah tiap epoch ─────────────────────────
def on_epoch_end(trainer):
    epoch = trainer.epoch + 1
    dest  = SAVE_DIR / "weights" / f"epoch{epoch}.pt"
    try:
        torch.save({"model": trainer.model.state_dict(), "epoch": epoch}, str(dest))
        print(f"\n[CALLBACK ✅] Epoch {epoch} weights saved → {dest}")
    except Exception:
        print(f"\n[CALLBACK ❌] torch.save gagal di epoch {epoch}:")
        traceback.print_exc()

def on_train_end(trainer):
    print(f"\n[CALLBACK] Training selesai. save_dir = {trainer.save_dir}")

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Torch version  : {torch.__version__}")
    print(f"[INFO] CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU            : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] data.yaml      : {DATA_YAML}")
    print()

    # Tes torch.save sederhana dulu
    print("[TEST] Menguji torch.save sederhana...")
    try:
        test_file = SAVE_DIR / "test_save.pt"
        dummy = {"a": torch.randn(10, 10)}
        torch.save(dummy, str(test_file))
        print(f"[TEST ✅] torch.save OK → {test_file}")
        test_file.unlink()
    except Exception:
        print("[TEST ❌] torch.save GAGAL:")
        traceback.print_exc()
        sys.exit(1)

    model = YOLO(MODEL)

    # Tambahkan callback manual
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.add_callback("on_train_end",       on_train_end)

    print("[INFO] Mulai training...")
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(SAVE_DIR.parent),
            name=SAVE_DIR.name,
            exist_ok=True,
            amp=False,
            workers=0,
            save=True,
            save_period=1,
            plots=False,
            verbose=True,
            patience=50,
        )
        print(f"\n[INFO] Training selesai. save_dir={getattr(results,'save_dir','N/A')}")
    except Exception:
        print("\n[ERROR] Exception saat model.train():")
        traceback.print_exc()

    # Cari file yang tersimpan
    print("\n[INFO] File yang tersimpan di SAVE_DIR:")
    for f in SAVE_DIR.rglob("*"):
        if f.is_file():
            print(f"       {f}")

    # Salin bobot terbaik ke weights/
    candidates = list(SAVE_DIR.rglob("best.pt")) + \
                 list(SAVE_DIR.rglob("last.pt")) + \
                 list(SAVE_DIR.rglob("epoch*.pt"))
    if candidates:
        best = candidates[0]
        dest = OUTPUT_DIR / "best.pt"
        shutil.copy2(best, dest)
        print(f"\n[SUCCESS] Bobot disalin ke: {dest.resolve()}")
    else:
        print("\n[ERROR] Tidak ada .pt ditemukan sama sekali.")
        sys.exit(1)

if __name__ == "__main__":
    main()
