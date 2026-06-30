"""
train_final.py  —  YOLO Training (100 epoch) dengan resume support
- Dataset : Kaggle RFT (8 kelas: bottle, grass, branch, milk-box,
             plastic-bag, plastic-garbage, ball, leaf)
- Save last.pt tiap epoch (full checkpoint: model + optimizer + epoch)
- Jika last.pt sudah ada → otomatis RESUME dari epoch terakhir
- Internal Ultralytics save=False (hindari crash), kita save manual via callback
"""
import os, sys, shutil, traceback
from pathlib import Path
from copy import deepcopy
from datetime import datetime

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["PYTHONIOENCODING"]      = "utf-8"

import torch
from ultralytics import YOLO

# ── Konfigurasi ─────────────────────────────────────────────────────────────
DATA_YAML  = Path("data/datasets/RFT.yaml").resolve()  # ← dataset Kaggle RFT
MODEL      = "yolo11n.pt"
EPOCHS     = 100
BATCH      = 4       # GTX 970 4GB; turunkan ke 2 jika OOM
IMGSZ      = 640
DEVICE     = 0
SAVE_DIR   = Path("C:/yolo_out/rft_run")
OUTPUT_DIR = Path("weights")

SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "weights").mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

best_fitness = 0.0
best_pt_path = SAVE_DIR / "weights" / "best.pt"
last_pt_path = SAVE_DIR / "weights" / "last.pt"

# ── Info dataset ─────────────────────────────────────────────────────────────
print(f"[INFO] Dataset : Kaggle RFT (8 kelas)")
print(f"[INFO] Kelas   : bottle, grass, branch, milk-box, plastic-bag, plastic-garbage, ball, leaf")


# ── Callback: simpan FULL checkpoint tiap epoch (model + optimizer) ──────────
def save_on_epoch_end(trainer):
    """
    Simpan checkpoint lengkap setelah epoch selesai.
    Menyimpan optimizer state agar training bisa di-resume kapan saja.
    """
    global best_fitness

    epoch = trainer.epoch + 1
    try:
        # ── Simpan last.pt (full checkpoint untuk resume) ──────────────────
        ckpt = {
            "epoch"      : epoch,
            "model"      : deepcopy(trainer.model).cpu(),   # full model object
            "optimizer"  : trainer.optimizer.state_dict()   # ← simpan optimizer!
                           if trainer.optimizer else None,
            "train_args" : vars(trainer.args),
            "date"       : datetime.now().isoformat(),
            "nc"         : trainer.model.nc,
            "names"      : trainer.model.names,
        }
        torch.save(ckpt, str(last_pt_path))
        print(f"\n[SAVE ✅] last.pt tersimpan (epoch {epoch}/{EPOCHS})")

        # ── Update best.pt berdasarkan fitness (mAP50) ────────────────────
        fitness = getattr(trainer, "fitness", None)
        if fitness is None and hasattr(trainer, "metrics"):
            fitness = trainer.metrics.get("metrics/mAP50(B)", 0.0)

        if fitness is not None and fitness >= best_fitness:
            best_fitness = fitness
            shutil.copy2(last_pt_path, best_pt_path)
            print(f"[SAVE ✅] best.pt diperbarui (fitness={fitness:.4f})")

    except Exception:
        print(f"\n[SAVE ❌] Gagal menyimpan checkpoint epoch {epoch}:")
        traceback.print_exc()


def on_train_end(trainer):
    epoch = trainer.epoch + 1
    print(f"\n[INFO] Training selesai di epoch {epoch}/{EPOCHS}")
    print(f"[INFO] last.pt : {last_pt_path}")
    print(f"[INFO] best.pt : {best_pt_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Deteksi apakah bisa RESUME ─────────────────────────────────────────
    can_resume = last_pt_path.exists()

    if can_resume:
        # Baca epoch terakhir yang tersimpan
        try:
            ckpt    = torch.load(str(last_pt_path), map_location="cpu", weights_only=False)
            last_ep = ckpt.get("epoch", "?")
            print(f"[RESUME] ✅ Checkpoint ditemukan: last.pt (epoch {last_ep})")
            print(f"[RESUME] Training akan lanjut dari epoch {last_ep} → {EPOCHS}")
        except Exception:
            print("[RESUME] ⚠️  Gagal baca last.pt, mulai dari awal.")
            can_resume = False

    if not can_resume:
        print(f"[INFO] Model  : {MODEL}")
        print(f"[INFO] Mulai training dari awal (epoch 1/{EPOCHS})")

    print(f"[INFO] Data   : {DATA_YAML}")
    print(f"[INFO] Epochs : {EPOCHS}, Batch: {BATCH}, ImgSz: {IMGSZ}")
    print(f"[INFO] Device : {DEVICE}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────
    if can_resume:
        # Load dari last.pt sebagai starting point (bukan resume= karena kita
        # pakai save=False; kita handle sendiri lewat callback)
        model = YOLO(str(last_pt_path))
        print("[RESUME] Model dimuat dari last.pt")
    else:
        model = YOLO(MODEL)

    model.add_callback("on_train_epoch_end", save_on_epoch_end)
    model.add_callback("on_train_end",       on_train_end)

    print("[INFO] Mulai training (save=False → kita handle checkpoint manual)...")
    try:
        results = model.train(
            data       = str(DATA_YAML),
            epochs     = EPOCHS,
            batch      = BATCH,
            imgsz      = IMGSZ,
            device     = DEVICE,
            project    = str(SAVE_DIR.parent),
            name       = SAVE_DIR.name,
            exist_ok   = True,
            amp        = False,
            workers    = 0,
            save       = False,      # ← matikan internal save (penyebab crash)
            save_period= 0,
            plots      = False,
            verbose    = True,
            patience   = 20,
            lr0        = 0.01,
            lrf        = 0.001,
            cos_lr     = True,
        )
    except Exception:
        print("\n[ERROR] Exception saat model.train():")
        traceback.print_exc()

    # ── Verifikasi bobot ────────────────────────────────────────────────────
    print("\n[INFO] Mencari file bobot...")
    for f in [best_pt_path, last_pt_path]:
        if f.exists():
            print(f"       ✅ {f} ({f.stat().st_size/1024:.0f} KB)")
        else:
            print(f"       ❌ {f} tidak ditemukan")

    # ── Salin ke weights/ proyek ────────────────────────────────────────────
    src = best_pt_path if best_pt_path.exists() else last_pt_path
    if src.exists():
        dest = OUTPUT_DIR / "best.pt"
        shutil.copy2(src, dest)
        print(f"\n[SUCCESS] Bobot disalin ke: {dest.resolve()}")
    else:
        print("\n[ERROR] Tidak ada bobot yang tersimpan.")
        sys.exit(1)


if __name__ == "__main__":
    main()
