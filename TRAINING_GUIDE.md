# 📖 Panduan Training & Penggunaan

> Dokumen ini menjelaskan langkah-langkah lengkap mulai dari persiapan dataset, training model,
> hingga menjalankan GUI interaktif.
>
> **Dataset:** Kaggle RFT Trash Detection (8 Kelas)
> **Model:** YOLOv11n | **Training:** 100 Epoch | **Inference:** CPU-only

---

## ✅ Status Pipeline

| # | Langkah | Status | Keterangan |
|---|---------|--------|------------|
| 1 | Persiapan Dataset | ✅ Selesai | Dataset Kaggle RFT tersedia di `data/datasets/` |
| 2 | Training Model | ✅ Selesai | 100 epoch, model tersimpan di `weights/best.pt` |
| 3 | Konversi Bobot | ✅ Selesai | `weights/best.pt` siap pakai |
| 4 | GUI Interaktif | ✅ Siap | Fitur YOLO, SAHI, Video, Polygon ROI |
| 5 | Evaluasi Benchmark | ⏳ Opsional | Jalankan `src/benchmark.py` untuk laporan |

---

## 🏷️ 8 Kelas yang Dideteksi

| ID | Nama Kelas | Keterangan |
|----|-----------|------------|
| 0 | `bottle` | Botol plastik |
| 1 | `grass` | Rumput |
| 2 | `branch` | Ranting / dahan |
| 3 | `milk-box` | Kotak susu / karton |
| 4 | `plastic-bag` | Kantong plastik |
| 5 | `plastic-garbage` | Sampah plastik umum |
| 6 | `ball` | Bola |
| 7 | `leaf` | Daun |

---

## 🏗️ Struktur File Penting

```
trash-detection-edge/
│
├── data/
│   └── datasets/             ← Dataset Kaggle RFT (train/valid/test)
│       ├── train/images/ & labels/
│       ├── valid/images/ & labels/
│       └── test/images/  & labels/
│
├── train_final.py            ← Script training utama
├── weights/
│   └── best.pt               ← Bobot hasil training 100 epoch ✅
│
├── src/
│   ├── gui.py                ← GUI interaktif utama ✅
│   ├── benchmark.py          ← Evaluasi YOLO vs YOLO+SAHI
│   ├── inference_sahi.py     ← Inference via SAHI
│   └── cpu_inference.py      ← Utilitas CPU-only inference
│
└── config/config.yaml        ← Konfigurasi proyek
```

---

## 🚀 Prasyarat

```powershell
# Di PowerShell, dari direktori proyek:
cd C:\Users\induk\Documents\trash-detection-edge

# Set environment variable (wajib di setiap sesi PowerShell baru)
$Env:POLARS_SKIP_CPU_CHECK = "1"
$Env:PYTHONIOENCODING     = "utf-8"
```

---

## LANGKAH 1 — Persiapan Dataset ✅ SELESAI

Dataset **Kaggle RFT Trash Detection** sudah tersedia di `data/datasets/`.

Struktur dataset yang diperlukan:
```
data/datasets/
├── train/
│   ├── images/   ← Gambar training
│   └── labels/   ← Anotasi YOLO format (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Jika perlu mengunduh ulang, download dari Kaggle dan sesuaikan struktur folder di atas.

---

## LANGKAH 2 — Training Model ✅ SELESAI

**File:** `train_final.py`

```powershell
$Env:POLARS_SKIP_CPU_CHECK = "1"
$Env:PYTHONIOENCODING     = "utf-8"
.\miniconda\python.exe train_final.py
```

**Konfigurasi training yang digunakan:**

```python
MODEL    = "yolo11n.pt"                        # Model base YOLOv11 nano
DATA     = "data/datasets/data.yaml"           # Dataset Kaggle RFT
EPOCHS   = 100                                 # Total epoch
BATCH    = 8                                   # Batch size
IMGSZ    = 640                                 # Resolusi input
SAVE_DIR = "C:/yolo_out/rft_run"              # Output training
```

**Hyperparameter:**
```python
lr0      = 0.01    # Learning rate awal
lrf      = 0.001   # Learning rate akhir (cosine decay)
cos_lr   = True    # Cosine LR scheduler
patience = 20      # Early stopping
```

**Output training:**
```
C:/yolo_out/rft_run/weights/
├── best.pt   ← Bobot terbaik (mAP50 tertinggi)
└── last.pt   ← Bobot epoch terakhir

weights/
└── best.pt   ← Disalin ke sini untuk dipakai GUI
```

---

## LANGKAH 3 — Jalankan GUI Interaktif ✅

**File:** `src/gui.py`

```powershell
$Env:POLARS_SKIP_CPU_CHECK = "1"
$Env:PYTHONIOENCODING     = "utf-8"
.\miniconda\python.exe src\gui.py
```

> Model default: `C:/yolo_out/rft_run/weights/best.pt`
> Atau gunakan custom path: `.\miniconda\python.exe src\gui.py --model weights\best.pt`

### Cara Menggunakan GUI

#### 📁 Deteksi Gambar
1. Klik **Upload Gambar** → pilih file `.jpg` / `.png`
2. Pilih mode: **🎯 YOLO-only** atau **🔲 YOLO + SAHI**
3. Atur **Slice Size** dan **Overlap** (untuk SAHI)
4. Atur **Confidence Threshold** (turunkan ke 0.10–0.15 jika kurang deteksi)
5. Klik **▶ Jalankan Deteksi**
6. Lihat hasil di tab **🎯 Hasil Deteksi**
7. Lihat grid slice di tab **🔲 Fragmentasi SAHI** (khusus mode SAHI)

#### 🎥 Deteksi Video
1. Klik **Upload Video** → pilih file `.mp4` / `.avi` / `.mov`
2. Pilih mode video: **🎯 YOLO-only** atau **🔲 YOLO+SAHI**
   > ⚠️ Mode SAHI lebih lambat karena memproses N patch per frame
3. Klik **▶ Play** untuk mulai inferensi real-time
4. Klik **⏸ Pause** / **■ Stop** untuk mengendalikan playback

#### 📐 Polygon ROI (Area Deteksi)
1. Upload gambar atau video terlebih dahulu
2. Klik **Tentukan ROI** di panel kiri
3. **Klik kiri** beberapa kali di gambar untuk membentuk batas area
4. **Klik kanan** untuk menutup poligon (minimal 3 titik)
5. Jalankan deteksi — hanya objek **di dalam area ROI** yang dihitung
6. Klik **Hapus ROI** untuk kembali ke deteksi penuh

---

## LANGKAH 4 — Evaluasi Benchmark (Opsional)

**File:** `src/benchmark.py`

```powershell
$Env:POLARS_SKIP_CPU_CHECK = "1"
$Env:PYTHONIOENCODING     = "utf-8"

# Evaluasi cepat (10 gambar)
.\miniconda\python.exe src\benchmark.py --model weights\best.pt --split test --max-images 10

# Evaluasi penuh untuk laporan skripsi
.\miniconda\python.exe src\benchmark.py --model weights\best.pt --split test --max-images 50 --runs 2
```

**Output:**
```
results/
├── benchmark_results.json   ← Hasil lengkap (mAP, latency, FPS, per-class AP)
└── benchmark_results.csv    ← Tabel ringkasan untuk laporan/skripsi
```

**Metrik yang diukur:**

| Metrik | YOLO-only | YOLO+SAHI |
|--------|:---------:|:---------:|
| mAP@0.5 | ✓ | ✓ |
| mAP@0.5:0.95 | ✓ | ✓ |
| Latency (ms) | ✓ | ✓ |
| FPS | ✓ | ✓ |
| Per-class AP | ✓ | ✓ |

---

## ⚙️ Perbandingan YOLO-only vs YOLO+SAHI

| Aspek | YOLO-only | YOLO+SAHI |
|-------|:---------:|:---------:|
| Input | Gambar penuh 640×640 | N patch 640×640 + overlap |
| Kecepatan | ✅ Cepat | ⚠️ Lebih lambat |
| Deteksi objek kecil | ⚠️ Lemah | ✅ Kuat |
| Cocok untuk | Real-time video | Akurasi tinggi / analisis |

---

## 🔧 Troubleshooting

### OOM (Out of Memory) saat Training
```python
# Di train_final.py, ubah:
BATCH = 4   # dari 8 ke 4
```

### Model kurang mendeteksi
Turunkan confidence threshold di GUI (slider ke 0.10) atau:
```powershell
.\miniconda\python.exe src\benchmark.py --model weights\best.pt --split test --conf 0.1
```

### Error UnicodeEncodeError (emoji di terminal)
```powershell
$Env:PYTHONIOENCODING = "utf-8"
```

### Error `POLARS_SKIP_CPU_CHECK`
```powershell
$Env:POLARS_SKIP_CPU_CHECK = "1"
```

---

## ⚡ Quick Start Lengkap

```powershell
# 1. Set environment
$Env:POLARS_SKIP_CPU_CHECK = "1"
$Env:PYTHONIOENCODING     = "utf-8"

# 2. (Jika perlu training ulang)
.\miniconda\python.exe train_final.py

# 3. Jalankan GUI
.\miniconda\python.exe src\gui.py

# 4. (Opsional) Benchmark
.\miniconda\python.exe src\benchmark.py --model weights\best.pt --split test --max-images 50
```

---

*Dokumentasi Skripsi — Sistem Deteksi Sampah Berbasis Deep Learning*
*Model: YOLOv11n | Dataset: Kaggle RFT (8 Kelas) | Metode: YOLO vs YOLO+SAHI + Polygon ROI*
*Terakhir diperbarui: 30 Juni 2026*
