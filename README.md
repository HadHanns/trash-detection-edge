# 🗑️ Trash Detection — YOLOv11 + SAHI + Polygon ROI
## Sistem Deteksi Sampah Berbasis Deep Learning | CPU-only Edge Simulation

> **Skripsi** — Deteksi Sampah di Lingkungan Terbuka Menggunakan YOLOv11n, SAHI (Slicing Aided Hyper Inference), dan GUI Interaktif  
> Dataset: **Kaggle RFT Trash Detection (8 Kelas)** | Training: 100 Epoch | Inference: CPU-only

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan sistem deteksi sampah skala kecil (botol, kantong plastik, daun, dll.) pada gambar maupun video. Fokus utama penelitian adalah **membandingkan performa YOLO-only vs. YOLO+SAHI**, dilengkapi dengan GUI interaktif untuk demonstrasi langsung, termasuk fitur **Polygon ROI** agar deteksi dapat dibatasi pada area tertentu.

| Komponen | Detail |
|----------|--------|
| **Model** | YOLOv11n (Ultralytics ≥ 8.3) |
| **Dataset** | Kaggle RFT Trash Detection — 8 Kelas |
| **Training** | 100 Epoch, Batch 8, Imgsz 640, CPU/GPU |
| **SAHI** | Slice 640×640, Overlap 20%, Postprocess: NMS |
| **Inference** | CPU-only (`torch.device('cpu')`) |
| **GUI** | Tkinter — Gambar & Video, YOLO vs YOLO+SAHI, Polygon ROI |
| **Metrik** | mAP@0.5, FPS, Latency (ms) — YOLO vs YOLO+SAHI |

---

## 🏷️ Label Kelas (8 Kelas)

| ID | Nama | Keterangan |
|----|------|------------|
| 0 | `bottle` | Botol plastik |
| 1 | `grass` | Rumput |
| 2 | `branch` | Ranting / dahan |
| 3 | `milk-box` | Kotak susu / dus karton |
| 4 | `plastic-bag` | Kantong plastik |
| 5 | `plastic-garbage` | Sampah plastik umum |
| 6 | `ball` | Bola |
| 7 | `leaf` | Daun |

---

## 🏗️ Struktur Proyek

```
trash-detection-edge/
├── config/
│   └── config.yaml              # Konfigurasi utama (SAHI, training)
├── data/
│   ├── datasets/                # Dataset Kaggle RFT (train/val/test)
│   └── yolo/                    # [Generated] Dataset YOLO format
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       └── data.yaml
├── src/
│   ├── gui.py                   # 🖥️  GUI Interaktif Utama
│   ├── cpu_inference.py         # Utilitas inferensi CPU-only
│   ├── trainer.py               # Fine-tuning YOLOv11n
│   ├── inference_sahi.py        # Inference via SAHI
│   └── utils/
│       ├── metrics.py           # mAP, FPS, latency, perbandingan
│       ├── visualization.py     # Bbox, side-by-side, plots
│       └── nms_utils.py         # NMS, Soft-NMS, WBF
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_inference_evaluation.ipynb
├── weights/
│   └── best.pt                  # Model hasil training 100 epoch
├── results/
│   ├── metrics/                 # CSV benchmark results
│   └── visualizations/          # Output detection images
├── tests/
│   └── test_pipeline.py
├── train_final.py               # Script training utama
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Menggunakan Miniconda (sudah tersedia di proyek)
.\miniconda\python.exe -m pip install -r requirements.txt
```

### 2. Siapkan Dataset

Dataset sudah tersedia di folder `data/datasets/` (dari Kaggle RFT Trash Detection).  
Pastikan struktur folder sesuai dengan format YOLO:
```
data/datasets/
├── train/images/ & labels/
├── valid/images/ & labels/
└── test/images/  & labels/
```

### 3. Training Model

```bash
.\miniconda\python.exe train_final.py
```

Model terbaik akan tersimpan di `C:/yolo_out/rft_run/weights/best.pt`.

### 4. Jalankan GUI

```bash
.\miniconda\python.exe src\gui.py

# Atau dengan path model custom
.\miniconda\python.exe src\gui.py --model weights/best.pt
```

---

## 🖥️ Fitur GUI Interaktif

GUI (`src/gui.py`) menyediakan antarmuka visual lengkap untuk demonstrasi dan pengujian sistem deteksi.

### Panel Kiri — Kontrol

| Seksi | Fungsi |
|-------|--------|
| **📁 GAMBAR** | Upload gambar, pilih mode deteksi (YOLO/SAHI), jalankan deteksi |
| **🎥 VIDEO** | Upload video, pilih mode deteksi, Play/Pause/Stop dengan inferensi real-time |
| **📐 AREA DETEKSI (ROI)** | Gambar poligon ROI interaktif — hanya deteksi dalam area tertentu |
| **🔧 PENGATURAN** | Slice size, overlap ratio, confidence threshold |
| **📊 HASIL** | Statistik (mode, jumlah deteksi, latency, FPS, patches) + daftar deteksi |

### Panel Kanan — Tab Visualisasi

| Tab | Konten |
|-----|--------|
| **🖼️ Gambar Asli** | Tampilan gambar yang di-upload |
| **🎯 Hasil Deteksi** | Output bounding box + label + confidence |
| **🔲 Fragmentasi SAHI** | Grid slice yang menunjukkan proses fragmentasi |
| **🎥 Video** | Playback video dengan inferensi real-time |

---

## 📐 Fitur Polygon ROI

Polygon ROI (Region of Interest) memungkinkan pengguna mendefinisikan area deteksi yang fleksibel secara visual. Hanya objek yang **berada di dalam area poligon** yang akan ditampilkan dan dihitung.

### Cara Menggunakan ROI

1. **Upload gambar atau video** terlebih dahulu.
2. Di panel kiri, klik tombol **"Tentukan ROI"** — status berubah menjadi "Menggambar...".
3. **Klik kiri** berulang kali di atas gambar/video untuk menambahkan titik batas area.
4. **Klik kanan** untuk menutup dan menyelesaikan poligon (minimal 3 titik).
5. Jalankan deteksi seperti biasa — hanya objek **di dalam area ROI** yang akan diproses.
6. Klik **"Hapus ROI"** untuk kembali ke mode deteksi penuh.

> **Catatan:** Fitur ROI berjalan pada mode **YOLO-only** maupun **YOLO+SAHI**, baik untuk gambar maupun video.

---

## ⚙️ Cara Kerja SAHI

```
Gambar / Frame Video (resolusi tinggi)
   ↓  Slicing: bagi menjadi patch 640×640 dengan overlap 20%
   ↓  YOLO per patch → deteksi lokal tiap patch
   ↓  Transform koordinat: patch space → original image space
   ↓  Multi-class NMS → gabungkan dan deduplikasi
   ↓  [Opsional] Filter ROI via cv2.pointPolygonTest
   ↓  Hasil final: deteksi objek kecil yang sebelumnya terlewat
```

**Parameter SAHI (dapat diubah di GUI):**

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| Slice size | 640 px | Ukuran tiap patch (128–1024) |
| Overlap | 0.20 | Rasio tumpang-tindih antar patch |
| Confidence | 0.15 | Threshold kepercayaan deteksi |

---

## 📊 Perbandingan YOLO-only vs YOLO+SAHI

| Aspek | YOLO-only | YOLO+SAHI |
|-------|-----------|-----------|
| **Metode** | Full image 640×640 | Slicing 640×640 + overlap 20% |
| **Kecepatan** | ✅ Lebih cepat | ⚠️ Lebih lambat (N patches) |
| **Akurasi objek kecil** | ⚠️ Lebih rendah | ✅ Lebih tinggi |
| **Penggunaan memori** | Rendah | Sedang (per patch) |
| **Cocok untuk** | Inferensi real-time | Akurasi tinggi / analisis |

---

## 📊 Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **mAP@0.5** | Mean Average Precision, IoU threshold = 0.5 |
| **mAP@0.5:0.95** | COCO standard mAP (rata-rata IoU 0.5–0.95) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **FPS** | Frame per second pada CPU-only |
| **Latency (ms)** | Waktu inferensi per gambar/frame |

---

## 🔬 Pipeline Deteksi Penuh

```bash
# 1. Training
.\miniconda\python.exe train_final.py

# 2. Evaluasi model (opsional)
.\miniconda\python.exe src\inference_sahi.py \
    --model weights/best.pt \
    --source data/datasets/test/images/ \
    --compare --benchmark

# 3. Demo GUI
.\miniconda\python.exe src\gui.py
```

---

## 📦 Dependencies Utama

```
ultralytics>=8.3.0   # YOLOv11n
torch>=2.0.0         # PyTorch (CPU mode)
torchvision>=0.15.0  # NMS utilities
opencv-python>=4.8.0 # Image/Video I/O
Pillow>=9.0.0        # Image drawing (bbox, ROI)
numpy>=1.24.0        # Array operations
psutil>=5.9.0        # Resource monitoring
```

---

## 📄 Referensi

- **YOLOv11**: [Ultralytics Docs](https://docs.ultralytics.com/) — YOLO11 (2024)
- **SAHI**: [Akyon et al., 2022](https://arxiv.org/abs/2202.06934) — Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection
- **Dataset**: [Kaggle River Floating Trash Datasets](https://www.kaggle.com/datasets/zhiaun/river-floating-trash-datasets/data) — 8 Kelas Sampah

---

## 👤 Penulis

Proyek Skripsi — Sistem Deteksi Sampah Berbasis Deep Learning  
Model: YOLOv11n | Metode: YOLO vs YOLO+SAHI | Interface: GUI Interaktif dengan Polygon ROI