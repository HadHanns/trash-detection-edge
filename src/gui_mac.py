#!/usr/bin/env python3
"""
gui_mac.py
==========
GUI Tkinter untuk pengujian deteksi sampah: YOLO-only vs YOLO+SAHI
Versi khusus macOS — penyesuaian:
  - Font sistem macOS (Helvetica Neue / SF Pro)
  - MPS device support (Apple Silicon M1/M2/M3)
  - Scroll trackpad macOS (<MouseWheel> dengan event.delta / 1)
  - Right-click = Button-2 di macOS (bukan Button-3)
  - Path separator Unix-style (/)
  - Font file fallback ke /System/Library/Fonts/

Penggunaan:
    python src/gui_mac.py
    python src/gui_mac.py --model weights/best_yolov8n.pt
"""

import os
import sys
import time
import math
import threading
import argparse
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import cv2
import torch

# ── Deteksi device terbaik (MPS untuk Apple Silicon, fallback CPU) ────────────
def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _best_device()

# ── Konstanta ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "bottle",            # 0
    "grass",             # 1
    "branch",            # 2
    "milk-box",          # 3
    "plastic-bag",       # 4
    "plastic-garbage",   # 5
    "ball",              # 6
    "leaf",              # 7
]

CLASS_COLORS_RGB = {
    0: (255, 128,   0),  # bottle          — orange
    1: (0,   200, 100),  # grass           — green
    2: (139,  90,  43),  # branch          — brown
    3: (0,   120, 255),  # milk-box        — blue
    4: (220,   0, 150),  # plastic-bag     — pink
    5: (140,   0, 255),  # plastic-garbage — purple
    6: (0,   220, 220),  # ball            — cyan
    7: (180, 200,  60),  # leaf            — yellow-green
}

CONF_THRESHOLD = 0.15
SLICE_SIZE     = 640
OVERLAP_RATIO  = 0.2

# ── Tema ──────────────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_MID    = "#16213e"
BG_CARD   = "#0f3460"
ACCENT    = "#e94560"
ACCENT2   = "#533483"
TEXT_MAIN = "#eaeaea"
TEXT_SUB  = "#a0a0b0"
SUCCESS   = "#4caf50"
WARNING   = "#ff9800"

# ── Font macOS ────────────────────────────────────────────────────────────────
# macOS tidak punya 'Segoe UI' atau 'Consolas' bawaan, gunakan yang ada
UI_FONT   = "Helvetica Neue"  # tersedia di semua macOS
MONO_FONT = "Menlo"           # mono font macOS (setara Consolas)

# Path font untuk PIL (ImageFont) — fallback ke default jika tidak ditemukan
MAC_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSText.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

def _get_pil_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in MAC_FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_yolo_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


# ── Detection Functions ───────────────────────────────────────────────────────

def run_yolo_full(model, img_bgr: np.ndarray, device=None, conf=0.15):
    """YOLO inference pada gambar penuh. Returns (boxes, scores, class_ids, ms)."""
    if device is None:
        device = DEVICE
    t0 = time.perf_counter()
    with torch.no_grad():
        results = model(
            img_bgr,
            conf=conf,
            iou=0.5,
            device=device,
            verbose=False,
            imgsz=SLICE_SIZE,
        )
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000

    r = results[0].boxes
    if r is None or len(r) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, int), ms

    boxes   = r.xyxy.cpu().numpy()
    scores  = r.conf.cpu().numpy()
    cls_ids = r.cls.cpu().numpy().astype(int)
    return boxes, scores, cls_ids, ms


def generate_slices(H, W, size=640, overlap=0.2):
    """Hasilkan daftar koordinat patch (x1,y1,x2,y2)."""
    stride = int(size * (1 - overlap))
    slices = []
    y = 0
    while True:
        x = 0
        while True:
            x1 = x; y1 = y
            x2 = min(x + size, W); y2 = min(y + size, H)
            slices.append((x1, y1, x2, y2))
            if x2 == W: break
            x += stride
        if y2 == H: break
        y += stride
    return slices


def run_yolo_on_patch(model, patch_bgr, device=None, conf=0.15):
    """Inference pada satu patch. Returns (boxes, scores, class_ids)."""
    if device is None:
        device = DEVICE
    with torch.no_grad():
        results = model(
            patch_bgr,
            conf=conf,
            iou=0.5,
            device=device,
            verbose=False,
            imgsz=SLICE_SIZE,
        )
    r = results[0].boxes
    if r is None or len(r) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, int)
    return (
        r.xyxy.cpu().numpy(),
        r.conf.cpu().numpy(),
        r.cls.cpu().numpy().astype(int),
    )


def multiclass_nms_numpy(boxes, scores, cls_ids, iou_thr=0.5):
    """Simple multi-class NMS."""
    if len(boxes) == 0:
        return boxes, scores, cls_ids

    import torchvision.ops as ops
    boxes_t  = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    cls_t    = torch.from_numpy(cls_ids.astype(np.float32))

    max_coord = boxes_t.max() + 1
    offsets   = cls_t * max_coord
    boxes_off = boxes_t + offsets[:, None]
    keep      = ops.nms(boxes_off, scores_t, iou_thr).numpy()
    return boxes[keep], scores[keep], cls_ids[keep]


# ── Drawing ───────────────────────────────────────────────────────────────────

def filter_boxes_by_roi(boxes, scores, cls_ids, roi_points):
    if not roi_points or len(roi_points) < 3 or len(boxes) == 0:
        return boxes, scores, cls_ids

    keep = []
    roi_poly = np.array(roi_points, dtype=np.int32)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0:
            keep.append(i)
    return boxes[keep], scores[keep], cls_ids[keep]


def draw_boxes_pil(pil_img: Image.Image, boxes, scores, cls_ids,
                   alpha_overlay=None, roi_points=None) -> Image.Image:
    """Gambar bounding box + label pada PIL Image."""
    draw = ImageDraw.Draw(pil_img, "RGBA")
    if roi_points and len(roi_points) >= 3:
        draw.polygon(roi_points, outline=(0, 255, 255, 200), width=3, fill=(0, 255, 255, 30))

    font_label = _get_pil_font(14)
    font_small = _get_pil_font(12)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid   = int(cls_ids[i]) if i < len(cls_ids) else 6
        score = float(scores[i]) if i < len(scores) else 0.0
        color = CLASS_COLORS_RGB.get(cid, (150, 150, 150))

        for t in range(2):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(*color, 255))

        label = f"{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else 'obj'} {score:.2f}"
        bbox  = draw.textbbox((x1, y1 - 18), label, font=font_label)
        draw.rectangle(bbox, fill=(*color, 230))
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255, 255), font=font_label)

    return pil_img


def draw_slice_grid(pil_img: Image.Image, slices, active_idx=-1,
                    done_indices=None) -> Image.Image:
    done_indices = done_indices or set()
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for i, (x1, y1, x2, y2) in enumerate(slices):
        if i == active_idx:
            fill    = (255, 80, 80, 80)
            outline = (255, 0, 0, 255)
            width   = 3
        elif i in done_indices:
            fill    = (80, 200, 80, 40)
            outline = (80, 200, 80, 180)
            width   = 1
        else:
            fill    = (255, 255, 255, 15)
            outline = (200, 200, 255, 120)
            width   = 1

        draw.rectangle([x1, y1, x2-1, y2-1], fill=fill, outline=outline, width=width)
        font = _get_pil_font(12)
        draw.text((x1 + 4, y1 + 4), f"#{i+1}", fill=(255, 255, 255, 200), font=font)

    return Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")


# ── GUI App ───────────────────────────────────────────────────────────────────

class TrashDetectionApp:
    def __init__(self, root: tk.Tk, model_path: str = "weights/best_yolov8n.pt"):
        self.root       = root
        self.model_path = model_path
        self.model      = None
        self.img_bgr    = None
        self.img_path   = None
        self._running   = False

        self._video_cap        = None
        self._video_running    = False
        self._video_paused     = False
        self._video_path       = None
        self._tk_img_video     = None
        self.roi_points        = []
        self._is_drawing_roi   = False
        self._video_last_frame = None

        self._setup_window()
        self._build_ui()
        self._load_model_async()

    # ── Window Setup ──────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("🗑️  Trash Detection: YOLO vs YOLO+SAHI")
        self.root.geometry("1280x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG_DARK)

        # macOS: aktifkan Retina scaling agar tidak blur
        try:
            self.root.tk.call("tk", "scaling", 2.0)
        except Exception:
            pass

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame",      background=BG_DARK)
        style.configure("Card.TFrame", background=BG_MID)
        style.configure(
            "Accent.TButton",
            background=ACCENT, foreground="white",
            font=(UI_FONT, 11, "bold"), padding=8,
        )
        style.map("Accent.TButton",
                  background=[("active", "#c73652"), ("disabled", "#555")])
        style.configure(
            "TLabel",
            background=BG_DARK, foreground=TEXT_MAIN,
            font=(UI_FONT, 10),
        )
        style.configure(
            "Stat.TLabel",
            background=BG_MID, foreground=TEXT_MAIN,
            font=(MONO_FONT, 10),
        )

    # ── UI Builder ────────────────────────────────────────────────────────

    def _build_ui(self):
        device_label = f"Device: {'MPS (Apple Silicon)' if DEVICE == 'mps' else 'CPU'}"
        top_bar = tk.Frame(self.root, bg=BG_CARD, height=40)
        top_bar.pack(side="top", fill="x")
        top_bar.pack_propagate(False)
        tk.Label(
            top_bar,
            text=f"🗑️  Kaggle RFT Trash Detection (8 Classes) — {device_label}",
            bg=BG_CARD, fg=TEXT_MAIN, font=(UI_FONT, 13, "bold")
        ).pack(side="left", padx=16, pady=8)

        self._paned = ttk.PanedWindow(self.root, orient="horizontal")
        self._paned.pack(fill="both", expand=True, padx=8, pady=8)

        self._left_panel = tk.Frame(self._paned, bg=BG_MID, width=320)
        self._paned.add(self._left_panel, weight=0)

        self._right_panel = tk.Frame(self._paned, bg=BG_DARK)
        self._paned.add(self._right_panel, weight=1)

        self._build_left_panel(self._left_panel)
        self._build_right_panel(self._right_panel)

        self._status_var = tk.StringVar(value="⏳ Menunggu model dimuat...")
        self._status_bar = tk.Label(
            self.root, textvariable=self._status_var,
            bg=BG_CARD, fg=TEXT_SUB, font=(UI_FONT, 10), anchor="w",
        )
        self._status_bar.pack(side="bottom", fill="x", ipady=4, padx=8)

    def _build_left_panel(self, parent):
        wrapper = tk.Frame(parent, bg=BG_MID)
        wrapper.pack(fill="both", expand=True)

        sb = ttk.Scrollbar(wrapper, orient="vertical")
        sb.pack(side="right", fill="y")

        canvas_scroll = tk.Canvas(wrapper, bg=BG_MID, highlightthickness=0,
                                  yscrollcommand=sb.set)
        canvas_scroll.pack(side="left", fill="both", expand=True)
        sb.config(command=canvas_scroll.yview)

        inner = tk.Frame(canvas_scroll, bg=BG_MID)
        inner_window = canvas_scroll.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(e):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        def _on_canvas_resize(e):
            canvas_scroll.itemconfig(inner_window, width=e.width)

        inner.bind("<Configure>", _on_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_resize)

        # ── macOS Scroll: gunakan <MouseWheel> dengan delta / 1 ────────────
        def _on_wheel_mac(e):
            # Pada macOS, event.delta biasanya 120 per notch (seperti Windows)
            # tetapi trackpad bisa menghasilkan nilai kecil; bagi dengan 120
            canvas_scroll.yview_scroll(int(-1 * (e.delta / 120)), "units")

        # macOS pakai <MouseWheel> bukan <Button-4>/<Button-5>
        canvas_scroll.bind_all("<MouseWheel>", _on_wheel_mac)

        p   = inner
        pad = {"padx": 12, "pady": 4}

        # ── SEKSI 1: GAMBAR ───────────────────────────────────────────────
        img_frame = tk.LabelFrame(p, text=" 📁 GAMBAR ", bg=BG_MID, fg=TEXT_MAIN,
                                  font=(UI_FONT, 10, "bold"), bd=1, padx=8, pady=8)
        img_frame.pack(fill="x", padx=12, pady=6)

        ttk.Button(img_frame, text="Upload Gambar", style="Accent.TButton",
                   command=self._upload_image).pack(fill="x", pady=(0, 4))

        self._filename_var = tk.StringVar(value="Belum ada gambar")
        tk.Label(img_frame, textvariable=self._filename_var, bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9), wraplength=210).pack(anchor="w", padx=4)

        tk.Label(img_frame, text="⚙️ Mode Deteksi:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).pack(anchor="w", pady=(8, 2))

        self._mode_var = tk.StringVar(value="yolo")
        mode_f = tk.Frame(img_frame, bg=BG_MID)
        mode_f.pack(fill="x")
        for val, lbl in [("yolo", "🎯 YOLO-only"), ("sahi", "🔲 YOLO + SAHI")]:
            tk.Radiobutton(
                mode_f, text=lbl, variable=self._mode_var, value=val,
                bg=BG_MID, fg=TEXT_MAIN, selectcolor=BG_CARD,
                activebackground=BG_MID, activeforeground=ACCENT,
                font=(UI_FONT, 10), indicatoron=True,
            ).pack(side="left", padx=(0, 12))

        self._detect_btn = ttk.Button(
            img_frame, text="▶  Jalankan Deteksi",
            style="Accent.TButton",
            command=self._run_detection,
            state="disabled",
        )
        self._detect_btn.pack(fill="x", pady=(12, 4))
        self._progress = ttk.Progressbar(img_frame, mode="indeterminate")
        self._progress.pack(fill="x")

        # ── SEKSI 2: VIDEO ────────────────────────────────────────────────
        vid_frame = tk.LabelFrame(p, text=" 🎥 VIDEO ", bg=BG_MID, fg=TEXT_MAIN,
                                  font=(UI_FONT, 10, "bold"), bd=1, padx=8, pady=8)
        vid_frame.pack(fill="x", padx=12, pady=6)

        ttk.Button(vid_frame, text="Upload Video", style="Accent.TButton",
                   command=self._upload_video).pack(fill="x", pady=(0, 4))

        self._videoname_var = tk.StringVar(value="Belum ada video")
        tk.Label(vid_frame, textvariable=self._videoname_var, bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9), wraplength=210).pack(anchor="w", padx=4)

        tk.Label(vid_frame, text="⚙️ Mode Deteksi:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).pack(anchor="w", pady=(8, 2))

        self._video_mode_var = tk.StringVar(value="yolo")
        vmode_f = tk.Frame(vid_frame, bg=BG_MID)
        vmode_f.pack(fill="x")
        for val, lbl in [("yolo", "🎯 YOLO-only"), ("sahi", "🔲 YOLO+SAHI")]:
            tk.Radiobutton(
                vmode_f, text=lbl, variable=self._video_mode_var, value=val,
                bg=BG_MID, fg=TEXT_MAIN, selectcolor=BG_CARD,
                activebackground=BG_MID, activeforeground=ACCENT,
                font=(UI_FONT, 10), indicatoron=True,
                command=self._on_video_mode_change,
            ).pack(side="left", padx=(0, 12))

        self._video_mode_warn = tk.Label(
            vid_frame, text="", bg=BG_MID, fg=WARNING,
            font=(UI_FONT, 8), wraplength=210)
        self._video_mode_warn.pack(anchor="w", pady=(2, 0))

        vctrl = tk.Frame(vid_frame, bg=BG_MID)
        vctrl.pack(fill="x", pady=(8, 4))

        self._play_btn = tk.Button(
            vctrl, text="▶ Play", bg=SUCCESS, fg="white",
            font=(UI_FONT, 10, "bold"), relief="flat", padx=6, pady=4,
            command=self._video_play_pause, state="disabled")
        self._play_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._stop_btn = tk.Button(
            vctrl, text="■ Stop", bg=ACCENT, fg="white",
            font=(UI_FONT, 10, "bold"), relief="flat", padx=6, pady=4,
            command=self._video_stop, state="disabled")
        self._stop_btn.pack(side="left", expand=True, fill="x")

        self._video_progress_var = tk.StringVar(value="Frame: —")
        tk.Label(vid_frame, textvariable=self._video_progress_var,
                 bg=BG_MID, fg=TEXT_SUB, font=(MONO_FONT, 9)).pack(anchor="w")

        # ── SEKSI ROI ─────────────────────────────────────────────────────
        roi_frame = tk.LabelFrame(p, text=" 📐 AREA DETEKSI (ROI) ", bg=BG_MID, fg=TEXT_MAIN,
                                  font=(UI_FONT, 10, "bold"), bd=1, padx=8, pady=8)
        roi_frame.pack(fill="x", padx=12, pady=6)

        self._roi_btn = ttk.Button(roi_frame, text="Tentukan ROI",
                                   style="Accent.TButton", command=self._toggle_roi_draw)
        self._roi_btn.pack(fill="x", pady=(0, 4))
        ttk.Button(roi_frame, text="Hapus ROI", command=self._clear_roi).pack(fill="x")

        self._roi_status_var = tk.StringVar(value="Status: Tidak aktif")
        tk.Label(roi_frame, textvariable=self._roi_status_var, bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).pack(anchor="w", pady=(4, 0))

        # ── SEKSI PENGATURAN ──────────────────────────────────────────────
        set_frame = tk.LabelFrame(p, text=" 🔧 PENGATURAN ", bg=BG_MID, fg=TEXT_MAIN,
                                  font=(UI_FONT, 10, "bold"), bd=1, padx=8, pady=8)
        set_frame.pack(fill="x", padx=12, pady=6)

        tk.Label(set_frame, text="Slice size:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).grid(row=0, column=0, sticky="w")
        self._slice_var = tk.IntVar(value=640)
        tk.Spinbox(set_frame, from_=128, to=1024, increment=64,
                   textvariable=self._slice_var, width=6,
                   bg=BG_CARD, fg=TEXT_MAIN, font=(MONO_FONT, 10),
                   buttonbackground=BG_CARD).grid(row=0, column=1, padx=4, pady=2)

        tk.Label(set_frame, text="Overlap:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).grid(row=1, column=0, sticky="w")
        self._overlap_var = tk.DoubleVar(value=0.2)
        tk.Spinbox(set_frame, from_=0.0, to=0.5, increment=0.05, format="%.2f",
                   textvariable=self._overlap_var, width=6,
                   bg=BG_CARD, fg=TEXT_MAIN, font=(MONO_FONT, 10),
                   buttonbackground=BG_CARD).grid(row=1, column=1, padx=4, pady=2)

        ttk.Separator(set_frame, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=6)

        tk.Label(set_frame, text="Confidence:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).grid(row=3, column=0, sticky="w")

        cf = tk.Frame(set_frame, bg=BG_MID)
        cf.grid(row=3, column=1, sticky="w")

        self._conf_var       = tk.DoubleVar(value=0.15)
        self._conf_label_var = tk.StringVar(value="0.15")

        def _update_conf_label(v):
            self._conf_label_var.set(f"{round(float(v), 2):.2f}")

        tk.Scale(cf, from_=0.05, to=0.9, resolution=0.05,
                 variable=self._conf_var, orient="horizontal",
                 bg=BG_MID, fg=TEXT_MAIN, troughcolor=BG_CARD,
                 highlightthickness=0, showvalue=False,
                 command=_update_conf_label, length=100).pack(side="left")
        tk.Label(cf, textvariable=self._conf_label_var,
                 bg=BG_MID, fg=ACCENT, font=(MONO_FONT, 11, "bold"),
                 width=4).pack(side="left")

        # ── SEKSI HASIL ───────────────────────────────────────────────────
        res_frame = tk.LabelFrame(p, text=" 📊 HASIL ", bg=BG_MID, fg=TEXT_MAIN,
                                  font=(UI_FONT, 10, "bold"), bd=1, padx=8, pady=8)
        res_frame.pack(fill="both", expand=True, padx=12, pady=6)

        self._stats_frame = tk.Frame(res_frame, bg=BG_MID)
        self._stats_frame.pack(fill="x", pady=(0, 8))

        self._stat_vars = {}
        for key, label in [
            ("mode",     "Mode"),
            ("detected", "Deteksi"),
            ("latency",  "Latency"),
            ("fps",      "FPS"),
            ("patches",  "Patches"),
        ]:
            row = tk.Frame(self._stats_frame, bg=BG_MID)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", bg=BG_MID, fg=TEXT_SUB,
                     font=(UI_FONT, 9), width=8, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=BG_MID, fg=TEXT_MAIN,
                     font=(MONO_FONT, 10)).pack(side="left")

        tk.Label(res_frame, text="🏷️ Daftar Deteksi:", bg=BG_MID, fg=TEXT_SUB,
                 font=(UI_FONT, 9)).pack(anchor="w")

        det_frame = tk.Frame(res_frame, bg=BG_MID)
        det_frame.pack(fill="both", expand=True, pady=(2, 0))

        det_sb = ttk.Scrollbar(det_frame)
        det_sb.pack(side="right", fill="y")

        self._det_list = tk.Listbox(
            det_frame, yscrollcommand=det_sb.set,
            bg=BG_CARD, fg=TEXT_MAIN,
            font=(MONO_FONT, 9), selectbackground=ACCENT,
            relief="flat", borderwidth=0, height=8,
        )
        self._det_list.pack(fill="both", expand=True)
        det_sb.config(command=self._det_list.yview)

    def _build_right_panel(self, parent):
        self._nb = ttk.Notebook(parent)
        self._nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("TNotebook",      background=BG_DARK)
        style.configure("TNotebook.Tab",  background=BG_MID, foreground=TEXT_MAIN,
                         font=(UI_FONT, 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", TEXT_MAIN)])

        self._tab_orig   = tk.Frame(self._nb, bg=BG_DARK)
        self._nb.add(self._tab_orig, text="🖼️  Gambar Asli")
        self._canvas_orig = tk.Canvas(self._tab_orig, bg="#111", highlightthickness=0)
        self._canvas_orig.pack(fill="both", expand=True)
        self._tk_img_orig = None

        self._tab_result = tk.Frame(self._nb, bg=BG_DARK)
        self._nb.add(self._tab_result, text="🎯  Hasil Deteksi")
        self._canvas_result = tk.Canvas(self._tab_result, bg="#111", highlightthickness=0)
        self._canvas_result.pack(fill="both", expand=True)
        self._tk_img_result = None

        self._tab_sahi = tk.Frame(self._nb, bg=BG_DARK)
        self._nb.add(self._tab_sahi, text="🔲  Fragmentasi SAHI")
        self._canvas_sahi = tk.Canvas(self._tab_sahi, bg="#111", highlightthickness=0)
        self._canvas_sahi.pack(fill="both", expand=True)
        self._tk_img_sahi = None

        self._tab_video = tk.Frame(self._nb, bg=BG_DARK)
        self._nb.add(self._tab_video, text="🎥  Video")
        self._canvas_video = tk.Canvas(self._tab_video, bg="#111", highlightthickness=0)
        self._canvas_video.pack(fill="both", expand=True)

        for canvas, txt in [
            (self._canvas_orig,   "Upload gambar untuk mulai"),
            (self._canvas_result, "Jalankan deteksi untuk melihat hasil"),
            (self._canvas_sahi,   "Tab ini menampilkan grid fragmentasi SAHI"),
            (self._canvas_video,  "Upload video lalu tekan ▶ Play"),
        ]:
            canvas.create_text(
                400, 300, text=txt,
                fill=TEXT_SUB, font=(UI_FONT, 14),
                tags="placeholder",
            )

    # ── Model Load ────────────────────────────────────────────────────────

    def _load_model_async(self):
        def _load():
            try:
                self.model = load_yolo_model(self.model_path)
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"❌ Gagal load model: {e}", "red"))
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self):
        self._set_status(
            f"✅ Model siap — {Path(self.model_path).name}  |  Device: {DEVICE.upper()}",
            SUCCESS
        )
        if self.img_bgr is not None:
            self._detect_btn.config(state="normal")

    # ── Image Upload ──────────────────────────────────────────────────────

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[
                ("Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Semua file", "*.*"),
            ],
        )
        if not path:
            return

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", f"Gagal membaca gambar:\n{path}")
            return

        self.img_bgr  = img_bgr
        self.img_path = path
        self._filename_var.set(Path(path).name)
        self._show_image_on_canvas(self._canvas_orig, img_bgr, tag="orig")
        self._nb.select(self._tab_orig)
        self._clear_result_canvas()
        self._clear_sahi_canvas()

        if self.model is not None:
            self._detect_btn.config(state="normal")

    # ── ROI ───────────────────────────────────────────────────────────────

    def _toggle_roi_draw(self):
        self._is_drawing_roi = not self._is_drawing_roi
        if self._is_drawing_roi:
            self.roi_points = []
            self._roi_btn.config(text="Selesai (Klik Kanan)")
            self._roi_status_var.set("Status: Menggambar... (Klik Kiri)")

            self._canvas_orig.bind("<Button-1>",  self._on_canvas_click)
            self._canvas_video.bind("<Button-1>", self._on_canvas_click)
            # macOS: right-click = Button-2 (bukan Button-3)
            self._canvas_orig.bind("<Button-2>",  self._on_canvas_right_click)
            self._canvas_video.bind("<Button-2>", self._on_canvas_right_click)
            # Tetap bind Button-3 sebagai fallback (Magic Mouse & external mouse)
            self._canvas_orig.bind("<Button-3>",  self._on_canvas_right_click)
            self._canvas_video.bind("<Button-3>", self._on_canvas_right_click)

            self._nb.select(self._tab_orig if self.img_bgr is not None else self._tab_video)
        else:
            self._roi_btn.config(text="Tentukan ROI")
            self._roi_status_var.set(
                f"Status: Aktif ({len(self.roi_points)} titik)"
                if len(self.roi_points) >= 3
                else "Status: Tidak aktif"
            )
            for canvas in (self._canvas_orig, self._canvas_video):
                canvas.unbind("<Button-1>")
                canvas.unbind("<Button-2>")
                canvas.unbind("<Button-3>")

            if len(self.roi_points) < 3:
                self.roi_points = []
                self._roi_status_var.set("Status: Tidak aktif")
            self._redraw_roi_on_orig()

    def _clear_roi(self):
        self.roi_points      = []
        self._is_drawing_roi = False
        self._roi_btn.config(text="Tentukan ROI")
        self._roi_status_var.set("Status: Tidak aktif")
        for canvas in (self._canvas_orig, self._canvas_video):
            canvas.unbind("<Button-1>")
            canvas.unbind("<Button-2>")
            canvas.unbind("<Button-3>")
        self._redraw_roi_on_orig()

    def _on_canvas_click(self, event):
        if not self._is_drawing_roi:
            return
        canvas   = event.widget
        cw, ch   = canvas.winfo_width(), canvas.winfo_height()
        tab_idx  = self._nb.index(self._nb.select())

        if tab_idx == 0 and self.img_bgr is not None:
            h, w = self.img_bgr.shape[:2]
        elif tab_idx == 3 and getattr(self, "_video_last_frame", None) is not None:
            h, w = self._video_last_frame.shape[:2]
        else:
            return

        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_x0 = (cw - nw) // 2
        img_y0 = (ch - nh) // 2

        real_x = int((event.x - img_x0) / scale)
        real_y = int((event.y - img_y0) / scale)

        if 0 <= real_x <= w and 0 <= real_y <= h:
            self.roi_points.append((real_x, real_y))
            self._redraw_roi_on_orig()

    def _on_canvas_right_click(self, event):
        if self._is_drawing_roi:
            self._toggle_roi_draw()

    def _redraw_roi_on_orig(self):
        tab_idx = self._nb.index(self._nb.select())
        if tab_idx == 0 and self.img_bgr is not None:
            img, canvas, tag = self.img_bgr, self._canvas_orig, "orig"
        elif tab_idx == 3 and getattr(self, "_video_last_frame", None) is not None:
            img, canvas, tag = self._video_last_frame, self._canvas_video, "video"
        else:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        if len(self.roi_points) > 0:
            draw = ImageDraw.Draw(pil_img, "RGBA")
            if len(self.roi_points) >= 3:
                draw.polygon(self.roi_points, outline=(0, 255, 255, 255), width=3,
                             fill=(0, 255, 255, 40))
            else:
                for pt in self.roi_points:
                    r = 4
                    draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r],
                                 fill=(0, 255, 255, 255))
        self._show_pil_on_canvas(canvas, pil_img, tag=tag)

    # ── Video Upload & Playback ────────────────────────────────────────────

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Pilih Video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"),
                ("Semua file", "*.*"),
            ],
        )
        if not path:
            return

        self._video_stop()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Gagal membuka video:\n{path}")
            return

        self._video_cap  = cap
        self._video_path = path
        self._videoname_var.set(Path(path).name)

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        self._video_fps_src = fps_src
        self._video_total   = total
        self._video_progress_var.set(f"Frame: 0 / {total}  |  {fps_src:.1f}fps")

        ret, frame = cap.read()
        if ret:
            self._video_last_frame = frame.copy()
            self._show_image_on_canvas(self._canvas_video, frame, tag="video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._nb.select(self._tab_video)
        self._play_btn.config(state="normal")
        self._stop_btn.config(state="normal")
        self._set_status(f"✅ Video dimuat: {Path(path).name}", SUCCESS)

    def _video_play_pause(self):
        if self._video_cap is None:
            return
        if not self._video_running:
            self._video_running = True
            self._video_paused  = False
            self._play_btn.config(text="⏸ Pause")
            self._nb.select(self._tab_video)
            threading.Thread(target=self._video_loop, daemon=True).start()
        else:
            self._video_paused = not self._video_paused
            if self._video_paused:
                self._play_btn.config(text="▶ Resume")
                self._set_status("⏸ Video dijeda", WARNING)
            else:
                self._play_btn.config(text="⏸ Pause")
                self._set_status("▶ Video berjalan...", SUCCESS)

    def _on_video_mode_change(self):
        if self._video_mode_var.get() == "sahi":
            self._video_mode_warn.config(
                text="⚠️ SAHI lebih lambat! FPS turun drastis. Cocok untuk analisis."
            )
        else:
            self._video_mode_warn.config(text="")

    def _video_stop(self):
        self._video_running = False
        self._video_paused  = False
        if self._video_cap is not None:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._play_btn.config(text="▶ Play")
        self._video_progress_var.set(f"Frame: 0 / {getattr(self, '_video_total', 0)}")
        self._set_status("⏹ Video dihentikan", TEXT_SUB)

    # ── Inference Helpers ─────────────────────────────────────────────────

    def _infer_yolo_frame(self, frame, conf):
        with torch.no_grad():
            results = self.model(
                frame, conf=conf, iou=0.5,
                device=DEVICE, verbose=False, imgsz=SLICE_SIZE,
            )
        r = results[0].boxes
        if r is not None and len(r) > 0:
            return (r.xyxy.cpu().numpy(), r.conf.cpu().numpy(),
                    r.cls.cpu().numpy().astype(int))
        return np.empty((0, 4)), np.empty(0), np.empty(0, int)

    def _infer_sahi_frame(self, frame, conf, slice_size, overlap):
        H, W   = frame.shape[:2]
        slices = generate_slices(H, W, slice_size, overlap)
        all_boxes, all_scores, all_cls = [], [], []

        for (x1, y1, x2, y2) in slices:
            patch = frame[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]
            if pw != slice_size or ph != slice_size:
                patch = cv2.resize(patch, (slice_size, slice_size))
                sx = (x2 - x1) / slice_size
                sy = (y2 - y1) / slice_size
            else:
                sx = sy = 1.0

            b, s, c = run_yolo_on_patch(self.model, patch, device=DEVICE, conf=conf)
            if len(b) > 0:
                b = b.copy()
                b[:, [0, 2]] = np.clip(b[:, [0, 2]] * sx + x1, 0, W)
                b[:, [1, 3]] = np.clip(b[:, [1, 3]] * sy + y1, 0, H)
                all_boxes.append(b); all_scores.append(s); all_cls.append(c)

        if all_boxes:
            fb, fs, fc = multiclass_nms_numpy(
                np.concatenate(all_boxes),
                np.concatenate(all_scores),
                np.concatenate(all_cls),
            )
        else:
            fb = np.empty((0, 4)); fs = np.empty(0); fc = np.empty(0, int)

        return fb, fs, fc, len(slices)

    def _video_loop(self):
        import time as _time
        cap        = self._video_cap
        fps_src    = getattr(self, "_video_fps_src", 30)
        total      = getattr(self, "_video_total", 0)
        delay      = 1.0 / max(fps_src, 1)
        conf       = float(self._conf_var.get())
        video_mode = self._video_mode_var.get()
        slice_size = self._slice_var.get()
        overlap    = self._overlap_var.get()

        while self._video_running:
            if self._video_paused:
                _time.sleep(0.05)
                continue

            t0 = _time.perf_counter()
            ret, frame = cap.read()
            if ret:
                self._video_last_frame = frame.copy()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(0, lambda: self._play_btn.config(text="▶ Play"))
                self.root.after(0, lambda: self._set_status("✅ Video selesai", SUCCESS))
                self._video_running = False
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if video_mode == "sahi":
                boxes, scores, cls_ids, n_patches = self._infer_sahi_frame(
                    frame, conf, slice_size, overlap
                )
                mode_label = f"Video-SAHI ({n_patches}p)"
            else:
                boxes, scores, cls_ids = self._infer_yolo_frame(frame, conf)
                n_patches  = None
                mode_label = "Video-YOLO"

            boxes, scores, cls_ids = filter_boxes_by_roi(
                boxes, scores, cls_ids, getattr(self, "roi_points", [])
            )
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_out  = draw_boxes_pil(
                Image.fromarray(img_rgb), boxes, scores, cls_ids,
                roi_points=getattr(self, "roi_points", [])
            )
            t1  = _time.perf_counter()
            ms  = (t1 - t0) * 1000
            fps = 1000.0 / ms if ms > 0 else 0

            def _update(pil_out=pil_out, n=len(boxes), ms=ms, fps=fps,
                        fi=frame_idx, tot=total, ml=mode_label, np_=n_patches):
                self._show_pil_on_canvas(self._canvas_video, pil_out, tag="video")
                patch_str = f" | {np_}patch" if np_ is not None else ""
                self._video_progress_var.set(
                    f"Frame: {fi}/{tot}{patch_str}  |  {n} obj  |  {ms:.0f}ms  |  {fps:.1f}fps"
                )
                self._update_stats(ml, n, ms, fps, patches=np_)

            self.root.after(0, _update)

            if video_mode == "yolo":
                elapsed = _time.perf_counter() - t0
                wait    = max(0.0, delay - elapsed)
                if wait > 0:
                    _time.sleep(wait)

    # ── Canvas Helpers ────────────────────────────────────────────────────

    def _show_image_on_canvas(self, canvas, img_bgr, tag="img"):
        canvas.update_idletasks()
        cw = max(100, canvas.winfo_width())
        ch = max(100, canvas.winfo_height())
        h, w   = img_bgr.shape[:2]
        scale  = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        tk_img  = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img, tags=tag)

        if tag == "orig":   self._tk_img_orig   = tk_img
        elif tag == "result": self._tk_img_result = tk_img
        elif tag == "sahi": self._tk_img_sahi   = tk_img
        else:               canvas._img_ref      = tk_img
        return scale, nw, nh

    def _show_pil_on_canvas(self, canvas, pil_img, tag="img"):
        canvas.update_idletasks()
        cw = max(100, canvas.winfo_width())
        ch = max(100, canvas.winfo_height())
        w, h   = pil_img.size
        scale  = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil_r  = pil_img.resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_r)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img, tags=tag)

        if tag == "result": self._tk_img_result = tk_img
        elif tag == "sahi": self._tk_img_sahi   = tk_img
        else:               canvas._img_ref      = tk_img

    def _clear_result_canvas(self):
        self._canvas_result.delete("all")
        self._canvas_result.create_text(
            400, 300, text="Jalankan deteksi untuk melihat hasil",
            fill=TEXT_SUB, font=(UI_FONT, 14), tags="placeholder"
        )

    def _clear_sahi_canvas(self):
        self._canvas_sahi.delete("all")
        self._canvas_sahi.create_text(
            400, 300, text="Tab ini menampilkan grid fragmentasi SAHI",
            fill=TEXT_SUB, font=(UI_FONT, 14), tags="placeholder"
        )

    # ── Detection ─────────────────────────────────────────────────────────

    def _run_detection(self):
        if self.img_bgr is None or self.model is None or self._running:
            return

        mode = self._mode_var.get()
        self._running = True
        self._detect_btn.config(state="disabled")
        self._progress.start(10)
        self._det_list.delete(0, "end")
        self._set_status("⏳ Mendeteksi...", WARNING)

        if mode == "yolo":
            threading.Thread(target=self._thread_yolo, daemon=True).start()
        else:
            threading.Thread(target=self._thread_sahi, daemon=True).start()

    def _thread_yolo(self):
        try:
            img  = self.img_bgr.copy()
            conf = float(self._conf_var.get())
            boxes, scores, cls_ids, ms = run_yolo_full(self.model, img, conf=conf)
            self.root.after(0, lambda: self._finish_yolo(img, boxes, scores, cls_ids, ms))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _finish_yolo(self, img_bgr, boxes, scores, cls_ids, ms):
        boxes, scores, cls_ids = filter_boxes_by_roi(
            boxes, scores, cls_ids, getattr(self, "roi_points", [])
        )
        self._stop_progress()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_res = draw_boxes_pil(
            Image.fromarray(img_rgb), boxes, scores, cls_ids,
            roi_points=getattr(self, "roi_points", [])
        )
        self._show_pil_on_canvas(self._canvas_result, pil_res, tag="result")
        self._nb.select(self._tab_result)

        fps = 1000.0 / ms if ms > 0 else 0
        self._update_stats("YOLO-only", len(boxes), ms, fps, patches=None)
        self._fill_detection_list(boxes, scores, cls_ids)
        self._set_status(
            f"✅ YOLO-only: {len(boxes)} deteksi | {ms:.0f}ms | {fps:.1f}FPS", SUCCESS
        )

    def _thread_sahi(self):
        try:
            img    = self.img_bgr.copy()
            H, W   = img.shape[:2]
            size   = self._slice_var.get()
            ovlp   = self._overlap_var.get()
            conf   = float(self._conf_var.get())
            slices = generate_slices(H, W, size, ovlp)
            n_slices = len(slices)

            self.root.after(0, lambda: self._show_sahi_grid(img, slices, -1, set()))
            self.root.after(0, lambda: self._nb.select(self._tab_sahi))

            all_boxes, all_scores, all_cls = [], [], []
            done = set()
            t_start = time.perf_counter()

            for i, (x1, y1, x2, y2) in enumerate(slices):
                self.root.after(0, lambda i=i: self._show_sahi_grid(img, slices, i, set(range(i))))
                self.root.after(0, lambda i=i, n=n_slices: self._set_status(
                    f"⏳ SAHI: patch {i+1}/{n}...", WARNING
                ))

                patch = img[y1:y2, x1:x2]
                ph, pw = patch.shape[:2]
                if pw != size or ph != size:
                    patch = cv2.resize(patch, (size, size))
                    sx = (x2 - x1) / size; sy = (y2 - y1) / size
                else:
                    sx = sy = 1.0

                b, s, c = run_yolo_on_patch(self.model, patch, device=DEVICE, conf=conf)

                if len(b) > 0:
                    b = b.copy()
                    b[:, [0, 2]] = np.clip(b[:, [0, 2]] * sx + x1, 0, W)
                    b[:, [1, 3]] = np.clip(b[:, [1, 3]] * sy + y1, 0, H)
                    all_boxes.append(b); all_scores.append(s); all_cls.append(c)

                done.add(i)

            ms = (time.perf_counter() - t_start) * 1000

            if all_boxes:
                final_b, final_s, final_c = multiclass_nms_numpy(
                    np.concatenate(all_boxes),
                    np.concatenate(all_scores),
                    np.concatenate(all_cls),
                )
            else:
                final_b = np.empty((0, 4)); final_s = np.empty(0); final_c = np.empty(0, int)

            self.root.after(0, lambda: self._finish_sahi(
                img, slices, done, final_b, final_s, final_c, ms, n_slices
            ))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: self._on_error(str(e)))

    def _show_sahi_grid(self, img_bgr, slices, active_idx, done_indices):
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_grid = draw_slice_grid(Image.fromarray(img_rgb), slices, active_idx, done_indices)
        self._show_pil_on_canvas(self._canvas_sahi, pil_grid, tag="sahi")

    def _finish_sahi(self, img_bgr, slices, done, boxes, scores, cls_ids, ms, n_slices):
        boxes, scores, cls_ids = filter_boxes_by_roi(
            boxes, scores, cls_ids, getattr(self, "roi_points", [])
        )
        self._stop_progress()
        self._show_sahi_grid(img_bgr, slices, -1, done)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_res = draw_boxes_pil(
            Image.fromarray(img_rgb), boxes, scores, cls_ids,
            roi_points=getattr(self, "roi_points", [])
        )
        self._show_pil_on_canvas(self._canvas_result, pil_res, tag="result")
        self._nb.select(self._tab_result)

        fps = 1000.0 / ms if ms > 0 else 0
        self._update_stats("YOLO+SAHI", len(boxes), ms, fps, n_slices)
        self._fill_detection_list(boxes, scores, cls_ids)
        self._set_status(
            f"✅ YOLO+SAHI: {len(boxes)} deteksi | {n_slices} patch | {ms:.0f}ms | {fps:.1f}FPS",
            SUCCESS
        )

    # ── UI Helpers ────────────────────────────────────────────────────────

    def _stop_progress(self):
        self._running = False
        self._progress.stop()
        self._detect_btn.config(state="normal")

    def _on_error(self, msg):
        self._stop_progress()
        self._set_status(f"❌ Error: {msg}", "red")
        messagebox.showerror("Error", msg)

    def _set_status(self, text, color=TEXT_MAIN):
        self._status_var.set(text)

    def _update_stats(self, mode, n_det, ms, fps, patches):
        self._stat_vars["mode"].set(mode)
        self._stat_vars["detected"].set(f"{n_det} objek")
        self._stat_vars["latency"].set(f"{ms:.0f} ms")
        self._stat_vars["fps"].set(f"{fps:.2f}")
        self._stat_vars["patches"].set(str(patches) if patches else "—")

    def _fill_detection_list(self, boxes, scores, cls_ids):
        self._det_list.delete(0, "end")
        for i in range(len(boxes)):
            cid   = int(cls_ids[i]) if i < len(cls_ids) else 0
            score = float(scores[i]) if i < len(scores) else 0.0
            name  = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "obj"
            self._det_list.insert("end", f"#{i+1:02d} {name} ({score:.2f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GUI Deteksi Sampah — macOS")
    parser.add_argument(
        "--model",
        default="weights/best_yolov8n.pt",  # YOLOv8n default
        # default="weights/best.pt",         # YOLOv11n
        help="Path model weights (.pt)"
    )
    args = parser.parse_args()

    root = tk.Tk()
    app  = TrashDetectionApp(root, model_path=args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
