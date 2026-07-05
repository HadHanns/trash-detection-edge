#!/usr/bin/env python3
"""
gui_pro.py
==========
Advanced GUI Tkinter untuk pengujian deteksi sampah: YOLO-only vs YOLO+SAHI
Meniru referensi desain light theme dengan akurasi tinggi:
  - Matplotlib integration untuk grafik trend
  - psutil untuk edge computing status
  - Telegram integration untuk alert
  - Layout & styling 100% mengikuti referensi
"""

import os
import sys
import time
import threading
import argparse
import json
from pathlib import Path
from copy import deepcopy
import datetime
import requests
from collections import deque

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
CONFIG_PATH = ROOT / "config.json"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import cv2
import psutil
import torch

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Deteksi device terbaik ────────────
def _best_device() -> str:
    if torch.cuda.is_available(): return "cuda:0"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = _best_device()

# ── Konstanta ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "bottle", "grass", "branch", "milk-box", "plastic-bag", "plastic-garbage", "ball", "leaf"
]

CLASS_COLORS_RGB = {
    0: (211, 47, 47),    # botol (Red)
    1: (56, 142, 60),    # organik (Green)
    2: (121, 85, 72),    # organik (Brown)
    3: (25, 118, 210),   # kertas/karton (Blue)
    4: (245, 124, 0),    # plastik (Orange)
    5: (123, 31, 162),   # styrofoam/other plastic (Purple)
    6: (0, 151, 167),    # lainnya (Cyan)
    7: (104, 159, 56),   # organik (Light Green)
}

GUI_CLASS_MAPPING = {
    0: "botol",
    1: "organik",
    2: "organik",
    3: "kertas/karton",
    4: "plastik",
    5: "styrofoam",
    6: "lainnya",
    7: "organik"
}

# ── Tema Light ─────────────────────────────────────────────────────────────────
BG_MAIN = "#f8f9fa"
BG_PANEL = "#ffffff"
TEXT_DARK = "#212529"
TEXT_MUTED = "#6c757d"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
ACCENT_ORANGE = "#f59e0b"
BORDER_COLOR = "#e5e7eb"

UI_FONT = ("Segoe UI", 9)
UI_FONT_BOLD = ("Segoe UI", 9, "bold")
HEADER_FONT = ("Segoe UI", 10, "bold")
TITLE_FONT = ("Segoe UI", 16, "bold")
SUBTITLE_FONT = ("Segoe UI", 11, "italic")
MONO_FONT = ("Consolas", 8)

# ── Helper Fungsi YOLO / SAHI ─────────────────────────────────────────────────

def load_yolo_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)

def generate_slices(H, W, size_h=640, size_w=640, overlap_h=0.2, overlap_w=0.2):
    stride_h = int(size_h * (1 - overlap_h))
    stride_w = int(size_w * (1 - overlap_w))
    slices = []
    y = 0
    while True:
        x = 0
        while True:
            x1, y1 = x, y
            x2, y2 = min(x + size_w, W), min(y + size_h, H)
            slices.append((x1, y1, x2, y2))
            if x2 == W: break
            x += stride_w
        if y2 == H: break
        y += stride_h
    return slices

def run_yolo_on_patch(model, patch_bgr, device=None, conf=0.25, iou=0.45):
    if device is None: device = DEVICE
    with torch.no_grad():
        results = model(patch_bgr, conf=conf, iou=iou, device=device, verbose=False)
    r = results[0].boxes
    if r is None or len(r) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, int)
    return r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy().astype(int)

def multiclass_nms_numpy(boxes, scores, cls_ids, iou_thr=0.45):
    if len(boxes) == 0: return boxes, scores, cls_ids
    import torchvision.ops as ops
    boxes_t = torch.from_numpy(boxes.astype(np.float32))
    scores_t = torch.from_numpy(scores.astype(np.float32))
    cls_t = torch.from_numpy(cls_ids.astype(np.float32))
    max_coord = boxes_t.max() + 1
    offsets = cls_t * max_coord
    boxes_off = boxes_t + offsets[:, None]
    keep = ops.nms(boxes_off, scores_t, iou_thr).numpy()
    return boxes[keep], scores[keep], cls_ids[keep]

def filter_boxes_by_roi(boxes, scores, cls_ids, roi_points):
    if not roi_points or len(roi_points) < 3 or len(boxes) == 0:
        return boxes, scores, cls_ids
    keep = []
    roi_poly = np.array(roi_points, dtype=np.int32)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0:
            keep.append(i)
    return boxes[keep], scores[keep], cls_ids[keep]

def draw_boxes_pil(pil_img, boxes, scores, cls_ids, roi_points=None):
    draw = ImageDraw.Draw(pil_img, "RGBA")
    
    # Draw ROI Polygon
    if roi_points and len(roi_points) >= 3:
        draw.polygon(roi_points, outline=(50, 255, 50, 255), width=2)
        for pt in roi_points:
            draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(50, 255, 50, 255))

    try:
        font_label = ImageFont.truetype("arial.ttf", 14)
    except:
        font_label = ImageFont.load_default()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cid = int(cls_ids[i]) if i < len(cls_ids) else 6
        score = float(scores[i]) if i < len(scores) else 0.0
        color = CLASS_COLORS_RGB.get(cid, (150, 150, 150))
        
        display_name = GUI_CLASS_MAPPING.get(cid, "obj")

        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)
        label = f"{display_name} {score:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), label, font=font_label)
        draw.rectangle(bbox, fill=(*color, 200))
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255, 255), font=font_label)

    return pil_img

def calculate_coverage(boxes, roi_points, img_shape):
    if len(boxes) == 0: return 0.0
    total_trash_area = 0
    for (x1, y1, x2, y2) in boxes:
        total_trash_area += max(0, (x2 - x1)) * max(0, (y2 - y1))
    if roi_points and len(roi_points) >= 3:
        roi_area = cv2.contourArea(np.array(roi_points, dtype=np.int32))
        if roi_area > 0:
            return min(100.0, (total_trash_area / roi_area) * 100)
    img_area = img_shape[0] * img_shape[1]
    return min(100.0, (total_trash_area / img_area) * 100)

def send_telegram_alert(token, chat_id, message):
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

def send_telegram_photo(token, chat_id, pil_img, caption):
    """Kirim foto snapshot beserta caption ke Telegram via sendPhoto API."""
    if not token or not chat_id or pil_img is None: return
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    try:
        requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("snapshot.jpg", buf, "image/jpeg")},
            timeout=15
        )
    except Exception as e:
        print(f"Telegram Photo Error: {e}")

# ── GUI Class ─────────────────────────────────────────────────────────────────

class AdvancedTrashGUI:
    def __init__(self, root, default_model):
        self.root = root
        self.root.title("Deteksi Sampah dengan YOLOv8 + SAHI (ROI Polygon) - Edge Computing Simulation")
        self.root.geometry("1440x900")
        self.root.configure(bg=BG_MAIN)
        self.root.state('zoomed') # Maximize on Windows
        
        self.model_path = default_model
        self.model = None
        self._running = False
        
        self.video_cap = None
        self.roi_points = []
        self._is_drawing_roi = False
        self.last_frame = None
        self.total_frames = 0
        self.zoom_factor = 1.0
        self._last_pil_img = None
        
        self.time_history = deque(maxlen=30)
        self.coverage_history = deque(maxlen=30)
        self.last_alert_time = 0
        self.table_items = []
        
        self._setup_styles()
        self._build_ui()
        self._load_config()   # Muat pengaturan tersimpan
        self._start_system_monitor()
        self.log_activity("Aplikasi dimulai.")
        
        threading.Thread(target=self._load_model, daemon=True).start()

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
            
        style.configure("TFrame", background=BG_MAIN)
        style.configure("Panel.TFrame", background=BG_PANEL)
        
        style.configure("TLabel", background=BG_MAIN, foreground=TEXT_DARK, font=UI_FONT)
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=TEXT_DARK, font=UI_FONT)
        style.configure("Header.TLabel", font=HEADER_FONT, background=BG_PANEL, foreground=TEXT_DARK)
        style.configure("Title.TLabel", font=TITLE_FONT, background=BG_MAIN, foreground=TEXT_DARK)
        style.configure("Subtitle.TLabel", font=SUBTITLE_FONT, foreground=TEXT_MUTED, background=BG_MAIN)
        
        style.configure("TButton", font=UI_FONT)
        style.configure("Blue.TButton", background=ACCENT_BLUE, foreground="white", font=UI_FONT_BOLD)
        style.map("Blue.TButton", background=[("active", "#2563eb")])
        style.configure("Green.TButton", background=ACCENT_GREEN, foreground="white", font=UI_FONT_BOLD)
        style.map("Green.TButton", background=[("active", "#16a34a")])
        style.configure("Red.TButton", background=ACCENT_RED, foreground="white", font=UI_FONT_BOLD)
        style.map("Red.TButton", background=[("active", "#dc2626")])
        style.configure("White.TButton", background="#ffffff", foreground=TEXT_DARK, font=UI_FONT_BOLD, borderwidth=1)
        
        # Style Combobox
        style.configure("TCombobox", padding=4)
        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab", background="#e9ecef", foreground=TEXT_DARK, padding=[10, 5], font=UI_FONT)
        style.map("TNotebook.Tab", background=[("selected", BG_PANEL)], font=[("selected", UI_FONT_BOLD)])

    def _build_ui(self):
        # ── Header ──
        header_frame = tk.Frame(self.root, bg=BG_MAIN)
        header_frame.pack(side="top", fill="x", pady=(10, 5))
        ttk.Label(header_frame, text="Implementasi YOLOv8 + SAHI untuk Deteksi Dini Penumpukan Sampah", style="Title.TLabel").pack(anchor="center")
        ttk.Label(header_frame, text="Simulasi Edge Computing - Saluran Air Perkotaan Skala Kecil", style="Subtitle.TLabel").pack(anchor="center")
        
        # ── Bottom Status Bar ──
        self.status_bar = tk.Frame(self.root, bg=BG_PANEL, height=35, bd=1, relief="solid")
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)
        
        def add_status_item(parent, default_text, fg=TEXT_MUTED):
            lbl = tk.Label(parent, text=default_text, bg=BG_PANEL, fg=fg, font=UI_FONT)
            lbl.pack(side="left", padx=15)
            # Add separator
            tk.Frame(parent, bg=BORDER_COLOR, width=1, height=20).pack(side="left", pady=7)
            return lbl
            
        self.lbl_status = add_status_item(self.status_bar, "Status: Idle", fg=TEXT_MUTED)
        self.lbl_source = add_status_item(self.status_bar, "Sumber: None")
        self.lbl_duration = add_status_item(self.status_bar, "Durasi: 00:00:00 / 00:00:00")
        self.lbl_frame_info = add_status_item(self.status_bar, "Frame: 0 / 0")
        self.lbl_model_stat = add_status_item(self.status_bar, "Model: YOLOv8n")
        self.lbl_sahi_stat = add_status_item(self.status_bar, "SAHI: Inactive")
        self.lbl_roi_stat = tk.Label(self.status_bar, text="ROI: Inactive", bg=BG_PANEL, fg=TEXT_MUTED, font=UI_FONT)
        self.lbl_roi_stat.pack(side="left", padx=15)
        
        # ── Main Container ──
        main_container = tk.Frame(self.root, bg=BG_MAIN)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Kolom Kiri
        left_col = tk.Frame(main_container, bg=BG_PANEL, width=260, bd=1, relief="solid")
        left_col.pack(side="left", fill="y", padx=(0, 5))
        left_col.pack_propagate(False)
        self._build_left_panel(left_col)
        
        # Kolom Kanan
        right_col = tk.Frame(main_container, bg=BG_PANEL, width=400, bd=1, relief="solid")
        right_col.pack(side="right", fill="y", padx=(5, 0))
        right_col.pack_propagate(False)
        self._build_right_panel(right_col)
        
        # Kolom Tengah
        mid_col = tk.Frame(main_container, bg=BG_MAIN, bd=1, relief="solid")
        mid_col.pack(side="left", fill="both", expand=True)
        self._build_mid_panel(mid_col)

    def _build_left_panel(self, parent_frame):
        # Create scrollable canvas
        canvas = tk.Canvas(parent_frame, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        parent = tk.Frame(canvas, bg=BG_PANEL)
        
        parent.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # We set width=240 to fit inside the 260px column minus scrollbar
        canvas.create_window((0, 0), window=parent, anchor="nw", width=240)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            # Cek apakan kursor di dalam area kanvas
            if str(event.widget).startswith(str(canvas)):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        parent_frame.bind_all("<MouseWheel>", _on_mousewheel)

        ttk.Label(parent, text="Kontrol", style="Header.TLabel").pack(anchor="w", padx=15, pady=10)
        
        def create_section_header(text):
            ttk.Label(parent, text=text, style="Header.TLabel").pack(anchor="w", padx=15, pady=(20, 5))
            
        # Sumber Input
        create_section_header("Sumber Input")
        rf = tk.Frame(parent, bg=BG_PANEL)
        rf.pack(fill="x", padx=15, pady=(0, 5))
        self.input_type = tk.StringVar(value="video")
        tk.Radiobutton(rf, text="Video", variable=self.input_type, value="video", bg=BG_PANEL, font=UI_FONT).pack(side="left", padx=(0, 10))
        tk.Radiobutton(rf, text="Webcam", variable=self.input_type, value="webcam", bg=BG_PANEL, font=UI_FONT).pack(side="left")
        
        b_f = tk.Frame(parent, bg=BG_PANEL)
        b_f.pack(fill="x", padx=15, pady=5)
        self.entry_source = ttk.Entry(b_f, font=UI_FONT)
        self.entry_source.pack(side="left", fill="x", expand=True, padx=(0,5))
        ttk.Button(b_f, text="Browse", command=self._browse_source, width=8).pack(side="left")
        
        # Model
        create_section_header("Weight YOLO")
        m_inner = tk.Frame(parent, bg=BG_PANEL)
        m_inner.pack(fill="x", padx=15)
        self.cb_model = ttk.Combobox(m_inner, values=[self.model_path], font=UI_FONT)
        self.cb_model.set(self.model_path)
        self.cb_model.pack(side="left", fill="x", expand=True, padx=(0,5))
        ttk.Button(m_inner, text="📂", width=3, command=self._browse_model).pack(side="left", padx=(0,5))
        ttk.Button(m_inner, text="Load", style="Blue.TButton", command=self._reload_model, width=6).pack(side="left")
        
        # SAHI Params
        create_section_header("SAHI - Slicing")
        def _add_combo_param(p, label, vals, default):
            f = tk.Frame(p, bg=BG_PANEL)
            f.pack(fill="x", pady=2, padx=15)
            ttk.Label(f, text=label, style="Panel.TLabel").pack(side="left")
            cb = ttk.Combobox(f, values=vals, width=6, font=UI_FONT, justify="center")
            cb.set(str(default))
            cb.pack(side="right")
            return cb
            
        self.var_sl_h = _add_combo_param(parent, "Slice Height", ["320", "512", "640"], 640)
        self.var_sl_w = _add_combo_param(parent, "Slice Width", ["320", "512", "640"], 640)
        self.var_ol_h = _add_combo_param(parent, "Overlap Height (%)", ["10", "20", "30"], 20)
        self.var_ol_w = _add_combo_param(parent, "Overlap Width (%)", ["10", "20", "30"], 20)
        
        self.var_sahi_enable = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Aktifkan Slicing (SAHI)", variable=self.var_sahi_enable, bg=BG_PANEL, font=UI_FONT).pack(anchor="w", padx=15, pady=(5,0))
        
        # ROI
        create_section_header("Region of Interest (ROI)")
        rb_f = tk.Frame(parent, bg=BG_PANEL)
        rb_f.pack(fill="x", padx=15, pady=2)
        self.btn_roi_draw = ttk.Button(rb_f, text="Buat Polygon ROI", style="Green.TButton", command=self._toggle_roi)
        self.btn_roi_draw.pack(side="left", expand=True, fill="x", padx=(0,2), ipady=3)
        ttk.Button(rb_f, text="Hapus ROI", style="Red.TButton", command=self._clear_roi).pack(side="left", expand=True, fill="x", padx=(2,0), ipady=3)
        ttk.Label(parent, text="Klik area video (minimal 3 titik)\nuntuk membuat Polygon ROI.", style="Panel.TLabel", foreground=TEXT_MUTED).pack(anchor="w", padx=15, pady=(5,0))
        
        # Parameter Inferensi
        create_section_header("Parameter Inferensi")
        
        device_opts = ["cpu"]
        if torch.cuda.is_available(): device_opts.insert(0, "cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device_opts.insert(0, "mps")
        self.var_device = _add_combo_param(parent, "Hardware Device", device_opts, device_opts[0])
        
        self.var_conf = _add_combo_param(parent, "Conf. Threshold", ["0.15", "0.25", "0.5"], 0.25)
        self.var_iou = _add_combo_param(parent, "IoU Threshold", ["0.45", "0.5", "0.65"], 0.45)
        self.var_sample_interval = _add_combo_param(parent, "Sample Interval (dtk)", ["0.5", "1", "2", "3", "5"], 1)
        
        self.var_roi_crop = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Crop ke ROI (percepat deteksi)", variable=self.var_roi_crop, bg=BG_PANEL, font=UI_FONT).pack(anchor="w", padx=15, pady=(5,0))
        
        # Kontrol
        create_section_header("Kontrol Sistem")
        cb_f = tk.Frame(parent, bg=BG_PANEL)
        cb_f.pack(fill="x", padx=15, pady=(5, 20))
        self.btn_start = ttk.Button(cb_f, text="▶ Mulai", style="Green.TButton", command=self._start_detection)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0,2), ipady=4)
        self.btn_stop = ttk.Button(cb_f, text="■ Stop", style="White.TButton", command=self._stop_detection)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=(2,0), ipady=4)

    def _build_mid_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab Deteksi Live
        self.tab_live = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(self.tab_live, text="Deteksi Live")
        
        self.canvas_video = tk.Canvas(self.tab_live, bg="#1a1a1a", highlightthickness=0)
        self.canvas_video.pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas_video.bind("<Button-1>", self._on_canvas_click)
        self.canvas_video.bind("<Button-2>", self._on_canvas_right_click)
        self.canvas_video.bind("<Button-3>", self._on_canvas_right_click)
        
        # Zoom Controls
        zoom_f = tk.Frame(self.tab_live, bg=BG_PANEL)
        zoom_f.pack(fill="x", padx=5, pady=5)
        ttk.Button(zoom_f, text="⊕", width=3, command=self._zoom_in).pack(side="left", padx=2)
        ttk.Button(zoom_f, text="⊖", width=3, command=self._zoom_out).pack(side="left", padx=2)
        ttk.Button(zoom_f, text="⛶", width=3, command=self._zoom_reset).pack(side="left", padx=2)
        ttk.Button(zoom_f, text="📷", width=3, command=self._save_snapshot_quick).pack(side="left", padx=2)
        ttk.Button(zoom_f, text="Simpan Snapshot", style="White.TButton", command=self._save_snapshot_dialog).pack(side="left", padx=5)

        # Tab Hasil & Statistik
        self.tab_stats = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(self.tab_stats, text="Hasil & Statistik")
        ttk.Label(self.tab_stats, text="Statistik akan muncul di sini.", style="Panel.TLabel").pack(pady=20)
        
        # Tab Pengaturan
        self.tab_settings = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(self.tab_settings, text="Pengaturan")
        self._build_settings_tab(self.tab_settings)
        
        # Tab Log
        self.tab_log = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(self.tab_log, text="Log")
        self.txt_log_full = scrolledtext.ScrolledText(self.tab_log, font=MONO_FONT, bg="#f8f9fa", bd=0)
        self.txt_log_full.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_settings_tab(self, parent):
        f = tk.Frame(parent, bg=BG_PANEL, padx=20, pady=20)
        f.pack(fill="both", expand=True)
        
        ttk.Label(f, text="Konfigurasi Notifikasi Telegram", style="Title.TLabel").pack(anchor="w", pady=(0,20))
        
        ttk.Label(f, text="Bot Token:", style="Panel.TLabel").pack(anchor="w")
        self.entry_tg_token = ttk.Entry(f, width=50, font=UI_FONT)
        self.entry_tg_token.pack(anchor="w", pady=(0,10))
        
        ttk.Label(f, text="Chat ID:", style="Panel.TLabel").pack(anchor="w")
        self.entry_tg_chatid = ttk.Entry(f, width=50, font=UI_FONT)
        self.entry_tg_chatid.pack(anchor="w", pady=(0,10))
        
        ttk.Label(f, text="Threshold Status (Coverage %):", style="Panel.TLabel").pack(anchor="w")
        f_thr = tk.Frame(f, bg=BG_PANEL)
        f_thr.pack(anchor="w", pady=(0,10))
        ttk.Label(f_thr, text="WASPADA >", style="Panel.TLabel").pack(side="left")
        self.var_thr_waspada = tk.StringVar(value="10")
        ttk.Entry(f_thr, textvariable=self.var_thr_waspada, width=5).pack(side="left", padx=5)
        ttk.Label(f_thr, text="%  |  KRITIS >", style="Panel.TLabel").pack(side="left")
        self.var_thr_kritis = tk.StringVar(value="25")
        ttk.Entry(f_thr, textvariable=self.var_thr_kritis, width=5).pack(side="left", padx=5)
        ttk.Label(f_thr, text="%", style="Panel.TLabel").pack(side="left")

        # Cooldown anti-spam
        ttk.Label(f, text="Jeda Minimum Antar Notifikasi (Anti-Spam):", style="Panel.TLabel").pack(anchor="w", pady=(10, 0))
        f_cd = tk.Frame(f, bg=BG_PANEL)
        f_cd.pack(anchor="w", pady=(0,10))
        self.var_alert_cooldown = tk.StringVar(value="300")
        ttk.Entry(f_cd, textvariable=self.var_alert_cooldown, width=6).pack(side="left")
        ttk.Label(f_cd, text=" detik  (300 = 5 menit, 60 = 1 menit)", style="Panel.TLabel").pack(side="left", padx=5)

        # Kirim snapshot
        self.var_send_snapshot = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text="Sertakan snapshot/foto saat mengirim alert", variable=self.var_send_snapshot, bg=BG_PANEL, font=UI_FONT).pack(anchor="w", pady=(0,10))

        ttk.Button(f, text="💾 Simpan & Test Kirim", style="Blue.TButton", command=self._test_telegram).pack(anchor="w", pady=10)
        ttk.Label(f, text="Pengaturan disimpan otomatis saat klik tombol di atas.", style="Panel.TLabel", foreground=TEXT_MUTED).pack(anchor="w")

    def _build_right_panel(self, parent_frame):
        # Create scrollable canvas
        canvas = tk.Canvas(parent_frame, bg=BG_PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        parent = tk.Frame(canvas, bg=BG_PANEL)
        
        parent.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=parent, anchor="nw", width=380)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel_r(event):
            if str(event.widget).startswith(str(canvas)):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        parent_frame.bind_all("<MouseWheel>", _on_mousewheel_r, add="+")

        # Ringkasan Deteksi
        sf = tk.Frame(parent, bg=BG_PANEL)
        sf.pack(fill="x", padx=15, pady=10)
        ttk.Label(sf, text="Ringkasan Deteksi (ROI)", style="Header.TLabel").pack(anchor="w", pady=(0,10))
        
        box_f = tk.Frame(sf, bg=BG_PANEL)
        box_f.pack(fill="x")
        
        def _make_box(p, title, val, color=TEXT_DARK):
            b = tk.Frame(p, bg=BG_PANEL, highlightbackground=BORDER_COLOR, highlightthickness=1)
            b.pack(side="left", expand=True, fill="both", padx=3, ipady=8)
            tk.Label(b, text=title, bg=BG_PANEL, fg=TEXT_MUTED, font=UI_FONT).pack(pady=(5,2))
            lbl = tk.Label(b, text=val, bg=BG_PANEL, fg=color, font=("Segoe UI", 13, "bold"))
            lbl.pack(pady=(0,5))
            return lbl
            
        self.lbl_tot_obj = _make_box(box_f, "Total Objek", "0")
        self.lbl_level = _make_box(box_f, "Level", "AMAN", ACCENT_GREEN)
        self.lbl_coverage = _make_box(box_f, "Coverage Area", "0.00 %")
        
        # Tabel Kelas (Grid Layout for precision)
        tbl_f = tk.Frame(sf, bg=BG_PANEL)
        tbl_f.pack(fill="x", pady=20)
        
        # Header Table
        th = tk.Frame(tbl_f, bg=BG_PANEL)
        th.pack(fill="x", pady=(0, 5))
        th.grid_columnconfigure(0, minsize=20) # Dot
        th.grid_columnconfigure(1, weight=3, minsize=100) # Kelas
        th.grid_columnconfigure(2, weight=1, minsize=60) # Jumlah
        th.grid_columnconfigure(3, weight=1, minsize=80) # Persentase
        
        tk.Label(th, text="", font=UI_FONT_BOLD, bg=BG_PANEL).grid(row=0, column=0)
        tk.Label(th, text="Kelas", font=UI_FONT_BOLD, bg=BG_PANEL, anchor="w").grid(row=0, column=1, sticky="w")
        tk.Label(th, text="Jumlah", font=UI_FONT_BOLD, bg=BG_PANEL, anchor="center").grid(row=0, column=2, sticky="ew")
        tk.Label(th, text="Persentase", font=UI_FONT_BOLD, bg=BG_PANEL, anchor="e").grid(row=0, column=3, sticky="e")
        tk.Frame(tbl_f, bg=BORDER_COLOR, height=1).pack(fill="x", pady=(0, 5))
        
        self.table_rows_frame = tk.Frame(tbl_f, bg=BG_PANEL)
        self.table_rows_frame.pack(fill="x")
        self.table_rows_frame.grid_columnconfigure(0, minsize=20)
        self.table_rows_frame.grid_columnconfigure(1, weight=3, minsize=100)
        self.table_rows_frame.grid_columnconfigure(2, weight=1, minsize=60)
        self.table_rows_frame.grid_columnconfigure(3, weight=1, minsize=80)
        
        self.table_rows_vars = {}
        # Pre-create rows as Grid
        all_classes = ["plastik", "botol", "styrofoam", "organik", "kertas/karton", "lainnya"]
        for i, cls_name in enumerate(all_classes):
            c_color = "#ccc"
            if cls_name == "plastik": c_color = "#f59e0b"
            elif cls_name == "botol": c_color = "#ef4444"
            elif cls_name == "styrofoam": c_color = "#7b1fa2"
            elif cls_name == "organik": c_color = "#16a34a"
            elif cls_name == "kertas/karton": c_color = "#2563eb"
            
            lbl_dot = tk.Label(self.table_rows_frame, text="●", fg=c_color, bg=BG_PANEL, font=("Segoe UI", 10))
            lbl_name = tk.Label(self.table_rows_frame, text=cls_name, font=UI_FONT, bg=BG_PANEL, anchor="w")
            lbl_cnt = tk.Label(self.table_rows_frame, text="0", font=UI_FONT, bg=BG_PANEL, anchor="center")
            lbl_pct = tk.Label(self.table_rows_frame, text="0.0%", font=UI_FONT, bg=BG_PANEL, anchor="e")
            
            self.table_rows_vars[cls_name] = {
                "dot": lbl_dot, "name": lbl_name, "count": lbl_cnt, "pct": lbl_pct, "row_idx": i
            }
        
        self.lbl_tot_row = tk.Label(tbl_f, text="Total: 0", font=UI_FONT_BOLD, bg=BG_PANEL)
        self.lbl_tot_row.pack(anchor="w", pady=(10, 5))
        
        # Tren (Matplotlib)
        tf = tk.Frame(parent, bg=BG_PANEL)
        tf.pack(fill="x", padx=15, pady=5)
        ttk.Label(tf, text="Tren Penumpukan (Coverage %)", style="Header.TLabel").pack(anchor="w")
        
        self.fig = Figure(figsize=(4, 1.8), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, 50)
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        self.fig.tight_layout(pad=1.0)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.line, = self.ax.plot([], [], marker='.', color=ACCENT_BLUE)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=tf)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        
        # Status Edge Computing & Log
        bottom_right = tk.Frame(parent, bg=BG_PANEL)
        bottom_right.pack(fill="x", padx=15, pady=(5, 20))
        
        ec_f = tk.Frame(bottom_right, bg=BG_PANEL, width=200)
        ec_f.pack(side="left", fill="both", padx=(0,15))
        
        ttk.Label(ec_f, text="Status Edge Computing", style="Header.TLabel").pack(anchor="w", pady=(0,10))
        
        ec_grid = tk.Frame(ec_f, bg=BG_PANEL)
        ec_grid.pack(fill="x")
        ec_grid.grid_columnconfigure(0, weight=1, minsize=110)
        ec_grid.grid_columnconfigure(1, weight=2)
        
        self._ec_row_idx = 0
        def _add_stat_row_grid(label, val_text=""):
            tk.Label(ec_grid, text=label, bg=BG_PANEL, font=UI_FONT, anchor="w").grid(row=self._ec_row_idx, column=0, sticky="w", pady=3)
            lbl = tk.Label(ec_grid, text=val_text, bg=BG_PANEL, font=UI_FONT, anchor="w")
            lbl.grid(row=self._ec_row_idx, column=1, sticky="w", pady=3)
            self._ec_row_idx += 1
            return lbl
            
        self.lbl_device = _add_stat_row_grid("Device", "Menunggu...")
        
        def _add_prog_grid(label):
            tk.Label(ec_grid, text=label, bg=BG_PANEL, font=UI_FONT, anchor="w").grid(row=self._ec_row_idx, column=0, sticky="w", pady=3)
            pf = tk.Frame(ec_grid, bg=BG_PANEL)
            pf.grid(row=self._ec_row_idx, column=1, sticky="w", pady=3)
            pb = ttk.Progressbar(pf, length=60, mode="determinate")
            pb.pack(side="left", padx=(0,5))
            val_lbl = tk.Label(pf, text="0%", bg=BG_PANEL, font=UI_FONT, anchor="w")
            val_lbl.pack(side="left")
            self._ec_row_idx += 1
            return pb, val_lbl
            
        self.pb_cpu, self.lbl_cpu_val = _add_prog_grid("CPU Usage")
        self.pb_ram, self.lbl_ram_val = _add_prog_grid("Memory Usage")
        self.lbl_inf_time = _add_stat_row_grid("Inference Time", "- ms")
        self.lbl_bw = _add_stat_row_grid("Bandwidth (Sim)", "1.25 Mbps")
        
        tk.Label(ec_grid, text="Status", bg=BG_PANEL, font=UI_FONT, anchor="w").grid(row=self._ec_row_idx, column=0, sticky="w", pady=3)
        self.lbl_ec_status = tk.Label(ec_grid, text="● Idle", bg=BG_PANEL, fg=TEXT_MUTED, font=UI_FONT_BOLD)
        self.lbl_ec_status.grid(row=self._ec_row_idx, column=1, sticky="w", pady=3)
        
        log_f = tk.Frame(bottom_right, bg=BG_PANEL)
        log_f.pack(side="left", fill="both", expand=True)
        ttk.Label(log_f, text="Log Aktivitas", style="Header.TLabel").pack(anchor="w", pady=(0,5))
        self.txt_log = scrolledtext.ScrolledText(log_f, font=MONO_FONT, bg="#f8f9fa", bd=1, relief="solid", height=10)
        self.txt_log.pack(fill="both", expand=True)

    def log_activity(self, msg):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{t}] {msg}\n"
        self.txt_log.insert("end", line)
        self.txt_log.see("end")
        self.txt_log_full.insert("end", line)
        self.txt_log_full.see("end")
        print(line.strip())

    def _save_config(self):
        """Simpan pengaturan Telegram ke config.json."""
        cfg = {
            "tg_token":       self.entry_tg_token.get(),
            "tg_chat_id":     self.entry_tg_chatid.get(),
            "thr_waspada":    self.var_thr_waspada.get(),
            "thr_kritis":     self.var_thr_kritis.get(),
            "alert_cooldown": self.var_alert_cooldown.get(),
            "send_snapshot":  self.var_send_snapshot.get(),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
                json.dump(cfg, fp, indent=2)
            self.log_activity(f"Pengaturan disimpan ke {CONFIG_PATH.name}")
        except Exception as e:
            self.log_activity(f"Gagal menyimpan config: {e}")

    def _load_config(self):
        """Muat pengaturan Telegram dari config.json jika ada."""
        if not CONFIG_PATH.exists():
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                cfg = json.load(fp)
            if cfg.get("tg_token"):
                self.entry_tg_token.delete(0, tk.END)
                self.entry_tg_token.insert(0, cfg["tg_token"])
            if cfg.get("tg_chat_id"):
                self.entry_tg_chatid.delete(0, tk.END)
                self.entry_tg_chatid.insert(0, cfg["tg_chat_id"])
            if cfg.get("thr_waspada"):
                self.var_thr_waspada.set(cfg["thr_waspada"])
            if cfg.get("thr_kritis"):
                self.var_thr_kritis.set(cfg["thr_kritis"])
            if cfg.get("alert_cooldown"):
                self.var_alert_cooldown.set(cfg["alert_cooldown"])
            if "send_snapshot" in cfg:
                self.var_send_snapshot.set(cfg["send_snapshot"])
            self.log_activity("Pengaturan Telegram dimuat dari config.json")
        except Exception as e:
            self.log_activity(f"Gagal memuat config: {e}")

    def _start_system_monitor(self):
        def _monitor():
            while True:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                try:
                    self.root.after(0, self._update_sys_ui, cpu, ram)
                except:
                    break
                time.sleep(1)
        threading.Thread(target=_monitor, daemon=True).start()

    def _update_sys_ui(self, cpu, ram):
        self.pb_cpu['value'] = cpu
        self.lbl_cpu_val.config(text=f"{cpu}%")
        self.pb_ram['value'] = ram
        self.lbl_ram_val.config(text=f"{ram}%")

    def _browse_source(self):
        path = filedialog.askopenfilename(filetypes=[("Video/Images", "*.mp4 *.avi *.jpg *.png"), ("All", "*.*")])
        if path:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, path)

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Pilih Model Weights", filetypes=[("YOLO Models", "*.pt *.engine *.onnx"), ("All Files", "*.*")])
        if path:
            self.cb_model.set(path)
            vals = list(self.cb_model['values'])
            if path not in vals:
                vals.append(path)
                self.cb_model['values'] = vals

    def _reload_model(self):
        path = self.cb_model.get()
        if os.path.exists(path):
            self.model_path = path
            threading.Thread(target=self._load_model, daemon=True).start()
        else:
            messagebox.showerror("Error", "Path model tidak ditemukan!")

    def _load_model(self):
        sel_device = getattr(self, 'var_device', None)
        dev_str = sel_device.get() if sel_device else DEVICE
        self.log_activity(f"Loading model: {self.model_path} (Target: {dev_str})...")
        self.model = None
        try:
            self.model = load_yolo_model(self.model_path)
            self.log_activity("Model loaded successfully.")
            self.root.after(0, lambda: self.lbl_model_stat.config(text=f"Model: {Path(self.model_path).stem}"))
        except Exception as e:
            self.log_activity(f"Error loading model: {e}")

    # ── ROI Logic ──
    def _toggle_roi(self):
        self._is_drawing_roi = not self._is_drawing_roi
        if self._is_drawing_roi:
            self.roi_points = []
            self.btn_roi_draw.config(text="Selesai (Klik Kanan)")
            self.log_activity("Mode ROI aktif. Klik kiri pada kanvas.")
            self.lbl_roi_stat.config(text="ROI: Drawing", fg=ACCENT_ORANGE)
        else:
            self.btn_roi_draw.config(text="Buat Polygon ROI")
            self.log_activity(f"ROI diset dengan {len(self.roi_points)} titik.")
            if len(self.roi_points) >= 3:
                self.lbl_roi_stat.config(text="ROI: Active", fg=ACCENT_GREEN)
            else:
                self.lbl_roi_stat.config(text="ROI: Inactive", fg=TEXT_MUTED)
            self._redraw_canvas()

    def _clear_roi(self):
        self.roi_points = []
        self._is_drawing_roi = False
        self.btn_roi_draw.config(text="Buat Polygon ROI")
        self.lbl_roi_stat.config(text="ROI: Inactive", fg=TEXT_MUTED)
        self.log_activity("ROI dibersihkan.")
        self._redraw_canvas()

    def _on_canvas_click(self, event):
        if not self._is_drawing_roi: return
        canvas = event.widget
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if self.last_frame is None: return
        
        h, w = self.last_frame.shape[:2]
        scale = min(cw/w, ch/h) * getattr(self, 'zoom_factor', 1.0)
        nw, nh = int(w*scale), int(h*scale)
        img_x0 = (cw - nw) // 2
        img_y0 = (ch - nh) // 2
        
        rx = int((event.x - img_x0) / scale)
        ry = int((event.y - img_y0) / scale)
        
        if 0 <= rx <= w and 0 <= ry <= h:
            self.roi_points.append((rx, ry))
            self._redraw_canvas()

    def _on_canvas_right_click(self, event):
        if self._is_drawing_roi:
            self._toggle_roi()

    def _redraw_canvas(self):
        if self.last_frame is None: return
        img_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        if self.roi_points:
            draw = ImageDraw.Draw(pil_img, "RGBA")
            if len(self.roi_points) >= 3:
                draw.polygon(self.roi_points, outline=(50, 255, 50, 255), width=2, fill=(50, 255, 50, 40))
            else:
                for pt in self.roi_points:
                    draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=(50,255,50))
                    
        self._show_pil_on_canvas(pil_img)

    def _show_pil_on_canvas(self, pil_img, fps=0, inf_time=0, patch_cnt=0, obj_cnt=0):
        self._last_pil_img = pil_img.copy()
        canvas = self.canvas_video
        canvas.update_idletasks()
        cw, ch = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
        w, h = pil_img.size
        
        scale = min(cw/w, ch/h) * getattr(self, 'zoom_factor', 1.0)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        
        pil_r = pil_img.resize((nw, nh), Image.LANCZOS)
        
        # Add Overlays
        draw = ImageDraw.Draw(pil_r, "RGBA")
        
        # ROI text top left
        try:
            _fnt_roi = ImageFont.truetype("arial.ttf", 16)
        except:
            _fnt_roi = ImageFont.load_default()
        if self.roi_points and len(self.roi_points) >= 3:
            draw.text((10, 10), "ROI (Polygon)", fill=(50,255,50,255), font=_fnt_roi)
            
        # Top right FPS
        fps_text = f"FPS: {fps:.1f} | Inference: {inf_time:.1f} ms"
        try: fnt = ImageFont.truetype("arial.ttf", 14)
        except: fnt = ImageFont.load_default()
        bbox = draw.textbbox((0,0), fps_text, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([nw-tw-20, 10, nw-10, 10+th+10], fill=(0,0,0,150))
        draw.text((nw-tw-15, 15), fps_text, fill=(255,255,255,255), font=fnt)
        
        # Bottom left stats
        b_txt = f"Slicing: {patch_cnt} tiles\nDeteksi: {obj_cnt} objek"
        draw.rectangle([10, nh-50, 150, nh-10], fill=(0,0,0,150))
        draw.text((15, nh-45), b_txt, fill=(255,255,255,255), font=fnt)
        
        # Legend bottom right
        draw.rectangle([nw-120, nh-90, nw-10, nh-10], fill=(0,0,0,150))
        draw.rectangle([nw-110, nh-80, nw-100, nh-70], fill=CLASS_COLORS_RGB[4]) # plastik orange
        draw.text((nw-90, nh-82), "plastik", fill=(255,255,255,255), font=fnt)
        draw.rectangle([nw-110, nh-60, nw-100, nh-50], fill=CLASS_COLORS_RGB[0]) # botol red
        draw.text((nw-90, nh-62), "botol", fill=(255,255,255,255), font=fnt)
        draw.rectangle([nw-110, nh-40, nw-100, nh-30], fill=CLASS_COLORS_RGB[5]) # styrofoam purple
        draw.text((nw-90, nh-42), "styrofoam", fill=(255,255,255,255), font=fnt)

        tk_img = ImageTk.PhotoImage(pil_r)
        canvas.delete("all")
        canvas.create_image(cw//2, ch//2, anchor="center", image=tk_img)
        canvas.image = tk_img 

    # ── Button Functions (Zoom & Save) ──
    def _zoom_in(self):
        self.zoom_factor *= 1.2
        if not self._running: self._redraw_canvas()

    def _zoom_out(self):
        self.zoom_factor /= 1.2
        if not self._running: self._redraw_canvas()

    def _zoom_reset(self):
        self.zoom_factor = 1.0
        if not self._running: self._redraw_canvas()

    def _save_snapshot_quick(self):
        if getattr(self, '_last_pil_img', None) is None:
            messagebox.showwarning("Warning", "Tidak ada frame untuk disimpan!")
            return
        os.makedirs("snapshots", exist_ok=True)
        fname = f"snapshots/snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        self._last_pil_img.save(fname)
        self.log_activity(f"Snapshot cepat tersimpan: {fname}")
        messagebox.showinfo("Info", f"Snapshot tersimpan di {fname}")

    def _save_snapshot_dialog(self):
        if getattr(self, '_last_pil_img', None) is None:
            messagebox.showwarning("Warning", "Tidak ada frame untuk disimpan!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        if path:
            self._last_pil_img.save(path)
            self.log_activity(f"Snapshot tersimpan: {path}")
            messagebox.showinfo("Info", f"Snapshot tersimpan di {path}")

    # ── Detection Loop ──
    def _start_detection(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Model belum diload!")
            return
            
        src = 0 if self.input_type.get() == "webcam" else self.entry_source.get()
        if not src and src != 0:
            messagebox.showwarning("Warning", "Sumber input kosong!")
            return
            
        self.video_cap = cv2.VideoCapture(src)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", f"Gagal membuka {src}")
            return
            
        self._running = True
        self.btn_start.config(state="disabled")
        self.lbl_status.config(text="Status: Running", fg=ACCENT_GREEN)
        self.lbl_ec_status.config(text="● Running", fg=ACCENT_GREEN)
        self.lbl_source.config(text=f"Sumber: {src}")
        
        sel_dev = self.var_device.get()
        self.lbl_device.config(text=f"Local Edge ({sel_dev.upper()})")
        
        if self.var_sahi_enable.get():
            self.lbl_sahi_stat.config(text="SAHI: Active", fg=ACCENT_GREEN)
        else:
            self.lbl_sahi_stat.config(text="SAHI: Inactive", fg=TEXT_MUTED)
            
        self.log_activity("Memulai deteksi...")
        
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(src, str) else 0
        self.fps_video = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        threading.Thread(target=self._process_loop, daemon=True).start()

    def _stop_detection(self):
        self._running = False
        if self.video_cap:
            self.video_cap.release()
        self.btn_start.config(state="normal")
        self.lbl_status.config(text="Status: Stopped", fg=TEXT_MUTED)
        self.lbl_ec_status.config(text="● Stopped", fg=TEXT_MUTED)
        self.log_activity("Deteksi dihentikan.")

    def _process_loop(self):
        cap = self.video_cap
        
        start_time = time.time()
        last_detect_time = 0.0  # Waktu terakhir deteksi dilakukan
        last_boxes = np.empty((0,4))
        last_scores = np.empty(0)
        last_cls_ids = np.empty(0, int)
        last_class_counts = {}
        last_coverage = 0.0
        last_inf_time = 0.0
        last_patch_cnt = 0
        frame_counter = 0
        
        while self._running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            
            # Baca parameter dari UI tiap frame (agar perubahan slider langsung berlaku)
            use_sahi = self.var_sahi_enable.get()
            sl_h = int(self.var_sl_h.get())
            sl_w = int(self.var_sl_w.get())
            ol_h = float(self.var_ol_h.get()) / 100.0
            ol_w = float(self.var_ol_w.get()) / 100.0
            conf = float(self.var_conf.get())
            iou = float(self.var_iou.get())
            
            if not ret:
                self.root.after(0, self._stop_detection)
                break
                
            self.last_frame = frame.copy()
            H, W = frame.shape[:2]
            frame_counter += 1
            
            now = time.perf_counter()
            sample_interval = float(self.var_sample_interval.get())
            
            # Hitung berapa frame yang harus di-skip berdasarkan interval dan FPS video
            # Jika belum waktunya deteksi, tampilkan frame terakhir dan lanjutkan
            if (now - last_detect_time) < sample_interval:
                # Tampilkan frame terkini tanpa menjalankan AI (hemat CPU/GPU)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_live = Image.fromarray(img_rgb)
                draw_boxes_pil(pil_live, last_boxes, last_scores, last_cls_ids, self.roi_points)
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.total_frames > 0 else frame_counter
                elapsed = time.time() - start_time
                tot_secs = (self.total_frames / self.fps_video) if (self.fps_video > 0 and self.total_frames > 0) else 0
                self.root.after(0, self._update_frame_ui, pil_live, 0, last_inf_time, frame_idx, len(last_boxes), last_coverage, last_class_counts, elapsed, tot_secs, last_patch_cnt)
                continue
            
            # ── Waktunya Deteksi ──
            last_detect_time = now
            t0 = time.perf_counter()
            
            patch_cnt = 0
            
            # ROI Crop: crop frame ke bounding box ROI sebelum masuk AI
            use_roi_crop = self.var_roi_crop.get() and len(self.roi_points) >= 3
            if use_roi_crop:
                roi_np = np.array(self.roi_points, dtype=np.int32)
                rx, ry, rw, rh = cv2.boundingRect(roi_np)
                # Pastikan tidak keluar dari batas frame
                rx, ry = max(0, rx), max(0, ry)
                rx2, ry2 = min(W, rx + rw), min(H, ry + rh)
                inference_frame = frame[ry:ry2, rx:rx2]
                x_offset, y_offset = rx, ry
                iH, iW = inference_frame.shape[:2]
            else:
                inference_frame = frame
                x_offset, y_offset = 0, 0
                iH, iW = H, W
            
            # Inference
            if use_sahi:
                slices = generate_slices(iH, iW, sl_h, sl_w, ol_h, ol_w)
                patch_cnt = len(slices)
                all_b, all_s, all_c = [], [], []
                
                for (x1, y1, x2, y2) in slices:
                    patch = inference_frame[y1:y2, x1:x2]
                    ph, pw = patch.shape[:2]
                    if pw != sl_w or ph != sl_h:
                        patch = cv2.resize(patch, (sl_w, sl_h))
                        sx, sy = (x2-x1)/sl_w, (y2-y1)/sl_h
                    else:
                        sx = sy = 1.0
                        
                    b, s, c = run_yolo_on_patch(self.model, patch, self.var_device.get(), conf, iou)
                    if len(b) > 0:
                        b = b.copy()
                        # Map ke koordinat inference_frame, lalu ke frame asli
                        b[:, [0,2]] = np.clip(b[:, [0,2]]*sx + x1, 0, iW) + x_offset
                        b[:, [1,3]] = np.clip(b[:, [1,3]]*sy + y1, 0, iH) + y_offset
                        all_b.append(b); all_s.append(s); all_c.append(c)
                        
                if all_b:
                    boxes, scores, cls_ids = multiclass_nms_numpy(
                        np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c), iou
                    )
                else:
                    boxes, scores, cls_ids = np.empty((0,4)), np.empty(0), np.empty(0, int)
            else:
                patch_cnt = 1
                with torch.no_grad():
                    res = self.model(inference_frame, conf=conf, iou=iou, device=self.var_device.get(), verbose=False)[0].boxes
                if res is not None and len(res) > 0:
                    boxes = res.xyxy.cpu().numpy()
                    scores = res.conf.cpu().numpy()
                    cls_ids = res.cls.cpu().numpy().astype(int)
                    # Map koordinat balik ke frame asli jika di-crop
                    if use_roi_crop:
                        boxes[:, [0,2]] += x_offset
                        boxes[:, [1,3]] += y_offset
                else:
                    boxes, scores, cls_ids = np.empty((0,4)), np.empty(0), np.empty(0, int)
                    
            # ROI Filter (masih dipakai untuk filter presisi polygon, setelah crop bounding box)
            boxes, scores, cls_ids = filter_boxes_by_roi(boxes, scores, cls_ids, getattr(self, "roi_points", []))
            
            # Stats Calculate
            t1 = time.perf_counter()
            inf_time = (t1 - t0) * 1000
            
            coverage = calculate_coverage(boxes, self.roi_points, (H, W))
            tot_obj = len(boxes)
            
            # Generate Table Data
            class_counts = {}
            for cid in cls_ids:
                name = GUI_CLASS_MAPPING.get(cid, "obj")
                class_counts[name] = class_counts.get(name, 0) + 1
            
            # Simpan hasil deteksi terakhir untuk frame yang di-skip
            last_boxes = boxes
            last_scores = scores
            last_cls_ids = cls_ids
            last_class_counts = class_counts
            last_coverage = coverage
            last_inf_time = inf_time
            last_patch_cnt = patch_cnt
                
            # Visualization
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_out = draw_boxes_pil(Image.fromarray(img_rgb), boxes, scores, cls_ids, self.roi_points)
            
            fps_display = 1000.0 / inf_time if inf_time > 0 else 0
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.total_frames > 0 else frame_counter
            
            elapsed = time.time() - start_time
            if self.fps_video > 0 and self.total_frames > 0:
                tot_secs = self.total_frames / self.fps_video
            else:
                tot_secs = 0
            
            # Update UI safely
            self.root.after(0, self._update_frame_ui, pil_out, fps_display, inf_time, frame_idx, tot_obj, coverage, class_counts, elapsed, tot_secs, patch_cnt)
            
            # Control Logic
            self._check_alert_logic(coverage, tot_obj)

    def _update_frame_ui(self, pil_out, fps, inf_time, f_idx, tot_obj, coverage, class_counts, elapsed, tot_secs, patch_cnt):
        self._show_pil_on_canvas(pil_out, fps, inf_time, patch_cnt, tot_obj)
        
        self.lbl_inf_time.config(text=f"{inf_time:.1f} ms")
        self.lbl_tot_obj.config(text=str(tot_obj))
        self.lbl_coverage.config(text=f"{coverage:.2f} %")
        
        # Format time
        def fmt_sec(s): return time.strftime('%H:%M:%S', time.gmtime(s))
        self.lbl_duration.config(text=f"Durasi: {fmt_sec(elapsed)} / {fmt_sec(tot_secs)}")
        
        if self.total_frames > 0:
            self.lbl_frame_info.config(text=f"Frame: {f_idx} / {self.total_frames}")
        else:
            self.lbl_frame_info.config(text=f"Frame: {f_idx}")
            
        # Update level penumpukan colors
        thr_w = float(self.var_thr_waspada.get())
        thr_k = float(self.var_thr_kritis.get())
        
        if coverage >= thr_k:
            self.lbl_level.config(text="KRITIS", fg=ACCENT_RED)
        elif coverage >= thr_w:
            self.lbl_level.config(text="WASPADA", fg=ACCENT_ORANGE)
        else:
            self.lbl_level.config(text="AMAN", fg=ACCENT_GREEN)
            
        # Update Table - optimized to avoid glitch (no destroy) and use grid
        for cls_name, vars in self.table_rows_vars.items():
            if cls_name in class_counts:
                count = class_counts[cls_name]
                pct = (count / tot_obj * 100) if tot_obj > 0 else 0
                vars["count"].config(text=str(count))
                vars["pct"].config(text=f"{pct:.1f}%")
                
                # Check if it's currently gridded
                if not vars["name"].winfo_ismapped():
                    r_idx = vars["row_idx"]
                    vars["dot"].grid(row=r_idx, column=0, pady=2)
                    vars["name"].grid(row=r_idx, column=1, sticky="w", pady=2)
                    vars["count"].grid(row=r_idx, column=2, sticky="ew", pady=2)
                    vars["pct"].grid(row=r_idx, column=3, sticky="e", pady=2)
            else:
                if vars["name"].winfo_ismapped():
                    vars["dot"].grid_forget()
                    vars["name"].grid_forget()
                    vars["count"].grid_forget()
                    vars["pct"].grid_forget()
                    
        self.lbl_tot_row.config(text=f"Total: {tot_obj}")
            
        # Update chart 
        if f_idx % 5 == 0 or self.total_frames == 0:
            now_str = datetime.datetime.now().strftime("%H:%M")
            self.time_history.append(now_str)
            self.coverage_history.append(coverage)
            
            self.line.set_data(range(len(self.coverage_history)), self.coverage_history)
            self.ax.set_xlim(0, max(10, len(self.coverage_history)-1))
            self.ax.set_xticks(range(len(self.time_history)))
            
            # Show tick every N items
            lbls = [t if i % max(1, len(self.time_history)//5) == 0 else "" for i, t in enumerate(self.time_history)]
            self.ax.set_xticklabels(lbls)
            
            # add bubble text at end
            [t.remove() for t in self.ax.texts]
            if len(self.coverage_history) > 0:
                self.ax.text(len(self.coverage_history)-1, coverage + 2, f"{coverage:.2f}%", color=ACCENT_BLUE, fontsize=8, ha='center')
                
            self.canvas_plot.draw()

    # ── Telegram Logic ──
    def _test_telegram(self):
        tok = self.entry_tg_token.get()
        cid = self.entry_tg_chatid.get()
        if not tok or not cid:
            messagebox.showwarning("Warning", "Token dan Chat ID harus diisi!")
            return
        
        self._save_config()  # Simpan otomatis saat klik tombol
        self.log_activity("Mengirim pesan test ke Telegram...")
        snap = getattr(self, '_last_pil_img', None)
        if self.var_send_snapshot.get() and snap is not None:
            threading.Thread(
                target=send_telegram_photo,
                args=(tok, cid, snap.copy(), "\u2705 <b>Test Notifikasi</b>\nIntegrasi Telegram dengan sistem deteksi sampah berhasil!"),
                daemon=True
            ).start()
        else:
            threading.Thread(target=send_telegram_alert, args=(tok, cid, "\u2705 <b>Test Notifikasi</b>\nIntegrasi Telegram dengan sistem deteksi sampah berhasil!"), daemon=True).start()
        messagebox.showinfo("Info", "Permintaan kirim ter-trigger. Cek Telegram Anda.")

    def _check_alert_logic(self, coverage, obj_count):
        thr_k = float(self.var_thr_kritis.get())
        thr_w = float(self.var_thr_waspada.get())
        
        if coverage >= thr_k:
            current_level = "KRITIS"
        elif coverage >= thr_w:
            current_level = "WASPADA"
        else:
            current_level = "AMAN"
            self.last_alert_level = getattr(self, 'last_alert_level', "AMAN")
            # Reset cooldown jika turun kembali ke AMAN (agar bisa alert lagi saat naik)
            if self.last_alert_level != "AMAN":
                self.last_alert_level = "AMAN"
            return
        
        now = time.time()
        cooldown = float(self.var_alert_cooldown.get()) if hasattr(self, 'var_alert_cooldown') else 300.0
        last_level = getattr(self, 'last_alert_level', "AMAN")
        
        # Kirim alert jika: level naik (baru kritis/waspada) ATAU sudah melewati cooldown
        level_escalated = (current_level == "KRITIS" and last_level != "KRITIS") or \
                          (current_level == "WASPADA" and last_level == "AMAN")
        cooldown_passed = (now - self.last_alert_time) > cooldown
        
        if level_escalated or cooldown_passed:
            self.last_alert_time = now
            self.last_alert_level = current_level
            tok = self.entry_tg_token.get()
            cid = self.entry_tg_chatid.get()
            if tok and cid:
                emoji = "\U0001f6a8" if current_level == "KRITIS" else "\u26a0\ufe0f"
                msg = (f"{emoji} <b>{current_level}</b>\n\n"
                       f"<b>Sistem Deteksi Sampah</b>\n"
                       f"Coverage Area: {coverage:.2f}%\n"
                       f"Total Objek: {obj_count}\n"
                       f"Waktu: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                snap = getattr(self, '_last_pil_img', None)
                send_snap = getattr(self.var_send_snapshot, 'get', lambda: False)()
                if send_snap and snap is not None:
                    threading.Thread(
                        target=send_telegram_photo,
                        args=(tok, cid, snap.copy(), msg),
                        daemon=True
                    ).start()
                else:
                    threading.Thread(target=send_telegram_alert, args=(tok, cid, msg), daemon=True).start()
                self.log_activity(f"Telegram Alert: {current_level} (coverage {coverage:.1f}%){' + snapshot' if send_snap else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/best_yolov8n.pt", help="Path model weights")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = AdvancedTrashGUI(root, args.model)
    root.mainloop()

if __name__ == "__main__":
    main()
