"""
eval_sahi_experiments.py
========================
Jalankan sendiri di terminal:

  # GPU saja (cepat, ~30-60 menit):
  python eval_sahi_experiments.py --gpu-only

  # GPU + CPU (lama, ~3-4 jam):
  python eval_sahi_experiments.py

  # Hanya konfigurasi tertentu:
  python eval_sahi_experiments.py --gpu-only --configs B0 S4 S5 S6

  # Tanpa mAP@0.5:0.95 (lebih cepat ~2x):
  python eval_sahi_experiments.py --gpu-only --no-map95
"""

import argparse
import os, sys, time, csv, json, datetime
import numpy as np
import cv2
import torch
import psutil
from pathlib import Path
from collections import defaultdict
import torchvision.ops as ops_tv

# ── Parse argumen ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-only",  action="store_true",
                    help="Hanya evaluasi GPU, skip CPU (lebih cepat)")
parser.add_argument("--cpu-only",  action="store_true",
                    help="Hanya evaluasi CPU, skip GPU")
parser.add_argument("--no-map95",  action="store_true",
                    help="Skip mAP@0.5:0.95 (hemat waktu ~2x)")
parser.add_argument("--configs",   nargs="*", default=None,
                    help="Konfigurasi yang dijalankan, mis: B0 S4 S5 S6")
parser.add_argument("--val-dir",   type=str, default=None,
                    help="Override folder validasi. Contoh: --val-dir path/ke/valid")
parser.add_argument("--max-images", type=int, default=None,
                    help="Batasi jumlah gambar yang dievaluasi (untuk tes FPS/CPU cepat)")
args = parser.parse_args()

# ── Konfigurasi ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
WEIGHT      = str(ROOT / "weights" / "best_yolov8sclass.pt")

# ── Folder validasi — bisa di-override dengan --val-dir ──────────────────────
if args.val_dir:
    VAL_BASE    = Path(args.val_dir)
    VAL_IMG_DIR = VAL_BASE / "images"
    VAL_LBL_DIR = VAL_BASE / "labels"
else:
    VAL_IMG_DIR = ROOT / "data" / "datasets" / "RFT" / "images" / "val"
    VAL_LBL_DIR = ROOT / "data" / "datasets" / "RFT" / "labels" / "val"

OUTPUT_DIR  = ROOT / "eval_output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"[INFO] Val images : {VAL_IMG_DIR}")
print(f"[INFO] Val labels : {VAL_LBL_DIR}")
if not VAL_IMG_DIR.exists():
    print(f"[ERROR] Folder tidak ditemukan: {VAL_IMG_DIR}")
    sys.exit(1)

CLASS_NAMES = ['bottle','grass','branch','milk-box','plastic-bag',
               'plastic-garbage','ball','leaf','pile']
NC          = len(CLASS_NAMES)
SMALL_RATIO = 0.05
CONF        = 0.25
IOU_NMS     = 0.45
IOU_MATCH   = 0.50

ALL_SAHI = {
    "S1": (320, 320, 0.10, 0.10),
    "S2": (320, 320, 0.20, 0.20),
    "S3": (320, 320, 0.30, 0.30),
    "S4": (512, 512, 0.10, 0.10),
    "S5": (512, 512, 0.20, 0.20),
    "S6": (512, 512, 0.30, 0.30),
    "S7": (640, 640, 0.10, 0.10),
    "S8": (640, 640, 0.20, 0.20),
    "S9": (640, 640, 0.30, 0.30),
}

# Filter konfigurasi jika --configs diberikan
ALL_ORDER = ["B0"] + [f"S{i}" for i in range(1, 10)]
if args.configs:
    RUN_ORDER = [c for c in ALL_ORDER if c in args.configs]
    SAHI_CONFIGS = {k: v for k, v in ALL_SAHI.items() if k in args.configs}
else:
    RUN_ORDER = ALL_ORDER
    SAHI_CONFIGS = ALL_SAHI

CATATAN = {
    "B0":"Baseline tanpa SAHI",
    "S1":"320 px, overlap 10%","S2":"320 px, overlap 20%","S3":"320 px, overlap 30%",
    "S4":"512 px, overlap 10%","S5":"512 px, overlap 20%","S6":"512 px, overlap 30%",
    "S7":"640 px, overlap 10%","S8":"640 px, overlap 20%","S9":"640 px, overlap 30%",
}

# ── Utility ───────────────────────────────────────────────────────────────────

def iou_box(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0.0

def load_labels(path, W, H):
    boxes=[]
    if not Path(path).exists(): return boxes
    with open(path) as f:
        for line in f:
            p=line.strip().split()
            if len(p)<5: continue
            c=int(p[0]); xc,yc,bw,bh=map(float,p[1:5])
            boxes.append((c,(xc-bw/2)*W,(yc-bh/2)*H,(xc+bw/2)*W,(yc+bh/2)*H))
    return boxes

def generate_slices(H, W, sh, sw, oh, ow):
    stride_h=int(sh*(1-oh)); stride_w=int(sw*(1-ow)); slices=[]
    y=0
    while True:
        y1=y; y2=min(y+sh,H)
        if y2==H: y1=max(0,H-sh)
        x=0
        while True:
            x1=x; x2=min(x+sw,W)
            if x2==W: x1=max(0,W-sw)
            slices.append((x1,y1,x2,y2))
            if x2==W: break
            x+=stride_w
        if y2==H: break
        y+=stride_h
    return slices

def nms_numpy(boxes, scores, cls_ids, iou_thr=0.45):
    if len(boxes)==0: return boxes, scores, cls_ids
    bt=torch.from_numpy(boxes.astype(np.float32))
    st=torch.from_numpy(scores.astype(np.float32))
    ct=torch.from_numpy(cls_ids.astype(np.float32))
    keep=ops_tv.nms(bt+ct[:,None]*(bt.max()+1), st, iou_thr).numpy()
    return boxes[keep], scores[keep], cls_ids[keep]

def compute_ap_11pt(rec, pre):
    ap=0.0
    for t in np.linspace(0,1,11):
        p=pre[rec>=t]; ap+=(np.max(p) if len(p)>0 else 0.0)
    return ap/11.0

# ── Metric Computation ────────────────────────────────────────────────────────

def _match_at_iou(all_preds, all_gts, iou_thr, small_only):
    gt_by_img = defaultdict(list)
    n_gt = defaultdict(int)
    for (img_idx,c,x1,y1,x2,y2,ar) in all_gts:
        if small_only and ar >= SMALL_RATIO: continue
        gt_by_img[img_idx].append([c,x1,y1,x2,y2,False])
        n_gt[c] += 1
    tp = defaultdict(list); fp = defaultdict(list)
    for pred in sorted(all_preds, key=lambda x: -x[2]):
        img_idx,pc,score,px1,py1,px2,py2 = pred
        gts = gt_by_img.get(img_idx,[])
        best=0.0; bj=-1
        for j,(gc,gx1,gy1,gx2,gy2,used) in enumerate(gts):
            if gc!=pc or used: continue
            v=iou_box([px1,py1,px2,py2],[gx1,gy1,gx2,gy2])
            if v>best: best=v; bj=j
        if best>=iou_thr and bj>=0:
            gts[bj][5]=True; tp[pc].append(1); fp[pc].append(0)
        else:
            tp[pc].append(0); fp[pc].append(1)
    return tp, fp, n_gt

def _ap_from_tp_fp(tp, fp, n_gt):
    ap={}; prec={}; rec={}; f1={}
    for c in range(NC):
        if n_gt[c]==0 or len(tp[c])==0:
            ap[c]=0.0; prec[c]=0.0; rec[c]=0.0; f1[c]=0.0; continue
        tpc=np.cumsum(tp[c]); fpc=np.cumsum(fp[c])
        r=tpc/(n_gt[c]+1e-10); p=tpc/(tpc+fpc+1e-10)
        ap[c]=float(compute_ap_11pt(r,p))
        prec[c]=float(p[-1]); rec[c]=float(r[-1])
        f1[c]=2*prec[c]*rec[c]/(prec[c]+rec[c]+1e-10)
    return ap, prec, rec, f1

def compute_metrics(all_preds, all_gts, small_only=False, compute_map95=True):
    """Hitung P, R, F1, mAP@0.5, dan (opsional) mAP@0.5:0.95."""
    tp50,fp50,ng50 = _match_at_iou(all_preds, all_gts, 0.50, small_only)
    ap50,p50,r50,f150 = _ap_from_tp_fp(tp50,fp50,ng50)
    cls_ok = [c for c in range(NC) if ng50[c]>0]

    mAP50 = float(np.mean([ap50[c] for c in cls_ok])) if cls_ok else 0.0
    mP    = float(np.mean([p50[c]  for c in cls_ok])) if cls_ok else 0.0
    mR    = float(np.mean([r50[c]  for c in cls_ok])) if cls_ok else 0.0
    mF1   = float(np.mean([f150[c] for c in cls_ok])) if cls_ok else 0.0

    # mAP@0.5:0.95 (opsional, ~10x lebih lambat)
    ap5095 = {c: 0.0 for c in range(NC)}
    mAP5095 = 0.0
    if compute_map95:
        print("    > Menghitung mAP@0.5:0.95 (10 IoU thresholds)...")
        ap5095_list = {c:[] for c in range(NC)}
        for iou_t in np.arange(0.50, 1.00, 0.05):
            tp_t,fp_t,ng_t = _match_at_iou(all_preds, all_gts, float(iou_t), small_only)
            ap_t,_,_,_ = _ap_from_tp_fp(tp_t,fp_t,ng_t)
            for c in range(NC): ap5095_list[c].append(ap_t[c])
        ap5095 = {c: float(np.mean(ap5095_list[c])) for c in range(NC)}
        mAP5095 = float(np.mean([ap5095[c] for c in cls_ok])) if cls_ok else 0.0

    return {
        "mP":mP,"mR":mR,"mF1":mF1,"mAP50":mAP50,"mAP5095":mAP5095,
        "per_class":{c:{"P":p50[c],"R":r50[c],"F1":f150[c],
                        "AP50":ap50[c],"AP5095":ap5095[c],"n_gt":ng50[c]}
                     for c in range(NC)}
    }

# ── Inference ─────────────────────────────────────────────────────────────────

def run_baseline(model, img, device):
    with torch.no_grad():
        res=model(img,conf=CONF,iou=IOU_NMS,device=device,verbose=False)[0].boxes
    if res is None or len(res)==0:
        return np.empty((0,4)),np.empty(0),np.empty(0,int)
    return res.xyxy.cpu().numpy(),res.conf.cpu().numpy(),res.cls.cpu().numpy().astype(int)

def run_sahi(model, img, device, sh, sw, oh, ow):
    H,W=img.shape[:2]; slices=generate_slices(H,W,sh,sw,oh,ow)
    ab,as_,ac=[],[],[]
    for (x1,y1,x2,y2) in slices:
        patch=img[y1:y2,x1:x2]
        with torch.no_grad():
            res=model(patch,conf=CONF,iou=IOU_NMS,device=device,verbose=False)[0].boxes
        if res is not None and len(res)>0:
            b=res.xyxy.cpu().numpy().copy()
            b[:,[0,2]]=np.clip(b[:,[0,2]]+x1,0,W)
            b[:,[1,3]]=np.clip(b[:,[1,3]]+y1,0,H)
            ab.append(b); as_.append(res.conf.cpu().numpy())
            ac.append(res.cls.cpu().numpy().astype(int))
    if ab:
        boxes,scores,cls_ids=nms_numpy(
            np.concatenate(ab),np.concatenate(as_),np.concatenate(ac),IOU_NMS)
    else:
        boxes,scores,cls_ids=np.empty((0,4)),np.empty(0),np.empty(0,int)
    return boxes,scores,cls_ids,len(slices)

# ── Evaluate One Config ───────────────────────────────────────────────────────

def evaluate_config(model, name, device, sahi_params=None, compute_map95=True):
    dev_str = "GPU" if "cuda" in device else "CPU"
    extra = f" | SAHI {sahi_params[0]}x{sahi_params[1]} ov={int(sahi_params[2]*100)}%" if sahi_params else ""
    print(f"\n{'='*60}")
    print(f"  [{name}] {dev_str}{extra}")
    print(f"{'='*60}")

    imgs = sorted(VAL_IMG_DIR.glob("*.jpg")) or sorted(VAL_IMG_DIR.glob("*.png"))
    if args.max_images:
        # Ambil subset gambar secara merata agar representatif
        step = max(1, len(imgs) // args.max_images)
        imgs = imgs[::step][:args.max_images]
    
    n_imgs = len(imgs)
    all_preds=[]; all_gts=[]; times=[]; cpus=[]; rams=[]; patches=[]

    t_start = time.time()
    for idx, img_path in enumerate(imgs):
        img=cv2.imread(str(img_path))
        if img is None: continue
        H,W=img.shape[:2]
        lbl=VAL_LBL_DIR/(img_path.stem+".txt")
        for (c,x1,y1,x2,y2) in load_labels(str(lbl),W,H):
            all_gts.append((idx,c,x1,y1,x2,y2,((x2-x1)*(y2-y1))/(H*W)))

        cb=psutil.cpu_percent(interval=None); rb=psutil.virtual_memory().percent
        t0=time.perf_counter()
        if sahi_params is None:
            boxes,scores,cls_ids=run_baseline(model,img,device); np_=1
        else:
            sh,sw,oh,ow=sahi_params
            boxes,scores,cls_ids,np_=run_sahi(model,img,device,sh,sw,oh,ow)
        t1=time.perf_counter()
        ca=psutil.cpu_percent(interval=None); ra=psutil.virtual_memory().percent

        lat=(t1-t0)*1000
        times.append(lat); cpus.append(max(cb,ca))
        rams.append(max(rb,ra)); patches.append(np_)
        for i in range(len(boxes)):
            all_preds.append((idx,int(cls_ids[i]),float(scores[i]),*boxes[i].tolist()))

        # Progress + ETA setiap 50 gambar
        if (idx+1)%50==0 or (idx+1)==n_imgs:
            avg_lat = np.mean(times)
            remaining = n_imgs - (idx+1)
            eta_s = remaining * avg_lat / 1000
            eta_str = f"{int(eta_s//60)}m{int(eta_s%60)}s" if eta_s>60 else f"{eta_s:.0f}s"
            print(f"  [{idx+1:>3}/{n_imgs}] avg={avg_lat:.0f}ms/img | "
                  f"FPS={1000/avg_lat:.1f} | ETA={eta_str} | patches={np_:.0f}")

    avg = float(np.mean(times))
    pf  = {"lat_ms":round(avg,2),
           "fps":round(1000/avg if avg>0 else 0,2),
           "cpu_pct":round(float(np.mean(cpus)),1),
           "ram_pct":round(float(np.mean(rams)),1),
           "avg_patches":round(float(np.mean(patches)),1)}

    print(f"  Menghitung metrik...")
    ma = compute_metrics(all_preds, all_gts, small_only=False, compute_map95=compute_map95)
    ms = compute_metrics(all_preds, all_gts, small_only=True,  compute_map95=False)  # small: cukup mAP50
    elapsed = time.time()-t_start
    map95_str = f" | mAP50:95={ma['mAP5095']:.4f}" if compute_map95 else ""
    print(f"  DONE [{elapsed/60:.1f} menit] | mAP50={ma['mAP50']:.4f}{map95_str} | FPS={pf['fps']:.1f}")
    return ma, ms, pf

# ── Print Tables ──────────────────────────────────────────────────────────────

def best_sahi(results, dev):
    keys=[f"{k}_{dev}" for k in ALL_ORDER[1:] if f"{k}_{dev}" in results]
    return max(keys, key=lambda k: results[k]["metrics_all"]["mAP50"]) if keys else None


def print_tabel_371(results, compute_map95):
    """Tabel 3.7.1 — Evaluasi Deteksi Keseluruhan."""
    for dev in ["GPU","CPU"]:
        rows=[f"{o}_{dev}" for o in RUN_ORDER if f"{o}_{dev}" in results]
        if not rows: continue
        W = 100 if compute_map95 else 80
        print(f"\n{'='*W}")
        print(f"  Tabel 3.7.1 — Hasil Evaluasi Deteksi Keseluruhan [{dev}]")
        print(f"{'='*W}")
        header = (f"{'Kode':<6} {'Precision':>10} {'Recall':>8} {'F1-score':>10} "
                  f"{'mAP@0.5':>9}")
        if compute_map95: header += f" {'mAP@0.5:0.95':>14}"
        header += "  Catatan"
        print(header); print("─"*W)
        for key in rows:
            kode=key.replace(f"_{dev}",""); m=results[key]["metrics_all"]
            row=(f"{kode:<6} {m['mP']:>10.4f} {m['mR']:>8.4f} {m['mF1']:>10.4f} "
                 f"{m['mAP50']:>9.4f}")
            if compute_map95: row += f" {m['mAP5095']:>14.4f}"
            row += f"  {CATATAN.get(kode,'')}"
            print(row)
        print("─"*W)


def print_tabel_372(results):
    """Tabel 3.7.2 — Evaluasi Objek Kecil (area_ratio < 5%)."""
    for dev in ["GPU","CPU"]:
        rows=[f"{o}_{dev}" for o in RUN_ORDER if f"{o}_{dev}" in results]
        if not rows: continue
        print(f"\n{'='*90}")
        print(f"  Tabel 3.7.2 — Hasil Evaluasi Objek Kecil [area_ratio < 5%] [{dev}]")
        print(f"{'='*90}")
        print(f"{'Kode':<6} {'Precision':>22} {'Recall':>20} {'F1':>16} {'mAP@0.5':>20}  Keterangan")
        print(f"{'':6} {'objek kecil':>22} {'objek kecil':>20} {'objek kecil':>16} {'objek kecil':>20}")
        print("─"*90)
        for key in rows:
            kode=key.replace(f"_{dev}",""); ms=results[key]["metrics_small"]
            n=sum(v["n_gt"] for v in ms["per_class"].values())
            print(f"{kode:<6} {ms['mP']:>22.4f} {ms['mR']:>20.4f} "
                  f"{ms['mF1']:>16.4f} {ms['mAP50']:>20.4f}  Area ratio < 5% (n={n})")
        print("─"*90)


def print_tabel_317(results):
    """
    Tabel 3.17 — Per Kelas, B0 vs SAHI terbaik (side-by-side).
    Kelas | Jumlah GT | Precision B0 | Recall B0 | mAP@0.5 B0 |
    Precision terbaik SAHI | Recall terbaik SAHI | mAP@0.5 terbaik SAHI
    """
    dev = "GPU" if any("GPU" in k for k in results) else "CPU"
    b0  = f"B0_{dev}"
    bk  = best_sahi(results, dev)
    if not bk or b0 not in results: return
    bkode = bk.replace(f"_{dev}","")
    b0_pc = results[b0]["metrics_all"]["per_class"]
    bk_pc = results[bk]["metrics_all"]["per_class"]
    W = 108
    print(f"\n{'='*W}")
    print(f"  Tabel 3.17 — Hasil Evaluasi Per Kelas [{dev}]")
    print(f"  SAHI terbaik: {bkode}  ({CATATAN.get(bkode,'')})")
    print(f"{'='*W}")
    print(f"{'Kelas':<18} {'Jumlah GT':>10}  "
          f"{'Precision B0':>13} {'Recall B0':>10} {'mAP@0.5 B0':>11}  "
          f"{'Precision':>11} {'Recall':>8} {'mAP@0.5':>9}")
    print(f"{'':18} {'':>10}  {'':>13} {'':>10} {'':>11}  "
          f"{'terbaik SAHI':>11} {'terbaik SAHI':>8} {'terbaik SAHI':>9}")
    print("─"*W)
    for c in range(NC):
        n  = b0_pc.get(c,{}).get("n_gt",0)
        pb = b0_pc.get(c,{}).get("P",0.0)
        rb = b0_pc.get(c,{}).get("R",0.0)
        ab = b0_pc.get(c,{}).get("AP50",0.0)
        ps = bk_pc.get(c,{}).get("P",0.0)
        rs = bk_pc.get(c,{}).get("R",0.0)
        as_= bk_pc.get(c,{}).get("AP50",0.0)
        print(f"{CLASS_NAMES[c]:<18} {n:>10}  "
              f"{pb:>13.4f} {rb:>10.4f} {ab:>11.4f}  "
              f"{ps:>11.4f} {rs:>8.4f} {as_:>9.4f}")
    print("─"*W)
    if b0 in results and bk in results:
        mb=results[b0]["metrics_all"]; mk=results[bk]["metrics_all"]
        print(f"{'Rata-rata (all)':<18} {'':>10}  "
              f"{mb['mP']:>13.4f} {mb['mR']:>10.4f} {mb['mAP50']:>11.4f}  "
              f"{mk['mP']:>11.4f} {mk['mR']:>8.4f} {mk['mAP50']:>9.4f}")


def print_tabel_318(results):
    """
    Tabel 3.18 — Kinerja CPU dan GPU.
    Mode | Kode | Latensi rata-rata (ms) | FPS rata-rata | CPU rata-rata (%) | RAM rata-rata | Patch rata-rata
    """
    print(f"\n{'='*90}")
    print(f"  Tabel 3.18 — Kinerja CPU dan GPU")
    print(f"{'='*90}")
    print(f"{'Mode':<6} {'Kode':<6} {'Latensi rata-rata':>19} {'FPS rata-rata':>14} "
          f"{'CPU rata-rata (%)':>18} {'RAM rata-rata':>14} {'Patch rata-rata':>16}")
    print(f"{'':6} {'':6} {'(ms)':>19} {'':14} {'':18} {'':14} {'':16}")
    print("─"*90)
    for dev in ["CPU","GPU"]:
        rows=[f"{o}_{dev}" for o in RUN_ORDER if f"{o}_{dev}" in results]
        if not rows: continue
        for key in rows:
            kode=key.replace(f"_{dev}",""); p=results[key]["perf"]
            print(f"{dev:<6} {kode:<6} {p['lat_ms']:>19.2f} {p['fps']:>14.2f} "
                  f"{p['cpu_pct']:>18.1f} {p['ram_pct']:>14.1f} {p['avg_patches']:>16.1f}")
        print("─"*90)


def print_rekomendasi(results):
    for dev in ["GPU","CPU"]:
        b0=f"B0_{dev}"; bk=best_sahi(results,dev)
        if not bk or b0 not in results: continue
        bkode=bk.replace(f"_{dev}","")
        bv=results[bk]; b0v=results[b0]
        dm   = bv['metrics_all']['mAP50']   - b0v['metrics_all']['mAP50']
        dm95 = bv['metrics_all']['mAP5095'] - b0v['metrics_all']['mAP5095']
        df   = bv['perf']['fps']             - b0v['perf']['fps']
        print(f"\n{'='*65}")
        print(f"  KONFIGURASI SAHI TERBAIK [{dev}]: {bkode}  ({CATATAN.get(bkode,'')})")
        print(f"  mAP@0.5      = {bv['metrics_all']['mAP50']:.4f}   (delta vs B0: {dm:+.4f})")
        if bv['metrics_all']['mAP5095'] > 0:
            print(f"  mAP@0.5:0.95 = {bv['metrics_all']['mAP5095']:.4f}   (delta vs B0: {dm95:+.4f})")
        print(f"  FPS          = {bv['perf']['fps']:.2f}             (delta vs B0: {df:+.2f})")
        print(f"  Patch rata-rata = {bv['perf']['avg_patches']:.1f}")
        print("="*65)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    use_gpu = torch.cuda.is_available()
    gpu_dev = "cuda:0" if use_gpu else None
    cpu_dev = "cpu"
    do_map95 = not args.no_map95

    print("="*65)
    print("  SAHI EVALUATION SUITE - Skripsi Trash Detection")
    print(f"  Model     : {WEIGHT}")
    print(f"  GPU only  : {args.gpu_only}")
    print(f"  mAP@0.5:95: {do_map95}")
    print(f"  Konfigurasi: {RUN_ORDER}")
    print("="*65)
    if use_gpu: print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Estimasi waktu
    n_cfg = len(RUN_ORDER)
    n_dev = 1 if (args.gpu_only or args.cpu_only) else (2 if use_gpu else 1)
    est_gpu = 40 * n_cfg   # ~40ms/img × 318 img / 1000 → ~13s per config GPU
    est_cpu = 850 * n_cfg  # ~850ms/img → ~270s per config CPU
    est_total = est_gpu + (0 if args.gpu_only else est_cpu)
    print(f"\n  Estimasi waktu: ~{est_total//60:.0f} menit "
          f"({n_cfg} konfigurasi × {n_dev} device)")
    if do_map95:
        print(f"  (+mAP@0.5:0.95: tambah ~50% waktu komputasi metrik)")
    print()

    model = YOLO(WEIGHT)
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # ── Jalankan evaluasi ─────────────────────────────────────────────────────
    for kode in RUN_ORDER:
        sparams = None if kode=="B0" else SAHI_CONFIGS.get(kode)
        if sparams is None and kode!="B0":
            continue  # skip jika kode tidak ada di SAHI_CONFIGS

        # GPU
        if use_gpu and not args.cpu_only:
            ma,ms,pf = evaluate_config(model, kode, gpu_dev, sparams, do_map95)
            results[f"{kode}_GPU"] = {"device":"GPU","metrics_all":ma,
                                      "metrics_small":ms,"perf":pf}
            if sparams: results[f"{kode}_GPU"]["sahi_params"]=list(sparams)

        # CPU (skip jika --gpu-only)
        if not args.gpu_only:
            ma,ms,pf = evaluate_config(model, kode, cpu_dev, sparams, do_map95)
            results[f"{kode}_CPU"] = {"device":"CPU","metrics_all":ma,
                                      "metrics_small":ms,"perf":pf}
            if sparams: results[f"{kode}_CPU"]["sahi_params"]=list(sparams)

    # ── Cetak semua tabel ─────────────────────────────────────────────────────
    print_tabel_371(results, do_map95)
    print_tabel_372(results)
    print_tabel_317(results)
    print_tabel_318(results)
    print_rekomendasi(results)

    # ── Simpan ────────────────────────────────────────────────────────────────
    csv_path  = OUTPUT_DIR / f"eval_results_{ts}.csv"
    json_path = OUTPUT_DIR / f"eval_results_{ts}.json"

    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["config","device",
                    "mP","mR","mF1","mAP50","mAP5095",
                    "small_mP","small_mR","small_mF1","small_mAP50",
                    "lat_ms","fps","cpu_pct","ram_pct","avg_patches"]
                   +[f"{n}_P"      for n in CLASS_NAMES]
                   +[f"{n}_R"      for n in CLASS_NAMES]
                   +[f"{n}_AP50"   for n in CLASS_NAMES]
                   +[f"{n}_AP5095" for n in CLASS_NAMES])
        for k,v in results.items():
            m=v["metrics_all"]; ms=v["metrics_small"]; p=v["perf"]; pc=m["per_class"]
            w.writerow([k,v["device"],
                        m["mP"],m["mR"],m["mF1"],m["mAP50"],m["mAP5095"],
                        ms["mP"],ms["mR"],ms["mF1"],ms["mAP50"],
                        p["lat_ms"],p["fps"],p["cpu_pct"],p["ram_pct"],p["avg_patches"]]
                       +[pc[c]["P"]      for c in range(NC)]
                       +[pc[c]["R"]      for c in range(NC)]
                       +[pc[c]["AP50"]   for c in range(NC)]
                       +[pc[c]["AP5095"] for c in range(NC)])

    with open(json_path,"w") as f: json.dump(results,f,indent=2)

    print(f"\nHasil tersimpan:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print("\nEvaluasi selesai!")

if __name__ == "__main__":
    main()
