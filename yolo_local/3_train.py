#!/usr/bin/env python3
"""
3_train.py — Train YOLO11m-seg on the bamboo dataset.
Tuned for RTX 3080 (10GB stock / 20GB modded) on Windows.

Run with:
    pixi run python 3_train.py
"""

import matplotlib
matplotlib.use('Agg')  # MUST be before ultralytics import

import sys
import time
import threading
import subprocess
from pathlib import Path

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path(r"C:/Users/13144/Documents/PhD/Robotics_ABE6399/advanced_perception_stuff/yolo_local/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

YOLO_ROOT = BASE / "yolo_dataset"
RUNS_DIR  = BASE / "yolo_runs"
YAML_PATH = YOLO_ROOT / "bamboo.yaml"
RUN_NAME  = "bamboo_seg_v1"
DEVICE    = 0

# ── 3080 VRAM settings ────────────────────────────────────────────────────────
# 10GB stock:  BATCH=8   IMGSZ=640
# 20GB modded: BATCH=16  IMGSZ=640  (or BATCH=4 IMGSZ=1280)
IMGSZ = 640
BATCH = 8

if not YAML_PATH.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {YAML_PATH}\nRun 2_convert_masks.py first.")

# ── heartbeat ─────────────────────────────────────────────────────────────────
'''def _heartbeat():
    i = 0
    while True:
        time.sleep(60)
        i += 1
        print(f"[heartbeat] still running — {i} min elapsed", flush=True)

threading.Thread(target=_heartbeat, daemon=True).start()
'''
# ── tensorboard ───────────────────────────────────────────────────────────────
try:
    tb = subprocess.Popen(
        ["tensorboard", "--logdir", str(RUNS_DIR), "--port", "6006",
         "--host", "127.0.0.1", "--reload_interval", "10"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[tensorboard] PID={tb.pid}  ->  http://localhost:6006", flush=True)
except Exception as e:
    print(f"[tensorboard] not available: {e} — continuing anyway", flush=True)

# ── train ─────────────────────────────────────────────────────────────────────
try:
    import torch
    print(f"[cuda] available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[cuda] device={torch.cuda.get_device_name(0)}", flush=True)
        print(f"[cuda] vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
    else:
        print("[cuda] WARNING: no CUDA — will train on CPU (very slow)", flush=True)

    from ultralytics import YOLO
    print(f"[train] imgsz={IMGSZ}  batch={BATCH}  device={DEVICE}", flush=True)
    print(f"[train] output -> {RUNS_DIR / RUN_NAME}", flush=True)
    print(f"[train] loading yolo11m-seg.pt ...", flush=True)

    model = YOLO("yolo11m-seg.pt")
    print(f"[train] model loaded — starting training loop ...", flush=True)

    results = model.train(
        data    = YAML_PATH.as_posix(),
        project = str(RUNS_DIR),
        name    = RUN_NAME,

        imgsz   = IMGSZ,
        batch   = BATCH,
        cache   = False,
        workers = 0,

        epochs        = 100,
        patience      = 20,
        optimizer     = "AdamW",
        lr0           = 0.001,
        warmup_epochs = 3,
        cos_lr        = True,
        amp           = True,

        mosaic     = 1.0,
        fliplr     = 0.5,
        scale      = 0.5,
        hsv_s      = 0.5,
        hsv_v      = 0.4,
        flipud     = 0.0,
        copy_paste = 0.0,
        degrees    = 0.0,
        hsv_h      = 0.0,

        device   = DEVICE,
        save     = True,
        plots    = True,
        val      = True,
        exist_ok = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"\n❌ best.pt missing at {best}", flush=True)
        sys.exit(1)

    print(f"\n✅ Training complete", flush=True)
    print(f"   Best weights : {best}", flush=True)
    print(f"   Results dir  : {results.save_dir}", flush=True)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted.", flush=True)
    sys.exit(1)
except Exception:
    import traceback
    print("\n❌ TRAINING CRASHED:", flush=True)
    traceback.print_exc()
    sys.exit(1)