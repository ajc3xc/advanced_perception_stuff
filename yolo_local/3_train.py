#!/usr/bin/env python3
"""
_3_train.py — Train YOLO11m-seg on the bamboo dataset.

Run with:
    nohup pixi run python _3_train.py > train.log 2>&1 &
    tail -f train.log
"""

import matplotlib
matplotlib.use('Agg')  # MUST be before ultralytics import — fixes silent crash on headless nodes

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

if not YAML_PATH.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {YAML_PATH}\nRun _2_convert_masks.py first.")

# ── heartbeat: prints every 60s so tail -f train.log shows it's alive ─────────
def _heartbeat():
    i = 0
    while True:
        time.sleep(60)
        i += 1
        print(f"[heartbeat] still running — {i} min elapsed", flush=True)

threading.Thread(target=_heartbeat, daemon=True).start()

# ── launch tensorboard in background on port 6007 ─────────────────────────────
try:
    tb = subprocess.Popen(
        ["tensorboard", "--logdir", str(RUNS_DIR), "--port", "6007",
         "--host", "0.0.0.0", "--reload_interval", "10"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[tensorboard] PID={tb.pid}  →  forward port 6007 in VS Code Ports tab", flush=True)
except Exception as e:
    print(f"[tensorboard] could not start: {e} — training continues anyway", flush=True)

# ── train ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO

    print(f"[train] starting — output → {RUNS_DIR / RUN_NAME}", flush=True)
    model   = YOLO("yolo11m-seg.pt")
    results = model.train(
        data    = str(YAML_PATH),
        project = str(RUNS_DIR),
        name    = RUN_NAME,

        imgsz   = 1280,
        batch   = 64,
        cache   = "disk",
        workers = 8,

        epochs        = 100,
        patience      = 20,
        optimizer     = "AdamW",
        lr0           = 0.001,
        warmup_epochs = 3,
        cos_lr        = True,
        amp           = True,

        mosaic     = 1.0,
        copy_paste = 0.3,
        fliplr     = 0.5,
        flipud     = 0.0,
        degrees    = 5.0,
        scale      = 0.5,
        hsv_h      = 0.02,
        hsv_s      = 0.5,
        hsv_v      = 0.4,

        device   = DEVICE,
        save     = True,
        plots    = True,
        val      = True,
        exist_ok = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"\n❌ ERROR: training finished but best.pt missing at {best}", flush=True)
        sys.exit(1)

    print(f"\n✅ Training complete", flush=True)
    print(f"   Best weights : {best}", flush=True)
    print(f"   Results dir  : {results.save_dir}", flush=True)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user.", flush=True)
    sys.exit(1)
except Exception:
    import traceback
    print("\n❌ TRAINING CRASHED:", flush=True)
    traceback.print_exc()
    sys.exit(1)
