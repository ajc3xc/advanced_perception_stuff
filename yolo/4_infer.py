#!/usr/bin/env python3
"""
4_infer.py — Run trained YOLO11m-seg on images or a video.
All paths auto-derived from BASE.

Usage:
    python 4_infer.py                        # runs on synthetic test frames
    python 4_infer.py --source /path/to/imgs
    python 4_infer.py --source /path/to/vid.mp4
    python 4_infer.py --weights /path/to/best.pt
"""

import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path("/blue/cli2/a.camerer/ABE6399_Robotics/advanced_robotic_systems/yolo/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

YOLO_ROOT  = BASE / "yolo_dataset"
RUNS_DIR   = BASE / "yolo_runs"
INFER_ROOT = BASE / "infer_results"

CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.45
IMGSZ          = 1280
MASK_ALPHA     = 0.45
MASK_COLOR     = (80, 255, 120)   # RGB

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_best_weights():
    """Auto-find the most recently trained best.pt under RUNS_DIR."""
    candidates = sorted(RUNS_DIR.rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under {RUNS_DIR}\nRun script 3 first."
        )
    return candidates[-1]


def draw_overlay(frame_bgr, result):
    out = frame_bgr.copy()
    if result.masks is None:
        return out
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    for i, m in enumerate(masks):
        if m.shape != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        binary = (m > 0.5).astype(np.uint8)
        layer  = np.zeros_like(out)
        layer[binary == 1] = MASK_COLOR[::-1]
        out = cv2.addWeighted(out, 1.0, layer, MASK_ALPHA, 0)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 200, 80), 2)
        if boxes is not None and i < len(boxes):
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.putText(out, f"bamboo {conf:.2f}", (xyxy[0], max(xyxy[1]-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 80), 2)
    return out


def infer_images(model, img_paths, out_dir):
    ovl_dir = out_dir / "overlays"
    ovl_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for img_path in img_paths:
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            continue
        r = model.predict(str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                          imgsz=IMGSZ, verbose=False)[0]
        n_inst = len(r.boxes) if r.boxes is not None else 0
        confs  = r.boxes.conf.cpu().tolist() if r.boxes is not None else []
        cv2.imwrite(str(ovl_dir / img_path.name),
                    draw_overlay(frame_bgr, r),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"file": img_path.name, "n_instances": n_inst,
                        "confidences": [round(c, 4) for c in confs]})
    return records


def infer_video(model, video_path, out_dir):
    ovl_dir = out_dir / "overlays"
    ovl_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp4 = out_dir / "overlay.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    records = []
    f_idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r      = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                               imgsz=IMGSZ, verbose=False)[0]
        n_inst = len(r.boxes) if r.boxes is not None else 0
        confs  = r.boxes.conf.cpu().tolist() if r.boxes is not None else []
        out_f  = draw_overlay(frame, r)
        cv2.imwrite(str(ovl_dir / f"{f_idx:06d}.jpg"), out_f,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        writer.write(out_f)
        records.append({"frame": f_idx, "n_instances": n_inst,
                        "confidences": [round(c, 4) for c in confs]})
        if f_idx % 30 == 0:
            print(f"  frame {f_idx}  instances={n_inst}", flush=True)
        f_idx += 1
    cap.release()
    writer.release()
    print(f"  video -> {mp4}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="Path to best.pt (auto-finds latest run if omitted)")
    ap.add_argument("--source",  default="SYNTH",
                    help="Image folder, video file, or SYNTH (default)")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else find_best_weights()
    print(f"Weights: {weights}")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = INFER_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    t0    = time.time()

    if args.source.upper() == "SYNTH":
        img_paths = sorted((YOLO_ROOT / "raw_frames").rglob("*.jpg"))
        print(f"SYNTH: {len(img_paths)} frames")
        records = infer_images(model, img_paths, out_dir)

    elif Path(args.source).suffix.lower() in VID_EXTS:
        records = infer_video(model, Path(args.source), out_dir)

    elif Path(args.source).is_dir():
        img_paths = sorted(p for p in Path(args.source).rglob("*")
                           if p.suffix.lower() in IMG_EXTS)
        print(f"Found {len(img_paths)} images")
        records = infer_images(model, img_paths, out_dir)

    else:
        records = infer_images(model, [Path(args.source)], out_dir)

    dt         = time.time() - t0
    all_confs  = [c for r in records for c in r.get("confidences", [])]
    total_inst = sum(r.get("n_instances", 0) for r in records)

    summary = {
        "weights":                str(weights),
        "source":                 args.source,
        "n_frames":               len(records),
        "total_instances":        total_inst,
        "avg_instances_per_frame": round(total_inst / max(len(records), 1), 2),
        "mean_conf":              round(float(np.mean(all_confs))  if all_confs else 0, 4),
        "min_conf":               round(float(np.min(all_confs))   if all_confs else 0, 4),
        "runtime_s":              round(dt, 2),
        "fps":                    round(len(records) / max(dt, 1e-6), 1),
        "output_dir":             str(out_dir),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone")
    print(f"  Frames              : {summary['n_frames']}")
    print(f"  Avg instances/frame : {summary['avg_instances_per_frame']}")
    print(f"  Mean confidence     : {summary['mean_conf']}")
    print(f"  Throughput          : {summary['fps']} fps")
    print(f"  Output              : {out_dir}")
    print("Next: python 5_check.py results")


if __name__ == "__main__":
    main()
