#!/usr/bin/env python3
"""
4_infer_roboflow.py — Run inference using Roboflow hosted API.

Usage:
    python _4_infer_roboflow.py                        # synthetic test frames
    python _4_infer_roboflow.py --source path/to/imgs
    python _4_infer_roboflow.py --source path/to/vid.mp4
"""

import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── EDIT THESE ────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "e20yDTNrBI9q2Sst9wA3"
WORKSPACE        = "xiaoyus-workspace-wtaot"
PROJECT          = "bamboo_synthetic"
MODEL_VERSION    = 1
BASE             = Path("synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

INFER_ROOT     = BASE / "infer_results"
CONF_THRESHOLD = 0.40
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MASK_COLOR     = (80, 255, 120)
MASK_ALPHA     = 0.45


def load_model():
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    return rf.workspace(WORKSPACE).project(PROJECT).version(MODEL_VERSION).model


def draw_overlay(frame_bgr, predictions):
    out = frame_bgr.copy()
    for pred in predictions:
        if pred.get("confidence", 0) < CONF_THRESHOLD:
            continue
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(out, f"bamboo {pred['confidence']:.2f}",
                    (x1, max(y1-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 80), 2)
        points = pred.get("points", [])
        if points:
            pts = np.array([[int(p["x"]), int(p["y"])] for p in points], np.int32)
            mask_layer = np.zeros_like(out)
            cv2.fillPoly(mask_layer, [pts], MASK_COLOR[::-1])
            out = cv2.addWeighted(out, 1.0, mask_layer, MASK_ALPHA, 0)
            cv2.polylines(out, [pts], True, (0, 200, 80), 2)
    return out


def infer_images(model, img_paths, out_dir):
    ovl_dir = out_dir / "overlays"
    ovl_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for img_path in img_paths:
        try:
            result    = model.predict(str(img_path), confidence=int(CONF_THRESHOLD*100)).json()
            preds     = result.get("predictions", [])
            frame_bgr = cv2.imread(str(img_path))
            if frame_bgr is not None:
                cv2.imwrite(str(ovl_dir / img_path.name),
                            draw_overlay(frame_bgr, preds),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({"file": img_path.name, "n_instances": len(preds),
                            "confidences": [round(p["confidence"], 4) for p in preds]})
        except Exception as e:
            print(f"  [error] {img_path.name}: {e}")
    return records


def infer_video(model, video_path, out_dir):
    ovl_dir = out_dir / "overlays"
    ovl_dir.mkdir(parents=True, exist_ok=True)
    cap    = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    records, f_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        tmp = out_dir / f"_tmp_{f_idx:06d}.jpg"
        cv2.imwrite(str(tmp), frame)
        try:
            preds = model.predict(str(tmp), confidence=int(CONF_THRESHOLD*100)).json().get("predictions", [])
        except Exception:
            preds = []
        tmp.unlink(missing_ok=True)
        out_frame = draw_overlay(frame, preds)
        cv2.imwrite(str(ovl_dir / f"{f_idx:06d}.jpg"), out_frame)
        writer.write(out_frame)
        records.append({"frame": f_idx, "n_instances": len(preds),
                        "confidences": [round(p["confidence"], 4) for p in preds]})
        if f_idx % 30 == 0:
            print(f"  frame {f_idx}  instances={len(preds)}", flush=True)
        f_idx += 1
    cap.release()
    writer.release()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="SYNTH")
    ap.add_argument("--out",    default=str(INFER_ROOT))
    args = ap.parse_args()

    print("Loading Roboflow model …")
    model   = load_model()
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.source.upper() == "SYNTH":
        img_paths = sorted((BASE / "yolo_dataset" / "raw_frames").rglob("*.jpg"))
        print(f"SYNTH: {len(img_paths)} frames")
        records = infer_images(model, img_paths, out_dir)
    elif Path(args.source).is_dir():
        img_paths = sorted(p for p in Path(args.source).rglob("*")
                           if p.suffix.lower() in IMG_EXTS)
        print(f"Found {len(img_paths)} images")
        records = infer_images(model, img_paths, out_dir)
    elif Path(args.source).suffix.lower() in VID_EXTS:
        records = infer_video(model, Path(args.source), out_dir)
    else:
        records = infer_images(model, [Path(args.source)], out_dir)

    dt         = time.time() - t0
    total_inst = sum(r.get("n_instances", 0) for r in records)
    all_confs  = [c for r in records for c in r.get("confidences", [])]
    summary = {
        "model":      f"{WORKSPACE}/{PROJECT}/v{MODEL_VERSION}",
        "source":     args.source,
        "n_frames":   len(records),
        "total_instances": total_inst,
        "avg_instances_per_frame": round(total_inst / max(len(records), 1), 2),
        "mean_conf":  round(float(np.mean(all_confs)) if all_confs else 0, 4),
        "runtime_s":  round(dt, 2),
        "output_dir": str(out_dir),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Done — {len(records)} frames in {dt:.1f}s")
    print(f"   Avg instances/frame : {summary['avg_instances_per_frame']}")
    print(f"   Mean confidence     : {summary['mean_conf']}")
    print(f"   Output              : {out_dir}")

if __name__ == "__main__":
    main()