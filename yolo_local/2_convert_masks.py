#!/usr/bin/env python3
"""
2_convert_masks.py — Convert SAM3 NPY label maps to YOLO segmentation .txt files.
All paths auto-derived from BASE.
"""

import cv2
import json
import shutil
import random
import numpy as np
from pathlib import Path

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path(r"C:/Users/13144/Documents/PhD/Robotics_ABE6399/advanced_perception_stuff/yolo_local/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

SAM3_ROOT         = BASE / "sam3_outputs"
YOLO_ROOT         = BASE / "yolo_dataset"
VAL_SPLIT         = 0.15
CLASS_ID          = 0
MIN_MASK_PX       = 150
POLY_EPSILON_FRAC = 0.002
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)


def load_instances(npy_path):
    arr = np.load(str(npy_path))
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.max() > 1:
        return [(arr == iid).astype(np.uint8) for iid in np.unique(arr) if iid > 0]
    binary = (arr > 0).astype(np.uint8)
    return [binary] if binary.sum() >= MIN_MASK_PX else []


def mask_to_poly(mask):
    if mask.sum() < MIN_MASK_PX:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_MASK_PX:
        return None
    eps = POLY_EPSILON_FRAC * cv2.arcLength(c, True)
    c   = cv2.approxPolyDP(c, eps, True)
    if len(c) < 3:
        return None
    h, w = mask.shape
    pts = c.reshape(-1, 2).astype(float)
    pts[:, 0] = (pts[:, 0] / w).clip(0, 1)
    pts[:, 1] = (pts[:, 1] / h).clip(0, 1)
    return pts.flatten().tolist()


def npy_to_lines(npy_path):
    try:
        instances = load_instances(npy_path)
    except Exception as e:
        print(f"  [warn] {npy_path.name}: {e}")
        return []
    lines = []
    for mask in instances:
        poly = mask_to_poly(mask)
        if poly:
            lines.append(f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in poly))
    return lines


def collect_pairs():
    pairs = []
    for stem_dir in sorted(SAM3_ROOT.iterdir()):
        if not stem_dir.is_dir():
            continue
        mask_dir  = stem_dir / "masks"
        frame_dir = YOLO_ROOT / "raw_frames" / stem_dir.name
        if not mask_dir.exists():
            continue
        if not frame_dir.exists():
            print(f"[skip] no raw_frames for '{stem_dir.name}' -- run script 0 or 1 first")
            continue
        for npy in sorted(mask_dir.glob("*.npy")):
            jpg = frame_dir / f"{npy.stem}.jpg"
            if jpg.exists():
                pairs.append((jpg, npy))
    return pairs


def main():
    pairs = collect_pairs()
    print(f"Found {len(pairs)} (image, mask) pairs")
    if not pairs:
        print("Nothing to convert.")
        return

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_SPLIT))

    for split in ["train", "val"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    counts  = {"train": 0, "val": 0}

    for i, (img_path, npy_path) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        lines = npy_to_lines(npy_path)
        if not lines:
            skipped += 1
            continue
        stem = f"{npy_path.parent.parent.name}_{npy_path.stem}"
        shutil.copy2(img_path, YOLO_ROOT / "images" / split / f"{stem}.jpg")
        (YOLO_ROOT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
        counts[split] += 1
        if written % 100 == 0:
            print(f"  {written}/{len(pairs)} ...", flush=True)

    # ── write YAML with forward slashes — YOLO breaks on Windows backslashes ──
    yolo_root_str = YOLO_ROOT.as_posix()
    yaml = (
        f"path: {yolo_root_str}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: bamboo\n"
    )
    yaml_path = YOLO_ROOT / "bamboo.yaml"
    yaml_path.write_text(yaml)

    report = {"total": len(pairs), "written": written, "skipped": skipped,
              "train": counts["train"], "val": counts["val"]}
    (YOLO_ROOT / "conversion_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone -- written={written}  skipped={skipped}")
    print(f"  train={counts['train']}  val={counts['val']}")
    print(f"  YAML -> {yaml_path}")
    print("Next: python 3_train.py")


if __name__ == "__main__":
    main()