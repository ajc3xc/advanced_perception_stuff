#!/usr/bin/env python3
"""
0_gen_data.py — Generate synthetic SAM3-style test data.
All paths auto-derived from BASE.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path("/blue/cli2/a.camerer/ABE6399_Robotics/advanced_robotic_systems/yolo/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

SAM3_ROOT = BASE / "sam3_outputs"
YOLO_ROOT = BASE / "yolo_dataset"

N_VIDEOS     = 4
N_FRAMES     = 60
IMG_W, IMG_H = 1280, 720
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

STALK_COLORS = [
    (30, 140, 30), (40, 160, 20), (20, 120, 40),
    (50, 150, 10), (25, 130, 35), (45, 155, 25),
]

def make_frame(w, h):
    n_stalks = random.randint(4, 12)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        bg[y] = [int(80*(1-t)+20*t), int(120*(1-t)+60*t), int(60*(1-t)+15*t)]
    noise = np.random.randint(0, 25, (h, w, 3), dtype=np.uint8)
    frame = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    label_map = np.zeros((h, w), dtype=np.uint8)

    for i in range(n_stalks):
        cx    = int(w * (i + 0.5 + random.uniform(-0.25, 0.25)) / n_stalks)
        cy    = h // 2 + random.randint(-40, 40)
        sw    = random.randint(14, 38)
        sh    = random.randint(int(h * 0.55), int(h * 0.92))
        angle = random.uniform(-7, 7)
        iid   = min(i + 1, 255)

        cv2.ellipse(label_map, (cx, cy), (sw//2, sh//2), angle, 0, 360, iid, -1)

        col = random.choice(STALK_COLORS)
        col_var = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in col)
        layer = np.zeros_like(frame)
        cv2.ellipse(layer, (cx, cy), (sw//2, sh//2), angle, 0, 360, col_var[::-1], -1)
        px = label_map == iid
        frame[px] = cv2.cvtColor(layer, cv2.COLOR_BGR2RGB)[px]

        for j in range(random.randint(3, 8)):
            ny = cy - sh//2 + int(sh * j / 7) + random.randint(-4, 4)
            ny = max(0, min(h-1, ny))
            dark = tuple(max(0, c-40) for c in col_var)
            cv2.line(frame, (cx-sw//2, ny), (cx+sw//2, ny),
                     dark[::-1], thickness=random.randint(1, 3))

    return frame, label_map


def colorize_overlay(frame_rgb, label_map):
    ov = frame_rgb.copy().astype(np.float32)
    palette = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255)]
    for iid in [v for v in np.unique(label_map) if v > 0]:
        c  = np.array(palette[(iid-1) % len(palette)], dtype=np.float32)
        px = label_map == iid
        ov[px] = ov[px] * 0.45 + c * 0.55
    return ov.clip(0, 255).astype(np.uint8)


def main():
    total = 0
    for vid_idx in range(N_VIDEOS):
        stem      = f"synthetic_video_{vid_idx:02d}"
        mask_dir  = SAM3_ROOT / stem / "masks"
        ovl_dir   = SAM3_ROOT / stem / "overlays"
        frame_dir = YOLO_ROOT / "raw_frames" / stem

        for d in [mask_dir, ovl_dir, frame_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"[{stem}] generating {N_FRAMES} frames ...", flush=True)

        for f in range(N_FRAMES):
            frame_rgb, lmap = make_frame(IMG_W, IMG_H)
            fn = f"{f:06d}"
            cv2.imwrite(str(frame_dir / f"{fn}.jpg"),
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.save(str(mask_dir / f"{fn}.npy"), lmap)
            Image.fromarray(lmap).save(mask_dir / f"{fn}.png")
            ov = colorize_overlay(frame_rgb, lmap)
            cv2.imwrite(str(ovl_dir / f"{fn}.jpg"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            total += 1

        print(f"  done -> {frame_dir}")

    print(f"\nDone -- {total} frames generated under {BASE}")
    print("Next: python 2_convert_masks.py")


if __name__ == "__main__":
    main()
