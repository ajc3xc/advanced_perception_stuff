#!/usr/bin/env python3
"""
1_extract_frames.py — Extract raw frames from real videos aligned to saved SAM3 masks.
Skip this when using synthetic data (script 0 writes raw_frames/ directly).
All paths auto-derived from BASE.
"""

import cv2
from pathlib import Path

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path(r"C:/Users/13144/Documents/PhD/Robotics_ABE6399/advanced_perception_stuff/yolo_local/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

SAM3_ROOT  = BASE / "sam3_outputs"
VIDEO_ROOT = BASE / "videos"          # put source videos here
YOLO_ROOT  = BASE / "yolo_dataset"
JPEG_QUALITY = 95
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract(video_path: Path, mask_dir: Path, out_dir: Path) -> int:
    indices = sorted(int(p.stem) for p in mask_dir.glob("*.npy") if p.stem.isdigit())
    if not indices:
        print(f"  [skip] no .npy files in {mask_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] cannot open {video_path}")
        return 0
    written = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [warn] frame {idx} unreadable")
            continue
        cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        written += 1
    cap.release()
    return written


def main():
    total = 0
    for stem_dir in sorted(SAM3_ROOT.iterdir()):
        if not stem_dir.is_dir():
            continue
        mask_dir = stem_dir / "masks"
        if not mask_dir.exists():
            continue
        stem = stem_dir.name
        video_path = next(
            (VIDEO_ROOT / f"{stem}{e}" for e in VID_EXTS
             if (VIDEO_ROOT / f"{stem}{e}").exists()), None
        )
        if video_path is None:
            print(f"[skip] no video for '{stem}' in {VIDEO_ROOT}")
            continue
        out_dir = YOLO_ROOT / "raw_frames" / stem
        print(f"[{stem}] extracting ...")
        n = extract(video_path, mask_dir, out_dir)
        print(f"  {n} frames -> {out_dir}")
        total += n
    print(f"\nDone -- {total} frames extracted")
    print("Next: python 2_convert_masks.py")


if __name__ == "__main__":
    main()
