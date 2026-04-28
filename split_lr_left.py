#!/usr/bin/env python3
"""
split_lr_left.py — Extract left half of all *_rgb_lr*.mp4 files recursively.
                   Uses ffmpeg with NVENC (GPU encode) + multiprocessing across files.

Input  : /blue/cli2/a.camerer/ABE6399_Robotics/inputs
Output : /blue/cli2/a.camerer/ABE6399_Robotics/inputs/lr_videos_left/
"""

import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

IN_DIR  = Path(r"C:\Users\13144\Documents\PhD\Robotics_ABE6399\inputs\videos")
OUT_DIR = Path(r"C:\Users\13144\Documents\PhD\Robotics_ABE6399\inputs\videos_left_preprocess")

# Number of files to process in parallel — keep <= number of GPUs to avoid
# NVENC session limits (NVIDIA caps at 8 concurrent NVENC sessions per GPU)
N_WORKERS = 4

# Global set by main() before pool is spawned
USE_NVENC = False


def has_nvenc() -> bool:
    """Check whether ffmpeg on this machine has NVENC support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True
        )
        return "h264_nvenc" in result.stdout
    except FileNotFoundError:
        return False


def split_left(src: Path) -> tuple[Path, bool, str]:
    """
    Crop left half with ffmpeg.
    GPU path : CUDA decode + NVENC encode  (fast)
    CPU path : software decode + libx264   (fallback)
    Returns (src, success, message).
    """
    out_path = OUT_DIR / f"{src.stem}_left.mp4"

    if out_path.exists():
        return src, True, f"⏭  Skipped (already exists): {out_path.name}"

    # crop=w:h:x:y  ->  left half = iw/2 : ih : 0 : 0
    crop_filter = "crop=iw/2:ih:0:0"

    if USE_NVENC:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",               # GPU decode
            "-hwaccel_output_format", "cuda",
            "-i", str(src),
            "-vf", f"hwdownload,format=nv12,{crop_filter}",
            "-c:v", "h264_nvenc",             # GPU encode
            "-preset", "p4",                  # p1=fastest ... p7=best quality
            "-cq", "23",                      # constant quality
            "-an",                            # no audio needed for SAM3
            "-y", str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-an",
            "-y", str(out_path),
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = out_path.stat().st_size / 1e6
        return src, True, f"✅  {src.name}  ->  {out_path.name}  ({size_mb:.1f} MB)"
    except subprocess.CalledProcessError as e:
        # NVENC session limit hit — fall back to CPU for this file
        if USE_NVENC:
            fallback_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vf", crop_filter,
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-an", "-y", str(out_path),
            ]
            try:
                subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
                size_mb = out_path.stat().st_size / 1e6
                return src, True, f"✅  {src.name}  ->  {out_path.name}  ({size_mb:.1f} MB)  [CPU fallback]"
            except subprocess.CalledProcessError as e2:
                return src, False, f"❌  {src.name}: {e2.stderr.strip()}"
        return src, False, f"❌  {src.name}: {e.stderr.strip()}"


def main():
    global USE_NVENC

    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "❌  ffmpeg not found — load the module first:\n"
            "    module load ffmpeg"
        )

    USE_NVENC = has_nvenc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in IN_DIR.rglob("*_rgb_lr*.mp4")
        if not p.stem.endswith("_left")   # skip already-split files
        and OUT_DIR not in p.parents      # skip anything inside the output folder
    )
    if not videos:
        print(f"⚠️  No *_rgb_lr*.mp4 files found under {IN_DIR}")
        return

    encoder_tag = "NVENC (GPU)" if USE_NVENC else "libx264 (CPU)"
    print(f"🔍  Found    : {len(videos)} file(s)")
    print(f"⚙️   Encoder  : {encoder_tag}")
    print(f"⚙️   Workers  : {N_WORKERS}  (of {cpu_count()} available CPUs)")
    print(f"📂  Output   : {OUT_DIR}\n")

    with Pool(processes=N_WORKERS) as pool:
        for src, ok, msg in pool.imap_unordered(split_left, videos):
            print(msg)

    print("\n🎉  All done.")


if __name__ == "__main__":
    main()
