#!/usr/bin/env python3
"""
Batch convert ZED .svo/.svo2 recordings to TWO MP4s each:
  1) RGB (LEFT+RIGHT AVI -> MP4)          mode 0
  2) RGB+DEPTH_VIEW (LEFT+DEPTH_VIEW AVI -> MP4)  mode 1

- Recurses INPUT_ROOT.
- Mirrors folder structure under OUTPUT_ROOT.
- Parallelizes up to 4 workers.
- Streams *live* stdout/stderr from ZED_SVO_Export and ffmpeg so you can see “CLI-like” progress.
- Uses GPU encoding via NVENC (h264_nvenc). Falls back to libx264 if NVENC not available.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple, List

# --------- YOUR PATHS ----------
INPUT_ROOT = Path(r"C:\Users\13144\Downloads\OneDrive_2026-03-04\Bamboo Data")
OUTPUT_ROOT = Path(r"C:\Users\13144\Documents\PhD\Robotics_ABE6399")
ZED_EXPORT = Path(r"C:\Program Files (x86)\ZED SDK\samples\bin\ZED_SVO_Export.exe")

# --------- SETTINGS ----------
MAX_WORKERS = 4
DELETE_AVI_AFTER_MP4 = True

# ZED modes:
# 0 = LEFT+RIGHT AVI
# 1 = LEFT+DEPTH_VIEW AVI
ZED_MODE_RGB = "0"
ZED_MODE_DEPTHVIEW = "1"

# NVENC settings (quality/speed tradeoff)
NVENC_PRESET = "p4"   # p1 fastest ... p7 best quality
NVENC_CQ = "19"       # lower = higher quality (18–23 typical)

# If ffmpeg missing or NVENC not available, we fall back to libx264
X264_CRF = "18"
X264_PRESET = "fast"


# ----------------- plumbing for live output -----------------
_print_lock = threading.Lock()

def _safe_print(line: str) -> None:
    with _print_lock:
        print(line, flush=True)

def stream_process(cmd: List[str], prefix: str) -> int:
    """
    Run a subprocess and stream its combined stdout/stderr live, prefixing each line.
    Returns returncode.
    """
    # Use text mode, line buffered
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        _safe_print(f"{prefix}{line}")
    return proc.wait()

def require_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")

def is_nontrivial_file(p: Path, min_bytes: int = 1024) -> bool:
    try:
        return p.exists() and p.stat().st_size >= min_bytes
    except OSError:
        return False


# ----------------- conversion logic -----------------
@dataclass(frozen=True)
class Job:
    svo: Path
    out_dir: Path
    avi_rgb: Path
    avi_depth: Path
    mp4_rgb: Path
    mp4_depth: Path

def build_jobs() -> List[Job]:
    svo_files = list(INPUT_ROOT.rglob("*.svo")) + list(INPUT_ROOT.rglob("*.svo2"))
    jobs: List[Job] = []
    for svo in svo_files:
        rel = svo.relative_to(INPUT_ROOT)
        out_dir = OUTPUT_ROOT / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = svo.stem
        avi_rgb = out_dir / f"{stem}_rgb_lr.avi"
        avi_depth = out_dir / f"{stem}_rgb_depthview.avi"
        mp4_rgb = out_dir / f"{stem}_rgb_lr.mp4"
        mp4_depth = out_dir / f"{stem}_rgb_depthview.mp4"

        jobs.append(Job(svo=svo, out_dir=out_dir, avi_rgb=avi_rgb, avi_depth=avi_depth, mp4_rgb=mp4_rgb, mp4_depth=mp4_depth))
    return jobs

def detect_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")

def ffmpeg_has_encoder(ffmpeg: str, encoder_name: str) -> bool:
    try:
        p = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return encoder_name in out
    except Exception:
        return False

def zed_export_to_avi(svo: Path, avi: Path, mode: str, prefix: str) -> None:
    avi.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ZED_EXPORT), str(svo), str(avi), mode]
    _safe_print(f"{prefix}>>> ZED export (mode {mode})\n{prefix}{' '.join(cmd)}")
    rc = stream_process(cmd, prefix)
    if rc != 0:
        raise RuntimeError(f"ZED_SVO_Export failed (mode {mode}) with code {rc}")
    if not is_nontrivial_file(avi):
        raise RuntimeError(f"ZED export produced no/too-small AVI: {avi}")

def encode_mp4(ffmpeg: str, inp: Path, out: Path, use_nvenc: bool, prefix: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if use_nvenc:
        cmd = [
            ffmpeg, "-y",
            "-i", str(inp),
            "-c:v", "h264_nvenc",
            "-preset", NVENC_PRESET,
            "-rc", "vbr",
            "-cq", NVENC_CQ,
            "-b:v", "0",
            str(out),
        ]
        _safe_print(f"{prefix}>>> ffmpeg NVENC encode\n{prefix}{' '.join(cmd)}")
    else:
        cmd = [
            ffmpeg, "-y",
            "-i", str(inp),
            "-c:v", "libx264",
            "-crf", X264_CRF,
            "-preset", X264_PRESET,
            str(out),
        ]
        _safe_print(f"{prefix}>>> ffmpeg x264 encode (fallback)\n{prefix}{' '.join(cmd)}")

    rc = stream_process(cmd, prefix)
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed with code {rc}")
    if not is_nontrivial_file(out):
        raise RuntimeError(f"ffmpeg produced no/too-small MP4: {out}")

def process_job(job: Job, job_idx: int, total: int, ffmpeg: str, use_nvenc: bool) -> Tuple[str, str]:
    """
    Returns (status, message).
    Streams all logs live.
    """
    prefix = f"[{job_idx+1:04d}/{total:04d}] "

    try:
        _safe_print(f"{prefix}=== START {job.svo}")

        # If both MP4s already exist, skip
        if is_nontrivial_file(job.mp4_rgb) and is_nontrivial_file(job.mp4_depth):
            _safe_print(f"{prefix}=== SKIP (MP4s already exist)\n{prefix}    {job.mp4_rgb}\n{prefix}    {job.mp4_depth}")
            return ("SKIP", str(job.svo))

        # 1) Export RGB AVI (L+R)
        if not is_nontrivial_file(job.avi_rgb):
            zed_export_to_avi(job.svo, job.avi_rgb, ZED_MODE_RGB, prefix)
        else:
            _safe_print(f"{prefix}>>> reuse existing AVI: {job.avi_rgb}")

        # 2) Export Depth-view AVI (L+DEPTH_VIEW)
        if not is_nontrivial_file(job.avi_depth):
            zed_export_to_avi(job.svo, job.avi_depth, ZED_MODE_DEPTHVIEW, prefix)
        else:
            _safe_print(f"{prefix}>>> reuse existing AVI: {job.avi_depth}")

        # 3) Encode MP4s
        if not is_nontrivial_file(job.mp4_rgb):
            encode_mp4(ffmpeg, job.avi_rgb, job.mp4_rgb, use_nvenc, prefix)
        else:
            _safe_print(f"{prefix}>>> MP4 exists: {job.mp4_rgb}")

        if not is_nontrivial_file(job.mp4_depth):
            encode_mp4(ffmpeg, job.avi_depth, job.mp4_depth, use_nvenc, prefix)
        else:
            _safe_print(f"{prefix}>>> MP4 exists: {job.mp4_depth}")

        # 4) Cleanup AVIs (optional)
        if DELETE_AVI_AFTER_MP4:
            for avi in (job.avi_rgb, job.avi_depth):
                try:
                    if avi.exists():
                        avi.unlink()
                        _safe_print(f"{prefix}>>> deleted AVI: {avi.name}")
                except OSError:
                    pass

        _safe_print(f"{prefix}=== DONE\n{prefix}    {job.mp4_rgb}\n{prefix}    {job.mp4_depth}")
        return ("OK", str(job.svo))

    except Exception as e:
        _safe_print(f"{prefix}=== FAIL {job.svo}\n{prefix}{e}")
        return ("FAIL", str(job.svo))


def main() -> int:
    # Basic checks
    try:
        require_exists(ZED_EXPORT, "ZED_SVO_Export.exe")
    except Exception as e:
        print(e, file=sys.stderr)
        return 2

    ffmpeg = detect_ffmpeg()
    if not ffmpeg:
        print("ERROR: ffmpeg not found on PATH. Install ffmpeg or add it to PATH, then retry.", file=sys.stderr)
        return 2

    use_nvenc = ffmpeg_has_encoder(ffmpeg, "h264_nvenc")
    if use_nvenc:
        _safe_print("Using GPU encoding: h264_nvenc (NVENC)")
    else:
        _safe_print("NVENC not available in your ffmpeg build; falling back to libx264 (CPU).")

    jobs = build_jobs()
    total = len(jobs)
    _safe_print(f"INPUT_ROOT : {INPUT_ROOT}")
    _safe_print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    _safe_print(f"Found {total} .svo/.svo2 files")
    _safe_print(f"Workers: {min(MAX_WORKERS, max(1,total))}\n")

    ok = skip = fail = 0

    # ThreadPool is ideal here: we are just running subprocesses and streaming I/O.
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, total))) as ex:
        futures = [ex.submit(process_job, job, i, total, ffmpeg, use_nvenc) for i, job in enumerate(jobs)]
        for fut in as_completed(futures):
            status, _ = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1

    _safe_print(f"\nSummary: OK={ok} SKIP={skip} FAIL={fail}")
    if fail:
        _safe_print("Some jobs failed. Scroll up to the [####/####] FAIL blocks for the exact command output.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())