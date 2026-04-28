#!/usr/bin/env python3
"""
5_check.py — Validate dataset and inspect inference results.

  python _5_check.py dataset   — validate YOLO labels
  python _5_check.py results   — summarise latest inference run
  python _5_check.py           — run both
"""

import sys
import json
import random
import numpy as np
from pathlib import Path

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path("synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

YOLO_ROOT  = BASE / "yolo_dataset"
INFER_ROOT = BASE / "infer_results"
SAMPLE_N   = 20


def check_dataset():
    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    report_path = YOLO_ROOT / "conversion_report.json"
    if report_path.exists():
        r = json.loads(report_path.read_text())
        print("  Conversion report:")
        for k, v in r.items():
            print(f"    {k:30s} {v}")
    else:
        print("  [warn] no conversion_report.json — run _2_convert_masks.py first")
        return

    issues = []
    for split in ["train", "val"]:
        img_dir = YOLO_ROOT / "images" / split
        lbl_dir = YOLO_ROOT / "labels" / split
        imgs    = list(img_dir.glob("*.jpg"))
        labels  = list(lbl_dir.glob("*.txt"))
        print(f"\n  {split}: {len(imgs)} images, {len(labels)} labels")

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in labels}
        if img_stems - lbl_stems:
            issues.append(f"{split}: {len(img_stems - lbl_stems)} images missing labels")
        if lbl_stems - img_stems:
            issues.append(f"{split}: {len(lbl_stems - img_stems)} labels missing images")

        sample = random.sample(labels, min(SAMPLE_N, len(labels)))
        bad, n_lines = 0, 0
        for lbl_path in sample:
            lines = lbl_path.read_text().strip().splitlines()
            if not lines:
                bad += 1; continue
            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    bad += 1; break
                try:
                    coords = list(map(float, parts[1:]))
                    if any(v < 0 or v > 1 for v in coords):
                        issues.append(f"out-of-range coords: {lbl_path.name}")
                except ValueError:
                    bad += 1; break
                n_lines += len(lines)

        avg = n_lines / max(len(sample), 1)
        print(f"    avg instances/frame (sample): {avg:.1f}")
        print(f"    {'✅ all sampled labels valid' if not bad else f'⚠️  {bad} bad label files'}")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for iss in issues[:20]:
            print(f"    • {iss}")
    else:
        print(f"\n  ✅ No issues found")


def check_results():
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS CHECK")
    print("=" * 60)

    if not INFER_ROOT.exists():
        print("  No inference results yet — run _4_infer_roboflow.py first.")
        return

    runs = sorted(INFER_ROOT.iterdir())
    if not runs:
        print("  No runs found.")
        return

    rpath = runs[-1] / "results.json"
    if not rpath.exists():
        print(f"  No results.json in {runs[-1]}")
        return

    r = json.loads(rpath.read_text())
    print(f"\n  Run         : {runs[-1].name}")
    print(f"  Model       : {r.get('model', 'n/a')}")
    print(f"  Frames      : {r.get('n_frames', 0)}")
    print(f"  Total inst  : {r.get('total_instances', 0)}")
    print(f"  Avg/frame   : {r.get('avg_instances_per_frame', 0)}")
    print(f"  Mean conf   : {r.get('mean_conf', 0):.4f}")
    print(f"  Runtime     : {r.get('runtime_s', 0)}s")

    mean = r.get("mean_conf", 0)
    if mean > 0.7:   note = "✅ model looks confident"
    elif mean > 0.5: note = "⚠️  moderate confidence — more data may help"
    else:            note = "❌ low confidence — check training converged"
    print(f"\n  {note}")


def main():
    args = sys.argv[1:] or ["dataset", "results"]
    if "dataset" in args: check_dataset()
    if "results" in args: check_results()

if __name__ == "__main__":
    main()