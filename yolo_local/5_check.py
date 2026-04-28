#!/usr/bin/env python3
"""
5_check.py — Validate dataset and inspect inference results.
All paths auto-derived from BASE.

Usage:
    python 5_check.py dataset          # validate converted YOLO labels
    python 5_check.py results          # summarise latest inference run
    python 5_check.py dataset results  # both (default)
"""

import sys
import json
import random
import numpy as np
from pathlib import Path

# ── ONLY EDIT THIS ────────────────────────────────────────────────────────────
BASE = Path("/blue/cli2/a.camerer/ABE6399_Robotics/advanced_robotic_systems/yolo/synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

YOLO_ROOT  = BASE / "yolo_dataset"
INFER_ROOT = BASE / "infer_results"
SAMPLE_N   = 20


def check_dataset():
    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)
    print(f"  Root: {YOLO_ROOT}")

    report_path = YOLO_ROOT / "conversion_report.json"
    if report_path.exists():
        r = json.loads(report_path.read_text())
        print("\n  Conversion report:")
        for k, v in r.items():
            print(f"    {k:35s} {v}")
    else:
        print("  [warn] no conversion_report.json")

    issues = []
    for split in ["train", "val"]:
        img_dir = YOLO_ROOT / "images" / split
        lbl_dir = YOLO_ROOT / "labels" / split

        imgs   = list(img_dir.glob("*.jpg"))
        labels = list(lbl_dir.glob("*.txt"))
        print(f"\n  {split}: {len(imgs)} images,  {len(labels)} labels")

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in labels}
        unpaired_imgs = img_stems - lbl_stems
        unpaired_lbls = lbl_stems - img_stems
        if unpaired_imgs:
            print(f"    WARNING: {len(unpaired_imgs)} images with no label")
            issues.append(f"{split}: {len(unpaired_imgs)} unpaired images")
        if unpaired_lbls:
            print(f"    WARNING: {len(unpaired_lbls)} labels with no image")
            issues.append(f"{split}: {len(unpaired_lbls)} unpaired labels")

        sample    = random.sample(labels, min(SAMPLE_N, len(labels)))
        n_lines   = 0
        bad_files = 0
        for lbl in sample:
            lines = lbl.read_text().strip().splitlines()
            if not lines:
                bad_files += 1
                issues.append(f"empty: {lbl.name}")
                continue
            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    bad_files += 1
                    issues.append(f"short polygon: {lbl.name}")
                    break
                try:
                    coords = list(map(float, parts[1:]))
                    oob = [v for v in coords if v < 0 or v > 1]
                    if oob:
                        issues.append(f"out-of-range coords: {lbl.name}")
                except ValueError as e:
                    bad_files += 1
                    issues.append(f"parse error: {lbl.name}: {e}")
                n_lines += len(lines)

        print(f"    avg instances/frame (sample): {n_lines / max(len(sample),1):.1f}")
        if bad_files:
            print(f"    WARNING: {bad_files} bad label files in sample")
        else:
            print(f"    OK -- all sampled label files valid")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for iss in issues[:20]:
            print(f"    * {iss}")
    else:
        print(f"\n  No issues found")


def check_results():
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS CHECK")
    print("=" * 60)
    print(f"  Root: {INFER_ROOT}")

    runs = sorted(INFER_ROOT.iterdir()) if INFER_ROOT.exists() else []
    if not runs:
        print("  No inference results -- run script 4 first.")
        return

    latest = runs[-1]
    rpath  = latest / "results.json"
    if not rpath.exists():
        print(f"  No results.json in {latest}")
        return

    r = json.loads(rpath.read_text())
    print(f"\n  Run                 : {latest.name}")
    print(f"  Weights             : {r.get('weights','n/a')}")
    print(f"  Frames              : {r.get('n_frames',0)}")
    print(f"  Total instances     : {r.get('total_instances',0)}")
    print(f"  Avg inst/frame      : {r.get('avg_instances_per_frame',0)}")
    print(f"  Mean confidence     : {r.get('mean_conf',0):.4f}")
    print(f"  Min confidence      : {r.get('min_conf',0):.4f}")
    print(f"  Throughput          : {r.get('fps',0)} fps")
    print(f"  Output dir          : {r.get('output_dir','n/a')}")

    mean = r.get("mean_conf", 0)
    if   mean > 0.70: note = "model looks confident"
    elif mean > 0.50: note = "moderate confidence -- more training data may help"
    else:             note = "low confidence -- check training converged"
    print(f"\n  Assessment: {note}")


def main():
    args = sys.argv[1:] or ["dataset", "results"]
    if "dataset" in args:
        check_dataset()
    if "results" in args:
        check_results()


if __name__ == "__main__":
    main()
