#!/usr/bin/env python3
"""
3_upload_to_roboflow.py — Upload converted YOLO dataset to Roboflow.

Before running:
  1. pip install roboflow  (already in pixi.toml)
  2. Go to app.roboflow.com -> Settings -> Roboflow API -> copy your API key
  3. Create a new project on Roboflow (type: Instance Segmentation)
  4. Fill in the three config values below
"""

from pathlib import Path
from roboflow import Roboflow

# ── EDIT THESE ────────────────────────────────────────────────────────────────
    # project you created on Roboflow
ROBOFLOW_API_KEY = "e20yDTNrBI9q2Sst9wA3"
WORKSPACE        = "xiaoyus-workspace-wtaot"
PROJECT          = "bamboo_synthetic"
BASE             = Path("synthetic_data")
# ─────────────────────────────────────────────────────────────────────────────

YOLO_ROOT    = BASE / "yolo_dataset"
IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
IMAGES_VAL   = YOLO_ROOT / "images" / "val"
LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
LABELS_VAL   = YOLO_ROOT / "labels" / "val"

for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    if not d.exists():
        raise FileNotFoundError(f"Missing: {d}\nRun _2_convert_masks.py first.")

rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)


def upload_split(images_dir: Path, labels_dir: Path, split: str):
    images = sorted(images_dir.glob("*.jpg"))
    print(f"\n[{split}] uploading {len(images)} images …")
    for img_path in images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"  [skip] no label for {img_path.name}")
            continue
        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(lbl_path),
                split=split,
                num_retry_uploads=3,
                batch_name=f"synthetic-{split}",
            )
        except Exception as e:
            print(f"  [error] {img_path.name}: {e}")
    print(f"  [{split}] done ✓")


upload_split(IMAGES_TRAIN, LABELS_TRAIN, "train")
upload_split(IMAGES_VAL,   LABELS_VAL,   "valid")

print(f"\n✅ Upload complete!")
print(f"   Go to app.roboflow.com/{WORKSPACE}/{PROJECT}")
print(f"   Click Train → wait ~45 min → done")