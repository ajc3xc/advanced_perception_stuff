#!/usr/bin/env python3
"""Batch bamboo-stalk instance labeling with SAM2 (Hugging Face).

Inputs default to:
  /blue/cli2/a.camerer/ABE6399_Robotics/inputs
Outputs default to:
  /blue/cli2/a.camerer/ABE6399_Robotics/outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"
DEFAULT_INPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/inputs")
DEFAULT_OUTPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/outputs")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Candidate:
    mask: np.ndarray
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    height: int
    width: int
    aspect_hw: float


def normalize_mask(mask_obj: object, shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(mask_obj)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.shape != shape:
        arr = np.array(Image.fromarray((arr > 0).astype(np.uint8) * 255).resize(shape[::-1], Image.NEAREST))
    return (arr > 0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def is_stalk_candidate(
    mask: np.ndarray,
    min_area: int,
    min_height: int,
    min_aspect: float,
    max_width: int,
) -> Candidate | None:
    area = int(mask.sum())
    if area < min_area:
        return None

    x0, y0, x1, y1 = bbox_from_mask(mask)
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    aspect_hw = height / float(width)

    if height < min_height:
        return None
    if width > max_width:
        return None
    if aspect_hw < min_aspect:
        return None

    # Fill ratio keeps thin vertical structure, removes very sparse noise.
    fill_ratio = area / float(width * height)
    if fill_ratio < 0.18:
        return None

    return Candidate(mask=mask, area=area, x0=x0, y0=y0, x1=x1, y1=y1, height=height, width=width, aspect_hw=aspect_hw)


def label_to_color_image_uint16(labels: np.ndarray) -> np.ndarray:
    """Map uint16 ids to RGB colors deterministically."""
    ids = labels.astype(np.uint32)
    r = (53 * ids + 29) % 256
    g = (97 * ids + 71) % 256
    b = (193 * ids + 11) % 256
    colors = np.stack([r, g, b], axis=-1).astype(np.uint8)
    colors[labels == 0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    b = np.zeros_like(labels, dtype=bool)
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return b


def process_image(
    img_path: Path,
    out_dir: Path,
    generator,
    alpha: float,
    min_area: int,
    min_height: int,
    min_aspect: float,
    max_width: int,
    max_iou: float,
) -> dict:
    image = Image.open(img_path).convert("RGB")
    h, w = image.height, image.width
    shape = (h, w)
    image_np = np.array(image, dtype=np.float32)

    print(f"[info] segmenting {img_path.name}", flush=True)
    outputs = generator(image)
    if isinstance(outputs, list):
        masks = outputs
    elif isinstance(outputs, dict):
        masks = outputs.get("masks", [])
    else:
        masks = []

    candidates: list[Candidate] = []
    for m in masks:
        mm = normalize_mask(m, shape)
        cand = is_stalk_candidate(mm, min_area=min_area, min_height=min_height, min_aspect=min_aspect, max_width=max_width)
        if cand is not None:
            candidates.append(cand)

    # Keep larger candidates first, suppress near-duplicates.
    candidates.sort(key=lambda c: c.area, reverse=True)
    kept: list[Candidate] = []
    for cand in candidates:
        if all(iou(cand.mask, k.mask) < max_iou for k in kept):
            kept.append(cand)

    labels = np.zeros(shape, dtype=np.uint16)
    for idx, cand in enumerate(kept, start=1):
        labels[cand.mask] = idx

    colors = label_to_color_image_uint16(labels)
    fg = labels > 0
    overlay = image_np.copy()
    overlay[fg] = (1.0 - alpha) * image_np[fg] + alpha * colors[fg].astype(np.float32)
    edges = label_boundaries(labels)
    overlay[edges] = np.array([45, 110, 255], dtype=np.float32)
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    label_png = out_dir / f"{stem}_stalk_labels_u16.png"
    label_npy = out_dir / f"{stem}_stalk_labels_u16.npy"
    color_png = out_dir / f"{stem}_stalk_colors.png"
    overlay_png = out_dir / f"{stem}_stalk_overlay.png"
    meta_json = out_dir / f"{stem}_stalk_meta.json"

    Image.fromarray(labels, mode="I;16").save(label_png)
    np.save(label_npy, labels)
    Image.fromarray(colors, mode="RGB").save(color_png)
    Image.fromarray(overlay_u8, mode="RGB").save(overlay_png)

    meta = {
        "image": str(img_path),
        "width": w,
        "height": h,
        "sam_masks_total": len(masks),
        "stalk_candidates": len(candidates),
        "stalk_instances": int(labels.max()),
        "outputs": {
            "label_png_u16": str(label_png),
            "label_npy_u16": str(label_npy),
            "color_png": str(color_png),
            "overlay_png": str(overlay_png),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    print(f"[ok] {img_path.name}: stalk_instances={labels.max()} -> {overlay_png.name}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch bamboo stalk labeling with SAM2.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--min-height", type=int, default=140)
    parser.add_argument("--min-aspect", type=float, default=2.2)
    parser.add_argument("--max-width", type=int, default=220)
    parser.add_argument("--max-iou", type=float, default=0.70)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    device = 0 if torch.cuda.is_available() else -1
    print(
        f"[info] torch={torch.__version__}, cuda={torch.version.cuda}, gpu_available={torch.cuda.is_available()}",
        flush=True,
    )
    print(f"[info] loading model: {args.model}", flush=True)
    generator = pipeline("mask-generation", model=args.model, device=device)

    images = sorted([p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")
    print(f"[info] found {len(images)} image(s)", flush=True)

    all_meta: list[dict] = []
    for img in images:
        all_meta.append(
            process_image(
                img_path=img,
                out_dir=args.output_dir,
                generator=generator,
                alpha=alpha,
                min_area=args.min_area,
                min_height=args.min_height,
                min_aspect=args.min_aspect,
                max_width=args.max_width,
                max_iou=args.max_iou,
            )
        )

    summary_path = args.output_dir / "sam2_bamboo_batch_summary.json"
    summary_path.write_text(json.dumps(all_meta, indent=2))
    print(f"[ok] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
