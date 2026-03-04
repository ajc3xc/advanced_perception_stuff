#!/usr/bin/env python3
"""Minimal SAM2 (Hugging Face) mask-generation test script.

Usage examples:
  pixi run python run_sam2_hf_test.py
  pixi run python run_sam2_hf_test.py --image /abs/path/image.jpg
  pixi run python run_sam2_hf_test.py --image /abs/path/image.jpg --output /abs/path/mask.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
print(".")
from PIL import Image, ImageDraw
print("..")
from transformers import pipeline


DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"
DEFAULT_OUTPUT = "sam2_combined_mask.png"
DEFAULT_LABEL_OUTPUT = "sam2_instance_labels.png"
DEFAULT_COLOR_OUTPUT = "sam2_instance_colors.png"
DEFAULT_OVERLAY_OUTPUT = "sam2_overlay.png"
DEFAULT_SAMPLE = "sam2_sample_input.png"


def make_sample_image(path: Path) -> Path:
    """Create a simple synthetic image if user did not provide one."""
    w, h = 960, 640
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle((120, 90, 430, 330), fill=(210, 70, 70), outline=(20, 20, 20), width=4)
    d.ellipse((520, 120, 860, 470), fill=(60, 120, 220), outline=(20, 20, 20), width=4)
    d.polygon([(130, 510), (370, 400), (560, 540)], fill=(80, 180, 110), outline=(20, 20, 20))
    img.save(path)
    return path


def normalize_mask(mask_obj: object, shape: tuple[int, int]) -> np.ndarray:
    """Convert model mask output to HxW uint8 boolean-like array."""
    if isinstance(mask_obj, Image.Image):
        arr = np.array(mask_obj)
    else:
        arr = np.array(mask_obj)

    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.shape != shape:
        arr = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize(shape[::-1], Image.NEAREST))

    return (arr > 0).astype(np.uint8)


def label_to_color_image(labels: np.ndarray) -> np.ndarray:
    """Map label ids to RGB colors for visualization."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[0] = np.array([0, 0, 0], dtype=np.uint8)
    for i in range(1, 256):
        lut[i] = np.array([(53 * i) % 256, (97 * i) % 256, (193 * i) % 256], dtype=np.uint8)
    return lut[labels]


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return boundary pixels where neighboring labels differ."""
    b = np.zeros_like(labels, dtype=bool)
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 automatic mask generation via Hugging Face.")
    parser.add_argument("--image", type=Path, default=None, help="Input image path. If omitted, create sample.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Combined binary mask output.")
    parser.add_argument(
        "--label-output",
        type=Path,
        default=Path(DEFAULT_LABEL_OUTPUT),
        help="Instance label map output (0=background, 1..255=object id).",
    )
    parser.add_argument(
        "--color-output",
        type=Path,
        default=Path(DEFAULT_COLOR_OUTPUT),
        help="Colorized instance map output.",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=Path(DEFAULT_OVERLAY_OUTPUT),
        help="Overlay image with colored instances and boundaries.",
    )
    parser.add_argument("--alpha", type=float, default=0.58, help="Overlay opacity in [0,1].")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    args = parser.parse_args()

    if args.image is None:
        image_path = make_sample_image(Path(DEFAULT_SAMPLE))
        print(f"[info] No --image provided, generated sample: {image_path.resolve()}", flush=True)
    else:
        image_path = args.image
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

    device = 0 if torch.cuda.is_available() else -1
    print(
        f"[info] torch={torch.__version__}, cuda={torch.version.cuda}, gpu_available={torch.cuda.is_available()}",
        flush=True,
    )
    print(f"[info] Loading model: {args.model} (first run may download weights)", flush=True)
    generator = pipeline("mask-generation", model=args.model, device=device)

    image = Image.open(image_path).convert("RGB")
    print(f"[info] Running SAM2 on: {image_path.resolve()}", flush=True)
    outputs = generator(image)

    if isinstance(outputs, list):
        masks = outputs
    elif isinstance(outputs, dict):
        masks = outputs.get("masks", [])
    else:
        masks = []

    if not masks:
        raise RuntimeError("No masks returned by model.")

    shape = (image.height, image.width)
    combined = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint8)

    max_instances = min(len(masks), 255)
    if len(masks) > 255:
        print(
            f"[warn] model returned {len(masks)} masks; only first {max_instances} labeled (uint8 limit).",
            flush=True,
        )

    for idx, m in enumerate(masks[:max_instances], start=1):
        current = normalize_mask(m, shape) > 0
        combined[current] = 255
        labels[current] = idx

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.label_output.parent.mkdir(parents=True, exist_ok=True)
    args.color_output.parent.mkdir(parents=True, exist_ok=True)
    args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(args.output)
    Image.fromarray(labels, mode="L").save(args.label_output)
    colors = label_to_color_image(labels)
    Image.fromarray(colors, mode="RGB").save(args.color_output)

    alpha = float(max(0.0, min(1.0, args.alpha)))
    image_np = np.array(image, dtype=np.float32)
    colors_np = colors.astype(np.float32)
    fg = labels > 0
    overlay = image_np.copy()
    overlay[fg] = (1.0 - alpha) * image_np[fg] + alpha * colors_np[fg]

    # Blue-ish boundaries similar to common SAM visualizations.
    edges = label_boundaries(labels)
    overlay[edges] = np.array([45, 110, 255], dtype=np.float32)
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay_u8, mode="RGB").save(args.overlay_output)

    print(f"[ok] masks={len(masks)}", flush=True)
    print(f"[ok] saved binary mask: {args.output.resolve()}", flush=True)
    print(f"[ok] saved label map: {args.label_output.resolve()}", flush=True)
    print(f"[ok] saved color map: {args.color_output.resolve()}", flush=True)
    print(f"[ok] saved overlay map: {args.overlay_output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
