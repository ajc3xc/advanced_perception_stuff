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
from PIL import Image, ImageDraw
from transformers import pipeline


DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"
DEFAULT_OUTPUT = "sam2_combined_mask.png"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2 automatic mask generation via Hugging Face.")
    parser.add_argument("--image", type=Path, default=None, help="Input image path. If omitted, create sample.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Combined binary mask output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    args = parser.parse_args()

    if args.image is None:
        image_path = make_sample_image(Path(DEFAULT_SAMPLE))
        print(f"[info] No --image provided, generated sample: {image_path.resolve()}")
    else:
        image_path = args.image
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[info] torch={torch.__version__}, cuda={torch.version.cuda}, gpu_available={torch.cuda.is_available()}")
    print(f"[info] Loading model: {args.model}")
    generator = pipeline("mask-generation", model=args.model, device=device)

    image = Image.open(image_path).convert("RGB")
    print(f"[info] Running SAM2 on: {image_path.resolve()}")
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
    for m in masks:
        combined[normalize_mask(m, shape) > 0] = 255

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(args.output)
    print(f"[ok] masks={len(masks)}")
    print(f"[ok] saved binary mask: {args.output.resolve()}")


if __name__ == "__main__":
    main()
