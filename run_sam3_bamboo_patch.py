#!/usr/bin/env python3
"""
Batch bamboo stalk labeling with SAM3 (pip installable).

Exports per image:
- uint16 instance label PNG + NPY
- colored debug label PNG
- overlay PNG
- meta JSON

Exports dataset:
- COCO instances JSON (RLE; pycocotools if installed, else uncompressed RLE)

Prompt is fixed to: "bamboo"

Usage:
  python sam3_bamboo_batch.py --input-dir /path/to/images --output-dir /path/to/out --recursive

Optional:
  pip install pycocotools   # for compressed COCO RLE (smaller JSON)

Notes:
- Uses the pip-style API: from sam3 import build_sam3_image_model + Sam3Processor
- Uses inference_mode + autocast (bf16 on cuda) for speed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

# --- SAM3 imports (pip-style) ---
# Your environment shows sam3 imports but lacks sam3.model_builder; this uses the style you requested.
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROMPT = "bamboo"
DEFAULT_INPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/inputs/images")
DEFAULT_OUTPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/outputs/images")
DEFAULT_HF_REPO = "jetjodh/sam3"
DEFAULT_HF_CKPT = "sam3.pt"


# ----------------------------- COCO RLE helpers -----------------------------
def _try_import_pycocotools():
    try:
        from pycocotools import mask as mask_utils  # type: ignore
        return mask_utils
    except Exception:
        return None


def rle_encode_uncompressed(mask: np.ndarray) -> Dict[str, Any]:
    """
    Uncompressed COCO-style RLE. Many tools can read this, but it's larger than pycocotools' compressed form.
    COCO expects Fortran order (column-major).
    """
    m = np.asfortranarray(mask.astype(np.uint8))
    h, w = m.shape
    flat = m.reshape(-1, order="F")

    counts: List[int] = []
    prev = 0
    run_len = 0
    for v in flat:
        v = int(v)
        if v == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = v
    counts.append(run_len)
    return {"size": [h, w], "counts": counts}


def rle_encode(mask: np.ndarray) -> Dict[str, Any]:
    """
    Prefer pycocotools compressed RLE; otherwise fall back to uncompressed.
    """
    mask_utils = _try_import_pycocotools()
    if mask_utils is None:
        return rle_encode_uncompressed(mask)

    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    # pycocotools returns bytes counts; JSON needs str
    if isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ----------------------------- geometry + debug helpers -----------------------------
def bbox_xywh_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))


def area_from_mask(mask: np.ndarray) -> int:
    return int(mask.sum())


def label_to_color_image_uint16(labels: np.ndarray) -> np.ndarray:
    """Deterministic mapping from uint16 ids to RGB colors."""
    ids = labels.astype(np.uint32)
    r = (53 * ids + 29) % 256
    g = (97 * ids + 71) % 256
    b = (193 * ids + 11) % 256
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[labels == 0] = 0
    return rgb


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    b = np.zeros_like(labels, dtype=bool)
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return b


def make_overlay(image_rgb_u8: np.ndarray, labels_u16: np.ndarray, alpha: float) -> np.ndarray:
    colors = label_to_color_image_uint16(labels_u16)
    img = image_rgb_u8.astype(np.float32)
    fg = labels_u16 > 0
    img[fg] = (1.0 - alpha) * img[fg] + alpha * colors[fg].astype(np.float32)

    # draw boundaries
    edges = label_boundaries(labels_u16)
    img[edges] = np.array([45, 110, 255], dtype=np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------- SAM3 inference -----------------------------
def sam3_predict(processor: Sam3Processor, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      masks: (N,H,W) boolean-ish
      boxes: (N,4) (format depends on implementation; we don't rely on it)
      scores:(N,)
    """
    state = processor.set_image(image)
    # per your requested API style: state = processor.set_text_prompt("person", state)
    state = processor.set_text_prompt(PROMPT, state)

    masks = state.get("masks", None)
    boxes = state.get("boxes", None)
    scores = state.get("scores", None)

    if masks is None:
        masks = np.zeros((0, image.height, image.width), dtype=np.uint8)
    if boxes is None:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores is None:
        scores = np.zeros((0,), dtype=np.float32)

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
        if masks.dtype == torch.bfloat16:
            masks = masks.float()
        masks = masks.numpy()
    else:
        masks = np.array(masks)

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu()
        if boxes.dtype == torch.bfloat16:
            boxes = boxes.float()
        boxes = boxes.numpy()
    else:
        boxes = np.array(boxes)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu()
        if scores.dtype == torch.bfloat16:
            scores = scores.float()
        scores = scores.numpy()
    else:
        scores = np.array(scores)

    # Normalize masks to shape (N,H,W)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 3:
        pass
    else:
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, :, :]

    return masks, boxes, scores


# ----------------------------- per-image processing -----------------------------
@dataclass
class PerImageResult:
    meta: Dict[str, Any]
    coco_annotations: List[Dict[str, Any]]
    width: int
    height: int
    file_name: str


def process_image(
    img_path: Path,
    out_dir: Path,
    processor: Sam3Processor,
    alpha: float,
    min_area: int,
    score_thresh: float,
) -> PerImageResult:
    image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    H, W = image_np.shape[:2]

    t0 = time.perf_counter()
    # speed: inference_mode + autocast on cuda
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    use_autocast = (device_type == "cuda")

    with torch.inference_mode():
        if use_autocast:
            # bf16 is usually a good speed/accuracy tradeoff on recent GPUs
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, _boxes, scores = sam3_predict(processor, image)
        else:
            masks, _boxes, scores = sam3_predict(processor, image)

    dt = time.perf_counter() - t0

    # Build uint16 instance labels
    labels = np.zeros((H, W), dtype=np.uint16)
    coco_anns: List[Dict[str, Any]] = []

    inst_id = 1
    total_masks = int(masks.shape[0]) if masks is not None else 0

    for i in range(total_masks):
        score = float(scores[i]) if i < len(scores) else 1.0
        if score < score_thresh:
            continue

        m = masks[i]
        m = (m > 0).astype(bool)

        a = area_from_mask(m)
        if a < min_area:
            continue

        labels[m] = inst_id

        x, y, w, h = bbox_xywh_from_mask(m)
        coco_anns.append(
            {
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": int(a),
                "segmentation": rle_encode(m),
                "iscrowd": 0,
                "score": float(score),  # non-standard; helpful for debugging
            }
        )
        inst_id += 1

    # Save debug + labels
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    label_png = out_dir / f"{stem}_labels_u16.png"
    label_npy = out_dir / f"{stem}_labels_u16.npy"
    colors_png = out_dir / f"{stem}_colors.png"
    overlay_png = out_dir / f"{stem}_overlay.png"
    meta_json = out_dir / f"{stem}_meta.json"

    Image.fromarray(labels, mode="I;16").save(label_png)
    np.save(label_npy, labels)

    colors = label_to_color_image_uint16(labels)
    Image.fromarray(colors, mode="RGB").save(colors_png)

    ov = make_overlay(image_np, labels, alpha=alpha)
    Image.fromarray(ov, mode="RGB").save(overlay_png)

    meta = {
        "image": str(img_path),
        "file_name": img_path.name,
        "width": int(W),
        "height": int(H),
        "prompt": PROMPT,
        "sam3_masks_total": int(total_masks),
        "score_thresh": float(score_thresh),
        "min_area": int(min_area),
        "instances_kept": int(labels.max()),
        "runtime_sec": float(dt),
        "outputs": {
            "labels_u16_png": str(label_png),
            "labels_u16_npy": str(label_npy),
            "colors_png": str(colors_png),
            "overlay_png": str(overlay_png),
            "meta_json": str(meta_json),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    return PerImageResult(meta=meta, coco_annotations=coco_anns, width=W, height=H, file_name=img_path.name)


# ----------------------------- dataset walk + COCO -----------------------------
def iter_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch bamboo stalk labeling with SAM3 (text prompt).")
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--recursive", action="store_true", default=True, help="Recurse into subfolders (mirrors structure under output).")
    ap.add_argument("--no-recursive", action="store_false", dest="recursive", help="Disable recursion.")

    # Filters / debug
    ap.add_argument("--confidence-threshold", type=float, default=0.5, help="SAM3 processor confidence_threshold.")
    ap.add_argument("--score-thresh", type=float, default=0.0, help="Extra score filter on returned instances.")
    ap.add_argument("--min-area", type=int, default=300, help="Drop masks smaller than this many pixels.")
    ap.add_argument("--alpha", type=float, default=0.60, help="Overlay alpha in [0,1].")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="Hugging Face repo containing SAM3 checkpoint.")
    ap.add_argument("--hf-ckpt", default=DEFAULT_HF_CKPT, help="Checkpoint filename within --hf-repo.")

    args = ap.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    # Speed toggles (safe defaults)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[info] downloading checkpoint: repo={args.hf_repo} file={args.hf_ckpt}", flush=True)
    checkpoint_path = hf_hub_download(repo_id=str(args.hf_repo), filename=str(args.hf_ckpt))
    print(f"[info] loading SAM3 LARGE model from checkpoint: {checkpoint_path}", flush=True)
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)

    # per your requested usage style
    processor = Sam3Processor(model, confidence_threshold=float(args.confidence_threshold))

    images = iter_images(args.input_dir, recursive=args.recursive)
    if not images:
        raise RuntimeError(f"No images found under: {args.input_dir}")
    print(f"[info] found {len(images)} image(s)", flush=True)
    print(f"[info] prompt='{PROMPT}'", flush=True)

    # COCO containers
    coco = {
        "info": {"description": "SAM3 bamboo stalk pseudo-labels", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "bamboo_stalk", "supercategory": "plant"}],
        "images": [],
        "annotations": [],
    }

    all_meta: List[Dict[str, Any]] = []
    ann_id = 1

    for img_id, img_path in enumerate(images, start=1):
        rel_parent = img_path.parent.relative_to(args.input_dir) if args.recursive else Path(".")
        out_dir = args.output_dir / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[info] ({img_id}/{len(images)}) {img_path}", flush=True)

        res = process_image(
            img_path=img_path,
            out_dir=out_dir,
            processor=processor,
            alpha=alpha,
            min_area=int(args.min_area),
            score_thresh=float(args.score_thresh),
        )

        all_meta.append(res.meta)

        coco["images"].append(
            {
                "id": img_id,
                "file_name": str((rel_parent / res.file_name).as_posix()) if args.recursive else res.file_name,
                "width": int(res.width),
                "height": int(res.height),
            }
        )

        for a in res.coco_annotations:
            a = dict(a)
            a["id"] = ann_id
            a["image_id"] = img_id
            coco["annotations"].append(a)
            ann_id += 1

    summary_path = args.output_dir / "sam3_bamboo_batch_summary.json"
    summary_path.write_text(json.dumps(all_meta, indent=2))

    coco_path = args.output_dir / "instances_bamboo_sam3_coco.json"
    coco_path.write_text(json.dumps(coco, indent=2))

    print(f"[ok] wrote summary: {summary_path}", flush=True)
    print(f"[ok] wrote COCO:    {coco_path}", flush=True)

    if _try_import_pycocotools() is None:
        print("[note] For smaller COCO JSON (compressed RLE), install: pip install pycocotools", flush=True)


if __name__ == "__main__":
    main()
