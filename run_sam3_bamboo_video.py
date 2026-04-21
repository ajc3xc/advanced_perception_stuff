#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------------------------------
# HARD CODED PATHS
# ------------------------------------------------

INPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/inputs")
OUTPUT_DIR = Path("/blue/cli2/a.camerer/ABE6399_Robotics/outputs")

PROMPT = "bamboo"
HF_REPO = "jetjodh/sam3"
HF_CKPT = "sam3.pt"

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


# ------------------------------------------------
# UTILITIES
# ------------------------------------------------

def bbox_from_mask(mask):

    ys, xs = np.where(mask)

    if ys.size == 0:
        return (0,0,0,0)

    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()

    return int(x0), int(y0), int(x1-x0+1), int(y1-y0+1)


def area(mask):
    return int(mask.sum())


def label_to_color(labels):

    ids = labels.astype(np.uint32)

    r = (53 * ids + 29) % 256
    g = (97 * ids + 71) % 256
    b = (193 * ids + 11) % 256

    rgb = np.stack([r,g,b], axis=-1).astype(np.uint8)

    rgb[labels == 0] = 0

    return rgb


def overlay(image, labels, alpha=0.6):

    colors = label_to_color(labels)

    img = image.astype(np.float32)

    fg = labels > 0

    img[fg] = (1-alpha)*img[fg] + alpha*colors[fg]

    return img.astype(np.uint8)


# ------------------------------------------------
# SAM3 INFERENCE
# ------------------------------------------------

def sam3_predict(processor, image):

    state = processor.set_image(image)

    state = processor.set_text_prompt(
        PROMPT,
        state
    )

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
        if masks.dtype == torch.bfloat16:
            masks = masks.float()
        masks = masks.numpy()

    if torch.is_tensor(scores):
        scores = scores.detach().cpu()
        if scores.dtype == torch.bfloat16:
            scores = scores.float()
        scores = scores.numpy()

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu()
        if boxes.dtype == torch.bfloat16:
            boxes = boxes.float()
        boxes = boxes.numpy()

    return masks, boxes, scores


# ------------------------------------------------
# PROCESS IMAGE
# ------------------------------------------------

def process_image(img_path, processor):

    image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

    image_np = np.array(image)

    H,W = image_np.shape[:2]

    with torch.inference_mode():

        if torch.cuda.is_available():

            with torch.autocast("cuda", dtype=torch.bfloat16):

                masks, boxes, scores = sam3_predict(processor, image)

        else:

            masks, boxes, scores = sam3_predict(processor, image)

    labels = np.zeros((H,W), dtype=np.uint16)

    coco_anns = []

    inst_id = 1

    for mask, score in zip(masks, scores):

        mask = mask.astype(bool)

        if mask.sum() < 300:
            continue

        labels[mask] = inst_id

        x,y,w,h = bbox_from_mask(mask)

        coco_anns.append({

            "bbox":[x,y,w,h],
            "area":area(mask),
            "category_id":1,
            "iscrowd":0,
            "segmentation":mask.tolist()

        })

        inst_id += 1


    colors = label_to_color(labels)

    ov = overlay(image_np, labels)

    stem = img_path.stem

    Image.fromarray(colors).save(OUTPUT_DIR / f"{stem}_colors.png")
    Image.fromarray(ov).save(OUTPUT_DIR / f"{stem}_overlay.png")

    Image.fromarray(labels, mode="I;16").save(
        OUTPUT_DIR / f"{stem}_labels_u16.png"
    )

    np.save(
        OUTPUT_DIR / f"{stem}_labels_u16.npy",
        labels
    )

    return labels, coco_anns, H, W


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"downloading checkpoint from {HF_REPO}/{HF_CKPT} ...")
    checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CKPT)
    print(f"loading SAM3 model from checkpoint: {checkpoint_path}")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)

    processor = Sam3Processor(
        model,
        confidence_threshold=0.5
    )

    images = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ])

    print("found", len(images), "images")

    coco = {

        "images":[],
        "annotations":[],
        "categories":[
            {"id":1,"name":"bamboo_stalk"}
        ]

    }

    ann_id = 1

    for img_id, img_path in enumerate(images):

        print("processing", img_path.name)

        labels, anns, H, W = process_image(
            img_path,
            processor
        )

        coco["images"].append({

            "id":img_id,
            "file_name":img_path.name,
            "width":W,
            "height":H

        })

        for a in anns:

            a["id"] = ann_id
            a["image_id"] = img_id

            coco["annotations"].append(a)

            ann_id += 1


    coco_path = OUTPUT_DIR / "instances_bamboo_sam3.json"

    with open(coco_path,"w") as f:

        json.dump(coco,f)

    print("COCO annotations saved:", coco_path)


if __name__ == "__main__":
    main()
