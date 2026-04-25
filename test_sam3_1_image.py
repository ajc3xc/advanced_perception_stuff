#!/usr/bin/env python3
"""
SAM3.1 hotswap probe.

Mirrors the prompt -> semantic mask path from app_v2.py (run_image_prompt_semantic),
but points the model loader at Comfy-Org/sam3.1 instead of jetjodh/sam3.

Goal: see whether the existing transformers Sam3Model/Sam3Processor classes accept
the 3.1 checkpoint cleanly. Three outcomes to watch for in stdout:
  1. Clean load + sensible mask  -> free upgrade.
  2. Load with "missing/unexpected keys" warning -> partially compatible.
  3. Hard error on .from_pretrained -> transformers doesn't grok 3.1 architecture yet.

No installs. First run downloads weights into the HF cache.
"""
from pathlib import Path
import os
import sys
import traceback

# Redirect HF cache to D: BEFORE any transformers / huggingface_hub import.
os.environ.setdefault("HF_HOME", r"D:\Xiaoyu's Life\1\hf_cache")

import numpy as np
import torch
from PIL import Image
import matplotlib

INPUT_IMG = Path(r"D:\Xiaoyu's Life\1\data\Field Photos\20241101_113429.jpg")
OUTPUT_DIR = Path(r"D:\Xiaoyu's Life\1\data\outputs")
MODEL_REPO = os.environ.get("SAM3_TEST_REPO", "jetjodh/sam3.1")
PROMPT = os.environ.get("SAM3_TEST_PROMPT", "bamboo")
CONF = float(os.environ.get("SAM3_TEST_CONF", "0.45"))


def overlay(base_pil: Image.Image, mask: np.ndarray, opacity: float = 0.5) -> Image.Image:
    base = base_pil.convert("RGBA")
    if mask is None:
        return base.convert("RGB")
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = np.any(m > 0, axis=0).astype(np.uint8)
    color = (70, 200, 120)
    fill = Image.new("RGBA", base.size, color + (0,))
    bm = Image.fromarray((m * 255).astype(np.uint8))
    if bm.size != base.size:
        bm = bm.resize(base.size, resample=Image.NEAREST)
    fill.putalpha(bm.point(lambda v: int(v * opacity) if v > 0 else 0))
    return Image.alpha_composite(base, fill).convert("RGB")


def main() -> int:
    if not INPUT_IMG.exists():
        print(f"[FAIL] Input not found: {INPUT_IMG}")
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] repo ={MODEL_REPO}")
    print(f"[info] image={INPUT_IMG}")
    print(f"[info] prompt={PROMPT!r}  conf={CONF}")

    try:
        from transformers import Sam3Model, Sam3Processor, Sam3Config
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print(f"[FAIL] transformers/huggingface_hub import: {e}")
        return 3

    try:
        print("[step] loading processor...")
        processor = Sam3Processor.from_pretrained(MODEL_REPO)

        # The 3.1 repo ships weights as `sam3.1_multiplex.pt`, which transformers
        # `from_pretrained` does not auto-discover. Manual load: build from config,
        # download the .pt, load_state_dict(strict=False), report key diff.
        weight_filename = os.environ.get("SAM3_TEST_WEIGHT_FILE", "sam3.1_multiplex.pt")
        print(f"[step] downloading weights ({weight_filename}) to HF cache...")
        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=weight_filename)
        print(f"[step] loading config + building empty Sam3Model...")
        config = Sam3Config.from_pretrained(MODEL_REPO)
        model = Sam3Model(config).to(device)

        print(f"[step] torch.load({Path(ckpt_path).name})...")
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            print("       (weights_only=True failed; retrying with weights_only=False)")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(state, dict):
            for k in ("model", "state_dict", "model_state_dict"):
                if k in state and isinstance(state[k], dict):
                    print(f"       unwrapping checkpoint dict via key '{k}'")
                    state = state[k]
                    break

        print("[step] load_state_dict(strict=False)...")
        result = model.load_state_dict(state, strict=False)
        n_miss = len(result.missing_keys)
        n_unx = len(result.unexpected_keys)
        n_ckpt = len(state) if isinstance(state, dict) else -1
        print(f"[diag] checkpoint keys: {n_ckpt}")
        print(f"[diag] missing  in model (model has, ckpt doesn't): {n_miss}")
        print(f"[diag] unexpected in ckpt (ckpt has, model doesn't): {n_unx}")
        if n_miss:
            print("       sample missing   :", result.missing_keys[:5])
        if n_unx:
            print("       sample unexpected:", result.unexpected_keys[:5])
        if n_miss == 0 and n_unx == 0:
            print("[diag] ✅ perfect key match — 3.1 weights fully fit transformers Sam3Model.")
        elif n_unx > 0 and n_miss == 0:
            print("[diag] ⚠️ ckpt has extra keys (likely the multiplex head). Architecture in"
                  " transformers is older — running as base SAM3 with 3.1 backbone weights.")
        elif n_miss > n_unx:
            print("[diag] ❌ many model layers got no weights. Output will be garbage.")
        model.eval()
    except Exception as e:
        print("[FAIL] model load.")
        print(f"       error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n[hint] fall back: SAM3_TEST_REPO=jetjodh/sam3 python test_sam3_1_image.py")
        return 4

    img = Image.open(INPUT_IMG).convert("RGB")

    try:
        inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        results = processor.post_process_instance_segmentation(
            out,
            threshold=CONF,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
    except Exception as e:
        print("[FAIL] inference / post-processing.")
        print(f"       error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 5

    masks = results.get("masks", None)
    scores = results.get("scores", None)
    if masks is None or len(masks) == 0:
        print("[warn] no masks returned (model loaded fine — prompt or threshold may need tuning).")
        sem = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    else:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[0]
        if masks.ndim == 2:
            sem = (masks > 0).astype(np.uint8)
        else:
            sem = np.any(masks > 0, axis=0).astype(np.uint8)
        n_inst = 1 if masks.ndim == 2 else masks.shape[0]
        s_str = "n/a"
        if scores is not None:
            s = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            s_str = f"min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}"
        print(f"[ok]   {n_inst} instances; score stats: {s_str}; sem coverage: {sem.mean()*100:.2f}%")

    stem = INPUT_IMG.stem + "__sam3_1_probe"
    mask_png = OUTPUT_DIR / f"{stem}_mask.png"
    overlay_png = OUTPUT_DIR / f"{stem}_overlay.png"
    npy_path = OUTPUT_DIR / f"{stem}_mask.npy"

    Image.fromarray((sem > 0).astype(np.uint8) * 255).save(mask_png)
    np.save(npy_path, (sem > 0).astype(np.uint8))
    overlay(img, sem, opacity=0.5).save(overlay_png)

    print(f"[save] {mask_png}")
    print(f"[save] {overlay_png}")
    print(f"[save] {npy_path}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
