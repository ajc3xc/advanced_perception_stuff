#!/usr/bin/env python3
import os
import cv2
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Iterable

# --- spaces (HF) optional fallback for local runs ---
try:
    import spaces  # type: ignore
except Exception:
    class _SpacesDummy:  # noqa: N801
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def deco(fn):
                return fn

            return deco
    spaces = _SpacesDummy()

import gradio as gr
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from transformers import (
    Sam3Model, Sam3Processor,
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerModel, Sam3TrackerProcessor
)

# ---------------- config ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_HISTORY = 10
DEFAULT_BATCH_INPUT_DIR = "/blue/cli2/a.camerer/ABE6399_Robotics/inputs/images"
DEFAULT_BATCH_OUTPUT_DIR = "/blue/cli2/a.camerer/ABE6399_Robotics/outputs/images_sam3gradio"

# ---------------- theme ----------------
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class CustomBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

app_theme = CustomBlueTheme()

# ---------------- model load ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO = os.environ.get("SAM3_MODEL_REPO", "jetjodh/sam3")
DEFAULT_IMAGE_PROMPT = os.environ.get("SAM3_DEFAULT_IMAGE_PROMPT", "bamboo")
DEFAULT_VIDEO_PROMPT = os.environ.get("SAM3_DEFAULT_VIDEO_PROMPT", "bamboo")
DEFAULT_BATCH_PROMPT = os.environ.get("SAM3_DEFAULT_BATCH_PROMPT", "bamboo")
print(f"🖥️ Using compute device: {device}")
print(f"📦 Using SAM3 repo: {MODEL_REPO}")
print("⏳ Loading SAM3 Models permanently into memory...")

try:
    print("   ... Loading Image Text Model")
    IMG_MODEL = Sam3Model.from_pretrained(MODEL_REPO).to(device)
    IMG_PROCESSOR = Sam3Processor.from_pretrained(MODEL_REPO)

    print("   ... Loading Image Tracker Model")
    TRK_MODEL = Sam3TrackerModel.from_pretrained(MODEL_REPO).to(device)
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(MODEL_REPO)

    print("   ... Loading Video Model")
    VID_MODEL = Sam3VideoModel.from_pretrained(MODEL_REPO).to(device, dtype=torch.bfloat16)
    VID_PROCESSOR = Sam3VideoProcessor.from_pretrained(MODEL_REPO)

    print("✅ All Models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
    IMG_MODEL = IMG_PROCESSOR = None
    TRK_MODEL = TRK_PROCESSOR = None
    VID_MODEL = VID_PROCESSOR = None

# ---------------- UTILS ----------------
def apply_mask_overlay(base_image, mask_data, opacity=0.5):
    """Draw segmentation mask(s) on top of image. mask_data can be (H,W) or (N,H,W)."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")

    if mask_data is None or (isinstance(mask_data, (list, tuple)) and len(mask_data) == 0):
        return base_image.convert("RGB")

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()
    mask_data = np.asarray(mask_data)

    # Normalize shapes
    if mask_data.ndim == 4:  # (B,N,H,W)
        mask_data = mask_data[0]
    if mask_data.ndim == 3 and mask_data.shape[0] == 1:
        # could be (1,H,W) => one mask
        pass

    if mask_data.ndim == 2:
        masks = [mask_data]
    elif mask_data.ndim == 3:
        masks = [mask_data[i] for i in range(mask_data.shape[0])]
    else:
        return base_image.convert("RGB")

    num_masks = len(masks)
    try:
        cmap = matplotlib.colormaps["rainbow"].resampled(max(num_masks, 1))
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("rainbow").resampled(max(num_masks, 1))

    rgb_colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_masks)]
    composite = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    for i, m in enumerate(masks):
        m = (m > 0).astype(np.uint8)
        mask_bitmap = Image.fromarray((m * 255).astype(np.uint8))
        if mask_bitmap.size != base_image.size:
            mask_bitmap = mask_bitmap.resize(base_image.size, resample=Image.NEAREST)

        fill_color = rgb_colors[i]
        color_fill = Image.new("RGBA", base_image.size, fill_color + (0,))
        mask_alpha = mask_bitmap.point(lambda v: int(v * opacity) if v > 0 else 0)
        color_fill.putalpha(mask_alpha)
        composite = Image.alpha_composite(composite, color_fill)

    return Image.alpha_composite(base_image, composite).convert("RGB")

def draw_points_on_image(image, points, modes=None):
    """Draw points. modes[i] in {'add','erase'} for coloring."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    if modes is None:
        modes = ["add"] * len(points)

    for (x, y), mode in zip(points, modes):
        r = 8
        if mode == "erase":
            fill = (255, 80, 80)     # red-ish
            outline = (255, 255, 255)
        else:
            fill = (80, 255, 120)    # green-ish
            outline = (0, 0, 0)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=3)

    return draw_img

def union_instance_masks(results) -> np.ndarray:
    """Union all instance masks into one semantic mask (H,W) uint8 {0,1}."""
    if results is None or "masks" not in results:
        return None
    masks = results["masks"]
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:  # (B,N,H,W)
        masks = masks[0]
    if masks.ndim == 3:  # (N,H,W)
        sem = np.any(masks > 0, axis=0)
    elif masks.ndim == 2:
        sem = masks > 0
    else:
        return None
    return sem.astype(np.uint8)

def safe_rel_path(root: Path, p: Path) -> Path:
    try:
        return p.relative_to(root)
    except Exception:
        return Path(p.name)


def _normalize_prompt(text_query: str | None, default_prompt: str) -> str:
    if text_query is None:
        return default_prompt
    normalized = str(text_query).strip()
    return normalized if normalized else default_prompt

# ---------------- IMAGE (text) ----------------
@spaces.GPU
def run_image_segmentation(source_img, text_query, conf_thresh=0.5):
    """Original annotated-image demo (unchanged)."""
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Models failed to load on startup.")
    if source_img is None:
        raise gr.Error("Please provide an image.")
    text_query = _normalize_prompt(text_query, DEFAULT_IMAGE_PROMPT)

    pil_image = source_img.convert("RGB")
    model_inputs = IMG_PROCESSOR(images=pil_image, text=text_query, return_tensors="pt").to(device)

    with torch.no_grad():
        inference_output = IMG_MODEL(**model_inputs)

    processed_results = IMG_PROCESSOR.post_process_instance_segmentation(
        inference_output,
        threshold=conf_thresh,
        mask_threshold=0.5,
        target_sizes=model_inputs.get("original_sizes").tolist()
    )[0]

    annotation_list = []
    raw_masks = processed_results.get("masks", None)
    raw_scores = processed_results.get("scores", None)

    if raw_masks is None:
        return (pil_image, annotation_list)

    raw_masks = raw_masks.cpu().numpy()
    raw_scores = raw_scores.cpu().numpy() if raw_scores is not None else np.ones((raw_masks.shape[0],), dtype=np.float32)

    for idx, mask_array in enumerate(raw_masks):
        label_str = f"{text_query} ({raw_scores[idx]:.2f})"
        annotation_list.append((mask_array, label_str))

    return (pil_image, annotation_list)

@spaces.GPU
def run_image_prompt_semantic(source_img, text_query, conf_thresh=0.5):
    """Prompt -> semantic union mask + overlay preview."""
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Models failed to load on startup.")
    if source_img is None:
        raise gr.Error("Please provide an image.")
    text_query = _normalize_prompt(text_query, DEFAULT_IMAGE_PROMPT)

    pil_image = source_img.convert("RGB")
    model_inputs = IMG_PROCESSOR(images=pil_image, text=text_query, return_tensors="pt").to(device)

    with torch.no_grad():
        inference_output = IMG_MODEL(**model_inputs)

    processed_results = IMG_PROCESSOR.post_process_instance_segmentation(
        inference_output,
        threshold=conf_thresh,
        mask_threshold=0.5,
        target_sizes=model_inputs.get("original_sizes").tolist()
    )[0]

    sem = union_instance_masks(processed_results)
    if sem is None:
        sem = np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.uint8)

    overlay = apply_mask_overlay(pil_image, sem, opacity=0.5)
    return pil_image, sem, overlay

# ---------------- TRACKER (single-click mask proposal) ----------------
@spaces.GPU
def tracker_single_click_mask(image_pil: Image.Image, x: int, y: int):
    """Return best tracker mask for a single positive click."""
    if TRK_MODEL is None or TRK_PROCESSOR is None:
        raise gr.Error("Tracker Model failed to load.")
    if image_pil is None:
        return None

    input_points = [[[[int(x), int(y)]]]]  # (img, obj, pt, xy)
    input_labels = [[[1]]]                 # positive click
    inputs = TRK_PROCESSOR(
        images=image_pil,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = TRK_MODEL(**inputs, multimask_output=True)

    # Post-process all masks (object_dim first)
    masks = TRK_PROCESSOR.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
        binarize=True
    )[0]  # shape: (num_objects, num_masks, H, W) OR (num_objects, H, W) depending

    # Choose best mask by iou_scores if available
    if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        ious = outputs.iou_scores.detach().cpu()
        # expected shape (B, obj, M)
        try:
            best_idx = int(torch.argmax(ious[0, 0]).item())
        except Exception:
            best_idx = 0
    else:
        best_idx = 0

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)

    if masks.ndim == 4:
        click_mask = masks[0, best_idx]
    elif masks.ndim == 3:
        # could be (obj, H, W) if multimask_output=False; treat first obj
        click_mask = masks[0]
    else:
        click_mask = None

    if click_mask is None:
        return None

    return (click_mask > 0).astype(np.uint8)

# ---------------- MASK EDITOR TAB LOGIC ----------------
def _push_history(hist, mask, pts, modes):
    hist = hist or []
    # store deep-ish copies
    hist.append((mask.copy(), list(pts), list(modes)))
    if len(hist) > MAX_HISTORY:
        hist = hist[-MAX_HISTORY:]
    return hist

def mask_editor_auto(source_img, prompt, conf, st_hist, st_pts, st_modes):
    pil, sem, overlay = run_image_prompt_semantic(source_img, prompt, conf)
    st_hist = []
    st_pts = []
    st_modes = []
    status = f"✅ Auto mask generated with prompt='{prompt}' (history cleared; max undo={MAX_HISTORY})."
    return pil, sem, overlay, st_hist, st_pts, st_modes, status

def mask_editor_click(evt: gr.SelectData, st_img, st_mask, st_mode, st_hist, st_pts, st_modes):
    if st_img is None or st_mask is None:
        return None, None, [], [], "Upload an image and click **Auto Mask** first."

    x, y = evt.index
    mode = st_mode or "add"

    # push history BEFORE edit
    st_hist = _push_history(st_hist, st_mask, st_pts, st_modes)

    # propose region mask from single click
    click_mask = tracker_single_click_mask(st_img, x, y)
    if click_mask is None:
        overlay = apply_mask_overlay(st_img, st_mask, opacity=0.5)
        overlay = draw_points_on_image(overlay, st_pts, st_modes)
        return overlay, st_mask, st_hist, st_pts, st_modes, "⚠️ Tracker returned no mask."

    # apply add/erase
    if mode == "erase":
        new_mask = (st_mask.astype(np.uint8) & (1 - click_mask.astype(np.uint8))).astype(np.uint8)
    else:
        new_mask = (st_mask.astype(np.uint8) | click_mask.astype(np.uint8)).astype(np.uint8)

    # record point for visualization only
    st_pts = list(st_pts or [])
    st_modes = list(st_modes or [])
    st_pts.append([int(x), int(y)])
    st_modes.append(mode)

    overlay = apply_mask_overlay(st_img, new_mask, opacity=0.5)
    overlay = draw_points_on_image(overlay, st_pts, st_modes)

    return overlay, new_mask, st_hist, st_pts, st_modes, f"✅ Applied {mode} at ({x}, {y})."

def mask_editor_undo(st_img, st_mask, st_hist):
    if st_img is None or st_mask is None:
        return None, None, [], [], [], "Nothing to undo yet."
    if not st_hist:
        overlay = apply_mask_overlay(st_img, st_mask, opacity=0.5)
        return overlay, st_mask, [], [], [], "Nothing to undo (history empty)."

    prev_mask, prev_pts, prev_modes = st_hist.pop()
    overlay = apply_mask_overlay(st_img, prev_mask, opacity=0.5)
    overlay = draw_points_on_image(overlay, prev_pts, prev_modes)
    return overlay, prev_mask, st_hist, prev_pts, prev_modes, f"↩️ Undo (remaining history: {len(st_hist)}/{MAX_HISTORY})."

def mask_editor_reset(st_img, st_auto_mask):
    if st_img is None or st_auto_mask is None:
        return None, None, [], [], [], "Upload an image and run Auto Mask first."
    overlay = apply_mask_overlay(st_img, st_auto_mask, opacity=0.5)
    return overlay, st_auto_mask, [], [], [], "🔄 Reset to auto mask (history cleared)."

def mask_editor_clear_points(st_img, st_mask):
    if st_img is None or st_mask is None:
        return None, None, [], [], "Nothing to clear yet."
    overlay = apply_mask_overlay(st_img, st_mask, opacity=0.5)
    return overlay, st_mask, [], [], "🧹 Cleared point markers (mask unchanged)."

def mask_editor_save(st_mask, out_dir, out_name_stem):
    if st_mask is None:
        return "No mask to save."
    out_dir = (out_dir or "").strip()
    if not out_dir:
        return "Please set an output directory."

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stem = (out_name_stem or "").strip()
    if not stem:
        stem = "mask_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = out_root / f"{stem}.png"
    npy_path = out_root / f"{stem}.npy"

    # save as 0/255 PNG
    m = (st_mask > 0).astype(np.uint8) * 255
    Image.fromarray(m).save(png_path)
    np.save(npy_path, (st_mask > 0).astype(np.uint8))

    return f"💾 Saved: {png_path} and {npy_path}"

# ---------------- VIDEO (unchanged) ----------------
def calc_timeout_duration(vid_file, *args):
    return args[-1] if args else 60

@spaces.GPU(duration=calc_timeout_duration)
def run_video_segmentation(source_vid, text_query, frame_limit, time_limit):
    if VID_MODEL is None or VID_PROCESSOR is None:
        raise gr.Error("Video Models failed to load on startup.")
    if not source_vid:
        raise gr.Error("Missing video.")
    text_query = _normalize_prompt(text_query, DEFAULT_VIDEO_PROMPT)

    try:
        video_cap = cv2.VideoCapture(source_vid)
        vid_fps = video_cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_frames = []
        counter = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret or (frame_limit > 0 and counter >= frame_limit):
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            counter += 1
        video_cap.release()

        session = VID_PROCESSOR.init_video_session(video=video_frames, inference_device=device, dtype=torch.bfloat16)
        session = VID_PROCESSOR.add_text_prompt(inference_session=session, text=text_query)

        temp_out_path = tempfile.mktemp(suffix=".mp4")
        video_writer = cv2.VideoWriter(temp_out_path, cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (vid_w, vid_h))

        for model_out in VID_MODEL.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(video_frames),
        ):
            post_processed = VID_PROCESSOR.postprocess_outputs(session, model_out)
            f_idx = model_out.frame_idx
            original_pil = Image.fromarray(video_frames[f_idx])

            if "masks" in post_processed:
                detected_masks = post_processed["masks"]
                if detected_masks.ndim == 4:
                    detected_masks = detected_masks.squeeze(1)
                final_frame = apply_mask_overlay(original_pil, detected_masks)
            else:
                final_frame = original_pil

            video_writer.write(cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR))

        video_writer.release()
        return temp_out_path, "Video processing completed successfully.✅"

    except Exception as e:
        return None, f"Error during video processing: {str(e)}"

# ---------------- BATCH FOLDER TAB ----------------
@spaces.GPU
def batch_folder_prompt_run(in_dir, out_dir, prompt, conf, recursive, overwrite, save_overlays, progress=gr.Progress()):
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Models failed to load on startup.")

    in_dir = (in_dir or "").strip() or DEFAULT_BATCH_INPUT_DIR
    out_dir = (out_dir or "").strip() or DEFAULT_BATCH_OUTPUT_DIR
    prompt = _normalize_prompt(prompt, DEFAULT_BATCH_PROMPT)

    in_root = Path(in_dir)
    out_root = Path(out_dir)
    if not in_root.exists():
        raise gr.Error(f"Input directory not found: {in_root}")

    out_masks = out_root / "masks"
    out_ovl = out_root / "overlays"
    out_meta = out_root / "meta"
    out_masks.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        out_ovl.mkdir(parents=True, exist_ok=True)

    # gather files
    if recursive:
        paths = [p for p in in_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in in_root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    paths = sorted(paths)
    if not paths:
        return f"No images found in {in_root}."

    t0 = time.time()
    lines = []
    progress(0, desc=f"Found {len(paths)} images. Running prompt='{prompt}'...")

    for i, p in enumerate(paths, start=1):
        rel = safe_rel_path(in_root, p)
        out_stem = rel.with_suffix("")  # keep subfolders
        mask_path = out_masks / (str(out_stem) + ".png")
        npy_path = out_masks / (str(out_stem) + ".npy")
        meta_path = out_meta / (str(out_stem) + ".json")
        ovl_path = out_ovl / (str(out_stem) + ".png")

        # create subdirs
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        if save_overlays:
            ovl_path.parent.mkdir(parents=True, exist_ok=True)

        if (not overwrite) and mask_path.exists():
            lines.append(f"[{i}/{len(paths)}] skip (exists): {rel}")
            progress(i / len(paths), desc=f"Skipping {rel} (exists)")
            continue

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            lines.append(f"[{i}/{len(paths)}] FAIL open {rel}: {e}")
            progress(i / len(paths), desc=f"Failed {rel}")
            continue

        # run prompt semantic
        pil, sem, overlay = run_image_prompt_semantic(img, prompt, conf)

        # save
        Image.fromarray((sem > 0).astype(np.uint8) * 255).save(mask_path)
        np.save(npy_path, (sem > 0).astype(np.uint8))

        meta = {
            "input": str(p),
            "relative": str(rel),
            "prompt": prompt,
            "confidence_threshold": float(conf),
            "shape_hw": [int(sem.shape[0]), int(sem.shape[1])],
            "timestamp": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if save_overlays:
            overlay.save(ovl_path)

        lines.append(f"[{i}/{len(paths)}] ok: {rel}")
        progress(i / len(paths), desc=f"Processed {rel}")

    dt = time.time() - t0
    summary = f"✅ Done. {len(paths)} images processed in {dt:.1f}s. Outputs in: {out_root}"
    return summary + "\n" + "\n".join(lines[-200:])  # last 200 lines

# ---------------- UI ----------------
custom_css = """
#col-container { margin: 0 auto; max-width: 1100px; }
#main-title h1 { font-size: 2.1em !important; }
"""

CTRLZ_JS = r"""
<script>
(function() {
  document.addEventListener('keydown', function(e) {
    const isUndo = (e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z');
    if (!isUndo) return;
    e.preventDefault();
    const btn = document.querySelector('#mask_editor_undo_btn button');
    if (btn) btn.click();
  }, {capture:true});
})();
</script>
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **SAM3: Segment Anything Model 3**", elem_id="main-title")
        gr.Markdown("Segment objects in image or video using **SAM3** with Text Prompts or Interactive Clicks.")

        with gr.Tabs():
            # -------- Original tabs (kept) --------
            with gr.Tab("Image Segmentation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Image", type="pil", height=350)
                        txt_prompt_img = gr.Textbox(
                            label="Text Prompt",
                            value=DEFAULT_IMAGE_PROMPT,
                            placeholder="e.g., bamboo",
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            conf_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                        btn_process_img = gr.Button("Segment Image", variant="primary")

                    with gr.Column(scale=3):
                        image_result = gr.AnnotatedImage(label="Segmented Result", height=410)

                        gr.Examples(
                            examples=[["examples/player.jpg", "player in white", 0.5]],
                            inputs=[image_input, txt_prompt_img, conf_slider],
                            outputs=[image_result],
                            fn=run_image_segmentation,
                            cache_examples=False,
                            label="Image Examples",
                        )

                        btn_process_img.click(
                            fn=run_image_segmentation,
                            inputs=[image_input, txt_prompt_img, conf_slider],
                            outputs=[image_result],
                        )

            with gr.Tab("Video Segmentation"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video", format="mp4", height=320)
                        txt_prompt_vid = gr.Textbox(
                            label="Text Prompt",
                            value=DEFAULT_VIDEO_PROMPT,
                            placeholder="e.g., person running",
                        )

                        with gr.Row():
                            frame_limiter = gr.Slider(10, 500, value=60, step=10, label="Max Frames")
                            time_limiter = gr.Radio([60, 120, 180], value=60, label="Timeout (seconds)")

                        btn_process_vid = gr.Button("Segment Video", variant="primary")

                    with gr.Column():
                        video_result = gr.Video(label="Processed Video")
                        process_status = gr.Textbox(label="System Status", interactive=False)

                        gr.Examples(
                            examples=[["examples/sample_video.mp4", "players", 120, 120]],
                            inputs=[video_input, txt_prompt_vid, frame_limiter, time_limiter],
                            outputs=[video_result, process_status],
                            fn=run_video_segmentation,
                            cache_examples=False,
                            label="Video Examples",
                        )

                btn_process_vid.click(
                    run_video_segmentation,
                    inputs=[video_input, txt_prompt_vid, frame_limiter, time_limiter],
                    outputs=[video_result, process_status],
                )

            with gr.Tab("Image Click Segmentation"):
                # (kept as-is, but now supports +/- label if you want later)
                with gr.Row():
                    with gr.Column(scale=1):
                        img_click_input = gr.Image(type="pil", label="Upload Image", interactive=True, height=450)
                        with gr.Row():
                            img_click_clear = gr.Button("Clear Points & Reset", variant="primary")
                        st_click_points = gr.State([])
                        st_click_labels = gr.State([])

                    with gr.Column(scale=1):
                        img_click_output = gr.Image(type="pil", label="Result Preview", height=450, interactive=False)

                @spaces.GPU
                def run_image_click_gpu(input_image, x, y, points_state, labels_state):
                    if TRK_MODEL is None or TRK_PROCESSOR is None:
                        raise gr.Error("Tracker Model failed to load.")
                    if input_image is None:
                        return input_image, [], []

                    if points_state is None:
                        points_state = []
                    if labels_state is None:
                        labels_state = []

                    points_state.append([x, y])
                    labels_state.append(1)

                    input_points = [[points_state]]
                    input_labels = [[labels_state]]

                    inputs = TRK_PROCESSOR(
                        images=input_image,
                        input_points=input_points,
                        input_labels=input_labels,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        outputs = TRK_MODEL(**inputs, multimask_output=False)

                    masks = TRK_PROCESSOR.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"],
                        binarize=True
                    )[0]

                    # show first object
                    final_img = apply_mask_overlay(input_image, masks[0])
                    final_img = draw_points_on_image(final_img, points_state, ["add"] * len(points_state))
                    return final_img, points_state, labels_state

                def image_click_handler(image, evt: gr.SelectData, points_state, labels_state):
                    x, y = evt.index
                    return run_image_click_gpu(image, x, y, points_state, labels_state)

                img_click_input.select(
                    image_click_handler,
                    inputs=[img_click_input, st_click_points, st_click_labels],
                    outputs=[img_click_output, st_click_points, st_click_labels],
                )

                img_click_clear.click(
                    lambda: (None, [], []),
                    outputs=[img_click_output, st_click_points, st_click_labels],
                )

            # -------- New: Mask Editor tab --------
            with gr.Tab("Mask Editor (Prompt + Add/Erase + Undo)"):
                gr.Markdown(
                    f"""
**Notes**
- **Ctrl+Z** = Undo (up to **{MAX_HISTORY}** steps).
- Workflow: **Auto Mask (Prompt)** → click **Add**/**Erase** regions → **Save Mask**.
- Clicks are applied **one at a time** (stable). Add/Erase controls how the click-mask is combined with the current mask.
"""
                )
                gr.HTML(CTRLZ_JS)

                with gr.Row():
                    with gr.Column(scale=1):
                        me_image_in = gr.Image(type="pil", label="Upload Image", height=420)
                        me_prompt = gr.Textbox(label="Prompt", value=DEFAULT_IMAGE_PROMPT)
                        me_conf = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                        me_mode = gr.Radio(["add", "erase"], value="add", label="Click Mode (combine)")
                        with gr.Row():
                            me_btn_auto = gr.Button("Auto Mask (Prompt)", variant="primary")
                            me_btn_reset = gr.Button("Reset to Auto", variant="secondary")
                        with gr.Row():
                            me_btn_undo = gr.Button("Undo (Ctrl+Z)", elem_id="mask_editor_undo_btn")
                            me_btn_clear_pts = gr.Button("Clear Point Markers")

                        gr.Markdown("**Save**")
                        me_out_dir = gr.Textbox(label="Output directory", placeholder="/path/to/output")
                        me_out_name = gr.Textbox(label="Output stem (no extension)", placeholder="img_0001_bamboo_mask")
                        me_btn_save = gr.Button("Save Mask", variant="primary")
                        me_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=6):
                        me_overlay = gr.Image(type="pil", label="Overlay Preview", height=520, interactive=True)
                        me_mask_bw = gr.Image(type="pil", label="Mask (B/W preview)", height=260, interactive=False)

                # States
                st_me_img = gr.State(None)       # PIL
                st_me_auto = gr.State(None)      # np uint8 HxW
                st_me_mask = gr.State(None)      # np uint8 HxW
                st_me_hist = gr.State([])        # list[(mask, pts, modes)]
                st_me_pts = gr.State([])         # list[[x,y]]
                st_me_modes = gr.State([])       # list['add'|'erase']

                def _bw_preview(mask_np):
                    if mask_np is None:
                        return None
                    m = (np.asarray(mask_np) > 0).astype(np.uint8) * 255
                    return Image.fromarray(m)

                # Auto mask
                def _auto_wrap(img, prompt, conf, hist, pts, modes):
                    pil, sem, overlay, hist, pts, modes, status = mask_editor_auto(img, prompt, conf, hist, pts, modes)
                    return (
                        overlay,
                        _bw_preview(sem),
                        pil, sem, sem,
                        hist, pts, modes,
                        status
                    )

                me_btn_auto.click(
                    _auto_wrap,
                    inputs=[me_image_in, me_prompt, me_conf, st_me_hist, st_me_pts, st_me_modes],
                    outputs=[me_overlay, me_mask_bw, st_me_img, st_me_auto, st_me_mask, st_me_hist, st_me_pts, st_me_modes, me_status],
                )

                # Click apply
                def _click_wrap(img, mask, mode, hist, pts, modes, evt: gr.SelectData):
                    overlay, new_mask, hist, pts, modes, status = mask_editor_click(evt, img, mask, mode, hist, pts, modes)
                    return overlay, _bw_preview(new_mask), new_mask, hist, pts, modes, status

                me_overlay.select(
                    _click_wrap,
                    inputs=[st_me_img, st_me_mask, me_mode, st_me_hist, st_me_pts, st_me_modes],
                    outputs=[me_overlay, me_mask_bw, st_me_mask, st_me_hist, st_me_pts, st_me_modes, me_status],
                )

                # Undo
                def _undo_wrap(img, mask, hist):
                    overlay, prev_mask, hist, pts, modes, status = mask_editor_undo(img, mask, hist)
                    return overlay, _bw_preview(prev_mask), prev_mask, hist, pts, modes, status

                me_btn_undo.click(
                    _undo_wrap,
                    inputs=[st_me_img, st_me_mask, st_me_hist],
                    outputs=[me_overlay, me_mask_bw, st_me_mask, st_me_hist, st_me_pts, st_me_modes, me_status],
                )

                # Reset to auto
                def _reset_wrap(img, auto_mask):
                    overlay, m, hist, pts, modes, status = mask_editor_reset(img, auto_mask)
                    return overlay, _bw_preview(m), m, hist, pts, modes, status

                me_btn_reset.click(
                    _reset_wrap,
                    inputs=[st_me_img, st_me_auto],
                    outputs=[me_overlay, me_mask_bw, st_me_mask, st_me_hist, st_me_pts, st_me_modes, me_status],
                )

                # Clear point markers
                def _clearpts_wrap(img, mask):
                    overlay, m, pts, modes, status = mask_editor_clear_points(img, mask)
                    return overlay, _bw_preview(m), m, pts, modes, status

                me_btn_clear_pts.click(
                    _clearpts_wrap,
                    inputs=[st_me_img, st_me_mask],
                    outputs=[me_overlay, me_mask_bw, st_me_mask, st_me_pts, st_me_modes, me_status],
                )

                # Save
                me_btn_save.click(
                    fn=mask_editor_save,
                    inputs=[st_me_mask, me_out_dir, me_out_name],
                    outputs=[me_status],
                )

            # -------- New: Batch folder prompt tab --------
            with gr.Tab("Batch Folder (Prompt → Save Masks)"):
                gr.Markdown(
                    """
Runs the **text prompt** over an entire folder and saves:
- `masks/*.png` + `masks/*.npy` (semantic union mask)
- `meta/*.json`
- optional `overlays/*.png`

This is the “one button” batch step for a prompt like **bamboo**.
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        bf_in = gr.Textbox(label="Input folder", value=DEFAULT_BATCH_INPUT_DIR, placeholder="/blue/cli2/a.camerer/ABE6399_Robotics/inputs/images")
                        bf_out = gr.Textbox(label="Output folder", value=DEFAULT_BATCH_OUTPUT_DIR, placeholder="/blue/cli2/a.camerer/ABE6399_Robotics/outputs/images_sam3gradio")
                        bf_prompt = gr.Textbox(label="Prompt", value=DEFAULT_BATCH_PROMPT)
                        bf_conf = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                        bf_recursive = gr.Checkbox(value=True, label="Recursive")
                        bf_overwrite = gr.Checkbox(value=False, label="Overwrite existing masks")
                        bf_save_overlays = gr.Checkbox(value=True, label="Save overlay PNGs")
                        bf_btn = gr.Button("Run Batch Folder", variant="primary")

                    with gr.Column(scale=6):
                        bf_log = gr.Textbox(label="Batch Log (tail)", lines=22, interactive=False)

                bf_btn.click(
                    fn=batch_folder_prompt_run,
                    inputs=[bf_in, bf_out, bf_prompt, bf_conf, bf_recursive, bf_overwrite, bf_save_overlays],
                    outputs=[bf_log],
                )

if __name__ == "__main__":
    demo.launch(css=custom_css, theme=app_theme, ssr_mode=False, mcp_server=False, show_error=True)
