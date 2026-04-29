#!/usr/bin/env python3
"""
SAM3.1 Bamboo Annotation Tool — app_v5.py

Identical to app_v4.py EXCEPT:
  - MODEL_REPO defaults to "jetjodh/sam3.1"  (Object Multiplex checkpoint)
  - sam3 package installed from facebookresearch/sam3 GitHub (has Object Multiplex code)
  - secret_token.txt read from parent directory (../secret_token.txt)

Object Multiplex (SAM3.1) processes all tracked objects jointly instead of one
at a time, giving ~7x speedup at 128 objects vs the November 2025 SAM3 release.
This directly fixes the VRAM death spiral (0x119 BSOD) seen with 50+ bamboo objects.

Tab 1 — Image Segmentation   : native SAM3 image predictor, PCS demo, self-contained
Tab 2 — Mask Editor          : image mode (PCS+PVS stateless) OR video mode
                               video mode uses native SAM3 video predictor session
                               PCS detects ALL bamboo instances simultaneously
                               PVS refines per-object via handle_request API
                               10-step undo history via session replay
                               shared session feeds Tab 3
Tab 3 — Video Propagator     : visible ONLY when video session active in Tab 2
                               propagates ALL tracked objects frame by frame
                               saves per-frame label maps + overlay MP4
                               min_region_px filter applied on output only
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import cv2
import json
import time
import psutil
import tempfile
import shutil
import threading
import glob
from pathlib import Path
from datetime import datetime

# --- HF Spaces GPU decorator (no-op locally) ---------------------------------
try:
    import spaces  # type: ignore
except Exception:
    class _SpacesDummy:
        @staticmethod
        def GPU(*args, **kwargs):
            def deco(fn): return fn
            return deco
    spaces = _SpacesDummy()

import gradio as gr
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# ── native SAM3 imports ───────────────────────────────────────────────────────
try:
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from sam3.visualization_utils import prepare_masks_for_visualization
    _SAM3_NATIVE = True
except Exception as e:
    print(f"⚠️  native sam3 import failed: {e}")
    _SAM3_NATIVE = False

# ── transformers fallback for image mode (Tab 1 + Tab 2 image) ───────────────
try:
    from transformers import (
        Sam3Model, Sam3Processor,
        Sam3TrackerModel, Sam3TrackerProcessor,
    )
    _TRANSFORMERS = True
except Exception:
    _TRANSFORMERS = False

# ── config ────────────────────────────────────────────────────────────────────
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS      = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VID_HIST  = 10    # max undo steps for video session
MAX_IMG_HIST  = 10    # max undo steps for image mask editor
DEFAULT_INPUT_DIR  = ""
DEFAULT_OUTPUT_DIR = ""
def _get_saved_input_dir():
    return _load_config().get("input_dir", DEFAULT_INPUT_DIR)
def _get_saved_output_dir():
    return _load_config().get("output_dir", DEFAULT_OUTPUT_DIR)
DEFAULT_PROMPT     = os.environ.get("SAM3_DEFAULT_PROMPT", "bamboo")
# ── v5 change: default to jetjodh/sam3.1 (Object Multiplex checkpoint) ───────
MODEL_REPO         = os.environ.get("SAM3_MODEL_REPO",    "jetjodh/sam3.1")
# jetjodh/sam3.1 only has the native checkpoint (sam3.pt), no transformers weights.
# Fall back to jetjodh/sam3 for the transformers image models (Tab 1 + Tab 2 image mode).
IMG_MODEL_REPO     = os.environ.get("SAM3_IMG_MODEL_REPO", "jetjodh/sam3")
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", "")
# Config lives next to the app file — always writable, no dependency on output dir
CONFIG_PATH        = Path(__file__).parent / ".sam3_ui_config.json"

def _load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}

def _save_config(key: str, value):
    try:
        cfg = _load_config()
        cfg[key] = value
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        print(f"[_save_config] {key}={value!r} → {CONFIG_PATH}")
    except Exception as e:
        print(f"[_save_config] failed: {e}")


def _norm_path(p: str) -> str:
    if not p:
        return p
    p = p.strip()
    if len(p) >= 2 and p[0] in ('"', "'") and p[-1] == p[0]:
        p = p[1:-1].strip()
    p = p.replace("\\", "/").replace("\\", "/")
    p = p.rstrip("/")
    return p


def _resolve_sam3_checkpoint() -> tuple[str | None, str | None]:
    """
    Return (checkpoint_path, bpe_path) for build_sam3_video_predictor.
    Priority: SAM3_CHECKPOINT_PATH env-var → HF hub download from MODEL_REPO.
    jetjodh/sam3.1 is ungated so no token required.
    """
    if SAM3_CHECKPOINT_PATH:
        ckpt = Path(SAM3_CHECKPOINT_PATH)
        if ckpt.exists():
            print(f"[_resolve_sam3_checkpoint] using SAM3_CHECKPOINT_PATH: {ckpt}")
            return str(ckpt), None
        print(f"[_resolve_sam3_checkpoint] SAM3_CHECKPOINT_PATH set but not found: {ckpt}")

    try:
        from huggingface_hub import hf_hub_download
        cache_dir = Path.home() / ".cache" / "sam3_local"
        cache_dir.mkdir(parents=True, exist_ok=True)

        hf_token = os.environ.get("HF_TOKEN") or None
        # ── v5 change: token file is one level up (../secret_token.txt) ───────
        token_file = Path(__file__).parent.parent / "secret_token.txt"
        if not hf_token and token_file.exists():
            try:
                hf_token = token_file.read_text().strip() or None
                if hf_token:
                    print(f"[_resolve_sam3_checkpoint] HF token loaded from {token_file}")
            except Exception:
                pass
        if not hf_token:
            print(f"[_resolve_sam3_checkpoint] no HF token — jetjodh/sam3.1 is ungated, should be fine")

        print(f"[_resolve_sam3_checkpoint] downloading sam3.1_multiplex.pt from facebook/sam3.1 …")
        ckpt_path = hf_hub_download(
            repo_id="facebook/sam3.1",
            filename="sam3.1_multiplex.pt",
            cache_dir=str(cache_dir),
            token=hf_token,
        )
        print(f"[_resolve_sam3_checkpoint] checkpoint → {ckpt_path}")

        bpe_path = None
        try:
            bpe_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="bpe_simple_vocab_16e6.txt.gz",
                cache_dir=str(cache_dir),
            )
            print(f"[_resolve_sam3_checkpoint] BPE vocab → {bpe_path}")
        except Exception as bpe_err:
            print(f"[_resolve_sam3_checkpoint] BPE not in repo ({bpe_err}), package will use bundled copy")

        return str(ckpt_path), bpe_path

    except Exception as e:
        print(f"[_resolve_sam3_checkpoint] download failed: {e}")
        return None, None

# ── instrumentation ───────────────────────────────────────────────────────────
def _mem() -> str:
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        return f"RAM={rss:.2f}GB VRAM={vram:.2f}GB"
    return f"RAM={rss:.2f}GB"

# ── theme ─────────────────────────────────────────────────────────────────────
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)

class CustomBlueTheme(Soft):
    def __init__(self, *, primary_hue=colors.gray, secondary_hue=colors.steel_blue,
                 neutral_hue=colors.slate, text_size=sizes.text_lg,
                 font=(fonts.GoogleFont("Outfit"), "Arial", "sans-serif"),
                 font_mono=(fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace")):
        super().__init__(primary_hue=primary_hue, secondary_hue=secondary_hue,
                         neutral_hue=neutral_hue, text_size=text_size, font=font, font_mono=font_mono)
        super().set(
            background_fill_primary="*primary_50",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            button_primary_text_color="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            slider_color="*secondary_500",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

app_theme = CustomBlueTheme()

# ── model load ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  device={device}  {_mem()}")

VID_PREDICTOR = None
IMG_MODEL     = None
IMG_PROCESSOR = None
TRK_MODEL     = None
TRK_PROCESSOR = None

# ── set HF_TOKEN before any HuggingFace downloads ────────────────────────────
# ── v5 change: secret_token.txt is one level up ──────────────────────────────
_token_file = Path(__file__).parent.parent / "secret_token.txt"
if _token_file.exists() and "HF_TOKEN" not in os.environ:
    _tok = _token_file.read_text().strip()
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        print(f"[models] HF_TOKEN set from {_token_file}")

# ── load transformers image models ──────────────────────────────────────────
# IMG_MODEL/PROCESSOR: lazy — NOT loaded at startup. Loaded on first use in
#   Tab 1 or Tab 2 image mode via _ensure_img_model(). Saves ~2GB VRAM
#   permanently if you only use video mode.
# TRK_MODEL/PROCESSOR: loaded to CPU now. Moved to GPU on demand for click
#   refinement, moved back to CPU immediately after. Saves ~1.5GB VRAM
#   during video propagation.
try:
    if _TRANSFORMERS:
        print("⏳  Loading TRK_MODEL to CPU (lazy GPU on click) …")
        TRK_MODEL     = Sam3TrackerModel.from_pretrained(IMG_MODEL_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True)  # stays on CPU
        TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(IMG_MODEL_REPO)
        print(f"✅  TRK_MODEL on CPU ready.  {_mem()}")
        print("ℹ️   IMG_MODEL deferred — will load on first image segmentation use.")
except BaseException as e:
    import traceback
    print(f"❌  TRK model load failed: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()

_img_model_loaded = False  # tracks whether IMG_MODEL has been loaded yet
_img_model_lock   = threading.Lock()

def _ensure_img_model():
    """Lazy-load IMG_MODEL/PROCESSOR to GPU on first use. Thread-safe."""
    global IMG_MODEL, IMG_PROCESSOR, _img_model_loaded
    if _img_model_loaded:
        return
    with _img_model_lock:
        if _img_model_loaded:  # double-check after acquiring lock
            return
        if not _TRANSFORMERS:
            raise gr.Error("transformers not available — image model cannot be loaded.")
        try:
            print(f"⏳  Lazy-loading IMG_MODEL to GPU …  {_mem()}")
            IMG_MODEL     = Sam3Model.from_pretrained(IMG_MODEL_REPO, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
            IMG_PROCESSOR = Sam3Processor.from_pretrained(IMG_MODEL_REPO)
            _img_model_loaded = True
            print(f"✅  IMG_MODEL loaded.  {_mem()}")
        except Exception as _le:
            raise gr.Error(f"IMG_MODEL load failed: {_le}")

def _trk_to_gpu():
    """Move TRK_MODEL to GPU for inference."""
    if TRK_MODEL is not None and next(TRK_MODEL.parameters()).device.type != device:
        print(f"[trk] moving TRK_MODEL → GPU  {_mem()}")
        TRK_MODEL.to(device)
        torch.cuda.synchronize()
        print(f"[trk] TRK_MODEL on GPU  {_mem()}")

def _trk_to_cpu():
    """Move TRK_MODEL back to CPU to free VRAM."""
    if TRK_MODEL is not None and next(TRK_MODEL.parameters()).device.type != 'cpu':
        print(f"[trk] moving TRK_MODEL → CPU  {_mem()}")
        TRK_MODEL.to('cpu')
        torch.cuda.empty_cache()
        print(f"[trk] TRK_MODEL on CPU  {_mem()}")

# ── Windows patches: apply sam3_vitdet.py fixes before loading predictor ─────
# Same patches as v4: triton→CPU connected-components, NMS, vitdet geometry.
try:
    _patch_script = Path(__file__).parent / "sam3_vitdet.py"
    if _patch_script.exists():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("sam3_vitdet", _patch_script)
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        print(f"[patches] sam3_vitdet.py applied from {_patch_script}")
    else:
        print(f"[patches] sam3_vitdet.py not found at {_patch_script} — skipping")
except Exception as _pe:
    print(f"[patches] sam3_vitdet.py failed: {_pe}")

# ── load native SAM3.1 video predictor ───────────────────────────────────────
_available_gb = psutil.virtual_memory().available / 1e9
if _available_gb < 3.0:
    print(f"⚠️  Low RAM warning: only {_available_gb:.1f}GB available before loading SAM3 predictor.")

try:
    if _SAM3_NATIVE:
        print("⏳  Loading native SAM3.1 video predictor (Object Multiplex) …")
        ckpt_path, bpe_path = _resolve_sam3_checkpoint()
        if ckpt_path:
            print(f"[VID_PREDICTOR] load_from_HF=False  ckpt={ckpt_path}")
            _build_kwargs = dict(
                checkpoint_path=ckpt_path,
                use_fa3=False,          # FA3 not available on Windows
                use_rope_real=True,     # required when use_fa3=False
                max_num_objects=32,     # Object Multiplex — main reason we use 3.1
            )
            if bpe_path:
                _build_kwargs["bpe_path"] = bpe_path
            VID_PREDICTOR = build_sam3_multiplex_video_predictor(**_build_kwargs)
        else:
            print("[VID_PREDICTOR] checkpoint download failed — falling back to load_from_HF=True")
            VID_PREDICTOR = build_sam3_multiplex_video_predictor(use_fa3=False, use_rope_real=True)
        print(f"✅  VID_PREDICTOR ready.  {_mem()}")
    else:
        print("⚠️  native sam3 not available — video tab disabled")
except Exception as e:
    print(f"❌  VID_PREDICTOR load failed: {e}")

# ── All bf16 autocast contexts are now disabled at construction time ────────
# (see patches to sam3_multiplex_base.py Sam3MultiplexTrackerPredictor and
# Sam3MultiplexPredictorWrapper). No runtime context cleanup needed.
if VID_PREDICTOR is not None:
    VID_PREDICTOR.model.to(torch.float32)
    # The tracker wrapper holds the inner VideoTrackingMultiplexDemo as .model;
    # it may have been built in bf16 inside model_builder. Cast everything.
    try:
        VID_PREDICTOR.model.tracker.model.to(torch.float32)
    except Exception as _e:
        print(f"[dtype] tracker.model cast: {_e}")
    torch.cuda.empty_cache()
    print(f"[dtype] VID_PREDICTOR running float32  {_mem()}")

    # ── VRAM→RAM offload: move maskmem outputs to CPU as frames are processed ─
    # Stops VRAM growing ~400MB/10frames by offloading frame outputs to RAM.
    # Your 31.5GB DDR5 can hold 433 frames × 32 objects of maskmem features.
    # Quality: zero impact — features are moved back to GPU on demand via
    # .cuda(non_blocking=True) in _prepare_memory_conditioned_features.
    # Risk: if RAM fills up Windows will page to disk and slow to a crawl.
    # Revert by setting offload_output_to_cpu_for_eval = False.
    try:
        inner = VID_PREDICTOR.model.tracker.model.tracker
        inner.offload_output_to_cpu_for_eval = True
        print(f"[vram-offload] offload_output_to_cpu_for_eval=True on {type(inner).__name__}")
    except Exception as _oe:
        print(f"[vram-offload] failed: {_oe}")

# ── Blanket dtype fix: hook every Conv2d+Linear+LayerNorm in the video predictor
if VID_PREDICTOR is not None:
    # Audit: count bf16 modules before hooking so we can verify the fix is needed
    try:
        _bf16_modules = [
            (type(m).__name__, name)
            for name, m in VID_PREDICTOR.model.named_modules()
            if hasattr(m, 'weight') and m.weight is not None
            and m.weight.dtype == torch.bfloat16
        ]
        print(f"[dtype-audit] {len(_bf16_modules)} bf16-weight modules found before hook registration")
        if _bf16_modules:
            # Show first 5 as a sample so we know exactly which layers need fixing
            for _mtype, _mname in _bf16_modules[:5]:
                print(f"  bf16 module: {_mtype}  path={_mname}")
            if len(_bf16_modules) > 5:
                print(f"  ... and {len(_bf16_modules) - 5} more")
        else:
            print("[dtype-audit] ✅ No bf16-weight modules — model is already fully float32")
    except Exception as _audit_e:
        print(f"[dtype-audit] audit failed (non-fatal): {_audit_e}")

    def _dtype_hook(module, args):
        x = args[0]
        if not isinstance(x, torch.Tensor): return
        # For LayerNorm: cast input to match weight dtype
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None and x.dtype != module.weight.dtype:
                print(f"[dtype-hook] LayerNorm cast {x.dtype} → {module.weight.dtype}  "
                      f"(module={type(module).__name__})")
                return (x.to(module.weight.dtype),) + args[1:]
        # For Conv2d/Linear: cast input to match weight dtype
        elif hasattr(module, 'weight') and module.weight is not None \
                and x.dtype != module.weight.dtype:
            return (x.to(module.weight.dtype),) + args[1:]

    # Register hooks — wrapped in try/except so a bad module never kills startup
    _n_hooks = 0
    _n_hook_errors = 0
    for _m in VID_PREDICTOR.model.modules():
        if isinstance(_m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.LayerNorm)):
            try:
                _m.register_forward_pre_hook(_dtype_hook)
                _n_hooks += 1
            except Exception as _he:
                _n_hook_errors += 1
                print(f"[dtype-hooks] hook registration failed on {type(_m).__name__}: {_he}")
    print(f"[dtype-hooks] ✅ registered on {_n_hooks} modules "
          f"(Conv2d+Linear+LayerNorm)  errors={_n_hook_errors}")

# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILS
# ═════════════════════════════════════════════════════════════════════════════
def apply_mask_overlay(base_image, mask_data, opacity: float = 0.5):
    """Rainbow per-instance overlay. mask_data: (H,W) or (N,H,W)."""
    if isinstance(base_image, np.ndarray):
        base_image = Image.fromarray(base_image)
    base_image = base_image.convert("RGBA")
    if mask_data is None or (isinstance(mask_data, (list, tuple)) and len(mask_data) == 0):
        return base_image.convert("RGB")
    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.detach().cpu().numpy()
    mask_data = np.asarray(mask_data)
    if mask_data.ndim == 4: mask_data = mask_data[0]
    masks = [mask_data] if mask_data.ndim == 2 else [mask_data[i] for i in range(mask_data.shape[0])]
    if not masks: return base_image.convert("RGB")
    try:
        cmap = matplotlib.colormaps["rainbow"].resampled(max(len(masks), 1))
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("rainbow").resampled(max(len(masks), 1))
    rgb_colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(len(masks))]
    composite  = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    for i, m in enumerate(masks):
        m   = (m > 0).astype(np.uint8)
        bmp = Image.fromarray((m * 255).astype(np.uint8))
        if bmp.size != base_image.size:
            bmp = bmp.resize(base_image.size, resample=Image.NEAREST)
        fill  = Image.new("RGBA", base_image.size, rgb_colors[i] + (0,))
        alpha = bmp.point(lambda v: int(v * opacity) if v > 0 else 0)
        fill.putalpha(alpha)
        composite = Image.alpha_composite(composite, fill)
    return Image.alpha_composite(base_image, composite).convert("RGB")


def draw_points_on_image(image, points, modes=None):
    if isinstance(image, np.ndarray): image = Image.fromarray(image)
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    if modes is None: modes = ["add"] * len(points)
    for (x, y), mode in zip(points, modes):
        r       = 8
        fill    = (80, 255, 120) if mode != "erase" else (255, 80, 80)
        outline = (0, 0, 0)      if mode != "erase" else (255, 255, 255)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=3)
    return img


def _bw_preview(mask_np) -> Image.Image | None:
    if mask_np is None: return None
    return Image.fromarray((np.asarray(mask_np) > 0).astype(np.uint8) * 255)


def _normalize_prompt(text_query, default: str = DEFAULT_PROMPT) -> str:
    if not text_query: return default
    s = str(text_query).strip()
    return s if s else default


def remove_small_regions(mask_np: np.ndarray, min_px: int) -> np.ndarray:
    if mask_np is None or min_px <= 0: return mask_np
    m = (mask_np > 0).astype(np.uint8)
    if m.sum() == 0: return m
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    clean = np.zeros_like(m)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == i] = 1
    return clean


def erase_connected_region(mask: np.ndarray, x: int, y: int):
    yi, xi = int(y), int(x)
    if yi >= mask.shape[0] or xi >= mask.shape[1] or yi < 0 or xi < 0:
        return mask, "⚠️ Click out of bounds."
    if mask[yi, xi] == 0:
        return mask, "⚠️ Clicked on background — nothing to erase here."
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    cid      = int(labels[yi, xi])
    new_mask = mask.copy()
    new_mask[labels == cid] = 0
    n_px = int(stats[cid, cv2.CC_STAT_AREA])
    return new_mask.astype(np.uint8), f"🧹 Erased region ({n_px} px) at ({x},{y})."


def union_instance_masks(results) -> np.ndarray | None:
    if results is None or "masks" not in results: return None
    masks = results["masks"]
    if isinstance(masks, torch.Tensor): masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4: masks = masks[0]
    if masks.ndim == 3: return np.any(masks > 0, axis=0).astype(np.uint8)
    if masks.ndim == 2: return (masks > 0).astype(np.uint8)
    return None


def _parse_native_outputs(outputs, vid_h: int, vid_w: int):
    results = []
    if outputs is None:
        return results

    def _add(obj_id, mask):
        if mask is None: return
        if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).squeeze()
        if mask.ndim != 2: return
        if not mask.any(): return
        if mask.shape != (vid_h, vid_w):
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (vid_w, vid_h), Image.NEAREST))
        results.append((int(obj_id), (mask > 0).astype(np.uint8)))

    try:
        if isinstance(outputs, dict) and "out_obj_ids" in outputs:
            ids   = outputs["out_obj_ids"]
            masks = outputs["out_binary_masks"]
            ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            for i, obj_id in enumerate(ids_list):
                _add(obj_id, masks[i])
            if results:
                return results

        if isinstance(outputs, dict) and outputs and all(
            isinstance(k, (int, np.integer)) for k in outputs.keys()
        ):
            for obj_id, mask in outputs.items():
                _add(obj_id, mask)
            if results:
                return results

        fmt = prepare_masks_for_visualization({0: outputs})
        for obj_id, mask in fmt.get(0, {}).items():
            _add(obj_id, mask)

    except Exception as e:
        print(f"[_parse_native_outputs] failed: {e}")
    return results


def _render_native_outputs(frame_pil: Image.Image, obj_masks: list) -> Image.Image:
    if not obj_masks:
        return frame_pil
    stack = np.stack([m for _, m in obj_masks], axis=0)
    return apply_mask_overlay(frame_pil, stack, opacity=0.55)


def _build_label_map(obj_masks: list, shape_hw: tuple) -> np.ndarray:
    label_map = np.zeros(shape_hw, dtype=np.uint8)
    for obj_id, mask in obj_masks:
        capped = min(int(obj_id), 255)
        label_map[(mask > 0) & (label_map == 0)] = capped
    return label_map


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — image segmentation
# ═════════════════════════════════════════════════════════════════════════════
def run_image_segmentation(source_img, text_query, conf_thresh: float = 0.5):
    print(f"[run_image_segmentation] START prompt='{text_query}' {_mem()}")
    _ensure_img_model()
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Image model not loaded.")
    if source_img is None:
        raise gr.Error("Please upload an image.")
    text_query = _normalize_prompt(text_query)
    pil    = source_img.convert("RGB")
    inputs = IMG_PROCESSOR(images=pil, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        out = IMG_MODEL(**inputs)
    res = IMG_PROCESSOR.post_process_instance_segmentation(
        out, threshold=conf_thresh, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    raw_masks  = res.get("masks", None)
    raw_scores = res.get("scores", None)
    if raw_masks is None:
        return (pil, [])
    raw_masks  = raw_masks.cpu().numpy()
    raw_scores = raw_scores.cpu().numpy() if raw_scores is not None else np.ones(len(raw_masks))
    annotations = [(m, f"{text_query} ({s:.2f})") for m, s in zip(raw_masks, raw_scores)]
    print(f"[run_image_segmentation] END n={len(annotations)} {_mem()}")
    return (pil, annotations)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — image mode
# ═════════════════════════════════════════════════════════════════════════════
def run_pcs_dual(source_img, text_query, conf_thresh: float = 0.5):
    print(f"[run_pcs_dual] START prompt='{text_query}' conf={conf_thresh} {_mem()}")
    _ensure_img_model()
    if IMG_MODEL is None or IMG_PROCESSOR is None:
        raise gr.Error("Image model not loaded.")
    if source_img is None:
        raise gr.Error("No image loaded.")
    text_query = _normalize_prompt(text_query)
    pil    = source_img.convert("RGB")
    inputs = IMG_PROCESSOR(images=pil, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        out = IMG_MODEL(**inputs)
    res = IMG_PROCESSOR.post_process_instance_segmentation(
        out, threshold=conf_thresh, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    binary    = union_instance_masks(res)
    if binary is None:
        binary = np.zeros((pil.size[1], pil.size[0]), dtype=np.uint8)
    raw_masks = res.get("masks", None)
    if raw_masks is not None:
        raw_masks_np = raw_masks.cpu().numpy()
        colored = apply_mask_overlay(pil, raw_masks_np, opacity=0.55)
        n_inst  = len(raw_masks_np)
    else:
        colored = apply_mask_overlay(pil, binary, opacity=0.55)
        raw_masks_np = None
        n_inst  = 0
    print(f"[run_pcs_dual] END n_instances={n_inst} {_mem()}")
    return pil, colored, binary, raw_masks_np


def tracker_single_click_mask(image_pil: Image.Image, x: int, y: int):
    print(f"[tracker_single_click_mask] click=({x},{y}) {_mem()}")
    if TRK_MODEL is None or TRK_PROCESSOR is None:
        raise gr.Error("Tracker model not loaded.")
    if image_pil is None: return None
    _trk_to_gpu()
    try:
        inputs = TRK_PROCESSOR(
            images=image_pil,
            input_points=[[[[int(x), int(y)]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = TRK_MODEL(**inputs, multimask_output=True)
        masks = TRK_PROCESSOR.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=True,
        )[0]
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            try:
                best = int(torch.argmax(outputs.iou_scores.detach().cpu()[0, 0]).item())
            except Exception:
                best = 0
        else:
            best = 0
        if isinstance(masks, torch.Tensor): masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if   masks.ndim == 4: click_mask = masks[0, best]
        elif masks.ndim == 3: click_mask = masks[0]
        else: return None
        print(f"[tracker_single_click_mask] END {_mem()}")
        return (click_mask > 0).astype(np.uint8)
    finally:
        _trk_to_cpu()


def _push_img_history(hist, mask, pts, modes):
    hist = list(hist or [])
    hist.append((mask.copy(), list(pts), list(modes)))
    return hist[-MAX_IMG_HIST:]


def mask_editor_click_image(evt: gr.SelectData, st_img, st_mask, st_mode, st_hist, st_pts, st_modes):
    if st_img is None or st_mask is None:
        return None, None, st_hist, st_pts, st_modes, "⚠️ Run Auto Detect (PCS) first."
    x, y = evt.index
    mode = st_mode or "add"
    st_hist = _push_img_history(st_hist, st_mask, st_pts, st_modes)
    if mode == "erase":
        new_mask, status = erase_connected_region(st_mask, x, y)
        st_pts   = list(st_pts or []) + [[int(x), int(y)]]
        st_modes = list(st_modes or []) + ["erase"]
        ov = apply_mask_overlay(st_img, new_mask, opacity=0.5)
        ov = draw_points_on_image(ov, st_pts, st_modes)
        return ov, new_mask, st_hist, st_pts, st_modes, status
    click_mask = tracker_single_click_mask(st_img, x, y)
    if click_mask is None:
        ov = apply_mask_overlay(st_img, st_mask, opacity=0.5)
        ov = draw_points_on_image(ov, st_pts, st_modes)
        return ov, st_mask, st_hist, st_pts, st_modes, "⚠️ Tracker returned no mask."
    if click_mask.shape != st_mask.shape:
        click_pil = Image.fromarray((click_mask * 255).astype(np.uint8))
        click_pil = click_pil.resize((st_mask.shape[1], st_mask.shape[0]), resample=Image.NEAREST)
        click_mask = (np.array(click_pil) > 127).astype(np.uint8)
    new_mask = (st_mask | click_mask).astype(np.uint8)
    st_pts   = list(st_pts or []) + [[int(x), int(y)]]
    st_modes = list(st_modes or []) + ["add"]
    ov = apply_mask_overlay(st_img, new_mask, opacity=0.5)
    ov = draw_points_on_image(ov, st_pts, st_modes)
    return ov, new_mask, st_hist, st_pts, st_modes, f"✅ add click at ({x},{y})."


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — video mode helpers
# ═════════════════════════════════════════════════════════════════════════════
def vid_start_session(video_path_str: str, prompt: str, conf: float,
                      target_fps: float, frame_limit: int):
    print(f"[vid_start_session] START path={video_path_str} {_mem()}")
    if VID_PREDICTOR is None:
        return None, None, [], None, [], 0, "❌ VID_PREDICTOR not loaded.", 0.0, 0, 0

    try:
        VID_PREDICTOR.model.score_threshold_detection = float(conf)
        VID_PREDICTOR.model.new_det_thresh = float(conf)
        print(f"[vid_start_session] detection thresholds → {conf:.2f}")
    except Exception as _e:
        print(f"[vid_start_session] could not set detection thresholds: {_e}")

    cap     = cv2.VideoCapture(video_path_str)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step    = max(1, round(fps / target_fps)) if target_fps > 0 else 1
    out_fps = fps / step
    ret, frame0_bgr = cap.read()
    cap.release()
    print(f"[vid_start_session] src_fps={fps} step={step} out_fps={out_fps} w={vid_w} h={vid_h}")

    if not ret:
        return None, None, [], None, [], 0, "❌ Could not read frame 0.", out_fps, vid_h, vid_w

    frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)

    _total_src_frames = int(cv2.VideoCapture(video_path_str).get(cv2.CAP_PROP_FRAME_COUNT))
    _total_at_fps     = max(1, _total_src_frames // max(step, 1))
    N_KEYFRAME_FRAMES = min(50, _total_at_fps)
    temp_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    frames_for_session = [frame0_rgb]
    cv2.imwrite(str(Path(temp_dir) / "00000.jpg"), frame0_bgr)

    cap2 = cv2.VideoCapture(video_path_str)
    raw_idx2 = 0
    while cap2.isOpened() and len(frames_for_session) < N_KEYFRAME_FRAMES:
        if raw_idx2 % step == 0 and raw_idx2 > 0:
            ret2, frm = cap2.read()
            if not ret2: break
            fi = len(frames_for_session)
            cv2.imwrite(str(Path(temp_dir) / f"{fi:05d}.jpg"), frm)
            frames_for_session.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        else:
            ret2 = cap2.grab()
            if not ret2: break
        raw_idx2 += 1
    cap2.release()
    print(f"[vid_start_session] keyframe window: {len(frames_for_session)} frames → {temp_dir}")

    try:
        resp       = VID_PREDICTOR.handle_request(dict(type="start_session", resource_path=temp_dir))
        session_id = resp["session_id"]
        print(f"[vid_start_session] session_id={session_id} ({len(frames_for_session)} frames)  {_mem()}")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, None, [], None, [], 0, f"❌ start_session failed: {e}", out_fps, vid_h, vid_w

    prompt = _normalize_prompt(prompt)
    try:
        resp      = VID_PREDICTOR.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=0, text=prompt,
        ))
        out0      = resp.get("outputs", {}) if isinstance(resp, dict) else {}
        obj_masks = _parse_native_outputs(out0, vid_h, vid_w)
        n_obj     = len(obj_masks)
        frame0_pil = Image.fromarray(frame0_rgb)
        overlay    = _render_native_outputs(frame0_pil, obj_masks)
        print(f"[vid_start_session] PCS detected {n_obj} objects  {_mem()}")
        status = (f"✅ PCS done — {n_obj} bamboo instance(s) on frame 0.  "
                  f"(full video will be extracted on Propagate)")
    except Exception as e:
        import traceback; traceback.print_exc()
        status     = f"⚠️ Session started but PCS failed: {e}"
        overlay    = Image.fromarray(frame0_rgb)
        obj_masks  = []
        n_obj      = 0

    return session_id, temp_dir, [frame0_rgb], overlay, obj_masks, n_obj, status, out_fps, vid_h, vid_w


def vid_expand_session_for_propagation(
    session_id: str, temp_dir: str, video_path_str: str,
    target_fps: float, frame_limit: int, prompt: str,
    prompt_history: list, vid_h: int, vid_w: int,
):
    print(f"[vid_expand_session] START  {_mem()}")

    try:
        VID_PREDICTOR.model.score_threshold_detection = float(
            getattr(VID_PREDICTOR.model, "score_threshold_detection", 0.5)
        )
    except Exception:
        pass

    cap     = cv2.VideoCapture(video_path_str)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, round(fps / target_fps)) if target_fps > 0 else 1
    if frame_limit > 0:
        total = min(total, frame_limit * step)

    frames  = []
    raw_idx = 0
    while cap.isOpened():
        if frame_limit > 0 and len(frames) >= frame_limit:
            break
        if raw_idx % step == 0:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            fname = Path(temp_dir) / f"{len(frames)-1:05d}.jpg"
            if not fname.exists():
                cv2.imwrite(str(fname), frame)
        else:
            ret = cap.grab()
            if not ret: break
        raw_idx += 1
    cap.release()
    n = len(frames)
    print(f"[vid_expand_session] extracted {n} frames → {temp_dir}  {_mem()}")

    if n == 0:
        return [], session_id, "❌ No frames extracted during expansion."

    try:
        resp = VID_PREDICTOR.handle_request(dict(
            type="start_session", resource_path=temp_dir))
        new_sid = resp["session_id"]
        print(f"[vid_expand_session] new session {new_sid} on {n} frames  {_mem()}")
        VID_PREDICTOR.handle_request(dict(
            type="add_prompt", session_id=new_sid, frame_index=0,
            text=_normalize_prompt(prompt),
        ))
        skipped = 0
        for entry in (prompt_history or []):
            reqs = entry.get("reqs", [entry]) if isinstance(entry, dict) and "reqs" in entry else [entry]
            for req in reqs:
                req_copy = {k: v for k, v in req.items() if k != "masks"}
                req_copy["session_id"] = new_sid
                try:
                    VID_PREDICTOR.handle_request(req_copy)
                except Exception as replay_e:
                    skipped += 1
                    print(f"[vid_expand_session] skipped replay {req_copy.get('type')} "
                          f"obj_id={req_copy.get('obj_id','?')}: {replay_e}")
        if skipped:
            print(f"[vid_expand_session] {skipped} replay entries skipped (obj_id mismatch)")
        print(f"[vid_expand_session] replayed ({len(prompt_history or [])} prompts, {skipped} skipped)  {_mem()}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return frames, session_id, f"⚠️ Session expansion failed: {e}"

    return frames, new_sid, f"✅ {n} frames ready for propagation."


def vid_add_point(session_id: str, frames: list, vid_h: int, vid_w: int,
                  x: int, y: int, label: int, obj_id: int | None):
    if VID_PREDICTOR is None or session_id is None:
        return None, [], "❌ No active session."
    try:
        W, H = Image.fromarray(frames[0]).size
        rel_x = x / W
        rel_y = y / H
        points_tensor = torch.tensor([[rel_x, rel_y]], dtype=torch.float32)
        labels_tensor = torch.tensor([label], dtype=torch.int32)
        if obj_id is None:
            try:
                state = VID_PREDICTOR._ALL_INFERENCE_STATES.get(session_id, {})
                inf   = state.get("state", {})
                meta  = inf.get("tracker_metadata", {}) if isinstance(inf, dict) else {}
                used  = meta.get("max_obj_id", -1) if isinstance(meta, dict) else -1
                obj_id = int(used) + 1 if used is not None and used >= 0 else 1000
            except Exception:
                obj_id = 1000
        req = dict(
            type="add_prompt", session_id=session_id, frame_index=0,
            points=points_tensor, point_labels=labels_tensor, obj_id=obj_id,
        )
        resp      = VID_PREDICTOR.handle_request(req)
        out0      = resp.get("outputs", {})
        obj_masks = _parse_native_outputs(out0, vid_h, vid_w)
        frame0_pil = Image.fromarray(frames[0])
        overlay    = _render_native_outputs(frame0_pil, obj_masks)
        lbl_str    = "+" if label == 1 else "-"
        status = (f"✅ Point {lbl_str} at ({x},{y}) — {len(obj_masks)} objects visible"
                  if obj_masks else
                  f"✅ Point {lbl_str} registered at ({x},{y}) — will appear after Propagate")
        print(f"[vid_add_point] {status}")
    except Exception as e:
        import traceback; traceback.print_exc()
        overlay   = Image.fromarray(frames[0])
        obj_masks = []
        status    = f"❌ add_prompt failed — see terminal"
    return overlay, obj_masks, status


def vid_remove_object(session_id: str, frames: list, vid_h: int, vid_w: int, obj_id: int):
    if VID_PREDICTOR is None or session_id is None:
        return None, [], "❌ No active session."
    try:
        VID_PREDICTOR.handle_request(dict(type="remove_object", session_id=session_id, obj_id=obj_id))
        resp = VID_PREDICTOR.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=0, text="bamboo",
        ))
        out0      = resp.get("outputs", {})
        obj_masks = _parse_native_outputs(out0, vid_h, vid_w)
        frame0_pil = Image.fromarray(frames[0])
        overlay    = _render_native_outputs(frame0_pil, obj_masks)
        status     = f"🗑️ Removed object {obj_id} — {len(obj_masks)} remaining"
    except Exception as e:
        overlay   = Image.fromarray(frames[0])
        obj_masks = []
        status    = f"❌ remove_object failed: {e}"
    return overlay, obj_masks, status


def vid_reset_to_pcs(session_id: str, frames: list, vid_h: int, vid_w: int, prompt: str):
    if VID_PREDICTOR is None or session_id is None:
        return None, [], "❌ No active session."
    try:
        VID_PREDICTOR.handle_request(dict(type="reset_session", session_id=session_id))
        prompt = _normalize_prompt(prompt)
        resp  = VID_PREDICTOR.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=0, text=prompt,
        ))
        out0      = resp.get("outputs", {})
        obj_masks = _parse_native_outputs(out0, vid_h, vid_w)
        frame0_pil = Image.fromarray(frames[0])
        overlay    = _render_native_outputs(frame0_pil, obj_masks)
        status     = f"🔄 Reset to PCS — {len(obj_masks)} objects"
    except Exception as e:
        overlay   = Image.fromarray(frames[0])
        obj_masks = []
        status    = f"❌ reset failed: {e}"
    return overlay, obj_masks, status


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — save mask
# ═════════════════════════════════════════════════════════════════════════════
def save_mask_with_meta(
    mask_np, inst_masks, pil_img, source_path_str, out_root_str,
    prompt, conf, clicks, modes,
    is_keyframe: bool = False, frame_idx: int | None = None,
):
    print(f"[save_mask_with_meta] {_mem()}")
    if mask_np is None:
        return "❌ No mask to save."
    out_root_str = (out_root_str or "").strip()
    if not out_root_str:
        return "❌ Please set an output folder."
    src    = Path(source_path_str) if source_path_str else None
    stem   = src.stem if src else ("mask_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    fsuffix = f"_frame{frame_idx:06d}" if frame_idx is not None else ""
    out_dir = Path(out_root_str) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fname     = f"mask{fsuffix}"
    png_path  = out_dir / f"{fname}.png"
    npy_path  = out_dir / f"{fname}.npy"
    json_path = out_dir / f"{fname}.json"
    m = (mask_np > 0).astype(np.uint8)

    if inst_masks is not None and len(inst_masks) > 0:
        if isinstance(inst_masks, list) and isinstance(inst_masks[0], tuple):
            label_map = _build_label_map(inst_masks, m.shape)
        else:
            inst_np   = np.asarray(inst_masks)
            if inst_np.ndim == 4: inst_np = inst_np[0]
            label_map = np.zeros(m.shape, dtype=np.uint8)
            for i, mask in enumerate(inst_np):
                obj_id = min(i + 1, 255)
                label_map[(mask > 0) & (label_map == 0)] = obj_id
        Image.fromarray(label_map).save(png_path)
        np.save(npy_path, label_map)
    else:
        Image.fromarray(m * 255).save(png_path)
        np.save(npy_path, m)

    if inst_masks is not None and pil_img is not None:
        try:
            src_pil = pil_img.convert("RGB")
            if isinstance(inst_masks, list) and isinstance(inst_masks[0], tuple):
                ovl = _render_native_outputs(src_pil, inst_masks)
            else:
                ovl = apply_mask_overlay(src_pil, np.asarray(inst_masks), opacity=0.55)
            ovl.save(out_dir / f"{fname}_overlay.png")
        except Exception as e:
            print(f"[save_mask_with_meta] overlay failed: {e}")

    meta = {
        "source_path": str(src) if src else None,
        "source_stem": stem,
        "is_keyframe": is_keyframe,
        "frame_idx": frame_idx,
        "prompt": str(prompt),
        "confidence_threshold": float(conf),
        "shape_hw": [int(m.shape[0]), int(m.shape[1])],
        "pvs_clicks": [[int(c[0]), int(c[1])] for c in (clicks or [])],
        "pvs_modes": list(modes or []),
        "timestamp": datetime.now().isoformat(),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    return f"💾 Saved → {out_dir}  ({fname}.png / .npy / .json)"


# ═════════════════════════════════════════════════════════════════════════════
# File utilities
# ═════════════════════════════════════════════════════════════════════════════
def list_folder_media(folder_path_str: str) -> list[str]:
    folder = Path((folder_path_str or "").strip())
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in (IMG_EXTS | VID_EXTS)
    )


def load_media_file(folder_path_str: str, filename: str):
    print(f"[load_media_file] '{filename}'  {_mem()}")
    if not folder_path_str or not filename:
        return None, False, None, "❌ Select a folder and a file.", 30.0
    full = Path(folder_path_str.strip()) / filename
    if not full.exists():
        return None, False, None, f"❌ Not found: {full}", 30.0
    ext = full.suffix.lower()
    if ext in VID_EXTS:
        cap = cv2.VideoCapture(str(full))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, False, None, f"❌ Could not read first frame.", 30.0
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return pil, True, str(full), f"🎬 Video loaded.  ({filename})", native_fps
    elif ext in IMG_EXTS:
        try:
            pil = Image.open(full).convert("RGB")
            return pil, False, str(full), f"🖼️ Image loaded.  ({filename})", 30.0
        except Exception as e:
            return None, False, None, f"❌ Could not open image: {e}", 30.0
    return None, False, None, f"❌ Unsupported type: {ext}", 30.0


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — video propagation
# ═════════════════════════════════════════════════════════════════════════════
def run_video_propagation(
    session_id: str, frames: list, temp_dir: str, video_path_str: str,
    out_root_str: str, out_fps: float, min_region_px: int,
    prompt: str = "", prompt_history: list = None,
    target_fps: float = 5.0, frame_limit: int = 0,
):
    print(f"[run_video_propagation] START session={session_id} {_mem()}")
    if VID_PREDICTOR is None:
        raise gr.Error("VID_PREDICTOR not loaded.")
    if not session_id:
        raise gr.Error("No active video session. Run PCS in Tab 2 first.")
    out_root_str = (out_root_str or "").strip()
    if not out_root_str:
        raise gr.Error("Please set an output folder.")

    video_path = Path(video_path_str)
    stem       = video_path.stem
    out_dir    = Path(out_root_str) / stem
    mask_dir   = out_dir / "masks"
    ovl_dir    = out_dir / "overlays"
    meta_dir   = out_dir / "meta"
    for d in [mask_dir, ovl_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n      = len(frames)
    vid_h, vid_w = frames[0].shape[:2]

    if n <= 1:
        yield None, f"⏳ Extracting frames from video …  {_mem()}"
        frames, new_session_id, expand_status = vid_expand_session_for_propagation(
            session_id=session_id, temp_dir=temp_dir, video_path_str=video_path_str,
            target_fps=target_fps, frame_limit=frame_limit, prompt=prompt,
            prompt_history=prompt_history, vid_h=vid_h, vid_w=vid_w,
        )
        session_id = new_session_id
        n = len(frames)
        print(f"[run_video_propagation] expansion: {expand_status}")
        if n == 0:
            yield None, f"❌ Frame expansion failed: {expand_status}"
            return
        yield None, f"{expand_status}  Now propagating {n} frames …"

    print(f"[run_video_propagation] {n} frames {vid_w}x{vid_h} out_fps={out_fps:.1f}")
    yield None, f"⏳ Propagating {n} frames …  {_mem()}"

    # Free the frames list from RAM — we'll read each frame from the temp_dir JPGs instead.
    # At 1 FPS, 433 frames × 1280×720×3 = ~1.1 GB held in RAM during propagation.
    frames_dir = Path(temp_dir)
    n_jpg = len(list(frames_dir.glob("*.jpg")))
    print(f"[run_video_propagation] frames_dir={frames_dir}  jpg_count={n_jpg}  freeing in-RAM frames list")
    frames = None  # allow GC

    tmp_mp4  = tempfile.mktemp(suffix=".mp4")
    real_mp4 = out_dir / "overlay.mp4"
    writer   = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                               max(out_fps, 1.0), (vid_w, vid_h))
    log_lines = []
    t0        = time.time()

    try:
        for response in VID_PREDICTOR.handle_stream_request(dict(
            type="propagate_in_video", session_id=session_id,
        )):
            f_idx     = response.get("frame_index", 0)
            raw_out   = response.get("outputs", {})

            # Per-frame try/except — a dtype mismatch or OOM on one frame
            # must not kill the entire propagation run.
            try:
                obj_masks = _parse_native_outputs(raw_out, vid_h, vid_w)
            except Exception as _pno_e:
                print(f"[run_video_propagation] ⚠️  frame {f_idx} parse failed: {_pno_e}")
                obj_masks = []

            if f_idx % 5 == 0:
                msg = f"⏳ Frame {f_idx}/{n}  {len(obj_masks)} objects  {_mem()}"
                print(f"[run_video_propagation] {msg}")
                yield None, "\n".join(reversed(log_lines[-20:])) + "\n" + msg

            # Periodic VRAM cache flush to prevent fragmentation creep
            if f_idx > 0 and f_idx % 10 == 0:
                torch.cuda.empty_cache()
                print(f"[run_video_propagation] empty_cache @ frame {f_idx}  {_mem()}")

            try:
                # Read frame from disk (JPG in temp_dir) to avoid holding 433 frames in RAM
                jpg_path = Path(temp_dir) / f"{f_idx:05d}.jpg"
                if jpg_path.exists():
                    orig = Image.open(jpg_path).convert("RGB")
                    print(f"[run_video_propagation] frame {f_idx} read from disk") if f_idx == 0 else None
                else:
                    # Fallback: if JPG missing, write a blank frame and warn
                    print(f"[run_video_propagation] ⚠️  frame {f_idx} JPG missing at {jpg_path}")
                    orig = Image.new("RGB", (vid_w, vid_h), (0, 0, 0))

                if obj_masks:
                    label_map = _build_label_map(obj_masks, (vid_h, vid_w))
                    if min_region_px > 0:
                        obj_masks_clean = [(oid, remove_small_regions(m, min_region_px)) for oid, m in obj_masks]
                        label_map = _build_label_map(obj_masks_clean, (vid_h, vid_w))
                    overlay_frame = _render_native_outputs(orig, obj_masks)
                else:
                    label_map     = np.zeros((vid_h, vid_w), dtype=np.uint8)
                    overlay_frame = orig

                fn = f"{f_idx:06d}"
                Image.fromarray(label_map).save(mask_dir / f"{fn}.png")
                np.save(mask_dir / f"{fn}.npy", label_map)
                Image.fromarray(np.array(overlay_frame)).save(ovl_dir / f"{fn}.png")
                (meta_dir / f"{fn}.json").write_text(json.dumps({
                    "frame_idx": f_idx, "n_objects": len(obj_masks),
                    "obj_ids": [int(oid) for oid, _ in obj_masks],
                    "shape_hw": [vid_h, vid_w], "timestamp": datetime.now().isoformat(),
                }, indent=2))
                writer.write(cv2.cvtColor(np.array(overlay_frame), cv2.COLOR_RGB2BGR))
                log_lines.append(f"[{f_idx:05d}/{n}] {len(obj_masks)} obj")
            except Exception as _frame_e:
                print(f"[run_video_propagation] ⚠️  frame {f_idx} save failed: {_frame_e}")
                log_lines.append(f"[{f_idx:05d}/{n}] ❌ {_frame_e}")

    except Exception as e:
        writer.release()
        import traceback; traceback.print_exc()
        yield None, f"❌ Propagation failed: {e}"
        return

    writer.release()
    shutil.copy2(tmp_mp4, real_mp4)
    dt = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps({
        "video_path": str(video_path), "stem": stem, "total_frames": n,
        "output_fps": out_fps, "runtime_s": round(dt, 2),
        "output_dir": str(out_dir), "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log = (f"✅ Done.  {n} frames in {dt:.1f}s\nOutput → {out_dir}\n\n"
           ) + "\n".join(reversed(log_lines[-150:]))
    print(f"[run_video_propagation] END {_mem()}")
    yield tmp_mp4, log


# ═════════════════════════════════════════════════════════════════════════════
# CSS + JS
# ═════════════════════════════════════════════════════════════════════════════
custom_css = """
#col-container  { margin: 0 auto; max-width: 1200px; }
#main-title h1  { font-size: 2.1em !important; }
.tabs > .tab-nav               { position: relative !important; z-index: 20 !important; }
.tabs > .tab-nav button        { pointer-events: auto !important;
                                  position: relative !important; z-index: 20 !important; }
.tabitem                       { position: relative !important; z-index: 1 !important; }
"""

CTRLZ_JS = r"""
<script>
(function(){
  document.addEventListener('keydown', function(e){
    if (!((e.ctrlKey||e.metaKey) && (e.key==='z'||e.key==='Z'))) return;
    e.preventDefault();
    var btn = document.querySelector('#me_undo_btn button');
    if (btn) btn.click();
  }, {capture:true});
})();
</script>
"""

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
with gr.Blocks() as demo:

    with gr.Column(elem_id="col-container"):
        # ── v5 change: updated title ──────────────────────────────────────────
        gr.Markdown("# **SAM3.1 v5: Multi-Object Bamboo Annotation** ⚡ Object Multiplex", elem_id="main-title")
        gr.Markdown(
            "**Tab 1**: quick image PCS test · "
            "**Tab 2**: image PCS+PVS OR video multi-object session · "
            "**Tab 3**: video propagation (appears when video session active)"
        )

    _startup_input_dir  = _get_saved_input_dir()
    _startup_output_dir = _get_saved_output_dir()
    _startup_target_fps  = _load_config().get("target_fps", 5)
    _startup_frame_limit = _load_config().get("frame_limit", 0)

    st_last_input_dir  = gr.State(_startup_input_dir)
    st_last_output_dir = gr.State(_startup_output_dir)

    st_out_dir = gr.Textbox(label="Output Folder (shared)", placeholder="/path/to/output",
                             value=_startup_output_dir)
    def _on_output_dir_change(v):
        v = _norm_path(v)
        _save_config("output_dir", v)
        return v
    st_out_dir.change(fn=_on_output_dir_change, inputs=[st_out_dir], outputs=[st_last_output_dir])

    shared_min_px = gr.Slider(0, 2000, value=_load_config().get("min_px", 50), step=10,
                               label="Min Region Size — output mask filter (0 = off)")
    st_min_px = gr.State(_load_config().get("min_px", 50))
    def _save_min_px(v):
        _save_config("min_px", int(v))
        return int(v)
    shared_min_px.release(fn=_save_min_px, inputs=[shared_min_px], outputs=[st_min_px])

    st_session_id    = gr.State(None)
    st_temp_dir      = gr.State(None)
    st_vid_frames    = gr.State([])
    st_vid_path      = gr.State(None)
    st_vid_h         = gr.State(0)
    st_vid_w         = gr.State(0)
    st_out_fps       = gr.State(5.0)
    st_vid_obj_masks = gr.State([])
    st_prompt_hist   = gr.State([])
    st_pcs_masks     = gr.State([])
    st_pending_pts   = gr.State([])
    st_last_file     = gr.State(None)
    st_shared_mask   = gr.State(None)

    with gr.Tabs(selected=1):

        with gr.Tab("Image Segmentation", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_img    = gr.Image(label="Upload Image", type="pil", height=350)
                    t1_prompt = gr.Textbox(label="Text Prompt", value=DEFAULT_PROMPT)
                    with gr.Accordion("Advanced", open=False):
                        t1_conf = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                    t1_btn = gr.Button("Segment Image", variant="primary")
                with gr.Column(scale=2):
                    t1_result = gr.AnnotatedImage(label="Segmented Result (per-instance colours)", height=430)
            t1_btn.click(fn=run_image_segmentation, inputs=[t1_img, t1_prompt, t1_conf], outputs=[t1_result])

        with gr.Tab("Mask Editor", id=1):
            # declare tab_video_prop early so it can be referenced in me_load_btn.click outputs
            tab_video_prop = gr.Tab("Video Propagator", visible=False, id=2)
            gr.Markdown(f"""
**Image mode:** Load image → Auto Detect (PCS) → click to add/erase regions (PVS) → Save.

**Video mode:** Load video → Auto Detect (PCS) → click to refine per-object → Propagate (Tab 3).
- PCS detects **all bamboo instances simultaneously** as separate tracked objects.
- Click in **Add mode** to add missed objects, **Erase mode** to remove false positives.
- **Ctrl+Z** = Undo (up to {MAX_VID_HIST} steps). **Reset to PCS** wipes all manual refinements.
""")
            gr.HTML(CTRLZ_JS)

            with gr.Row():
                me_folder  = gr.Textbox(label="Input Folder", value=_startup_input_dir, scale=4)
                me_refresh = gr.Button("🔄 Refresh", scale=0, min_width=90)
            me_file_dd  = gr.Dropdown(label="Select File",
                                       choices=list_folder_media(_startup_input_dir),
                                       value=_load_config().get("last_file", None),
                                       interactive=True, allow_custom_value=True)
            me_load_btn    = gr.Button("Load Selected File", variant="secondary")
            me_load_status = gr.Textbox(label="Load Status", interactive=False, lines=1)
            def _on_input_dir_change(v):
                v = _norm_path(v)
                _save_config("input_dir", v)
                return v
            me_folder.change(fn=_on_input_dir_change, inputs=[me_folder], outputs=[st_last_input_dir])

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    me_loaded_preview = gr.Image(type="pil", label="Loaded Frame / Image",
                                                 height=280, interactive=False)
                    me_prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
                    me_conf   = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Confidence Threshold")
                    me_mode   = gr.Radio(["add", "erase"], value="add",
                                         label="Click Mode  (add=new object / erase=remove object)")
                    with gr.Row():
                        me_btn_auto  = gr.Button("🔍 Auto Detect (PCS)", variant="primary")
                        me_btn_reset = gr.Button("Reset to PCS", variant="secondary")
                    with gr.Row():
                        me_btn_undo  = gr.Button("↩ Undo (Ctrl+Z)", elem_id="me_undo_btn")
                        me_btn_clrpt = gr.Button("Clear Markers")
                    me_btn_submit = gr.Button("✅ Submit Points", variant="primary",
                                              visible=False, elem_id="me_submit_btn")
                    me_btn_pop    = gr.Button("⌫ Pop Last Point", variant="secondary",
                                              visible=False, elem_id="me_pop_btn")
                    gr.Markdown("**Save / Propagate**")
                    me_btn_save = gr.Button("💾 Save Frame Mask", variant="primary")
                    me_status   = gr.Textbox(label="Status", interactive=False, lines=3)

                with gr.Column(scale=6):
                    me_colored     = gr.Image(type="pil", label="PCS Result — multi-colour per instance",
                                              height=280, interactive=False)
                    me_pvs_overlay = gr.Image(type="pil", label="Session Overlay — click to refine",
                                              height=360, interactive=True)
                    me_mask_bw     = gr.Image(type="pil", label="Binary Mask Preview",
                                              height=200, interactive=False)

            st_me_img   = gr.State(None)
            st_me_auto  = gr.State(None)
            st_me_mask  = gr.State(None)
            st_me_inst  = gr.State(None)
            st_me_hist  = gr.State([])
            st_me_pts   = gr.State([])
            st_me_modes = gr.State([])
            st_is_video = gr.State(False)

            vp_keyframe_preview = gr.Image(type="pil", visible=False,
                                           label="Frame 0 Detection Preview", height=200, interactive=False)
            vp_target_fps = gr.Slider(0, 30, value=_startup_target_fps, step=1, visible=False, label="Target FPS (0 = source FPS)")
            vp_framelim   = gr.Slider(0, 2000, value=_startup_frame_limit, step=10, visible=False, label="Frame Limit (0 = all frames)")

            def _refresh(folder, last_file):
                folder = _norm_path(folder or "")
                if folder: _save_config("input_dir", folder)
                files = list_folder_media(folder)
                if not files:
                    return gr.update(choices=[], value=None), "⚠️ No files found."
                value = last_file if last_file in files else files[0]
                return gr.update(choices=files, value=value), f"Found {len(files)} file(s)."
            me_refresh.click(fn=_refresh, inputs=[me_folder, st_last_file], outputs=[me_file_dd, me_load_status])

            def _load(folder, filename, out_dir):
                print(f"[_load] called: folder={folder!r} filename={filename!r} out_dir={out_dir!r}")
                # Guard: don't run if no file is selected yet
                if not filename:
                    print("[_load] no filename — returning early")
                    return (None, None, False, None, "", gr.update(visible=False),
                            None, None, None, [], [], [],
                            None, None, None, None, None, gr.update(),
                            None, [], None)
                folder  = _norm_path(folder or "")
                out_dir = _norm_path(out_dir or "")
                pil, is_vid, full_path, status, native_fps = load_media_file(folder, filename)
                print(f"[_load] is_vid={is_vid} full_path={full_path} native_fps={native_fps}")
                tab3_vis   = gr.update(visible=is_vid)
                # Use saved target_fps from config, capped to native_fps
                saved_fps = _load_config().get("target_fps", 5)
                fps_val   = min(int(saved_fps), int(native_fps)) if is_vid else 5
                fps_update = gr.update(maximum=int(native_fps), value=fps_val,
                                       label=f"Target FPS (0 = keep source {native_fps:.1f}fps)"
                                       ) if is_vid else gr.update()
                print(f"[_load] fps_val={fps_val} (saved={saved_fps}, native={native_fps})")
                existing_mask = existing_colored = existing_pvs_ov = existing_bw = None
                if full_path and out_dir and out_dir.strip() and pil is not None:
                    stem  = Path(full_path).stem
                    # Video: check for both propagation masks/frame output AND keyframe save
                    candidates = []
                    if is_vid:
                        candidates = [
                            Path(out_dir.strip()) / stem / "masks" / "000000.png",      # propagation output
                            Path(out_dir.strip()) / stem / "mask_frame000000.png",      # keyframe save (video)
                            Path(out_dir.strip()) / stem / "overlays" / "000000.png",   # overlay direct
                        ]
                    else:
                        candidates = [Path(out_dir.strip()) / stem / "mask.png"]
                    print(f"[_load] checking cache candidates: {[str(c) for c in candidates]}")
                    mask_path = next((c for c in candidates if c.exists()), None)
                    if mask_path:
                        print(f"[_load] found cached mask: {mask_path}")
                        try:
                            m = np.array(Image.open(mask_path).convert("L"))
                            existing_mask = (m > 0).astype(np.uint8)
                            # look for overlay: same stem + _overlay.png, or in overlays/ folder
                            ovl_candidates = [
                                mask_path.parent / f"{mask_path.stem}_overlay.png",
                                Path(out_dir.strip()) / stem / "overlays" / "000000.png",
                            ]
                            ovl_path = next((o for o in ovl_candidates if o.exists()), None)
                            if ovl_path:
                                print(f"[_load] found overlay: {ovl_path}")
                                existing_colored = existing_pvs_ov = Image.open(ovl_path).convert("RGB")
                            else:
                                print(f"[_load] no overlay found, generating from mask")
                                existing_colored = apply_mask_overlay(pil, existing_mask, opacity=0.55)
                                existing_pvs_ov  = apply_mask_overlay(pil, existing_mask, opacity=0.5)
                            existing_bw = _bw_preview(existing_mask)
                            status += f"  ✅ Cached mask loaded ({mask_path.name}) — run Auto Detect (PCS) to start a new session."
                        except Exception as e:
                            print(f"[_load] mask load error: {e}")
                            status += f"  ⚠️ Mask load failed: {e}"
                    else:
                        print(f"[_load] no cached mask found for stem={stem!r} in {out_dir!r}")
                print(f"[_load] returning: existing_mask={'yes' if existing_mask is not None else 'no'}")
                # Use overlay as the loaded preview when cache exists, else raw frame
                preview_img = existing_pvs_ov if existing_pvs_ov is not None else pil
                print(f"[_load] preview_img source={'cached overlay' if existing_pvs_ov is not None else 'raw frame'}")
                return (preview_img, pil, is_vid, full_path, status, tab3_vis,
                        existing_mask, existing_mask, existing_mask,
                        [], [], [],
                        existing_colored, existing_pvs_ov, existing_bw,
                        existing_bw, filename, fps_update,
                        None, [], None)  # always clear session on fresh load

            me_load_btn.click(
                fn=_load, inputs=[me_folder, me_file_dd, st_out_dir],
                outputs=[
                    me_loaded_preview, st_me_img, st_is_video, st_vid_path, me_load_status,
                    tab_video_prop,
                    st_me_auto, st_me_mask, st_shared_mask,
                    st_me_hist, st_me_pts, st_me_modes,
                    me_colored, me_pvs_overlay, me_mask_bw,
                    vp_keyframe_preview, st_last_file, vp_target_fps,
                    st_session_id, st_vid_frames, st_temp_dir,
                ],
            )
            me_load_btn.click(fn=lambda f: _save_config("last_file", f), inputs=[me_file_dd], outputs=[])

            def _auto_detect(img, is_vid, vid_path, prompt, conf, target_fps, frame_lim, hist, pts, modes):
                if is_vid:
                    if VID_PREDICTOR is None:
                        return (None, None, None, None, [], [], None, [], 0.0, 0, 0, [], None,
                                "❌ VID_PREDICTOR not loaded.", None, None, [], [])
                    session_id, temp_dir, frames, overlay, obj_masks_out, n_obj, status, out_fps, vid_h, vid_w = \
                        vid_start_session(vid_path, prompt, conf, float(target_fps), int(frame_lim))
                    binary = (_build_label_map(obj_masks_out, (vid_h, vid_w)) > 0).astype(np.uint8) if obj_masks_out else None
                    return (overlay, overlay, _bw_preview(binary),
                            session_id, frames, temp_dir, binary, obj_masks_out,
                            out_fps, vid_h, vid_w, [], overlay, status,
                            None, None, obj_masks_out, [])
                else:
                    if img is None:
                        return (None, None, None, None, [], None, None, [], 0.0, 0, 0, [], None,
                                "⚠️ No image loaded — load a file first.", None, None, [], [])
                    pil, colored, binary, inst_masks = run_pcs_dual(img, prompt, conf)
                    pvs_src = inst_masks if inst_masks is not None else binary
                    pvs_ov  = apply_mask_overlay(pil, pvs_src, opacity=0.5)
                    status  = f"✅ PCS done. {0 if inst_masks is None else len(inst_masks)} instances."
                    return (colored, pvs_ov, _bw_preview(binary),
                            None, [], None, binary, [], 0.0, 0, 0,
                            [], None, status, binary, inst_masks, [], [])

            me_btn_auto.click(
                fn=_auto_detect,
                inputs=[st_me_img, st_is_video, st_vid_path, me_prompt, me_conf,
                        vp_target_fps, vp_framelim, st_me_hist, st_me_pts, st_me_modes],
                outputs=[me_colored, me_pvs_overlay, me_mask_bw,
                         st_session_id, st_vid_frames, st_temp_dir,
                         st_shared_mask, st_vid_obj_masks,
                         st_out_fps, st_vid_h, st_vid_w,
                         st_prompt_hist, vp_keyframe_preview, me_status,
                         st_me_mask, st_me_inst, st_pcs_masks, st_pending_pts],
            )

            def _click(img, is_vid, session_id, frames, vid_h, vid_w,
                       mask, mode, hist, pts, modes,
                       prompt_hist, prompt, prev_obj_masks, pending_pts,
                       evt: gr.SelectData):
                x, y = evt.index
                mode = mode or "add"

                if is_vid and session_id:
                    frame0_pil = Image.fromarray(frames[0])
                    if mode == "erase":
                        clicked_obj_id = None
                        yi, xi = int(y), int(x)
                        for obj_id, obj_mask in (prev_obj_masks or []):
                            if (0 <= yi < obj_mask.shape[0] and 0 <= xi < obj_mask.shape[1]
                                    and obj_mask[yi, xi] > 0):
                                clicked_obj_id = obj_id
                                break
                        if clicked_obj_id is None:
                            overlay = _render_native_outputs(frame0_pil, prev_obj_masks or [])
                            if pending_pts:
                                overlay = draw_points_on_image(overlay,
                                    [[p[0], p[1]] for p in pending_pts], ["add"]*len(pending_pts))
                            binary = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                            return (overlay, _bw_preview(binary), prompt_hist,
                                    pts, modes, "⚠️ Clicked background — no object to erase.",
                                    binary, prev_obj_masks or [], pending_pts,
                                    gr.update(visible=bool(pending_pts)),
                                    gr.update(visible=bool(pending_pts)))
                        _, _, status = vid_remove_object(session_id, frames, vid_h, vid_w, clicked_obj_id)
                        merged_list = [(oid, m) for oid, m in (prev_obj_masks or []) if oid != clicked_obj_id]
                        binary  = (_build_label_map(merged_list, (vid_h, vid_w)) > 0).astype(np.uint8) if merged_list else None
                        overlay = _render_native_outputs(frame0_pil, merged_list)
                        req_entry = dict(type="remove_object", frame_index=0, obj_id=clicked_obj_id)
                        new_hist  = (list(prompt_hist) + [{"reqs": [req_entry], "masks": merged_list}])[-MAX_VID_HIST:]
                        return (overlay, _bw_preview(binary), new_hist,
                                pts, modes, status, binary, merged_list, pending_pts,
                                gr.update(visible=bool(pending_pts)),
                                gr.update(visible=bool(pending_pts)))
                    else:
                        yi, xi = int(y), int(x)
                        already_covered = any(
                            0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi] > 0
                            for _, m in (prev_obj_masks or [])
                        )
                        if already_covered:
                            overlay = _render_native_outputs(frame0_pil, prev_obj_masks or [])
                            if pending_pts:
                                overlay = draw_points_on_image(overlay,
                                    [[p[0], p[1]] for p in pending_pts], ["add"]*len(pending_pts))
                            binary = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                            n_staged = len(pending_pts or [])
                            status = (f"⚠️ Click lands inside an existing segment — ignored."
                                      + (f" ({n_staged} point(s) still staged)" if n_staged else ""))
                            return (overlay, _bw_preview(binary), prompt_hist,
                                    pts, modes, status, binary, prev_obj_masks or [], pending_pts,
                                    gr.update(visible=bool(pending_pts)),
                                    gr.update(visible=bool(pending_pts)))
                        new_pending = list(pending_pts or []) + [[int(x), int(y)]]
                        overlay = _render_native_outputs(frame0_pil, prev_obj_masks or [])
                        overlay = draw_points_on_image(overlay,
                            [[p[0], p[1]] for p in new_pending], ["add"]*len(new_pending))
                        binary  = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                        status  = f"🟢 {len(new_pending)} point(s) staged — click Submit Points to register"
                        return (overlay, _bw_preview(binary), prompt_hist,
                                pts, modes, status, binary, prev_obj_masks or [], new_pending,
                                gr.update(visible=True), gr.update(visible=True))
                else:
                    ov, new_mask, hist, pts, modes, status = \
                        mask_editor_click_image(evt, img, mask, mode, hist, pts, modes)
                    return (ov, _bw_preview(new_mask), hist,
                            pts, modes, status, new_mask, [], [],
                            gr.update(visible=False), gr.update(visible=False))

            me_pvs_overlay.select(
                fn=_click,
                inputs=[st_me_img, st_is_video, st_session_id, st_vid_frames,
                        st_vid_h, st_vid_w, st_me_mask, me_mode, st_me_hist,
                        st_me_pts, st_me_modes, st_prompt_hist, me_prompt,
                        st_vid_obj_masks, st_pending_pts],
                outputs=[me_pvs_overlay, me_mask_bw, st_prompt_hist, st_me_pts, st_me_modes,
                         me_status, st_me_mask, st_vid_obj_masks, st_pending_pts,
                         me_btn_submit, me_btn_pop],
            )

            def _submit_batch(session_id, frames, vid_h, vid_w, prev_obj_masks, pending_pts, prompt_hist):
                if not session_id or not pending_pts:
                    frame0_pil = Image.fromarray(frames[0]) if frames else None
                    overlay = _render_native_outputs(frame0_pil, prev_obj_masks or []) if frame0_pil else None
                    binary  = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                    return (overlay, _bw_preview(binary), prompt_hist,
                            "⚠️ No points to submit.", binary, prev_obj_masks or [], [],
                            gr.update(visible=False), gr.update(visible=False))
                frame0_pil = Image.fromarray(frames[0])
                merged = {oid: m for oid, m in (prev_obj_masks or [])}
                batch_reqs = []
                base_id = max((oid for oid, _ in (prev_obj_masks or [])), default=998) + 1
                if base_id < 1000: base_id = 1000
                for i, (x, y) in enumerate(pending_pts):
                    obj_id = base_id + i
                    _, new_masks, _ = vid_add_point(session_id, frames, vid_h, vid_w, int(x), int(y), 1, obj_id)
                    for oid, m in new_masks:
                        merged[oid] = m
                    batch_reqs.append(dict(type="add_prompt", frame_index=0,
                                           points=[[x/max(vid_w,1), y/max(vid_h,1)]],
                                           point_labels=[1], obj_id=obj_id))
                merged_list = list(merged.items())
                binary  = (_build_label_map(merged_list, (vid_h, vid_w)) > 0).astype(np.uint8) if merged_list else None
                overlay = _render_native_outputs(frame0_pil, merged_list)
                new_hist = (list(prompt_hist) + [{"reqs": batch_reqs, "masks": merged_list}])[-MAX_VID_HIST:]
                status  = f"✅ {len(pending_pts)} point(s) submitted — will be tracked in Propagate"
                return (overlay, _bw_preview(binary), new_hist,
                        status, binary, merged_list, [],
                        gr.update(visible=False), gr.update(visible=False))

            me_btn_submit.click(
                fn=_submit_batch,
                inputs=[st_session_id, st_vid_frames, st_vid_h, st_vid_w,
                        st_vid_obj_masks, st_pending_pts, st_prompt_hist],
                outputs=[me_pvs_overlay, me_mask_bw, st_prompt_hist,
                         me_status, st_me_mask, st_vid_obj_masks, st_pending_pts,
                         me_btn_submit, me_btn_pop],
            )

            def _pop_point(frames, vid_h, vid_w, prev_obj_masks, pending_pts):
                if not pending_pts:
                    frame0_pil = Image.fromarray(frames[0]) if frames else None
                    overlay = _render_native_outputs(frame0_pil, prev_obj_masks or []) if frame0_pil else None
                    binary  = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                    return (overlay, _bw_preview(binary), [], "⚠️ No staged points to remove.",
                            gr.update(visible=False), gr.update(visible=False))
                new_pending = pending_pts[:-1]
                frame0_pil  = Image.fromarray(frames[0])
                overlay     = _render_native_outputs(frame0_pil, prev_obj_masks or [])
                if new_pending:
                    overlay = draw_points_on_image(overlay,
                        [[p[0], p[1]] for p in new_pending], ["add"]*len(new_pending))
                binary  = (_build_label_map(prev_obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if prev_obj_masks else None
                removed = pending_pts[-1]
                status  = (f"⌫ Removed point at ({removed[0]},{removed[1]})."
                           + (f" {len(new_pending)} remaining." if new_pending else " Queue empty."))
                return (overlay, _bw_preview(binary), new_pending, status,
                        gr.update(visible=bool(new_pending)),
                        gr.update(visible=bool(new_pending)))

            me_btn_pop.click(
                fn=_pop_point,
                inputs=[st_vid_frames, st_vid_h, st_vid_w, st_vid_obj_masks, st_pending_pts],
                outputs=[me_pvs_overlay, me_mask_bw, st_pending_pts, me_status,
                         me_btn_submit, me_btn_pop],
            )

            def _undo(is_vid, session_id, frames, vid_h, vid_w,
                      prompt, prompt_hist, pcs_masks, img, mask, hist):
                if is_vid and session_id:
                    if not prompt_hist:
                        overlay = _render_native_outputs(Image.fromarray(frames[0]), pcs_masks or [])
                        binary  = (_build_label_map(pcs_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if pcs_masks else None
                        vid_reset_to_pcs(session_id, frames, vid_h, vid_w, prompt)
                        return overlay, _bw_preview(binary), [], [], [], "↩️ Nothing to undo — restored to PCS.", binary, pcs_masks or [], gr.update(visible=False), gr.update(visible=False)
                    new_hist = prompt_hist[:-1]
                    try:
                        VID_PREDICTOR.handle_request(dict(type="reset_session", session_id=session_id))
                        VID_PREDICTOR.handle_request(dict(
                            type="add_prompt", session_id=session_id, frame_index=0,
                            text=_normalize_prompt(prompt)))
                        for entry in new_hist:
                            reqs = entry.get("reqs", [entry]) if isinstance(entry, dict) else [entry]
                            for req in reqs:
                                r = {k: v for k, v in req.items() if k not in ("masks",)}
                                r["session_id"] = session_id
                                VID_PREDICTOR.handle_request(r)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                    display_masks = (new_hist[-1].get("masks", pcs_masks or [])
                                     if new_hist and isinstance(new_hist[-1], dict)
                                     else (pcs_masks or []))
                    overlay = _render_native_outputs(Image.fromarray(frames[0]), display_masks)
                    binary  = (_build_label_map(display_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if display_masks else None
                    status  = f"↩️ Undo — {len(new_hist)} action(s) remaining"
                    return overlay, _bw_preview(binary), new_hist, [], [], status, binary, display_masks, gr.update(visible=False), gr.update(visible=False)
                else:
                    if img is None or mask is None:
                        return None, None, hist, [], [], "Nothing to undo.", mask, [], gr.update(), gr.update()
                    if not hist:
                        ov = apply_mask_overlay(img, mask, opacity=0.5)
                        return ov, _bw_preview(mask), [], [], [], "History empty.", mask, [], gr.update(), gr.update()
                    prev_mask, prev_pts, prev_modes = hist.pop()
                    ov = apply_mask_overlay(img, prev_mask, opacity=0.5)
                    ov = draw_points_on_image(ov, prev_pts, prev_modes)
                    return (ov, _bw_preview(prev_mask), hist, prev_pts, prev_modes,
                            f"↩️ Undo — {len(hist)} step(s) remaining.", prev_mask, [], gr.update(), gr.update())

            me_btn_undo.click(
                fn=_undo,
                inputs=[st_is_video, st_session_id, st_vid_frames, st_vid_h, st_vid_w,
                        me_prompt, st_prompt_hist, st_pcs_masks, st_me_img, st_me_mask, st_me_hist],
                outputs=[me_pvs_overlay, me_mask_bw, st_prompt_hist, st_me_pts, st_me_modes,
                         me_status, st_me_mask, st_vid_obj_masks, me_btn_submit, me_btn_pop],
            )

            def _reset(is_vid, session_id, frames, vid_h, vid_w, prompt, img, auto_mask):
                if is_vid and session_id:
                    overlay, obj_masks, status = vid_reset_to_pcs(session_id, frames, vid_h, vid_w, prompt)
                    binary = (_build_label_map(obj_masks, (vid_h, vid_w)) > 0).astype(np.uint8) if obj_masks else None
                    return overlay, _bw_preview(binary), [], [], [], status, binary, obj_masks
                else:
                    if img is None or auto_mask is None:
                        return None, None, [], [], [], "Run Auto Detect first.", auto_mask, []
                    ov = apply_mask_overlay(img, auto_mask, opacity=0.5)
                    return ov, _bw_preview(auto_mask), [], [], [], "🔄 Reset to auto mask.", auto_mask, []

            me_btn_reset.click(
                fn=_reset,
                inputs=[st_is_video, st_session_id, st_vid_frames, st_vid_h, st_vid_w,
                        me_prompt, st_me_img, st_me_auto],
                outputs=[me_pvs_overlay, me_mask_bw, st_prompt_hist, st_me_pts, st_me_modes,
                         me_status, st_me_mask, st_vid_obj_masks],
            )

            def _clrpt(img, mask):
                if img is None or mask is None:
                    return None, mask, [], [], "Nothing to clear."
                ov = apply_mask_overlay(img, mask, opacity=0.5)
                return ov, mask, [], [], "🧹 Markers cleared."
            me_btn_clrpt.click(fn=_clrpt, inputs=[st_me_img, st_me_mask],
                               outputs=[me_pvs_overlay, st_me_mask, st_me_pts, st_me_modes, me_status])

            def _save(is_vid, mask, inst_masks, obj_masks, pil_img, vid_frames, vid_h, vid_w,
                      src_path, out_dir, prompt, conf, pts, modes, min_px):
                if is_vid and obj_masks:
                    label_map  = _build_label_map(obj_masks, (vid_h, vid_w))
                    binary     = (label_map > 0).astype(np.uint8)
                    frame0_pil = Image.fromarray(vid_frames[0]) if vid_frames else pil_img
                    status = save_mask_with_meta(binary, obj_masks, frame0_pil, src_path, out_dir,
                                                 prompt, conf, [], [], is_keyframe=True, frame_idx=0)
                    return binary, status
                elif not is_vid and mask is not None:
                    clean = remove_small_regions(mask, int(min_px))
                    is_video_path = src_path is not None and Path(src_path).suffix.lower() in VID_EXTS
                    status = save_mask_with_meta(clean, inst_masks, pil_img, src_path, out_dir,
                                                 prompt, conf, pts, modes,
                                                 is_keyframe=is_video_path,
                                                 frame_idx=0 if is_video_path else None)
                    return clean, status
                return mask, "⚠️ Nothing to save."

            me_btn_save.click(
                fn=_save,
                inputs=[st_is_video, st_me_mask, st_me_inst, st_vid_obj_masks,
                        st_me_img, st_vid_frames, st_vid_h, st_vid_w,
                        st_vid_path, st_out_dir, me_prompt, me_conf,
                        st_me_pts, st_me_modes, shared_min_px],
                outputs=[st_shared_mask, me_status],
            )

        with tab_video_prop:
            gr.Markdown("""
Propagates **all detected objects** from Tab 2's active session across all video frames.

**Run PCS in Tab 2 first**, then refine if needed, then click Propagate here.

**Outputs** → `output_folder/{video_stem}/`:
- `masks/{N:06d}.png` + `.npy` — per-frame instance label map (0=bg, 1..N=object IDs)
- `overlays/{N:06d}.png` — per-frame rainbow overlay
- `meta/{N:06d}.json` — per-frame object count + IDs
- `overlay.mp4` — stitched overlay video
- `summary.json`
""")
            with gr.Row():
                with gr.Column(scale=4):
                    vp_video_out = gr.Video(label="Overlay Video", height=430)
                    vp_log       = gr.Textbox(label="Propagation Log", lines=14, interactive=False)
            with gr.Row():
                vp_video_info = gr.Textbox(label="Video (from Tab 2)", interactive=False, scale=3)
            with gr.Row():
                with gr.Column(scale=1):
                    vp_keyframe_preview  # renders here (already declared visible=False, toggled by _load)
                    vp_framelim_tab3  = gr.Slider(0, 2000, value=_startup_frame_limit, step=10,
                                                   label="Frame Limit (0 = all frames)")
                    vp_target_fps_tab3 = gr.Slider(0, 30, value=_startup_target_fps, step=1,
                                                   label="Target FPS (0 = source FPS)")
            with gr.Row():
                vp_btn = gr.Button("🚀 Propagate Video", variant="primary")

            vp_target_fps.release(
                fn=lambda v: (_save_config("target_fps", int(v)) or int(v)),
                inputs=[vp_target_fps], outputs=[vp_target_fps]
            )
            vp_framelim.release(
                fn=lambda v: (_save_config("frame_limit", int(v)) or int(v)),
                inputs=[vp_framelim], outputs=[vp_framelim]
            )
            # Tab 3 slider save-to-config and sync back to hidden Tab 2 states
            vp_target_fps_tab3.release(
                fn=lambda v: (_save_config("target_fps", int(v)) or int(v)),
                inputs=[vp_target_fps_tab3], outputs=[vp_target_fps_tab3]
            )
            vp_framelim_tab3.release(
                fn=lambda v: (_save_config("frame_limit", int(v)) or int(v)),
                inputs=[vp_framelim_tab3], outputs=[vp_framelim_tab3]
            )

            def _propagate(session_id, frames, temp_dir, vid_path, out_dir, out_fps, min_px,
                           prompt, prompt_history, target_fps, frame_limit):
                vid_display = vid_path or "(none loaded)"
                print(f"[_propagate] session_id={session_id!r} n_frames={len(frames) if frames else 0} "
                      f"vid_path={vid_path!r} out_fps={out_fps} target_fps={target_fps}")
                if not session_id:
                    print("[_propagate] ❌ no session_id — aborting")
                    yield vid_display, None, None, "❌ No session. Run Auto Detect (PCS) in Tab 2."
                    return
                yield vid_display, None, None, "⏳ Starting propagation …"
                for mp4, log in run_video_propagation(
                    session_id, frames, temp_dir, vid_path, out_dir,
                    float(out_fps), int(min_px), prompt=prompt,
                    prompt_history=prompt_history,
                    target_fps=float(target_fps), frame_limit=int(frame_limit),
                ):
                    yield vid_display, None, mp4, log

            vp_btn.click(
                fn=_propagate,
                inputs=[st_session_id, st_vid_frames, st_temp_dir, st_vid_path,
                        st_out_dir, st_out_fps, shared_min_px,
                        me_prompt, st_prompt_hist, vp_target_fps_tab3, vp_framelim_tab3],
                outputs=[vp_video_info, vp_keyframe_preview, vp_video_out, vp_log],
            )

demo.queue(default_concurrency_limit=2, max_size=4)

if __name__ == "__main__":
    demo.launch(
        theme=app_theme,
        css=custom_css,
        ssr_mode=False,
        mcp_server=False,
        show_error=True,
        max_threads=2,
    )
