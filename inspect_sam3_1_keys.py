#!/usr/bin/env python3
"""
SAM3.1 key/shape inspector — Tier 1 diagnostic.

Loads the native 3.1 checkpoint (already cached from the previous run) and
empties of all three transformers SAM3 classes. Tries naive prefix stripping
('tracker.model.' -> '') and counts how many checkpoint tensors match each
transformers class by (name, shape).

Output:
- Console: summary counts per class, top unmatched prefix histograms.
- File:    full per-class match log at OUTPUT_DIR / sam3_1_key_diagnostic.txt
"""
from pathlib import Path
import os
import sys
from collections import Counter

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    Sam3Model, Sam3Config,
    Sam3TrackerModel, Sam3TrackerConfig,
    Sam3VideoModel, Sam3VideoConfig,
)

REPO = "jetjodh/sam3.1"
WEIGHT_FILE = "sam3.1_multiplex.pt"
OUTPUT_DIR = Path(r"D:\Xiaoyu's Life\1\data\outputs")
LOG_PATH = OUTPUT_DIR / "sam3_1_key_diagnostic.txt"


def common_prefix_strip(keys: list[str]) -> tuple[str, list[str]]:
    """Find the single longest dotted prefix shared by ALL keys; return (prefix, stripped)."""
    if not keys:
        return "", keys
    parts0 = keys[0].split(".")
    for k in keys[1:]:
        p = k.split(".")
        i = 0
        while i < len(parts0) and i < len(p) and parts0[i] == p[i]:
            i += 1
        parts0 = parts0[:i]
        if not parts0:
            break
    prefix = ".".join(parts0)
    if prefix:
        prefix_dot = prefix + "."
        return prefix, [k[len(prefix_dot):] if k.startswith(prefix_dot) else k for k in keys]
    return "", keys


def compare(name: str, model_sd: dict, ckpt_sd: dict, log) -> dict:
    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())
    name_overlap = model_keys & ckpt_keys
    shape_match = sum(1 for k in name_overlap if model_sd[k].shape == ckpt_sd[k].shape)
    shape_mismatch = len(name_overlap) - shape_match
    only_model = model_keys - ckpt_keys
    only_ckpt = ckpt_keys - model_keys

    line = (
        f"\n=== {name} ===\n"
        f"  model keys:        {len(model_keys)}\n"
        f"  ckpt  keys:        {len(ckpt_keys)}\n"
        f"  name overlap:      {len(name_overlap)}\n"
        f"  shape-exact match: {shape_match}\n"
        f"  shape mismatch:    {shape_mismatch}\n"
        f"  only-in-model:     {len(only_model)}\n"
        f"  only-in-ckpt:      {len(only_ckpt)}\n"
    )
    print(line)
    log.write(line)

    def top_prefixes(keys, n=10, depth=2):
        c = Counter(".".join(k.split(".")[:depth]) for k in keys)
        return c.most_common(n)

    if only_model:
        log.write(f"  top only-in-model prefixes (depth=2):\n")
        for p, n in top_prefixes(only_model):
            log.write(f"    {n:5d}  {p}\n")
    if only_ckpt:
        log.write(f"  top only-in-ckpt  prefixes (depth=2):\n")
        for p, n in top_prefixes(only_ckpt):
            log.write(f"    {n:5d}  {p}\n")
    if name_overlap and shape_mismatch > 0:
        log.write(f"  shape-mismatched (sample 10):\n")
        for k in list(name_overlap)[:10]:
            if model_sd[k].shape != ckpt_sd[k].shape:
                log.write(f"    {k}  model={tuple(model_sd[k].shape)} ckpt={tuple(ckpt_sd[k].shape)}\n")

    return {
        "model_keys": len(model_keys),
        "ckpt_keys": len(ckpt_keys),
        "shape_match": shape_match,
        "shape_mismatch": shape_mismatch,
        "only_model": len(only_model),
        "only_ckpt": len(only_ckpt),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[step] downloading / locating cached checkpoint...")
    ckpt_path = hf_hub_download(repo_id=REPO, filename=WEIGHT_FILE)
    print(f"[step] torch.load({Path(ckpt_path).name})...")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in state and isinstance(state[k], dict):
                print(f"       unwrapped via key '{k}'")
                state = state[k]
                break

    raw_keys = list(state.keys())
    print(f"[info] raw checkpoint has {len(raw_keys)} tensors")
    print(f"[info] sample raw keys: {raw_keys[:3]}")

    # Strip the tracker.model.* prefix that we saw earlier.
    common, stripped_keys = common_prefix_strip(raw_keys)
    print(f"[info] common prefix across all keys: {common!r}")
    if common:
        ckpt_sd = {nk: state[ok] for ok, nk in zip(raw_keys, stripped_keys)}
    else:
        ckpt_sd = state

    print(f"[info] sample stripped keys: {list(ckpt_sd.keys())[:5]}")

    log = LOG_PATH.open("w", encoding="utf-8")
    log.write(f"SAM3.1 key/shape diagnostic\n")
    log.write(f"repo={REPO} file={WEIGHT_FILE}\n")
    log.write(f"raw ckpt keys: {len(raw_keys)} | stripped prefix: {common!r}\n")
    log.write(f"sample stripped keys: {list(ckpt_sd.keys())[:5]}\n")

    summary = {}
    for cls_name, ModelCls, ConfigCls in [
        ("Sam3Model",        Sam3Model,        Sam3Config),
        ("Sam3TrackerModel", Sam3TrackerModel, Sam3TrackerConfig),
        ("Sam3VideoModel",   Sam3VideoModel,   Sam3VideoConfig),
    ]:
        try:
            print(f"[step] building empty {cls_name}...")
            cfg = ConfigCls.from_pretrained(REPO)
            model = ModelCls(cfg)
            model_sd = {k: v for k, v in model.state_dict().items()}
        except Exception as e:
            print(f"  [warn] {cls_name} failed to build from {REPO} config: {type(e).__name__}: {e}")
            log.write(f"\n=== {cls_name} ===\n  build failed: {type(e).__name__}: {e}\n")
            continue
        summary[cls_name] = compare(cls_name, model_sd, ckpt_sd, log)
        del model

    log.write("\n=== summary ===\n")
    for k, v in summary.items():
        log.write(f"  {k}: shape_match={v['shape_match']}/{v['ckpt_keys']} (model_total={v['model_keys']})\n")
    log.close()

    print(f"\n[done] full log -> {LOG_PATH}")
    print(f"[summary] shape-exact matches per class:")
    for k, v in summary.items():
        pct = 100.0 * v["shape_match"] / max(v["ckpt_keys"], 1)
        print(f"  {k:20s}  {v['shape_match']:5d} / {v['ckpt_keys']:5d}  ({pct:5.1f}% of ckpt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
