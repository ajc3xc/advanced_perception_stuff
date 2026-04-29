#!/usr/bin/env python3
"""
patch_sam3_vitdet.py — Fix SAM3 BFloat16/Float32 dtype mismatch on Windows.
Run once: pixi run python sam3_vitdet.py

Patches three specific input-to-weight dtype mismatches:
  1. vitdet.py            — PatchEmbed.proj (image backbone)
  2. geometry_encoders.py — points_direct_project (coordinate encoder)
  3. sam3_video_base.py   — conv_s0 on FPN features (mask decoder)

Also cleans any previously injected code from sam3_video_predictor.py.
  4. connected_components.py — triton fallback → CPU fallback (Windows has no triton)
"""
import sys, re
from pathlib import Path

VITDET_MARKER = "# [PATCHED] dtype cast for Windows BF16 compat"
GEO_MARKER    = "# [PATCHED-GEO] dtype cast for Windows BF16 compat"
BASE_MARKER   = "# [PATCHED-BASE] dtype cast for Windows BF16 compat"
ANY_PATCH_TAG = "[PATCHED"  # prefix shared by all markers


def find_sam3_dir():
    try:
        import sam3
        return Path(sam3.__file__).parent
    except Exception:
        import site
        for sp in site.getsitepackages():
            c = Path(sp) / "sam3"
            if c.exists(): return c
    return None


def _apply_single_patch(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    if not path.exists():
        print(f"❌ {label}: file not found at {path}"); return False
    content = path.read_text(encoding="utf-8")
    if marker in content:
        print(f"✅ {label}: already patched"); return True
    if old not in content:
        print(f"❌ {label}: pattern not found")
        for i, l in enumerate(content.splitlines()):
            if any(kw in l for kw in ["proj(", "points_direct", "conv_s0"]):
                print(f"   line {i+1}: {l!r}")
        return False
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"✅ {label}: patched"); return True


def patch_vitdet(sam3_dir):
    return _apply_single_patch(
        sam3_dir / "model" / "vitdet.py",
        old="        x = self.proj(x)\n",
        new=f"        x = self.proj(x.to(self.proj.weight.dtype))  {VITDET_MARKER}\n",
        marker=VITDET_MARKER, label="vitdet.py",
    )


def patch_geometry_encoders(sam3_dir):
    path = sam3_dir / "model" / "geometry_encoders.py"
    if not path.exists():
        print(f"❌ geometry_encoders.py: not found"); return False
    content = path.read_text(encoding="utf-8")
    if GEO_MARKER in content:
        print(f"✅ geometry_encoders.py: already patched"); return True
    for line in content.splitlines():
        if "self.points_direct_project(points)" in line and GEO_MARKER not in line:
            indent = len(line) - len(line.lstrip())
            sp = " " * indent
            old = f"{sp}proj = self.points_direct_project(points)\n"
            new = f"{sp}proj = self.points_direct_project(points.to(self.points_direct_project.weight.dtype))  {GEO_MARKER}\n"
            if old in content:
                path.write_text(content.replace(old, new, 1), encoding="utf-8")
                print(f"✅ geometry_encoders.py: patched"); return True
    print(f"❌ geometry_encoders.py: pattern not found"); return False


BASE_MARKER2 = "# [PATCHED-BASE2] dtype cast for Windows BF16 compat"


def patch_video_base(sam3_dir):
    path = sam3_dir / "model" / "sam3_video_base.py"
    if not path.exists():
        print(f"❌ sam3_video_base.py: not found"); return False
    content = path.read_text(encoding="utf-8")

    ok0 = True
    if BASE_MARKER not in content:
        OLD0 = 'sam_mask_decoder.conv_s0(sam3_image_out["tracker_backbone_fpn_0"]),'
        NEW0 = f'sam_mask_decoder.conv_s0(sam3_image_out["tracker_backbone_fpn_0"].to(sam_mask_decoder.conv_s0.weight.dtype)),  {BASE_MARKER}'
        if OLD0 not in content:
            print(f"❌ sam3_video_base.py: conv_s0 pattern not found")
            ok0 = False
        else:
            content = content.replace(OLD0, NEW0, 1)
            print(f"✅ sam3_video_base.py: conv_s0 patched")
    else:
        print(f"✅ sam3_video_base.py: conv_s0 already patched")

    ok1 = True
    if BASE_MARKER2 not in content:
        OLD1 = 'sam_mask_decoder.conv_s1(sam3_image_out["tracker_backbone_fpn_1"]),'
        NEW1 = f'sam_mask_decoder.conv_s1(sam3_image_out["tracker_backbone_fpn_1"].to(sam_mask_decoder.conv_s1.weight.dtype)),  {BASE_MARKER2}'
        if OLD1 not in content:
            print(f"❌ sam3_video_base.py: conv_s1 pattern not found")
            ok1 = False
        else:
            content = content.replace(OLD1, NEW1, 1)
            print(f"✅ sam3_video_base.py: conv_s1 patched")
    else:
        print(f"✅ sam3_video_base.py: conv_s1 already patched")

    if ok0 or ok1:
        path.write_text(content, encoding="utf-8")
    return ok0 and ok1


def patch_connected_components(sam3_dir):
    path = sam3_dir / "perflib" / "connected_components.py"
    marker = "# [PATCHED-CC] triton fallback -> CPU fallback for Windows"
    if not path.exists():
        print(f"❌ connected_components.py: not found"); return False
    content = path.read_text(encoding="utf-8")
    if marker in content:
        print(f"✅ connected_components.py: already patched"); return True
    OLD = ('            # triton fallback\n'
           '            from sam3.perflib.triton.connected_components import (\n'
           '                connected_components_triton,\n'
           '            )\n'
           '\n'
           '            return connected_components_triton(input_tensor)')
    NEW = ('            try:  ' + marker + '\n'
           '                from sam3.perflib.triton.connected_components import (\n'
           '                    connected_components_triton,\n'
           '                )\n'
           '                return connected_components_triton(input_tensor)\n'
           '            except (ImportError, ModuleNotFoundError):\n'
           '                logging.debug("triton not available, falling back to CPU connected components")\n'
           '                return connected_components_cpu(input_tensor.cpu())')
    if OLD not in content:
        print(f"❌ connected_components.py: pattern not found")
        return False
    path.write_text(content.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"✅ connected_components.py: patched"); return True


def clean_video_predictor(sam3_dir):
    path = sam3_dir / "model" / "sam3_video_predictor.py"
    if not path.exists(): return True
    content = path.read_text(encoding="utf-8")
    if ANY_PATCH_TAG not in content:
        print(f"✅ sam3_video_predictor.py: clean (nothing to remove)"); return True

    lines = content.splitlines(keepends=True)
    out = []
    i = 0
    removed = 0
    while i < len(lines):
        line = lines[i]
        if ANY_PATCH_TAG in line:
            while out and out[-1].strip() in ('try:', ''):
                out.pop()
                removed += 1
            inject_indent = len(line) - len(line.lstrip())
            i += 1
            removed += 1
            while i < len(lines):
                l = lines[i]
                stripped = l.strip()
                if not stripped:
                    i += 1; removed += 1; continue
                cur_indent = len(l) - len(l.lstrip())
                if stripped.startswith('except') and cur_indent <= inject_indent:
                    i += 1; removed += 1
                    while i < len(lines):
                        bl = lines[i]
                        bs = bl.strip()
                        if not bs: i += 1; removed += 1; continue
                        bi = len(bl) - len(bl.lstrip())
                        if bi > cur_indent: i += 1; removed += 1
                        else: break
                    break
                elif cur_indent > inject_indent or ANY_PATCH_TAG in l:
                    i += 1; removed += 1
                else:
                    break
        else:
            out.append(line)
            i += 1

    new_content = ''.join(out)
    path.write_text(new_content, encoding="utf-8")
    print(f"✅ sam3_video_predictor.py: removed {removed} injected lines")
    try:
        compile(new_content, str(path), 'exec')
        print(f"   syntax OK")
    except SyntaxError as e:
        print(f"   ⚠️  still has syntax error at line {e.lineno}: {e.msg}")
    return True


def patch_edt(sam3_dir):
    """Guard triton import in edt.py so sam3 can be imported on Windows."""
    path = sam3_dir / "model" / "edt.py"
    marker = "# [PATCHED-EDT] triton import guarded for Windows"
    if not path.exists():
        print(f"⚠️ edt.py: not found (skipping)"); return True
    content = path.read_text(encoding="utf-8")
    if marker in content:
        print(f"✅ edt.py: already patched"); return True
    old = "import torch\nimport triton\nimport triton.language as tl"
    new = (
        f"import torch\n"
        f"{marker}\n"
        f"try:\n"
        f"    import triton\n"
        f"    import triton.language as tl\n"
        f"    _TRITON_AVAILABLE = True\n"
        f"except (ImportError, ModuleNotFoundError):\n"
        f"    _TRITON_AVAILABLE = False"
    )
    if old not in content:
        print(f"⚠️ edt.py: pattern not found (may already be patched differently)"); return True
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"✅ edt.py: patched"); return True


def main():
    sam3_dir = find_sam3_dir()
    if sam3_dir is None:
        print("❌ sam3 package not found."); sys.exit(1)
    print(f"SAM3 package: {sam3_dir}\n")

    clean_video_predictor(sam3_dir)
    print()
    ok1 = patch_vitdet(sam3_dir)
    ok2 = patch_geometry_encoders(sam3_dir)
    ok3 = patch_video_base(sam3_dir)
    ok4 = patch_connected_components(sam3_dir)
    ok5 = patch_edt(sam3_dir)

    print()
    if all([ok1, ok2, ok3, ok4, ok5]):
        print("✅ All done — restart app_v5.py")
    else:
        print("⚠️  Some patches failed — check above")
        sys.exit(1)

if __name__ == "__main__":
    main()
