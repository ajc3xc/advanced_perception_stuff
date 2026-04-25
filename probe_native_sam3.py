#!/usr/bin/env python3
"""
Native sam3 package probe.

Checks whether the conda-forge `sam3` package is installed in this pixi env,
prints what it exposes, and attempts to load the cached 3.1 checkpoint with it.
No model construction yet — just import + introspect.
"""
import sys
import traceback
from pathlib import Path

print("[step] import sam3...")
try:
    import sam3
    print(f"[ok]   sam3 imported. version: {getattr(sam3, '__version__', '?')}")
    print(f"       location: {Path(sam3.__file__).parent}")
except Exception as e:
    print(f"[FAIL] cannot import sam3: {type(e).__name__}: {e}")
    print("       -> native package not installed in this env. Tier 2 = add to pixi.toml + reinstall.")
    sys.exit(2)

# Walk the public surface
print("\n[step] sam3 public attributes:")
public = [a for a in dir(sam3) if not a.startswith("_")]
for a in public:
    print(f"   - {a}")

# Try common submodules
print("\n[step] probe common submodules:")
for sub in ["build_sam3", "sam3_image_predictor", "sam3_video_predictor",
            "build_sam", "modeling", "configs", "utils"]:
    try:
        mod = __import__(f"sam3.{sub}", fromlist=["*"])
        names = [n for n in dir(mod) if not n.startswith("_")][:8]
        print(f"   - sam3.{sub}: {names}")
    except Exception as e:
        print(f"   - sam3.{sub}: NOT FOUND ({type(e).__name__})")

# Try locating a build function and a config
print("\n[step] hunting build_* / *Predictor entry points...")
import pkgutil
for finder, name, ispkg in pkgutil.walk_packages(sam3.__path__, prefix="sam3."):
    if any(t in name for t in ("build", "predictor", "config")):
        print(f"   - {name}")
