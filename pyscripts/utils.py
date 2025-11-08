#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — dependency check/installer for diagnostics and visualization scripts.

Behavior:
- If run with no arguments, shows extended help with workflow guidance.
- By default, verifies and installs any missing packages.

Typical usage:
  python utils.py                  # verify and install anything missing
  python utils.py --check          # verify only (no installation)
  python utils.py --help-extended  # show extended help

Notes:
- For MP4 export, having 'ffmpeg' in your PATH improves compatibility.
"""
from __future__ import annotations
import sys, subprocess, shutil, argparse, importlib
from typing import List, Tuple

# Required packages (specs for pip)
REQ_PKGS = [
    "numpy>=1.20",
    "matplotlib>=3.5",
    "imageio>=2.31",
    "imageio-ffmpeg>=0.4.9"
]

def have_module(mod: str) -> bool:
    """Return True if the importable module `mod` is available."""
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

def pip_install(specs: List[str]) -> Tuple[bool, List[str]]:
    """Install each package spec with pip; return (success_all, failed_specs)."""
    failed = []
    for spec in specs:
        print(f"[SETUP] Installing {spec} ...")
        code = subprocess.call([sys.executable, "-m", "pip", "install", spec])
        if code != 0:
            failed.append(spec)
    return (len(failed) == 0), failed

def ensure_deps(install_missing: bool = True) -> bool:
    """
    Verify required modules are present; optionally install missing ones.
    Returns True if environment is ready, False otherwise.
    """
    spec2mod = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "imageio": "imageio",
        "imageio-ffmpeg": "imageio_ffmpeg",
    }
    missing = []
    for spec in REQ_PKGS:
        root = spec.split(">=")[0].split("==")[0]
        mod  = spec2mod[root]
        if not have_module(mod):
            missing.append(spec)

    if not missing:
        print("[OK] All dependencies are present.")
        return True

    print("[INFO] Missing dependencies:", ", ".join(missing))
    if not install_missing:
        print("[SKIP] --check was given: not installing packages.")
        return False

    ok, failed = pip_install(missing)
    if not ok:
        print("[ERR] Failed to install:", ", ".join(failed))
        return False

    # Optional: ffmpeg in PATH is useful for MP4 export
    if shutil.which("ffmpeg") is None:
        print("[WARN] 'ffmpeg' not found in PATH. For MP4 export, install ffmpeg "
              "(Linux: apt/brew; Windows: winget/choco; macOS: brew), or let imageio-ffmpeg use its bundled binary.")

    print("[OK] Dependencies installed successfully.")
    return True

def print_extended_help() -> None:
    """Print extended human-friendly guidance."""
    txt = r"""
=== Extended Help (utils.py) ===

This script ensures the Python environment is ready for diagnostics and visualization.

What it does
------------
- Verifies required packages are installed: numpy, matplotlib, imageio, imageio-ffmpeg
- Optionally installs anything missing using pip

Suggested workflow
------------------
1) python utils.py
2) python diagnostics.py --help-extended
3) python viz.py --help-extended

Examples
--------
# Verify only (no installs)
python utils.py --check

# Verify and install anything missing
python utils.py

Notes
-----
- For better MP4 compatibility, ensure 'ffmpeg' is available on your system PATH.
"""
    print(txt.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Verify only (do not install)")
    ap.add_argument("--help-extended", action="store_true", help="Show extended guidance")
    args, unknown = ap.parse_known_args()

    # Show extended help if no arguments were provided
    if len(sys.argv) == 1 or args.help_extended:
        print_extended_help()
        return

    ensure_deps(install_missing=(not args.check))

if __name__ == "__main__":
    main()

