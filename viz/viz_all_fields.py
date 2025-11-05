#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-field 2D viewer (R–Z) in a single window.

Examples:
  python viz/viz_all_fields.py --run-dir ./data/step7_zpinch_kScan --make-mp4 --every 2 --sym vr vz Bth p
  python viz/viz_all_fields.py --run-dir ./data/step7_zpinch_kScan --grid 2x3 --pclip 1,99 vr vz Bth Br Bz p
"""
import argparse, re
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- I/O helpers --------------------
def load_meta(meta_path: Path):
    """Parse meta like:
       t=1.234e-05
       Nr=128, Nz=256, Ng=2
       Rmax=0.02, Zmax=0.10
    """
    s = meta_path.read_text(encoding="utf-8")
    # Intento 1: regex “bonito”
    rt  = re.search(r"t=([0-9eE.+\-]+)", s)
    rNZ = re.search(r"Nr=(\d+),\s*Nz=(\d+),\s*Ng=(\d+)", s)
    rRZ = re.search(r"Rmax=([0-9eE.+\-]+),\s*Zmax=([0-9eE.+\-]+)", s)
    if rt and rNZ and rRZ:
        t = float(rt.group(1))
        Nr, Nz, Ng = map(int, rNZ.groups())
        Rmax, Zmax = map(float, rRZ.groups())
        return t, Nr, Nz, Ng, Rmax, Zmax
    # Intento 2: parser “key=value” tolerante
    kv = {}
    for tok in re.split(r"[,\s]+", s.strip()):
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
    try:
        t    = float(kv["t"])
        Nr   = int(kv["Nr"]); Nz = int(kv["Nz"]); Ng = int(kv["Ng"])
        Rmax = float(kv["Rmax"]); Zmax = float(kv["Zmax"])
        return t, Nr, Nz, Ng, Rmax, Zmax
    except Exception as e:
        raise ValueError(f"[meta parse] No pude leer {meta_path}: {e}")

def find_steps(run_dir: Path):
    return sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("fields_*_meta.txt"))

def load_field(run_dir: Path, step: int, field: str):
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", dtype=float)

# -------------------- geometry & crop --------------------
def reshape_crop(vec, Nr, Nz, Ng, crop_inner=1):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    i0 = Ng + crop_inner; i1 = Ng + Nr - crop_inner
    k0 = Ng + crop_inner; k1 = Ng + Nz - crop_inner
    if not (0 <= i0 < i1 <= NrT and 0 <= k0 < k1 <= NzT):  # fallback
        i0, i1, k0, k1 = Ng, Ng+Nr, Ng, Ng+Nz
    return A[i0:i1, k0:k1], (i0, k0)

# -------------------- color-limits --------------------
def compute_limits(run_dir, steps, field, Nr, Nz, Ng, pclip=(2.0,98.0), sample_frames=30, symmetric=False, crop_inner=1, log1p=False):
    vals = []
    take = max(1, len(steps)//sample_frames)
    for s in steps[::take]:
        vec = load_field(run_dir, s, field)
        if vec is None:
            continue
        F,_ = reshape_crop(vec, Nr, Nz, Ng, crop_inner=crop_inner)
        if log1p:
            F = np.sign(F) * np.log1p(np.abs(F))
        vals.append(F.ravel())
    if not vals:
        return None, None
    big = np.concatenate(vals)
    lo, hi = np.percentile(big, pclip)
    if symmetric:
        m = float(max(abs(lo), abs(hi)))
        return -m, m
    return float(lo), float(hi)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--pclip", type=str, default="2,98", help="percentile lo,hi")
    ap.add_argument("--sym", action="store_true", help="límites simétricos (para campos con signo)")
    ap.add_argument("--crop-inner", type=int, default=1)
    ap.add_argument("--log1p", action="store_true", help="mostrar log1p(|F|)*sign(F)")
    ap.add_argument("--grid", type=str, default="", help="forzar cuadricula RxC, ej: 2x3")
    ap.add_argument("--width", type=int, default=1200, help="ancho px del canvas total")
    ap.add_argument("fields", nargs="*", default=["rho","p","vr","vz","Bth","Br","Bz","vth"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    steps = find_steps(run_dir)
    if not steps:
        raise SystemExit("[ERR] No hay snapshots (fields_*_meta.txt).")

    # meta y malla
    t0,Nr_full,Nz_full,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr = Rmax / Nr_full; dz = Zmax / Nz_full

    # filtra campos existentes
    fields = []
    for f in args.fields:
        if (run_dir / f"fields_{steps[0]}_{f}.csv").exists() or any((run_dir / f"fields_{s}_{f}.csv").exists() for s in steps):
            fields.append(f)
    if not fields:
        raise SystemExit("[ERR] Ninguno de los campos existe en los snapshots.")

    # límites por campo
    p1,p2 = [float(x) for x in args.pclip.split(",")]
    limits = {}
    for f in fields:
        vmin,vmax = compute_limits(run_dir, steps, f, Nr_full, Nz_full, Ng,
                                   pclip=(p1,p2), symmetric=args.sym,
                                   crop_inner=args.crop_inner, log1p=args.log1p)
        limits[f]=(vmin,vmax)

    # primer frame (para tamaños/extents)
    vec0 = load_field(run_dir, steps[0], fields[0])
    F0,(i0crop,k0crop) = reshape_crop(vec0, Nr_full, Nz_full, Ng, crop_inner=args.crop_inner)
    if args.log1p: F0 = np.sign(F0)*np.log1p(np.abs(F0))
    Nr = F0.shape[0]; Nz = F0.shape[1]
    i_globals = i0crop + np.arange(Nr)
    k_globals = k0crop + np.arange(Nz)
    r_vals = (i_globals - Ng + 0.5)*dr
    z_vals = (k_globals - Ng + 0.5)*dz
    extent0 = [z_vals[0], z_vals[-1], r_vals[0], r_vals[-1]]

    # layout
    n = len(fields)
    if args.grid:
        R,C = map(int, args.grid.lower().split("x"))
    else:
        C = int(np.ceil(np.sqrt(n)))
        R = int(np.ceil(n / C))
    if R*C < n:
        C = int(np.ceil(n / R))

    # figura
    target_w = args.width
    subplot_w = target_w / max(C,1)
    subplot_h = subplot_w * 2/3
    H = int(np.ceil(subplot_h*R))
    fig = plt.figure(figsize=(target_w/100, H/100), dpi=100)
    axs, ims, cbs = [], [], []
    for idx,f in enumerate(fields):
        ax = fig.add_subplot(R, C, idx+1)
        axs.append(ax)
        vmin,vmax = limits[f]
        im = ax.imshow(F0*0.0, origin="lower", aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax, extent=extent0)
        ims.append(im)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label(f)
        cbs.append(cb)
        ax.set_xlabel("z [m]"); ax.set_ylabel("r [m]")
        ax.set_title(f)
    suptxt = fig.suptitle("", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])

    writer = None
    if args.make_mp4:
        out_mp4 = run_dir / f"multifield_movie.mp4"
        writer = imageio.get_writer(out_mp4, fps=args.fps, macro_block_size=None)

    # frames
    for step in steps[::args.every]:
        meta = run_dir / f"fields_{step}_meta.txt"
        if not meta.exists(): 
            continue
        t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)
        i_globals = (NgF + args.crop_inner) + np.arange(NrF - 2*args.crop_inner)
        k_globals = (NgF + args.crop_inner) + np.arange(NzF - 2*args.crop_inner)
        r_vals = (i_globals - NgF + 0.5) * (RmaxF / NrF)
        z_vals = (k_globals - NgF + 0.5) * (ZmaxF / NzF)
        extent = [z_vals[0], z_vals[-1], r_vals[0], r_vals[-1]]

        for im, f in zip(ims, fields):
            vec = load_field(run_dir, step, f)
            if vec is None:
                continue
            F,_ = reshape_crop(vec, NrF, NzF, NgF, crop_inner=args.crop_inner)
            if args.log1p: F = np.sign(F)*np.log1p(np.abs(F))
            vmin,vmax = limits[f]
            F = np.clip(F, vmin, vmax)
            im.set_data(F); im.set_extent(extent)

        suptxt.set_text(f"t = {t*1e6:.2f} µs   step={step}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        if writer:
            writer.append_data(frame)

    out_png = run_dir / "multifield_last.png"
    fig.savefig(out_png, dpi=args.dpi)
    if writer:
        writer.close()
        print("Saved:", out_mp4)
    plt.close(fig)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()

