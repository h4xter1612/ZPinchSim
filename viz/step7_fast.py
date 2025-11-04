#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_meta(p: Path):
    s = p.read_text(encoding="utf-8")
    t   = float(re.search(r"t=([0-9eE\.\+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)", s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

def find_steps(run_dir: Path):
    return sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("fields_*_meta.txt"))

def load_field(run_dir: Path, step: int, field: str):
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists(): return None
    return np.loadtxt(p, delimiter=",", dtype=float)

def reshape_crop(vec, Nr, Nz, Ng, crop_inner=1):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.array(vec).reshape(NrT, NzT)
    # Recorta ghosts y además 1 celda física interior para evitar rayas en BC
    i0 = Ng + crop_inner; i1 = Ng + Nr - crop_inner
    k0 = Ng + crop_inner; k1 = Ng + Nz - crop_inner
    return A[i0:i1, k0:k1]

def compute_global_limits(run_dir, steps, field, Nr, Nz, Ng, every, pclip):
    vals = []
    step_samples = steps[::max(1, len(steps)//30)]  # muestrea ~30 frames
    for s in step_samples:
        vec = load_field(run_dir, s, field)
        if vec is None: continue
        F = reshape_crop(vec, Nr, Nz, Ng, crop_inner=1)
        vals.append(F.ravel())
    if not vals:
        return None, None
    big = np.concatenate(vals)
    lo, hi = np.percentile(big, pclip)
    # centra en 0 si el campo es con signo (vr, vz, B*, etc.)
    if field.lower() in ("vr","vz","br","bz","bth","vth"):
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--field", required=True,
                    choices=["rho","p","vr","vz","Bth","Br","Bz","vth"])
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--pclip", type=str, default="2,98")  # percentiles globales
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    steps = find_steps(run_dir)
    if not steps:
        print("[ERR] No hay snapshots."); return

    # Geometría del primer frame
    t0,Nr,Nz,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")

    # Límites globales
    p1,p2 = [float(x) for x in args.pclip.split(",")]
    vmin, vmax = compute_global_limits(run_dir, steps, args.field, Nr, Nz, Ng, args.every, (p1,p2))
    if vmin is None:
        print("[ERR] No hay campos para pintar."); return

    # Figura reutilizable (tamaño múltiplo de 16 px para evitar warnings de ffmpeg)
    W, H = 640, 400  # 40x25 px * 16
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax  = fig.add_subplot(111)
    im  = ax.imshow(np.zeros((Nr-2, Nz-2)), origin="lower", aspect="auto",
                    interpolation="nearest", vmin=vmin, vmax=vmax)
    cb  = fig.colorbar(im); cb.set_label(args.field)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("r [mm]")
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, ha="left", va="top",
                  color="w", fontsize=9,
                  bbox=dict(facecolor="0.0", alpha=0.35, pad=2, edgecolor="none"))
    fig.tight_layout()

    writer = None
    if args.make_mp4:
        writer = imageio.get_writer(run_dir / f"{args.field}_movie.mp4", fps=args.fps, macro_block_size=None)

    for idx, step in enumerate(steps[::args.every]):
        meta = run_dir / f"fields_{step}_meta.txt"
        if not meta.exists(): continue
        t, Nr, Nz, Ng, Rmax, Zmax = load_meta(meta)

        vec = load_field(run_dir, step, args.field)
        if vec is None: continue
        F = reshape_crop(vec, Nr, Nz, Ng, crop_inner=1)
        # Clip estático global
        F = np.clip(F, vmin, vmax)

        im.set_data(F)
        txt.set_text(f"t={t*1e6:.2f} µs    step={step}")
        fig.canvas.draw()
        # buffer sin deprecación
        image = np.asarray(fig.canvas.buffer_rgba())
        if writer:
            writer.append_data(image)
    if writer: writer.close()

    # guarda también el último frame como PNG útil
    out_png = run_dir / f"{args.field}_last.png"
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)
    print("Saved:", out_png)
    if args.make_mp4:
        print("Saved:", run_dir / f"{args.field}_movie.mp4")

if __name__ == "__main__":
    main()

