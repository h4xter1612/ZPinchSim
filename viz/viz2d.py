#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_meta(meta_path: Path):
    s = meta_path.read_text(encoding="utf-8")
    t   = float(re.search(r"t=([0-9eE\.\+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)", s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

def find_steps(run_dir: Path):
    steps = []
    for p in run_dir.glob("fields_*_meta.txt"):
        try:
            steps.append(int(p.stem.split("_")[1]))
        except Exception:
            pass
    return sorted(set(steps))

def load_field(run_dir: Path, step: int, field: str):
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", dtype=float)

def reshape_crop(vec, Nr, Nz, Ng, crop_inner=1):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.array(vec, dtype=float).reshape(NrT, NzT)
    i0 = Ng + crop_inner; i1 = Ng + Nr - crop_inner
    k0 = Ng + crop_inner; k1 = Ng + Nz - crop_inner
    if not (0 <= i0 < i1 <= NrT and 0 <= k0 < k1 <= NzT):
        # fallback sin recorte si parámetros extremos
        i0, i1, k0, k1 = Ng, Ng+Nr, Ng, Ng+Nz
    return A[i0:i1, k0:k1], (i0, k0)

def compute_global_limits(run_dir, steps, field, Nr, Nz, Ng, pclip=(2.0, 98.0), sample_frames=30, symmetric_auto=False):
    vals = []
    take = max(1, len(steps)//sample_frames)
    for s in steps[::take]:
        vec = load_field(run_dir, s, field)
        if vec is None: 
            continue
        F, _ = reshape_crop(vec, Nr, Nz, Ng, crop_inner=1)
        if np.any(~np.isfinite(F)): 
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        vals.append(F.ravel())
    if not vals:
        return None, None
    big = np.concatenate(vals)
    lo, hi = np.percentile(big, pclip)
    if symmetric_auto:
        m = float(max(abs(lo), abs(hi)))
        return -m, m
    return float(lo), float(hi)

def amplitude_maxabs(line):
    return float(np.nanmax(np.abs(line)))

def amplitude_projk(line, dz, k_target):
    Nz = line.size
    freqs = np.fft.rfftfreq(Nz, d=dz)   # cycles/m
    ks = 2.0*np.pi*freqs                # rad/m
    spec = np.fft.rfft(line)
    j = int(np.argmin(np.abs(ks - k_target)))
    return float(np.abs(spec[j]) / Nz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--field", required=True,
                    choices=["rho","p","vr","vz","Bth","Br","Bz","vth"])
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--pclip", type=str, default="2,98")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--crop-inner", type=int, default=1)
    ap.add_argument("--sym", action="store_true", help="límites simétricos (campos con signo)")
    ap.add_argument("--log1p", action="store_true", help="usar log1p(|F|)*sign(F) para mostrar")
    # Amplitud
    ap.add_argument("--amp-mode", choices=["none","maxabs","projk"], default="none")
    ap.add_argument("--r", type=float, default=0.30, help="radio [m] para amplitud (línea en z)")
    ap.add_argument("--k", type=float, default=2*np.pi*10.0, help="k [rad/m] si amp-mode=projk")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    steps = find_steps(run_dir)
    if not steps:
        raise SystemExit("[ERR] No hay snapshots (fields_*_meta.txt)")

    # meta base
    t0,Nr_full,Nz_full,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr = Rmax / Nr_full
    dz = Zmax / Nz_full

    # límites globales
    p1,p2 = [float(x) for x in args.pclip.split(",")]
    vmin, vmax = compute_global_limits(run_dir, steps, args.field,
                                       Nr_full, Nz_full, Ng, (p1,p2),
                                       symmetric_auto=args.sym)
    if vmin is None:
        raise SystemExit("[ERR] No se pudieron calcular límites de color.")

    # primer frame
    vec0 = load_field(run_dir, steps[0], args.field)
    if vec0 is None:
        raise SystemExit(f"[ERR] No existe {args.field} en step {steps[0]}")
    F0, (i0crop, k0crop) = reshape_crop(vec0, Nr_full, Nz_full, Ng, crop_inner=args.crop_inner)
    if args.log1p:
        F0 = np.sign(F0) * np.log1p(np.abs(F0))

    Nr = F0.shape[0]; Nz = F0.shape[1]
    # coordenadas para labels
    i_globals = i0crop + np.arange(Nr)
    k_globals = k0crop + np.arange(Nz)
    r_vals = (i_globals - Ng + 0.5) * dr
    z_vals = (k_globals - Ng + 0.5) * dz

    # figura
    W, H = 640, 400  # múltiplos de 16 px para video
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax  = fig.add_subplot(111)
    im  = ax.imshow(F0*0.0, origin="lower", aspect="auto",
                    interpolation="nearest", vmin=vmin, vmax=vmax,
                    extent=[z_vals[0], z_vals[-1], r_vals[0], r_vals[-1]])
    cb  = fig.colorbar(im); cb.set_label(args.field)
    ax.set_xlabel("z [m]"); ax.set_ylabel("r [m]")
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, ha="left", va="top",
                  color="w", fontsize=9,
                  bbox=dict(facecolor="0.0", alpha=0.35, pad=2, edgecolor="none"))
    fig.tight_layout()

    writer = None
    if args.make_mp4:
        writer = imageio.get_writer(run_dir / f"{args.field}_movie.mp4",
                                    fps=args.fps, macro_block_size=None)

    # amplitud
    amp_times, amp_vals = [], []
    r_index = None
    r_used = None

    # loop de frames
    for step in steps[::args.every]:
        meta = run_dir / f"fields_{step}_meta.txt"
        if not meta.exists(): 
            continue
        t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)

        vec = load_field(run_dir, step, args.field)
        if vec is None:
            continue

        F, (i0, k0) = reshape_crop(vec, NrF, NzF, NgF, crop_inner=args.crop_inner)
        if args.log1p:
            F = np.sign(F) * np.log1p(np.abs(F))
        F = np.clip(F, vmin, vmax)

        # extents (por si cambian Z/NZ — normal no debería)
        i_globals = i0 + np.arange(F.shape[0])
        k_globals = k0 + np.arange(F.shape[1])
        r_vals = (i_globals - Ng + 0.5) * dr
        z_vals = (k_globals - Ng + 0.5) * dz

        im.set_data(F)
        im.set_extent([z_vals[0], z_vals[-1], r_vals[0], r_vals[-1]])
        txt.set_text(f"t={t*1e6:.2f} µs    step={step}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        if writer:
            writer.append_data(frame)

        if args.amp_mode != "none":
            if r_index is None:
                i_globals0 = i0 + np.arange(F.shape[0])
                r_coords0 = (i_globals0 - Ng + 0.5) * dr
                r_index = int(np.argmin(np.abs(r_coords0 - args.r)))
                r_used  = float(r_coords0[r_index])
            line = F[r_index, :]
            if args.amp_mode == "maxabs":
                amp = amplitude_maxabs(line)
            else:
                dz_local = (z_vals[-1] - z_vals[0]) / F.shape[1]
                amp = amplitude_projk(line, dz_local, args.k)
            amp_times.append(t); amp_vals.append(amp)

    # guardados
    out_png = run_dir / f"{args.field}_last.png"
    fig.savefig(out_png, dpi=args.dpi)
    if writer: 
        writer.close()
    plt.close(fig)
    print("Saved:", out_png)
    if writer:
        print("Saved:", run_dir / f"{args.field}_movie.mp4")

    if args.amp_mode != "none" and amp_times:
        amp_times = np.array(amp_times); amp_vals = np.array(amp_vals)
        tag = "maxabs" if args.amp_mode=="maxabs" else f"k{args.k:.3f}"
        tag = f"r{r_used:.3f}_{tag}"
        csv_path = run_dir / f"amplitude_{args.field}_{tag}.csv"
        np.savetxt(csv_path, np.column_stack([amp_times, amp_vals]),
                   delimiter=",", header="t,amplitude", comments="")
        print("Saved:", csv_path)
        # plot rápido
        plt.figure(figsize=(6,3), dpi=args.dpi)
        plt.plot(amp_times*1e6, amp_vals, lw=1.5)
        plt.xlabel("t [µs]"); plt.ylabel(f"|{args.field}| amplitude")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        png_path = run_dir / f"amplitude_{args.field}_{tag}.png"
        plt.savefig(png_path, dpi=args.dpi); plt.close()
        print("Saved:", png_path)

if __name__ == "__main__":
    main()

