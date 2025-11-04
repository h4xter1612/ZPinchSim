#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- I/O helpers --------------------
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
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", dtype=float)

# -------------------- geometry & crop --------------------
def reshape_crop(vec, Nr, Nz, Ng, crop_inner=2):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.array(vec, dtype=float).reshape(NrT, NzT)
    # recorta ghosts y además 'crop_inner' celdas físicas internas
    i0 = Ng + crop_inner; i1 = Ng + Nr - crop_inner
    k0 = Ng + crop_inner; k1 = Ng + Nz - crop_inner
    return A[i0:i1, k0:k1], (i0, k0)

# -------------------- color limits (global static) --------------------
def compute_global_limits(run_dir, steps, field, Nr, Nz, Ng, pclip=(2.0, 98.0), sample_frames=30):
    vals = []
    take = max(1, len(steps)//sample_frames)
    for s in steps[::take]:
        vec = load_field(run_dir, s, field)
        if vec is None:
            continue
        F, _ = reshape_crop(vec, Nr, Nz, Ng, crop_inner=1)
        vals.append(F.ravel())
    if not vals:
        return None, None
    big = np.concatenate(vals)
    lo, hi = np.percentile(big, pclip)
    # centra en 0 si el campo es con signo
    if field.lower() in ("vr","vz","br","bz","bth","vth"):
        m = float(max(abs(lo), abs(hi)))
        return -m, m
    return float(lo), float(hi)

# -------------------- amplitude extractors --------------------
def nearest_index_r(i0, Nr, Rmax, dr, r_target):
    # i_global = i0 + i_local ; r = (i_global - Ng + 0.5)*dr
    # aquí i0 ya incluye Ng+crop_inner, por lo que basta:
    r_local = np.arange(Nr) + 0.5
    r_vals = r_local * dr + (i0 - 0)*0.0  # dr ya es físico
    # corrección exacta: r(i_global) = (i_global - Ng + 0.5)*dr
    # pero como trabajamos en recorte, reconstruimos i_global con i0
    i_global = i0 + np.arange(Nr)
    r_vals = (i_global - 0 + 0.5) * dr  # Ng ya se compensó en el meta con dr = Rmax/Nr_full
    j = int(np.argmin(np.abs(r_vals - r_target)))
    return j, r_vals[j]

def amplitude_maxabs(F_rz):
    return float(np.max(np.abs(F_rz)))

def amplitude_projk(F_rz_line, Zmax, k_target):
    # FFT en z y toma la componente más cercana a k_target
    Nz = F_rz_line.size
    dz = Zmax / Nz
    freqs = np.fft.rfftfreq(Nz, d=dz)      # [cycles/m]
    ks = 2.0*np.pi*freqs                   # [rad/m]
    spec = np.fft.rfft(F_rz_line)
    j = int(np.argmin(np.abs(ks - k_target)))
    return float(np.abs(spec[j]) / Nz)     # amplitud normalizada

# -------------------- main --------------------
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
    # amplitud
    ap.add_argument("--amp-mode", choices=["none","maxabs","projk"], default="none")
    ap.add_argument("--r", type=float, default=0.30, help="radio [m] para medir amplitud")
    ap.add_argument("--k", type=float, default=2*np.pi*10.0, help="k [rad/m] si amp-mode=projk")
    ap.add_argument("--crop-inner", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    steps = find_steps(run_dir)
    if not steps:
        print("[ERR] No hay snapshots.")
        return

    # meta del primer frame
    t0,Nr_full,Nz_full,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr = Rmax / Nr_full
    dz = Zmax / Nz_full

    # límites globales
    p1,p2 = [float(x) for x in args.pclip.split(",")]
    vmin, vmax = compute_global_limits(run_dir, steps, args.field,
                                       Nr_full, Nz_full, Ng, (p1,p2))
    if vmin is None:
        print("[ERR] No hay campos para pintar.")
        return

    # primer frame para dimensionar
    vec0 = load_field(run_dir, steps[0], args.field)
    F0, (i0crop, k0crop) = reshape_crop(vec0, Nr_full, Nz_full, Ng, crop_inner=args.crop_inner)
    Nr = F0.shape[0]; Nz = F0.shape[1]

    # figura (múltiplo de 16 px para evitar warnings de ffmpeg)
    W, H = 640, 400
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax  = fig.add_subplot(111)
    im  = ax.imshow(F0*0.0, origin="lower", aspect="auto",
                    interpolation="nearest", vmin=vmin, vmax=vmax)
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

    # preparación para amplitud
    amp_times = []
    amp_vals  = []
    r_index   = None
    r_used    = None

    # preconstruye coordenada z del recorte (para labels si quisieras)
    # z_local index -> z = (k_global - Ng + 0.5)*dz
    # k_global arranca en k0crop
    k_globals = k0crop + np.arange(Nz)
    z_vals = (k_globals - Ng + 0.5) * dz

    # recorre frames
    for step in steps[::args.every]:
        meta = run_dir / f"fields_{step}_meta.txt"
        if not meta.exists():
            continue
        t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)
        vec = load_field(run_dir, step, args.field)
        if vec is None:
            continue

        F, (i0, k0) = reshape_crop(vec, NrF, NzF, NgF, crop_inner=args.crop_inner)
        F = np.clip(F, vmin, vmax)
        im.set_data(F)
        txt.set_text(f"t={t*1e6:.2f} µs    step={step}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        if writer:
            writer.append_data(frame)

        # amplitud
        if args.amp_mode != "none":
            if r_index is None:
                # índice radial más cercano al r solicitado
                # r(i_global) = (i_global - Ng + 0.5) * dr
                # i_global del recorte: i0 + i_local
                i_globals = i0 + np.arange(F.shape[0])
                r_coords = (i_globals - Ng + 0.5) * dr
                r_index = int(np.argmin(np.abs(r_coords - args.r)))
                r_used  = float(r_coords[r_index])

            line = F[r_index, :]  # perfil en z a r≈r_used

            if args.amp_mode == "maxabs":
                amp = amplitude_maxabs(line)
            else:  # projk
                amp = amplitude_projk(line, ZmaxF * (Nz / NzF), args.k)

            amp_times.append(t)
            amp_vals.append(amp)

    if writer:
        writer.close()

    # guarda último frame
    out_png = run_dir / f"{args.field}_last.png"
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)
    print("Saved:", out_png)
    if args.make_mp4:
        print("Saved:", run_dir / f"{args.field}_movie.mp4")

    # guarda amplitud si aplica
    if args.amp_mode != "none" and amp_times:
        amp_times = np.array(amp_times)
        amp_vals  = np.array(amp_vals)
        if args.amp_mode == "maxabs":
            tag = f"r{r_used:.2f}_maxabs"
        else:
            tag = f"r{r_used:.2f}_k{args.k:.3f}"

        csv_path = run_dir / f"amplitude_{args.field}_{tag}.csv"
        np.savetxt(csv_path,
                   np.column_stack([amp_times, amp_vals]),
                   delimiter=",", header="t,amplitude", comments="")
        print("Saved:", csv_path)

        # plot rápido
        plt.figure(figsize=(6,3), dpi=args.dpi)
        plt.plot(amp_times*1e6, amp_vals, lw=1.5)
        plt.xlabel("t [µs]")
        plt.ylabel(f"|{args.field}| amplitude")
        plt.grid(True, alpha=0.3)
        png_path = run_dir / f"amplitude_{args.field}_{tag}.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=args.dpi)
        plt.close()
        print("Saved:", png_path)

if __name__ == "__main__":
    main()

