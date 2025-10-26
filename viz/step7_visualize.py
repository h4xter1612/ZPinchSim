#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent

def load_meta(meta_path: Path):
    t = None; Nr=Nz=Ng=None; Rmax=Zmax=None
    s = meta_path.read_text(encoding="utf-8")
    m = re.search(r"t=([0-9eE\.\+\-]+)", s);          t = float(m.group(1)) if m else 0.0
    m = re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s); Nr, Nz, Ng = map(int, m.groups())
    m = re.search(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)", s); Rmax, Zmax = map(float, m.groups())
    return t, Nr, Nz, Ng, Rmax, Zmax

def load_field_csv(csv_path: Path):
    # vector 1D -> numpy 1D
    return np.loadtxt(csv_path, delimiter=",", dtype=float)

def reshape_crop(vec, Nr, Nz, Ng):
    NrT = Nr + 2*Ng; NzT = Nz + 2*Ng
    arr = np.array(vec).reshape(NrT, NzT)  # (i,k) = (r,z)
    return arr[Ng:Ng+Nr, Ng:Ng+Nz]         # recorte ghosts

def find_steps(run_dir: Path):
    # Detecta todos los snapshots basados en fields_*_meta.txt
    metas = sorted(run_dir.glob("fields_*_meta.txt"))
    steps = []
    for m in metas:
        step = int(m.stem.split("_")[1])   # fields_<step>_meta
        steps.append(step)
    return sorted(steps)

def main():
    ap = argparse.ArgumentParser(description="Step7 visualización: mapas 2D + video + amplitud")
    ap.add_argument("--run-dir", required=True, help="Carpeta del caso (contiene fields_*.csv)")
    ap.add_argument("--field", default="vr", choices=["vr","rho","p","Bth"], help="Campo a visualizar")
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--r-over-R", type=float, default=0.30)
    ap.add_argument("--k", type=float, default=62.8318530718)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    frames_dir = run_dir / f"frames_{args.field}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    steps = find_steps(run_dir)
    if len(steps) == 0:
        print("[WARN] No snapshots.")
        return

    # Cargar meta del primer snapshot para geometría
    t0, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    r = (np.arange(Nr) + 0.5) * (Rmax/Nr)
    z = (np.arange(Nz) + 0.5) * (Zmax/Nz)

    # Serie de amplitud (proyección 1D en z a un r/R dado)
    amp_t = []
    amp_vals = []

    # Render frames
    for step in steps:
        meta = run_dir / f"fields_{step}_meta.txt"
        t, Nr, Nz, Ng, Rmax, Zmax = load_meta(meta)

        # cargar campo
        base = run_dir / f"fields_{step}_{args.field}.csv"
        if not base.exists():  # si falta ese campo, saltar
            continue
        vec = load_field_csv(base)
        F = reshape_crop(vec, Nr, Nz, Ng)

        # proyección modal 1D en z a r/R≈args.r_over_R (nearest)
        ridx = int(min(max(round(args.r_over_R * Nr), 0), Nr-1))
        line = F[ridx, :]           # campo vs z
        # extraer componente a k usando FFT
        Lz = Zmax
        kz = np.fft.rfftfreq(Nz, d=Lz/Nz) * 2*np.pi
        spec = np.fft.rfft(line)
        # índice más cercano a k target
        kidx = int(np.argmin(np.abs(kz - args.k)))
        amp = 2.0*np.abs(spec[kidx]) / Nz  # amplitud simple
        amp_t.append(t); amp_vals.append(amp)

        # dibujar frame 2D
        plt.figure(figsize=(6,4))
        extent = [0, Zmax*1e3, 0, Rmax*1e3]  # mm
        plt.imshow(F, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("z [mm]"); plt.ylabel("r [mm]")
        plt.title(f"{args.field}  t={t*1e6:.2f} µs")
        cbar = plt.colorbar(); cbar.set_label(args.field)
        out_png = frames_dir / f"{args.field}_{step:06d}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        plt.close()

    # Curva de amplitud
    if len(amp_t) > 5:
        amp_t = np.array(amp_t); amp_vals = np.array(amp_vals)
        plt.figure()
        plt.semilogy(amp_t, amp_vals+1e-300)  # evitar log(0)
        plt.xlabel("t [s]"); plt.ylabel(f"|{args.field}|_k at r/R={args.r_over_R}")
        plt.title("Evolución de amplitud modal")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        out_csv = run_dir / f"amplitude_{args.field}_r{args.r_over_R:.2f}_k{args.k:.3f}.csv"
        np.savetxt(out_csv, np.c_[amp_t, amp_vals], delimiter=",", header="t,amp", comments="")
        out_png = run_dir / f"amplitude_{args.field}_r{args.r_over_R:.2f}_k{args.k:.3f}.png"
        plt.savefig(out_png, dpi=140)
        plt.close()
        print("Saved:", out_csv)
        print("Saved:", out_png)

    # Video opcional (MP4)
    if args.make_mp4:
        mp4_path = run_dir / f"{args.field}_movie.mp4"
        imgs = []
        for step in steps:
            p = frames_dir / f"{args.field}_{step:06d}.png"
            if p.exists():
                imgs.append(imageio.imread(p))
        if imgs:
            imageio.mimsave(mp4_path, imgs, fps=args.fps)
            print("Saved:", mp4_path)

if __name__ == "__main__":
    main()

