#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_meta(meta_path: Path):
    s = meta_path.read_text(encoding="utf-8")
    t   = float(re.search(r"t=([0-9eE\.\+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)", s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

def load_field(rd: Path, step: int, name: str):
    return np.loadtxt(rd/f"fields_{step}_{name}.csv", delimiter=",")

def reshape_center(vec, Nr, Nz, Ng):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--field", required=True, choices=["rho","p","vr","vz","Bth","Br","Bz"])
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--step", default="last", help="número o 'last'")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    steps = sorted({int(p.stem.split("_")[1]) for p in rd.glob("fields_*_meta.txt")})
    if not steps: raise SystemExit("No snapshots.")
    if args.step=="last": step = steps[-1]
    else: step = int(args.step)

    t,Nr,Nz,Ng,Rmax,Zmax = load_meta(rd/f"fields_{step}_meta.txt")
    F = reshape_center(load_field(rd, step, args.field), Nr, Nz, Ng)
    dr, dz = Rmax/Nr, Zmax/Nz
    r = (np.arange(Nr)+0.5)*dr
    j = int(np.argmin(np.abs(r - args.r)))
    line = F[j,:] - F[j,:].mean()

    freqs = np.fft.rfftfreq(Nz, d=dz)      # cycles/m
    ks = 2.0*np.pi*freqs                    # rad/m
    spec = np.abs(np.fft.rfft(line))/Nz

    plt.figure(figsize=(6,3), dpi=args.dpi)
    plt.plot(ks, spec, lw=1.5)
    plt.xlabel("k [rad/m]"); plt.ylabel("|F_k|"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = rd / f"spectrum_{args.field}_r{args.r:.4f}_step{step}.png"
    plt.savefig(out, dpi=args.dpi); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()

