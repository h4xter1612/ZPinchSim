#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU0 = 4e-7*np.pi

def load_meta(meta_path: Path):
    s = meta_path.read_text(encoding="utf-8")
    t   = float(re.search(r"t=([0-9eE.+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+),\s*Nz=(\d+),\s*Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE.+\-]+),\s*Zmax=([0-9eE.+\-]+)", s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

def find_steps(run_dir: Path):
    out = []
    for p in run_dir.glob("fields_*_meta.txt"):
        try: out.append(int(p.stem.split("_")[1]))
        except: pass
    return sorted(set(out))

def load_field(run_dir: Path, step: int, field: str, Nr=None, Nz=None, Ng=None):
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists(): return None
    arr = np.loadtxt(p, delimiter=",", dtype=float)
    if arr.ndim==0 or arr.size<=1: return None
    if Nr is None or Nz is None or Ng is None:
        return arr
    try:
        NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
        _ = arr.reshape(NrT, NzT)  # verifica
    except Exception:
        return None
    return arr

def center(vec, Nr, Nz, Ng):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    return np.asarray(vec, float).reshape(NrT, NzT)[Ng:Ng+Nr, Ng:Ng+Nz]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gamma", type=float, default=5.0/3.0)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    steps = find_steps(rd)
    if not steps: raise SystemExit("[ERR] no snapshots")

    t0,Nr,Nz,Ng,Rmax,Zmax = load_meta(rd / f"fields_{steps[0]}_meta.txt")
    dr=Rmax/Nr; dz=Zmax/Nz
    r = (np.arange(Nr)+0.5)*dr
    w_r = 2.0*np.pi*r  # peso cilindrico

    T=[]; Ek=[]; EBth=[]; EBp=[]; Ep=[]

    for s in steps:
        meta = rd / f"fields_{s}_meta.txt"
        if not meta.exists(): continue
        try:
            t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)
        except Exception:
            continue
        # carga segura (si falla algo → None)
        rho = load_field(rd,s,"rho",NrF,NzF,NgF)
        p   = load_field(rd,s,"p",  NrF,NzF,NgF)
        vr  = load_field(rd,s,"vr", NrF,NzF,NgF)
        vz  = load_field(rd,s,"vz", NrF,NzF,NgF)
        vth = load_field(rd,s,"vth",NrF,NzF,NgF)
        Bth = load_field(rd,s,"Bth",NrF,NzF,NgF)
        Br  = load_field(rd,s,"Br", NrF,NzF,NgF)
        Bz  = load_field(rd,s,"Bz", NrF,NzF,NgF)

        # mínimos indispensables
        if rho is None or p is None:
            continue

        # centra
        rho = center(rho,NrF,NzF,NgF)
        p   = center(p,  NrF,NzF,NgF)
        vr  = center(vr,NrF,NzF,NgF)  if vr  is not None else 0.0
        vz  = center(vz,NrF,NzF,NgF)  if vz  is not None else 0.0
        vth = center(vth,NrF,NzF,NgF) if vth is not None else 0.0
        Bth = center(Bth,NrF,NzF,NgF) if Bth is not None else 0.0
        Br  = center(Br,NrF,NzF,NgF)  if Br  is not None else 0.0
        Bz  = center(Bz,NrF,NzF,NgF)  if Bz  is not None else 0.0

        # energías densidad
        v2 = (0 if np.isscalar(vr) else vr**2) + (0 if np.isscalar(vz) else vz**2) + (0 if np.isscalar(vth) else vth**2)
        ek = 0.5 * rho * v2
        ebth = (0 if np.isscalar(Bth) else Bth**2)/(2*MU0)
        ebp  = (0 if np.isscalar(Br)  else Br**2 + (0 if np.isscalar(Bz) else Bz**2))/(2*MU0)
        ep   = p/(args.gamma - 1.0)

        # integra ∫∫ (...) 2π r dr dz
        def cyl_int(Q):
            if np.isscalar(Q): return 0.0
            q_r = (Q * w_r[:,None]).sum(axis=0) * dr
            return float(q_r.sum() * dz)

        T.append(t)
        Ek.append(cyl_int(ek))
        EBth.append(cyl_int(ebth))
        EBp.append(cyl_int(ebp))
        Ep.append(cyl_int(ep))

    if not T: raise SystemExit("[ERR] no se pudieron integrar energías en ningún frame")

    T = np.array(T); order = np.argsort(T)
    T,Ek,EBth,EBp,Ep = T[order], np.array(Ek)[order], np.array(EBth)[order], np.array(EBp)[order], np.array(Ep)[order]
    Etot = Ek + EBth + EBp + Ep

    # CSV
    out_csv = rd / "diagnostics_energy.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t,Ek,EBth,EBp,Ep,Etot\n")
        for i in range(T.size):
            f.write(f"{T[i]:.9e},{Ek[i]:.9e},{EBth[i]:.9e},{EBp[i]:.9e},{Ep[i]:.9e},{Etot[i]:.9e}\n")
    print("Saved:", out_csv)

    # Plots
    for name, Y in [("Ek",Ek), ("EBth",EBth), ("EBp",EBp), ("Ep",Ep), ("Etot",Etot)]:
        plt.figure(figsize=(6,3), dpi=args.dpi)
        plt.plot(T*1e6, Y, lw=1.6)
        plt.xlabel("t [µs]"); plt.ylabel(name)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(rd / f"diagnostics_{name}.png", dpi=args.dpi); plt.close()

if __name__ == "__main__":
    main()

