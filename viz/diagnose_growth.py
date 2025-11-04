#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, re
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

def reshape_center(vec, Nr, Nz, Ng):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.array(vec, dtype=float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

def Ak_from_metrics(csv_path: Path):
    t = []
    ak = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if not header:
            return None, None
        # buscamos la columna Ak (exacto) o similar
        try:
            j_t = header.index("t")
            j_ak = header.index("Ak")
        except ValueError:
            return None, None
        for row in rd:
            try:
                t.append(float(row[j_t])); ak.append(float(row[j_ak]))
            except Exception:
                pass
    if not t:
        return None, None
    return np.array(t), np.array(ak)

def Ak_from_snapshots(run_dir: Path, from_field: str, k: float):
    steps = find_steps(run_dir)
    if not steps:
        return None, None
    t_list, A_list = [], []
    # meta (constante)
    t0,Nr,Nz,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr = Rmax / Nr
    dz = Zmax / Nz
    # integración en r del cos(k z) con peso 2π r
    two_pi = 2.0*np.pi
    for s in steps:
        meta = run_dir / f"fields_{s}_meta.txt"
        if not meta.exists():
            continue
        t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)
        vec = load_field(run_dir, s, from_field)
        if vec is None: 
            continue
        F = reshape_center(vec, NrF, NzF, NgF)
        # proyección en z
        z = (np.arange(Nz) + 0.5) * dz
        cosk = np.cos(k * z)[None, :]  # (1, Nz)
        proj_z = (F * cosk).sum(axis=1) * dz   # (Nr,)
        # integral en r con peso 2π r
        r = (np.arange(Nr) + 0.5) * dr
        Ak = np.abs((proj_z * (two_pi * r)).sum() * dr)
        t_list.append(t); A_list.append(Ak)
    if not t_list:
        return None, None
    return np.array(t_list), np.array(A_list)

def fit_lnA(t, A, tmin=None, tmax=None):
    idx = np.ones_like(t, dtype=bool)
    if tmin is not None: idx &= (t>=tmin)
    if tmax is not None: idx &= (t<=tmax)
    tt = t[idx]; AA = A[idx]
    # filtro valores no válidos
    tt = tt[np.isfinite(AA) & (AA>0.0)]
    AA = AA[np.isfinite(AA) & (AA>0.0)]
    if tt.size < 3:
        return None
    y = np.log(AA)
    # regresión lineal
    X = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    gamma, b = coef
    yhat = X @ coef
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return dict(gamma=gamma, tau=(1.0/gamma if gamma>0 else np.inf),
                intercept=b, r2=r2, tmin=(tt.min() if tt.size else None), tmax=(tt.max() if tt.size else None),
                tt=tt, lnA=y, lnA_fit=yhat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=float, default=2*np.pi*10.0, help="k [rad/m] si hay que reconstruir Ak")
    ap.add_argument("--from", dest="from_field", choices=["rho","p"], default="rho", help="campo base para Ak si no existe en métricas")
    ap.add_argument("--tmin", type=float, default=None, help="inicio de ventana [s]")
    ap.add_argument("--tmax", type=float, default=None, help="fin de ventana [s]")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    rundir = Path(args.run_dir)
    metrics = rundir / "debug/2d_mhd_metrics.csv"

    if metrics.exists():
        t, Ak = Ak_from_metrics(metrics)
    else:
        t, Ak = None, None

    if t is None or Ak is None:
        print("[warn] No Ak en métricas; reconstruyendo desde snapshots…")
        t, Ak = Ak_from_snapshots(rundir, args.from_field, args.k)
    if t is None or Ak is None:
        raise SystemExit("[ERR] No se pudo obtener A_k(t).")

    fit = fit_lnA(t, Ak, tmin=args.tmin, tmax=args.tmax)
    if fit is None:
        raise SystemExit("[ERR] Ventana demasiado corta o datos inválidos para el ajuste.")

    # Reporte
    print(f"gamma = {fit['gamma']:.3e}  [1/s]")
    print(f"tau   = {fit['tau']:.3e}  [s]")
    print(f"R^2   = {fit['r2']:.4f}")
    if fit['tmin'] is not None and fit['tmax'] is not None:
        print(f"Ventana: [{fit['tmin']:.3e}, {fit['tmax']:.3e}] s")

    # Guardados
    # 1) CSV con Ak(t)
    out_csv = rundir / "diagnostics_Ak.csv"
    np.savetxt(out_csv, np.column_stack([t, Ak]), delimiter=",", header="t,Ak", comments="")
    print("Saved:", out_csv)

    # 2) PNG con ln A y ajuste
    plt.figure(figsize=(6,3), dpi=args.dpi)
    plt.plot(t*1e6, np.log(Ak), lw=1.2, label="ln A_k(t)")
    plt.plot(fit["tt"]*1e6, fit["lnA_fit"], lw=1.8, ls="--", label=f"fit: γ={fit['gamma']:.2e} 1/s")
    plt.xlabel("t [µs]"); plt.ylabel("ln A_k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = rundir / "diagnostics_lnA_fit.png"
    plt.savefig(out_png, dpi=args.dpi); plt.close()
    print("Saved:", out_png)

    # 3) CSV con parámetros del ajuste
    out_fit = rundir / "diagnostics_fit_gamma.csv"
    with out_fit.open("w", encoding="utf-8") as f:
        f.write("gamma,tau,intercept,R2,tmin,tmax\n")
        f.write(f"{fit['gamma']:.16e},{fit['tau']:.16e},{fit['intercept']:.16e},{fit['r2']:.6f},"
                f"{(fit['tmin'] if fit['tmin'] is not None else np.nan)},"
                f"{(fit['tmax'] if fit['tmax'] is not None else np.nan)}\n")
    print("Saved:", out_fit)

if __name__ == "__main__":
    main()

