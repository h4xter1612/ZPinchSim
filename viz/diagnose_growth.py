#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- I/O helpers --------------------
def load_meta(meta_path: Path):
    s = meta_path.read_text(encoding="utf-8")
    # robust regex
    t   = float(re.search(r"t=([0-9eE.\+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE.\+\-]+), Zmax=([0-9eE.\+\-]+)", s).groups())
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
    try:
        arr = np.loadtxt(p, delimiter=",", dtype=float)
    except Exception:
        # fallback: maybe single value (bad write) -> return None to skip
        return None
    return arr

def reshape_center(vec, Nr, Nz, Ng):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

# -------------------- z-projection helpers --------------------
def window_hann(n):
    if n <= 1:
        return np.ones(n)
    w = np.hanning(n)
    # unity RMS normalization to keep amplitudes comparable
    rms = np.sqrt((w*w).mean()) + 1e-30
    return w / rms

def proj_fft_z(F_rz, dz, k, jwin=0, demean=False, use_hann=False):
    """
    Proyecta en z a número de onda ~k usando FFT en cada r, luego promedia banda (±jwin)
    y devuelve el perfil radial |A_k(r)|. La integración volumétrica se hace aparte.
    """
    Nr, Nz = F_rz.shape
    # opcional de-mean por fila (en z)
    if demean:
        F_rz = F_rz - F_rz.mean(axis=1, keepdims=True)
    # ventana Hann opcional
    if use_hann:
        w = window_hann(Nz)
        F_rz = F_rz * w[None, :]

    # FFT y selección de bin
    freqs = np.fft.rfftfreq(Nz, d=dz)        # cycles/m
    ks = 2.0*np.pi*freqs                     # rad/m
    spec = np.fft.rfft(F_rz, axis=1)         # (Nr, Nz_r)
    j0 = int(np.argmin(np.abs(ks - k)))
    jL = max(0, j0 - int(jwin))
    jR = min(spec.shape[1]-1, j0 + int(jwin))
    band = spec[:, jL:jR+1]
    ak_r = np.linalg.norm(band, axis=1) / Nz  # energía de banda, normalizada
    return ak_r

def proj_direct_z(F_rz, dz, k, demean=False):
    """
    Proyección directa en z contra cos(k z). Devuelve perfil radial |A_k(r)|.
    La integración volumétrica se hace aparte.
    """
    Nr, Nz = F_rz.shape
    if demean:
        F_rz = F_rz - F_rz.mean(axis=1, keepdims=True)
    z = (np.arange(Nz) + 0.5) * dz
    cosk = np.cos(k * z)[None, :]
    ak_r = np.abs((F_rz * cosk).sum(axis=1)) * dz / Nz
    return ak_r

# -------------------- extractors --------------------
def Ak_time_series(run_dir: Path, field: str, mode: str, method: str,
                   k: float, r_pick: float, jwin: int,
                   apply_demean: bool, apply_hann: bool):
    steps = find_steps(run_dir)
    if not steps:
        return None, None
    t_list, A_list = [], []

    # meta del primer snapshot
    _, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr = Rmax / Nr
    dz = Zmax / Nz
    two_pi = 2.0*np.pi

    # índice radial objetivo si se usa centerline
    r_index = None
    if mode == "centerline":
        r_coords = (np.arange(Nr) + 0.5) * dr
        r_index = int(np.argmin(np.abs(r_coords - r_pick)))
        r_pick = float(r_coords[r_index])  # snap to grid

    for s in steps:
        meta = run_dir / f"fields_{s}_meta.txt"
        if not meta.exists():
            continue
        t, NrF, NzF, NgF, RmaxF, ZmaxF = load_meta(meta)
        vec = load_field(run_dir, s, field)
        if vec is None:
            continue

        try:
            F = reshape_center(vec, NrF, NzF, NgF)
        except Exception:
            # tamaño inesperado -> saltar
            continue

        if method == "fft":
            ak_r = proj_fft_z(F, dz, k, jwin=jwin, demean=apply_demean, use_hann=apply_hann)
        else:
            ak_r = proj_direct_z(F, dz, k, demean=apply_demean)

        # integrar o tomar línea
        if mode == "volume":
            r = (np.arange(Nr) + 0.5) * dr
            Ak = np.abs((ak_r * (two_pi * r)).sum()) * dr
        else:
            Ak = float(np.abs(ak_r[r_index]))

        t_list.append(t)
        A_list.append(Ak)

    if not t_list:
        return None, None
    return np.array(t_list), np.array(A_list)

# -------------------- fit helper (FIXED MASK LOGIC) --------------------
def fit_lnA(t, A, tmin=None, tmax=None):
    """
    Ajusta ln(A) = gamma * t + b con máscara CONSISTENTE.
    - Combina: finitud & A>0 & ventana de tiempo en UNA sola máscara.
    """
    t = np.asarray(t, float)
    A = np.asarray(A, float)

    ok = np.isfinite(t) & np.isfinite(A) & (A > 0.0)
    if tmin is not None:
        ok &= (t >= tmin)
    if tmax is not None:
        ok &= (t <= tmax)

    tt = t[ok]
    AA = A[ok]
    if tt.size < 3:
        return None

    y = np.log(AA)
    X = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    gamma, b = coef
    yhat = X @ coef
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return dict(gamma=gamma, tau=(1.0/gamma if gamma>0 else np.inf),
                intercept=b, r2=r2, tmin=(tt.min() if tt.size else None),
                tmax=(tt.max() if tt.size else None),
                tt=tt, lnA=y, lnA_fit=yhat)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--field", required=True, choices=["rho","p","vr","vz","Bth","Br","Bz","vth"])
    ap.add_argument("--mode", choices=["volume","centerline"], default="volume")
    ap.add_argument("--method", choices=["fft","direct"], default="fft")
    ap.add_argument("--k", type=float, default=2*math.pi*10.0, help="k [rad/m]")
    ap.add_argument("--jwin", type=int, default=0, help="±bins alrededor de k (solo fft)")
    ap.add_argument("--r", type=float, default=0.006, help="radio para centerline [m]")
    ap.add_argument("--demean-z", action="store_true")
    ap.add_argument("--hann", action="store_true")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    rundir = Path(args.run_dir)
    t, Ak = Ak_time_series(rundir, args.field, args.mode, args.method,
                           args.k, args.r, args.jwin,
                           args.demean_z, args.hann)
    if t is None or Ak is None or t.size == 0:
        raise SystemExit("[ERR] No se pudo obtener A_k(t).")

    # Fit robusto con máscara consistente
    fit = fit_lnA(t, Ak, tmin=args.tmin, tmax=args.tmax)
    if fit is None:
        print("[WARN] La ventana no es suficiente para ajustar ln A_k(t). Guardando A_k(t) sin fit.")
    else:
        print(f"gamma = {fit['gamma']:.3e}  [1/s]")
        print(f"tau   = {fit['tau']:.3e}  [s]")
        print(f"R^2   = {fit['r2']:.4f}")
        if fit['tmin'] is not None and fit['tmax'] is not None:
            print(f"Ventana: [{fit['tmin']:.3e}, {fit['tmax']:.3e}] s")

    # Guardados básicos
    out_csv = rundir / "diagnostics_Ak.csv"
    np.savetxt(out_csv, np.column_stack([t, Ak]), delimiter=",", header="t,Ak", comments="")
    print("Saved:", out_csv)

    # Plot ln A_k y ajuste (si existe)
    plt.figure(figsize=(6,3), dpi=args.dpi)
    # filtra solo A>0 para graficar ln
    pos = Ak > 0
    if np.any(pos):
        plt.plot(t[pos]*1e6, np.log(Ak[pos]), lw=1.2, label="ln A_k(t)")
    else:
        plt.plot(t*1e6, Ak, lw=1.2, label="A_k(t)")

    if fit is not None:
        plt.plot(fit["tt"]*1e6, fit["lnA_fit"], lw=1.8, ls="--", label=f"fit: γ={fit['gamma']:.2e} 1/s")

    plt.xlabel("t [µs]")
    plt.ylabel("ln A_k" if np.any(pos) else "A_k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = rundir / "diagnostics_lnA_fit.png"
    plt.savefig(out_png, dpi=args.dpi); plt.close()
    print("Saved:", out_png)

    # CSV con parámetros del ajuste (o NaN si no hubo ajuste)
    out_fit = rundir / "diagnostics_fit_gamma.csv"
    if fit is None:
        with out_fit.open("w", encoding="utf-8") as f:
            f.write("gamma,tau,intercept,R2,tmin,tmax\n")
            f.write("nan,nan,nan,nan,nan,nan\n")
    else:
        with out_fit.open("w", encoding="utf-8") as f:
            f.write("gamma,tau,intercept,R2,tmin,tmax\n")
            f.write(f"{fit['gamma']:.16e},{fit['tau']:.16e},{fit['intercept']:.16e},{fit['r2']:.6f},"
                    f"{fit['tmin']:.16e},{fit['tmax']:.16e}\n")
    print("Saved:", out_fit)

if __name__ == "__main__":
    main()

