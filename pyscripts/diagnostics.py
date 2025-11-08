#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics.py — Analysis pipeline for 2D runs (e.g., Z-pinch/MHD).

Key features:
- Computes A_k(t) using FFT or direct projection (volume or centerline).
- Auto-detects dominant axial wavenumber k from first snapshot (if --k=0).
- Saves time-series CSVs and ln(A_k) fit plot; reports magnetic/kinetic/internal energies.
- Extended help appears automatically when no arguments are provided.

Reminder: run `python utils.py` first to ensure dependencies are installed.
"""
from __future__ import annotations
import argparse, re, json, functools, sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Precompiled regex (faster metadata parsing) ----------
_RE_T     = re.compile(r"t=([0-9eE\.\+\-]+)")
_RE_NRNZ  = re.compile(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)")
_RE_RZMAX = re.compile(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)")

def _parse_meta_text(s: str) -> Tuple[float,int,int,int,float,float]:
    """Parse meta text into (t, Nr, Nz, Ng, Rmax, Zmax)."""
    t   = float(_RE_T.search(s).group(1))
    Nr,Nz,Ng = map(int, _RE_NRNZ.search(s).groups())
    Rmax,Zmax= map(float, _RE_RZMAX.search(s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

@functools.lru_cache(maxsize=2048)
def load_meta(meta_path: Path) -> Tuple[float,int,int,int,float,float]:
    """Load and parse a single meta file, with LRU caching."""
    s = meta_path.read_text(encoding="utf-8")
    return _parse_meta_text(s)

def list_steps(run_dir: Path) -> List[int]:
    """Return sorted integer step indices found in run_dir."""
    steps=[]
    for p in run_dir.glob("fields_*_meta.txt"):
        try:
            steps.append(int(p.stem.split("_")[1]))
        except Exception:
            pass
    steps.sort()
    return steps

def load_field(run_dir: Path, step: int, field: str) -> Optional[np.ndarray]:
    """Load a CSV field for a given step; return None if missing."""
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",", dtype=float)

def reshape_center(vec: np.ndarray, Nr: int, Nz: int, Ng: int) -> np.ndarray:
    """Strip guard cells and return the center domain (Nr x Nz)."""
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

# ------------------ k-space projections ------------------
def hann(n: int) -> np.ndarray:
    """Hann window with guard for n<=1."""
    if n <= 1:
        return np.ones((n,), float)
    i = np.arange(n, dtype=float)
    return 0.5 - 0.5*np.cos(2*np.pi*i/(n-1))

def pick_k_fft(z: np.ndarray, qz: np.ndarray, k_hint: Optional[float]) -> Tuple[np.ndarray,np.ndarray,int]:
    """Return (k_grid, amplitude_spectrum, index_of_peak_or_hint)."""
    dz = z[1] - z[0]
    Lz = z[-1] - z[0] + dz
    w  = hann(len(z))
    x  = (qz - qz.mean()) * w
    Qk = np.fft.rfft(x)
    dk = 2.0*np.pi / Lz
    k_grid = np.arange(Qk.size, dtype=float) * dk
    amp = np.abs(Qk)
    if not k_hint or k_hint <= 0.0:
        j_star = int(np.argmax(amp[1:]))+1 if amp.size>1 else 0
    else:
        j_star = int(np.argmin(np.abs(k_grid - k_hint)))
    return k_grid, amp, j_star

def Ak_volume(r, z, Qrz, k, method="fft", demean=True) -> float:
    """Volume-weighted A_k using FFT (default) or direct cosine projection."""
    Nr,_ = Qrz.shape
    if method=="direct":
        proj = np.trapz(Qrz * np.cos(k*z[np.newaxis,:]), z, axis=1)
        Ak = np.abs(np.trapz(np.abs(proj) * (2.0*np.pi*r), r))
        return float(Ak)
    amps=[]
    for i in range(Nr):
        qz = Qrz[i,:]
        if demean: qz = qz - qz.mean()
        _, A, j = pick_k_fft(z, qz, k)
        amps.append(A[j])
    Ak = np.trapz(np.asarray(amps) * (2.0*np.pi*r), r)
    return float(Ak)

def Ak_centerline(z, qz, k, method="fft", demean=True) -> float:
    """Centerline A_k at chosen radius index (or r given upstream)."""
    if demean: qz = qz - qz.mean()
    if method=="direct":
        return float(np.abs(np.trapz(qz*np.cos(k*z), z)))
    _, A, j = pick_k_fft(z, qz, k)
    return float(A[j])

# ------------------ ln(A) fit ------------------
def fit_lnA(t, A, tmin=None, tmax=None) -> Optional[Dict[str,float]]:
    """Least-squares linear fit to ln(A) over [tmin, tmax]."""
    t = np.asarray(t, float); A = np.asarray(A, float)
    ok = np.isfinite(t) & np.isfinite(A) & (A>0.0)
    if ok.sum() < 5: 
        return None
    t  = t[ok]; A = A[ok]
    if tmin is None: tmin = t[0] + 0.15*(t[-1]-t[0])
    if tmax is None: tmax = t[0] + 0.75*(t[-1]-t[0])
    msk = (t>=tmin) & (t<=tmax)
    if msk.sum() < 3:
        return None
    tt = t[msk]; y = np.log(A[msk])
    X = np.vstack([tt, np.ones_like(tt)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    gamma = beta[0]; b = beta[1]
    yhat = X@beta
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y-y.mean())**2) + 1e-30)
    R2 = 1.0 - ss_res/ss_tot
    return dict(gamma=float(gamma), tau=(1.0/gamma if gamma!=0 else np.inf),
                R2=float(R2), tmin=float(tt[0]), tmax=float(tt[-1]), beta0=float(b))

# ------------------ per-run pipeline ------------------
def run_diagnostics_one(
    run_dir: Path, field: str, mode: str, r_center: Optional[float],
    method: str, k_user: float, demean_z: bool,
    tmin: Optional[float], tmax: Optional[float], save_spectrum: bool,
    gamma_gas: float, mu0: float, save_timeseries: bool
) -> Dict[str, Any]:
    """Process a single run directory and return a summary dict."""
    steps = list_steps(run_dir)
    if not steps: 
        raise SystemExit(f"[ERR] No snapshots in {run_dir}")

    t0,Nr,Nz,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr, dz = Rmax/Nr, Zmax/Nz
    r = (np.arange(Nr)+0.5)*dr
    z = (np.arange(Nz)+0.5)*dz
    r_cl = r_center if (r_center is not None) else (0.30*Rmax)
    w_r = 2.0*np.pi*r

    T, Ak_series = [], []
    EB, EK, Eint, Erms = [], [], [], []
    k_detected_once: Optional[float] = None
    spectrum_written = False

    for s in steps:
        meta = run_dir / f"fields_{s}_meta.txt"
        if not meta.exists(): 
            continue
        t,NrS,NzS,NgS,RmaxS,ZmaxS = load_meta(meta)
        if (NrS,NzS,NgS)!=(Nr,Nz,Ng) or (RmaxS!=Rmax) or (ZmaxS!=Zmax):
            Nr,Nz,Ng,Rmax,Zmax = NrS,NzS,NgS,RmaxS,ZmaxS
            dr, dz = Rmax/Nr, Zmax/Nz
            r = (np.arange(Nr)+0.5)*dr
            z = (np.arange(Nz)+0.5)*dz
            w_r = 2.0*np.pi*r
            r_cl = r_center if (r_center is not None) else (0.30*Rmax)

        Q = load_field(run_dir, s, field)
        if Q is None: 
            continue
        Qrz = reshape_center(Q, Nr, Nz, Ng)

        # k selection (auto FFT if user provided k<=0)
        k_use = k_user
        if k_use<=0.0:
            qz_mean = np.average(Qrz, axis=0, weights=w_r)
            k_grid, amp, j = pick_k_fft(z, qz_mean, None)
            k_use = float(k_grid[j])
            if k_detected_once is None: 
                k_detected_once = float(k_use)
            if save_spectrum and not spectrum_written:
                np.savetxt(run_dir / "diagnostics_spectrum_k.csv",
                           np.c_[k_grid, amp], delimiter=",",
                           header="k,amp", comments="")
                spectrum_written = True

        # A_k (volume or centerline)
        if mode=="centerline":
            i = int(np.clip(np.round(r_cl/dr - 0.5), 0, Nr-1))
            qz = Qrz[i,:]
            Ak = Ak_centerline(z, qz, k_use, method=method, demean=demean_z)
        else:
            Ak = Ak_volume(r, z, Qrz, k_use, method=method, demean=demean_z)

        # Volume RMS for field (demean along z if requested)
        Qdm = Qrz - Qrz.mean(axis=1, keepdims=True) if demean_z else Qrz
        Q2_r = np.trapz(Qdm**2, z, axis=1)
        vol_Q2 = np.trapz(Q2_r * w_r, r)
        Vcol = (2.0*np.pi) * (0.5*(Rmax**2)) * (Zmax)
        rms_vol = np.sqrt(max(vol_Q2, 0.0) / max(Vcol, 1e-30))

        # Energies
        Emag = np.nan
        Bth = load_field(run_dir, s, "Bth")
        Bz  = load_field(run_dir, s, "Bz")
        if (Bth is not None) and (Bz is not None):
            Bth = reshape_center(Bth, Nr, Nz, Ng)
            Bz  = reshape_center(Bz,  Nr, Nz, Ng)
            UB  = 0.5*(Bth**2 + Bz**2)/mu0
            Emag = float(np.trapz(np.trapz(UB, z, axis=1) * w_r, r))

        Ekin = np.nan
        rho = load_field(run_dir, s, "rho")
        vr  = load_field(run_dir, s, "vr")
        if (rho is not None) and (vr is not None):
            rho = reshape_center(rho, Nr, Nz, Ng)
            vr  = reshape_center(vr,  Nr, Nz, Ng)
            UK  = 0.5*rho*(vr**2)
            Ekin = float(np.trapz(np.trapz(UK, z, axis=1) * w_r, r))

        Eint_i = np.nan
        p = load_field(run_dir, s, "p")
        if p is not None:
            p = reshape_center(p, Nr, Nz, Ng)
            Uint = p/(gamma_gas-1.0)
            Eint_i = float(np.trapz(np.trapz(Uint, z, axis=1) * w_r, r))

        T.append(t); Ak_series.append(Ak)
        EB.append(Emag); EK.append(Ekin); Eint.append(Eint_i); Erms.append(rms_vol)

    # Save outputs
    T = np.asarray(T, float); Ak_series = np.asarray(Ak_series, float)
    EB = np.asarray(EB); EK = np.asarray(EK); Eint = np.asarray(Eint); Erms = np.asarray(Erms)

    out_Ak = run_dir / "diagnostics_Ak.csv"
    np.savetxt(out_Ak, np.c_[T, Ak_series], delimiter=",", header="t,A_k", comments="")
    out_E = run_dir / "diagnostics_energy.csv"
    np.savetxt(out_E, np.c_[T, EB, EK, Eint, Erms], delimiter=",",
               header="t,EB,EK,Eint,RMS_field", comments="")

    fit = fit_lnA(T, Ak_series, tmin=tmin, tmax=tmax)

    out_fit = None
    if fit is not None:
        gamma = fit["gamma"]; R2 = fit["R2"]; tau = fit["tau"]
        out_fit = run_dir / "diagnostics_fit_gamma.csv"
        with open(out_fit, "w", encoding="utf-8") as f:
            f.write("tmin,tmax,gamma,tau,R2\n")
            f.write(f"{fit['tmin']:.16e},{fit['tmax']:.16e},{gamma:.16e},{tau:.16e},{R2:.6f}\n")
        ok = np.isfinite(T) & np.isfinite(Ak_series) & (Ak_series>0.0)
        tt = T[ok]; yy = np.log(Ak_series[ok])
        fig = plt.figure(figsize=(7,4), dpi=140); ax=fig.add_subplot(111)
        ax.plot(tt*1e6, yy, ".", ms=3, label="ln A_k(t)")
        msk=(tt>=fit["tmin"])&(tt<=fit["tmax"]); tx=tt[msk]
        ax.plot(tx*1e6, (gamma*tx + fit["beta0"]), "-", lw=2, label=f"fit γ={gamma:.2e} s⁻¹, R²={R2:.3f}")
        ax.set_xlabel("t [µs]"); ax.set_ylabel("ln A_k")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "diagnostics_lnA_fit.png", bbox_inches="tight"); plt.close(fig)

    # JSON summary
    summary = {
        "run_dir": str(run_dir),
        "k_peak": float(k_detected_once) if k_detected_once is not None else None,
        "paths": {
            "Ak_csv": str(out_Ak),
            "energy_csv": str(out_E),
            "fit_csv": (str(out_fit) if out_fit else None),
            "fit_png": (str((run_dir / "diagnostics_lnA_fit.png")) if out_fit else None),
            "spectrum_csv": (str(run_dir / "diagnostics_spectrum_k.csv")
                             if (run_dir / "diagnostics_spectrum_k.csv").exists() else None),
        }
    }
    if fit is not None:
        summary["fit"] = {k: (float(v) if np.isfinite(v) else None) for k,v in fit.items()}

    with open(run_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    def last_finite(x: np.ndarray) -> float:
        m = np.isfinite(x)
        return float(x[m][-1]) if m.any() else float('nan')

    print("\n====================== ZPINCH DIAGNOSTICS ======================")
    print(f"run-dir: {run_dir}")
    if k_detected_once is not None:
        print(f"k_peak (FFT, first sample): {k_detected_once:.4f} rad/m")
    else:
        print("k_peak: (not estimated; k was provided by user)")
    if fit is not None:
        print(f"gamma: {fit['gamma']:.6e}  [1/s]")
        print(f"tau  : {fit['tau']:.6e}  [s]")
        print(f"R^2  : {fit['R2']:.4f}")
        print(f"fit window: [{fit['tmin']:.3e}, {fit['tmax']:.3e}] s")
    else:
        print("gamma: N/A (could not fit ln A_k)")
    print("\n--- Energies (last finite value) ---")
    print(f"EB   (mag) : {last_finite(EB):.6e}  [J]")
    print(f"EK   (kin) : {last_finite(EK):.6e}  [J]   (proxy with vr)")
    print(f"Eint (int) : {last_finite(Eint):.6e}  [J]   (γ={gamma_gas:.3f})")
    print(f"RMS({field}) volume : {last_finite(Erms):.6e}")
    print("================================================================\n")

    return {
        "run_dir": str(run_dir),
        "k_peak": float(k_detected_once) if k_detected_once is not None else None,
        "gamma": (float(fit["gamma"]) if fit else None),
        "tau": (float(fit["tau"]) if fit else None),
        "R2": (float(fit["R2"]) if fit else None),
        "EB_last": last_finite(EB),
        "EK_last": last_finite(EK),
        "Eint_last": last_finite(Eint),
        "RMS_last": last_finite(Erms)
    }

def print_extended_help() -> None:
    """Human-friendly extended help with workflow and examples."""
    txt = r"""
=== Extended Help (diagnostics.py) ===

Recommended flow
----------------
1) python utils.py
2) python diagnostics.py --run-dir <RUN> --field p --save-spectrum --demean-z
3) Inspect: diagnostics_Ak.csv, diagnostics_energy.csv, diagnostics_lnA_fit.png

Important flags
---------------
--run-dir / --run-dirs : One or more simulation directories.
--field                : p | rho | vr | Bth | Bz (variable for A_k and RMS).
--mode                 : volume | centerline (volume integral or line at r=--r).
--r                    : Radius [m] for centerline (default 0.30*Rmax if omitted).
--method               : fft (fast) | direct (cosine integral).
--k                    : k [rad/m]. If 0, auto-detect via FFT on first snapshot.
--demean-z             : Subtract axial mean before projection (recommended).
--tmin / --tmax        : Time window for ln(A_k) fit.
--save-spectrum        : Save k-spectrum at t0 when k=0.
--gamma-gas            : Gas gamma for internal energy (default 5/3).
--mu0                  : Permeability (default 4π×1e-7).
--summary-csv          : Combined CSV when processing multiple runs.

Your recommended commands (verbatim)
------------------------------------
python pyscripts\diagnostics.py --run-dir .\data\sausage_m0_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts\diagnostics.py --run-dir .\data\kink_m1_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts\diagnostics.py --run-dir .\data\stable_ref_2p5d --field p --mode volume --demean-z --save-spectrum

Examples (generic)
------------------
# Single run, auto-detected k, volume projection:
python diagnostics.py --run-dir out/runA --field p --demean-z --save-spectrum

# Centerline at r=4 mm, direct method with fit window:
python diagnostics.py --run-dir out/runA --field p --mode centerline --r 0.004 \
                      --method direct --tmin 1e-6 --tmax 3e-6

# Multiple runs with combined CSV:
python diagnostics.py --run-dirs out/runA out/runB --summary-csv diag_all.csv
"""
    print(txt.strip())

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--help-extended", action="store_true", help="Show extended guidance")
    ap.add_argument("--run-dir", help="single run directory")
    ap.add_argument("--run-dirs", nargs="+", help="one or more run directories")
    ap.add_argument("--field", choices=["p","rho","vr","Bth","Bz"], default="p")
    ap.add_argument("--mode", choices=["volume","centerline"], default="volume")
    ap.add_argument("--r", type=float, default=None, help="for centerline: radius [m]")
    ap.add_argument("--method", choices=["fft","direct"], default="fft")
    ap.add_argument("--k", type=float, default=0.0, help="k [rad/m]; 0 = auto-detect via FFT")
    ap.add_argument("--demean-z", action="store_true")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--save-spectrum", action="store_true", help="save k-spectrum at t0 if k=0")
    # energies
    ap.add_argument("--gamma-gas", type=float, default=5.0/3.0)
    ap.add_argument("--mu0", type=float, default=4e-7*np.pi)
    ap.add_argument("--save-timeseries", action="store_true")
    ap.add_argument("--summary-csv", type=str, default="diagnostics_summary_all.csv",
                    help="combined CSV for multiple runs (saved alongside the first run)")
    args, unknown = ap.parse_known_args()

    # Show extended help if no CLI args or if requested
    if len(sys.argv) == 1 or args.help_extended:
        print_extended_help()
        return

    runs: List[Path] = []
    if args.run_dirs: runs.extend([Path(p) for p in args.run_dirs])
    if args.run_dir:  runs.append(Path(args.run_dir))
    if not runs:
        raise SystemExit("[ERR] You must pass --run-dir or --run-dirs ...")

    all_rows = []
    for rd in runs:
        info = run_diagnostics_one(
            rd, args.field, args.mode, args.r, args.method, args.k,
            args.demean_z, args.tmin, args.tmax, args.save_spectrum,
            args.gamma_gas, args.mu0, args.save_timeseries
        )
        all_rows.append(info)

    # Combined CSV (string+numeric safe formatting)
    out_root = runs[0] if runs else Path(".")
    out_csv = out_root / args.summary_csv
    header = "run_dir,k_peak,gamma,tau,R2,EB_last,EK_last,Eint_last,RMS_last"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(header+"\n")
        for r in all_rows:
            kpk = r["k_peak"] if r["k_peak"] is not None else np.nan
            g   = r["gamma"]  if r["gamma"]  is not None else np.nan
            tau = r["tau"]    if r["tau"]    is not None else np.nan
            R2  = r["R2"]     if r["R2"]     is not None else np.nan
            f.write(f"{r['run_dir']},{kpk:.10e},{g:.10e},{tau:.10e},{R2:.6f},"
                    f"{r['EB_last']:.10e},{r['EK_last']:.10e},{r['Eint_last']:.10e},{r['RMS_last']:.10e}\n")
    print(f"[OK] Combined summary: {out_csv}")

if __name__ == "__main__":
    main()

