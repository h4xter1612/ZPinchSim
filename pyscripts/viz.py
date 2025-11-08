#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz.py — Synthetic view rendering (kink/sausage/stable) and MP4 export.

Key features:
- Extended help appears automatically when no arguments are provided.
- Reads fitted gamma from diagnostics (if available) or estimates it from early snapshots.
- Global vmin/vmax estimation (fast sampling) with optional manual overrides and colormap.
- Smooth "sausage" clamp (no hard edges) with tunable parameters.

Reminder: run `python utils.py` first to ensure dependencies are installed.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ------------------ I/O helpers ------------------
_RE_T     = re.compile(r"t=([0-9eE\.\+\-]+)")
_RE_NRNZ  = re.compile(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)")
_RE_RZMAX = re.compile(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)")

def load_meta(meta_path: Path):
    """Parse a meta file with lines like: t=..., Nr=..., Nz=..., Ng=..., Rmax=..., Zmax=..."""
    s = meta_path.read_text(encoding="utf-8")
    t   = float(_RE_T.search(s).group(1))
    Nr,Nz,Ng = map(int, _RE_NRNZ.search(s).groups())
    Rmax,Zmax= map(float, _RE_RZMAX.search(s).groups())
    return t,Nz and Nr,Ng,Rmax,Zmax  # intentional unpack below to keep order explicit

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

def load_field(run_dir: Path, step: int, field: str):
    """Load the raw CSV array for a given field/step; None if missing."""
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists(): return None
    return np.loadtxt(p, delimiter=",", dtype=float)

def reshape_center(vec, Nr, Nz, Ng):
    """Strip guard cells in both dimensions and return the core (Nr x Nz)."""
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

def try_read_gamma_csv(run_dir: Path) -> Optional[float]:
    """Read the last finite gamma from diagnostics_fit_gamma.csv if present."""
    p = run_dir / "diagnostics_fit_gamma.csv"
    if not p.exists(): return None
    try:
        rows = [ln.strip().split(",") for ln in p.read_text(encoding="utf-8").strip().splitlines()]
        if len(rows) < 2: return None
        for row in reversed(rows[1:]):
            if len(row) >= 3:
                g = float(row[2])
                if np.isfinite(g): return g
    except Exception:
        return None
    return None

# ---------- Gamma estimator (fallback) ----------
def estimate_gamma_from_snaps(run_dir: Path, field: str, frac_core=0.35, frac_window=0.25):
    """
    Estimate growth rate gamma from early-time RMS inside a core radius.
    Uses a simple log-linear fit over a window of early snapshots.
    """
    steps = list_steps(run_dir)
    if len(steps) < 10: return None
    t0,Nr,Nz,Ng,Rmax,Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")
    dr, dz = Rmax/Nr, Zmax/Nz
    r = (np.arange(Nr)+0.5)*dr
    R0 = frac_core * Rmax
    w_r = (r <= R0).astype(float)
    if w_r.sum() <= 1: return None
    w_r /= w_r.sum()
    nfit = max(10, int(len(steps)*frac_window))
    times, amps = [], []
    for s in steps[:nfit]:
        meta = run_dir / f"fields_{s}_meta.txt"
        Qraw = load_field(run_dir, s, field)
        if (not meta.exists()) or (Qraw is None): continue
        t, NrS, NzS, NgS, RmaxS, ZmaxS = load_meta(meta)
        Q = reshape_center(Qraw, NrS, NzS, NgS)
        Qdm = Q - Q.mean(axis=1, keepdims=True)
        q_rms_r = np.sqrt((Qdm**2).mean(axis=1))
        A = float(np.sum(q_rms_r * w_r[:NrS]))
        if A > 0 and np.isfinite(A):
            times.append(t); amps.append(A)
    if len(times) < 5: return None
    times = np.asarray(times); amps = np.asarray(amps)
    m = amps > 0
    if m.sum() < 5: return None
    x = times[m] - times[m][0]
    y = np.log(amps[m])
    X = np.vstack([np.ones_like(x), x]).T
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        gamma = float(coef[1])
        return gamma if np.isfinite(gamma) else None
    except Exception:
        return None

# ------------------ Visualization utilities ------------------
def interp_radial_linear(F0_rz, r_base, dr):
    """Linear interpolation along radius for synthetic cross-sections."""
    Nr, Nz = F0_rz.shape
    ii = np.clip((r_base/dr - 0.5).astype(np.int64), 0, Nr-2)
    w  = np.clip((r_base - (ii+0.5)*dr)/dr, 0.0, 1.0)
    Fexp = F0_rz[:, None, :]
    ii3  = ii[None, :, :]
    fL = np.take_along_axis(Fexp, ii3, axis=0)[0]
    fR = np.take_along_axis(Fexp, (ii3+1), axis=0)[0]
    return (1.0 - w)*fL + w*fR

def soft_clamp_sym(x, lo, hi, k=4.0):
    """Smooth symmetric clamp using a logistic-like transition."""
    mid = 0.5*(lo+hi)
    sig = 1.0/(1.0 + np.exp(-k*(x - mid)))
    return lo + (hi-lo)*sig

# ------------------ Frame generators ------------------
def frame_kink(F0_rz, z, Rmax, a, k, c, t, x, dr):
    """Kink-like lateral displacement of the core centerline."""
    x_c = a * np.cos(k*(z - c*t))[None, :]
    r_base = np.abs(x[:, None] - x_c)
    r_base = np.clip(r_base, 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_base, dr)

def frame_sausage(F0_rz, z, Rmax, a, k, c, t, x, dr,
                  stretch_lo=0.75, stretch_hi=1.25, k_soft=4.0, beta=0.65):
    """Sausage-like radial stretch/compression around the core radius."""
    R0 = 0.30 * Rmax
    s_raw  = (a / (R0 + 1e-12)) * np.cos(k*(z - c*t))[None, :]
    stretch_raw = 1.0 + s_raw
    stretch = soft_clamp_sym(stretch_raw, stretch_lo, stretch_hi, k=k_soft)
    r_eff = np.abs(x[:, None]) / np.power(stretch, beta)
    r_eff = np.clip(r_eff, 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_eff, dr)

def frame_stable(F0_rz, z, Rmax, x, dr):
    """Stable reference (no axial modulation)."""
    r_base = np.clip(np.abs(x[:, None]), 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_base, dr)

def infer_mode_from_name(run_dir: Path, default_mode: str):
    """Guess mode from folder name."""
    name = run_dir.name.lower()
    if "kink" in name: return "kink"
    if "sausage" in name: return "sausage"
    if "stable" in name or "ref" in name: return "stable"
    return default_mode

# --------- Amplitude evolution ---------
def evolve_amplitude(times, a0_init, gamma, a_sat, p=1.0):
    """Integrate a' = gamma*a*(1 - (a/a_sat)^p) along given times (explicit Euler)."""
    a = a0_init
    out = [a]
    t_prev = times[0]
    for t in times[1:]:
        dt = max(t - t_prev, 0.0)
        if gamma != 0.0 and dt > 0.0 and a_sat > 0.0:
            a = a + gamma * a * (1.0 - (a / a_sat)**p) * dt
            if a < 0.0: a = 0.0
        out.append(a)
        t_prev = t
    return np.asarray(out, float)

# ------------------ Single-run renderer ------------------
def render_one_run(run_dir: Path, field: str, mode_arg: str, amp_mode: str, k_arg: Optional[float],
                   a0_arg: Optional[float], gamma_arg: float, phase_speed: float,
                   rview_arg: Optional[float], every: int, make_mp4: bool, fps: int, dpi: int,
                   nx: int, t_ref_step: Optional[int], a_sat_frac: float,
                   auto_core_frac: float, auto_window_frac: float,
                   sausage_bounds: Tuple[float,float], sausage_ksoft: float, sausage_beta: float,
                   logistic_power: float, cmap: str, vmin_user: Optional[float], vmax_user: Optional[float]):

    steps = list_steps(run_dir)
    if not steps:
        print(f"[WARN] No snapshots in {run_dir}")
        return

    # Choose a reference meta (explicit step if provided)
    if (t_ref_step is not None) and (t_ref_step in steps):
        t0, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{t_ref_step}_meta.txt")
    else:
        t0, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")

    dr, dz = Rmax/Nr, Zmax/Nz
    z = (np.arange(Nz)+0.5)*dz
    k = k_arg if k_arg is not None else (2.0*np.pi/Zmax)
    Rview = rview_arg if rview_arg is not None else 1.05*Rmax
    x = np.linspace(-Rview, Rview, nx)

    # Mode (auto by folder name if requested)
    mode = mode_arg
    if mode == "auto":
        mode = infer_mode_from_name(run_dir, "kink")

    # Growth rate
    if mode == "stable":
        gamma_use = 0.0
    else:
        gamma_use = try_read_gamma_csv(run_dir)
        if gamma_use is None:
            gamma_use = estimate_gamma_from_snaps(run_dir, field,
                                                  frac_core=auto_core_frac,
                                                  frac_window=auto_window_frac) or 0.0

    # Amplitude initial value and saturation cap
    a0_default = 0.1*Rmax
    a0_base = a0_arg if (amp_mode == "fixed" and a0_arg is not None) else max(3.0*dr, 0.10*(0.30*Rmax))
    a_sat = a_sat_frac * Rmax

    # Global color limits via sampling (unless user overrides)
    vals = []
    pick = max(1, len(steps)//30)
    for s in steps[::pick]:
        Q0 = load_field(run_dir, s, field)
        if Q0 is None: continue
        F0rz = reshape_center(Q0, Nr, Nz, Ng)
        vals.append(F0rz.ravel())
    vlims = None
    if vals and (vmin_user is None or vmax_user is None):
        big = np.concatenate(vals)
        vmin_est, vmax_est = np.percentile(big, [2, 98])
        vlims = (vmin_est, vmax_est)

    # Build frame time list
    times_all = []
    metas = {}
    for s in steps[::every]:
        meta = run_dir / f"fields_{s}_meta.txt"
        if not meta.exists(): continue
        t, NrS, NzS, NgS, RmaxS, ZmaxS = load_meta(meta)
        times_all.append(t)
        metas[s] = (t, NrS, NzS, NgS, RmaxS, ZmaxS)
    if not times_all:
        print(f"[WARN] No readable metadata in {run_dir}")
        return
    times_all = np.asarray(times_all, float)

    # Amplitude time series
    if mode == "stable":
        a_t_series = np.zeros_like(times_all)
    elif amp_mode == "fixed":
        gamma_eff = gamma_arg
        a_t_series = a0_base * np.exp(gamma_eff*(times_all - times_all[0])) if gamma_eff!=0 else np.full_like(times_all, a0_base)
        a_t_series = np.clip(a_t_series, 0.0, a_sat)
    elif amp_mode == "auto":
        gamma_eff = gamma_use
        a_t_series = a0_base * np.exp(gamma_eff*(times_all - times_all[0])) if gamma_eff!=0 else np.full_like(times_all, a0_base)
        a_t_series = np.clip(a_t_series, 0.0, a_sat)
    elif amp_mode == "auto-logistic":
        gamma_eff = gamma_use
        p = max(0.5, float(logistic_power))
        a_t_series = evolve_amplitude(times_all, a0_init=a0_base, gamma=gamma_eff, a_sat=a_sat, p=p)
    elif amp_mode == "auto-energy":
        gamma_eff = gamma_use
        a_t_series = a0_base * np.exp(gamma_eff*(times_all - times_all[0])) if gamma_eff!=0 else np.full_like(times_all, a0_base)
        a_t_series = np.clip(a_t_series, 0.0, a_sat)
    else:
        gamma_eff = 0.0
        a_t_series = np.full_like(times_all, a0_default)

    # Optional MP4 writer
    writer = None
    out_mp4 = None
    if make_mp4:
        out_mp4 = run_dir / f"viz_{field}_{mode}.mp4"
        writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=None)

    # Single figure reused for all frames (faster)
    fig = None; ax = None; im = None; cb = None
    last_png = run_dir / f"viz_{field}_{mode}_last.png"

    lo, hi = sausage_bounds

    for idx, s in enumerate(steps[::every]):
        if s not in metas: continue
        t, NrS, NzS, NgS, RmaxS, ZmaxS = metas[s]
        Q = load_field(run_dir, s, field)
        if Q is None: continue
        F0_rz = reshape_center(Q, NrS, NzS, NgS)

        a_t = float(a_t_series[idx])

        if mode == "kink":
            img = frame_kink(F0_rz, z, RmaxS, a_t, k, phase_speed, t, x, dr)
        elif mode == "sausage":
            img = frame_sausage(F0_rz, z, RmaxS, a_t, k, phase_speed, t, x, dr,
                                stretch_lo=lo, stretch_hi=hi, k_soft=sausage_ksoft, beta=sausage_beta)
        else:
            img = frame_stable(F0_rz, z, RmaxS, x, dr)

        title = f"{field} {mode} | t={t*1e6:.2f} µs | a={a_t*1e3:.2f} mm | k={k:.2f} rad/m | γ={gamma_use:.2e} s⁻¹"

        if fig is None:
            fig = plt.figure(figsize=(9,4.8), dpi=dpi, constrained_layout=True)
            ax  = fig.add_subplot(111)
            if vmin_user is not None and vmax_user is not None:
                vmin, vmax = vmin_user, vmax_user
            else:
                vmin, vmax = (np.percentile(img, [2,98]) if vlims is None else vlims)
            im = ax.imshow(img, origin="lower", aspect="auto",
                           extent=[z[0], z[-1], -Rview, Rview],
                           vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            cb = fig.colorbar(im); cb.set_label("field")
            ax.set_xlabel("z [m]"); ax.set_ylabel("x [m]")
        else:
            im.set_data(img)

        ax.set_title(title)

        if writer is not None:
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame)

    if fig is not None:
        fig.savefig(last_png, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", last_png)
    if writer is not None:
        writer.close()
        print("Saved:", out_mp4)

def print_extended_help() -> None:
    """Human-friendly extended help with workflow and examples."""
    txt = r"""
=== Extended Help (viz.py) ===

Suggested workflow
------------------
1) python utils.py
2) python diagnostics.py --run-dir <RUN> ...
3) python viz.py --run-dir <RUN> --mode auto --make-mp4 --fps 30

Key flags
---------
--run-dir / --run-dirs     : One or more runs to render.
--field                    : p | rho | vr | Bth | Bz
--mode                     : kink | sausage | stable | auto  (auto infers from run folder name)
--amp-mode                 : fixed | auto | auto-logistic | auto-energy
--k                        : Axial wavenumber [rad/m]. If None => 2π/Zmax.
--a0, --gamma              : Initial amplitude and γ for 'fixed'.
--phase-speed              : Phase speed [m/s] for axial modulation.
--rview                    : Lateral half-width shown (default 1.05*Rmax).
--every                    : Skip snapshots to speed up (1 = use all).
--make-mp4 --fps --dpi     : Export video and control quality.
--nx                       : Horizontal resolution (x) of synthetic render.
--t-ref-step               : Step used to anchor t0 (if None uses first).
--a-sat-frac               : Geometric saturation (a_sat = frac*Rmax).
--auto-core-frac/window    : Params to estimate γ automatically if fit is absent.
--sausage-bounds/ksoft/beta: Smooth clamp and mapping controls for 'sausage'.
--logistic-power           : Exponent p for generalized logistic growth.
--cmap                     : Matplotlib colormap (e.g., viridis, plasma, magma).
--vmin/--vmax              : Manual color limits (otherwise estimated globally).

Your recommended commands (verbatim)
------------------------------------
python pyscripts/viz.py --run-dir ./data/sausage_m0_2p5d --field p --mode auto --amp-mode auto-logistic --logistic-power 1.8 --a-sat-frac 0.65 --sausage-bounds 0.60 1.60 --sausage-ksoft 6.0 --sausage-beta 0.95 --make-mp4 --fps 24

python pyscripts/viz.py --run-dir ./data/kink_m1_2p5d --field p --mode auto --amp-mode auto-logistic --logistic-power 1.6 --a-sat-frac 0.60 --make-mp4 --fps 24
python pyscripts\viz.py --run-dir .\data\kink_m1_2p5d --field p  --mode auto --amp-mode auto-logistic --a-sat-frac 0.5 --make-mp4 --fps 24

python pyscripts\viz.py --run-dir .\data\stable_ref_2p5d --field p --mode stable --make-mp4 --fps 24

Examples (generic)
------------------
# Auto-gamma (use fit if present; else estimate):
python viz.py --run-dir out/runA --mode auto --make-mp4 --fps 24

# Sausage with smooth clamp and manual color limits:
python viz.py --run-dir out/runA --mode sausage --cmap magma --vmin -2 --vmax 2 \
              --sausage-bounds 0.8 1.3 --sausage-ksoft 5.0 --sausage-beta 0.6

# Kink with fixed parameters:
python viz.py --run-dir out/runB --mode kink --amp-mode fixed --a0 0.003 --gamma 5e4 --make-mp4
"""
    print(txt.strip())

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--help-extended", action="store_true", help="Show extended guidance")
    ap.add_argument("--run-dir", help="single run directory")
    ap.add_argument("--run-dirs", nargs="+", help="one or more run directories")
    ap.add_argument("--field", choices=["p","rho","vr","Bth","Bz"], default="p")
    ap.add_argument("--mode", choices=["kink","sausage","stable","auto"], default="auto")
    ap.add_argument("--amp-mode", choices=["fixed","auto","auto-logistic","auto-energy"], default="auto")
    ap.add_argument("--k", type=float, default=None, help="rad/m; if None uses 2π/Zmax")
    ap.add_argument("--a0", type=float, default=None, help="initial amplitude [m] if 'fixed'")
    ap.add_argument("--gamma", type=float, default=0.0, help="γ [1/s] if 'fixed'")
    ap.add_argument("--phase-speed", type=float, default=0.0, help="c [m/s]")
    ap.add_argument("--rview", type=float, default=None, help="lateral half-width shown [m]")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--nx", type=int, default=700)
    # Auto / saturation
    ap.add_argument("--t-ref-step", type=int, default=None)
    ap.add_argument("--a-sat-frac", type=float, default=0.50, help="saturation a_sat = frac*Rmax")
    ap.add_argument("--auto-core-frac", type=float, default=0.35)
    ap.add_argument("--auto-window-frac", type=float, default=0.25)
    # Sausage soft-clamp
    ap.add_argument("--sausage-bounds", nargs=2, type=float, default=[0.75, 1.25],
                    metavar=("LO","HI"), help="smooth limits for stretch")
    ap.add_argument("--sausage-ksoft", type=float, default=4.0, help="smoothness of clamp")
    ap.add_argument("--sausage-beta", type=float, default=0.65, help="stretch exponent in mapping")
    # Generalized logistic
    ap.add_argument("--logistic-power", type=float, default=1.0,
                    help="exponent p for generalized logistic (1.0 = classical)")
    # Visual extras
    ap.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
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

    for rd in runs:
        render_one_run(
            run_dir=rd, field=args.field, mode_arg=args.mode, amp_mode=args.amp_mode,
            k_arg=args.k, a0_arg=args.a0, gamma_arg=args.gamma, phase_speed=args.phase_speed,
            rview_arg=args.rview, every=args.every, make_mp4=args.make_mp4, fps=args.fps,
            dpi=args.dpi, nx=args.nx, t_ref_step=args.t_ref_step, a_sat_frac=args.a_sat_frac,
            auto_core_frac=args.auto_core_frac, auto_window_frac=args.auto_window_frac,
            sausage_bounds=(float(args.sausage_bounds[0]), float(args.sausage_bounds[1])),
            sausage_ksoft=args.sausage_ksoft, sausage_beta=args.sausage_beta,
            logistic_power=args.logistic_power, cmap=args.cmap,
            vmin_user=args.vmin, vmax_user=args.vmax
        )

if __name__ == "__main__":
    main()

