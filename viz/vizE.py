#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ------------------ E/S ------------------
def load_meta(meta_path: Path):
    s = meta_path.read_text(encoding="utf-8")
    # Formato: t=..., Nr=..., Nz=..., Ng=..., Rmax=..., Zmax=...
    t   = float(re.search(r"t=([0-9eE\.\+\-]+)", s).group(1))
    Nr,Nz,Ng = map(int, re.search(r"Nr=(\d+), Nz=(\d+), Ng=(\d+)", s).groups())
    Rmax,Zmax = map(float, re.search(r"Rmax=([0-9eE\.\+\-]+), Zmax=([0-9eE\.\+\-]+)", s).groups())
    return t,Nr,Nz,Ng,Rmax,Zmax

def list_steps(run_dir: Path):
    steps=[]
    for p in run_dir.glob("fields_*_meta.txt"):
        try: steps.append(int(p.stem.split("_")[1]))
        except: pass
    return sorted(steps)

def load_field(run_dir: Path, step: int, field: str):
    p = run_dir / f"fields_{step}_{field}.csv"
    if not p.exists(): return None
    return np.loadtxt(p, delimiter=",", dtype=float)

def reshape_center(vec, Nr, Nz, Ng):
    NrT, NzT = Nr + 2*Ng, Nz + 2*Ng
    A = np.asarray(vec, float).reshape(NrT, NzT)
    return A[Ng:Ng+Nr, Ng:Ng+Nz]

def try_read_gamma_csv(run_dir: Path):
    p = run_dir / "diagnostics_fit_gamma.csv"
    if not p.exists(): return None
    try:
        rows = [ln.strip().split(",") for ln in p.read_text(encoding="utf-8").strip().splitlines()]
        if len(rows) < 2: return None
        for row in reversed(rows[1:]):
            if len(row) >= 3:
                g = float(row[2])
                if np.isfinite(g): return g
    except:
        return None
    return None

# ---------- estimador de gamma (fallback) ----------
def estimate_gamma_from_snaps(run_dir: Path, field: str, frac_core=0.35, frac_window=0.25):
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
    except:
        return None

# ------------------ utilidades visuales ------------------
def interp_radial_linear(F0_rz, r_base, dr):
    Nr, Nz = F0_rz.shape
    ii = np.clip((r_base/dr - 0.5).astype(np.int64), 0, Nr-2)
    w  = np.clip((r_base - (ii+0.5)*dr)/dr, 0.0, 1.0)
    Fexp = F0_rz[:, None, :]
    ii3  = ii[None, :, :]
    fL = np.take_along_axis(Fexp, ii3, axis=0)[0]
    fR = np.take_along_axis(Fexp, (ii3+1), axis=0)[0]
    return (1.0 - w)*fL + w*fR

def soft_clamp_sym(x, lo, hi, k=4.0):
    # clamp suave simétrico con transición logística
    mid = 0.5*(lo+hi)
    sig = 1.0/(1.0 + np.exp(-k*(x - mid)))
    return lo + (hi-lo)*sig

# ------------------ generadores de frames ------------------
def frame_kink(F0_rz, z, Rmax, a, k, c, t, x, dr):
    x_c = a * np.cos(k*(z - c*t))[None, :]
    r_base = np.abs(x[:, None] - x_c)
    r_base = np.clip(r_base, 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_base, dr)

def frame_sausage(F0_rz, z, Rmax, a, k, c, t, x, dr,
                  stretch_lo=0.75, stretch_hi=1.25, k_soft=4.0, beta=0.65):
    R0 = 0.30 * Rmax
    s_raw  = (a / (R0 + 1e-12)) * np.cos(k*(z - c*t))[None, :]   # (1, Nz)
    stretch_raw = 1.0 + s_raw
    # clamp suave (sin esquinas duras)
    stretch = soft_clamp_sym(stretch_raw, stretch_lo, stretch_hi, k=k_soft)
    # mapeo con exponente beta para realzar/atenuar el contraste radial
    r_eff = np.abs(x[:, None]) / np.power(stretch, beta)
    r_eff = np.clip(r_eff, 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_eff, dr)

def frame_stable(F0_rz, z, Rmax, x, dr):
    r_base = np.clip(np.abs(x[:, None]), 0.0, Rmax - 1e-12)
    return interp_radial_linear(F0_rz, r_base, dr)

def infer_mode_from_name(run_dir: Path, default_mode: str):
    name = run_dir.name.lower()
    if "kink" in name: return "kink"
    if "sausage" in name: return "sausage"
    if "stable" in name or "ref" in name: return "stable"
    return default_mode

# --------- evolución de amplitud ---------
def evolve_amplitude(times, a0_init, gamma, a_sat, p=1.0):
    """
    Integra a' = gamma * a * (1 - (a/a_sat)^p) sobre los tiempos dados.
    Devuelve a(t_i) con a(t_0)=a0_init. p=1 -> logística clásica.
    """
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

# ------------------ render de UN run ------------------
def render_one_run(run_dir: Path, field: str, mode_arg: str, amp_mode: str, k_arg: float,
                   a0_arg: float, gamma_arg: float, phase_speed: float,
                   rview_arg: float, every: int, make_mp4: bool, fps: int, dpi: int,
                   nx: int, t_ref_step: int, a_sat_frac: float,
                   auto_core_frac: float, auto_window_frac: float,
                   sausage_bounds: tuple, sausage_ksoft: float, sausage_beta: float,
                   logistic_power: float):

    steps = list_steps(run_dir)
    if not steps:
        print(f"[WARN] No hay snapshots en {run_dir}")
        return

    # t0
    if (t_ref_step is not None) and (t_ref_step in steps):
        t0, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{t_ref_step}_meta.txt")
    else:
        t0, Nr, Nz, Ng, Rmax, Zmax = load_meta(run_dir / f"fields_{steps[0]}_meta.txt")

    dr, dz = Rmax/Nr, Zmax/Nz
    z = (np.arange(Nz)+0.5)*dz
    k = k_arg if k_arg is not None else (2.0*np.pi/Zmax)
    Rview = rview_arg if rview_arg is not None else 1.05*Rmax
    x = np.linspace(-Rview, Rview, nx)

    # modo (auto por nombre si así se pidió)
    mode = mode_arg
    if mode == "auto":
        mode = infer_mode_from_name(run_dir, "kink")

    # gamma base
    if mode == "stable":
        gamma_use = 0.0
    else:
        gamma_use = try_read_gamma_csv(run_dir)
        if gamma_use is None:
            gamma_use = estimate_gamma_from_snaps(run_dir, field,
                                                  frac_core=auto_core_frac,
                                                  frac_window=auto_window_frac) or 0.0

    # amplitud inicial y saturación
    a0_default = 0.1*Rmax
    a0_base = a0_arg if (amp_mode == "fixed" and a0_arg is not None) else max(3.0*dr, 0.10*(0.30*Rmax))
    a_sat = a_sat_frac * Rmax  # saturación geométrica

    # vlims globales
    vals = []
    pick = max(1, len(steps)//30)
    for s in steps[::pick]:
        Q0 = load_field(run_dir, s, field)
        if Q0 is None: continue
        F0rz = reshape_center(Q0, Nr, Nz, Ng)
        vals.append(F0rz.ravel())
    vlims = None
    if vals:
        big = np.concatenate(vals)
        vlims = (np.percentile(big, 2), np.percentile(big, 98))

    # tiempos de frames
    times_all = []
    metas = {}
    for s in steps[::every]:
        meta = run_dir / f"fields_{s}_meta.txt"
        if not meta.exists(): continue
        t, NrS, NzS, NgS, RmaxS, ZmaxS = load_meta(meta)
        times_all.append(t)
        metas[s] = (t, NrS, NzS, NgS, RmaxS, ZmaxS)
    if not times_all:
        print(f"[WARN] No hay metadatos legibles en {run_dir}")
        return
    times_all = np.asarray(times_all, float)

    # construir a(t) según amp_mode
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
        p = max(0.5, float(logistic_power))  # logística generalizada
        a_t_series = evolve_amplitude(times_all, a0_init=a0_base, gamma=gamma_eff, a_sat=a_sat, p=p)
    elif amp_mode == "auto-energy":
        gamma_eff = gamma_use
        a_t_series = a0_base * np.exp(gamma_eff*(times_all - times_all[0])) if gamma_eff!=0 else np.full_like(times_all, a0_base)
        a_t_series = np.clip(a_t_series, 0.0, a_sat)
    else:
        gamma_eff = 0.0
        a_t_series = np.full_like(times_all, a0_default)

    # escritor de video
    writer = None
    if make_mp4:
        out_mp4 = run_dir / f"viz_{field}_{mode}.mp4"
        writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=None)

    # figura única (evita warnings)
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
            vmin, vmax = vlims if vlims is not None else np.percentile(img, [2,98])
            im = ax.imshow(img, origin="lower", aspect="auto",
                           extent=[z[0], z[-1], -Rview, Rview],
                           vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
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

# ------------------ main (multi-runs) ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="un solo directorio de run")
    ap.add_argument("--run-dirs", nargs="+", help="uno o más directorios de run")
    ap.add_argument("--field", choices=["p","rho","vr","Bth","Bz"], default="p")
    ap.add_argument("--mode", choices=["kink","sausage","stable","auto"], default="auto")
    ap.add_argument("--amp-mode", choices=["fixed","auto","auto-logistic","auto-energy"], default="auto")
    ap.add_argument("--k", type=float, default=None, help="rad/m; si None usa 2π/Zmax")
    ap.add_argument("--a0", type=float, default=None, help="amplitud inicial [m] si fixed")
    ap.add_argument("--gamma", type=float, default=0.0, help="γ [1/s] si fixed")
    ap.add_argument("--phase-speed", type=float, default=0.0, help="c [m/s]")
    ap.add_argument("--rview", type=float, default=None, help="semi-ancho lateral mostrado [m]")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--make-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--nx", type=int, default=700)
    # Auto / saturación
    ap.add_argument("--t-ref-step", type=int, default=None)
    ap.add_argument("--a-sat-frac", type=float, default=0.50, help="saturación geométrica a_sat = frac*Rmax")
    ap.add_argument("--auto-core-frac", type=float, default=0.35)
    ap.add_argument("--auto-window-frac", type=float, default=0.25)
    # Sausage soft-clamp
    ap.add_argument("--sausage-bounds", nargs=2, type=float, default=[0.75, 1.25],
                    metavar=("LO","HI"), help="límites suaves del stretch")
    ap.add_argument("--sausage-ksoft", type=float, default=4.0, help="suavidad del clamp")
    ap.add_argument("--sausage-beta", type=float, default=0.65, help="exponente del stretch en el mapeo")
    # Logística generalizada
    ap.add_argument("--logistic-power", type=float, default=1.0,
                    help="exponente p de la logística generalizada (1.0 = clásica)")
    args = ap.parse_args()

    runs = []
    if args.run_dirs: runs.extend([Path(p) for p in args.run_dirs])
    if args.run_dir:  runs.append(Path(args.run_dir))
    if not runs:
        raise SystemExit("[ERR] Debes pasar --run-dir o --run-dirs ...")

    for rd in runs:
        render_one_run(
            run_dir=rd, field=args.field, mode_arg=args.mode, amp_mode=args.amp_mode,
            k_arg=args.k, a0_arg=args.a0, gamma_arg=args.gamma, phase_speed=args.phase_speed,
            rview_arg=args.rview, every=args.every, make_mp4=args.make_mp4, fps=args.fps,
            dpi=args.dpi, nx=args.nx, t_ref_step=args.t_ref_step, a_sat_frac=args.a_sat_frac,
            auto_core_frac=args.auto_core_frac, auto_window_frac=args.auto_window_frac,
            sausage_bounds=(float(args.sausage_bounds[0]), float(args.sausage_bounds[1])),
            sausage_ksoft=args.sausage_ksoft, sausage_beta=args.sausage_beta,
            logistic_power=args.logistic_power
        )

if __name__ == "__main__":
    main()

