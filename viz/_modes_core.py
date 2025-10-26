# Lightweight core for modal growth analysis (step6-ready)
import numpy as np
import pandas as pd
from pathlib import Path
import re

def _read_meta(path: Path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if "#" in ln:
                ln = ln.split("#", 1)[0].strip()
            for part in [p.strip() for p in ln.replace(":", "=").split(",")]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    d[k.strip()] = v.strip()
    return d

def _step_from_name(p: Path):
    import re
    m = re.search(r"fields_(\d+)_", p.name)
    return int(m.group(1)) if m else None

def _load_times(diag_csv: Path):
    out = {}
    if diag_csv.exists():
        try:
            df = pd.read_csv(diag_csv)
            if "step" in df.columns and "t" in df.columns:
                for _, row in df.iterrows():
                    try: out[int(row["step"])] = float(row["t"])
                    except: pass
        except: pass
    return out

def _amplitude_k(slice_z: np.ndarray, k_target: float, Lz: float):
    Nz_local = slice_z.shape[0]
    dz = Lz / Nz_local
    k_axis = 2.0 * np.pi * np.fft.rfftfreq(Nz_local, d=dz)
    F = np.fft.rfft(slice_z)
    j = int(np.argmin(np.abs(k_axis - k_target)))
    return float(np.abs(F[j])), float(k_axis[j])

def compute_growth_rate(run_dir: Path, r_over_R: float=0.3, field: str="vr",
                        k_target: float=40.0, fit_frac: float=0.5):
    run_dir = Path(run_dir)
    metas = sorted(run_dir.glob("fields_*_meta.txt"))
    if not metas:
        raise FileNotFoundError(f"No snapshots in {run_dir}")
    meta = _read_meta(metas[-1])
    Nr   = int(meta.get("Nr", "128"))
    Nz   = int(meta.get("Nz", "256"))
    Ng   = int(meta.get("Ng", "0"))
    Zmax = float(meta.get("Zmax", "0.10"))
    files = sorted(run_dir.glob(f"fields_*_{field}.csv"))
    interior = Nr * Nz
    full = (Nr + 2*Ng) * (Nz + 2*Ng)
    arrs, used = [], []
    for p in files:
        try:
            a = np.loadtxt(p, delimiter=",")
        except:
            continue
        n = a.size
        if n == interior:
            arrs.append(a.reshape(Nr, Nz)); used.append(p)
        elif n == full:
            A = a.reshape(Nr + 2*Ng, Nz + 2*Ng)[Ng:Ng+Nr, Ng:Ng+Nz]
            arrs.append(A); used.append(p)
    if not arrs:
        raise RuntimeError(f"No readable {field} snapshots in {run_dir}")

    t_by_step = _load_times(run_dir / "diag.csv")
    ts = []
    for p in used:
        stp = _step_from_name(p)
        if stp is not None and stp in t_by_step:
            ts.append(t_by_step[stp])
        else:
            ts.append(np.nan)
    if np.any(~np.isfinite(ts)):
        ts = list(np.arange(len(arrs), dtype=float))

    ridx = int(np.clip(round(r_over_R * Nr), 0, Nr-1))

    amps = []
    k_used = None
    for A, tt in zip(arrs, ts):
        if ridx < 0 or ridx >= A.shape[0]:
            continue
        slice_z = A[ridx, :]
        amp, k_eff = _amplitude_k(slice_z, k_target, Zmax)
        k_used = k_eff
        amps.append((tt, amp))
    arr = np.array([(tt, aa) for tt, aa in amps if np.isfinite(tt) and np.isfinite(aa)])
    if arr.shape[0] < 8:
        return {"ok": False, "reason": "insufficient_samples", "samples": int(arr.shape[0])}

    arr = arr[arr[:,0].argsort()]
    tt = arr[:,0]
    AA = arr[:,1]
    mask = AA > 0
    tt = tt[mask]; AA = AA[mask]
    if len(tt) < 8:
        return {"ok": False, "reason": "nonpositive_amplitudes", "samples": int(len(tt))}

    m = max(8, int(len(tt)*fit_frac))
    x = tt[-m:]; y = np.log(AA[-m:])
    coef = np.polyfit(x, y, 1)
    gamma = float(coef[0])
    return {
        "ok": True, "field": field, "snapshots": int(len(tt)),
        "k_target": float(k_target), "k_used": float(k_used),
        "r_over_R": float(r_over_R), "growth_rate": float(gamma),
        "last_amp": float(AA[-1])
    }
