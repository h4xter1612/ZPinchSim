#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys

BASE = Path(".")
DATA = BASE / "data"

# ------------------ utilidades meta / yaml-lite ------------------
def read_meta(path: Path):
    """Parser tolerante para lines tipo 'Nr: 128, Nz=256 # comment'."""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if "#" in ln:
                ln = ln.split("#", 1)[0].strip()
            # admite ":" o "=" y múltiples pares separados por coma
            for part in [p.strip() for p in ln.replace(":", "=").split(",")]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if v and ((v[0] == v[-1] == "'") or (v[0] == v[-1] == '"')):
                        v = v[1:-1].strip()
                    if k:
                        d[k] = v
    return d

def clean_val_colon_line(s: str) -> str:
    """Extrae valor a la derecha de ':' y limpia comentario."""
    v = s.split(":", 1)[1].strip()
    if "#" in v:
        v = v.split("#", 1)[0].strip()
    if len(v) >= 2 and (v[0] == v[-1] == '"' or v[0] == v[-1] == "'"):
        v = v[1:-1].strip()
    return v

def read_yaml_knobs():
    """Lee knobs mínimos desde configs/run2d_modes.yaml (yaml-lite)."""
    r_over_R = 0.30
    field = "vr"         # vr | Bth | p | rho
    k_target = 40.0
    y = BASE / "configs" / "run2d_modes.yaml"
    if not y.exists():
        return r_over_R, field, k_target
    with open(y, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or ":" not in s:
                continue
            key = s.split(":", 1)[0].strip()
            if key == "r_over_R":
                try:
                    r_over_R = float(clean_val_colon_line(s))
                except:
                    pass
            elif key == "field":
                v = clean_val_colon_line(s).split()[0]
                v = "Bth" if v.lower() == "bth" else v
                if v in ("vr", "Bth", "p", "rho"):
                    field = v
            elif key in ("k_target", "k", "k_mode"):
                try:
                    k_target = float(clean_val_colon_line(s))
                except:
                    pass
    return r_over_R, field, k_target

def step_from_name(p: Path):
    m = re.search(r"fields_(\d+)_", p.name)
    return int(m.group(1)) if m else None

def load_times_from_diag():
    out = {}
    diag = DATA / "diag.csv"
    if diag.exists():
        try:
            df = pd.read_csv(diag)
            if "step" in df.columns and "t" in df.columns:
                for _, row in df.iterrows():
                    try:
                        out[int(row["step"])] = float(row["t"])
                    except:
                        pass
        except:
            pass
    return out

# ------------------ detecta grid ------------------
META = sorted(DATA.glob("fields_*_meta.txt"))
if not META:
    print("[FAIL] falta meta (data/fields_*_meta.txt).")
    sys.exit(1)

meta = read_meta(META[-1])
Nr   = int(meta.get("Nr", "128"))
Nz   = int(meta.get("Nz", "256"))
Ng   = int(meta.get("Ng", meta.get("ghost", "0")))
Rmax = float(meta.get("Rmax", "0.02"))
Zmax = float(meta.get("Zmax", "0.10"))

# ------------------ carga serie de un campo ------------------
def load_field_series(field_name: str):
    """Devuelve (lista_matrices_Nr_x_Nz, lista_paths_usados). Silencioso."""
    files = sorted(DATA.glob(f"fields_*_{field_name}.csv"))
    arrs, used = [], []
    if not files:
        return arrs, used
    interior = Nr * Nz
    full = (Nr + 2*Ng) * (Nz + 2*Ng)
    for p in files:
        try:
            a = np.loadtxt(p, delimiter=",")
        except:
            continue
        n = a.size
        if n == interior:
            arrs.append(a.reshape(Nr, Nz)); used.append(p)
        elif Ng > 0 and n == full:
            A = a.reshape(Nr + 2*Ng, Nz + 2*Ng)[Ng:Ng+Nr, Ng:Ng+Nz]
            arrs.append(A); used.append(p)
        # si no cuadra, se omite en silencio
    return arrs, used

# ------------------ FFT en z ------------------
def amplitude_k(slice_z: np.ndarray, k_target: float, Lz: float):
    Nz_local = slice_z.shape[0]
    dz = Lz / Nz_local
    k_axis = 2.0 * np.pi * np.fft.rfftfreq(Nz_local, d=dz)
    F = np.fft.rfft(slice_z)
    j = int(np.argmin(np.abs(k_axis - k_target)))
    return float(np.abs(F[j])), float(k_axis[j])

def pick_ring_index(r_over_R: float):
    # mismo criterio que tu script “que sí leía bien”
    ridx = int(np.clip(round(r_over_R * Nr), 0, Nr-1))
    return ridx

# ------------------ main ------------------
r_over_R, field_req, k_target = read_yaml_knobs()

# intenta el field del yaml y luego fallbacks
candidates = [field_req] + [x for x in ("vr", "Bth", "p", "rho") if x != field_req]

series = []
files_used = []
chosen_field = None
for fld in candidates:
    series, files_used = load_field_series(fld)
    if series:
        chosen_field = fld
        break

if not series:
    print(f"[FAIL] no hay snapshots para: {candidates}")
    sys.exit(1)

times_by_step = load_times_from_diag()
ts = []
for p in files_used:
    stp = step_from_name(p)
    if stp is not None and stp in times_by_step:
        ts.append(times_by_step[stp])
    else:
        ts.append(np.nan)

# si faltan tiempos, usa índice (silencioso)
if np.any(~np.isfinite(ts)):
    ts = list(np.arange(len(series), dtype=float))

ridx = pick_ring_index(r_over_R)

amps = []
k_used = None
for A, tt in zip(series, ts):
    if ridx < 0 or ridx >= A.shape[0]:
        continue
    slice_z = A[ridx, :]
    amp, k_eff = amplitude_k(slice_z, k_target, Zmax)
    k_used = k_eff
    amps.append((tt, amp))

arr = np.array([(tt, aa) for tt, aa in amps if np.isfinite(tt) and np.isfinite(aa)])
if arr.shape[0] < 4:
    # salida compacta
    if arr.shape[0] > 0:
        print(f"[MODES] field={chosen_field} samples={arr.shape[0]} last_amp={arr[-1,1]:.3e}")
    else:
        print("[MODES] sin muestras válidas.")
    sys.exit(0)

arr = arr[arr[:, 0].argsort()]
tt = arr[:, 0]
AA = arr[:, 1]
mask = AA > 0
tt = tt[mask]; AA = AA[mask]

if len(tt) < 4:
    print(f"[MODES] field={chosen_field} samples={len(tt)} last_amp={AA[-1]:.3e}")
    sys.exit(0)

# usa últimos m puntos (más robusto si hay fase no exponencial al inicio)
m = max(5, len(tt)//2)
x = tt[-m:]; y = np.log(AA[-m:])
coef = np.polyfit(x, y, 1)
gamma = float(coef[0])

# --------- salida mínima y clara ---------
print(f"[MODES] field={chosen_field}  r_over_R={r_over_R:.3f}  snaps={len(tt)}")
print(f"[MODES] k_target={k_target:.3f} rad/m  k_used≈{k_used:.3f} rad/m")
print(f"[MODES] growth_rate={gamma:.3e} 1/s  last_amp={AA[-1]:.3e}")

