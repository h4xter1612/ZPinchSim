# viz/check_2d_mhd.py  (reemplaza el contenido)
import os, sys, pandas as pd, numpy as np

m = "./data/debug/2d_mhd_metrics.csv"
if not os.path.exists(m):
    print("[FAIL] missing", m); sys.exit(2)

# lee y fuerza numérico
M = pd.read_csv(m, comment="#")
for col in ["t","divB_L2","Etot"]:
    M[col] = pd.to_numeric(M[col], errors="coerce")

# elimina filas vacías o no numéricas
M = M.dropna(subset=["t","divB_L2","Etot"])
if len(M) < 2:
    print("[WARN] not enough numeric rows."); sys.exit(0)

divB = M["divB_L2"].to_numpy()
Etot = M["Etot"].to_numpy()

print(f"[2D_MHD_TOY] divB_L2 last: {divB[-1]:.3e}")
print(f"[2D_MHD_TOY] Etot first -> last: {Etot[0]:.6e} -> {Etot[-1]:.6e}")

ok = np.isfinite(divB[-1]) and np.isfinite(Etot[-1]) and (Etot[-1] < 10*np.nanmax(Etot[:max(3,len(Etot))]))
print("[OK] sanity passed." if ok else "[WARN] energy grew too much or NaNs.")

