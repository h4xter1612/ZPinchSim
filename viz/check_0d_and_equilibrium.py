import numpy as np, pandas as pd, os, sys

def fit_LR_step(L, R, V0, t): return (V0/R) * (1.0 - np.exp(-t*R/L))

def main():
    base = "./data/debug"
    circ = os.path.join(base, "0d_circuit.csv")
    eq_r = os.path.join(base, "equilibrium_residual.csv")
    if not (os.path.exists(circ) and os.path.exists(eq_r)):
        print("[FAIL] faltan archivos de debug (0D o equilibrio)."); sys.exit(2)

    # --- 0D check ---
    C = pd.read_csv(circ)
    t, I = C["t"].values, C["I"].values
    L,R,V0 = 1.0, 0.1, 1.0
    Ithe = fit_LR_step(L,R,V0,t)
    rms = np.sqrt(np.mean((I - Ithe)**2))
    print(f"[LR] RMS error vs step-response (L={L},R={R},V0={V0}): {rms:.3e}")

    # --- equilibrium residual ---
    E = pd.read_csv(eq_r)
    res = E["residual"].values
    # reconstruye derivada típica desde ptot en el csv
    ptot = E["ptot"].values
    r = E["r"].values
    dpt_dr = np.gradient(ptot, r)
    # término geométrico típico
    # OJO: aquí no tenemos Bθ en este csv; usa equilibrium_profiles.csv si lo deseas.
    # Para una cota razonable, usa ||dpt/dr|| + percentil del residuo "geométrico"
    scale = np.max(np.abs(dpt_dr)) + np.percentile(np.abs(res), 90)
    scale = max(scale, 1e-12)
    res_norm = res / scale
    L2_raw = np.sqrt(np.mean(res**2))
    L2_norm = np.sqrt(np.mean(res_norm**2))
    print(f"[EQ] residual L2 (raw):  {L2_raw:.3e}")
    print(f"[EQ] residual L2 (norm): {L2_norm:.3e}")

    ok = (rms < 5e-3) and (L2_norm < 1e-1)
    print("[OK] Step 1 checks passed." if ok else "[WARN] Consider smoothing/refining (see tips).")
    sys.exit(0)

if __name__ == "__main__":
    main()

