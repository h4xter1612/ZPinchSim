
import pandas as pd, numpy as np, os, sys, glob

def main():
    metrics = "./data/debug/2d_ct_metrics.csv"
    if not os.path.exists(metrics):
        print("[FAIL] missing", metrics); sys.exit(2)
    M = pd.read_csv(metrics)
    if len(M) < 3:
        print("[WARN] not enough metric rows, run longer."); sys.exit(0)
    divB = M["divB_L2"].values
    Emag = M["Emag"].values
    tol = max(1e-12, 1e-9*np.max(Emag))
    noninc = np.all(np.diff(Emag) <= tol)
    print(f"[2D_CT] divB_L2 last: {divB[-1]:.3e}")
    print(f"[2D_CT] Emag first -> last: {Emag[0]:.6e} -> {Emag[-1]:.6e}")
    print(f"[2D_CT] non-increasing energy: {noninc}")
    if np.isfinite(divB[-1]) and divB[-1] < 1e-6:
        print("[OK] CT divergence small.")
    else:
        print("[WARN] CT divergence not small; check dt/grid.")
    if noninc:
        print("[OK] Magnetic energy non-increasing (with eta_ct>=0).")
    else:
        print("[NOTE] Energy may change due to advection by synthetic flow (eta_ct=0). Try eta_ct>0 for damping.")
if __name__ == "__main__":
    main()
