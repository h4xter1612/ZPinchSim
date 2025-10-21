import pandas as pd, numpy as np, os, sys, glob
def main():
    metrics="./data/debug/1d_r_metrics.csv"
    if not os.path.exists(metrics):
        print("[FAIL] missing", metrics); sys.exit(2)
    M=pd.read_csv(metrics)
    if len(M)<3:
        print("[WARN] not enough metric rows, run longer or change output_every."); sys.exit(0)
    Emag=M["Emag"].values; diffs=np.diff(Emag)
    tol = max(1e-12, 1e-9 * np.max(Emag))  # tolerancia relativa chiquita
    noninc = np.all(np.diff(Emag) <= tol)
    # noninc = np.all(diffs<=1e-10)
    print(f"[1D_R] Emag first -> last: {Emag[0]:.6e} -> {Emag[-1]:.6e}")
    print(f"[1D_R] non-increasing energy: {noninc}")
    divB = M["divB_L2"].values
    print(f"[1D_R] divB_L2 last: {divB[-1]:.3e}")
    if noninc and np.isfinite(divB[-1]) and divB[-1] < 1e-2:
        print("[OK] 1D_R diffusion sanity passed.")
    else:
        print("[WARN] 1D_R sanity suggests tuning dt/Nr/eta.")
if __name__=="__main__": main()
