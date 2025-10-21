
import json, glob, sys, os

def main():
    debug_files = sorted(glob.glob(os.path.join("data","debug","debug_step_*.json")))
    if not debug_files:
        print("[FAIL] No debug JSON files found in data/debug/. Did you run the code?")
        sys.exit(1)
    last = debug_files[-1]
    with open(last) as f:
        J = json.load(f)
    ok = True
    if J.get("nan_count", 1) != 0:
        print(f"[FAIL] Found NaNs: nan_count={J['nan_count']} in {last}")
        ok = False
    cfl = J.get("cfl", None)
    if cfl is None or not (0.0 < cfl <= 1.0):
        print(f"[FAIL] CFL out of range: {cfl} in {last}")
        ok = False
    divB = J.get("divB_L2", None)
    if divB is None or divB != divB:  # NaN check
        print(f"[FAIL] divB_L2 invalid: {divB}")
        ok = False
    print(f"[INFO] Checked {last}: CFL={cfl}, divB_L2={divB}, nan_count={J.get('nan_count')}")
    if ok:
        print("[OK] Sanity check passed.")
        sys.exit(0)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
