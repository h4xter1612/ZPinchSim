#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from _modes_core import compute_growth_rate  # ya lo tienes

def slurp(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def make_config_from_template(tmpl_path: Path, out_path: Path, bz0: float, out_dir: Path):
    s = slurp(tmpl_path)
    s = s.replace("${Bz0}", f"{bz0}")
    s = s.replace("${OUT}", str(out_dir).replace("\\","/"))
    out_path.write_text(s, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Paso 6: barrido de Bz0 y análisis de crecimiento modal")
    ap.add_argument("--exe", default="./zpinch_run.exe", help="Ruta al ejecutable")
    ap.add_argument("--template", default="./configs/run2d_modes_template.yaml",
                    help="YAML plantilla con ${Bz0} y ${OUT}")
    ap.add_argument("--out-root", default="./data_runs/Bz_sweep", help="Carpeta base para resultados")
    ap.add_argument("--bzs", default="0.05,0.1,0.2,0.5,1,2", help="Lista de Bz0 separados por coma")
    ap.add_argument("--dirs", default="", help="(Opcional) alias de carpetas (mismo orden que --bzs), p.ej. Bzlow,Bzhigh")
    ap.add_argument("--field", default="vr")
    ap.add_argument("--k", type=float, default=40.0)
    ap.add_argument("--r-over-R", type=float, default=0.3)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--only-analyze", action="store_true", help="Solo analizar carpetas existentes (no correr simulaciones)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    exe = Path(args.exe)
    tmpl = Path(args.template)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bz_vals = [float(x) for x in args.bzs.split(",")]
    dir_aliases = [d for d in args.dirs.split(",") if d] if args.dirs else []
    if dir_aliases and len(dir_aliases) != len(bz_vals):
        print("[ERROR] --dirs y --bzs deben tener la misma cantidad de elementos.")
        sys.exit(2)

    rows = []
    for idx, bz in enumerate(bz_vals):
        run_dir = (out_root / dir_aliases[idx]) if dir_aliases else (out_root / f"Bz{bz}")
        cfg_path = out_root / f"run_Bz{bz}.yaml"

        if not args.only_analyze:
            make_config_from_template(tmpl, cfg_path, bz, run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [str(exe), str(cfg_path)]
            if not args.quiet:
                print("[RUN]", " ".join(cmd))
            # Si quiet=True, escondemos stdout/err; si no, dejamos pasar a consola
            ret = subprocess.run(cmd, capture_output=args.quiet)
            if ret.returncode != 0:
                print(f"[FAIL] run Bz={bz} rc={ret.returncode}")
                if args.quiet:
                    try:
                        print(ret.stdout.decode("utf-8","ignore"))
                        print(ret.stderr.decode("utf-8","ignore"))
                    except Exception:
                        pass
                rows.append({"Bz0": bz, "ok": False, "reason": f"run failed rc={ret.returncode}"})
                continue

        if not run_dir.exists():
            rows.append({"Bz0": bz, "ok": False, "reason": f"run_dir not found: {run_dir}"})
            if not args.quiet:
                print(f"[SKIP] {run_dir} no existe")
            continue

        res = compute_growth_rate(run_dir, r_over_R=args.r_over_R, field=args.field,
                                  k_target=args.k, fit_frac=args.fit_frac)
        if not res.get("ok", False):
            row = {"Bz0": bz, "ok": False, "reason": res.get("reason","")}
        else:
            row = {"Bz0": bz, "ok": True,
                   "growth_rate": res["growth_rate"],
                   "k_used": res["k_used"],
                   "snapshots": res["snapshots"],
                   "last_amp": res["last_amp"]}
        rows.append(row)
        if not args.quiet:
            print("[Bz]", bz, row)

    df = pd.DataFrame(rows)
    csv_path = out_root / "step6_growth_vs_Bz0.csv"
    df.to_csv(csv_path, index=False)
    if not args.quiet:
        print("Saved:", csv_path)

    # Plot rápido
    try:
        import matplotlib.pyplot as plt
        good = df[df["ok"]==True]
        if len(good)>0:
            plt.figure()
            plt.plot(good["Bz0"], good["growth_rate"], marker="o")
            plt.xlabel("Bz0")
            plt.ylabel("growth_rate (1/s)")
            plt.title("Step6: growth_rate vs Bz0")
            fig_path = out_root / "step6_growth_vs_Bz0.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            if not args.quiet:
                print("Saved:", fig_path)
    except Exception as e:
        if not args.quiet:
            print("[WARN] plot failed:", e)

if __name__ == "__main__":
    main()

