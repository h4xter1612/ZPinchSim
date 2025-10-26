#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
from _modes_core import read_yaml_knobs, measure_growth

def main():
    ap = argparse.ArgumentParser(
        description="Paso 5: comparar crecimiento de modos entre Bzhigh y Bzlow."
    )
    ap.add_argument("--high-dir", default="data_runs/Bzhigh", help="Carpeta data para Bzhigh")
    ap.add_argument("--low-dir",  default="data_runs/Bzlow",  help="Carpeta data para Bzlow")
    ap.add_argument("--field", default=None, help="Override del campo (vr|Bth|p|rho)")
    ap.add_argument("--k", type=float, default=None, help="Override de k_target [rad/m]")
    ap.add_argument("--r", type=float, default=None, help="Override de r_over_R [0..1]")
    ap.add_argument("--quiet", action="store_true", help="Menos verbosidad")
    ap.add_argument("--out-csv", default=None, help="Si se indica, guarda resultados a CSV")
    args = ap.parse_args()

    base = Path(".")
    r0, f0, k0 = read_yaml_knobs(base, quiet=args.quiet)

    r_over_R = args.r if args.r is not None else r0
    field    = args.field if args.field is not None else f0
    k_target = args.k if args.k is not None else k0

    high_dir = Path(args.high_dir)
    low_dir  = Path(args.low_dir)

    if not high_dir.exists():
        print(f"[FAIL] no existe {high_dir}")
        return
    if not low_dir.exists():
        print(f"[FAIL] no existe {low_dir}")
        return

    # Mide ambos
    res_high = measure_growth(high_dir, r_over_R, field, k_target, quiet=args.quiet)
    res_low  = measure_growth(low_dir,  r_over_R, field, k_target, quiet=args.quiet)

    # DataFrame para comparación
    df = pd.DataFrame([
        {"run":"Bzhigh", **res_high},
        {"run":"Bzlow",  **res_low},
    ])

    # Salida concisa
    def fmt_row(row):
        return (
            f"[MODES] {row['run']:6s} | field={row['field']} r/R={row['r_over_R']:.3f} "
            f"snaps={int(row['snaps'])}  k_target={row['k_target']:.3f} "
            f"k_used≈{row['k_used']:.3f}  mode={row['mode']}  "
            f"growth={row['growth_rate']:.3e}  last_amp={row['last_amp']:.3e}"
        )

    print(fmt_row(df.iloc[0]))
    print(fmt_row(df.iloc[1]))

    # Diferencia de crecimiento
    if pd.notna(df.loc[0,"growth_rate"]) and pd.notna(df.loc[1,"growth_rate"]):
        dg = df.loc[0,"growth_rate"] - df.loc[1,"growth_rate"]
        print(f"[COMPARE] Δgrowth (high - low) = {dg:.3e} 1/s")

    # CSV opcional
    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        if not args.quiet:
            print(f"[INFO] guardado {outp}")

if __name__ == "__main__":
    main()

