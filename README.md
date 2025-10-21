
# ZPinchSim — Minimal Resistive MHD Z‑Pinch Simulator (C++17)

This is a clean, from‑scratch scaffold to simulate the time evolution of a Z‑pinch in cylindrical coordinates.  
Scope: **resistive MHD** with optional axial field `Bz0` to study **m=0 (sausage)** and **m=1 (kink)** instabilities.

## Roadmap
1) 0D LRC circuit + radially averaged pinch (sanity check).  
2) 1D (r) equilibrium builder + wall BCs.  
3) 2D (r,z) axisymmetric MHD with **Constrained Transport** and HLL family Riemann solver.  
4) Pseudo‑3D θ‑Fourier modes to capture **m=0** and **m=1** (sausage/kink) growth.  
5) Diagnostics: growth rates, energy spectra, line density, pinch radius, β, and q‑like metrics.

## Equations (non‑dimensional resistive MHD)
We solve in (r,z) with cylindrical source terms and Fourier modes for θ. Unknowns U = [ρ, ρv_r, ρv_z, ρv_θ, B_r, B_z, B_θ, E].  
- Continuity: ∂t ρ + ∇·(ρ v) = 0  
- Momentum: ∂t (ρ v) + ∇·(ρ v v + pI + (B^2/2)I − B B) = S_cyl  
- Induction (CT): ∂t B = −∇×E, with E = −v×B + η J, J = ∇×B  
- Energy: ∂t E + ∇·((E+p_t) v − (B·v) B + η J×B) = S_cyl  
Here p_t = p + B^2/2, ideal gas p = (γ−1)(E − ½ρ|v|^2 − ½|B|^2).

Optional terms later: Hall + viscosity + radiation losses.

## Numerics
- Finite Volume (MUSCL), 2nd‑order in space, TVD limiter.  
- Riemann: HLL (start) → HLLD (later).  
- Time: SSP RK2 (start) → RK3. CFL controlled.  
- **∇·B=0**: face‑centered B + Constrained Transport (Yee grid).  
- Cylindrical sources: well‑balanced discretization near r=0; axis BCs enforced.  
- Boundaries: axis (symmetry), outer wall (perfect conductor), z‑periodic or sponge.  
- Seeding instabilities: add perturbations `∝ cos(mθ + k z)` with small ε.

## Layout
- `include/` headers for grid, state, physics, riemann, CT, IO.
- `src/` implementations and `main.cpp` (loads `configs/run.yaml`).
- `viz/` Python scripts to plot slices and monitor growth rates.
- `data/` output `.npz` snapshots for quick Python consumption.
- `tests/` tiny unit/integration tests (e.g., grid shape, CFL step).

## Build & run
```bash
mkdir -p build && cd build
cmake .. && cmake --build . -j
./zpinch_run ../configs/run.yaml
```
Outputs to `data/`:
- `fields_0000.npz, ...` arrays: rho, vr, vz, vtheta, Br, Bz, Btheta, p.
- `diag.csv` time series (radius, energies, growth rates).

## First experiments
- **Equilibrium check**: `Bθ ~ μ0 I / (2πr)` with small `Bz0`; no perturbation → should hold.  
- **Sausage (m=0)**: ε·cos(kz) density or current perturbation.  
- **Kink (m=1)**: helical velocity/B perturbation with small ε.

> This scaffold compiles and runs a *toy* update loop that writes dummy arrays. Fill in physics gradually following the TODOs in code.


## Step 0: Debug & Units
- Added `utils` with `DebugFrame`, `divB_L2`, `total_energy`, `count_nans`.
- Sanity script: `viz/check_sanity.py` reads `data/debug/debug_step_####.json`.
