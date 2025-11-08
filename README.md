# ZPinchSim — Minimal Resistive MHD Z‑Pinch Simulator (C++17)

![C++17](https://img.shields.io/badge/Language-C++17-blue)  ![Python](https://img.shields.io/badge/Visualization-Python-green)  
![Plasma Physics](https://img.shields.io/badge/Physics-Plasma_Physics-red)

---

## Overview

**ZPinchSim** is a clean, from‑scratch scaffold to simulate the time evolution of a **Z‑pinch** in cylindrical coordinates. The present scope is **resistive MHD** in \((r,z)\) with optional axial field \(B_z\) and seeded perturbations to study the onset and nonlinear behavior of the classic **sausage** \((m=0)\) and **kink** \((m=1)\) instabilities. A **stable** reference mode is also provided for baselines and regression tests.

The simulator outputs compact CSV snapshots that are consumed by Python utilities to compute growth‑rate diagnostics and to synthesize cross‑section renders and MP4s of the evolving modes.

> **Quick start:** run the C++ executable with one of the YAML configs, then run the Python diagnostics and visualization. See **Execution Flow** and **Commands** below.

---

## Physics Model

We solve single‑fluid **resistive MHD** in \((r,z)\) with cylindrical source terms and face‑centered magnetic field for \(\nabla\cdot \mathbf{B}=0\) enforcement via Constrained Transport (CT). Unknowns in conservative form:

\[
\mathbf{U} = \left[\rho,\; \rho v_r,\; \rho v_z,\; \rho v_\theta,\; B_r,\; B_z,\; B_\theta,\; E \right].
\]

**Equations (in SI, written schematically):**
- **Continuity:** \(\partial_t \rho + \nabla\cdot(\rho \mathbf{v}) = 0\)
- **Momentum:** \(\partial_t (\rho \mathbf{v}) + \nabla\cdot\big[\rho \mathbf{v}\mathbf{v} + (p + \tfrac{B^2}{2\mu_0})\mathbf{I} - \tfrac{1}{\mu_0}\mathbf{B}\mathbf{B}\big] = \mathbf{S}_{\mathrm{cyl}}\)
- **Induction (CT):** \(\partial_t \mathbf{B} = -\nabla\times\mathbf{E}\), with \(\mathbf{E} = -\mathbf{v}\times\mathbf{B} + \eta \mathbf{J}\), \(\mathbf{J} = \tfrac{1}{\mu_0}\nabla\times\mathbf{B}\)
- **Total Energy:** \(\partial_t E + \nabla\cdot\big[(E + p_t)\mathbf{v} - (\mathbf{B}\cdot\mathbf{v})\mathbf{B}/\mu_0 + \eta \mathbf{J}\times\mathbf{B}\big] = S_{\mathrm{cyl}}\),  
  with \(p_t = p + \tfrac{B^2}{2\mu_0}\), \(p=(\gamma-1)\big[E - \tfrac{1}{2}\rho|\mathbf{v}|^2 - \tfrac{|\mathbf{B}|^2}{2\mu_0}\big]\).

**Instability seeding:** small perturbations of the form \(\propto \cos(m\theta + kz)\) with amplitude \(\epsilon \ll 1\) to target sausage (\(m=0\)) or kink (\(m=1\)). Optional uniform \(B_z\) can modify stability boundaries and mode structure.

> Extensions on the roadmap (guarded with config toggles): explicit viscosity, Hall MHD, radiative losses, sponge layers.

---

## Numerics

- **Discretization:** Finite Volume (cell‑centered hydrodynamic conserved variables; face‑centered \( \mathbf{B}\)).  
- **Reconstruction:** MUSCL (2nd‑order), TVD limiter (see `include/reconstruction.hpp`).  
- **Riemann Solver:** HLL family (`include/rsolver.hpp`), structured to allow HLLD upgrade.  
- **Time Integration:** SSP‑RK2 (upgrade path to SSP‑RK3).  
- **Divergence Control:** **Constrained Transport** (Yee staggered mesh) to preserve \(\nabla\cdot\mathbf{B}=0\).  
- **Cylindrical Sources:** well‑balanced handling at \(r\to 0\); axis regularity and symmetry BCs.  
- **Boundaries:** axis (symmetry), outer wall (perfect conductor), periodic in \(z\) (default) or sponge.  
- **CFL:** configurable via YAML; automatic dt from fastest wavespeeds.

---

## Project Structure

```
ZPinchSim/
├── configs/
│   ├── helper.yaml       # Usage cheatsheet and parameter glossary
│   ├── kink.yaml         # m=1 seeded run
│   ├── sausage.yaml      # m=0 seeded run
│   └── stable.yaml       # unseeded/stable reference
├── include/              # Core headers (grid, state, physics, CT, Riemann, IO)
├── src/                  # C++ implementations and main.cpp
├── pyscripts/            # Python tools (run utils.py first)
│   ├── utils.py          # Dependency checker/installer
│   ├── diagnostics.py    # Growth‑rate & energy diagnostics
│   └── viz.py            # Synthetic renders and MP4 export
├── data/                 # Output folders per run (created at runtime)
├── CMakeLists.txt
└── README.md
```

Your current layout (Windows):

```
C:\Users\conej\Documents\Fusion\EPFL\Simulations\Git\ZPinchSim
├── configs\{helper.yaml,kink.yaml,sausage.yaml,stable.yaml}
├── include\{grid.hpp,io.hpp,physics.hpp,reconstruction.hpp,rsolver.hpp,state.hpp,utils.hpp}
├── pyscripts\{diagnostics.py,utils.py,viz.py}
├── src\{grid.cpp,io.cpp,main.cpp,physics.cpp,state.cpp,utils.cpp}
└── (plus build files, LICENSE, etc.)
```

---

## Configuration Files (`configs/*.yaml`)

Full usage instructions live in **`configs/helper.yaml`**. Each run YAML typically sets:

- **Grid/domain:** `Nr`, `Nz`, `Ng` (guard cells), `Rmax`, `Zmax`, periodicity.  
- **Physical params:** `gamma` (ratio of specific heats), `mu0`, resistivity `eta` (constant/profile).  
- **Initial state:** radial profiles for \(\rho, p, \mathbf{B}, \mathbf{v}\); optional axial field `Bz0`.  
- **Perturbation:** `m` (0 or 1), axial wavenumber `k`, small amplitude `eps`, random seed.  
- **Time control:** `t_end`, `cfl`, diagnostic cadence, snapshot frequency.  
- **Output:** target folder name, CSV/PNG toggles.

Use/modify:
- `configs/kink.yaml` → m=1 kink
- `configs/sausage.yaml` → m=0 sausage
- `configs/stable.yaml` → baseline (no growth expected)

---

## Execution Flow

1) **Compile the simulator** (CMake, C++17):
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j
```
The binary appears as `zpinch_run.exe` (Windows) or `zpinch_run` (Linux/macOS).

2) **Run a physical scenario** (choose a YAML):
```powershell
.\zpinch_run.exe .\configs\kink.yaml
.\zpinch_run.exe .\configs\sausage.yaml
.\zpinch_run.exe .\configs\stable.yaml
.\zpinch_run.exe .\configs\custom.yaml
```
Each run creates a folder under `data/` (e.g., `data\kink_m1_2p5d`, `data\sausage_m0_2p5d`, `data\stable_ref_2p5d`) with files like:
```
fields_<step>_<field>.csv
fields_<step>_meta.txt
```
where `<field>` ∈ { `p`, `rho`, `vr`, `Bth`, `Bz` } (and others as enabled).

3) **Prepare Python environment** (one‑time or whenever you change machines):
```bash
python pyscripts/utils.py            # installs numpy, matplotlib, imageio, imageio-ffmpeg
```
> You can run `python pyscripts/diagnostics.py` or `python pyscripts/viz.py` **without arguments** to see **Extended Help** at any time.

4) **Compute diagnostics** (growth rate, energies, RMS):
```powershell
python pyscripts\diagnostics.py --run-dir .\data\sausage_m0_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts\diagnostics.py --run-dir .\data\kink_m1_2p5d    --field p --mode volume --demean-z --save-spectrum
python pyscripts\diagnostics.py --run-dir .\data\stable_ref_2p5d --field p --mode volume --demean-z --save-spectrum
```
Artifacts per run:
- `diagnostics_Ak.csv`, `diagnostics_energy.csv`
- `diagnostics_fit_gamma.csv`, `diagnostics_lnA_fit.png`
- Optional `diagnostics_spectrum_k.csv` if `--save-spectrum` and `--k=0`

5) **Visualize** (synthetic slices + MP4 export):
```powershell
python pyscripts/viz.py --run-dir ./data/sausage_m0_2p5d --field p --mode auto \
    --amp-mode auto-logistic --logistic-power 1.8 --a-sat-frac 0.65 \
    --sausage-bounds 0.60 1.60 --sausage-ksoft 6.0 --sausage-beta 0.95 \
    --make-mp4 --fps 24

python pyscripts/viz.py --run-dir ./data/kink_m1_2p5d --field p --mode auto \
    --amp-mode auto-logistic --logistic-power 1.6 --a-sat-frac 0.60 \
    --make-mp4 --fps 24

python pyscripts\viz.py --run-dir .\data\kink_m1_2p5d --field p --mode auto \
    --amp-mode auto-logistic --a-sat-frac 0.5 --make-mp4 --fps 24

# Stable baseline:
# NOTE: if you don't have 'vizE.py', use viz.py with --mode stable (they are equivalent here).
python pyscripts\vizE.py --run-dir .\data\stable_ref_2p5d --field p --mode stable --make-mp4 --fps 24
# or
python pyscripts\viz.py  --run-dir .\data\stable_ref_2p5d --field p --mode stable --make-mp4 --fps 24
```

---

## What the Python Tools Do

- **`pyscripts/diagnostics.py`**  
  Computes \(A_k(t)\) either by **FFT projection** (default) or **direct cosine integrals**, volume‑weighted or at a chosen centerline radius. If `--k=0`, the dominant axial wavenumber is auto‑detected from the first snapshot. Performs a linear **fit of \(\ln A_k\)** over a configurable time window to report the growth rate \(\gamma\) and time constant \(\tau=1/\gamma\). Also integrates **magnetic, kinetic, and internal energies** and a volume RMS for the chosen field. Saves CSVs and a PNG of the \(\ln A_k\) fit.

- **`pyscripts/viz.py`**  
  Builds **synthetic cross‑section frames** for *kink*, *sausage*, or *stable* modes by re‑mapping the equilibrium field according to the instantaneous modal amplitude \(a(t)\) and phase \(\cos(k(z-ct))\). The amplitude can be:
  - `fixed`: \(a(t)=a_0\,e^{\gamma t}\) with user‑provided \(a_0,\gamma\),
  - `auto`: use \(\gamma\) from diagnostics or estimate it from early snapshots,
  - `auto-logistic`: integrates \(a'=\gamma\,a\,[1-(a/a_{\text{sat}})^p]\) to mimic nonlinear saturation,
  - `auto-energy`: simplified energy‑consistent scaling (same envelope as `auto` here).  
  Exports the last frame PNG and, if requested, an MP4 (`--make-mp4`).

Both scripts show **Extended Help** if you run them **without arguments**.

---

## Example Results (place your GIFs here)

> Save your animations as:
> - `src/sausage.gif`
> - `src/kink.gif`
> - `src/stable.gif`

**Sausage (m=0):** axial modulation of radius; the column periodically **narrows and widens**, changing cross‑sectional area. Expect exponential growth at early times if unstable; shear or line‑tying may reduce growth.

**Kink (m=1):** helical lateral **displacement of the column**; the core wobbles with a single helix pattern along \(z\). Sensitive to axial field and current profile; stabilization can occur for sufficient shear or safety factor analogs.

**Stable:** reference run with parameters below instability thresholds; serves to validate numerics and boundary conditions (no secular growth).

<p align="center">
  <img src="src/sausage.gif" alt="sausage mode" width="31%"/>
  <img src="src/kink.gif"    alt="kink mode"    width="31%"/>
  <img src="src/stable.gif"  alt="stable ref"   width="31%"/>
</p>

---

## Commands (copy‑paste)

**Simulator (Windows / PowerShell):**
```powershell
.\zpinch_run.exe .\configs\kink.yaml
.\zpinch_run.exe .\configs\sausage.yaml
.\zpinch_run.exe .\configs\stable.yaml
.\zpinch_run.exe .\configs\custom.yaml
```

**Diagnostics and Visualization:**
```powershell
python pyscripts\diagnostics.py --run-dir .\data\sausage_m0_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts/viz.py --run-dir ./data/sausage_m0_2p5d --field p --mode auto --amp-mode auto-logistic --logistic-power 1.8 --a-sat-frac 0.65 --sausage-bounds 0.60 1.60 --sausage-ksoft 6.0 --sausage-beta 0.95 --make-mp4 --fps 24

python pyscripts\diagnostics.py --run-dir .\data\kink_m1_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts/viz.py --run-dir ./data/kink_m1_2p5d --field p --mode auto --amp-mode auto-logistic --logistic-power 1.6 --a-sat-frac 0.60 --make-mp4 --fps 24
python pyscripts\viz.py --run-dir .\data\kink_m1_2p5d --field p  --mode auto --amp-mode auto-logistic --a-sat-frac 0.5 --make-mp4 --fps 24

python pyscripts\diagnostics.py --run-dir .\data\stable_ref_2p5d --field p --mode volume --demean-z --save-spectrum
python pyscripts\vizE.py --run-dir .\data\stable_ref_2p5d --field p --mode stable --make-mp4 --fps 24
# If vizE.py is not present, use:
# python pyscripts\viz.py --run-dir .\data\stable_ref_2p5d --field p --mode stable --make-mp4 --fps 24
```

---

## Output Files (per run)

- `fields_<step>_<field>.csv` — raw fields (center + guards trimmed by Python readers).  
- `fields_<step>_meta.txt` — metadata (`t, Nr, Nz, Ng, Rmax, Zmax`).  
- `diagnostics_Ak.csv` — time series of \(t, A_k\).  
- `diagnostics_energy.csv` — time series of \(E_B, E_K, E_{\mathrm{int}}, \mathrm{RMS}\).  
- `diagnostics_fit_gamma.csv` — fitted \(\gamma, \tau, R^2\) and fit window.  
- `diagnostics_lnA_fit.png` — plot of \(\ln A_k\) with linear fit.  
- `viz_*_last.png`, `viz_*.mp4` — last synthesized frame and optional animation.

---

## Tips & Validation

- Use **stable** config first to verify boundary conditions and \(\nabla\cdot\mathbf{B}\) preservation.  
- Compare early‑time \(\gamma\) from `diagnostics.py` to linear theory where applicable.  
- Enable `--demean-z` for cleaner axial spectral peaks and more robust \(A_k\).  
- Keep perturbation amplitude **small** to remain in the linear regime when measuring growth rates.  
- For MP4 export, having **ffmpeg** in your PATH improves compatibility.

---

## References

- **Primary (as requested):**  
  K. Tummel *et al.*, “Kinetic simulations of sheared flow stabilization in high‑temperature Z‑pinch plasmas,” *Physics of Plasmas* **26** (2019) 062506. DOI: 10.1063/1.5092241.

- **Books (suggested):**  
  - F. F. Chen, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed., Springer.  
  - J. P. Freidberg, *Ideal MHD*, Cambridge University Press.

> These sources contextualize MHD modeling, mode stability, and the role of shear flow in Z‑pinches; the code here focuses on a resistive‑MHD fluid picture with clean numerics and diagnostics.

---

## License

MIT © 2025 Juan Pablo Solís Ruiz

---

## Contact

- **Author:** Juan Pablo Solís Ruiz  
- **Email:** jp.sruiz18.tec@gmail.com  
- **GitHub:** h4xter1612

