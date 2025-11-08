// =============================
// physics.cpp — OpenMP-friendly (MSVC), no 'omp simd', identical numerics
// =============================
/**
 * @file physics.cpp
 * @brief 2D axisymmetric resistive MHD toy solver (r–z) core time integrator and helpers.
 *
 * @details
 * ### Scope
 * This translation unit implements:
 *  - Problem initialization (equilibrium-like radial pressure from Bθ(r), standard tests).
 *  - Optional injected axial flow profiles v_z(r).
 *  - Randomized/controlled modal seeding for perturbations (m=0/1).
 *  - Axisymmetric boundary conditions in r and selectable periodic/non-periodic BCs in z.
 *  - A constrained-transport–like update for (Br, Bz) via the azimuthal electric field Eθ.
 *  - An explicit advection-diffusion update for the azimuthal field Bθ.
 *  - Directional Godunov sweeps (r then z) using slope-limited reconstructions and HLL fluxes.
 *  - Selective KO (Kreiss–Oliger–like) smoothing near domain edges and a sponge layer.
 *  - Lightweight diagnostics: energy in Bθ, amplitude of a chosen axial mode k, and CSV metrics.
 *
 * ### Geometry & Units
 * - Cylindrical symmetry with azimuthal invariance (∂/∂θ = 0). State variables are cell-centered.
 * - Grid geometry, strides, and ghost extents are provided by Fields::g.
 * - Typical MHD units with μ₀ = 1; total pressure used in Riemann solves adds ½ Bθ² to p.
 *
 * ### Numerics (high level)
 * - MUSCL-type reconstruction with slope limiters (Minmod/MC).
 * - HLL fluxes in r and z; radial sweep uses conservative form for r-weighted fluxes.
 * - CT-like update for (Br,Bz) using Eθ = −(v×B)_θ + η_CT(∂Bz/∂r − ∂Br/∂z).
 * - Explicit Bθ advection + resistive diffusion in r and z with cylindrical corrections.
 * - Explicit time stepping with adaptive dt from CFL using fast magnetosonic estimate.
 *
 * @note All parallel loops are guarded by a soft threshold to avoid oversubscription on small grids.
 * @warning This is a pedagogical “toy” code; stability is improved by edge KO filters + sponge.
 */

#include "physics.hpp"
#include "rsolver.hpp"
#include "reconstruction.hpp"
#include "io.hpp"
#include "utils.hpp"

#include <chrono>
#include <sstream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
  #include <omp.h>
#else
  // Fallback stubs to keep code single-source even without OpenMP.
  inline int omp_get_max_threads(){ return 1; }
  inline int omp_get_num_threads(){ return 1; }
  inline int omp_get_thread_num(){ return 0; }
#endif

namespace physics {

static constexpr double PI = 3.141592653589793238462643383279502884;

/**
 * @brief Heuristic to decide if we should spawn OpenMP threads.
 * @param Nr Number of physical radial cells (no ghosts).
 * @param Nz Number of physical axial cells (no ghosts).
 * @return True if (Nr*Nz) exceeds a soft threshold and OpenMP has >1 threads available.
 * @note Prevents thread overhead from dominating small runs.
 */
static inline bool should_parallelize(std::size_t Nr, std::size_t Nz) {
    const std::size_t Ncrit = 30000; // ~3e4 cells
    return (Nr * Nz) >= Ncrit && omp_get_max_threads() > 1;
}
static constexpr int OMP_CHUNK = 16;

// ============================================================================
// Problem initialization
// ============================================================================

/**
 * @brief Construct initial state: background Bz, prescribed Bθ(r), pressure p(r) from radial balance.
 *
 * @details
 * The azimuthal magnetic field Bθ(r) is prescribed as:
 * \f[
 *   B_\theta(r) = B_{\theta0} \frac{x}{1+x^2}, \quad x = \frac{r}{R_0}.
 * \f]
 * A radially varying pressure is then integrated from \f$r=0\f$ by discretizing the MHD
 * radial balance (neglecting inertia at t=0):
 * \f[
 *   \frac{dp}{dr} + B_\theta \frac{dB_\theta}{dr} + \frac{B_\theta^2}{r} = 0,
 * \f]
 * using first/central differences and a mild smoothing pass to remove staircasing.
 * Density is uniform and velocities are zeroed, except for later optional v_z injection.
 *
 * For selected test problems (e.g., "blast", "brio_wu") this routine overrides parts
 * of the above to match canonical initial conditions in z.
 *
 * @param F   In/out field arrays to be filled at all (r,z) including ghosts.
 * @param cfg Run-wide configuration (Bz0 used here).
 * @param mhd Physics configuration (problem selection, γ, etc.).
 */
static void init_problem(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhd){
    const auto& g = F.g;

    // --- Baseline “equilibrium-like” profile parameters (toy) ---
    const double rho0  = 1.0;
    const double p_axis= 1.0e-2;
    const double R0    = 0.3 * g.Rmax;
    const double Bz_bg = cfg.phys.Bz0;
    const double Bth0  = 0.15;

    const std::size_t NrT = g.size_r();
    const std::size_t NzT = g.size_z();
    std::vector<double> rcoord(NrT, 0.0), Bth_r(NrT, 0.0), p_r(NrT, 0.0);

    // Cell-center radii (ghosts included)
    for (std::size_t i=0; i<NrT; ++i){
        rcoord[i] = (int(i)-int(g.Ng)+0.5)*g.dr;
    }
    // Prescribed Bθ(r)
    for (std::size_t i=0; i<NrT; ++i){
        const double r = std::fabs(rcoord[i]);
        const double x = r / (R0 + 1e-12);
        Bth_r[i] = Bth0 * (x / (1.0 + x*x));
    }

    // Integrate dp/dr from the axis using the discretized balance
    p_r[0] = p_axis;
    auto dBth_dr = [&](std::size_t i)->double{
        if (i==0)         return (Bth_r[1] - Bth_r[0]) / g.dr;
        if (i==NrT-1)     return (Bth_r[NrT-1] - Bth_r[NrT-2]) / g.dr;
        return (Bth_r[i+1] - Bth_r[i-1]) / (2.0*g.dr);
    };
    for (std::size_t i=1; i<NrT; ++i){
        const double rL = std::max(std::fabs(rcoord[i-1]), 0.5*g.dr);
        const double rC = std::max(std::fabs(rcoord[i  ]), 0.5*g.dr);
        const double Bm   = Bth_r[i-1];
        const double dBdr = dBth_dr(i-1);
        const double B2_r = (Bm*Bm) / rL;
        const double dp = ( - Bm*dBdr - B2_r ) * (rC - rL);
        p_r[i] = std::max(1e-8, p_r[i-1] + dp);
    }
    // Mild smoothing to reduce small oscillations
    {
        std::vector<double> tmp = p_r;
        for (std::size_t i=1; i+1<NrT; ++i){
            p_r[i] = 0.25*tmp[i-1] + 0.5*tmp[i] + 0.25*tmp[i+1];
        }
    }

    // Fill 2D arrays from 1D profiles (and/or selected tests)
    if (should_parallelize(g.Nr, g.Nz)) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static, OMP_CHUNK)
            for (int ii=int(0); ii<int(NrT); ++ii){
                for (std::size_t k=0; k<NzT; ++k){
                    const std::size_t id = g.idx(ii,k);
                    F.rho[id] = rho0;
                    F.vr [id] = 0.0;
                    F.vz [id] = 0.0;
                    F.vth[id] = 0.0;
                    F.Br [id] = 0.0;
                    F.Bz [id] = Bz_bg;
                    F.Bth[id] = Bth_r[ii];
                    F.p  [id] = p_r[ii];
                }
            }
        }
    } else {
        for (std::size_t i=0; i<NrT; ++i){
            for (std::size_t k=0; k<NzT; ++k){
                const std::size_t id = g.idx(i,k);
                F.rho[id] = rho0;
                F.vr [id] = 0.0;
                F.vz [id] = 0.0;
                F.vth[id] = 0.0;
                F.Br [id] = 0.0;
                F.Bz [id] = Bz_bg;
                F.Bth[id] = Bth_r[i];
                F.p  [id] = p_r[i];
            }
        }
    }

    // Override for named problems (standard tests)
    if (mhd.problem=="blast"){
        if (should_parallelize(g.Nr, g.Nz)) {
            #pragma omp parallel
            {
                #pragma omp for schedule(static, OMP_CHUNK)
                for (int ii=int(0); ii<int(NrT); ++ii){
                    const double r = (int(ii)-int(g.Ng)+0.5)*g.dr;
                    for (std::size_t k=0;k<NzT;++k){
                        const double z = (int(k)-int(g.Ng)+0.5)*g.dz;
                        const std::size_t id = g.idx(ii,k);
                        const double rc = 0.2*std::min(g.Rmax, g.Zmax);
                        const double rr = std::sqrt(r*r + (z-0.5*g.Zmax)*(z-0.5*g.Zmax));
                        F.rho[id] = 1.0;
                        F.p  [id] = (rr<rc)? 1e-1 : 1e-2;
                        F.Br [id] = 0.0;
                        F.Bz [id] = Bz_bg;
                        F.Bth[id] = 0.0;
                        F.vr [id] = F.vz[id] = F.vth[id] = 0.0;
                    }
                }
            }
        } else {
            for (std::size_t i=0;i<NrT;++i){
                const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
                for (std::size_t k=0;k<NzT;++k){
                    const double z = (int(k)-int(g.Ng)+0.5)*g.dz;
                    const std::size_t id = g.idx(i,k);
                    const double rc = 0.2*std::min(g.Rmax, g.Zmax);
                    const double rr = std::sqrt(r*r + (z-0.5*g.Zmax)*(z-0.5*g.Zmax));
                    F.rho[id] = 1.0;
                    F.p  [id] = (rr<rc)? 1e-1 : 1e-2;
                    F.Br [id] = 0.0;
                    F.Bz [id] = Bz_bg;
                    F.Bth[id] = 0.0;
                    F.vr [id] = F.vz[id] = F.vth[id] = 0.0;
                }
            }
        }
    } else if (mhd.problem=="brio_wu"){
        if (should_parallelize(g.Nr, g.Nz)) {
            #pragma omp parallel
            {
                #pragma omp for schedule(static, OMP_CHUNK)
                for (int ii=int(0); ii<int(NrT); ++ii){
                    for (std::size_t k=0;k<NzT;++k){
                        const double z = (int(k)-int(g.Ng)+0.5)*g.dz;
                        const std::size_t id = g.idx(ii,k);
                        F.rho[id] = (z < 0.5*g.Zmax)? 1.0 : 0.125;
                        F.p  [id] = (z < 0.5*g.Zmax)? 1.0 : 0.1;
                        F.Br [id] = 0.0;
                        F.Bz [id] = Bz_bg;
                        F.Bth[id] = 0.0;
                        F.vr [id] = F.vz[id] = F.vth[id] = 0.0;
                    }
                }
            }
        } else {
            for (std::size_t i=0;i<NrT;++i){
                for (std::size_t k=0;k<NzT;++k){
                    const double z = (int(k)-int(g.Ng)+0.5)*g.dz;
                    const std::size_t id = g.idx(i,k);
                    F.rho[id] = (z < 0.5*g.Zmax)? 1.0 : 0.125;
                    F.p  [id] = (z < 0.5*g.Zmax)? 1.0 : 0.1;
                    F.Br [id] = 0.0;
                    F.Bz [id] = Bz_bg;
                    F.Bth[id] = 0.0;
                    F.vr [id] = F.vz[id] = F.vth[id] = 0.0;
                }
            }
        }
    }

    // ---- Extended radial-balance diagnostics (for sanity checking) ----
    {
        double resid_L2 = 0.0, norm_L2 = 0.0, resid_Linf = 0.0;
        double dpdr_min = +1e300, dpdr_max = -1e300;

        // Location of |Bθ| peak (proxy for current ring)
        double Bth_abs_max = 0.0;
        std::size_t i_Bth_max = 0;
        for (std::size_t i = 0; i < NrT; ++i){
            double a = std::abs(Bth_r[i]);
            if (a > Bth_abs_max){ Bth_abs_max = a; i_Bth_max = i; }
        }
        const double r_Bth_max = (int(i_Bth_max) - int(g.Ng) + 0.5) * g.dr;

        for (std::size_t i = 1; i + 1 < NrT; ++i){
            const double r    = std::max(std::fabs(rcoord[i]), 0.5 * g.dr);
            const double dpdr = (p_r[i+1] - p_r[i-1]) / (2.0 * g.dr);
            const double dBdr = (Bth_r[i+1] - Bth_r[i-1]) / (2.0 * g.dr);
            const double R    = dpdr + Bth_r[i]*dBdr + (Bth_r[i]*Bth_r[i]) / r; // residual

            resid_L2   += R*R;
            norm_L2    += dpdr*dpdr;
            resid_Linf  = std::max(resid_Linf, std::abs(R));
            dpdr_min    = std::min(dpdr_min, dpdr);
            dpdr_max    = std::max(dpdr_max, dpdr);
        }
        const std::size_t N = std::max<std::size_t>(1, NrT - 2);
        resid_L2 = std::sqrt(resid_L2 / N);
        norm_L2  = std::sqrt(norm_L2  / N);
        const double rel_L2 = (norm_L2 > 0.0) ? (resid_L2 / norm_L2) : 0.0;

        // Characteristic speeds near the “center”
        const std::size_t ic = g.Ng + g.Nr/2;
        const double p0   = p_r[ic];
        const double rhoC = rho0;
        const double cs0  = std::sqrt(std::max(0.0, mhd.gamma * p0 / std::max(1e-12, rhoC)));
        const double B2_0 = cfg.phys.Bz0*cfg.phys.Bz0 + Bth_r[ic]*Bth_r[ic];
        const double vA0  = std::sqrt(B2_0 / std::max(1e-12, rhoC));
        const double cf0  = std::sqrt(cs0*cs0 + vA0*vA0);
        const double beta0= (B2_0 > 0.0) ? (2.0 * p0 / B2_0) : 0.0;

        // Initial CFL estimate (v≈0 at t=0)
        const double hmin     = std::min(g.dr, g.dz);
        const double denom    = std::max(1e-12, cf0);
        const double dt_cfl0  = mhd.cfl * hmin / denom;

        // Mesh info
        std::cerr << std::setprecision(6) << std::scientific;
        std::cerr << "[INIT] grid: Nr=" << g.Nr << " Nz=" << g.Nz
                  << " dr=" << g.dr << " dz=" << g.dz
                  << " Rmax=" << g.Rmax << " Zmax=" << g.Zmax << "\n";

        // Balance diagnostics
        std::cerr << "[INIT] radial_balance_L2=" << resid_L2
                  << " (norm=" << norm_L2 << ", rel=" << rel_L2 << ")\n";
        std::cerr << "[INIT] radial_balance_Linf=" << resid_Linf << "\n";
        std::cerr << "[INIT] dp/dr range = [" << dpdr_min << ", " << dpdr_max << "]\n";

        // Fields & thermodynamics
        std::cerr << "[INIT] center: p0=" << p0
                  << " Bz0=" << cfg.phys.Bz0
                  << " Bth0=" << Bth_r[ic]
                  << " beta0=" << beta0
                  << " cs0=" << cs0
                  << " vA0=" << vA0
                  << " cf0=" << cf0 << "\n";

        // Bθ geometry
        std::cerr << "[INIT] Bth_peak=" << Bth_abs_max
                  << " at r=" << r_Bth_max << " (R0=" << R0 << ")\n";

        // Suggested initial dt by CFL
        std::cerr << "[INIT] CFL(dt0) ~ " << dt_cfl0 << "  (v≈0, using cf0)\n";
    }
}

/**
 * @brief Inject an axial flow profile v_z(r) after initialization.
 *
 * @details
 * Two optional shapes (set by mhd.flow.type):
 *  - "linear":   v_z(r) = v0 · min(|r|,R0)/R0
 *  - "localized":v_z(r) = v0 · exp(-0.5 ((r-R0)/σ)^2)
 *
 * Here v0 is specified in units of local sound speed at the domain center:
 *   v0 = (mhd.flow.v0) * c_s(center). The profile is copied uniformly along z.
 *
 * @param F    In/out fields (vz is modified).
 * @param cfg  Unused here (kept for symmetry).
 * @param mhd  Flow parameters (enable/type/v0/r0_frac/sigma_frac).
 */
static void apply_vz_profile(Fields& F, [[maybe_unused]] const RunConfig& cfg, const MHD2DConfig& mhd){
    const auto& g = F.g;
    if (mhd.flow.type=="off" || std::abs(mhd.flow.v0)<=0.0) return;

    const std::size_t NrT = g.size_r();

    const std::size_t ic = g.Ng + g.Nr/2, kc = g.Ng + g.Nz/2;
    const double p0   = std::max(1e-12, F.p[g.idx(ic,kc)]);
    const double rho0 = std::max(1e-12, F.rho[g.idx(ic,kc)]);
    const double cs0  = std::sqrt(std::max(0.0, mhd.gamma * p0 / rho0));
    const double v0   = mhd.flow.v0 * cs0;

    const double R0   = mhd.flow.r0_frac * g.Rmax;
    const double SIG  = std::max(1e-12, mhd.flow.sigma_frac * g.Rmax);

    const bool do_omp = should_parallelize(g.Nr, g.Nz);
    if (do_omp) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static, OMP_CHUNK)
            for (int ii=int(0); ii<int(NrT); ++ii){
                const double r = (int(ii)-int(g.Ng)+0.5)*g.dr;
                double vzr = 0.0;
                if (mhd.flow.type=="linear"){
                    const double rr = std::min(std::abs(r), R0);
                    vzr = v0 * (rr/(R0+1e-12));
                } else if (mhd.flow.type=="localized"){
                    const double ga = std::exp(-0.5*std::pow((r - R0)/SIG,2.0));
                    vzr = v0 * ga;
                }
                for (int kk=int(g.Ng); kk<int(g.Ng+g.Nz); ++kk){
                    F.vz[g.idx(ii, std::size_t(kk))] += vzr;
                }
            }
        }
    } else {
        for (std::size_t i=0;i<g.size_r();++i){
            const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
            double vzr = 0.0;
            if (mhd.flow.type=="linear"){
                const double rr = std::min(std::abs(r), R0);
                vzr = v0 * (rr/(R0+1e-12));
            } else if (mhd.flow.type=="localized"){
                const double ga = std::exp(-0.5*std::pow((r - R0)/SIG,2.0));
                vzr = v0 * ga;
            }
            for (std::size_t k=g.Ng;k<g.Ng+g.Nz;++k){
                F.vz[g.idx(i,k)] += vzr;
            }
        }
    }
}

// ============================================================================
// Small utilities (edge treatment, KO filters, etc.)
// ============================================================================

/**
 * @brief Optional wave-speed capping for stability experiments.
 * @param vmax_raw Measured characteristic + flow speed.
 * @param vmax_guard If >0, returns min(vmax_raw, vmax_guard); otherwise returns vmax_raw.
 */
static inline double cap_wave_speed(double vmax_raw, double vmax_guard){
    if (vmax_guard <= 0.0) return vmax_raw;
    return std::min(vmax_raw, vmax_guard);
}

/**
 * @brief Zero out reconstructed slopes near edges (indices [L,R) in 1D buffer).
 * @param darr Slope array to modify.
 * @param L    Left bound (inclusive).
 * @param R    Right bound (exclusive).
 * @note Helps to avoid spurious overshoots at ghost-adjacent interfaces.
 */
static inline void kill_edge_slopes(std::vector<double>& darr, std::size_t L, std::size_t R){
    if (R <= L) return;
    if (R - L >= 4) { darr[L+0] = 0.0; darr[R-1] = 0.0; }
    else { for (std::size_t q=L; q<R; ++q) darr[q] = 0.0; }
}

/**
 * @brief Apply a localized KO (5-point) filter near r-edges for several fields.
 * @param F  In/out fields (rho, vr, vz, p, Br, Bz, Bth filtered).
 * @param dt Time step size (scales viscosity).
 * @param fac Non-dimensional strength (default 0.012).
 * @details
 * The discrete operator along r for a fixed k is:
 * \f[
 *  \nu \,(-1, 4, -6, 4, -1) \star A(i)
 * \f]
 * applied only to two cells next to each physical boundary (if available).
 */
static inline void ko_filter_edges_r(Fields& F, double dt, double fac=0.012){
    const auto& g = F.g;
    if (g.Nr < 8) return;
    const double h  = g.dr;
    const double nu = fac * dt / (h + 1e-12);

    auto apply_ko = [&](std::vector<double>& A){
        std::vector<double> B = A;
        const std::size_t iL = g.Ng, iR = g.Ng + g.Nr - 1;

        auto KO = [&](std::size_t i, std::size_t k){
            auto I = [&](std::size_t ii){ return g.idx(ii,k); };
            return -B[I(i-2)] + 4.0*B[I(i-1)] - 6.0*B[I(i)] + 4.0*B[I(i+1)] - B[I(i+2)];
        };

        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            for (std::size_t i : {iL+1, iL+2, iR-2, iR-1}){
                if (i>=iL+2 && i+2<=iR) A[g.idx(i,k)] += nu*KO(i,k);
            }
        }
    };

    apply_ko(F.rho);
    apply_ko(F.vr);
    apply_ko(F.vz);
    apply_ko(F.p);
    apply_ko(F.Br);
    apply_ko(F.Bz);
    apply_ko(F.Bth);
}

/**
 * @brief Apply a localized KO (5-point) filter near z-edges for selected fields.
 * @param F  In/out fields (rho, vr, vz filtered here).
 * @param dt Time step size (scales viscosity).
 * @param fac Non-dimensional strength (default 0.012).
 */
static inline void ko_filter_edges_z(Fields& F, double dt, double fac=0.012){
    const auto& g = F.g;
    if (g.Nz < 8) return;
    const double h  = g.dz;
       const double nu = fac * dt / (h + 1e-12);

    auto apply_ko = [&](std::vector<double>& A){
        std::vector<double> B = A;
        const std::size_t kL = g.Ng, kR = g.Ng + g.Nz - 1;
        auto KO = [&](std::size_t i, std::size_t k){
            auto I = [&](std::size_t ii,std::size_t kk){ return g.idx(ii,kk); };
            return -B[I(i,k-2)] + 4.0*B[I(i,k-1)] - 6.0*B[I(i,k)] + 4.0*B[I(i,k+1)] - B[I(i,k+2)];
        };
        for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
            for (std::size_t k : {kL+1, kL+2, kR-2, kR-1}){
                if (k>=kL+2 && k+2<=kR) A[g.idx(i,k)] += nu*KO(i,k);
            }
        }
    };
    apply_ko(F.rho);
    apply_ko(F.vr);
    apply_ko(F.vz);
}

/**
 * @brief Seed user-controlled perturbation modes (m=0,1) in vr and/or Bθ.
 *
 * @details
 * For a given axial wavenumber k, phase φ = k z, and radial envelope
 * \f$ \mathrm{shape}_r = \exp\{-r^2/R_0^2\} \f$, we add:
 *  - If seed_vr:
 *    - m=0: vr += ε · shape_r · cos(φ)
 *    - m=1: vr1c += ε · (r/R0) · shape_r · cos(φ)   (stored in m=1 cosine slot)
 *  - If seed_bth:
 *    - m=0: Bθ *= (1 + 0.1ε · cos(φ))
 *    - m=1: Bθ1c += 0.1ε · Bθ · cos(φ)               (stored in m=1 cosine slot)
 *
 * @param F   In/out fields (vr, Bth or their m=1 components modified).
 * @param cfg Unused (kept for symmetry with other hooks).
 * @param m   Modal seeding parameters (enable, m, k, ε, r0_frac).
 */
static void seed_modes(Fields& F, [[maybe_unused]] const RunConfig& cfg, const MHD2DConfig& m){
    if (!m.modes.enable || (m.modes.eps<=0.0)) return;
    const auto& g = F.g;
    const double k = m.modes.k;
    const int    M = m.modes.m;
    const double eps = m.modes.eps;
    const double R0 = m.modes.r0_frac * g.Rmax;

    const bool do_omp = should_parallelize(g.Nr, g.Nz);
    if (do_omp) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static, OMP_CHUNK)
            for (int ii=int(0); ii<int(g.size_r()); ++ii){
                const double r = (int(ii)-int(g.Ng)+0.5)*g.dr;
                const double shape_r = std::exp(-(r*r)/(R0*R0+1e-16));
                for (int kk=int(0); kk<int(g.size_z()); ++kk){
                    const double z = (int(kk)-int(g.Ng)+0.5)*g.dz;
                    const std::size_t id = g.idx(std::size_t(ii), std::size_t(kk));
                    const double phase = k*z;
                    const double fz = std::cos(phase);

                    if (m.modes.seed_vr){
                        if (M==0){
                            F.vr[id] += eps * shape_r * fz;
                        } else if (M==1){
                            F.vr1c[id] += eps * (r/(R0+1e-12)) * shape_r * fz;
                        } else {
                            F.vr[id] += eps * shape_r * fz;
                        }
                    }
                    if (m.modes.seed_bth){
                        if (M==0){
                            F.Bth[id] *= (1.0 + 0.1*eps * fz);
                        } else if (M==1){
                            F.Bth1c[id] += 0.1 * eps * F.Bth[id] * fz;
                        } else {
                            F.Bth[id] *= (1.0 + 0.1*eps * fz);
                        }
                    }
                }
            }
        }
    } else {
        for (std::size_t i=0;i<g.size_r();++i){
            const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
            const double shape_r = std::exp(-(r*r)/(R0*R0+1e-16));
            for (std::size_t kidx=0;kidx<g.size_z();++kidx){
                const double z = (int(kidx)-int(g.Ng)+0.5)*g.dz;
                const std::size_t id = g.idx(i,kidx);
                const double phase = k*z;
                const double fz = std::cos(phase);

                if (m.modes.seed_vr){
                    if (M==0){
                        F.vr[id] += eps * shape_r * fz;
                    } else if (M==1){
                        F.vr1c[id] += eps * (r/(R0+1e-12)) * shape_r * fz;
                    } else {
                        F.vr[id] += eps * shape_r * fz;
                    }
                }
                if (m.modes.seed_bth){
                    if (M==0){
                        F.Bth[id] *= (1.0 + 0.1*eps * fz);
                    } else if (M==1){
                        F.Bth1c[id] += 0.1 * eps * F.Bth[id] * fz;
                    } else {
                        F.Bth[id] *= (1.0 + 0.1*eps * fz);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Boundary conditions and source-like terms
// ============================================================================

/**
 * @brief Axis boundary conditions in r (wall at iL, outflow-like at iR).
 *
 * @details
 * - At inner wall: vr=0, Br=0, Bθ=0; copy-outflow for (vz,p,ρ,Bz).
 * - At outer edge: vr=0, Br=0; copy-outflow for (vz,p,ρ,Bz,Bθ).
 * This is a simple, robust set of conditions for the toy problem.
 */
static inline void apply_bc_r(Fields& F){
    const auto& g = F.g;
    const std::size_t iL = g.Ng, iR = g.Ng + g.Nr - 1;
    for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
        F.vr [g.idx(iL,k)] = 0.0;
        F.Br [g.idx(iL,k)] = 0.0;
        F.Bth[g.idx(iL,k)] = 0.0;
        F.vz [g.idx(iL,k)] = F.vz[g.idx(iL+1,k)];
        F.p  [g.idx(iL,k)] = F.p [g.idx(iL+1,k)];
        F.rho[g.idx(iL,k)] = F.rho[g.idx(iL+1,k)];
        F.Bz [g.idx(iL,k)] = F.Bz[g.idx(iL+1,k)];

        F.vr [g.idx(iR,k)] = 0.0;
        F.vz [g.idx(iR,k)] = F.vz[g.idx(iR-1,k)];
        F.p  [g.idx(iR,k)] = F.p [g.idx(iR-1,k)];
        F.rho[g.idx(iR,k)] = F.rho[g.idx(iR-1,k)];
        F.Br [g.idx(iR,k)] = 0.0;
        F.Bz [g.idx(iR,k)] = F.Bz[g.idx(iR-1,k)];
        F.Bth[g.idx(iR,k)] = F.Bth[g.idx(iR-1,k)];
    }
}

/**
 * @brief Axisymmetric geometric source for vr from hoop-stress term (Bθ²/r)/ρ.
 * @param F  In/out fields (vr incremented).
 * @param dt Time increment.
 *
 * @details Adds:
 * \f[
 *   \Delta v_r \leftarrow \Delta t \,\frac{B_\theta^2}{r\,\rho}.
 * \f]
 * A minimal stabilizing source capturing inward/outward Lorentz forcing from Bθ curvature.
 */
static inline void apply_axisym_sources(Fields& F, double dt){
    const auto& g = F.g;
    for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
        const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        const double rinv = (r>0.0)? 1.0/r : 0.0;
        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            const std::size_t id = g.idx(i,k);
            const double rho = std::max(1e-12, F.rho[id]);
            const double Svr = (F.Bth[id]*F.Bth[id]) * rinv / rho;
            F.vr[id] += dt * Svr;
        }
    }
}

/**
 * @brief Sponge layer near outer r and endcaps in z; exponential damping of all components.
 * @param F   In/out fields.
 * @param cfg Unused (reserved).
 * @param m   Unused (reserved).
 * @param dt  Time step for scaling the damping factor.
 *
 * @details Damping factor is exp(-α f dt), where f∈[0,1] increases quadratically into the sponge.
 */
static inline void apply_sponge(Fields& F, [[maybe_unused]] const RunConfig& cfg, [[maybe_unused]] const MHD2DConfig& m, double dt){
    const auto& g = F.g;
    const double Rcut   = 0.88 * g.Rmax;
    const double Zcut   = 0.92 * g.Zmax;
    const double alpha0 = 30.0;

    for (std::size_t i=0;i<g.size_r();++i){
        const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        double fr = (r>Rcut)? std::min(1.0,(r-Rcut)/(g.Rmax-Rcut+1e-12)) : 0.0;

        for (std::size_t k=0;k<g.size_z();++k){
            const double z   = (int(k)-int(g.Ng)+0.5)*g.dz;
            const double z0  = 0.5*g.Zmax;
            const double dzb = std::max(0.0, std::fabs(z-z0) - Zcut);
            double fz = (dzb>0.0)? std::min(1.0, dzb/(0.5*g.Zmax - Zcut + 1e-12)) : 0.0;

            fr *= fr; fz *= fz;
            const double f = std::max(fr,fz);
            if (f<=0.0) continue;

            const std::size_t id = g.idx(i,k);
            const double damp = std::exp(-alpha0 * f * dt);

            F.vr[id]  *= damp;
            F.vz[id]  *= damp;
            F.vth[id] *= damp;
            F.Bth[id] *= damp;
            F.Br[id]  *= damp;
            F.Bz[id]  *= damp;
        }
    }
}

/**
 * @brief z-boundary conditions: periodic copy or Neumann copy-outflow into ghosts.
 * @param F         In/out fields.
 * @param periodic  If true, wrap-around (periodic); else extrapolate edge values.
 */
static inline void apply_bc_z(Fields& F, bool periodic){
    const auto& g = F.g;
    const std::size_t kL = g.Ng, kR = g.Ng + g.Nz - 1;
    for (std::size_t i=0; i<g.size_r(); ++i){
        if (periodic){
            for (std::size_t q=1; q<=g.Ng; ++q){
                const std::size_t kghostL = kL - q, ksrcL = kR - (q-1);
                const std::size_t kghostR = kR + q, ksrcR = kL + (q-1);
                F.rho[g.idx(i,kghostL)] = F.rho[g.idx(i,ksrcL)];
                F.vr [g.idx(i,kghostL)] = F.vr [g.idx(i,ksrcL)];
                F.vz [g.idx(i,kghostL)] = F.vz [g.idx(i,ksrcL)];
                F.p  [g.idx(i,kghostL)] = F.p  [g.idx(i,ksrcL)];
                F.Br [g.idx(i,kghostL)] = F.Br [g.idx(i,ksrcL)];
                F.Bz [g.idx(i,kghostL)] = F.Bz [g.idx(i,ksrcL)];
                F.Bth[g.idx(i,kghostL)] = F.Bth[g.idx(i,ksrcL)];

                F.rho[g.idx(i,kghostR)] = F.rho[g.idx(i,ksrcR)];
                F.vr [g.idx(i,kghostR)] = F.vr [g.idx(i,ksrcR)];
                F.vz [g.idx(i,kghostR)] = F.vz [g.idx(i,ksrcR)];
                F.p  [g.idx(i,kghostR)] = F.p  [g.idx(i,ksrcR)];
                F.Br [g.idx(i,kghostR)] = F.Br [g.idx(i,ksrcR)];
                F.Bz [g.idx(i,kghostR)] = F.Bz [g.idx(i,ksrcR)];
                F.Bth[g.idx(i,kghostR)] = F.Bth[g.idx(i,ksrcR)];
            }
        } else {
            for (std::size_t q=1; q<=g.Ng; ++q){
                const std::size_t kghostL = kL - q, kghostR = kR + q;
                F.rho[g.idx(i,kghostL)] = F.rho[g.idx(i,kL)];
                F.vr [g.idx(i,kghostL)] = F.vr [g.idx(i,kL)];
                F.vz [g.idx(i,kghostL)] = F.vz [g.idx(i,kL)];
                F.p  [g.idx(i,kghostL)] = F.p  [g.idx(i,kL)];
                F.Br [g.idx(i,kghostL)] = F.Br [g.idx(i,kL)];
                F.Bz [g.idx(i,kghostL)] = F.Bz [g.idx(i,kL)];
                F.Bth[g.idx(i,kghostL)] = F.Bth[g.idx(i,kL)];

                F.rho[g.idx(i,kghostR)] = F.rho[g.idx(i,kR)];
                F.vr [g.idx(i,kghostR)] = F.vr [g.idx(i,kR)];
                F.vz [g.idx(i,kghostR)] = F.vz [g.idx(i,kR)];
                F.p  [g.idx(i,kghostR)] = F.p  [g.idx(i,kR)];
                F.Br [g.idx(i,kghostR)] = F.Br [g.idx(i,kR)];
                F.Bz [g.idx(i,kghostR)] = F.Bz [g.idx(i,kR)];
                F.Bth[g.idx(i,kghostR)] = F.Bth[g.idx(i,kR)];
            }
        }
    }
}

// ============================================================================
// Magnetic updates: Bθ advection-diffusion; CT-like update for (Br,Bz)
// ============================================================================

/**
 * @brief Explicit update of Bθ with upwind advection and resistive diffusion in r and z.
 *
 * @param F          In/out fields (Bth updated in place).
 * @param dt         Time step.
 * @param eta_theta  Resistivity-like coefficient for Bθ diffusion.
 * @param periodic_z Unused in current stencil (kept for symmetry).
 *
 * @details
 * The update follows (schematically):
 * \f[
 *   B_\theta^{n+1} = B_\theta^n
 *     - \Delta t \left[\nabla\cdot(\mathbf{v} B_\theta)\right]
 *     + \Delta t \,\eta_\theta \left( \nabla^2 B_\theta - \frac{B_\theta}{r^2}
 *                                     + \frac{1}{r}\frac{\partial}{\partial r}
 *                                       \left(r \frac{\partial B_\theta}{\partial r}\right)\right)
 * \f]
 * where the cylindrical corrections are included via discrete terms (lap_r + lap_z).
 * Advection uses simple face upwinding with arithmetic-averaged velocities.
 */
static void update_Btheta_axisym(Fields& F, double dt, double eta_theta, bool periodic_z){
    (void)periodic_z;
    const auto& g = F.g;
    std::vector<double> Bth_new(F.Bth);

    for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
        double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        double rinv = (r>0.0)? 1.0/r : 0.0;

        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            const std::size_t id = g.idx(i,k);

            // Upwinded advection
            double vr_L = 0.5*(F.vr[g.idx(i-1,k)] + F.vr[g.idx(i,k)]);
            double vr_R = 0.5*(F.vr[g.idx(i,k)]   + F.vr[g.idx(i+1,k)]);
            double vz_D = 0.5*(F.vz[g.idx(i,k-1)] + F.vz[g.idx(i,k)]);
            double vz_U = 0.5*(F.vz[g.idx(i,k)]   + F.vz[g.idx(i,k+1)]);

            double B_L  = (vr_L>=0) ? F.Bth[g.idx(i-1,k)] : F.Bth[g.idx(i,k)];
            double B_R  = (vr_R>=0) ? F.Bth[g.idx(i,k)]   : F.Bth[g.idx(i+1,k)];
            double B_D  = (vz_D>=0) ? F.Bth[g.idx(i,k-1)] : F.Bth[g.idx(i,k)];
            double B_U  = (vz_U>=0) ? F.Bth[g.idx(i,k)]   : F.Bth[g.idx(i,k+1)];

            double Fr_L = B_L * vr_L;
            double Fr_R = B_R * vr_R;
            double Fz_D = B_D * vz_D;
            double Fz_U = B_U * vz_U;

            double adv = (Fr_R - Fr_L)/g.dr + (Fz_U - Fz_D)/g.dz + F.Bth[id]*F.vr[id]*rinv;

            // Diffusion with cylindrical correction
            double BrL = F.Bth[g.idx(i-1,k)], Bc = F.Bth[id], BrR = F.Bth[g.idx(i+1,k)];
            double BzD = F.Bth[g.idx(i,k-1)], BzU = F.Bth[g.idx(i,k+1)];
            double dBdr_L = (Bc - BrL)/g.dr;
            double dBdr_R = (BrR - Bc)/g.dr;
            double r_dBdr = r * (dBdr_R - dBdr_L)/g.dr;
            double lap_r  = rinv * r_dBdr - Bc * rinv * rinv;
            double lap_z  = (BzU - 2.0*Bc + BzD) / (g.dz*g.dz);
            double diff   = eta_theta * (lap_r + lap_z);

            double Bn = Bc - dt*adv + dt*diff;
            Bth_new[id] = std::isfinite(Bn) ? Bn : 0.0;
        }
    }
    F.Bth.swap(Bth_new);
}

/**
 * @brief CT-like update for (Br, Bz) using the azimuthal electric field Eθ.
 *
 * @param F          In/out fields (Br,Bz updated).
 * @param cfg        Unused here (symmetry).
 * @param dt         Time step.
 * @param eta_ct     Resistive term weight in Eθ.
 * @param periodic_z Pass-through to z-BC application after the update.
 *
 * @details
 * Discretization mirrors Faraday’s law in cylindrical form (axisymmetric):
 * \f[
 *   \partial_t B_r = -\partial_z E_\theta, \qquad
 *   \partial_t B_z = \frac{1}{r} \partial_r ( r E_\theta ).
 * \f]
 * with
 * \f[
 *   E_\theta = - (v \times B)_\theta + \eta_{ct}\left(\frac{\partial B_z}{\partial r} - \frac{\partial B_r}{\partial z}\right).
 * \f]
 * Centered approximations are used consistently with the storage layout.
 */
static void ct_update(Fields& F, [[maybe_unused]] const RunConfig& cfg, double dt, double eta_ct, bool periodic_z){
    (void)cfg;
    const auto& g = F.g;
    std::vector<double> E(g.size_r()*g.size_z(), 0.0);
    auto Eidx = [&](std::size_t i, std::size_t k){ return g.idx(i,k); };

    // Build Eθ at centers
    for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            double vr = F.vr[g.idx(i,k)];
            double vz = F.vz[g.idx(i,k)];
            double Brc= F.Br[g.idx(i,k)];
            double Bzc= F.Bz[g.idx(i,k)];
            double Etheta = -(vr*Bzc - vz*Brc);
            if (eta_ct>0.0){
                double dBz_dr = (F.Bz[g.idx(i,k)] - F.Bz[g.idx(i-1,k)]) / g.dr;
                double dBr_dz = (F.Br[g.idx(i,k)] - F.Br[g.idx(i,k-1)]) / g.dz;
                Etheta += eta_ct * (dBz_dr - dBr_dz);
            }
            E[Eidx(i,k)] = Etheta;
        }
    }

    // Update Br, Bz
    std::vector<double> Br_new(F.Br), Bz_new(F.Bz);
    for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
        double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        double rph = r + 0.5*g.dr;
        double rmh = std::max(r - 0.5*g.dr, 0.5*g.dr);
        double rinv = (r>0.0)? 1.0/r : 0.0;

        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            double dE_dz = (E[Eidx(i,k+1)] - E[Eidx(i,k)]) / g.dz;
            Br_new[g.idx(i,k)] = F.Br[g.idx(i,k)] - dt * dE_dz;

            double rE_rhs = ( rph*E[Eidx(i+1,k)] - rmh*E[Eidx(i,k)] ) / g.dr;
            Bz_new[g.idx(i,k)] = F.Bz[g.idx(i,k)] + dt * rinv * rE_rhs;
        }
    }

    F.Br.swap(Br_new);
    F.Bz.swap(Bz_new);

    // Neumann copies in r-ghosts for (Br,Bz), then enforce z-BCs
    for (std::size_t k=0;k<g.size_z();++k){
        F.Br[g.idx(g.Ng,k)]                       = F.Br[g.idx(g.Ng+1,k)];
        F.Br[g.idx(g.size_r()-g.Ng-1,k)]         = F.Br[g.idx(g.size_r()-g.Ng-2,k)];
        F.Bz[g.idx(g.Ng,k)]                       = F.Bz[g.idx(g.Ng+1,k)];
        F.Bz[g.idx(g.size_r()-g.Ng-1,k)]         = F.Bz[g.idx(g.size_r()-g.Ng-2,k)];
    }
    apply_bc_z(F, periodic_z);
}

// ============================================================================
// Directional sweeps (Godunov): r-sweep and z-sweep
// ============================================================================

/**
 * @brief Godunov sweep along r with cylindrical conservative update (r-weighted fluxes).
 *
 * @param F     In/out fields (ρ, vr, vz, p updated).
 * @param cfg   Unused here.
 * @param gamma Ratio of specific heats (γ).
 * @param lim   Slope limiter for reconstruction (Minmod/MC).
 * @param dt    Time step.
 *
 * @details
 * - Reconstruct MUSCL left/right states for (ρ, vr, vz, p+½Bθ², Br, Bz).
 * - Compute HLL fluxes in r using rsolver::hll_r; set magnetic fluxes (Br,Bz) to zero
 *   to avoid double-updating them (they are advanced by CT).
 * - Conservative update integrates r-weighted fluxes:
 *   \f[
 *     r U^{n+1} = r U^{n} - \frac{\Delta t}{\Delta r} \big( r_{i+1/2} F_{i+1/2}
 *                    - r_{i-1/2} F_{i-1/2} \big).
 *   \f]
 * - Convert back to primitives; apply simple floors and velocity clamps.
 */
static void mhd_sweep_r(Fields& F, [[maybe_unused]] const RunConfig& cfg, double gamma,
                        recon::Limiter lim, double dt){
    (void)cfg;
    const auto& g = F.g;
    const std::size_t i0 = g.Ng, i1 = g.Ng + g.Nr;
    const std::size_t k0 = g.Ng, k1 = g.Ng + g.Nz;
    constexpr double UMAX = 3e2;

    const bool do_omp = should_parallelize(g.Nr, g.Nz);

    if (do_omp) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static, OMP_CHUNK)
            for (int kk=int(k0); kk<int(k1); ++kk){
                std::vector<double> rho(g.size_r()), vr(g.size_r()), vz(g.size_r()),
                                    p(g.size_r()),   Br(g.size_r()), Bz(g.size_r()), Bth(g.size_r());
                // Gather line
                for (std::size_t i=i0; i<i1; ++i){
                    const std::size_t id = g.idx(i,std::size_t(kk));
                    rho[i] = std::max(1e-12, F.rho[id]);
                    vr[i]  = F.vr[id];
                    vz[i]  = F.vz[id];
                    p[i]   = std::max(1e-12, F.p[id] + 0.5*F.Bth[id]*F.Bth[id]); // add ½Bθ²
                    Br[i]  = F.Br[id];
                    Bz[i]  = F.Bz[id];
                    Bth[i] = F.Bth[id];
                }

                // Slopes
                std::vector<double> drho,dvr,dvz,dpv,dBr,dBz;
                recon::slope_limited(rho, drho, i0, i1, lim);
                recon::slope_limited(vr,  dvr,  i0, i1, lim);
                recon::slope_limited(vz,  dvz,  i0, i1, lim);
                recon::slope_limited(p,   dpv,  i0, i1, lim);
                recon::slope_limited(Br,  dBr,  i0, i1, lim);
                recon::slope_limited(Bz,  dBz,  i0, i1, lim);

                kill_edge_slopes(drho, i0, i1);
                kill_edge_slopes(dvr,  i0, i1);
                kill_edge_slopes(dvz,  i0, i1);
                kill_edge_slopes(dpv,  i0, i1);
                kill_edge_slopes(dBr,  i0, i1);
                kill_edge_slopes(dBz,  i0, i1);

                // Interfaces
                std::vector<double> rhoL,rhoR, vrL,vrR, vzL,vzR, pL,pR, BrL,BrR, BzL,BzR;
                recon::interfaces_lr(rho, drho, rhoL, rhoR, i0, i1);
                recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  i0, i1);
                recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  i0, i1);
                recon::interfaces_lr(p,   dpv,  pL,   pR,   i0, i1);
                recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  i0, i1);
                recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  i0, i1);

                // HLL fluxes at faces (Br,Bz fluxes nulled; CT owns B-updates)
                const std::size_t f0 = i0, f1 = i1 - 1;
                std::vector<rsolver::Flux> FH(g.size_r());
                for (std::size_t f=f0; f<f1; ++f){
                    rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
                    rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
                    rsolver::hll_r(gamma, WL, WR, FH[f]);
                    FH[f][4]=0.0; FH[f][5]=0.0; // zero magnetic fluxes here
                }

                // Conservative update (r-weighted)
                for (std::size_t i=i0+1; i+1<i1; ++i){
                    const std::size_t id = g.idx(i,std::size_t(kk));
                    double r   = (int(i)-int(g.Ng)+0.5)*g.dr;
                    double r_imh = std::max(r - 0.5*g.dr, 0.5*g.dr);
                    double r_iph = r + 0.5*g.dr;

                    rsolver::MHDPrim W{
                        std::max(1e-12,F.rho[id]), F.vr[id], F.vz[id],
                        std::max(1e-12,F.p[id]), F.Br[id], F.Bz[id]
                    };
                    rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);

                    for (int n=0;n<6;++n){
                        double rF_left  = r_imh * FH[i-1][n];
                        double rF_right = r_iph * FH[i][n];
                        double Ui = r * U[n];
                        Ui -= dt * (rF_right - rF_left) / g.dr;
                        U[n] = Ui / r;
                        if (!std::isfinite(U[n])) U[n] = 0.0;
                    }
                    rsolver::cons_to_prim(gamma, U, W);

                    // Floors and clamps
                    const double rho_floor=1e-6, p_floor=1e-6;
                    W.rho = std::max(W.rho, rho_floor);
                    W.p   = std::max(W.p,   p_floor);
                    W.vr  = std::clamp(W.vr,-UMAX,UMAX);
                    W.vz  = std::clamp(W.vz,-UMAX,UMAX);

                    F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
                }

                // Simple r-ghost copies (consistent with apply_bc_r after full sweep)
                F.rho[g.idx(i0,std::size_t(kk))]   = F.rho[g.idx(i0+1,std::size_t(kk))];
                F.rho[g.idx(i1-1,std::size_t(kk))] = F.rho[g.idx(i1-2,std::size_t(kk))];
                F.vr [g.idx(i0,std::size_t(kk))]   = F.vr [g.idx(i0+1,std::size_t(kk))];
                F.vr [g.idx(i1-1,std::size_t(kk))] = F.vr [g.idx(i1-2,std::size_t(kk))];
                F.vz [g.idx(i0,std::size_t(kk))]   = F.vz [g.idx(i0+1,std::size_t(kk))];
                F.vz [g.idx(i1-1,std::size_t(kk))] = F.vz [g.idx(i1-2,std::size_t(kk))];
                F.p  [g.idx(i0,std::size_t(kk))]   = F.p  [g.idx(i0+1,std::size_t(kk))];
                F.p  [g.idx(i1-1,std::size_t(kk))] = F.p  [g.idx(i1-2,std::size_t(kk))];
            }
        }
    } else {
        for (std::size_t k=k0; k<k1; ++k){
            std::vector<double> rho(g.size_r()), vr(g.size_r()), vz(g.size_r()),
                                p(g.size_r()),   Br(g.size_r()), Bz(g.size_r()),
                                Bth(g.size_r());
            for (std::size_t i=i0; i<i1; ++i){
                const std::size_t id = g.idx(i,k);
                rho[i] = std::max(1e-12, F.rho[id]);
                vr[i]  = F.vr[id];
                vz[i]  = F.vz[id];
                p[i]   = std::max(1e-12, F.p[id] + 0.5*F.Bth[id]*F.Bth[id]);
                Br[i]  = F.Br[id];
                Bz[i]  = F.Bz[id];
                Bth[i] = F.Bth[id];
            }

            std::vector<double> drho,dvr,dvz,dpv,dBr,dBz;
            recon::slope_limited(rho, drho, i0, i1, lim);
            recon::slope_limited(vr,  dvr,  i0, i1, lim);
            recon::slope_limited(vz,  dvz,  i0, i1, lim);
            recon::slope_limited(p,   dpv,  i0, i1, lim);
            recon::slope_limited(Br,  dBr,  i0, i1, lim);
            recon::slope_limited(Bz,  dBz,  i0, i1, lim);

            kill_edge_slopes(drho, i0, i1);
            kill_edge_slopes(dvr,  i0, i1);
            kill_edge_slopes(dvz,  i0, i1);
            kill_edge_slopes(dpv,  i0, i1);
            kill_edge_slopes(dBr,  i0, i1);
            kill_edge_slopes(dBz,  i0, i1);

            std::vector<double> rhoL,rhoR, vrL,vrR, vzL,vzR, pL,pR, BrL,BrR, BzL,BzR;
            recon::interfaces_lr(rho, drho, rhoL, rhoR, i0, i1);
            recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  i0, i1);
            recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  i0, i1);
            recon::interfaces_lr(p,   dpv,  pL,   pR,   i0, i1);
            recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  i0, i1);
            recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  i0, i1);

            const std::size_t f0 = i0, f1 = i1 - 1;
            std::vector<rsolver::Flux> FH(g.size_r());
            for (std::size_t f=f0; f<f1; ++f){
                rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
                rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
                rsolver::hll_r(gamma, WL, WR, FH[f]);
                FH[f][4]=0.0; FH[f][5]=0.0;
            }

            for (std::size_t i=i0+1; i+1<i1; ++i){
                const std::size_t id = g.idx(i,k);
                double r   = (int(i)-int(g.Ng)+0.5)*g.dr;
                double r_imh = std::max(r - 0.5*g.dr, 0.5*g.dr);
                double r_iph = r + 0.5*g.dr;

                rsolver::MHDPrim W{
                    std::max(1e-12,F.rho[id]), F.vr[id], F.vz[id],
                    std::max(1e-12,F.p[id]), F.Br[id], F.Bz[id]
                };
                rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);

                for (int n=0;n<6;++n){
                    double rF_left  = r_imh * FH[i-1][n];
                    double rF_right = r_iph * FH[i][n];
                    double Ui = r * U[n];
                    Ui -= dt * (rF_right - rF_left) / g.dr;
                    U[n] = Ui / r;
                    if (!std::isfinite(U[n])) U[n] = 0.0;
                }
                rsolver::cons_to_prim(gamma, U, W);

                const double rho_floor=1e-6, p_floor=1e-6;
                W.rho = std::max(W.rho, rho_floor);
                W.p   = std::max(W.p,   p_floor);
                W.vr  = std::clamp(W.vr,-UMAX,UMAX);
                W.vz  = std::clamp(W.vz,-UMAX,UMAX);

                F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
            }

            F.rho[g.idx(i0,k)]   = F.rho[g.idx(i0+1,k)];
            F.rho[g.idx(i1-1,k)] = F.rho[g.idx(i1-2,k)];
            F.vr [g.idx(i0,k)]   = F.vr [g.idx(i0+1,k)];
            F.vr [g.idx(i1-1,k)] = F.vr [g.idx(i1-2,k)];
            F.vz [g.idx(i0,k)]   = F.vz [g.idx(i0+1,k)];
            F.vz [g.idx(i1-1,k)] = F.vz [g.idx(i1-2,k)];
            F.p  [g.idx(i0,k)]   = F.p  [g.idx(i0+1,k)];
            F.p  [g.idx(i1-1,k)] = F.p  [g.idx(i1-2,k)];
        }
    }
}

/**
 * @brief Godunov sweep along z (standard Cartesian conservative update).
 *
 * @param F          In/out fields (ρ, vr, vz, p updated).
 * @param cfg        Unused.
 * @param gamma      Ratio of specific heats (γ).
 * @param lim        Slope limiter (Minmod/MC).
 * @param dt         Time step.
 * @param periodic_z If true, periodic BCs are applied in z at the end of each line update.
 *
 * @details
 * Similar to the r-sweep but without cylindrical r-weighting:
 * \f[
 *   U^{n+1} = U^n - \frac{\Delta t}{\Delta z} (F_{k+1/2} - F_{k-1/2}).
 * \f]
 * Magnetic fluxes for (Br,Bz) are again nulled here; CT handles their evolution.
 */
static void mhd_sweep_z(Fields& F, [[maybe_unused]] const RunConfig& cfg, double gamma,
                        recon::Limiter lim, double dt, bool periodic_z){
    (void)cfg; (void)periodic_z;
    const auto& g = F.g;
    const std::size_t i0 = g.Ng, i1 = g.Ng + g.Nr;
    const std::size_t k0 = g.Ng, k1 = g.Ng + g.Nz;
    constexpr double UMAX = 3e2;

    const bool do_omp = should_parallelize(g.Nr, g.Nz);

    if (do_omp) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static, OMP_CHUNK)
            for (int ii=int(i0); ii<int(i1); ++ii){
                std::vector<double> rho(g.size_z()), vr(g.size_z()), vz(g.size_z()),
                                    p(g.size_z()),   Br(g.size_z()), Bz(g.size_z());
                // Gather column
                for (std::size_t k=k0; k<k1; ++k){
                    const std::size_t id = g.idx(std::size_t(ii),k);
                    rho[k] = std::max(1e-12, F.rho[id]);
                    vr[k]  = F.vr[id];
                    vz[k]  = F.vz[id];
                    p[k]   = std::max(1e-12, F.p[id] + 0.5*F.Bth[id]*F.Bth[id]);
                    Br[k]  = F.Br[id];
                    Bz[k]  = F.Bz[id];
                }

                std::vector<double> drho,dvr,dvz,dpv,dBr,dBz;
                recon::slope_limited(rho, drho, k0, k1, lim);
                recon::slope_limited(vr,  dvr,  k0, k1, lim);
                recon::slope_limited(vz,  dvz,  k0, k1, lim);
                recon::slope_limited(p,   dpv,  k0, k1, lim);
                recon::slope_limited(Br,  dBr,  k0, k1, lim);
                recon::slope_limited(Bz,  dBz,  k0, k1, lim);

                kill_edge_slopes(drho, k0, k1);
                kill_edge_slopes(dvr,  k0, k1);
                kill_edge_slopes(dvz,  k0, k1);
                kill_edge_slopes(dpv,  k0, k1);
                kill_edge_slopes(dBr,  k0, k1);
                kill_edge_slopes(dBz,  k0, k1);

                std::vector<double> rhoL,rhoR, vrL,vrR, vzL,vzR, pL,pR, BrL,BrR, BzL,BzR;
                recon::interfaces_lr(rho, drho, rhoL, rhoR, k0, k1);
                recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  k0, k1);
                recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  k0, k1);
                recon::interfaces_lr(p,   dpv,  pL,   pR,   k0, k1);
                recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  k0, k1);
                recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  k0, k1);

                const std::size_t f0 = k0, f1 = k1 - 1;
                std::vector<rsolver::Flux> FH(g.size_z());
                for (std::size_t f=f0; f<f1; ++f){
                    rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
                    rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
                    rsolver::hll_z(gamma, WL, WR, FH[f]);
                    FH[f][4]=0.0; FH[f][5]=0.0;
                }

                for (std::size_t k=k0+1; k+1<k1; ++k){
                    const std::size_t id = g.idx(std::size_t(ii),k);
                    rsolver::MHDPrim W{
                        std::max(1e-12,F.rho[id]), F.vr[id], F.vz[id],
                        std::max(1e-12,F.p[id]), F.Br[id], F.Bz[id]
                    };
                    rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);
                    const auto& Fl = FH[k-1];
                    const auto& Fr = FH[k];
                    for (int n=0;n<6;++n){
                        U[n] -= dt/g.dz * (Fr[n] - Fl[n]);
                        if (!std::isfinite(U[n])) U[n] = 0.0;
                    }
                    rsolver::cons_to_prim(gamma, U, W);

                    const double rho_floor=1e-6, p_floor=1e-6;
                    W.rho = std::max(W.rho, rho_floor);
                    W.p   = std::max(W.p,   p_floor);
                    W.vr  = std::clamp(W.vr,-UMAX,UMAX);
                    W.vz  = std::clamp(W.vz,-UMAX,UMAX);

                    F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
                }
                apply_bc_z(F, periodic_z);
            }
        }
    } else {
        for (std::size_t i=i0; i<i1; ++i){
            std::vector<double> rho(g.size_z()), vr(g.size_z()), vz(g.size_z()),
                                p(g.size_z()),   Br(g.size_z()), Bz(g.size_z());
            for (std::size_t k=k0; k<k1; ++k){
                const std::size_t id = g.idx(i,k);
                rho[k] = std::max(1e-12, F.rho[id]);
                vr[k]  = F.vr[id];
                vz[k]  = F.vz[id];
                p[k]   = std::max(1e-12, F.p[id] + 0.5*F.Bth[id]*F.Bth[id]);
                Br[k]  = F.Br[id];
                Bz[k]  = F.Bz[id];
            }

            std::vector<double> drho,dvr,dvz,dpv,dBr,dBz;
            recon::slope_limited(rho, drho, k0, k1, lim);
            recon::slope_limited(vr,  dvr,  k0, k1, lim);
            recon::slope_limited(vz,  dvz,  k0, k1, lim);
            recon::slope_limited(p,   dpv,  k0, k1, lim);
            recon::slope_limited(Br,  dBr,  k0, k1, lim);
            recon::slope_limited(Bz,  dBz,  k0, k1, lim);

            kill_edge_slopes(drho, k0, k1);
            kill_edge_slopes(dvr,  k0, k1);
            kill_edge_slopes(dvz,  k0, k1);
            kill_edge_slopes(dpv,  k0, k1);
            kill_edge_slopes(dBr,  k0, k1);
            kill_edge_slopes(dBz,  k0, k1);

            std::vector<double> rhoL,rhoR, vrL,vrR, vzL,vzR, pL,pR, BrL,BrR, BzL,BzR;
            recon::interfaces_lr(rho, drho, rhoL, rhoR, k0, k1);
            recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  k0, k1);
            recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  k0, k1);
            recon::interfaces_lr(p,   dpv,  pL,   pR,   k0, k1);
            recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  k0, k1);
            recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  k0, k1);

            const std::size_t f0 = k0, f1 = k1 - 1;
            std::vector<rsolver::Flux> FH(g.size_z());
            for (std::size_t f=f0; f<f1; ++f){
                rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
                rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
                rsolver::hll_z(gamma, WL, WR, FH[f]);
                FH[f][4]=0.0; FH[f][5]=0.0;
            }

            for (std::size_t k=k0+1; k+1<k1; ++k){
                const std::size_t id = g.idx(i,k);
                rsolver::MHDPrim W{
                    std::max(1e-12,F.rho[id]), F.vr[id], F.vz[id],
                    std::max(1e-12,F.p[id]), F.Br[id], F.Bz[id]
                };
                rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);
                const auto& Fl = FH[k-1];
                const auto& Fr = FH[k];
                for (int n=0;n<6;++n){
                    U[n] -= dt/g.dz * (Fr[n] - Fl[n]);
                    if (!std::isfinite(U[n])) U[n] = 0.0;
                }
                rsolver::cons_to_prim(gamma, U, W);

                const double rho_floor=1e-6, p_floor=1e-6;
                W.rho = std::max(W.rho, rho_floor);
                W.p   = std::max(W.p,   p_floor);
                W.vr  = std::clamp(W.vr,-UMAX,UMAX);
                W.vz  = std::clamp(W.vz,-UMAX,UMAX);

                F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
            }
            apply_bc_z(F, periodic_z);
        }
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

/**
 * @brief Compute total magnetic energy stored in Bθ (integrated over volume).
 * @param F Fields.
 * @return \f$ \int \frac{1}{2} B_\theta^2 \, dV \f$ with \f$ dV = 2\pi r\, dr\, dz \f$.
 */
static double energy_Btheta(const Fields& F){
    const auto& g = F.g;
    double sum = 0.0;
    for (std::size_t i=g.Ng;i<g.Ng+g.Nr;++i){
        const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        for (std::size_t k=g.Ng;k<g.Ng+g.Nz;++k){
            const std::size_t id = g.idx(i,k);
            sum += 0.5 * F.Bth[id]*F.Bth[id] * (2.0*PI*r) * g.dr * g.dz;
        }
    }
    return sum;
}

/**
 * @brief Axial mode amplitude integral for a scalar q (pressure or density).
 *
 * @param F    Fields (reads p or rho).
 * @param k    Axial wavenumber.
 * @param from "pressure" or anything else (then density).
 * @return \f$ \int | \int q(r,z)\cos(kz)\,dz | \, 2\pi r\,dr \f$.
 * @note A simple proxy for mode growth along z for axisymmetric projections.
 */
static double mode_amplitude_k(const Fields& F, double k, const std::string& from){
    if (k<=0.0) return 0.0;
    const auto& g = F.g;
    double Ak = 0.0;
    for (std::size_t i=g.Ng;i<g.Ng+g.Nr;++i){
        const double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        double proj = 0.0;
        for (std::size_t kz=g.Ng;kz<g.Ng+g.Nz;++kz){
            const double z = (int(kz)-int(g.Ng)+0.5)*g.dz;
            const std::size_t id = g.idx(i,kz);
            const double q = (from=="pressure")? F.p[id] : F.rho[id];
            proj += q * std::cos(k*z);
        }
        proj *= g.dz;
        Ak += std::abs(proj) * (2.0*PI*r) * g.dr;
    }
    return Ak;
}

// ============================================================================
// Main driver
// ============================================================================

/**
 * @brief Run the 2D axisymmetric MHD toy simulation until t_end.
 *
 * @param F       In/out fields (initialized here, advanced in time, then dumped).
 * @param cfg     Run configuration (I/O cadence, output dir, background fields).
 * @param mhdcfg  Physics & numerics configuration (γ, CFL, limiters, viscosities, etc.).
 *
 * @details
 * Algorithmic skeleton per step:
 *  1) Compute max wave speed (|v| + c_f) over the grid; set dt by CFL and dt_max.
 *  2) Directional split:
 *     - Full dt:  sweep_r → sweep_z → CT → Bθ update → sources → KO → BC_r → sponge → BC_z
 *     - Half dt:  same sequence with 0.5·dt (Strang-like stabilization).
 *  3) Periodically write snapshots, 2.5D m=1 projections, diag row, and metrics.
 *
 * Basic safeguards:
 *  - NaN detection → write a debug frame and abort.
 *  - Floors on ρ and p; clamps on velocities.
 *  - KO edge filters and sponge layer to reduce spurious reflections.
 */
void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg){
    const auto& g = F.g;
    namespace fs = std::filesystem;
    fs::create_directories(cfg.out_dir + "/debug");

    // Fast I/O init once
    {
        static bool io_inited = false;
        if (!io_inited) {
            std::ios::sync_with_stdio(false);
            std::cin.tie(nullptr);
            std::cout.tie(nullptr);
            std::setvbuf(stdout, nullptr, _IOFBF, 1<<20); // 1 MiB stdout buffer
            io_inited = true;
        }
    }

    #ifdef _OPENMP
    {
        int th = 0;
        #pragma omp parallel
        {
            #pragma omp master
            { th = omp_get_num_threads(); }
        }
        std::cout << "[OMP] Using " << th << " thread(s)\n";
    }
    #else
        std::cout << "[OMP] Disabled (single-thread)\n";
    #endif

    // Initial state + optional modifications
    init_problem(F, cfg, mhdcfg);
    apply_vz_profile(F, cfg, mhdcfg);
    seed_modes(F, cfg, mhdcfg);

    // Initial dumps
    io::write_snapshot(F, cfg, /*step=*/0, /*t=*/0.0);
    double k_proj = (mhdcfg.modes.k > 0.0) ? mhdcfg.modes.k
                  : ((mhdcfg.k_diag > 0.0) ? mhdcfg.k_diag : 0.0);
    if (k_proj > 0.0) {
        io::write_snapshot_2p5D(F, cfg, /*step=*/0, /*t=*/0.0, k_proj);
    }

    recon::Limiter lim = (mhdcfg.limiter=="minmod") ? recon::Limiter::Minmod : recon::Limiter::MC;

    // Metrics file with persistent buffer
    std::ofstream metrics(cfg.out_dir + "/debug/2d_mhd_metrics.csv", std::ios::out);
    static std::vector<char> fbuf(1<<20);
    metrics.rdbuf()->pubsetbuf(fbuf.data(), static_cast<std::streamsize>(fbuf.size()));
    metrics << "t,divB_L2,Etot,E_Bth,vmax_raw,dt,Ak\n";

    const bool periodic_z = (mhdcfg.bc_z == "periodic");

    double t = 0.0;
    int    step = 0;

    // Ultra-light progress bar (throttled)
    auto t0 = std::chrono::steady_clock::now();
    const int  PROG_EVERY = 100;                            // refresh every N steps
    const auto UI_DT      = std::chrono::milliseconds(500); // and ≥0.5 s apart
    auto last_ui = t0;

    auto print_progress = [&](double frac, int step_, double t_sim, double dt_last, double vmax){
        frac = std::clamp(frac, 0.0, 1.0);
        constexpr int W = 40;
        int filled = (int)std::lround(frac * W);

        // ETA from elapsed time and fraction
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double eta = (frac > 1e-9) ? elapsed * (1.0 - frac) / frac : 0.0;

        static char bar[W+1];
        for (int i=0;i<W;i++) bar[i] = (i<filled ? '#' : '-');
        bar[W] = '\0';

        int s = (int)std::llround(eta);
        int h = s/3600; s%=3600; int m = s/60; s%=60;

        static char buf[256];
        int n = std::snprintf(buf, sizeof(buf),
            "\r[%s] %5.1f%%  t=%.2e  dt=%.2e  vmax=%.3g  ETA %02d:%02d:%02d  (step %d)",
            bar, 100.0*frac, t_sim, dt_last, vmax, h, m, s, step_);
        if (n < 0) return;

        std::fwrite(buf, 1, std::min<int>(n, (int)sizeof(buf)), stdout); // no flush
    };

    while (t < mhdcfg.t_end - 1e-16) {
        // --- CFL-based dt ---
        double vmax_raw = 1e-6;
        for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
            for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
                const std::size_t id  = g.idx(i,k);
                const double rho = std::max(1e-12, F.rho[id]);
                const double p   = std::max(1e-12, F.p[id]);
                const double cs  = std::sqrt(std::max(0.0, mhdcfg.gamma * p / rho));
                const double rho_safe = std::max(1e-10, rho);
                const double B2  = F.Br[id]*F.Br[id] + F.Bz[id]*F.Bz[id] + F.Bth[id]*F.Bth[id];
                const double vA  = std::sqrt(B2 / rho_safe);
                const double cf  = std::sqrt(cs*cs + vA*vA);
                vmax_raw = std::max(vmax_raw, std::abs(F.vr[id]) + cf);
                vmax_raw = std::max(vmax_raw, std::abs(F.vz[id]) + cf);
            }
        }
        if (!std::isfinite(vmax_raw) || vmax_raw <= 0.0){
            std::cerr << "[ABORT] vmax_raw invalid: " << vmax_raw << "\n"; break;
        }
        vmax_raw = cap_wave_speed(vmax_raw, mhdcfg.vmax_guard);

        double dt_cfl = mhdcfg.cfl * std::min(g.dr, g.dz) / vmax_raw;
        double dt = dt_cfl;
        if (mhdcfg.dt_max > 0.0) dt = std::min(dt, mhdcfg.dt_max);
        dt = std::clamp(dt, 1e-12, mhdcfg.t_end - t);

        // --- Directional splitting: full step ---
        mhd_sweep_r(F, cfg, mhdcfg.gamma, lim, dt);
        mhd_sweep_z(F, cfg, mhdcfg.gamma, lim, dt, periodic_z);
        ct_update(F, cfg, dt, mhdcfg.eta_ct, periodic_z);
        update_Btheta_axisym(F, dt, /*eta_theta=*/1.5*mhdcfg.eta_ct, periodic_z);
        apply_axisym_sources(F, /*dt=*/dt);
        ko_filter_edges_r(F, dt, /*fac=*/0.012);
        ko_filter_edges_z(F, dt, /*fac=*/0.012);
        apply_bc_r(F);
        apply_sponge(F, cfg, mhdcfg, dt);

        // --- Directional splitting: half step (stabilization) ---
        mhd_sweep_r(F, cfg, mhdcfg.gamma, lim, 0.5*dt);
        mhd_sweep_z(F, cfg, mhdcfg.gamma, lim, 0.5*dt, periodic_z);
        ct_update(F, cfg, 0.5*dt, mhdcfg.eta_ct, periodic_z);
        update_Btheta_axisym(F, 0.5*dt, /*eta_theta=*/1.5*mhdcfg.eta_ct, periodic_z);
        apply_axisym_sources(F, /*dt=*/0.5*dt);
        ko_filter_edges_r(F, 0.5*dt, /*fac=*/0.012);
        ko_filter_edges_z(F, 0.5*dt, /*fac=*/0.012);
        apply_bc_r(F);
        apply_sponge(F, cfg, mhdcfg, 0.5*dt);

        t += dt; step++;

        // --- Progress bar (throttled) ---
        if (step % PROG_EVERY == 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_ui >= UI_DT) {
                double frac = (mhdcfg.t_end > 0.0) ? (t / mhdcfg.t_end) : 0.0;
                print_progress(frac, step, t, dt, vmax_raw);
                std::fflush(stdout);
                last_ui = now;
            }
        }

        // --- Safety/diagnostics ---
        if (utils::count_nans(F) > 0) {
            std::cerr << "\n[ABORT] NaNs at step " << step << ", t=" << t << "\n";
            utils::DebugFrame dbg;
            dbg.t=t; dbg.dt=dt; dbg.cfl=mhdcfg.cfl; dbg.max_wave=vmax_raw;
            dbg.divB_L2_val=utils::divB_L2(F);
            dbg.energy_tot=utils::total_energy(F, cfg);
            dbg.nan_count=utils::count_nans(F);
            dbg.notes="abort on NaN";
            dbg.write_json(cfg.out_dir, step);
            break;
        }

        // --- Snapshots ---
        if (step % cfg.output_every == 0){
            io::write_snapshot(F, cfg, step, t);
            io::write_snapshot_2p5D(F, cfg, step, t, k_proj);
            io::write_diag(cfg.out_dir, step, t, vmax_raw);
        }

        // --- Metrics ---
        if (step % mhdcfg.diag_every == 0){
            const double EBth = energy_Btheta(F);
            const double Ak   = mhdcfg.write_mode_amp ? mode_amplitude_k(F, mhdcfg.k_diag, mhdcfg.amp_from) : 0.0;
            metrics << std::setprecision(16)
                    << t << "," << utils::divB_L2(F) << "," << utils::total_energy(F,cfg)
                    << "," << EBth << "," << vmax_raw << "," << dt << "," << Ak << "\n";
        }
    }

    // Final dump & close
    io::write_snapshot(F, cfg, /*step=*/step, /*t=*/t);
    if (k_proj > 0.0) {
        io::write_snapshot_2p5D(F, cfg, /*step=*/0, /*t=*/0.0, k_proj);
    }
    metrics << std::setprecision(16) << t << "," << utils::divB_L2(F) << ","
            << utils::total_energy(F,cfg) << "," << energy_Btheta(F)
            << "," << 0.0 << "," << 0.0 << "," << 0.0 << "\n";
    metrics.flush();

    std::fwrite("\n", 1, 1, stdout); // newline to finish the progress line
    std::fflush(stdout);
}

} // namespace physics

