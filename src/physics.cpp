// physics.cpp  — OpenMP-friendly (MSVC), sin 'omp simd', misma lógica numérica
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
  inline int omp_get_max_threads(){ return 1; }
  inline int omp_get_num_threads(){ return 1; }
  inline int omp_get_thread_num(){ return 0; }
#endif

namespace physics {

static constexpr double PI = 3.141592653589793238462643383279502884;

static inline bool should_parallelize(std::size_t Nr, std::size_t Nz) {
    const std::size_t Ncrit = 30000; // ~3e4 celdas
    return (Nr * Nz) >= Ncrit && omp_get_max_threads() > 1;
}
static constexpr int OMP_CHUNK = 16;

// -------------------- Inicialización del problema --------------------
static void init_problem(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhd){
    const auto& g = F.g;

    const double rho0  = 1.0;
    const double p_axis= 1.0e-2;
    const double R0    = 0.3 * g.Rmax;
    const double Bz_bg = cfg.phys.Bz0;
    const double Bth0  = 0.15;

    const std::size_t NrT = g.size_r();
    const std::size_t NzT = g.size_z();
    std::vector<double> rcoord(NrT, 0.0), Bth_r(NrT, 0.0), p_r(NrT, 0.0);

    for (std::size_t i=0; i<NrT; ++i){
        rcoord[i] = (int(i)-int(g.Ng)+0.5)*g.dr;
    }
    for (std::size_t i=0; i<NrT; ++i){
        const double r = std::fabs(rcoord[i]);
        const double x = r / (R0 + 1e-12);
        Bth_r[i] = Bth0 * (x / (1.0 + x*x));
    }

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
    {   // suavizado leve
        std::vector<double> tmp = p_r;
        for (std::size_t i=1; i+1<NrT; ++i){
            p_r[i] = 0.25*tmp[i-1] + 0.5*tmp[i] + 0.25*tmp[i+1];
        }
    }

    // 2D
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

    {   // diagnóstico balance radial — extendido
        double resid_L2 = 0.0, norm_L2 = 0.0, resid_Linf = 0.0;
        double dpdr_min = +1e300, dpdr_max = -1e300;

        // Pico de Btheta y su radio
        double Bth_abs_max = 0.0;
        std::size_t i_Bth_max = 0;
        for (std::size_t i = 0; i < NrT; ++i){
            double a = std::abs(Bth_r[i]);
            if (a > Bth_abs_max){ Bth_abs_max = a; i_Bth_max = i; }
        }
        const double r_Bth_max = (int(i_Bth_max) - int(g.Ng) + 0.5) * g.dr;

        // Acumuladores para normas y rangos
        for (std::size_t i = 1; i + 1 < NrT; ++i){
            const double r    = std::max(std::fabs(rcoord[i]), 0.5 * g.dr);
            const double dpdr = (p_r[i+1] - p_r[i-1]) / (2.0 * g.dr);
            const double dBdr = (Bth_r[i+1] - Bth_r[i-1]) / (2.0 * g.dr);
            const double R    = dpdr + Bth_r[i]*dBdr + (Bth_r[i]*Bth_r[i]) / r; // equilibrio radial

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

        // Parámetros en el centro (i_c ~ radio medio)
        const std::size_t ic = g.Ng + g.Nr/2;
        const double p0   = p_r[ic];
        const double rhoC = rho0; // consistente con arriba
        const double cs0  = std::sqrt(std::max(0.0, mhd.gamma * p0 / std::max(1e-12, rhoC)));
        const double B2_0 = cfg.phys.Bz0*cfg.phys.Bz0 + Bth_r[ic]*Bth_r[ic];
        const double vA0  = std::sqrt(B2_0 / std::max(1e-12, rhoC));
        const double cf0  = std::sqrt(cs0*cs0 + vA0*vA0);
        const double beta0= (B2_0 > 0.0) ? (2.0 * p0 / B2_0) : 0.0; // μ0=1 en unidades MHD típicas

        // Estimación de CFL inicial (v≈0 al inicio)
        const double hmin     = std::min(g.dr, g.dz);
        const double denom    = std::max(1e-12, cf0);
        const double dt_cfl0  = mhd.cfl * hmin / denom;

        // Info de malla
        std::cerr << std::setprecision(6) << std::scientific;
        std::cerr << "[INIT] grid: Nr=" << g.Nr << " Nz=" << g.Nz
                  << " dr=" << g.dr << " dz=" << g.dz
                  << " Rmax=" << g.Rmax << " Zmax=" << g.Zmax << "\n";

        // Equilibrio
        std::cerr << "[INIT] radial_balance_L2=" << resid_L2
                  << " (norm=" << norm_L2 << ", rel=" << rel_L2 << ")\n";
        std::cerr << "[INIT] radial_balance_Linf=" << resid_Linf << "\n";
        std::cerr << "[INIT] dp/dr range = [" << dpdr_min << ", " << dpdr_max << "]\n";

        // Campos y termodinámica
        std::cerr << "[INIT] center: p0=" << p0
                  << " Bz0=" << cfg.phys.Bz0
                  << " Bth0=" << Bth_r[ic]
                  << " beta0=" << beta0
                  << " cs0=" << cs0
                  << " vA0=" << vA0
                  << " cf0=" << cf0 << "\n";

        // Geometría de Bθ
        std::cerr << "[INIT] Bth_peak=" << Bth_abs_max
                  << " at r=" << r_Bth_max << " (R0=" << R0 << ")\n";

        // Sugerencia de paso inicial por CFL
        std::cerr << "[INIT] CFL(dt0) ~ " << dt_cfl0 << "  (v~0, usando cf0)\n";
    }
}

// --- NEW: perfiles v_z(r) inyectados tras init_problem ---
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

static inline double cap_wave_speed(double vmax_raw, double vmax_guard){
    if (vmax_guard <= 0.0) return vmax_raw;
    return std::min(vmax_raw, vmax_guard);
}

static inline void kill_edge_slopes(std::vector<double>& darr, std::size_t L, std::size_t R){
    if (R <= L) return;
    if (R - L >= 4) { darr[L+0] = 0.0; darr[R-1] = 0.0; }
    else { for (std::size_t q=L; q<R; ++q) darr[q] = 0.0; }
}

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

static void update_Btheta_axisym(Fields& F, double dt, double eta_theta, bool periodic_z){
    (void)periodic_z;
    const auto& g = F.g;
    std::vector<double> Bth_new(F.Bth);

    for (std::size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
        double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        double rinv = (r>0.0)? 1.0/r : 0.0;

        for (std::size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
            const std::size_t id = g.idx(i,k);

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

static void ct_update(Fields& F, [[maybe_unused]] const RunConfig& cfg, double dt, double eta_ct, bool periodic_z){
    (void)cfg;
    const auto& g = F.g;
    std::vector<double> E(g.size_r()*g.size_z(), 0.0);
    auto Eidx = [&](std::size_t i, std::size_t k){ return g.idx(i,k); };

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

    for (std::size_t k=0;k<g.size_z();++k){
        F.Br[g.idx(g.Ng,k)]                       = F.Br[g.idx(g.Ng+1,k)];
        F.Br[g.idx(g.size_r()-g.Ng-1,k)]         = F.Br[g.idx(g.size_r()-g.Ng-2,k)];
        F.Bz[g.idx(g.Ng,k)]                       = F.Bz[g.idx(g.Ng+1,k)];
        F.Bz[g.idx(g.size_r()-g.Ng-1,k)]         = F.Bz[g.idx(g.size_r()-g.Ng-2,k)];
    }
    apply_bc_z(F, periodic_z);
}

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
                for (std::size_t i=i0; i<i1; ++i){
                    const std::size_t id = g.idx(i,std::size_t(kk));
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

                    const double rho_floor=1e-6, p_floor=1e-6;
                    W.rho = std::max(W.rho, rho_floor);
                    W.p   = std::max(W.p,   p_floor);
                    W.vr  = std::clamp(W.vr,-UMAX,UMAX);
                    W.vz  = std::clamp(W.vz,-UMAX,UMAX);

                    F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
                }

                // ghosts en r
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

// -------------------- Diagnósticos --------------------
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

// -------------------- Driver principal --------------------
void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg){
    const auto& g = F.g;
    namespace fs = std::filesystem;
    fs::create_directories(cfg.out_dir + "/debug");

    // ---- Inicialización de I/O rápido (una sola vez) ----
    {
        static bool io_inited = false;
        if (!io_inited) {
            std::ios::sync_with_stdio(false);
            std::cin.tie(nullptr);
            std::cout.tie(nullptr);
            std::setvbuf(stdout, nullptr, _IOFBF, 1<<20); // 1 MiB buffer a stdout
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

    init_problem(F, cfg, mhdcfg);
    apply_vz_profile(F, cfg, mhdcfg);
    seed_modes(F, cfg, mhdcfg);

    io::write_snapshot(F, cfg, /*step=*/0, /*t=*/0.0);
    double k_proj = (mhdcfg.modes.k > 0.0) ? mhdcfg.modes.k
                  : ((mhdcfg.k_diag > 0.0) ? mhdcfg.k_diag : 0.0);
    if (k_proj > 0.0) {
        io::write_snapshot_2p5D(F, cfg, /*step=*/0, /*t=*/0.0, k_proj);
    }

    recon::Limiter lim = (mhdcfg.limiter=="minmod") ? recon::Limiter::Minmod : recon::Limiter::MC;

    // --- Archivo de métricas con búfer persistente ---
    std::ofstream metrics(cfg.out_dir + "/debug/2d_mhd_metrics.csv", std::ios::out);
    static std::vector<char> fbuf(1<<20);
    metrics.rdbuf()->pubsetbuf(fbuf.data(), static_cast<std::streamsize>(fbuf.size()));
    metrics << "t,divB_L2,Etot,E_Bth,vmax_raw,dt,Ak\n";

    const bool periodic_z = (mhdcfg.bc_z == "periodic");

    double t = 0.0;
    int    step = 0;

    // --- Barra de progreso ultra-ligera ---
    auto t0 = std::chrono::steady_clock::now();
    const int  PROG_EVERY = 100;                          // refresca cada N pasos
    const auto UI_DT      = std::chrono::milliseconds(500); // y al menos cada 0.5 s
    auto last_ui = t0;

    auto print_progress = [&](double frac, int step_, double t_sim, double dt_last, double vmax){
        frac = std::clamp(frac, 0.0, 1.0);
        constexpr int W = 40;
        int filled = (int)std::lround(frac * W);

        // ETA basado en tiempo transcurrido
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        double eta = (frac > 1e-9) ? elapsed * (1.0 - frac) / frac : 0.0;

        // Construcción de la barra y línea con snprintf
        static char bar[W+1];
        for (int i=0;i<W;i++) bar[i] = (i<filled ? '#' : '-');
        bar[W] = '\0';

        int s = (int)std::llround(eta);
        int h = s/3600; s%=3600; int m = s/60; s%=60;

        static char buf[256];
        int n = std::snprintf(buf, sizeof(buf),
            "\r[%s] %5.1f%%  t=%.2e  dt=%.2e  vmax=%.3g  ETA %02d:%02d:%02d  (step %d)",
            bar, 100.0*frac, t_sim, dt_last, vmax, h, m, s, step_);
        if (n < 0) return; // por seguridad

        std::fwrite(buf, 1, std::min<int>(n, (int)sizeof(buf)), stdout); // sin flush
    };

    while (t < mhdcfg.t_end - 1e-16) {
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

        // Paso completo + half-step con las mismas fases numéricas que ya tenías
        mhd_sweep_r(F, cfg, mhdcfg.gamma, lim, dt);
        mhd_sweep_z(F, cfg, mhdcfg.gamma, lim, dt, periodic_z);
        ct_update(F, cfg, dt, mhdcfg.eta_ct, periodic_z);
        update_Btheta_axisym(F, dt, /*eta_theta=*/1.5*mhdcfg.eta_ct, periodic_z);
        apply_axisym_sources(F, /*dt=*/dt);
        ko_filter_edges_r(F, dt, /*fac=*/0.012);
        ko_filter_edges_z(F, dt, /*fac=*/0.012);
        apply_bc_r(F);
        apply_sponge(F, cfg, mhdcfg, dt);

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

        // ---- Barra de progreso: throttle por tiempo + cada N pasos ----
        if (step % PROG_EVERY == 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_ui >= UI_DT) {
                double frac = (mhdcfg.t_end > 0.0) ? (t / mhdcfg.t_end) : 0.0;
                print_progress(frac, step, t, dt, vmax_raw);
                std::fflush(stdout);  // flush sólo cada UI_DT
                last_ui = now;
            }
        }

        // ---- Seguridad/diagnóstico ----
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

        // ---- Snapshots (los mantengo tal cual, según output_every) ----
        if (step % cfg.output_every == 0){
            io::write_snapshot(F, cfg, step, t);
            io::write_snapshot_2p5D(F, cfg, step, t, k_proj);
            io::write_diag(cfg.out_dir, step, t, vmax_raw);
        }

        // ---- Métricas (archivo con buffer, poco flush) ----
        if (step % mhdcfg.diag_every == 0){
            const double EBth = energy_Btheta(F);
            const double Ak   = mhdcfg.write_mode_amp ? mode_amplitude_k(F, mhdcfg.k_diag, mhdcfg.amp_from) : 0.0;
            metrics << std::setprecision(16)
                    << t << "," << utils::divB_L2(F) << "," << utils::total_energy(F,cfg)
                    << "," << EBth << "," << vmax_raw << "," << dt << "," << Ak << "\n";
            // sin metrics.flush(); dejemos que el buffer haga su trabajo
        }
    }

    // ---- Finalización: último snapshot y cierre prolijo ----
    io::write_snapshot(F, cfg, /*step=*/step, /*t=*/t);
    if (k_proj > 0.0) {
        io::write_snapshot_2p5D(F, cfg, /*step=*/0, /*t=*/0.0, k_proj);
    }
    metrics << std::setprecision(16) << t << "," << utils::divB_L2(F) << ","
            << utils::total_energy(F,cfg) << "," << energy_Btheta(F)
            << "," << 0.0 << "," << 0.0 << "," << 0.0 << "\n";
    metrics.flush(); // flush final del archivo

    std::fwrite("\n", 1, 1, stdout); // cerrar la línea de la barra
    std::fflush(stdout);
}

} // namespace physics

