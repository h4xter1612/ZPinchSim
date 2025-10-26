#include "physics.hpp"
#include "riemann.hpp"
#include "reconstruction.hpp"
#include "io.hpp"
#include "utils.hpp"
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <iostream>

namespace physics {

static void init_problem(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhd){
    const auto& g = F.g;
    for (size_t i=0;i<g.size_r();++i){
        double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        for (size_t k=0;k<g.size_z();++k){
            double z = (int(k)-int(g.Ng)+0.5)*g.dz;
            size_t id = g.idx(i,k);
            if (mhd.problem=="blast"){
                double rc = 0.2*std::min(g.Rmax, g.Zmax);
                double rr = std::sqrt(r*r + (z-0.5*g.Zmax)*(z-0.5*g.Zmax));
                F.rho[id] = 1.0;
                F.p[id]   = (rr<rc)? 1e-1 : 1e-2;
            } else { // brio_wu o default
                F.rho[id] = (z < 0.5*g.Zmax)? 1.0 : 0.125;
                F.p[id]   = (z < 0.5*g.Zmax)? 1.0 : 0.1;
            }
            F.vr[id]=F.vz[id]=F.vth[id]=0.0;
            F.Br[id]=0.0;
            F.Bz[id]=cfg.phys.Bz0;
        }
    }
}

static void ct_update(Fields& F, const RunConfig& cfg, double dt, double eta_ct){
    const auto& g = F.g;
    std::vector<double> E(g.size_r()*g.size_z(), 0.0);
    auto Eidx = [&](size_t i, size_t k){ return g.idx(i,k); };
    for (size_t i=1; i<g.size_r()-1; ++i){
        for (size_t k=1; k<g.size_z()-1; ++k){
            double vr = 0.25*(F.vr[g.idx(i,k)]+F.vr[g.idx(i-1,k)]+F.vr[g.idx(i,k-1)]+F.vr[g.idx(i-1,k-1)]);
            double vz = 0.25*(F.vz[g.idx(i,k)]+F.vz[g.idx(i-1,k)]+F.vz[g.idx(i,k-1)]+F.vz[g.idx(i-1,k-1)]);
            double Brc= 0.25*(F.Br[g.idx(i,k)]+F.Br[g.idx(i-1,k)]+F.Br[g.idx(i,k-1)]+F.Br[g.idx(i-1,k-1)]);
            double Bzc= 0.25*(F.Bz[g.idx(i,k)]+F.Bz[g.idx(i-1,k)]+F.Bz[g.idx(i,k-1)]+F.Bz[g.idx(i-1,k-1)]);
            double Etheta = -(vr*Bzc - vz*Brc);
            if (eta_ct>0.0){
                double dBr_dz = (F.Br[g.idx(i,k)] - F.Br[g.idx(i,k-1)]) / g.dz;
                double dBz_dr = (F.Bz[g.idx(i,k)] - F.Bz[g.idx(i-1,k)]) / g.dr;
                Etheta += eta_ct * (dBr_dz - dBz_dr); // signo resistivo estable
            }
            E[Eidx(i,k)] = Etheta;
        }
    }
    std::vector<double> Br_new(F.Br), Bz_new(F.Bz);
    for (size_t i=1; i<g.size_r()-1; ++i){
        for (size_t k=1; k<g.size_z()-1; ++k){
            double dE_dz = (E[Eidx(i,k+1)] - E[Eidx(i,k)]) / g.dz;
            double dE_dr = (E[Eidx(i+1,k)] - E[Eidx(i,k)]) / g.dr;
            Br_new[g.idx(i,k)] = F.Br[g.idx(i,k)] - dt * dE_dz;
            Bz_new[g.idx(i,k)] = F.Bz[g.idx(i,k)] + dt * dE_dr;
        }
    }
    F.Br.swap(Br_new); F.Bz.swap(Bz_new);
    // BCs de espejo
    for (size_t k=0;k<g.size_z();++k){
        F.Br[g.idx(g.Ng,k)] = F.Br[g.idx(g.Ng+1,k)];
        F.Br[g.idx(g.size_r()-g.Ng-1,k)] = F.Br[g.idx(g.size_r()-g.Ng-2,k)];
        F.Bz[g.idx(g.Ng,k)] = F.Bz[g.idx(g.Ng+1,k)];
        F.Bz[g.idx(g.size_r()-g.Ng-1,k)] = F.Bz[g.idx(g.size_r()-g.Ng-2,k)];
    }
    for (size_t i=0;i<g.size_r();++i){
        F.Br[g.idx(i,g.Ng)] = F.Br[g.idx(i,g.Ng+1)];
        F.Br[g.idx(i,g.size_z()-g.Ng-1)] = F.Br[g.idx(i,g.size_z()-g.Ng-2)];
        F.Bz[g.idx(i,g.Ng)] = F.Bz[g.idx(i,g.Ng+1)];
        F.Bz[g.idx(i,g.size_z()-g.Ng-1)] = F.Bz[g.idx(i,g.size_z()-g.Ng-2)];
    }
}

// ===== Barrido MHD en r (MUSCL + HLL); B no se actualiza aquí =====
static void mhd_sweep_r(Fields& F, const RunConfig& cfg, double gamma,
                        recon::Limiter lim, double dt){
    const auto& g = F.g;
    const size_t i0 = g.Ng, i1 = g.Ng + g.Nr;
    const size_t k0 = g.Ng, k1 = g.Ng + g.Nz;
    constexpr double UMAX = 3e2;

    for (size_t k=k0; k<k1; ++k){
        std::vector<double> rho(g.size_r()), vr(g.size_r()), vz(g.size_r()),
                            p(g.size_r()),   Br(g.size_r()), Bz(g.size_r());
        for (size_t i=i0; i<i1; ++i){
            const size_t id = g.idx(i,k);
            rho[i] = std::max(1e-12, F.rho[id]);
            vr[i]  = F.vr[id];
            vz[i]  = F.vz[id];
            p[i]   = std::max(1e-12, F.p[id]);
            Br[i]  = F.Br[id];
            Bz[i]  = F.Bz[id];
        }
        std::vector<double> drho, dvr, dvz, dpv, dBr, dBz;
        recon::slope_limited(rho, drho, i0, i1, lim);
        recon::slope_limited(vr,  dvr,  i0, i1, lim);
        recon::slope_limited(vz,  dvz,  i0, i1, lim);
        recon::slope_limited(p,   dpv,  i0, i1, lim);
        recon::slope_limited(Br,  dBr,  i0, i1, lim);
        recon::slope_limited(Bz,  dBz,  i0, i1, lim);

        std::vector<double> rhoL, rhoR, vrL, vrR, vzL, vzR, pL, pR, BrL, BrR, BzL, BzR;
        recon::interfaces_lr(rho, drho, rhoL, rhoR, i0, i1);
        recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  i0, i1);
        recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  i0, i1);
        recon::interfaces_lr(p,   dpv,  pL,   pR,   i0, i1);
        recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  i0, i1);
        recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  i0, i1);

        const size_t f0 = i0, f1 = i1 - 1;
        std::vector<rsolver::Flux> FH(g.size_r());
        for (size_t f=f0; f<f1; ++f){
            rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
            rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
            rsolver::hll_r(gamma, WL, WR, FH[f]);
            FH[f][4]=0.0; FH[f][5]=0.0; // B solo por CT
        }

        for (size_t i=i0+1; i+1<i1; ++i){
            const size_t id = g.idx(i,k);
            rsolver::MHDPrim W{rho[i], vr[i], vz[i], p[i], Br[i], Bz[i]};
            rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);
            const auto& Fl = FH[i-1];
            const auto& Fr = FH[i];
            for (int n=0;n<6;++n){
                U[n] -= dt/g.dr * (Fr[n] - Fl[n]);
                if (!std::isfinite(U[n])) U[n] = 0.0;
            }
            rsolver::cons_to_prim(gamma, U, W);
            // W.vr = std::clamp(W.vr, -UMAX, UMAX);
            // W.vz = std::clamp(W.vz, -UMAX, UMAX);
            const double rho_floor = 1e-8;
            const double p_floor   = 1e-8;
            W.rho = std::max(W.rho, rho_floor);
            W.p   = std::max(W.p,   p_floor);
            W.vr  = std::clamp(W.vr, -UMAX, UMAX);
            W.vz  = std::clamp(W.vz, -UMAX, UMAX);

            F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
        }
        // BCs
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

// ===== Barrido MHD en z (MUSCL + HLL); B no se actualiza aquí =====
static void mhd_sweep_z(Fields& F, const RunConfig& cfg, double gamma,
                        recon::Limiter lim, double dt){
    const auto& g = F.g;
    const size_t i0 = g.Ng, i1 = g.Ng + g.Nr;
    const size_t k0 = g.Ng, k1 = g.Ng + g.Nz;
    constexpr double UMAX = 3e2;

    for (size_t i=i0; i<i1; ++i){
        std::vector<double> rho(g.size_z()), vr(g.size_z()), vz(g.size_z()),
                            p(g.size_z()),   Br(g.size_z()), Bz(g.size_z());
        for (size_t k=k0; k<k1; ++k){
            const size_t id = g.idx(i,k);
            rho[k] = std::max(1e-12, F.rho[id]);
            vr[k]  = F.vr[id];
            vz[k]  = F.vz[id];
            p[k]   = std::max(1e-12, F.p[id]);
            Br[k]  = F.Br[id];
            Bz[k]  = F.Bz[id];
        }
        std::vector<double> drho, dvr, dvz, dpv, dBr, dBz;
        recon::slope_limited(rho, drho, k0, k1, lim);
        recon::slope_limited(vr,  dvr,  k0, k1, lim);
        recon::slope_limited(vz,  dvz,  k0, k1, lim);
        recon::slope_limited(p,   dpv,  k0, k1, lim);
        recon::slope_limited(Br,  dBr,  k0, k1, lim);
        recon::slope_limited(Bz,  dBz,  k0, k1, lim);

        std::vector<double> rhoL, rhoR, vrL, vrR, vzL, vzR, pL, pR, BrL, BrR, BzL, BzR;
        recon::interfaces_lr(rho, drho, rhoL, rhoR, k0, k1);
        recon::interfaces_lr(vr,  dvr,  vrL,  vrR,  k0, k1);
        recon::interfaces_lr(vz,  dvz,  vzL,  vzR,  k0, k1);
        recon::interfaces_lr(p,   dpv,  pL,   pR,   k0, k1);
        recon::interfaces_lr(Br,  dBr,  BrL,  BrR,  k0, k1);
        recon::interfaces_lr(Bz,  dBz,  BzL,  BzR,  k0, k1);

        const size_t f0 = k0, f1 = k1 - 1;
        std::vector<rsolver::Flux> FH(g.size_z());
        for (size_t f=f0; f<f1; ++f){
            rsolver::MHDPrim WL{rhoR[f], vrR[f], vzR[f], pR[f], BrR[f], BzR[f]};
            rsolver::MHDPrim WR{rhoL[f+1], vrL[f+1], vzL[f+1], pL[f+1], BrL[f+1], BzL[f+1]};
            rsolver::hll_z(gamma, WL, WR, FH[f]);
            FH[f][4]=0.0; FH[f][5]=0.0; // B solo por CT
        }

        for (size_t k=k0+1; k+1<k1; ++k){
            const size_t id = g.idx(i,k);
            rsolver::MHDPrim W{rho[k], vr[k], vz[k], p[k], Br[k], Bz[k]};
            rsolver::Cons U; rsolver::prim_to_cons(gamma, W, U);
            const auto& Fl = FH[k-1];
            const auto& Fr = FH[k];
            for (int n=0;n<6;++n){
                U[n] -= dt/g.dz * (Fr[n] - Fl[n]);
                if (!std::isfinite(U[n])) U[n] = 0.0;
            }
            rsolver::cons_to_prim(gamma, U, W);
            // W.vr = std::clamp(W.vr, -UMAX, UMAX);
            // W.vz = std::clamp(W.vz, -UMAX, UMAX);
            const double rho_floor = 1e-8;
            const double p_floor   = 1e-8;
            W.rho = std::max(W.rho, rho_floor);
            W.p   = std::max(W.p,   p_floor);
            W.vr  = std::clamp(W.vr, -UMAX, UMAX);
            W.vz  = std::clamp(W.vz, -UMAX, UMAX);
            F.rho[id]=W.rho; F.vr[id]=W.vr; F.vz[id]=W.vz; F.p[id]=W.p;
        }
        // BCs
        F.rho[g.idx(i,k0)]   = F.rho[g.idx(i,k0+1)];
        F.rho[g.idx(i,k1-1)] = F.rho[g.idx(i,k1-2)];
        F.vr [g.idx(i,k0)]   = F.vr [g.idx(i,k0+1)];
        F.vr [g.idx(i,k1-1)] = F.vr [g.idx(i,k1-2)];
        F.vz [g.idx(i,k0)]   = F.vz [g.idx(i,k0+1)];
        F.vz [g.idx(i,k1-1)] = F.vz [g.idx(i,k1-2)];
        F.p  [g.idx(i,k0)]   = F.p  [g.idx(i,k0+1)];
        F.p  [g.idx(i,k1-1)] = F.p  [g.idx(i,k1-2)];
    }
}

void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg){
    const auto& g = F.g;
    namespace fs=std::filesystem; fs::create_directories(cfg.out_dir + "/debug");

    init_problem(F, cfg, mhdcfg);

    // snapshot inicial
    io::write_snapshot(F, cfg, /*step=*/0, /*t=*/0.0);

    recon::Limiter lim = (mhdcfg.limiter=="minmod") ? recon::Limiter::Minmod : recon::Limiter::MC;

    { std::ofstream(cfg.out_dir + "/debug/2d_mhd_metrics.csv") << "t,divB_L2,Etot,vmax_raw,vmax_cap,dt\n"; }

    double t=0.0; int step=0;
    while (t < mhdcfg.t_end - 1e-16){
        // velocidad característica (rápida magnética)
        double vmax_raw = 1e-6; // reiniciar cada paso
        for (size_t i=g.Ng; i<g.Ng+g.Nr; ++i){
            for (size_t k=g.Ng; k<g.Ng+g.Nz; ++k){
                const size_t id = g.idx(i,k);
                const double rho = std::max(1e-12, F.rho[id]);
                const double p   = std::max(1e-12, F.p[id]);
                const double cs  = std::sqrt(std::max(0.0, mhdcfg.gamma * p / rho));
                // const double vA  = std::sqrt( (F.Br[id]*F.Br[id] + F.Bz[id]*F.Bz[id]) / rho );
                const double rho_safe = std::max(1e-10, rho); // ya tenías 1e-12; subir a 1e-10 aquí ayuda
                const double B2 = F.Br[id]*F.Br[id] + F.Bz[id]*F.Bz[id];
                // const double vA  = std::sqrt(B2 / rho_safe);

                const double vA_cap = 5e3; // por ejemplo
                const double vA = std::min(std::sqrt(B2 / rho_safe), vA_cap);
                const double cf  = std::sqrt(cs*cs + vA*vA);
                vmax_raw = std::max(vmax_raw, std::abs(F.vr[id]) + cf);
                vmax_raw = std::max(vmax_raw, std::abs(F.vz[id]) + cf);
            }
        }
        double vmax = vmax_raw;
        if (mhdcfg.vmax_guard > 0.0) vmax = std::min(vmax, mhdcfg.vmax_guard);

        if (!std::isfinite(vmax) || vmax <= 0){
            std::cerr << "[ABORT] vmax invalid: " << vmax << "\n";
            break;
        }

        if (!std::isfinite(vmax_raw)) {
            std::cerr << "[ABORT] vmax_raw non-finite at step " << step << ", t=" << t << "\n";
            break;
        }

        double dt_cfl = mhdcfg.cfl * std::min(g.dr, g.dz) / vmax;
        double dt = dt_cfl;
        if (mhdcfg.dt_max > 0.0) dt = std::min(dt, mhdcfg.dt_max);
        dt = std::clamp(dt, 1e-10, mhdcfg.t_end - t);

        // Paso completo
        mhd_sweep_r(F, cfg, mhdcfg.gamma, lim, dt);
        mhd_sweep_z(F, cfg, mhdcfg.gamma, lim, dt);
        ct_update(F, cfg, dt, mhdcfg.eta_ct);

        // Medio paso RK
        mhd_sweep_r(F, cfg, mhdcfg.gamma, lim, 0.5*dt);
        mhd_sweep_z(F, cfg, mhdcfg.gamma, lim, 0.5*dt);
        ct_update(F, cfg, 0.5*dt, mhdcfg.eta_ct);

        t += dt; step++;

        if (step % 100 == 0) {
            std::cout << "[2D_MHD_TOY] t=" << t << " step=" << step 
                      << " dt=" << dt << " vmax_raw=" << vmax_raw << " vmax=" << vmax << "\n";
        }

        if (utils::count_nans(F) > 0) {
            std::cerr << "[ABORT] NaNs detected at step " << step << ", t=" << t << "\n";
            utils::DebugFrame dbg;
            dbg.t=t; dbg.dt=dt; dbg.cfl=mhdcfg.cfl; dbg.max_wave=vmax;
            dbg.divB_L2_val=utils::divB_L2(F); 
            dbg.energy_tot=utils::total_energy(F, cfg);
            dbg.nan_count=utils::count_nans(F);
            dbg.notes="abort on NaN";
            dbg.write_json(cfg.out_dir, step);
            break;
        }

        if (step % cfg.output_every == 0){
            io::write_snapshot(F, cfg, step, t);
        }
        // diag.csv con frecuencia configurable
        if (mhdcfg.diag_every < 1) {
            io::write_diag(cfg.out_dir, step, t, vmax);
        } else if (step % mhdcfg.diag_every == 0) {
            io::write_diag(cfg.out_dir, step, t, vmax);
        }

        // métricas
        if (step % cfg.output_every == 0){
            double divB = utils::divB_L2(F);
            double Etot=0.0;
            for (size_t i=0;i<g.size_r();++i){
                for (size_t k=0;k<g.size_z();++k){
                    size_t id=g.idx(i,k);
                    double ke = 0.5 * F.rho[id]*(F.vr[id]*F.vr[id]+F.vz[id]*F.vz[id]);
                    double ge = F.p[id]/(mhdcfg.gamma-1.0);
                    double me = 0.5*(F.Br[id]*F.Br[id]+F.Bz[id]*F.Bz[id]+F.Bth[id]*F.Bth[id]);
                    Etot += ke + ge + me;
                }
            }
            std::ofstream(cfg.out_dir + "/debug/2d_mhd_metrics.csv", std::ios::app)
                << std::setprecision(16) << t << "," << divB << "," << Etot << ","
                << vmax_raw << "," << vmax << "," << dt << "\n";
        }
    }

    // snapshot final garantizado + métrica final
    io::write_snapshot(F, cfg, /*step=*/step, /*t=*/t);
    double divB = utils::divB_L2(F);
    double Etot = 0.0;
    for (size_t i=0;i<g.size_r();++i){
        for (size_t k=0;k<g.size_z();++k){
            size_t id=g.idx(i,k);
            double ke = 0.5 * F.rho[id]*(F.vr[id]*F.vr[id]+F.vz[id]*F.vz[id]);
            double ge = F.p[id]/(mhdcfg.gamma-1.0);
            double me = 0.5*(F.Br[id]*F.Br[id]+F.Bz[id]*F.Bz[id]+F.Bth[id]*F.Bth[id]);
            Etot += ke + ge + me;
        }
    }
    std::ofstream(cfg.out_dir + "/debug/2d_mhd_metrics.csv", std::ios::app)
        << std::setprecision(16) << t << "," << divB << "," << Etot << ","
        << 0.0 << "," << 0.0 << "," << 0.0 << "\n";

}

} // namespace physics
