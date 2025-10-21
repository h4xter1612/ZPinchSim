#pragma once
#include "state.hpp"
#include "physics.hpp"
#include <cmath>
#include <algorithm>

namespace modes {

/**
 * Seed a pseudo-3D (theta) perturbation in a 2D (r,z) grid.
 * - Perturba en un anillo radial r ~ r0_frac * Rmax con anchura sigma_frac*Rmax.
 * - Dependencia sinusoidal en z con wavenumber k (1/m).
 * - m atenúa/amplifica radialmente: weight ~ (r/r0)^m (crudo, sólo para “pseudo-3D”).
 *
 * eps es adimensional; se escala con Bz0 para Bθ y con c_s local para v_r.
 */
inline void seed(Fields& F, const RunConfig& cfg,
                 int m, double k, double eps,
                 double r0_frac, double sigma_frac,
                 bool seed_vr, bool seed_bth)
{
    const Grid& g   = F.g;
    const double R0 = r0_frac  * g.Rmax;
    const double SIG= std::max(1.0e-12, sigma_frac * g.Rmax);
    const double Bref = std::max(1.0e-12, static_cast<double>(cfg.phys.Bz0));
    const double gam  = std::max(1.01,    static_cast<double>(cfg.phys.gamma));

    // ---- c_s de referencia desde el CAMPO (celda central) ----
    const size_t ic = g.Ng + g.Nr/2;
    const size_t kc = g.Ng + g.Nz/2;
    const size_t idc= g.idx(ic, kc);
    const double p0   = std::max(1.0e-12, F.p[idc]);
    const double rho0 = std::max(1.0e-12, F.rho[idc]);
    const double cs_ref = std::sqrt(gam * p0 / rho0);

    for (size_t i=0; i<g.size_r(); ++i){
        // coordenadas centro de celda
        const double r = (static_cast<double>(static_cast<int>(i) - static_cast<int>(g.Ng)) + 0.5) * g.dr;

        for (size_t kz=0; kz<g.size_z(); ++kz){
            const double z  = (static_cast<double>(static_cast<int>(kz) - static_cast<int>(g.Ng)) + 0.5) * g.dz;
            const size_t id = g.idx(i, kz);

            // Envolvente radial tipo gaussiana centrada en R0
            const double gauss_r = std::exp(-0.5 * std::pow((r - R0)/SIG, 2.0));

            // Peso radial crudo que “depende de m” (evita r=0 y R0=0)
            const double r_safe  = std::max(1.0e-6, r);
            const double R0_safe = std::max(1.0e-6, R0);
            const double radial_weight = std::pow(r_safe / R0_safe, std::max(0, m));

            const double envelope = gauss_r * radial_weight;

            // Sinusoide en z
            const double s = std::cos(k * z);

            if (seed_vr){
                const double dv = eps * cs_ref * envelope * s;
                F.vr[id] += dv;
            }
            if (seed_bth){
                const double dB = eps * Bref * envelope * s;
                F.Bth[id] += dB;
            }
        }
    }
}

} // namespace modes

