
#include "time_integrator.hpp"
#include "riemann.hpp"
#include "physics.hpp"
#include "boundary.hpp"

namespace timeint {
double step(Fields& F, const RunConfig& cfg, double t, double dt){
    (void)t;
    Fields k1 = F;
    Fields dU(F.g); dU.zero();

    // fluxes
    riemann::flux_r(F, dU, cfg);
    riemann::flux_z(F, dU, cfg);

    // apply sources/BC
    physics::apply_sources(F, cfg, dt);
    bc::apply(F, cfg);

    // toy forward Euler update
    for (size_t i=0;i<F.rho.size();++i){
        F.rho[i] += dt * dU.rho[i];
        F.vr[i]  += dt * dU.vr[i];
        F.vz[i]  += dt * dU.vz[i];
        F.vth[i] += dt * dU.vth[i];
        F.Br[i]  += dt * dU.Br[i];
        F.Bz[i]  += dt * dU.Bz[i];
        F.Bth[i] += dt * dU.Bth[i];
        F.p[i]   += dt * dU.p[i];
        F.E[i]   += dt * dU.E[i];
    }
    return dt;
}
}
