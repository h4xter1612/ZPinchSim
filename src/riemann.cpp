
#include "riemann.hpp"
namespace riemann {
void flux_r(const Fields& /*F*/, Fields& dU, const RunConfig& /*cfg*/) {
    // TODO: compute radial fluxes; add to dU
    (void)dU;
}
void flux_z(const Fields& /*F*/, Fields& dU, const RunConfig& /*cfg*/) {
    // TODO: compute axial fluxes; add to dU
    (void)dU;
}
}
