
#include "state.hpp"
#include <algorithm>

Fields::Fields(const Grid& grid) : g(grid),
 rho(g.size_r()*g.size_z()), vr(rho.size()), vz(rho.size()), vth(rho.size()),
 Br(rho.size()), Bz(rho.size()), Bth(rho.size()), p(rho.size()), E(rho.size()) {
    zero();
}

void Fields::zero() {
    std::fill(rho.begin(), rho.end(), 0.0);
    std::fill(vr.begin(), vr.end(), 0.0);
    std::fill(vz.begin(), vz.end(), 0.0);
    std::fill(vth.begin(), vth.end(), 0.0);
    std::fill(Br.begin(), Br.end(), 0.0);
    std::fill(Bz.begin(), Bz.end(), 0.0);
    std::fill(Bth.begin(), Bth.end(), 0.0);
    std::fill(p.begin(), p.end(), 0.0);
    std::fill(E.begin(), E.end(), 0.0);
}
