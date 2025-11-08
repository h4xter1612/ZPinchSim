#include "state.hpp"
#include <algorithm>

/**
 * @brief Allocate all field arrays to (Nr+2Ng)*(Nz+2Ng) and zero them.
 */
Fields::Fields(const Grid& grid) 
    : g(grid)
    , rho(g.size_r()*g.size_z())
    , vr (rho.size())
    , vz (rho.size())
    , vth(rho.size())
    , Br (rho.size())
    , Bz (rho.size())
    , Bth(rho.size())
    , p  (rho.size())
    , E  (rho.size())
    // --- modal m=1 (cos/sin) with same size ---
    , rho1c(rho.size())
    , rho1s(rho.size())
    , p1c  (rho.size())
    , p1s  (rho.size())
    , vr1c (rho.size())
    , vr1s (rho.size())
    , vz1c (rho.size())
    , vz1s (rho.size())
    , Bth1c(rho.size())
    , Bth1s(rho.size())
    , Bz1c (rho.size())
    , Bz1s (rho.size())
{
    zero();
}

/**
 * @brief Set all arrays (mode-0 and mode-1) to zero.
 */
void Fields::zero() {
    auto Z = [](auto& v){ std::fill(v.begin(), v.end(), 0.0); };

    // mode 0
    Z(rho); Z(vr); Z(vz); Z(vth); Z(Br); Z(Bz); Z(Bth); Z(p); Z(E);

    // mode 1
    Z(rho1c); Z(rho1s);
    Z(p1c);   Z(p1s);
    Z(vr1c);  Z(vr1s);
    Z(vz1c);  Z(vz1s);
    Z(Bth1c); Z(Bth1s);
    Z(Bz1c);  Z(Bz1s);
}

