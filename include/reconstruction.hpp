#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace recon {

/**
 * @brief Slope limiter options.
 *  - Minmod: more diffusive, very robust
 *  - MC    : Monotonized Central (less diffusive, still TVD)
 */
enum class Limiter { Minmod, MC };

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

inline double mc(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    const double mm = minmod(2.0 * a, 2.0 * b);
    return minmod(0.5 * (a + b), mm);
}

/**
 * @brief Compute limited slopes dq for q[i] on [i0, i1).
 * dq has the same size as q; only interior indices (i0+1 .. i1-2) are filled.
 */
inline void slope_limited(const std::vector<double>& q, std::vector<double>& dq,
                          std::size_t i0, std::size_t i1, Limiter lim) {
    dq.assign(q.size(), 0.0);
    for (std::size_t i = i0 + 1; i + 1 < i1; ++i) {
        const double dl = q[i]   - q[i - 1];
        const double dr = q[i+1] - q[i];
        dq[i] = (lim == Limiter::Minmod) ? minmod(dl, dr) : mc(dl, dr);
    }
}

/**
 * @brief Build left/right interface states (piecewise linear).
 * Produces:
 *   qR[i]   = value at right face of cell i
 *   qL[i+1] = value at left  face of cell i+1
 */
inline void interfaces_lr(const std::vector<double>& q, const std::vector<double>& dq,
                          std::vector<double>& qL, std::vector<double>& qR,
                          std::size_t i0, std::size_t i1) {
    qL.assign(q.size(), 0.0);
    qR.assign(q.size(), 0.0);
    for (std::size_t i = i0 + 1; i + 1 < i1; ++i) {
        qL[i + 1] = q[i] + 0.5 * dq[i];
        qR[i]     = q[i] - 0.5 * dq[i];
    }
}

} // namespace recon

