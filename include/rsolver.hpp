#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace rsolver {

/**
 * @brief Primitive state for the poloidal MHD subsystem (axisymmetric scaffold).
 * Contains (ρ, v_r, v_z, p, B_r, B_z).
 */
struct MHDPrim { double rho, vr, vz, p, Br, Bz; };

/// Conserved and flux array layout: [rho, m_r, m_z, E, B_r, B_z]
using Cons = std::array<double, 6>;
using Flux = std::array<double, 6>;

inline double pt(const MHDPrim& W) {
    return W.p + 0.5 * (W.Br * W.Br + W.Bz * W.Bz);
}

inline void prim_to_cons(double gamma, const MHDPrim& W, Cons& U) {
    const double v2 = W.vr * W.vr + W.vz * W.vz;
    const double Em = 0.5 * (W.Br * W.Br + W.Bz * W.Bz);
    const double Ei = W.p / (gamma - 1.0);
    U[0] = W.rho;
    U[1] = W.rho * W.vr;
    U[2] = W.rho * W.vz;
    U[3] = Ei + 0.5 * W.rho * v2 + Em;
    U[4] = W.Br;
    U[5] = W.Bz;
}

inline void cons_to_prim(double gamma, const Cons& U, MHDPrim& W) {
    W.rho = std::max(1e-12, U[0]);
    W.vr  = U[1] / W.rho;
    W.vz  = U[2] / W.rho;
    W.Br  = U[4];
    W.Bz  = U[5];
    const double v2 = W.vr * W.vr + W.vz * W.vz;
    const double Em = 0.5 * (W.Br * W.Br + W.Bz * W.Bz);
    const double Ei = U[3] - 0.5 * W.rho * v2 - Em;
    W.p = std::max(1e-12, (gamma - 1.0) * Ei);
}

/**
 * @brief Euler–MHD flux in r-direction for the reduced (Br,Bz) system.
 * Note: Bθ terms are handled outside this Riemann solver (CT, source terms).
 */
inline void flux_r(double gamma, const MHDPrim& W, Flux& F) {
    (void)gamma;
    const double ptot = pt(W);
    const double vn   = W.vr;
    const double vb   = W.vr * W.Br + W.vz * W.Bz;
    const double Etot = W.p / (gamma - 1.0)
                      + 0.5 * W.rho * (W.vr * W.vr + W.vz * W.vz)
                      + 0.5 * (W.Br * W.Br + W.Bz * W.Bz);
    F[0] = W.rho * vn;
    F[1] = W.rho * W.vr * W.vr + ptot - W.Br * W.Br;
    F[2] = W.rho * W.vr * W.vz - W.Br * W.Bz;
    F[3] = (Etot + ptot) * vn - W.Br * vb;
    F[4] = 0.0; // Br/Bz updated by CT
    F[5] = 0.0;
}

/**
 * @brief Euler–MHD flux in z-direction for the reduced (Br,Bz) system.
 */
inline void flux_z(double gamma, const MHDPrim& W, Flux& F) {
    (void)gamma;
    const double ptot = pt(W);
    const double vn   = W.vz;
    const double vb   = W.vr * W.Br + W.vz * W.Bz;
    const double Etot = W.p / (gamma - 1.0)
                      + 0.5 * W.rho * (W.vr * W.vr + W.vz * W.vz)
                      + 0.5 * (W.Br * W.Br + W.Bz * W.Bz);
    F[0] = W.rho * vn;
    F[1] = W.rho * W.vr * W.vz - W.Br * W.Bz;
    F[2] = W.rho * W.vz * W.vz + ptot - W.Bz * W.Bz;
    F[3] = (Etot + ptot) * vn - W.Bz * vb;
    F[4] = 0.0; // Br/Bz updated by CT
    F[5] = 0.0;
}

/** @brief Sound speed squared. */
inline double cs2(double gamma, const MHDPrim& W) {
    return gamma * W.p / std::max(1e-12, W.rho);
}

/** @brief Fast magnetosonic speed approximation: cf ≈ sqrt(cs^2 + vA^2). */
inline double a_fast(double gamma, const MHDPrim& W, bool /*normal_r*/) {
    const double cs  = std::sqrt(std::max(0.0, cs2(gamma, W)));
    const double vA2 = (W.Br * W.Br + W.Bz * W.Bz) / std::max(1e-12, W.rho);
    return std::sqrt(cs * cs + vA2);
}

/**
 * @brief HLL Riemann solver in r-direction.
 */
inline void hll_r(double gamma, const MHDPrim& WL, const MHDPrim& WR, Flux& FH) {
    Cons UL, UR; prim_to_cons(gamma, WL, UL); prim_to_cons(gamma, WR, UR);
    Flux FL, FR; flux_r(gamma, WL, FL);       flux_r(gamma, WR, FR);
    const double aL = a_fast(gamma, WL, true),  aR = a_fast(gamma, WR, true);
    const double sL = std::min(WL.vr - aL, WR.vr - aR);
    const double sR = std::max(WL.vr + aL, WR.vr + aR);
    if (sL >= 0) { FH = FL; return; }
    if (sR <= 0) { FH = FR; return; }
    for (int n = 0; n < 6; ++n) {
        FH[n] = (sR * FL[n] - sL * FR[n] + sR * sL * (UR[n] - UL[n])) / (sR - sL);
        if (!std::isfinite(FH[n])) FH[n] = 0.0;
    }
}

/**
 * @brief HLL Riemann solver in z-direction.
 */
inline void hll_z(double gamma, const MHDPrim& WL, const MHDPrim& WR, Flux& FH) {
    Cons UL, UR; prim_to_cons(gamma, WL, UL); prim_to_cons(gamma, WR, UR);
    Flux FL, FR; flux_z(gamma, WL, FL);       flux_z(gamma, WR, FR);
    const double aL = a_fast(gamma, WL, false), aR = a_fast(gamma, WR, false);
    const double sL = std::min(WL.vz - aL, WR.vz - aR);
    const double sR = std::max(WL.vz + aL, WR.vz + aR);
    if (sL >= 0) { FH = FL; return; }
    if (sR <= 0) { FH = FR; return; }
    for (int n = 0; n < 6; ++n) {
        FH[n] = (sR * FL[n] - sL * FR[n] + sR * sL * (UR[n] - UL[n])) / (sR - sL);
        if (!std::isfinite(FH[n])) FH[n] = 0.0;
    }
}

} // namespace rsolver

