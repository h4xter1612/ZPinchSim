#pragma once
#include "state.hpp"
#include <string>

namespace physics {

/**
 * @brief Prescribed axial flow profile v_z(r).
 * type:
 *  - "off"       : disabled
 *  - "linear"    : |vz| grows ~ r up to r0_frac*Rmax (capped)
 *  - "localized" : Gaussian around r0_frac*Rmax with width sigma_frac*Rmax
 * v0 is interpreted in Mach units relative to c_s at the axis at initialization.
 */
struct FlowProfile {
    std::string type{"off"};
    double v0{0.0};
    double r0_frac{0.30};
    double sigma_frac{0.12};
};

/**
 * @brief Simple CT (Constrained Transport) toy runner config (not used in the main 2D_MHD_TOY).
 */
struct CTConfig {
    double vel_amp{0.0};
    std::string vel_type{"solid"};
    double resistivity{0.0};
};
void run_2d_ct(Fields& F, const RunConfig& cfg, const CTConfig& ctcfg);

/**
 * @brief Modal seeding parameters (pseudo-θ).
 * m = 0 (sausage) or m = 1 (kink-like). Only m=0/1 are used in this toy setup.
 */
struct ModeSeed {
    bool enable{false};
    int m{0};
    double k{0.0};         ///< axial wavenumber
    double eps{0.0};       ///< seed amplitude (non-dimensional)
    double r0_frac{0.3};   ///< radial localization (fraction of Rmax)
    bool seed_vr{true};    ///< seed radial velocity
    bool seed_bth{false};  ///< optionally modulate B_theta
};

/**
 * @brief Main configuration for the 2D MHD toy model.
 *
 * Notes:
 *  - limiter: "mc" or "minmod"
 *  - bc_z   : "copy" or "periodic"
 *  - vmax_guard softly caps the wave speed estimate to avoid overly small dt.
 *  - dt_max optionally limits the time step from above.
 *  - diag.mode_amp (via fields below) enables a CSV amplitude trace.
 */
struct MHD2DConfig {
    double      gamma{1.6666666667};
    std::string limiter{"mc"};
    double      eta_ct{0.0};
    double      cfl{0.4};
    double      t_end{1e-4};
    int         output_every{50};
    std::string problem{"brio_wu"};

    // stability / step control
    double vmax_guard{1.0e3};
    double dt_max{5e-8};
    int    diag_every{50};

    // axial boundary condition
    std::string bc_z{"copy"}; // "copy" or "periodic"

    // modal seeding
    ModeSeed    modes;

    // axial flow & diagnostics
    FlowProfile flow;
    bool        write_mode_amp{true};
    double      k_diag{0.0};           ///< k>0 enables amplitude measurement
    std::string amp_from{"density"};   ///< "density" | "pressure"
};

/**
 * @brief Driver for the 2D MHD toy simulation (poloidal MHD + CT + axisymmetric Bθ update).
 */
void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg);

} // namespace physics

