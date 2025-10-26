#pragma once
#include "state.hpp"
#include <string>

namespace physics {

struct CTConfig { double vel_amp{0.0}; std::string vel_type{"solid"}; double resistivity{0.0}; };
void run_2d_ct(Fields& F, const RunConfig& cfg, const CTConfig& ctcfg);

struct MHD2DConfig {
    double gamma{1.6666666667};
    std::string limiter{"mc"};
    double eta_ct{0.0};
    double cfl{0.4};
    double t_end{1e-4};
    int output_every{50};
    std::string problem{"brio_wu"};
    // NEW in step6
    double vmax_guard{1.0e3};   // if <=0, disabled
    double dt_max{5e-8};        // if <=0, disabled
    int diag_every{50};         // write diag.csv every N steps (>=1)
};
void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg);

} // namespace physics
