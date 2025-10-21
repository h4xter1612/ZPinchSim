
#pragma once
#include <cstddef>
#include <string>
#include "state.hpp"

namespace utils {

// Compute L2 norm of div(B) using central differences on cell centers.
double divB_L2(const Fields& F);

// Compute approximate total energy integral (cell sum) for diagnostics.
double total_energy(const Fields& F, const RunConfig& cfg);

// Count NaNs across all primary fields.
std::size_t count_nans(const Fields& F);

// Simple debug frame to be written as JSON for each step or checkpoint.
struct DebugFrame {
    double t{0.0};
    double dt{0.0};
    double cfl{0.0};
    double max_wave{0.0};
    double divB_L2_val{0.0};
    double energy_tot{0.0};
    std::size_t nan_count{0};
    std::string notes;

    // Write JSON to data/debug/debug_step_####.json
    void write_json(const std::string& out_dir, int step) const;
};

} // namespace utils
