#pragma once
#include <cstddef>
#include <string>
#include "state.hpp"

namespace utils {

/**
 * @brief L2 norm of ∇·B on cell centers (toy diagnostic; CT should control this).
 */
double divB_L2(const Fields& F);

/**
 * @brief Approximate total energy integral (simple cell sum).
 * Note: not the conserved energy; used only as a scalar diagnostic.
 */
double total_energy(const Fields& F, const RunConfig& cfg);

/**
 * @brief Count NaN/Inf occurrences in all main field arrays.
 */
std::size_t count_nans(const Fields& F);

/**
 * @brief Minimal step snapshot written as JSON for post-mortem debugging.
 * The file path is: <out_dir>/debug/debug_step_####.json
 */
struct DebugFrame {
    double t{0.0};
    double dt{0.0};
    double cfl{0.0};
    double max_wave{0.0};
    double divB_L2_val{0.0};
    double energy_tot{0.0};
    std::size_t nan_count{0};
    std::string notes;

    void write_json(const std::string& out_dir, int step) const;
};

} // namespace utils

