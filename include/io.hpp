#pragma once
#include "state.hpp"
#include <string>

namespace io {

/**
 * @brief Write m=1 (2.5D) cosine/sine projections for multiple fields.
 *        Produces <base>_<field>_1c.csv and <base>_<field>_1s.csv plus a meta file.
 * @param F       Simulation fields (with grid metadata).
 * @param cfg     Run configuration (used for output directory selection).
 * @param step    Integer step id, used in the file prefix.
 * @param t       Physical time (written into meta).
 * @param k_proj  Axial wavenumber used for the z-projection (<=0 disables).
 */
void write_snapshot_2p5D(const Fields& F,
                         const RunConfig& cfg,
                         int step, double t,
                         double k_proj);

/**
 * @brief Write a standard snapshot of main arrays to CSVs and a small meta file.
 */
void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t);

/**
 * @brief Append one line of scalar diagnostics (step, t, max wave speed) to diag.csv.
 */
void write_diag(const std::string& out_dir, int step, double t, double max_c);

/**
 * @brief Copy the YAML used and write a short text file with a few run parameters.
 */
void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0);

} // namespace io

