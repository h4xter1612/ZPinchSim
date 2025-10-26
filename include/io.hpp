
#pragma once
#include "state.hpp"
#include <string>

namespace io {
// write npz-like simple CSV bundles for quick Python loading (toy)
void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t);
void write_diag(const std::string& out_dir, int step, double t, double max_c);
void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0);
}
