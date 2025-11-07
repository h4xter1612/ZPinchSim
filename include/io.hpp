
#pragma once
#include "state.hpp"
#include <string>

namespace io {
// write npz-like simple CSV bundles for quick Python loading (toy)
// Escribe componentes 2.5D (m=1) para varios campos: *_1c.csv y *_1s.csv
// Usa un modelo separable: A_c(r) = (2/Zmax) ∫ [F(r,z) - <F>_z] cos(k z) dz (idem para sin)
// y luego F1c(r,z)=A_c(r) cos(k z), F1s(r,z)=A_s(r) sin(k z).
void write_snapshot_2p5D(const Fields& F,
                         const RunConfig& cfg,
                         int step, double t,
                         double k_proj);

void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t);
void write_diag(const std::string& out_dir, int step, double t, double max_c);
void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0);
}
