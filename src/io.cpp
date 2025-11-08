// =============================
// io.cpp — micro-optimizations (no logic changes)
// =============================
#include "io.hpp"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace io {

static void write_csv_vec(std::ofstream& f, const std::vector<double>& v){
    f << std::setprecision(16);
    for (size_t i = 0; i < v.size(); ++i){ f << v[i]; if (i + 1 < v.size()) f << ","; }
    f << "\n";
}

// m=1 projection helpers with precomputed trig
static void project_m1_full_rz(const Fields& F, const std::vector<double>& vec_full,
                               double kproj,
                               std::vector<double>& F1c_center,
                               std::vector<double>& F1s_center)
{
    const auto& g = F.g; const std::size_t Nr = g.Nr, Nz = g.Nz, Ng = g.Ng; const double dz = g.dz;
    F1c_center.assign(Nr*Nz, 0.0); F1s_center.assign(Nr*Nz, 0.0);
    auto CIDX = [&](std::size_t i, std::size_t k){ return i*Nz + k; };
    auto FIDX = [&](std::size_t ii, std::size_t kk){ return (ii)*(g.Nz + 2*Ng) + (kk); };

    // Precompute cos/sin(k z)
    std::vector<double> cosk(Nz), sink(Nz);
    for (std::size_t k=0; k<Nz; ++k){ const double z = (k + 0.5) * dz; cosk[k]=std::cos(kproj*z); sink[k]=std::sin(kproj*z); }

    for (std::size_t i=0; i<Nr; ++i){
        const std::size_t ii = Ng + i;
        double mean_z = 0.0;
        for (std::size_t k=0; k<Nz; ++k){ const std::size_t kk = Ng + k; mean_z += vec_full[FIDX(ii, kk)]; }
        mean_z /= static_cast<double>(std::max<std::size_t>(1, Nz));

        double Ac = 0.0, As = 0.0;
        for (std::size_t k=0; k<Nz; ++k){ const std::size_t kk = Ng + k; const double f = vec_full[FIDX(ii, kk)] - mean_z; Ac += f * cosk[k]; As += f * sink[k]; }
        const double fac = (2.0 / g.Zmax) * dz; Ac *= fac; As *= fac;

        for (std::size_t k=0; k<Nz; ++k){ F1c_center[CIDX(i,k)] = Ac * cosk[k]; F1s_center[CIDX(i,k)] = As * sink[k]; }
    }
}

static void write_center_csv(const std::string& base_path, const std::string& tag,
                             const std::vector<double>& center, std::size_t Nr, std::size_t Nz)
{
    std::ofstream f(base_path + "_" + tag + ".csv"); f << std::setprecision(16);
    for (std::size_t i=0; i<Nr; ++i){ const std::size_t off = i*Nz; for (std::size_t k=0; k<Nz; ++k){ f << center[off + k]; if (k+1 < Nz) f << ","; } f << "\n"; }
}

void write_snapshot_2p5D(const Fields& F, const RunConfig& cfg, int step, double t, double k_proj){
    if (!(k_proj > 0.0)) return; namespace fs = std::filesystem; fs::create_directories(cfg.out_dir);
    const auto& g = F.g; const std::size_t Nr = g.Nr; const std::size_t Nz = g.Nz; const std::string base = cfg.out_dir + "/fields_" + std::to_string(step);
    struct Item { const char* name; const std::vector<double>& vec; };
    std::vector<Item> items = {{"vr",F.vr},{"vz",F.vz},{"Bth",F.Bth},{"Br",F.Br},{"Bz",F.Bz},{"p",F.p},{"rho",F.rho}};
    for (const auto& it : items){ std::vector<double> F1c_center, F1s_center; project_m1_full_rz(F, it.vec, k_proj, F1c_center, F1s_center); write_center_csv(base, std::string(it.name) + "_1c", F1c_center, Nr, Nz); write_center_csv(base, std::string(it.name) + "_1s", F1s_center, Nr, Nz); }
    { std::ofstream meta(base + "_m1_meta.txt"); meta << "t=" << std::setprecision(16) << t << "\n";
      meta << "Nr=" << g.Nr << ", Nz=" << g.Nz << ", Ng=" << g.Ng << "\n";
      meta << "Rmax=" << g.Rmax << ", Zmax=" << g.Zmax << "\n";
      meta << "k_proj=" << std::setprecision(16) << k_proj << "\n"; }
}

void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t){
    namespace fs = std::filesystem; fs::create_directories(cfg.out_dir);
    const std::string base = cfg.out_dir + "/fields_" + std::to_string(step);
    { std::ofstream f(base + "_rho.csv");  write_csv_vec(f, F.rho); }
    { std::ofstream f(base + "_p.csv");    write_csv_vec(f, F.p); }
    { std::ofstream f(base + "_vr.csv");   write_csv_vec(f, F.vr); }
    { std::ofstream f(base + "_Bth.csv");  write_csv_vec(f, F.Bth); }
    { std::ofstream f(base + "_Bz.csv");   write_csv_vec(f, F.Bz); }
    { std::ofstream f(base + "_rho1c.csv"); write_csv_vec(f, F.rho1c); }
    { std::ofstream f(base + "_rho1s.csv"); write_csv_vec(f, F.rho1s); }
    { std::ofstream f(base + "_p1c.csv");   write_csv_vec(f, F.p1c); }
    { std::ofstream f(base + "_p1s.csv");   write_csv_vec(f, F.p1s); }
    { std::ofstream f(base + "_vr1c.csv");  write_csv_vec(f, F.vr1c); }
    { std::ofstream f(base + "_vr1s.csv");  write_csv_vec(f, F.vr1s); }
    { std::ofstream f(base + "_Bth1c.csv"); write_csv_vec(f, F.Bth1c); }
    { std::ofstream f(base + "_Bth1s.csv"); write_csv_vec(f, F.Bth1s); }
    { std::ofstream f(base + "_Bz1c.csv");  write_csv_vec(f, F.Bz1c); }
    { std::ofstream f(base + "_Bz1s.csv");  write_csv_vec(f, F.Bz1s); }
    { std::ofstream meta(base + "_meta.txt"); meta << "t=" << std::setprecision(16) << t << "\n";
      meta << "Nr=" << F.g.Nr << ", Nz=" << F.g.Nz << ", Ng=" << F.g.Ng << "\n";
      meta << "Rmax=" << F.g.Rmax << ", Zmax=" << F.g.Zmax << "\n";
      meta << "Bz0=" << cfg.phys.Bz0 << "\n"; meta << "has_m1=1\n"; }
}

void write_diag(const std::string& out_dir, int step, double t, double max_c){
    namespace fs = std::filesystem; fs::create_directories(out_dir);
    const std::string path = out_dir + "/diag.csv"; const bool need_header = !fs::exists(path);
    std::ofstream f(path, std::ios::app); f << std::setprecision(16);
    if (need_header){ f << "step,t,max_c\n"; }
    f << step << "," << t << "," << max_c << "\n";
}

void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0){
    namespace fs = std::filesystem; fs::create_directories(out_dir);
    try { fs::copy_file(cfg_path, out_dir + "/_config.yaml", fs::copy_options::overwrite_existing); } catch(...) { /* no-op */ }
    std::ofstream f(out_dir + "/_run_info.txt"); if (f) { f << "Bz0=" << std::setprecision(16) << Bz0 << "\n"; }
}

} // namespace io
