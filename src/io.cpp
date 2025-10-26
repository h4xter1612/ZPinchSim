#include "io.hpp"
#include <fstream>
#include <filesystem>
#include <iomanip>

namespace io {

static void write_csv_vec(std::ofstream& f, const std::vector<double>& v){
    // usa precisión alta para evitar redondeos raros
    f << std::setprecision(16);
    for (size_t i = 0; i < v.size(); ++i){
        f << v[i];
        if (i + 1 < v.size()) f << ",";
    }
    f << "\n";
}

void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t){
    namespace fs = std::filesystem;
    fs::create_directories(cfg.out_dir);

    const std::string base = cfg.out_dir + "/fields_" + std::to_string(step);

    // Campos básicos
    {
        std::ofstream f(base + "_rho.csv");
        write_csv_vec(f, F.rho);
    }
    {
        std::ofstream f(base + "_p.csv");
        write_csv_vec(f, F.p);
    }

    // Campos extra para análisis de modos
    {
        std::ofstream f(base + "_vr.csv");
        write_csv_vec(f, F.vr);
    }
    {
        std::ofstream f(base + "_Bth.csv");
        write_csv_vec(f, F.Bth);
    }

    // Metadata de la malla / tiempo (para scripts)
    {
        std::ofstream meta(base + "_meta.txt");
        meta << "t=" << std::setprecision(16) << t << "\n";
        meta << "Nr=" << F.g.Nr << ", Nz=" << F.g.Nz << ", Ng=" << F.g.Ng << "\n";
        meta << "Rmax=" << F.g.Rmax << ", Zmax=" << F.g.Zmax << "\n";
        meta << "Bz0=" << cfg.phys.Bz0 << "\n";
    }
}

void write_diag(const std::string& out_dir, int step, double t, double max_c){
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    const std::string path = out_dir + "/diag.csv";
    // si es el primer write crea cabecera
    const bool need_header = !fs::exists(path);

    std::ofstream f(path, std::ios::app);
    f << std::setprecision(16);
    if (need_header){
        f << "step,t,max_c\n";
    }
    f << step << "," << t << "," << max_c << "\n";
}

void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0){
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);
    // copia del YAML para trazabilidad (best-effort)
    try {
        fs::copy_file(cfg_path, out_dir + "/_config.yaml", fs::copy_options::overwrite_existing);
    } catch(...) { /* no-op */ }
    // info rápida
    std::ofstream f(out_dir + "/_run_info.txt");
    if (f) {
        f << "Bz0=" << std::setprecision(16) << Bz0 << "\n";
    }
}

} // namespace io

