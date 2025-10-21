
#include "utils.hpp"
#include <cmath>
#include <fstream>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <iostream>
#include <cstdio>

namespace utils {

static inline bool is_bad(double x){
    return std::isnan(x) || !std::isfinite(x);
}

double divB_L2(const Fields& F){
    const auto& g = F.g;
    const std::size_t Nr = g.size_r();
    const std::size_t Nz = g.size_z();
    double acc = 0.0;
    std::size_t n = 0;

    // Central differences on interior cells (toy diagnostic; CT will replace later)
    for (std::size_t i=1; i<Nr-1; ++i){
        double r = (int(i)-int(g.Ng)+0.5)*g.dr;
        for (std::size_t k=1; k<Nz-1; ++k){
            auto id = g.idx(i,k);
            double dBr_dr = (F.Br[g.idx(i+1,k)] - F.Br[g.idx(i-1,k)])/(2.0*g.dr);
            double dBz_dz = (F.Bz[g.idx(i,k+1)] - F.Bz[g.idx(i,k-1)])/(2.0*g.dz);
            // Cylindrical: divB = (1/r) d(r Br)/dr + dBz/dz + (1/r) dBθ/dθ (last term=0 in axisymmetry)
            double d_rBr_dr = ( (r+g.dr*0.5)*F.Br[g.idx(i+1,k)] - (r-g.dr*0.5)*F.Br[g.idx(i-1,k)] )/(2.0*g.dr);
            double divB = (r>1e-14 ? d_rBr_dr/r : dBr_dr) + dBz_dz;
            acc += divB*divB;
            ++n;
        }
    }
    return std::sqrt(acc / std::max<std::size_t>(n,1));
}

double total_energy(const Fields& F, const RunConfig& cfg){
    const auto& g = F.g;
    double sum = 0.0;
    const double dr = g.dr, dz = g.dz;
    const double cell_area = dr*dz; // axisymmetry; strictly area element is 2π r dr dz for integrals,
                                    // but for a diagnostic L2-like sum we keep uniform weighting.
    for(std::size_t i=0;i<g.size_r();++i){
        for(std::size_t k=0;k<g.size_z();++k){
            auto id = g.idx(i,k);
            // Primitive-to-energy approximation for diagnostic (not the conserved E):
            double ke = 0.5 * F.rho[id]*(F.vr[id]*F.vr[id] + F.vz[id]*F.vz[id] + F.vth[id]*F.vth[id]);
            double mag = 0.5 * (F.Br[id]*F.Br[id] + F.Bz[id]*F.Bz[id] + F.Bth[id]*F.Bth[id]);
            double pe = F.p[id]/(cfg.phys.gamma - 1.0);
            sum += (ke + mag + pe) * cell_area;
        }
    }
    return sum;
}

std::size_t count_nans(const Fields& F){
    std::size_t c = 0;
    auto scan = [&](const std::vector<double>& v){
        for(double x: v){ if (is_bad(x)) ++c; }
    };
    scan(F.rho); scan(F.vr); scan(F.vz); scan(F.vth);
    scan(F.Br);  scan(F.Bz); scan(F.Bth);
    scan(F.p);   scan(F.E);
    return c;
}

void DebugFrame::write_json(const std::string& out_dir, int step) const{
    namespace fs = std::filesystem;
    fs::path base = fs::path(out_dir) / "debug";
    std::error_code ec;
    fs::create_directories(base, ec);
    if (ec){
        std::cerr << "[DEBUG] create_directories failed for " << base.string()
                  << " : " << ec.message() << "\n";
    }
    char fname[64];
    std::snprintf(fname, sizeof(fname), "debug_step_%04d.json", step);
    fs::path fpath = base / fname;

    std::ofstream f(fpath);
    if (!f){
        std::cerr << "[DEBUG] cannot open " << fpath.string() 
                  << " for writing. out_dir='" << out_dir << "'\n";
        return;
    }
    f << "{\n";
    f << "  \"t\": " << std::setprecision(16) << t << ",\n";
    f << "  \"dt\": " << std::setprecision(16) << dt << ",\n";
    f << "  \"cfl\": " << std::setprecision(16) << cfl << ",\n";
    f << "  \"max_wave\": " << std::setprecision(16) << max_wave << ",\n";
    f << "  \"divB_L2\": " << std::setprecision(16) << divB_L2_val << ",\n";
    f << "  \"energy_tot\": " << std::setprecision(16) << energy_tot << ",\n";
    f << "  \"nan_count\": " << nan_count << ",\n";
    f << "  \"notes\": \"" << notes << "\"\n";
    f << "}\n";
}

} // namespace utils
