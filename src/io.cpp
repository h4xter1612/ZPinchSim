// =============================
// io.cpp — micro-optimizations (no logic changes)
// =============================
/**
 * @file io.cpp
 * @brief CSV snapshot writers, m=1 (2.5D) projection helpers, diagnostics, and run metadata.
 * @details
 *   Pure I/O helpers:
 *   - Write flat CSV rows for raw field arrays.
 *   - Compute and dump m=1 (cos/sin) projected fields at cell centers (a.k.a. "2.5D").
 *   - Append simple diagnostics to a CSV log.
 *   - Persist minimal run metadata and a copy of the YAML config.
 *
 *   This translation unit contains no physics logic and only performs file output
 *   using standard C++ I/O. All functions are exception-safe in the sense that
 *   failures to open or copy files do not throw and instead no-op where sensible.
 *
 * @note Geometry and indexing come from the @c Fields::g grid (includes ghosts).
 */

#include "io.hpp"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace io {

/**
 * @brief Write a contiguous vector of @c double as a single CSV row.
 * @param f Already-opened output stream (text mode).
 * @param v Values to write; each element is separated by a comma, then a newline.
 * @post The stream precision is set to 16 for this write.
 * @note Intended for large flat arrays; avoids per-element formatting overhead.
 */
static void write_csv_vec(std::ofstream& f, const std::vector<double>& v){
    f << std::setprecision(16);
    for (size_t i = 0; i < v.size(); ++i){
        f << v[i];
        if (i + 1 < v.size()) f << ",";
    }
    f << "\n";
}

/**
 * @brief Project a full (ghost-including) field onto m=1 along @c z at cell centers.
 *
 * @details
 *   For each radius index @c i (cell-center in @c r), the routine:
 *    1) Computes the @c z-mean of the provided full field (including ghosts) and removes it.
 *    2) Accumulates cosine and sine coefficients @c (Ac, As) for a given wavenumber @c kproj.
 *    3) Reconstructs center-plane values @c Ac*cos(k z) and @c As*sin(k z) at every (i,k) center.
 *
 *   The output arrays are sized @c (Nr * Nz) and indexed as a flat (i,k) grid:
 *   @code
 *     CIDX(i,k) = i * Nz + k
 *   @endcode
 *
 * @param F        Fields container (only geometry/strides are used).
 * @param vec_full Input field over the *full* (Nr + 2*Ng) x (Nz + 2*Ng) domain, ghosts included.
 * @param kproj    Wavenumber k used for the m=1 projection (units consistent with z).
 * @param[out] F1c_center Output: Ac*cos(k z) at cell centers, flattened (Nr * Nz).
 * @param[out] F1s_center Output: As*sin(k z) at cell centers, flattened (Nr * Nz).
 *
 * @warning Assumes uniform spacing in z; relies on @c g.dz and @c g.Zmax for normalization.
 * @note Cos/Sin(k z) are precomputed per z-slice to reduce trig overhead.
 */
static void project_m1_full_rz(const Fields& F, const std::vector<double>& vec_full,
                               double kproj,
                               std::vector<double>& F1c_center,
                               std::vector<double>& F1s_center)
{
    const auto& g = F.g;
    const std::size_t Nr = g.Nr, Nz = g.Nz, Ng = g.Ng;
    const double dz = g.dz;

    F1c_center.assign(Nr*Nz, 0.0);
    F1s_center.assign(Nr*Nz, 0.0);

    auto CIDX = [&](std::size_t i, std::size_t k){ return i*Nz + k; };
    auto FIDX = [&](std::size_t ii, std::size_t kk){ return (ii)*(g.Nz + 2*Ng) + (kk); };

    // Precompute cos/sin(k z) at center locations
    std::vector<double> cosk(Nz), sink(Nz);
    for (std::size_t k=0; k<Nz; ++k){
        const double z = (k + 0.5) * dz;
        cosk[k] = std::cos(kproj * z);
        sink[k] = std::sin(kproj * z);
    }

    // Per-radius projection (remove mean, then project)
    for (std::size_t i=0; i<Nr; ++i){
        const std::size_t ii = Ng + i;

        // 1) z-mean (over centers)
        double mean_z = 0.0;
        for (std::size_t k=0; k<Nz; ++k){
            const std::size_t kk = Ng + k;
            mean_z += vec_full[FIDX(ii, kk)];
        }
        mean_z /= static_cast<double>(std::max<std::size_t>(1, Nz));

        // 2) accumulate Ac, As
        double Ac = 0.0, As = 0.0;
        for (std::size_t k=0; k<Nz; ++k){
            const std::size_t kk = Ng + k;
            const double f = vec_full[FIDX(ii, kk)] - mean_z;
            Ac += f * cosk[k];
            As += f * sink[k];
        }
        // Normalize integral over z (continuous interpretation)
        const double fac = (2.0 / g.Zmax) * dz;
        Ac *= fac;
        As *= fac;

        // 3) reconstruct Ac*cos and As*sin on the center plane
        for (std::size_t k=0; k<Nz; ++k){
            F1c_center[CIDX(i,k)] = Ac * cosk[k];
            F1s_center[CIDX(i,k)] = As * sink[k];
        }
    }
}

/**
 * @brief Write a center-plane rectangular (Nr x Nz) array to CSV (row-major by r, inner loop z).
 * @param base_path Base path (without extension) used to build the final filename.
 * @param tag       Suffix tag appended to @p base_path as "<base>_<tag>.csv".
 * @param center    Flattened (Nr*Nz) array to write.
 * @param Nr        Number of radial cells (no ghosts).
 * @param Nz        Number of axial cells (no ghosts).
 * @post Produces one CSV file with Nr lines; each line has Nz comma-separated values.
 */
static void write_center_csv(const std::string& base_path, const std::string& tag,
                             const std::vector<double>& center, std::size_t Nr, std::size_t Nz)
{
    std::ofstream f(base_path + "_" + tag + ".csv");
    f << std::setprecision(16);
    for (std::size_t i=0; i<Nr; ++i){
        const std::size_t off = i*Nz;
        for (std::size_t k=0; k<Nz; ++k){
            f << center[off + k];
            if (k+1 < Nz) f << ",";
        }
        f << "\n";
    }
}

/**
 * @brief Dump m=1-projected fields ("2.5D") for several variables at a given step/time.
 *
 * @details For each selected primitive (vr, vz, Bth, Br, Bz, p, rho), writes:
 *   - "<base>_<name>_1c.csv" with Ac*cos(k z)
 *   - "<base>_<name>_1s.csv" with As*sin(k z)
 *   where base is "<out_dir>/fields_<step>".
 *
 *   Also writes a metadata text file with geometry, k_proj, and time.
 *
 * @param F      Fields container (data + geometry).
 * @param cfg    Run configuration (for output directory).
 * @param step   Current time-step index.
 * @param t      Physical time associated with the dump.
 * @param k_proj Wavenumber k used for the projection; if <= 0, the function returns immediately.
 *
 * @note Creates the output directory if missing.
 * @warning Assumes @p k_proj > 0 and uniform grid in z.
 */
void write_snapshot_2p5D(const Fields& F, const RunConfig& cfg, int step, double t, double k_proj){
    if (!(k_proj > 0.0)) return;

    namespace fs = std::filesystem;
    fs::create_directories(cfg.out_dir);

    const auto& g = F.g;
    const std::size_t Nr = g.Nr;
    const std::size_t Nz = g.Nz;
    const std::string base = cfg.out_dir + "/fields_" + std::to_string(step);

    struct Item { const char* name; const std::vector<double>& vec; };
    std::vector<Item> items = {
        {"vr",  F.vr},  {"vz",  F.vz},  {"Bth", F.Bth},
        {"Br",  F.Br},  {"Bz",  F.Bz},  {"p",   F.p},
        {"rho", F.rho}
    };

    for (const auto& it : items){
        std::vector<double> F1c_center, F1s_center;
        project_m1_full_rz(F, it.vec, k_proj, F1c_center, F1s_center);
        write_center_csv(base, std::string(it.name) + "_1c", F1c_center, Nr, Nz);
        write_center_csv(base, std::string(it.name) + "_1s", F1s_center, Nr, Nz);
    }

    // Minimal metadata sidecar
    {
        std::ofstream meta(base + "_m1_meta.txt");
        meta << "t=" << std::setprecision(16) << t << "\n";
        meta << "Nr=" << g.Nr << ", Nz=" << g.Nz << ", Ng=" << g.Ng << "\n";
        meta << "Rmax=" << g.Rmax << ", Zmax=" << g.Zmax << "\n";
        meta << "k_proj=" << std::setprecision(16) << k_proj << "\n";
    }
}

/**
 * @brief Dump raw (flat) CSV snapshots for core fields and stored m=1 components.
 *
 * @details Produces several "<base>_<field>.csv" files where base is
 *   "<out_dir>/fields_<step>". Also writes a small "<base>_meta.txt" with grid meta
 *   and Bz0.
 *
 * @param F    Fields container (data + geometry).
 * @param cfg  Run configuration (output dir, Bz0 metadata).
 * @param step Step index to embed in filenames.
 * @param t    Physical time of this snapshot.
 *
 * @note This function writes both axisymmetric fields and any pre-computed m=1
 *       arrays stored on F (e.g., vr1c, vr1s, ...).
 */
void write_snapshot(const Fields& F, const RunConfig& cfg, int step, double t){
    namespace fs = std::filesystem;
    fs::create_directories(cfg.out_dir);

    const std::string base = cfg.out_dir + "/fields_" + std::to_string(step);

    { std::ofstream f(base + "_rho.csv");   write_csv_vec(f, F.rho);  }
    { std::ofstream f(base + "_p.csv");     write_csv_vec(f, F.p);    }
    { std::ofstream f(base + "_vr.csv");    write_csv_vec(f, F.vr);   }
    { std::ofstream f(base + "_Bth.csv");   write_csv_vec(f, F.Bth);  }
    { std::ofstream f(base + "_Bz.csv");    write_csv_vec(f, F.Bz);   }

    { std::ofstream f(base + "_rho1c.csv"); write_csv_vec(f, F.rho1c); }
    { std::ofstream f(base + "_rho1s.csv"); write_csv_vec(f, F.rho1s); }
    { std::ofstream f(base + "_p1c.csv");   write_csv_vec(f, F.p1c);   }
    { std::ofstream f(base + "_p1s.csv");   write_csv_vec(f, F.p1s);   }
    { std::ofstream f(base + "_vr1c.csv");  write_csv_vec(f, F.vr1c);  }
    { std::ofstream f(base + "_vr1s.csv");  write_csv_vec(f, F.vr1s);  }
    { std::ofstream f(base + "_Bth1c.csv"); write_csv_vec(f, F.Bth1c); }
    { std::ofstream f(base + "_Bth1s.csv"); write_csv_vec(f, F.Bth1s); }
    { std::ofstream f(base + "_Bz1c.csv");  write_csv_vec(f, F.Bz1c);  }
    { std::ofstream f(base + "_Bz1s.csv");  write_csv_vec(f, F.Bz1s);  }

    // Tiny metadata file
    {
        std::ofstream meta(base + "_meta.txt");
        meta << "t=" << std::setprecision(16) << t << "\n";
        meta << "Nr=" << F.g.Nr << ", Nz=" << F.g.Nz << ", Ng=" << F.g.Ng << "\n";
        meta << "Rmax=" << F.g.Rmax << ", Zmax=" << F.g.Zmax << "\n";
        meta << "Bz0=" << cfg.phys.Bz0 << "\n";
        meta << "has_m1=1\n";
    }
}

/**
 * @brief Append a single-line diagnostic row to @c <out_dir>/diag.csv .
 *
 * @details If the file does not exist, a header "step,t,max_c" is written first.
 *          Then a new line with the provided values is appended.
 *
 * @param out_dir Output directory where @c diag.csv lives.
 * @param step    Current step index.
 * @param t       Physical time.
 * @param max_c   Reported maximum characteristic speed (or any scalar metric).
 */
void write_diag(const std::string& out_dir, int step, double t, double max_c){
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    const std::string path = out_dir + "/diag.csv";
    const bool need_header = !fs::exists(path);

    std::ofstream f(path, std::ios::app);
    f << std::setprecision(16);
    if (need_header){
        f << "step,t,max_c\n";
    }
    f << step << "," << t << "," << max_c << "\n";
}

/**
 * @brief Persist minimal run info and copy the YAML config into the output directory.
 *
 * @details
 *   - Attempts to copy @p cfg_path to "<out_dir>/_config.yaml" (overwrites if present).
 *   - Writes "<out_dir>/_run_info.txt" with a single line: "Bz0=<value>".
 *
 * @param out_dir Output directory.
 * @param cfg_path Path to the input YAML/CFG file used to launch the run.
 * @param Bz0      Background axial field value to record.
 *
 * @note File copy errors are swallowed (best-effort archival).
 */
void write_run_info(const std::string& out_dir, const std::string& cfg_path, double Bz0){
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    try {
        fs::copy_file(cfg_path, out_dir + "/_config.yaml", fs::copy_options::overwrite_existing);
    } catch(...) {
        // best-effort: ignore copy failures
    }

    std::ofstream f(out_dir + "/_run_info.txt");
    if (f) {
        f << "Bz0=" << std::setprecision(16) << Bz0 << "\n";
    }
}

} // namespace io

