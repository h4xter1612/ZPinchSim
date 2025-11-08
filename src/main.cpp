#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <omp.h>
#include "grid.hpp"
#include "state.hpp"
#include "physics.hpp"
#include "utils.hpp"
#include "io.hpp"

/**
 * @brief Read entire file content into a single string.
 */
static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/**
 * @brief Extract a double value for key `key:` from a YAML-ish string `s`.
 *        Returns `def` if the key is missing or parsing fails.
 *        Ignores trailing comments starting with '#'.
 */
static double read_val(const std::string& s, const std::string& key, double def) {
    auto p = s.find(key + ":");
    if (p == std::string::npos) return def;
    auto e = s.find("\n", p);
    std::string line = (e == std::string::npos) ? s.substr(p) : s.substr(p, e - p);
    auto c = line.find(":");
    std::string v = line.substr(c + 1);
    auto h = v.find('#');
    if (h != std::string::npos) v = v.substr(0, h);
    auto clip = [](int ch){ return ch==' ' || ch=='\t' || ch=='\r'; };
    size_t a = 0, b = v.size();
    while (a < b && clip(v[a])) ++a;
    while (b > a && clip(v[b-1])) --b;
    if (a >= b) return def;
    try { return std::stod(v.substr(a, b - a)); } catch (...) { return def; }
}

/**
 * @brief Same as read_val but returns an int.
 */
static int read_ival(const std::string& s, const std::string& key, int def) {
    return static_cast<int>(read_val(s, key, def));
}

/**
 * @brief Extract a string value for key `key:` from a YAML-ish string `s`.
 *        Trims braces/commas and strips surrounding quotes if present.
 *        Returns `def` if the key is missing or empty.
 */
static std::string read_sval(const std::string& s, const std::string& key, const std::string& def) {
    auto p = s.find(key + ":");
    if (p == std::string::npos) return def;
    auto c = s.find(":", p);
    auto e = s.find("\n", p);
    std::string v = (e == std::string::npos) ? s.substr(c + 1) : s.substr(c + 1, e - c - 1);
    auto cutpos = v.find_first_of("#},");
    if (cutpos != std::string::npos) v = v.substr(0, cutpos);
    auto clip = [](int ch){
        return ch==' ' || ch=='\t' || ch=='\r' || ch=='{' || ch=='}' || ch==',';
    };
    size_t a = 0, b = v.size();
    while (a < b && clip(v[a])) ++a;
    while (b > a && clip(v[b-1])) --b;
    std::string x = (a < b) ? v.substr(a, b - a) : std::string();
    if (x.size() >= 2 && ((x.front()=='\"' && x.back()=='\"') || (x.front()=='\'' && x.back()=='\''))) {
        x = x.substr(1, x.size() - 2);
    }
    return x.empty() ? def : x;
}

/**
 * @brief Extract a boolean for key `key:`. Accepts 1/0, true/false, yes/no (case-insensitive on 'true/false').
 *        Returns `def` if unparsable.
 */
static bool read_bval(const std::string& s, const std::string& key, bool def) {
    auto v = read_sval(s, key, def ? "true" : "false");
    if (v=="1" || v=="true" || v=="True" || v=="yes")  return true;
    if (v=="0" || v=="false"|| v=="False"|| v=="no")   return false;
    return def;
}

/**
 * @brief Print a short usage with example config files and where to find the full parameter list.
 */
static void print_help() {
    std::cerr
        << "Usage:" << " zpinch_run configs/<config.yaml>\n"
        << "\n"
        << "Examples:\n"
        << "  " << " zpinch_run configs/sausage_m0.yaml   # sausage (m=0)\n"
        << "  " << " zpinch_run configs/kink_m1.yaml      # kink (m=1)\n"
        << "  " << " zpinch_run configs/stable.yaml       # stable reference\n"
        << "  " << " zpinch_run configs/custom.yaml       # start from your custom file\n"
        << "\n"
        << "Tip: For a full list of parameters and documentation, see configs/helper.yaml\n";
}

int main(int argc, char** argv) {
    // Handle --help/-h or missing config path.
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_help();
        return (argc < 2) ? 1 : 0;
    }

    // Load configuration file as a single string.
    const std::string cfgs = slurp(argv[1]);

    // Parse core grid/time/run parameters from config.
    const int    Nr           = read_ival(cfgs, "Nr", 128);
    const int    Nz           = read_ival(cfgs, "Nz", 256);
    const int    Ng           = read_ival(cfgs, "ghost", 2);
    const double Rmax         = read_val(cfgs, "Rmax", 0.01);
    const double Zmax         = read_val(cfgs, "Zmax", 0.10);
    const double t_end        = read_val(cfgs, "t_end", 1.0e-4);
    const double cfl          = read_val(cfgs, "cfl", 0.4);
    const double dt_min       = read_val(cfgs, "dt_min", 1.0e-9);
    const int    output_every = read_ival(cfgs, "output_every", 50);
    const std::string out_dir = read_sval(cfgs, "out_dir", "./data");
    const std::string stage   = read_sval(cfgs, "stage", "2D_MHD_TOY");
    const bool verbose        = read_bval(cfgs, "verbose", false);

    // Build grid and run configuration; allocate field arrays.
    Grid g(Nr, Nz, Ng, Rmax, Zmax);
    RunConfig rc{g, {}, t_end, cfl, dt_min, output_every, out_dir};
    Fields F(g);

    // Read background Bz (fallback chain: Bz0 <- Bz <- Bz_background).
    const double Bz0 = read_val(cfgs, "Bz0",
                       read_val(cfgs, "Bz",
                       read_val(cfgs, "Bz_background", 0.0)));
    rc.phys.Bz0 = Bz0;

    // Select stage/driver. Main path: 2D MHD toy model.
    if (stage == "2D_MHD_TOY") {
        physics::MHD2DConfig mc;

        // Physics / numerics
        mc.gamma        = read_val(cfgs, "gamma", 1.6666666667);
        mc.limiter      = read_sval(cfgs, "limiter", "mc");
        mc.eta_ct       = read_val(cfgs, "eta_ct", 0.0);
        mc.cfl          = read_val(cfgs, "cfl", 0.4);
        mc.t_end        = read_val(cfgs, "t_end", 1.0e-4);
        mc.output_every = read_ival(cfgs, "output_every", 50);
        mc.problem      = read_sval(cfgs, "problem", "brio_wu");
        mc.vmax_guard   = read_val(cfgs, "vmax_guard", 1.0e3);
        mc.dt_max       = read_val(cfgs, "dt_max", 5e-8);
        mc.diag_every   = read_ival(cfgs, "diag_every", 50);
        mc.bc_z         = read_sval(cfgs, "bc_z", "copy");

        // Modal seeding (m=0 sausage / m=1 kink-like)
        mc.modes.enable   = read_bval(cfgs, "modes.enable", false);
        mc.modes.m        = static_cast<int>(read_val(cfgs, "modes.m", 0));
        mc.modes.k        = read_val(cfgs, "modes.k", 0.0);
        mc.modes.eps      = read_val(cfgs, "modes.eps", 0.0);
        mc.modes.r0_frac  = read_val(cfgs, "modes.r0_frac", 0.3);
        mc.modes.seed_vr  = read_bval(cfgs, "modes.seed_vr", true);
        mc.modes.seed_bth = read_bval(cfgs, "modes.seed_bth", false);

        // Optional axial flow profile
        mc.flow.type       = read_sval(cfgs, "flow.type", "off");
        mc.flow.v0         = read_val (cfgs, "flow.v0", 0.0);
        mc.flow.r0_frac    = read_val (cfgs, "flow.r0_frac", 0.30);
        mc.flow.sigma_frac = read_val (cfgs, "flow.sigma_frac", 0.12);

        // Optional modal amplitude diagnostic
        mc.write_mode_amp  = read_bval(cfgs, "diag.mode_amp", true);
        mc.k_diag          = read_val (cfgs, "diag.k_diag", 0.0);
        mc.amp_from        = read_sval(cfgs, "diag.amp_from", "density");

        // Optional config echo
        if (verbose) {
            std::cout << "[CFG] out_dir=" << out_dir
                      << " Bz0=" << rc.phys.Bz0
                      << " vmax_guard=" << mc.vmax_guard
                      << " dt_max=" << mc.dt_max
                      << " bc_z=" << mc.bc_z
                      << " flow=" << mc.flow.type
                      << " v0=" << mc.flow.v0
                      << " modes=" << (mc.modes.enable ? "on" : "off")
                      << " config=" << argv[1] << "\n";
        }

        // ---- Run the simulation driver ----
        physics::run_2d_mhd_toy(F, rc, mc);

        // Friendly end message with next steps for analysis.
        std::cout << "Simulation completed.\n"
                  << "For analysis/visualization, run Python scripts in the 'viz' folder, e.g.:\n"
                  << "  python viz/diagnostics.py --run-dir " << out_dir << " [...]\n"
                  << "  python viz/viz.py         --run-dir " << out_dir << " [...]\n";
        return 0;
    }

    // Unknown stage.
    if (verbose) std::cout << "No recognized stage; exiting.\n";
    return 0;
}

