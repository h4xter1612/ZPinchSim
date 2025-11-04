#pragma once
#include "state.hpp"
#include <string>

namespace physics {

// --- NEW: Perfil de flujo axial v_z(r)
struct FlowProfile {
    std::string type{"off"};   // "off" | "linear" | "localized"
    double v0{0.0};            // pico, interpretado como Mach (escala con c_s0)
    double r0_frac{0.30};      // fracción de Rmax para el pico
    double sigma_frac{0.12};   // ancho gaussiano (solo "localized")
};

struct CTConfig { double vel_amp{0.0}; std::string vel_type{"solid"}; double resistivity{0.0}; };
void run_2d_ct(Fields& F, const RunConfig& cfg, const CTConfig& ctcfg);

struct ModeSeed {
    bool enable{false};
    int m{0};               // 0=sausage, 1=kink (pseudo)
    double k{0.0};          // número de onda axial
    double eps{0.0};        // amplitud (no-dim)
    double r0_frac{0.3};    // radio característico (fracción de Rmax)
    bool seed_vr{true};
    bool seed_bth{false};   // opcional: ondular Bθ
};

struct MHD2DConfig {
    double gamma{1.6666666667};
    std::string limiter{"mc"};
    double eta_ct{0.0};
    double cfl{0.4};
    double t_end{1e-4};
    int output_every{50};
    std::string problem{"brio_wu"};
    // step6/7
    double vmax_guard{1.0e3};
    double dt_max{5e-8};
    int diag_every{50};
    // NEW
    std::string bc_z{"copy"};   // "copy" o "periodic"
    ModeSeed modes;             // semilla modal

    // --- NEW: flujo axial y diagnósticos
    FlowProfile flow;           // perfil de v_z(r)
    bool write_mode_amp{true};  // escribir amplitud modal en CSV
    double k_diag{0.0};         // k a proyectar (>0 activa medición)
    std::string amp_from{"density"}; // "density" | "pressure"
};

void run_2d_mhd_toy(Fields& F, const RunConfig& cfg, const MHD2DConfig& mhdcfg);

} // namespace physics

