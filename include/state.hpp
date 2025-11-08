#pragma once
#include <string>
#include <vector>
#include "grid.hpp"

/**
 * @brief Cell-centered primitive/conserved fields.
 *
 * Mode-0 arrays represent the axisymmetric scaffold.
 * Mode-1 arrays store pseudo-θ (m=1) cosine/sine components that we write/visualize.
 */
struct Fields {
    Grid g;

    // ===== mode 0 (axisymmetric scaffold) =====
    std::vector<double> rho, vr, vz, vth, Br, Bz, Bth, p, E;

    // ===== mode 1 (m=1) cosine/sine components that we output/plot =====
    std::vector<double> rho1c, rho1s;
    std::vector<double> p1c,   p1s;
    std::vector<double> vr1c,  vr1s;
    std::vector<double> vz1c,  vz1s;
    std::vector<double> Bth1c, Bth1s;
    std::vector<double> Bz1c,  Bz1s;

    explicit Fields(const Grid&);
    void zero();
};

/**
 * @brief Basic physical parameters for initialization/diagnostics.
 */
struct SimParams {
    double      gamma = 5.0/3.0;
    double      eta   = 1e-6;
    double      Bz0   = 0.0;
    std::string init_profile = "tophat";
};

/**
 * @brief Global run configuration passed around the solver.
 */
struct RunConfig {
    Grid        grid;
    SimParams   phys;
    double      t_end;
    double      cfl;
    double      dt_min;
    int         output_every;
    std::string out_dir;
};

