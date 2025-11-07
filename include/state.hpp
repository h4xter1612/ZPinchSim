#pragma once
#include <vector>
#include <string>
#include "grid.hpp"

// Cell-centered primitive/conserved fields (toy scaffold)
struct Fields {
    Grid g;

    // ===== modo 0 (axisimétrico actual) =====
    std::vector<double> rho, vr, vz, vth, Br, Bz, Bth, p, E;

    // ===== modo 1 (m=1) en forma cos/sin =====
    // sólo los campos que sí escribimos / sí visualizamos
    std::vector<double> rho1c, rho1s;
    std::vector<double> p1c,   p1s;
    std::vector<double> vr1c,  vr1s;
    std::vector<double> vz1c,  vz1s;
    std::vector<double> Bth1c, Bth1s;
    std::vector<double> Bz1c,  Bz1s;

    explicit Fields(const Grid&);
    void zero();
};

struct SimParams {
    double gamma=5.0/3.0, eta=1e-6, Bz0=0.0;
    std::string init_profile="tophat";
};

struct RunConfig {
    Grid grid;
    SimParams phys;
    double t_end, cfl, dt_min;
    int output_every;
    std::string out_dir;
};

