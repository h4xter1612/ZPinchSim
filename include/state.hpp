
#pragma once
#include <vector>
#include <string>
#include "grid.hpp"

// Cell-centered primitive/conserved fields (toy scaffold)
struct Fields {
    Grid g;
    std::vector<double> rho, vr, vz, vth, Br, Bz, Bth, p, E;
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
