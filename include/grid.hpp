
#pragma once
#include <vector>
#include <cstddef>

struct Grid {
    size_t Nr, Nz, Ng;
    double Rmax, Zmax, dr, dz;
    Grid(size_t Nr_, size_t Nz_, size_t Ng_, double Rmax_, double Zmax_);
    inline size_t idx(size_t i, size_t k) const { return i*(Nz+2*Ng)+k; }
    size_t size_r() const { return Nr + 2*Ng; }
    size_t size_z() const { return Nz + 2*Ng; }
};
