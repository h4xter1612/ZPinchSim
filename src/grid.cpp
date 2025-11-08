#include "grid.hpp"

/**
 * @brief Construct grid geometry and spacing.
 * @param Nr_   number of physical cells in r
 * @param Nz_   number of physical cells in z
 * @param Ng_   number of ghost cells on each side
 * @param Rmax_ domain size in r
 * @param Zmax_ domain size in z
 */
Grid::Grid(size_t Nr_, size_t Nz_, size_t Ng_, double Rmax_, double Zmax_)
: Nr(Nr_), Nz(Nz_), Ng(Ng_), Rmax(Rmax_), Zmax(Zmax_) {
    dr = Rmax / Nr;
    dz = Zmax / Nz;
}

