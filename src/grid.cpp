
#include "grid.hpp"
Grid::Grid(size_t Nr_, size_t Nz_, size_t Ng_, double Rmax_, double Zmax_)
: Nr(Nr_), Nz(Nz_), Ng(Ng_), Rmax(Rmax_), Zmax(Zmax_) {
    dr = Rmax / Nr;
    dz = Zmax / Nz;
}
