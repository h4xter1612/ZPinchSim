#pragma once
#include <cstddef>

/**
 * @brief Structured cylindrical grid (r,z) with ghost cells.
 *
 * Indexing is cell-centered. The full storage in each dimension includes ghosts:
 *  - size_r() = Nr + 2*Ng
 *  - size_z() = Nz + 2*Ng
 *
 * Flattened index order: row-major with r as the slow index, z as the fast index:
 *   idx(i,k) = i*(Nz + 2*Ng) + k
 */
struct Grid {
    std::size_t Nr, Nz, Ng;  ///< interior cells in r, z; ghost cells on each side
    double Rmax, Zmax;       ///< physical domain extents
    double dr, dz;           ///< cell sizes

    Grid(std::size_t Nr_, std::size_t Nz_, std::size_t Ng_, double Rmax_, double Zmax_);

    /// Flattened index for (i,k) into a 1D array laid out with z as the fast index.
    inline std::size_t idx(std::size_t i, std::size_t k) const { return i*(Nz + 2*Ng) + k; }

    /// Total allocated size in r and z (including ghost layers).
    std::size_t size_r() const { return Nr + 2*Ng; }
    std::size_t size_z() const { return Nz + 2*Ng; }
};

