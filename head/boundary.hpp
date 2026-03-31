#pragma once

#include "grid.hpp"

inline void apply_transmissive_boundary(Grid2D& grid) {
    const int ng = grid.ng();
    const int ib = grid.i_begin();
    const int ie = grid.i_end();
    const int jb = grid.j_begin();
    const int je = grid.j_end();

    const int total_nx = grid.total_nx();
    // const int total_ny = grid.total_ny();

    for (int j = jb; j < je; ++j) {
        // Left
        for (int g = 0; g < ng; ++g) {
            grid(ib - 1 - g, j) = grid(ib, j);
        }

        // Right
        for (int g = 0; g < ng; ++g) {
            grid(ie + g, j) = grid(ie - 1, j);
        }
    }

    for (int i = 0; i < total_nx; ++i) {
        // Bottom
        for (int g = 0; g < ng; ++g) {
            grid(i, jb - 1 - g) = grid(i, jb);
        }

        // Top
        for (int g = 0; g < ng; ++g) {
            grid(i, je + g) = grid(i, je - 1);
        }
    }
}