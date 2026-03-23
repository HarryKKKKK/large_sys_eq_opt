#pragma once

#include "grid.hpp"

double compute_dt(const Grid2D& grid, double cfl);

void advance_first_order(
    const Grid2D& Uold,
    Grid2D& Unew,
    double dt
);

inline double minmod(double a, double b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (a > 0.0) ? std::min(a, b) : std::max(a, b);
}